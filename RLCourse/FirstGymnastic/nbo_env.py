"""
NBO Custom Gymnasium Environment
---------------------------------
Models a retail bank's Next Best Offer problem:
  - 7 products x 3 channels = 21 product-channel combos
  - Weekly dynamic action masking (some campaigns disabled)
  - 7-day delayed reward (n-step buffer simulates attribution window)
  - Customer state: demographics + transaction history + product holdings + CVI
  - No-action (action 0) baseline included
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Product / channel metadata ──────────────────────────────────────────────
PRODUCTS = [
    "CreditCard", "PersonalLoan", "Mortgage",
    "SavingsAccount", "Insurance", "InvestmentFund", "OverdraftLimit",
]
CHANNELS = ["SMS", "Call", "Push"]

# Base conversion probability per product (used in the synthetic reward model)
BASE_CONV_PROB = {
    "CreditCard":      0.08,
    "PersonalLoan":    0.06,
    "Mortgage":        0.03,
    "SavingsAccount":  0.10,
    "Insurance":       0.05,
    "InvestmentFund":  0.04,
    "OverdraftLimit":  0.07,
}

# Estimated lifetime value per product (normalised)
PRODUCT_VALUE = {
    "CreditCard":      1.0,
    "PersonalLoan":    1.5,
    "Mortgage":        5.0,
    "SavingsAccount":  0.8,
    "Insurance":       2.0,
    "InvestmentFund":  2.5,
    "OverdraftLimit":  0.6,
}

# Channel multiplier on conversion probability
CHANNEL_MULT = {"SMS": 1.0, "Call": 1.4, "Push": 0.7}

# Contact cost (regulatory / fatigue penalty)
CONTACT_COST = 0.05

# Number of actions: 1 no-action + 21 product-channel combos
N_PRODUCTS  = len(PRODUCTS)   # 7
N_CHANNELS  = len(CHANNELS)   # 3
N_OFFERS    = N_PRODUCTS * N_CHANNELS  # 21
N_ACTIONS   = 1 + N_OFFERS    # 22  (0 = no-action)

# Delayed reward window (days)
DELAY_WINDOW = 7


def action_to_offer(action: int):
    """Map flat action index → (product_name, channel_name) or None."""
    if action == 0:
        return None, None
    idx = action - 1
    p   = PRODUCTS[idx // N_CHANNELS]
    c   = CHANNELS[idx % N_CHANNELS]
    return p, c


# ── State space dimensions ──────────────────────────────────────────────────
# Demographics       : age_norm, income_norm, tenure_norm, region_onehot(5)  → 8
# Transaction history: avg_monthly_txn, recency_days_norm, txn_volatility     → 3
# Product holdings   : binary vector over 7 products                          → 7
# CVI                : scalar [0, 1]                                           → 1
# Recent contacts    : count of contacts in last 30 days (norm)               → 1
# Total                                                                        → 20
STATE_DIM = 20


class NBOEnvironment(gym.Env):
    """
    Gymnasium environment simulating a single customer interaction episode.

    Each episode = one customer.
    Each step    = one weekly decision point (up to max_steps weeks).

    Observation : continuous vector of shape (STATE_DIM + N_ACTIONS,)
                  = customer features  +  binary action-availability mask
    Action      : Discrete(N_ACTIONS)  — 0 is no-action
    Reward      : immediate contact cost  +  delayed conversion reward
                  (delayed portion arrives after DELAY_WINDOW steps via buffer)
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 12, seed: int | None = None):
        super().__init__()
        self.max_steps   = max_steps
        self.rng         = np.random.default_rng(seed)

        # ── Spaces ──────────────────────────────────────────────────────────
        # Observation = state features + action mask (appended so SB3 can see it)
        obs_dim = STATE_DIM + N_ACTIONS
        self.observation_space = spaces.Box(
            low   = np.zeros(obs_dim, dtype=np.float32),
            high  = np.ones(obs_dim,  dtype=np.float32),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Internal buffers
        self._state        = None
        self._action_mask  = None
        self._step_count   = 0
        self._reward_queue = []   # [(arrival_step, reward_value)]
        self._holdings     = None
        self._contact_count = 0

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count    = 0
        self._reward_queue  = []
        self._contact_count = 0

        # Randomise a synthetic customer
        self._customer = self._sample_customer()
        self._holdings = self._customer["holdings"].copy()
        self._action_mask = self._sample_action_mask()

        obs = self._build_obs()
        return obs, {"action_mask": self._action_mask.copy()}

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        immediate_reward = 0.0
        info = {}

        product, channel = action_to_offer(action)

        if product is not None:
            # ── Validity check (hard constraint) ────────────────────────────
            if not self._action_mask[action]:
                # Agent chose a masked action — penalise and skip
                immediate_reward = -0.2
                info["invalid_action"] = True
            else:
                immediate_reward -= CONTACT_COST
                self._contact_count += 1

                # ── Simulate conversion with 7-day delay ────────────────────
                conv_prob = self._conversion_prob(product, channel)
                converted = self.rng.random() < conv_prob

                if converted:
                    delayed_r = PRODUCT_VALUE[product]
                    arrival   = self._step_count + DELAY_WINDOW
                    self._reward_queue.append((arrival, delayed_r))

                # Update holdings on conversion
                if converted:
                    p_idx = PRODUCTS.index(product)
                    self._holdings[p_idx] = 1.0

                info["product"]   = product
                info["channel"]   = channel
                info["converted"] = converted

        # ── Drain any rewards whose delay has elapsed ────────────────────────
        pending_next = []
        for (arrival_step, r_val) in self._reward_queue:
            if arrival_step <= self._step_count:
                immediate_reward += r_val      # delayed reward now arrives
            else:
                pending_next.append((arrival_step, r_val))
        self._reward_queue = pending_next

        # ── Advance step & refresh action mask weekly ────────────────────────
        self._step_count += 1
        if self._step_count % 4 == 0:          # rotate campaigns every ~4 weeks
            self._action_mask = self._sample_action_mask()

        terminated = self._step_count >= self.max_steps
        truncated  = False

        # Flush remaining delayed rewards on episode end
        if terminated:
            for (_, r_val) in self._reward_queue:
                immediate_reward += r_val
            self._reward_queue = []

        obs  = self._build_obs()
        info["action_mask"]    = self._action_mask.copy()
        info["contact_count"]  = self._contact_count

        return obs, immediate_reward, terminated, truncated, info

    # ── Action mask (for SB3 MaskablePPO / custom wrappers) ─────────────────
    def action_masks(self) -> np.ndarray:
        """Returns boolean mask of shape (N_ACTIONS,). True = valid."""
        return self._action_mask.copy()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _sample_customer(self) -> dict:
        age        = self.rng.uniform(0, 1)         # normalised age
        income     = self.rng.uniform(0, 1)         # normalised income
        tenure     = self.rng.uniform(0, 1)         # years with bank (norm)
        region     = np.eye(5)[self.rng.integers(5)]  # one-hot region
        avg_txn    = self.rng.uniform(0, 1)
        recency    = self.rng.uniform(0, 1)
        volatility = self.rng.uniform(0, 1)
        holdings   = (self.rng.random(N_PRODUCTS) > 0.7).astype(np.float32)
        cvi        = self.rng.uniform(0, 1)
        return dict(
            age=age, income=income, tenure=tenure, region=region,
            avg_txn=avg_txn, recency=recency, volatility=volatility,
            holdings=holdings, cvi=cvi,
        )

    def _sample_action_mask(self) -> np.ndarray:
        """
        Simulate weekly campaign availability.
        No-action (0) is always valid.
        Each product-channel combo is available with prob 0.75,
        but disabled if customer already holds that product.
        """
        mask = np.ones(N_ACTIONS, dtype=bool)
        for i in range(1, N_ACTIONS):
            product, _ = action_to_offer(i)
            p_idx      = PRODUCTS.index(product)
            # Disable if already held
            if self._holdings[p_idx] == 1.0:
                mask[i] = False
            # Randomly disable ~25% of campaigns (weekly rotation)
            elif self.rng.random() < 0.25:
                mask[i] = False
        return mask

    def _build_obs(self) -> np.ndarray:
        c   = self._customer
        rec = self._holdings          # updated holdings
        state = np.array([
            c["age"], c["income"], c["tenure"],
            *c["region"],             # 5 dims
            c["avg_txn"], c["recency"], c["volatility"],
            *rec,                     # 7 dims
            c["cvi"],
            min(self._contact_count / 10.0, 1.0),  # contact count norm
        ], dtype=np.float32)
        # Append action mask so the policy can see availability
        mask_float = self._action_mask.astype(np.float32)
        return np.concatenate([state, mask_float])

    def _conversion_prob(self, product: str, channel: str) -> float:
        """
        Synthetic conversion model:
        P(convert) = base_prob * channel_mult * income_modifier * cvi_modifier
        CVI is purely a state modifier here — NOT in the reward directly.
        """
        c    = self._customer
        base = BASE_CONV_PROB[product]
        ch   = CHANNEL_MULT[channel]
        # Higher income → more likely mortgage / investment; lower for overdraft
        income_mod = 1.0
        if product in ("Mortgage", "InvestmentFund"):
            income_mod = 0.5 + c["income"]
        elif product == "OverdraftLimit":
            income_mod = 1.5 - c["income"]
        # CVI as state feature modifying probability (not the reward itself)
        cvi_mod = 0.7 + 0.6 * c["cvi"]
        return float(np.clip(base * ch * income_mod * cvi_mod, 0.0, 0.95))
