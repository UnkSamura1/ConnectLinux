"""
Neural Contextual Bandit — NBO Agent
--------------------------------------
Architecture:
  - Shared MLP encoder   : state → embedding (128-d)
  - Per-arm linear heads : embedding → scalar reward estimate
  - Exploration          : Thompson Sampling via diagonal posterior on last layer
  - Action masking       : invalid actions excluded before arm selection

This is intentionally SB3-compatible in spirit but self-contained so
you can inspect every component without framework magic.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ── Config ───────────────────────────────────────────────────────────────────
@dataclass
class BanditConfig:
    state_dim:     int   = 20          # pure customer features (no mask appended)
    n_actions:     int   = 22          # including no-action
    embed_dim:     int   = 128         # shared encoder output dimension
    hidden_dim:    int   = 256         # hidden layer width
    n_hidden:      int   = 2           # number of hidden layers in encoder
    lr:            float = 3e-4        # Adam learning rate
    buffer_size:   int   = 10_000      # replay buffer capacity
    batch_size:    int   = 256
    ts_noise:      float = 0.1         # Thompson Sampling std multiplier
    train_every:   int   = 32          # update every N steps
    gamma:         float = 0.95        # not used in pure bandit; kept for n-step
    device:        str   = "cpu"


# ── Replay Buffer ────────────────────────────────────────────────────────────
@dataclass
class Transition:
    state:   np.ndarray
    action:  int
    reward:  float
    mask:    np.ndarray   # boolean action mask at decision time


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self._buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        idx = np.random.choice(len(self._buf), size=batch_size, replace=False)
        return [self._buf[i] for i in idx]

    def __len__(self):
        return len(self._buf)


# ── Neural Contextual Bandit ─────────────────────────────────────────────────
class NeuralContextualBandit(nn.Module):
    """
    Shared encoder + independent linear head per arm.
    Thompson Sampling: at inference, add Gaussian noise scaled by
    ts_noise to each arm's weight vector before computing Q-value.
    """

    def __init__(self, cfg: BanditConfig):
        super().__init__()
        self.cfg = cfg

        # Shared MLP encoder
        layers = []
        in_dim = cfg.state_dim
        for _ in range(cfg.n_hidden):
            layers += [nn.Linear(in_dim, cfg.hidden_dim), nn.ReLU()]
            in_dim  = cfg.hidden_dim
        layers += [nn.Linear(in_dim, cfg.embed_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

        # Per-arm linear heads (no bias keeps it as pure dot product)
        self.arm_heads = nn.ModuleList([
            nn.Linear(cfg.embed_dim, 1, bias=True)
            for _ in range(cfg.n_actions)
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
        Returns:
            q_values: (B, n_actions)
        """
        emb = self.encoder(state)                          # (B, embed_dim)
        qs  = [head(emb) for head in self.arm_heads]      # n_actions × (B, 1)
        return torch.cat(qs, dim=1)                        # (B, n_actions)

    @torch.no_grad()
    def thompson_sample(
        self,
        state: np.ndarray,
        mask:  np.ndarray,
        noise: float,
    ) -> int:
        """
        Select action via Thompson Sampling.
        Adds Gaussian noise to arm weights before scoring, then masks.
        """
        s_t  = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        emb  = self.encoder(s_t)                           # (1, embed_dim)

        scores = np.full(self.cfg.n_actions, -np.inf)
        for a in range(self.cfg.n_actions):
            if not mask[a]:
                continue
            # Perturb weights for Thompson Sampling
            w     = self.arm_heads[a].weight.data.clone()
            b_val = self.arm_heads[a].bias.data.clone()
            w_perturbed = w + noise * torch.randn_like(w)
            score = (emb @ w_perturbed.T + b_val).item()
            scores[a] = score

        return int(np.argmax(scores))


# ── Training Loop ────────────────────────────────────────────────────────────
class NBOBanditAgent:
    """
    Wraps NeuralContextualBandit with a replay buffer,
    Adam optimiser, and a training loop compatible with
    our NBOEnvironment.
    """

    def __init__(self, cfg: Optional[BanditConfig] = None):
        self.cfg    = cfg or BanditConfig()
        self.device = torch.device(self.cfg.device)
        self.model  = NeuralContextualBandit(self.cfg).to(self.device)
        self.optim  = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.loss_fn = nn.HuberLoss()
        self.buffer  = ReplayBuffer(self.cfg.buffer_size)
        self._steps  = 0

    # ── Select action ────────────────────────────────────────────────────────
    def select_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        # obs has STATE_DIM + N_ACTIONS dims; we only pass state part to model
        state = obs[:self.cfg.state_dim]
        return self.model.thompson_sample(state, mask, self.cfg.ts_noise)

    # ── Store transition ─────────────────────────────────────────────────────
    def store(self, obs: np.ndarray, action: int, reward: float, mask: np.ndarray):
        state = obs[:self.cfg.state_dim]
        self.buffer.push(Transition(state, action, reward, mask))
        self._steps += 1
        if (self._steps % self.cfg.train_every == 0
                and len(self.buffer) >= self.cfg.batch_size):
            loss = self._train_step()
            return loss
        return None

    # ── One gradient update ──────────────────────────────────────────────────
    def _train_step(self) -> float:
        batch  = self.buffer.sample(self.cfg.batch_size)
        states  = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32
        ).to(self.device)

        # Q-values for all arms, then gather the taken action's Q
        q_all    = self.model(states)                      # (B, n_actions)
        q_taken  = q_all.gather(1, actions.unsqueeze(1).to(self.device)).squeeze(1)

        # Contextual bandit target = observed (potentially delayed) reward
        loss = self.loss_fn(q_taken, rewards)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        return loss.item()

    # ── Greedy evaluation (no noise) ────────────────────────────────────────
    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        state  = obs[:self.cfg.state_dim]
        s_t    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_all  = self.model(s_t).squeeze(0).cpu().numpy()
        q_all[~mask] = -np.inf
        return int(np.argmax(q_all))
