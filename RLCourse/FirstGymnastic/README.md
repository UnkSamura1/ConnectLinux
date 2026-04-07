# NBO Neural Contextual Bandit — Prototype

A minimal but complete sandbox for testing Next Best Offer ideas in a
retail banking setting. No real customer data required.

## Files

| File | Purpose |
|---|---|
| `nbo_env.py` | Custom Gymnasium environment |
| `nbo_bandit.py` | Neural Contextual Bandit agent (Thompson Sampling) |
| `train.py` | Training + evaluation + plotting loop |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
python train.py                          # 2 000 episodes (fast, ~2 min)
python train.py --episodes 5000 --eval-every 200
```

## What the environment models

- **7 products × 3 channels = 21 offers + 1 no-action** = 22 discrete actions
- **Dynamic action masking**: ~25 % of campaigns randomly disabled each week;
  products the customer already holds are always masked
- **7-day delayed reward**: conversions are added to a queue and
  released after `DELAY_WINDOW` steps
- **State (20 dims)**: demographics, transaction history, product holdings,
  CVI (monthly scalar), contact count
- **CVI is in state only** — never part of the reward function

## What the agent does

- Shared MLP encoder (256 → 256 → 128) learns customer representations
- Per-arm linear heads estimate expected reward per offer
- **Thompson Sampling**: Gaussian noise added to arm weights at inference;
  noise scale controlled by `BanditConfig.ts_noise`
- Invalid (masked) actions are excluded before arm selection
- Huber loss on observed (delayed) rewards; replay buffer of 10 k transitions

## Experiments to try

| Idea | How to test |
|---|---|
| Reward hacking via CVI | Move `cvi` from state into reward in `_conversion_prob` and compare eval curves |
| No-action impact | Set `CONTACT_COST = 0` and watch no-action rate collapse |
| Delay sensitivity | Change `DELAY_WINDOW` from 0 → 14 and observe convergence speed |
| Weekly mask churn | Change 0.25 → 0.60 in `_sample_action_mask` for aggressive rotation |
| Thompson noise | Vary `ts_noise` in BanditConfig (0.0 = greedy, 0.5 = high exploration) |

## Next upgrade: Offline RL (IQL)

Once you validate the bandit converges and want sequential credit assignment:
1. Collect bandit interaction logs (state, action, reward, next_state)
2. Replace `nbo_bandit.py` with an IQL agent using the same `NBOEnvironment`
3. Compare eval reward curves — if IQL > bandit by >5 %, the MDP assumption is worth it
