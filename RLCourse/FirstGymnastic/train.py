"""
train.py — Run the NBO Bandit experiment
-----------------------------------------
Usage:
    python train.py                     # default 2000 episodes
    python train.py --episodes 5000 --eval-every 200
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from RLCourse.FirstGymnastic.nbo_env import NBOEnvironment, N_ACTIONS, STATE_DIM
from RLCourse.FirstGymnastic.nbo_bandit import NBOBanditAgent, BanditConfig


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(agent: NBOBanditAgent, n_episodes: int = 100, seed: int = 42) -> dict:
    """Run greedy policy (no Thompson noise) and collect metrics."""
    env     = NBOEnvironment(seed=seed)
    metrics = defaultdict(list)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        mask      = info["action_mask"]
        ep_reward = 0.0
        ep_conversions  = 0
        ep_contacts     = 0
        ep_no_actions   = 0

        while True:
            action              = agent.greedy_action(obs, mask)
            obs, reward, done, _, info = env.step(action)
            mask                = info["action_mask"]
            ep_reward          += reward
            ep_contacts         = info["contact_count"]
            if reward > 0.0:
                ep_conversions += 1
            if action == 0:
                ep_no_actions += 1
            if done:
                break

        metrics["reward"].append(ep_reward)
        metrics["conversions"].append(ep_conversions)
        metrics["contacts"].append(ep_contacts)
        metrics["no_action_rate"].append(ep_no_actions / env.max_steps)

    return {k: np.mean(v) for k, v in metrics.items()}


# ── Training loop ─────────────────────────────────────────────────────────────
def train(n_episodes: int = 2000, eval_every: int = 200, seed: int = 0):
    cfg   = BanditConfig(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    agent = NBOBanditAgent(cfg)
    env   = NBOEnvironment(seed=seed)

    history = defaultdict(list)
    losses  = []

    print(f"{'Episode':>8}  {'AvgReward':>10}  {'Conversions':>12}  "
          f"{'Contacts':>9}  {'NoActionRate':>13}")
    print("-" * 60)

    for ep in range(n_episodes):
        obs, info = env.reset()
        mask      = info["action_mask"]
        ep_reward = 0.0

        while True:
            action                     = agent.select_action(obs, mask)
            next_obs, reward, done, _, info = env.step(action)
            next_mask                  = info["action_mask"]

            loss = agent.store(obs, action, reward, mask)
            if loss is not None:
                losses.append(loss)

            obs, mask  = next_obs, next_mask
            ep_reward += reward
            if done:
                break

        history["train_reward"].append(ep_reward)

        # ── Periodic evaluation ──────────────────────────────────────────────
        if (ep + 1) % eval_every == 0:
            metrics = evaluate(agent, n_episodes=100)
            for k, v in metrics.items():
                history[f"eval_{k}"].append(v)

            avg_loss = np.mean(losses[-200:]) if losses else float("nan")
            print(
                f"{ep+1:>8}  {metrics['reward']:>10.3f}  "
                f"{metrics['conversions']:>12.3f}  "
                f"{metrics['contacts']:>9.2f}  "
                f"{metrics['no_action_rate']:>13.3f}  "
                f"loss={avg_loss:.4f}"
            )

    return agent, history, losses


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(history: dict, losses: list, save_path: str = "results.png"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("NBO Neural Contextual Bandit — Training Results", fontsize=14)

    # 1. Training reward (smoothed)
    ax = axes[0, 0]
    r  = np.array(history["train_reward"])
    window = max(1, len(r) // 50)
    smoothed = np.convolve(r, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, color="#2196F3")
    ax.set_title("Training Episode Reward (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True, alpha=0.3)

    # 2. Eval reward
    ax = axes[0, 1]
    if "eval_reward" in history:
        ax.plot(history["eval_reward"], marker="o", color="#4CAF50")
        ax.set_title("Eval Average Reward")
        ax.set_xlabel("Eval checkpoint")
        ax.set_ylabel("Avg Reward (100 eps)")
        ax.grid(True, alpha=0.3)

    # 3. Conversions
    ax = axes[0, 2]
    if "eval_conversions" in history:
        ax.plot(history["eval_conversions"], marker="s", color="#FF9800")
        ax.set_title("Eval Average Conversions / Episode")
        ax.set_xlabel("Eval checkpoint")
        ax.set_ylabel("Conversions")
        ax.grid(True, alpha=0.3)

    # 4. Training loss
    ax = axes[1, 0]
    if losses:
        w = max(1, len(losses) // 50)
        sl = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(sl, color="#E91E63", alpha=0.8)
        ax.set_title("Huber Loss (smoothed)")
        ax.set_xlabel("Update step")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # 5. No-action rate
    ax = axes[1, 1]
    if "eval_no_action_rate" in history:
        ax.plot(history["eval_no_action_rate"], marker="^", color="#9C27B0")
        ax.set_title("No-Action Rate (eval)")
        ax.set_xlabel("Eval checkpoint")
        ax.set_ylabel("Fraction of steps")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # 6. Contact rate
    ax = axes[1, 2]
    if "eval_contacts" in history:
        ax.plot(history["eval_contacts"], marker="D", color="#00BCD4")
        ax.set_title("Avg Contacts / Episode (eval)")
        ax.set_xlabel("Eval checkpoint")
        ax.set_ylabel("# contacts")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    agent, history, losses = train(args.episodes, args.eval_every, args.seed)
    plot_results(history, losses, save_path="results.png")

    # Final evaluation
    print("\n=== Final Evaluation (200 episodes, greedy policy) ===")
    final = evaluate(agent, n_episodes=200)
    for k, v in final.items():
        print(f"  {k:20s}: {v:.4f}")
