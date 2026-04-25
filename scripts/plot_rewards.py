"""Generate reward plots from episode JSON logs.

Usage:
    python scripts/plot_rewards.py --logs outputs/logs --out outputs/plots

Produces:
    - outputs/plots/reward_curve.png  (mean per-step reward curve)
    - outputs/plots/reward_hist.png   (histogram / boxplot of total rewards)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import matplotlib.pyplot as plt

try:
    import numpy as np
except Exception:
    np = None


def load_episodes(log_dir: Path):
    files = sorted(log_dir.glob("episode_*.json"))
    episodes = []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # try common shapes
        total = d.get("total_reward") if isinstance(d, dict) else None
        if total is None:
            total = d.get("reward") if isinstance(d, dict) else None
        steps = d.get("steps") or d.get("episode") or d.get("trajectory") or []
        if total is None and steps:
            total = sum(s.get("reward", 0) for s in steps if isinstance(s, dict))
        # extract per-step rewards
        per_step = []
        if steps and isinstance(steps, list):
            for s in steps:
                if isinstance(s, dict):
                    per_step.append(float(s.get("reward", 0)))
                else:
                    # if step is numeric
                    try:
                        per_step.append(float(s))
                    except Exception:
                        pass

        episodes.append({"name": p.name, "total": total, "per_step": per_step})
    return episodes


def ensure_out(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)


def plot_reward_curve(episodes, out_path: Path):
    # build aligned per-step arrays
    per_steps = [e["per_step"] for e in episodes if e["per_step"]]
    if not per_steps:
        print("No per-step data found to plot reward curve.")
        return
    max_len = max(len(p) for p in per_steps)
    if np is not None:
        arr = np.full((len(per_steps), max_len), np.nan, dtype=float)
        for i, p in enumerate(per_steps):
            arr[i, : len(p)] = p
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        x = list(range(1, len(mean) + 1))
    else:
        # fallback: compute mean manually
        sums = [0.0] * max_len
        counts = [0] * max_len
        for p in per_steps:
            for i, v in enumerate(p):
                sums[i] += v
                counts[i] += 1
        mean = [sums[i] / counts[i] if counts[i] else 0.0 for i in range(max_len)]
        std = [0.0] * max_len
        x = list(range(1, max_len + 1))

    plt.figure(figsize=(9, 4))
    plt.plot(x, mean, label="mean reward per step")
    try:
        plt.fill_between(x, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], alpha=0.2)
    except Exception:
        pass
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Mean per-step reward (shaded = std)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "reward_curve.png", dpi=150)
    plt.close()


def plot_total_hist(episodes, out_path: Path):
    totals = [e["total"] for e in episodes if e["total"] is not None]
    if not totals:
        print("No total reward data found to plot histogram.")
        return
    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    plt.hist(totals, bins=min(30, max(5, len(totals)//2)), alpha=0.7)
    plt.xlabel("Total reward")
    plt.ylabel("Count")
    plt.title("Total reward distribution")

    plt.subplot(1, 2, 2)
    plt.boxplot(totals, vert=False)
    plt.title("Total reward (box)")
    plt.tight_layout()
    plt.savefig(out_path / "reward_hist.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()
    if not args.logs.exists():
        print(f"Logs directory not found: {args.logs}")
        return
    ensure_out(args.out)
    episodes = load_episodes(args.logs)
    print(f"Found {len(episodes)} episode files in {args.logs}")
    plot_reward_curve(episodes, args.out)
    plot_total_hist(episodes, args.out)
    print(f"Saved plots to {args.out}")


if __name__ == "__main__":
    main()
