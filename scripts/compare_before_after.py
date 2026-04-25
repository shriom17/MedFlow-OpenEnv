"""Compare 'before' and 'after' episode logs visually.

Usage:
    python scripts/compare_before_after.py --before_dir outputs/logs/before --after_dir outputs/logs/after --out outputs/plots

Produces:
    - outputs/plots/compare_reward.png

If `scipy` is available, a t-test or Wilcoxon (paired) result will be shown on the plot.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy import stats
except Exception:
    stats = None


def load_totals_and_steps(dirpath: Path):
    files = sorted(dirpath.glob("episode_*.json"))
    totals = []
    per_steps = []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        total = d.get("total_reward") or d.get("reward")
        steps = d.get("steps") or d.get("episode") or d.get("trajectory") or []
        if total is None and steps:
            total = sum(s.get("reward", 0) for s in steps if isinstance(s, dict))
        per_step = [float(s.get("reward", 0)) if isinstance(s, dict) else float(s) for s in steps if (isinstance(s, dict) or isinstance(s, (int, float)))]
        if total is not None:
            totals.append(float(total))
        if per_step:
            per_steps.append(per_step)
    return totals, per_steps


def ensure_out(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)


def mean_std_across(per_steps):
    if not per_steps:
        return [], []
    max_len = max(len(p) for p in per_steps)
    if np is not None:
        arr = np.full((len(per_steps), max_len), np.nan, dtype=float)
        for i, p in enumerate(per_steps):
            arr[i, : len(p)] = p
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        return mean, std
    else:
        sums = [0.0] * max_len
        counts = [0] * max_len
        for p in per_steps:
            for i, v in enumerate(p):
                sums[i] += v
                counts[i] += 1
        mean = [sums[i] / counts[i] if counts[i] else 0.0 for i in range(max_len)]
        std = [0.0] * max_len
        return mean, std


def compare_and_plot(before_totals, after_totals, before_steps, after_steps, out_path: Path):
    ensure_out(out_path)
    plt.figure(figsize=(10, 4))
    # boxplot of totals
    plt.subplot(1, 2, 1)
    data = [before_totals, after_totals]
    plt.boxplot(data, labels=["before", "after"])
    plt.title("Total reward: before vs after")

    # stats
    stat_text = ""
    try:
        if stats is not None and len(before_totals) and len(after_totals):
            if len(before_totals) == len(after_totals):
                # paired
                try:
                    w, p = stats.wilcoxon(before_totals, after_totals)
                    stat_text = f"Wilcoxon p={p:.3g}"
                except Exception:
                    t, p = stats.ttest_rel(before_totals, after_totals)
                    stat_text = f"paired t p={p:.3g}"
            else:
                t, p = stats.ttest_ind(before_totals, after_totals, equal_var=False)
                stat_text = f"ind t p={p:.3g}"
    except Exception:
        stat_text = "stat test failed"

    # per-step mean curves
    plt.subplot(1, 2, 2)
    before_mean, before_std = mean_std_across(before_steps)
    after_mean, after_std = mean_std_across(after_steps)
    x1 = list(range(1, len(before_mean) + 1))
    x2 = list(range(1, len(after_mean) + 1))
    if len(before_mean):
        plt.plot(x1, before_mean, label="before mean")
        try:
            plt.fill_between(x1, [m - s for m, s in zip(before_mean, before_std)], [m + s for m, s in zip(before_mean, before_std)], alpha=0.2)
        except Exception:
            pass
    if len(after_mean):
        plt.plot(x2, after_mean, label="after mean")
        try:
            plt.fill_between(x2, [m - s for m, s in zip(after_mean, after_std)], [m + s for m, s in zip(after_mean, after_std)], alpha=0.2)
        except Exception:
            pass
    plt.xlabel("Step")
    plt.ylabel("Reward")
    title = "Mean per-step reward"
    if stat_text:
        title = f"{title} — {stat_text}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "compare_reward.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before_dir", type=Path, required=True)
    parser.add_argument("--after_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()
    if not args.before_dir.exists():
        print(f"Before dir not found: {args.before_dir}")
        return
    if not args.after_dir.exists():
        print(f"After dir not found: {args.after_dir}")
        return
    before_totals, before_steps = load_totals_and_steps(args.before_dir)
    after_totals, after_steps = load_totals_and_steps(args.after_dir)
    print(f"Loaded {len(before_totals)} before totals, {len(after_totals)} after totals")
    compare_and_plot(before_totals, after_totals, before_steps, after_steps, args.out)
    print(f"Saved comparison plot to {args.out / 'compare_reward.png'}")


if __name__ == "__main__":
    main()
