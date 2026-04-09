import re
import matplotlib.pyplot as plt

def _read_log_text(path: str) -> str:
    # PowerShell redirection often creates UTF-16LE files; fall back to UTF-8.
    for enc in ("utf-16", "utf-8", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeError:
            continue
    with open(path, "r", errors="ignore") as f:
        return f.read()


data = _read_log_text("output.txt")

# extract rewards
rewards = [float(x) for x in re.findall(r"reward=([-0-9.]+)", data)]

if not rewards:
    print("No reward values found in output.txt. Re-generate log with: python inference.py ... > output.txt")
    raise SystemExit(1)

plt.figure(figsize=(8, 4.5))
plt.plot(rewards, marker="o", markersize=2, linewidth=1)
plt.title("Reward vs Steps")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reward_plot.png")

print(f"Graph saved as reward_plot.png (points={len(rewards)})")