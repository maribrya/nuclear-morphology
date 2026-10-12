from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def plot_recall():
    summary = json.loads((ROOT / "restoration_summary.json").read_text())
    row = summary["paired_comparisons"][-1]
    groups = ["typical", "rare"]
    values = []
    low = []
    high = []
    for group in groups:
        gain = row[group + "_recall_gain"] * 100
        interval = [x * 100 for x in row[group + "_recall_gain_ci95"]]
        values.append(gain)
        low.append(gain - interval[0])
        high.append(interval[1] - gain)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.errorbar(values, [1, 0], xerr=[low, high], fmt="o", color="#243e5c", capsize=5)
    ax.axvline(0, color="0.6", linewidth=1)
    ax.set_yticks([1, 0], ["Typical nuclei", "Elongated nuclei"])
    ax.set_xlabel("Recall gain after restoration, percentage points")
    ax.set_title("Strong blur: paired 95% intervals")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    plot_recall().savefig(folder / "recall.png", dpi=160)
