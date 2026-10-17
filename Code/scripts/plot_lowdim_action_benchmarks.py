"""Plot the generated low-dimensional benchmark snapshots for inspection."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lowdim_action_benchmarks"
OUT = ROOT / "../Doc/Experiment2/Figures" / "lowdim_action_benchmarks.png"
DATASETS = [
    ("gaussian_translation", "Translating Gaussian"),
    ("gaussian_breathing", "Breathing Gaussian"),
    ("gmm_split", "Splitting Gaussian mixture"),
    ("gmm_rotate_reweight", "Rotating / reweighted mixture"),
    ("double_well_sde", "Double-well SDE"),
    ("ou_sde_5d", "5D OU-SDE (x1-x2 projection)"),
    ("two_moons_morph", "Two-moons shape / mass change"),
]
COLORS = ["#5e3c99", "#3288bd", "#66c2a5", "#fdae61"]


def main():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, (name, title) in zip(axes.flat, DATASETS):
        frame = pd.read_csv(DATA / f"{name}.csv")
        for time, color in enumerate(COLORS):
            subset = frame.loc[frame["samples"] == time]
            ax.scatter(subset["x1"], subset["x2"], s=8, alpha=0.55,
                       color=color, label=f"t={time}")
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.15)
        ax.set_aspect("equal", adjustable="datalim")
    axes.flat[len(DATASETS)].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=4, frameon=False)
    fig.suptitle("Low-dimensional action-comparison benchmarks", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
