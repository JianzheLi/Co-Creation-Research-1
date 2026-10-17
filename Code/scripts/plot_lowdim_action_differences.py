"""Plot paired Monte Carlo Action gaps for the low-dimensional benchmark suite."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "../Results" / "lowdim_action_benchmarks"
OUT = ROOT / "../Doc/Experiment2/Figures" / "lowdim_action_differences.png"
MC_DIR = "action_monte_carlo_20_seed1000"
NAMES = {
    "gaussian_translation": "Translating Gaussian",
    "gaussian_breathing": "Breathing Gaussian",
    "gmm_split": "Splitting Gaussian mixture",
    "gmm_rotate_reweight": "Rotating / reweighted GMM",
    "double_well_sde": "Double-well SDE",
    "ou_sde_5d": "5D OU-SDE",
    "two_moons_morph": "Two-moons morph",
}


def main():
    rows = []
    for summary_path in sorted(RUNS.glob(f"*/{MC_DIR}/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run = summary_path.parent.parent
        ruot_mean = summary["method_summaries"]["RUOT"]["total_action"]["mean"]
        paired = summary["paired_wfr_minus_ruot"]["total_action"]
        with (summary_path.parent / "replicates.csv").open(newline="", encoding="utf-8-sig") as handle:
            raw = list(csv.DictReader(handle))
        by_seed = {}
        for item in raw:
            by_seed.setdefault(int(item["seed"]), {})[item["method"]] = float(item["total_action"])
        delta_pct = np.asarray([
            100 * (item["WFR-FM bridge"] - item["RUOT"]) / ruot_mean
            for item in by_seed.values()
        ])
        base, alpha = run.name.rsplit("_alpha", 1)
        title = NAMES.get(base, base.replace("_", " "))
        ci = 100 * np.asarray(paired["ci95"], dtype=np.float64) / ruot_mean
        rows.append({"label": f"{title}, α={alpha.replace('p', '.')}\n",
                     "mean_pct": 100 * paired["mean"] / ruot_mean,
                     "ci_pct": ci, "values": delta_pct,
                     "clear": abs(100 * paired["mean"] / ruot_mean) >= 1.0
                     and (paired["ci95"][0] > 0 or paired["ci95"][1] < 0)})
    rows.sort(key=lambda row: row["mean_pct"])
    if not rows:
        raise SystemExit("No paired Action summaries found.")

    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.78 * len(rows) + 1.8)))
    positions = np.arange(1, len(rows) + 1)
    data = [row["values"] for row in rows]
    artists = ax.boxplot(data, vert=False, positions=positions, widths=0.52,
                         patch_artist=True, showfliers=False, manage_ticks=False,
                         medianprops={"color": "black", "linewidth": 1.4},
                         whiskerprops={"color": "#555555"},
                         capprops={"color": "#555555"})
    jitter = np.random.default_rng(43)
    for position, row, box in zip(positions, rows, artists["boxes"]):
        color = "#66bd63" if row["clear"] else "#bdbdbd"
        box.set_facecolor(color)
        box.set_alpha(0.72)
        y = position + jitter.uniform(-0.10, 0.10, size=len(row["values"]))
        ax.scatter(row["values"], y, s=15, color=color, edgecolor="white",
                   linewidth=0.25, alpha=0.8, zorder=3)
        lo, hi = row["ci_pct"]
        ax.errorbar(row["mean_pct"], position, xerr=[[row["mean_pct"] - lo], [hi - row["mean_pct"]]],
                    fmt="D", markersize=4.5, color="#1f1f1f", ecolor="#1f1f1f",
                    elinewidth=1.2, capsize=3, zorder=4)
    ax.set_yticks(positions, [row["label"] for row in rows])
    ax.axvline(0, color="#555555", linewidth=1)
    ax.axvline(-1, color="#a66a00", linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(1, color="#a66a00", linestyle="--", linewidth=1, alpha=0.8)
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlabel("Paired Action difference: WFR-FM − RUOT (% of RUOT mean)")
    ax.set_title("20 paired Monte Carlo rollouts per data / α setting")
    ax.invert_yaxis()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
