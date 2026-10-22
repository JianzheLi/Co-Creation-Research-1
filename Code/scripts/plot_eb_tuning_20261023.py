"""Plot the completed EB tuning evidence from the report's CSV artifacts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "Doc" / "Experiment2"


def main():
    reconstruction = pd.read_csv(DOC / "eb_tuning_20261023_seed_metrics.csv")
    reconstruction = reconstruction.loc[reconstruction.time > 0]
    per_training_seed = reconstruction.groupby(["training_seed", "time"]).W1.mean()
    mean = per_training_seed.groupby("time").mean()
    sd = per_training_seed.groupby("time").std(ddof=1)
    holdout = pd.read_csv(DOC / "eb_tuning_20261023_holdout.csv").sort_values("heldout_time")
    baseline_reconstruction = pd.read_csv(DOC / "all_timepoint_metrics.csv")
    baseline_reconstruction = baseline_reconstruction.loc[
        (baseline_reconstruction.dataset == "eb_noscale") & (baseline_reconstruction.time > 0)]
    baseline_holdout = pd.read_csv(REPO / "Doc" / "Experiment3" / "baseline_holdout_w1.csv")
    baseline_holdout = baseline_holdout.loc[baseline_holdout.dataset == "eb_noscale"]

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.27, wspace=0.23)
    ours_color = "#0072B2"
    axes[0].plot(mean.index, mean.values, "o-", color=ours_color, lw=2.3, label="Ours: selected recipe")
    axes[0].fill_between(mean.index, mean - sd, mean + sd, color=ours_color, alpha=0.18)
    axes[1].plot(holdout.heldout_time, holdout.W1, "o-", color=ours_color, lw=2.3)
    for rec_name, hold_name, label, color, marker in [
        ("wfr_fm", "wfr_fm", "WFR-FM", "#D55E00", "s"),
        ("ot_cfm", "ot_cfm", "OT-CFM", "#009E73", "^"),
        ("vgfm", "VGFM", "VGFM", "#CC79A7", "D"),
    ]:
        rec = baseline_reconstruction.loc[baseline_reconstruction.method == rec_name].sort_values("time")
        held = baseline_holdout.loc[baseline_holdout.method == hold_name].sort_values("heldout_time")
        axes[0].plot(rec.time, rec.W1, marker=marker, color=color, lw=1.6, label=label)
        axes[1].plot(held.heldout_time, held.W1, marker=marker, color=color, lw=1.6)

    axes[0].set(title="Reconstruction: 3 Matching training seeds", xlabel="Observed time", xticks=[1, 2, 3, 4])
    axes[1].set(title="Holdout transfer: one full retraining per split", xlabel="Held-out time", xticks=[1, 2, 3])
    for ax in axes:
        ax.set_ylabel("Full empirical W1 (lower is better)")
        ax.grid(axis="y", alpha=0.23)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=4, frameon=False)
    fig.suptitle("EB: 100 dimensions, no additional normalization", fontsize=13, y=0.97)
    fig.text(0.5, 0.025,
             "Shading: ±1 SD across Matching seeds, with a fixed path and couplings. Baselines are archived runs.\n"
             "The recipe was selected on reconstruction before evaluating holdout predictions; budgets are not matched.",
             ha="center", va="bottom", fontsize=8)
    output = DOC / "Figures"
    output.mkdir(exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"eb_tuning_20261023.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
