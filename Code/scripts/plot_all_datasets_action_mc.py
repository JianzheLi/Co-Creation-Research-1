from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SIGMAS = [0.1, 0.25, 0.5, 1.0]
METHODS = ["RUOT", "WFR+Brownian bridge", "WFR coupling + Brownian bridge (log-linear mass)"]
COLORS = {"RUOT": "#2878B5", "WFR+Brownian bridge": "#E87532",
          "WFR coupling + Brownian bridge (log-linear mass)": "#2A9D72"}
DATASETS = [
    ("Simulation Gene", "gene"),
    ("Mouse", "mouse"),
    ("EMT", "emt"),
    ("WFR-FM Dygen", "dygen"),
]


def read_action_samples(root: Path, datasets=DATASETS) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_label, dataset in datasets:
        for sigma in SIGMAS:
            sigma_tag = str(sigma).replace(".", "p")
            path = root / dataset / f"sigma_{sigma_tag}" / "action_replicates.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(newline="", encoding="utf-8-sig") as f:
                current = list(csv.DictReader(f))
            by_method: dict[str, dict[int, dict[str, str]]] = {m: {} for m in METHODS}
            for row in current:
                method = row["method"]
                if method not in by_method:
                    continue
                seed = int(row["seed"])
                by_method[method][seed] = row
                finite = row["finite"].strip().lower() in {"true", "1", "yes"}
                action = float(row["total_action"])
                if not finite or not np.isfinite(action) or action <= 0:
                    raise ValueError(f"Non-finite/non-positive Action: {path}, {method}, seed={seed}")
                rows.append(
                    {
                        "dataset": dataset_label,
                        "dataset_id": dataset,
                        "sigma": sigma,
                        "method": method,
                        "seed": seed,
                        "Action": action,
                    }
                )
            ruot_seeds = set(by_method["RUOT"])
            wfr_seeds = set(by_method["WFR+Brownian bridge"])
            bridge_seeds = set(by_method[METHODS[2]])
            if (len(ruot_seeds) != 20 or ruot_seeds != wfr_seeds
                    or ruot_seeds != bridge_seeds):
                raise ValueError(
                    f"Expected 20 paired seeds for {dataset}, σ={sigma}; "
                    f"RUOT={len(ruot_seeds)}, WFR={len(wfr_seeds)}, bridge={len(bridge_seeds)}"
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../Results/experiment1_alpha2_sigma_sweep"),
    )
    parser.add_argument("--datasets", nargs="+", choices=[d[1] for d in DATASETS], default=[d[1] for d in DATASETS])
    args = parser.parse_args()
    root = args.output_root
    selected_ids = set(args.datasets)
    datasets = [item for item in DATASETS if item[1] in selected_ids]
    if not datasets:
        raise SystemExit("Select at least one dataset.")
    focused_ids = {"gene", "emt", "dygen"}
    suffix = "" if len(datasets) == len(DATASETS) else ("_three_datasets" if selected_ids == focused_ids else "_focused")
    rows = read_action_samples(root, datasets)

    csv_name = "action_mc_samples_all_datasets.csv" if not suffix else f"action_mc_samples{suffix}.csv"
    csv_path = root / csv_name
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS[m], label=m, markersize=7)
        for m in METHODS
    ]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(15.5, 4.15 * len(datasets)), squeeze=False, constrained_layout=True)
    positions = {sigma: i + 1 for i, sigma in enumerate(SIGMAS)}
    method_offsets = {"RUOT": -0.23, "WFR+Brownian bridge": 0.0,
                      METHODS[2]: 0.23}

    for row_idx, (dataset_label, dataset) in enumerate(datasets):
        subset = [r for r in rows if r["dataset_id"] == dataset]
        ax_scatter, ax_box = axes[row_idx]

        # Paired seed segments make the within-seed method comparison visible.
        for sigma in SIGMAS:
            group = [r for r in subset if r["sigma"] == sigma]
            by_method = {
                method: {int(r["seed"]): float(r["Action"]) for r in group if r["method"] == method}
                for method in METHODS
            }
            ordered_seeds = sorted(by_method["RUOT"])
            jitter = np.linspace(-0.055, 0.055, len(ordered_seeds))
            for seed, dx in zip(ordered_seeds, jitter):
                x_ruot = positions[sigma] + method_offsets["RUOT"] + dx
                y_ruot = by_method["RUOT"][seed]
                for method in METHODS[1:]:
                    x_method = positions[sigma] + method_offsets[method] + dx
                    y_method = by_method[method][seed]
                    ax_scatter.plot([x_ruot, x_method], [y_ruot, y_method], color="#8C9299", alpha=0.18, linewidth=0.55, zorder=1)
                    ax_scatter.scatter(x_method, y_method, color=COLORS[method], s=20, alpha=0.78, zorder=3)
                ax_scatter.scatter(x_ruot, y_ruot, color=COLORS["RUOT"], s=20, alpha=0.78, zorder=3)

        ax_scatter.set_title(dataset_label, loc="left", fontweight="bold")
        ax_scatter.set_ylabel("Rollout Action (log scale)")
        ax_scatter.set_yscale("log")
        ax_scatter.set_xticks(list(positions.values()), [f"{s:g}" for s in SIGMAS])
        ax_scatter.set_xlim(0.5, 4.5)
        ax_scatter.grid(axis="y", which="both", alpha=0.22)
        ax_scatter.legend(handles=method_handles, loc="upper left", fontsize=8, frameon=True)

        data: list[list[float]] = []
        box_positions: list[float] = []
        box_methods: list[str] = []
        for sigma in SIGMAS:
            for method in METHODS:
                vals = [
                    float(r["Action"])
                    for r in subset
                    if r["sigma"] == sigma and r["method"] == method
                ]
                data.append(vals)
                box_positions.append(positions[sigma] + method_offsets[method])
                box_methods.append(method)
        bp = ax_box.boxplot(
            data,
            positions=box_positions,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.4},
            whiskerprops={"color": "#555555", "linewidth": 0.9},
            capprops={"color": "#555555", "linewidth": 0.9},
        )
        for patch, method in zip(bp["boxes"], box_methods):
            patch.set_facecolor(COLORS[method])
            patch.set_edgecolor(COLORS[method])
            patch.set_alpha(0.38)
        for pos, vals, method in zip(box_positions, data, box_methods):
            jitter = np.linspace(-0.052, 0.052, len(vals))
            ax_box.scatter(
                [pos + dx for dx in jitter],
                vals,
                color=COLORS[method],
                edgecolors="white",
                linewidths=0.35,
                s=18,
                alpha=0.78,
                zorder=3,
            )
        ax_box.set_title(f"{dataset_label} (20 paired rollouts per method/σ)", loc="left", fontweight="bold")
        ax_box.set_ylabel("Rollout Action (log scale)")
        ax_box.set_yscale("log")
        ax_box.set_xticks(list(positions.values()), [f"{s:g}" for s in SIGMAS])
        ax_box.set_xlim(0.5, 4.5)
        ax_box.grid(axis="y", which="both", alpha=0.22)
        ax_box.legend(handles=method_handles, loc="upper left", fontsize=8, frameon=True)

    axes[-1, 0].set_xlabel("σ")
    axes[-1, 1].set_xlabel("σ")
    fig.suptitle(
        f"Experiment 1: paired Monte Carlo Action across {len(datasets)} datasets (α=2)",
        fontsize=15,
    )
    fig.text(
        0.5,
        -0.004,
        "Each point is one rollout seed (20 paired seeds per method and σ); gray segments connect RUOT to each baseline. "
        "Each box summarizes those 20 Action values. One training seed (42) was used per dataset and σ.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    scatter_path = root / f"all_datasets_action_scatter{suffix}.png"
    boxplot_path = root / f"all_datasets_action_boxplot{suffix}.png"
    # Both panels are saved separately too, so each plot is legible at full size.
    combined_path = root / f"all_datasets_action_scatter_boxplot{suffix}.png"
    fig.savefig(combined_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Standalone scatter plot, retaining the paired-seed connectors.
    # Crop the left column from the composite into a fresh figure for comfortable viewing.
    fig_s, axes_s = plt.subplots(len(datasets), 1, figsize=(8.5, 3.9 * len(datasets)), squeeze=False, constrained_layout=True)
    for row_idx, (dataset_label, dataset) in enumerate(datasets):
        ax = axes_s[row_idx, 0]
        subset = [r for r in rows if r["dataset_id"] == dataset]
        for sigma in SIGMAS:
            group = [r for r in subset if r["sigma"] == sigma]
            by_method = {
                method: {int(r["seed"]): float(r["Action"]) for r in group if r["method"] == method}
                for method in METHODS
            }
            seeds = sorted(by_method["RUOT"])
            for seed, dx in zip(seeds, np.linspace(-0.055, 0.055, len(seeds))):
                x1 = positions[sigma] + method_offsets["RUOT"] + dx
                y1 = by_method["RUOT"][seed]
                for method in METHODS[1:]:
                    x2 = positions[sigma] + method_offsets[method] + dx
                    y2 = by_method[method][seed]
                    ax.plot([x1, x2], [y1, y2], color="#8C9299", alpha=0.18, linewidth=0.55, zorder=1)
                    ax.scatter(x2, y2, color=COLORS[method], s=20, alpha=0.78, zorder=3)
                ax.scatter(x1, y1, color=COLORS["RUOT"], s=20, alpha=0.78, zorder=3)
        ax.set_title(dataset_label, loc="left", fontweight="bold")
        ax.set_ylabel("Rollout Action")
        ax.set_yscale("log")
        ax.set_xticks(list(positions.values()), [f"{s:g}" for s in SIGMAS])
        ax.set_xlim(0.5, 4.5)
        ax.grid(axis="y", which="both", alpha=0.22)
        ax.legend(handles=method_handles, loc="upper left", fontsize=8, frameon=True)
    axes_s[-1, 0].set_xlabel("σ")
    fig_s.suptitle("Paired Action scatter (α=2; 20 rollouts per method/σ)", fontsize=14)
    fig_s.savefig(scatter_path, dpi=220, bbox_inches="tight")
    plt.close(fig_s)

    # Standalone boxplot view with the individual paired Monte Carlo values overlaid.
    fig_b, axes_b = plt.subplots(len(datasets), 1, figsize=(8.5, 3.9 * len(datasets)), squeeze=False, constrained_layout=True)
    for row_idx, (dataset_label, dataset) in enumerate(datasets):
        ax = axes_b[row_idx, 0]
        subset = [r for r in rows if r["dataset_id"] == dataset]
        data: list[list[float]] = []
        box_positions: list[float] = []
        box_methods: list[str] = []
        for sigma in SIGMAS:
            for method in METHODS:
                vals = [float(r["Action"]) for r in subset if r["sigma"] == sigma and r["method"] == method]
                data.append(vals)
                box_positions.append(positions[sigma] + method_offsets[method])
                box_methods.append(method)
        bp = ax.boxplot(data, positions=box_positions, widths=0.18, patch_artist=True, showfliers=False,
                        medianprops={"color": "#202020", "linewidth": 1.4},
                        whiskerprops={"color": "#555555", "linewidth": 0.9},
                        capprops={"color": "#555555", "linewidth": 0.9})
        for patch, method in zip(bp["boxes"], box_methods):
            patch.set_facecolor(COLORS[method]); patch.set_edgecolor(COLORS[method]); patch.set_alpha(0.38)
        for pos, vals, method in zip(box_positions, data, box_methods):
            ax.scatter([pos + dx for dx in np.linspace(-0.052, 0.052, len(vals))], vals,
                       color=COLORS[method], edgecolors="white", linewidths=0.35, s=18, alpha=0.78, zorder=3)
        ax.set_title(dataset_label, loc="left", fontweight="bold")
        ax.set_ylabel("Rollout Action")
        ax.set_yscale("log")
        ax.set_xticks(list(positions.values()), [f"{s:g}" for s in SIGMAS])
        ax.set_xlim(0.5, 4.5)
        ax.grid(axis="y", which="both", alpha=0.22)
        ax.legend(handles=method_handles, loc="upper left", fontsize=8, frameon=True)
    axes_b[-1, 0].set_xlabel("σ")
    fig_b.suptitle("Paired Action boxplots (α=2; 20 rollouts per method/σ)", fontsize=14)
    fig_b.text(0.5, -0.003, "Boxes summarize 20 rollout seeds; overlaid dots are the observed Action values.",
               ha="center", fontsize=9, color="#444444")
    fig_b.savefig(boxplot_path, dpi=220, bbox_inches="tight")
    plt.close(fig_b)

    print(f"scatter={scatter_path.resolve()}")
    print(f"boxplot={boxplot_path.resolve()}")
    print(f"combined={combined_path.resolve()}")
    print(f"csv={csv_path.resolve()}; rows={len(rows)}")


if __name__ == "__main__":
    main()
