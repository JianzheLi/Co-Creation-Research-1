from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SIGMAS = [0.1, 0.25, 0.5, 1.0]
METHODS = ["RUOT", "WFR+Brownian bridge"]
COLORS = {"RUOT": "#2878B5", "WFR+Brownian bridge": "#E87532"}
MARKERS = {0.1: "o", 0.25: "s", 0.5: "^", 1.0: "D"}
DATASETS = [
    ("Simulation Gene", "gene", "../Results/experiment1_alpha2_sigma_sweep"),
    ("Mouse", "mouse", "../Results/experiment1_alpha2_sigma_sweep"),
    ("EMT", "emt", "../Results/experiment1_alpha2_sigma_sweep"),
    ("WFR-FM Dygen", "dygen", "../Results/experiment1_alpha2_sigma_sweep"),
]


def read_metrics(workspace: Path) -> tuple[list[dict[str, object]], dict[str, list[int]]]:
    rows: list[dict[str, object]] = []
    dataset_times: dict[str, list[int]] = {}
    for label, dataset, output_root in DATASETS:
        expected_times: list[int] | None = None
        for sigma in SIGMAS:
            sigma_tag = str(sigma).replace(".", "p")
            for method, folder in (
                ("RUOT", "ruot_seed42"),
                ("WFR+Brownian bridge", "wfr_brownian_bridge_seed42"),
            ):
                path = workspace / output_root / dataset / f"sigma_{sigma_tag}" / folder / "metrics.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                with path.open(newline="", encoding="utf-8-sig") as f:
                    current = list(csv.DictReader(f))
                times = [int(float(r["time"])) for r in current]
                if expected_times is None:
                    expected_times = times
                elif times != expected_times:
                    raise ValueError(
                        f"Observation times mismatch for {label}, σ={sigma}, {method}: {times} != {expected_times}"
                    )
                for r in current:
                    rows.append(
                        {
                            "dataset": label,
                            "dataset_id": dataset,
                            "sigma": sigma,
                            "method": method,
                            "time": int(float(r["time"])),
                            "W1_normalized": float(r["W1_normalized"]),
                            "W2_normalized": float(r["W2_normalized"]),
                            "mass_pred": float(r["mass_pred"]),
                            "mass_true": float(r["mass_true"]),
                            "mass_relative_error": float(r["mass_relative_error"]),
                            "metric_seed": int(r["seed"]),
                        }
                    )
        assert expected_times is not None
        dataset_times[label] = expected_times
    return rows, dataset_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Results/experiment1_alpha2_sigma_sweep"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, dataset_times = read_metrics(workspace)
    csv_path = output_dir / "all_timepoint_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS[m], label=m, markersize=7)
        for m in METHODS
    ]
    sigma_handles = [
        Line2D([0], [0], marker=MARKERS[s], linestyle="", color="#555555", label=f"σ={s:g}", markersize=7)
        for s in SIGMAS
    ]

    # Scatter view: every method × sigma × observation time is shown explicitly.
    fig_scatter, axes = plt.subplots(4, 1, figsize=(12.5, 15.5), constrained_layout=True)
    for ax, (label, _, _root) in zip(axes, DATASETS):
        offset = {"RUOT": -0.055, "WFR+Brownian bridge": 0.055}
        dataset_rows = [r for r in rows if r["dataset"] == label]
        for sigma in SIGMAS:
            for method in METHODS:
                subset = [r for r in dataset_rows if r["sigma"] == sigma and r["method"] == method]
                ax.scatter(
                    [int(r["time"]) + offset[method] for r in subset],
                    [float(r["W1_normalized"]) for r in subset],
                    color=COLORS[method],
                    marker=MARKERS[sigma],
                    s=60,
                    edgecolors="white",
                    linewidths=0.65,
                    alpha=0.93,
                    zorder=3,
                )
        times = dataset_times[label]
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylabel("Normalized W1")
        ax.set_xticks(times, [str(t) for t in times])
        ax.set_xlim(min(times) - 0.4, max(times) + 0.4)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(handles=method_handles + sigma_handles, ncol=3, fontsize=8, loc="upper left", frameon=True)
    axes[-1].set_xlabel("Observation time")
    fig_scatter.suptitle(
        "All observation-time W1 values: RUOT vs WFR + Brownian bridge (α=2)", fontsize=15
    )
    scatter_path = output_dir / "all_datasets_timepoint_w1_scatter.png"
    fig_scatter.savefig(scatter_path, dpi=220, bbox_inches="tight")
    plt.close(fig_scatter)

    # Boxplot view: each box summarizes the W1 values over that dataset's observation times.
    fig_box, axes = plt.subplots(4, 1, figsize=(12.5, 15.5), constrained_layout=True)
    centers = list(range(1, len(SIGMAS) + 1))
    for ax, (label, _, _root) in zip(axes, DATASETS):
        dataset_rows = [r for r in rows if r["dataset"] == label]
        positions: list[float] = []
        box_data: list[list[float]] = []
        box_methods: list[str] = []
        for center, sigma in zip(centers, SIGMAS):
            for method, offset in (("RUOT", -0.19), ("WFR+Brownian bridge", 0.19)):
                values = [
                    float(r["W1_normalized"])
                    for r in dataset_rows
                    if r["sigma"] == sigma and r["method"] == method
                ]
                positions.append(center + offset)
                box_data.append(values)
                box_methods.append(method)
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.4},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
        )
        for patch, method in zip(bp["boxes"], box_methods):
            patch.set_facecolor(COLORS[method])
            patch.set_edgecolor(COLORS[method])
            patch.set_alpha(0.38)
        for pos, values, method in zip(positions, box_data, box_methods):
            jitter = [
                (i - (len(values) - 1) / 2) * 0.035
                for i in range(len(values))
            ]
            ax.scatter(
                [pos + dx for dx in jitter],
                values,
                color=COLORS[method],
                edgecolors="white",
                linewidths=0.45,
                s=26,
                alpha=0.95,
                zorder=4,
            )
        times = dataset_times[label]
        ax.set_title(f"{label} (n={len(times)} timepoints per box)", loc="left", fontweight="bold")
        ax.set_ylabel("Normalized W1")
        ax.set_xticks(centers, [f"{s:g}" for s in SIGMAS])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(handles=method_handles, ncol=2, fontsize=8, loc="upper left", frameon=True)
    axes[-1].set_xlabel("σ")
    fig_box.suptitle(
        "W1 distribution across all observation times, grouped by σ and method (α=2)", fontsize=15
    )
    fig_box.text(
        0.5,
        -0.005,
        "Each box contains the per-timepoint W1 values for one trained model; dots show individual times. "
        "Boxes summarize variation across observation times, not Monte Carlo uncertainty.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    box_path = output_dir / "all_datasets_timepoint_w1_boxplot.png"
    fig_box.savefig(box_path, dpi=220, bbox_inches="tight")
    plt.close(fig_box)

    print(f"scatter={scatter_path.resolve()}")
    print(f"boxplot={box_path.resolve()}")
    print(f"csv={csv_path.resolve()}")
    print(f"rows={len(rows)}; timepoints={dataset_times}")


if __name__ == "__main__":
    main()
