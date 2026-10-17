"""Compare SDE W1 across common rollout seeds for Matching architectures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code: src/
from src.evaluation_utils import emd2_with_retry as _emd2_with_retry
from src.experiment import make_feedback
from src.matching import validate_feedback_architecture
from src.runtime import configure, file_hash, load_config, snapshots_for
from src.sample import rollout
from src.evaluate import particle_mass_and_weights


EVALUATION_VERSION = 2


def evaluate_sde_seed(run: Path, seed: int, steps: int | None = None) -> pd.DataFrame:
    run = run.resolve()
    config = load_config(run / "config.yaml")
    device = configure(config)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("matching", {}).get("status") != "completed":
        raise RuntimeError(f"Matching is incomplete: {run}")
    checkpoint_path = run / "feedback.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_feedback_architecture(checkpoint, config)
    if checkpoint["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt"):
        raise RuntimeError("feedback checkpoint belongs to a different pair path")
    if checkpoint["coupling_manifest_sha256"] != file_hash(run / "couplings" / "manifest.json"):
        raise RuntimeError("feedback checkpoint belongs to different couplings")
    model = make_feedback(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    truth = snapshots_for(config, full=True)
    n = len(truth[0].x)
    n_steps = steps or config["generation"]["steps_per_unit_interval"]
    generated = rollout(
        model=model, first_snapshot=truth[0].x, times=config["model_times"],
        sigma=config["model_sigma"], n_particles=n, steps_per_unit=n_steps,
        seed=seed, endpoint_guard=config["generation"]["endpoint_guard"],
        initial_x=truth[0].x, stochastic=True,
    )
    positions = generated["positions"].astype(np.float64)
    log_weights = np.asarray(generated["log_weights"], dtype=np.float64)
    statistics = [particle_mass_and_weights(row) for row in log_weights]
    start_mass = statistics[0][0]
    rows = []
    for k, snapshot in enumerate(truth):
        mass_total, normalized_mass, _ = statistics[k]
        if k == 0 and np.allclose(positions[k], snapshot.x, rtol=1e-5, atol=1e-5):
            w1 = 0.0
        else:
            cost = cdist(positions[k], snapshot.x, metric="euclidean")
            a = normalized_mass
            b = np.full(len(snapshot.x), 1.0 / len(snapshot.x), dtype=np.float64)
            w1 = _emd2_with_retry(a, b, cost)
        pred_mass_ratio = float(mass_total / start_mass)
        obs_mass_ratio = float(snapshot.mass / truth[0].mass)
        rows.append(dict(
            architecture=run.name, dataset=config["dataset"], inference="SDE",
            time=float(snapshot.time), W1=float(w1), seed=int(seed),
            predicted_mass_ratio=pred_mass_ratio, observed_mass_ratio=obs_mass_ratio,
            mass_relative_error=abs(pred_mass_ratio - obs_mass_ratio) / max(abs(obs_mass_ratio), 1e-12),
            prediction_points=len(log_weights[k]), target_points=len(snapshot.x),
            sigma=config["model_sigma"], alpha=config["growth_penalty"],
            steps_per_unit=n_steps, feedback_checkpoint_sha256=file_hash(checkpoint_path),
            status="completed", evaluation_version=EVALUATION_VERSION,
        ))
    frame = pd.DataFrame(rows)
    frame.to_csv(run / f"sde_seed_{seed}_metrics.csv", index=False)
    return frame


def failed_seed_frame(run: Path, seed: int, error: Exception) -> pd.DataFrame:
    config = load_config(run / "config.yaml")
    return pd.DataFrame([dict(
        architecture=run.name, dataset=config["dataset"], inference="SDE",
        time=float(t), W1=np.nan, seed=int(seed), predicted_mass_ratio=np.nan,
        observed_mass_ratio=np.nan, mass_relative_error=np.nan,
        prediction_points=np.nan, target_points=np.nan,
        sigma=config["model_sigma"], alpha=config["growth_penalty"],
        steps_per_unit=0, feedback_checkpoint_sha256="", status="failed",
        failure_reason=f"{type(error).__name__}: {error}",
        evaluation_version=EVALUATION_VERSION,
    ) for t in config["model_times"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[342, 343, 344, 345, 346])
    parser.add_argument("--steps-per-unit", type=int)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    runs = sorted(path for path in run_root.iterdir()
                  if path.is_dir() and (path / "config.yaml").exists()
                  and (path / "feedback.pt").exists())
    if not runs:
        raise RuntimeError(f"no trained architecture runs found in {run_root}")

    frames = []
    for run in runs:
        print(f"\n=== {run.name}: SDE seeds {args.seeds} ===", flush=True)
        for seed in args.seeds:
            cached = run / f"sde_seed_{seed}_metrics.csv"
            frame = None
            if cached.exists():
                cached_frame = pd.read_csv(cached)
                if ("evaluation_version" in cached_frame and
                        np.all(cached_frame.evaluation_version == EVALUATION_VERSION)):
                    frame = cached_frame
                    print(f"Reuse {run.name}, seed={seed}", flush=True)
            if frame is None and (run / "sde_metrics.csv").exists():
                primary = pd.read_csv(run / "sde_metrics.csv")
                if ("evaluation_version" in primary and
                        np.all(primary.evaluation_version == EVALUATION_VERSION) and
                        "seed" in primary and np.all(primary.seed.to_numpy() == seed)):
                    frame = primary.copy()
                    frame["architecture"] = run.name
                    frame.to_csv(cached, index=False)
                    print(f"Reuse primary evaluation {run.name}, seed={seed}", flush=True)
            if frame is None:
                try:
                    frame = evaluate_sde_seed(run, seed, args.steps_per_unit)
                except (FloatingPointError, RuntimeError) as error:
                    frame = failed_seed_frame(run, seed, error)
                    frame.to_csv(cached, index=False)
                    print(f"seed={seed}: failed: {error}", flush=True)
            frames.append(frame)
            mean_w1 = frame.loc[frame.time > 0, "W1"].mean()
            print(f"seed={seed}: mean noninitial W1={mean_w1:.6g}", flush=True)

    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(run_root / "w1_seed_metrics.csv", index=False)
    per_seed = (metrics[metrics.time > 0].groupby(["architecture", "seed"], as_index=False)
                .agg(mean_W1=("W1", "mean"), median_W1=("W1", "median"),
                     mean_mass_relative_error=("mass_relative_error", "mean"),
                     failed=("status", lambda x: bool((x != "completed").any()))))
    summary = (per_seed.groupby("architecture", as_index=False)
               .agg(mean_W1=("mean_W1", "mean"), sd_W1=("mean_W1", "std"),
                    median_W1=("mean_W1", "median"),
                    mean_mass_relative_error=("mean_mass_relative_error", "mean"),
                    failed_seeds=("failed", "sum"), total_seeds=("failed", "size")))
    summary["sd_W1"] = summary["sd_W1"].fillna(0.0)
    summary["failure_rate"] = summary.failed_seeds / summary.total_seeds
    summary = summary.sort_values("mean_W1")
    summary.to_csv(run_root / "w1_seed_summary.csv", index=False)
    per_seed.to_csv(run_root / "w1_seed_per_run.csv", index=False)

    order = summary.architecture.tolist()
    values = [per_seed.loc[(per_seed.architecture == name) & per_seed.mean_W1.notna(), "mean_W1"].to_numpy()
              for name in order]
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(order)), 5))
    ax.boxplot(values, labels=order, showfliers=False)
    rng = np.random.default_rng(1701)
    for i, sample in enumerate(values, 1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(sample)), sample, alpha=0.8, s=24)
    ax.set_ylabel("Mean full empirical SDE W1 over noninitial times")
    ax.set_title(f"{metrics.dataset.iloc[0]}: shared-seed SDE W1 comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_root / "w1_seed_boxplot.png", dpi=170)
    plt.close(fig)
    print("\n=== Mean W1 ± SD across rollout seeds (lower is better) ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
