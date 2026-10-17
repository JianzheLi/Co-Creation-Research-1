"""Evaluate the same trained RUOT feedback model with its flow ODE and SDE."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DeepRUOT/Code: src/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # workspace root: Baselines/
from Baselines.experiment2.evaluate import _emd2_with_retry
from src.experiment import make_feedback
from src.matching import validate_feedback_architecture
from src.runtime import configure, file_hash, load_config, snapshots_for
from src.sample import rollout
from src.evaluate import particle_mass_and_weights


def evaluate_run(run: Path, steps: int | None, seed: int | None) -> pd.DataFrame:
    run = run.resolve()
    config = load_config(run / "config.yaml")
    device = configure(config)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("matching", {}).get("status") != "completed":
        raise RuntimeError("complete Matching is required before ODE/SDE evaluation")
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
    eval_seed = seed if seed is not None else config["seed"] + 300
    output_rows = []
    modes = [("ODE", False), ("SDE", True)]
    for label, stochastic in modes:
        start = time.perf_counter()
        generated = rollout(
            model=model,
            first_snapshot=truth[0].x,
            times=config["model_times"],
            sigma=config["model_sigma"],
            n_particles=n,
            steps_per_unit=n_steps,
            seed=eval_seed,
            endpoint_guard=config["generation"]["endpoint_guard"],
            initial_x=truth[0].x,
            stochastic=stochastic,
        )
        positions = generated["positions"].astype(np.float64)
        # Preserve the exact full-data initial sample and its dataset order.
        positions[0] = truth[0].x
        log_weights = np.asarray(generated["log_weights"], dtype=np.float64)
        statistics = [particle_mass_and_weights(row) for row in log_weights]
        weights = np.exp(log_weights)
        if not np.isfinite(weights).all():
            raise FloatingPointError(f"raw particle weights are not representable for {label}")
        predictions = run / f"{label.lower()}_predictions.npz"
        np.savez_compressed(predictions, times=np.asarray(config["model_times"], dtype=np.float64),
                            positions=positions, weights=weights,
                            sigma=config["model_sigma"], seed=eval_seed,
                            inference=label, stochastic=stochastic,
                            feedback_checkpoint_sha256=file_hash(checkpoint_path))
        start_mass = statistics[0][0]
        for k, snapshot in enumerate(truth):
            mass_total, normalized_mass, _ = statistics[k]
            if k == 0 and np.allclose(positions[k], snapshot.x, rtol=1e-5, atol=1e-5) and np.allclose(normalized_mass, 1/n, rtol=1e-10, atol=1e-12):
                w1 = 0.0
            else:
                print(f"Full-data W1: {run.name}/{label} t={k}, n_pred={len(log_weights[k])}, n_obs={len(snapshot.x)}", flush=True)
                cost = cdist(positions[k], snapshot.x, metric="euclidean")
                a = normalized_mass
                b = np.full(len(snapshot.x), 1.0 / len(snapshot.x), dtype=np.float64)
                w1 = _emd2_with_retry(a, b, cost)
            pred_mass_ratio = float(mass_total / start_mass)
            obs_mass_ratio = float(snapshot.mass / truth[0].mass)
            row = dict(dataset=config["dataset"], inference=label, time=float(snapshot.time),
                       W1=w1, predicted_mass_ratio=pred_mass_ratio,
                       observed_mass_ratio=obs_mass_ratio,
                       mass_relative_error=abs(pred_mass_ratio-obs_mass_ratio)/max(abs(obs_mass_ratio), 1e-12),
                       prediction_points=len(log_weights[k]), target_points=len(snapshot.x),
                       sigma=config["model_sigma"], alpha=config["growth_penalty"],
                       seed=eval_seed, steps_per_unit=n_steps)
            output_rows.append(row)
            print(f"{label} t={k}: W1={w1:.6g}; mass ratio={pred_mass_ratio:.6g}/{obs_mass_ratio:.6g}", flush=True)
        pd.DataFrame([r for r in output_rows if r["inference"] == label]).to_csv(run / f"{label.lower()}_metrics.csv", index=False)
        print(f"{label} rollout and full evaluation finished in {time.perf_counter()-start:.1f}s", flush=True)
    metrics = pd.DataFrame(output_rows)
    metrics.to_csv(run / "ode_sde_metrics.csv", index=False)
    noninitial = metrics[metrics.time > 0]
    summary = (noninitial.groupby("inference")
               .agg(mean_W1=("W1", "mean"), median_W1=("W1", "median"),
                    mean_mass_relative_error=("mass_relative_error", "mean"))
               .reset_index())
    summary.to_csv(run / "ode_sde_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode, style in (("ODE", "o-"), ("SDE", "s-")):
        part = metrics[metrics.inference == mode]
        axes[0].plot(part.time, part.W1, style, label=mode)
        axes[1].plot(part.time, part.predicted_mass_ratio, style, label=f"{mode} predicted")
    axes[1].plot(metrics[metrics.inference == "ODE"].time,
                 metrics[metrics.inference == "ODE"].observed_mass_ratio, "k--", label="observed")
    axes[0].set(xlabel="Observed time", ylabel="Full empirical W1")
    axes[1].set(xlabel="Observed time", ylabel="Mass ratio to initial time")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle(f"{config['dataset']}: same RUOT training, sigma={config['model_sigma']}")
    fig.tight_layout()
    fig.savefig(run / "ode_sde_comparison.png", dpi=160)
    plt.close(fig)
    endpoint_regularized = config.get("matching", {}).get("field_parameterization") == "endpoint_regularized"
    metadata = dict(dataset=config["dataset"], sigma=config["model_sigma"], growth_penalty=config["growth_penalty"],
                   inference_modes=["probability-flow ODE (V)", "SDE (V + sigma^2 Score / 2 drift, sigma noise)"],
                   initial_points=n, full_point_W1=True, evaluation_times=config["model_times"],
                   steps_per_unit=n_steps, seed=eval_seed,
                   ode_integrator=("classical RK4 in s, t=sin^2(pi*s/2)" if endpoint_regularized
                                   else "explicit Euler in physical time"),
                   sde_integrator=("Euler-Maruyama in s, t=sin^2(pi*s/2), with exact physical-time Brownian variance"
                                   if endpoint_regularized else "explicit Euler in physical time"),
                   feedback_checkpoint_sha256=file_hash(checkpoint_path))
    (run / "ode_sde_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps-per-unit", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(evaluate_run(args.run_dir, args.steps_per_unit, args.seed).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
