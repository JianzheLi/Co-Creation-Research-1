"""Probe whether bounded Score/Growth readouts stabilize a trained SDE without retraining."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DeepRUOT/Code: src/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # workspace root: Baselines/
from Baselines.experiment2.evaluate import _emd2_with_retry
from src.evaluate import particle_mass_and_weights
from src.experiment import make_feedback
from src.runtime import configure, file_hash, load_config, snapshots_for
from src.sample import rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(342, 362)))
    parser.add_argument("--w1-seeds", type=int, nargs="*", default=[342, 343])
    parser.add_argument("--bound-velocity", action="store_true",
                        help="use the checkpoint's output-normalized velocity fields too")
    parser.add_argument("--use-checkpoint-architecture", action="store_true",
                        help="evaluate the trained architecture exactly as configured")
    parser.add_argument("--trace-file", type=Path,
                        help="write per-step SDE state/field maxima for diagnosing rollout instability")
    args = parser.parse_args()
    if args.bound_velocity and args.use_checkpoint_architecture:
        raise ValueError("choose either --bound-velocity or --use-checkpoint-architecture")
    run = args.run_dir.resolve()
    trace_file = args.trace_file.resolve() if args.trace_file else None
    if trace_file is not None and trace_file.exists():
        raise FileExistsError(f"refusing to overwrite SDE trace: {trace_file}")
    if trace_file is not None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(run / "config.yaml")
    device = configure(config)
    checkpoint_path = run / "feedback.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The checkpoint parameter shapes are unchanged: non-affine LayerNorm adds
    # no trainable values. This inference-only probe isolates the OOD output
    # constraint before paying for another Matching run.
    if not args.use_checkpoint_architecture:
        config["matching"]["network_variant"] = ("plain_output_norm" if args.bound_velocity else "plain")
        config["matching"]["growth_network_variant"] = "plain_output_norm"
        config["matching"]["score_network_variant"] = "plain_output_norm"
    model = make_feedback(config, device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    truth = snapshots_for(config, full=True)
    n = len(truth[0].x)
    generated_rows = []
    started = time.perf_counter()
    for seed in args.seeds:
        seed_started = time.perf_counter()
        trace_rows = []
        try:
            generated = rollout(
                model=model, first_snapshot=truth[0].x, times=config["model_times"],
                sigma=config["model_sigma"], n_particles=n,
                steps_per_unit=config["generation"]["steps_per_unit_interval"],
                seed=seed, endpoint_guard=config["generation"]["endpoint_guard"],
                initial_x=truth[0].x, stochastic=True,
                diagnostic_callback=(trace_rows.append if trace_file is not None else None),
            )
        except FloatingPointError as error:
            if trace_file is not None:
                with trace_file.open("a", encoding="utf-8") as handle:
                    for row in trace_rows:
                        handle.write(json.dumps({"seed": int(seed), **row}) + "\n")
            seed_row = {"seed": int(seed), "status": "rollout_failure",
                        "failure_reason": str(error), "metrics": [],
                        "elapsed_seconds": time.perf_counter() - seed_started}
            generated_rows.append(seed_row)
            print(json.dumps(seed_row), flush=True)
            continue
        if trace_file is not None:
            with trace_file.open("a", encoding="utf-8") as handle:
                for row in trace_rows:
                    handle.write(json.dumps({"seed": int(seed), **row}) + "\n")
        positions = generated["positions"].astype(np.float64)
        log_weights = np.asarray(generated["log_weights"], dtype=np.float64)
        seed_row = {"seed": int(seed), "status": "completed", "metrics": []}
        try:
            stats = [particle_mass_and_weights(lw) for lw in log_weights]
        except FloatingPointError as error:
            seed_row["status"] = "mass_overflow"
            seed_row["failure_reason"] = str(error)
            stats = []
        if stats:
            initial_mass = stats[0][0]
            for k, snapshot in enumerate(truth):
                mass, normalized, log_mass = stats[k]
                predicted_ratio = mass / initial_mass
                observed_ratio = snapshot.mass / truth[0].mass
                row = {
                    "time": float(snapshot.time),
                    "predicted_mass_ratio": float(predicted_ratio),
                    "observed_mass_ratio": float(observed_ratio),
                    "mass_relative_error": float(abs(predicted_ratio-observed_ratio)/max(abs(observed_ratio), 1e-12)),
                    "ESS": float(1.0 / np.square(normalized).sum()),
                    "max_abs_position": float(np.abs(positions[k]).max()),
                    "max_abs_log_weight": float(np.abs(log_weights[k]).max()),
                    "log_total_mass": float(log_mass),
                }
                if seed in args.w1_seeds and k > 0:
                    cost = cdist(positions[k], snapshot.x, metric="euclidean")
                    target_weights = np.full(len(snapshot.x), 1.0 / len(snapshot.x))
                    row["W1"] = float(_emd2_with_retry(normalized, target_weights, cost))
                seed_row["metrics"].append(row)
        else:
            for k, snapshot in enumerate(truth):
                seed_row["metrics"].append({
                    "time": float(snapshot.time),
                    "max_abs_position": float(np.abs(positions[k]).max()),
                    "max_abs_log_weight": float(np.abs(log_weights[k]).max()),
                })
        seed_row["elapsed_seconds"] = time.perf_counter() - seed_started
        generated_rows.append(seed_row)
        final = seed_row["metrics"][-1]
        print(json.dumps({"seed": seed, "status": seed_row["status"], **final,
                          "elapsed_seconds": seed_row["elapsed_seconds"]}), flush=True)

    successful = [item for item in generated_rows if item["status"] == "completed"]
    result = {
        "source_run": str(run),
        "source_checkpoint_sha256": file_hash(checkpoint_path),
        "dataset": config["dataset"], "sigma": config["model_sigma"],
        "alpha": config["growth_penalty"], "steps_per_unit": config["generation"]["steps_per_unit_interval"],
        "intervention": ("trained checkpoint architecture" if args.use_checkpoint_architecture else
                         "non-affine LayerNorm on all four field readouts" if args.bound_velocity else
                         "non-affine LayerNorm on Score and Growth readouts only"),
        "training_objective_changed": False,
        "checkpoint_finetuned": bool(args.bound_velocity),
        "seeds": generated_rows,
        "summary": {
            "completed_seeds": len(successful),
            "failed_seeds": len(generated_rows) - len(successful),
            "max_abs_position": max((m["max_abs_position"] for s in successful for m in s["metrics"]), default=None),
            "max_abs_log_weight": max((m["max_abs_log_weight"] for s in successful for m in s["metrics"]), default=None),
            "mean_mass_error_by_time": {
                str(t): float(np.mean([m["mass_relative_error"] for s in successful for m in s["metrics"]
                                       if m["time"] == t]))
                for t in config["model_times"][1:] if successful
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if trace_file is not None:
        filename = "sde_trace_" + "_".join(str(seed) for seed in args.seeds) + ".json"
    else:
        filename = ("checkpoint_architecture_20seeds_stability.json" if args.use_checkpoint_architecture else
                    "all_field_output_norm_20seeds_stability.json" if args.bound_velocity else
                    "score_growth_output_norm_probe_20seeds.json")
    out = run / "evaluation_all_initial_points" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
