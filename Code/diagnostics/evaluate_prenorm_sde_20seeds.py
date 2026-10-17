import json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.spatial.distance import cdist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DeepRUOT/Code: src/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # workspace root: Baselines/
from Baselines.experiment2.evaluate import _emd2_with_retry
from src.runtime import configure, load_config, snapshots_for, file_hash
from src.experiment import make_feedback
from src.matching import validate_feedback_architecture
from src.sample import rollout

run = Path(r"../Results/experiment2/eb_noscale_leakyrelu_prenorm_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42").resolve()
outdir = run / "evaluation_all_initial_points"
outdir.mkdir(parents=True, exist_ok=True)
cfg = load_config(run / "config.yaml")
device = configure(cfg)
truth = snapshots_for(cfg, full=True)
checkpoint = torch.load(run / "feedback.pt", map_location=device, weights_only=False)
validate_feedback_architecture(checkpoint, cfg)
model = make_feedback(cfg, device)
model.load_state_dict(checkpoint["model"])
model.eval()
results = {"dataset": cfg["dataset"], "sigma": cfg["model_sigma"], "alpha": cfg["growth_penalty"],
           "steps_per_unit": cfg["generation"]["steps_per_unit_interval"],
           "source_points": len(truth[0].x), "checkpoint_sha256": file_hash(run / "feedback.pt"),
           "objective_unchanged": True, "seeds": []}
outpath = outdir / "sde_full_20seeds_metrics.json"
for seed in range(340, 360):
    started = time.perf_counter()
    entry = {"seed": seed}
    try:
        generated = rollout(model, truth[0].x, times=cfg["model_times"], sigma=cfg["model_sigma"],
                            n_particles=len(truth[0].x), steps_per_unit=cfg["generation"]["steps_per_unit_interval"],
                            seed=seed, endpoint_guard=cfg["generation"]["endpoint_guard"],
                            initial_x=truth[0].x, stochastic=True)
        positions = generated["positions"].astype(np.float64)
        positions[0] = truth[0].x
        weights = np.exp(generated["log_weights"])
        initial_mass = float(weights[0].sum())
        rows = []
        for k, snapshot in enumerate(truth):
            mass = np.maximum(np.nan_to_num(weights[k], nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            if mass.sum() <= 0:
                raise FloatingPointError(f"all weights vanished at time={snapshot.time}")
            cost = cdist(positions[k], snapshot.x, metric="euclidean")
            a = mass / mass.sum()
            b = np.full(len(snapshot.x), 1.0 / len(snapshot.x), dtype=np.float64)
            w1 = float(_emd2_with_retry(a, b, cost))
            mass_ratio = float(mass.sum() / initial_mass)
            observed_ratio = float(snapshot.mass / truth[0].mass)
            rows.append({"time": float(snapshot.time), "W1": w1,
                         "predicted_mass_ratio": mass_ratio, "observed_mass_ratio": observed_ratio,
                         "mass_relative_error": abs(mass_ratio - observed_ratio) / max(abs(observed_ratio), 1e-12),
                         "ESS": float(mass.sum() ** 2 / (mass @ mass)),
                         "max_abs_position": float(np.abs(positions[k]).max()),
                         "max_abs_log_weight": float(np.abs(generated["log_weights"][k]).max())})
        entry.update(status="completed", metrics=rows)
    except Exception as exc:
        entry.update(status="failed", error=repr(exc))
    entry["elapsed_seconds"] = time.perf_counter() - started
    results["seeds"].append(entry)
    outpath.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if entry["status"] == "completed":
        w1 = [row["W1"] for row in entry["metrics"][1:]]
        print(f"seed={seed} complete mean_W1={np.mean(w1):.6g} max_abs_x={max(r['max_abs_position'] for r in entry['metrics']):.6g} seconds={entry['elapsed_seconds']:.1f}", flush=True)
    else:
        print(f"seed={seed} FAILED {entry['error']} seconds={entry['elapsed_seconds']:.1f}", flush=True)

completed = [e for e in results["seeds"] if e["status"] == "completed"]
if completed:
    pointwise = {}
    for j in range(1, len(truth)):
        vals = np.asarray([e["metrics"][j]["W1"] for e in completed])
        masses = np.asarray([e["metrics"][j]["mass_relative_error"] for e in completed])
        pointwise[str(truth[j].time)] = {"W1_mean": float(vals.mean()), "W1_sd": float(vals.std(ddof=1)),
                                         "W1_median": float(np.median(vals)),
                                         "W1_p95": float(np.quantile(vals, .95)),
                                         "mass_error_mean": float(masses.mean()),
                                         "mass_error_p95": float(np.quantile(masses, .95))}
    results["summary"] = {"completed_seeds": len(completed), "failed_seeds": len(results["seeds"]) - len(completed),
                          "mean_W1_across_seeds_and_times": float(np.mean([e["metrics"][j]["W1"] for e in completed for j in range(1, len(truth))])),
                          "pointwise": pointwise}
    outpath.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("saved", outpath, flush=True)
