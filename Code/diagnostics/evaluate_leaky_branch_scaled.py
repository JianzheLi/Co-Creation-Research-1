import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.spatial.distance import cdist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code: src/
from src.evaluation_utils import emd2_with_retry as _emd2_with_retry
from src.runtime import configure, load_config, snapshots_for, file_hash
from src.experiment import make_feedback
from src.matching import validate_feedback_architecture
from src.sample import rollout

run = Path(r"../Results/experiment2/eb_noscale_leakyrelu_residual_branch_scaled_w512x8_a1225_sigma05_blocks8_fm30000_seed42").resolve()
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
results = {"dataset": cfg["dataset"], "sigma": cfg["model_sigma"], "seed": 342,
           "steps_per_unit": cfg["generation"]["steps_per_unit_interval"],
           "source_points": len(truth[0].x), "loss_weights": json.loads((run / "matching_loss_weights.json").read_text()),
           "checkpoint_sha256": file_hash(run / "feedback.pt"), "modes": {}}
for label, stochastic in (("ODE", False), ("SDE", True)):
    started = time.perf_counter()
    try:
        generated = rollout(model, truth[0].x, times=cfg["model_times"], sigma=cfg["model_sigma"],
                            n_particles=len(truth[0].x), steps_per_unit=cfg["generation"]["steps_per_unit_interval"],
                            seed=342, endpoint_guard=cfg["generation"]["endpoint_guard"],
                            initial_x=truth[0].x, stochastic=stochastic)
        positions = generated["positions"].astype(np.float64)
        positions[0] = truth[0].x
        weights = np.exp(generated["log_weights"])
        rows = []
        for k, snapshot in enumerate(truth):
            mass = np.maximum(np.nan_to_num(weights[k], nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            a = mass / mass.sum()
            b = np.full(len(snapshot.x), 1.0 / len(snapshot.x), dtype=np.float64)
            cost = cdist(positions[k], snapshot.x, metric="euclidean")
            w1 = float(_emd2_with_retry(a, b, cost))
            w2 = float(np.sqrt(_emd2_with_retry(a, b, cost * cost)))
            mr = float(weights[k].sum() / weights[0].sum())
            obs = float(snapshot.mass / truth[0].mass)
            rows.append({"time": float(snapshot.time), "W1": w1, "W2": w2,
                         "predicted_mass_ratio": mr, "observed_mass_ratio": obs,
                         "mass_relative_error": abs(mr - obs) / max(abs(obs), 1e-12),
                         "ESS": float(weights[k].sum() ** 2 / (weights[k] @ weights[k])),
                         "max_abs_position": float(np.abs(positions[k]).max())})
        np.savez_compressed(outdir / f"{label.lower()}_predictions_seed342.npz",
                            times=generated["times"], positions=positions, weights=weights, seed=342,
                            sigma=cfg["model_sigma"], inference=label)
        results["modes"][label] = {"status": "completed", "metrics": rows,
                                   "elapsed_seconds": time.perf_counter() - started}
    except Exception as exc:
        results["modes"][label] = {"status": "failed", "error": repr(exc),
                                   "elapsed_seconds": time.perf_counter() - started}
    (outdir / "branch_scaled_seed342_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(label, json.dumps(results["modes"][label], ensure_ascii=False), flush=True)
print("saved", outdir / "branch_scaled_seed342_metrics.json", flush=True)
