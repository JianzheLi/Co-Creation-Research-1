"""Inspect states and weights for selected Action Monte Carlo seeds."""
from pathlib import Path
import argparse
import json
import math
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.matching import make_feedback
from src.runtime import configure, file_hash, json_write, load_config, snapshots_for


@torch.no_grad()
def diagnose(model, first_snapshot, config, seed):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    generation = config["generation"]
    n = int(generation["n_particles"])
    steps = int(generation["steps_per_unit_interval"])
    times = config["model_times"]
    sigma = float(config["model_sigma"])
    alpha = float(config["growth_penalty"])
    dt = 1 / steps
    generator = torch.Generator(device=device).manual_seed(seed + 300)
    ids = torch.randint(len(first_snapshot), (n,), device=device, generator=generator)
    x = first_snapshot[ids].clone()
    logw = torch.full((n,), -math.log(n), dtype=torch.float64, device=device)
    interval_kinetic = torch.zeros((len(times) - 1, n), dtype=torch.float64, device=device)
    interval_growth = torch.zeros_like(interval_kinetic)
    peak_abs_growth, peak_drift_norm, peak_x_norm, peak_logw = [], [], [], []
    boundaries = []
    position_snapshots = [x.clone()]
    logweight_snapshots = [logw.clone()]

    for k in range(len(times) - 1):
        g_trace, drift_trace, x_trace, logw_trace = [], [], [], []
        for local_step in range(steps):
            time = times[k] + local_step * dt
            local = min(max(time - times[k], generation["endpoint_guard"]), 1 - generation["endpoint_guard"])
            t_input = torch.full((n, 1), times[k] + local, dtype=dtype, device=device)
            velocity, growth, score = model(t_input, x)
            drift = velocity + 0.5 * sigma**2 * score
            g = growth.squeeze(-1)

            particle_mass = logw.exp()
            interval_kinetic[k] += particle_mass * (0.5 * drift.double().square().sum(-1)) * dt
            interval_growth[k] += particle_mass * (0.5 * alpha * g.double().square()) * dt

            g_trace.append(g.abs().max())
            drift_trace.append(drift.norm(dim=-1).max())
            x_trace.append(x.norm(dim=-1).max())
            logw_trace.append(logw.max())
            noise = torch.randn(x.shape, dtype=dtype, device=device, generator=generator) * math.sqrt(dt)
            x = x + drift * dt + sigma * noise
            logw = logw + g.double() * dt
            if not torch.isfinite(x).all() or not torch.isfinite(logw).all():
                raise FloatingPointError(f"nonfinite rollout at t={time + dt}")

        traces = [torch.stack(trace) for trace in (g_trace, drift_trace, x_trace, logw_trace)]
        for name, trace in zip(("abs_growth", "drift_norm", "position_norm", "log_weight"), traces):
            index = int(trace.argmax().item())
            peak = {"value": float(trace[index].item()), "time": float(times[k] + index * dt)}
            (peak_abs_growth if name == "abs_growth" else
             peak_drift_norm if name == "drift_norm" else
             peak_x_norm if name == "position_norm" else peak_logw).append(peak)

        masses = logw.exp()
        mass_sum = masses.sum()
        boundaries.append({
            "time": float(times[k + 1]),
            "mass": float(mass_sum.item()),
            "max_log_weight": float(logw.max().item()),
            "max_normalized_particle_weight": float((masses.max() / mass_sum).item()),
            "effective_sample_size": float((mass_sum.square() / masses.square().sum()).item()),
            "max_position_norm": float(x.norm(dim=-1).max().item()),
            "p99_position_norm": float(torch.quantile(x.norm(dim=-1), 0.99).item()),
        })
        position_snapshots.append(x.clone())
        logweight_snapshots.append(logw.clone())

    kinetic_particles = interval_kinetic.sum(0) * n
    growth_particles = interval_growth.sum(0) * n
    total_particles = kinetic_particles + growth_particles
    _, top_indices = torch.topk(total_particles, k=5)
    selected_positions = torch.stack(position_snapshots)[:, top_indices].cpu().tolist()
    selected_logweights = torch.stack(logweight_snapshots)[:, top_indices].cpu().tolist()
    top_contributors = [{
        "particle": int(index.item()),
        "total_action_contribution": float(total_particles[index].item()),
        "kinetic_contribution": float(kinetic_particles[index].item()),
        "growth_contribution": float(growth_particles[index].item()),
        "position_path": [
            {"time": float(times[t]), "x": [float(a) for a in selected_positions[t][rank]],
             "norm": float(torch.linalg.vector_norm(torch.tensor(selected_positions[t][rank])).item())}
            for t in range(len(times))
        ],
        "log_weight_path": [float(selected_logweights[t][rank]) for t in range(len(times))],
    } for rank, index in enumerate(top_indices)]
    intervals = []
    for k, (kinetic, growth) in enumerate(zip(interval_kinetic.sum(-1).tolist(), interval_growth.sum(-1).tolist())):
        intervals.append({"start": float(times[k]), "end": float(times[k + 1]),
                          "kinetic": float(kinetic), "growth": float(growth),
                          "total": float(kinetic + growth)})
    return {
        "total_action": float(total_particles.mean().item()),
        "particle_mc_standard_error": float(total_particles.std(unbiased=True).item() / math.sqrt(n)),
        "intervals": intervals,
        "per_interval_peaks": {
            "abs_growth": peak_abs_growth,
            "drift_norm": peak_drift_norm,
            "position_norm": peak_x_norm,
            "log_weight": peak_logw,
        },
        "boundary_diagnostics": boundaries,
        "top_particle_action_contributors": top_contributors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruot-run", default="../Results/gene_steps_gpu_4000_seed42")
    parser.add_argument("--wfr-run", default="../Results/gene_wfr_bridge_alpha10_sigma025_seed42")
    parser.add_argument("--seeds", default="42,46,51,52")
    parser.add_argument("--out", default="../Results/action_monte_carlo_20_seed42/diagnostics.json")
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]
    runs = {"RUOT": Path(args.ruot_run), "WFR-FM bridge": Path(args.wfr_run)}
    configs = {name: load_config(run / "config.yaml") for name, run in runs.items()}
    config = configs["RUOT"]
    device = configure(config)
    if device.type != "cuda":
        raise RuntimeError("GPU is required for this diagnostic")
    truth = snapshots_for(config, full=True)
    first = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)
    models = {}
    for name, run in runs.items():
        model = make_feedback(configs[name], device)
        checkpoint = torch.load(run / "feedback.pt", map_location="cpu", weights_only=False)
        if (checkpoint["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt")
                or checkpoint["coupling_manifest_sha256"] != file_hash(run / "couplings" / "manifest.json")):
            raise RuntimeError(f"feedback checkpoint is stale: {run}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[name] = model

    results = {"device": torch.cuda.get_device_name(device), "seeds": {}}
    for seed in seeds:
        results["seeds"][str(seed)] = {}
        for name in runs:
            results["seeds"][str(seed)][name] = diagnose(models[name], first, configs[name], seed)
        print(f"seed={seed} RUOT={results['seeds'][str(seed)]['RUOT']['total_action']:.6f} "
              f"WFR-FM={results['seeds'][str(seed)]['WFR-FM bridge']['total_action']:.6f}", flush=True)
    json_write(args.out, results)
    print(f"Saved diagnostics to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
