"""Paired 20-seed mass-trace diagnostic for trained feedback rollouts."""
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from evaluate_matching_stability import checked_checkpoint
from src.matching import make_feedback, validate_feedback_architecture
from src.runtime import configure, file_hash, json_write, load_config, snapshots_for


@torch.no_grad()
def rollout_mass(model, x0, noises, config):
    device = x0.device
    n_repeats, n_particles, dimension = x0.shape
    dtype = next(model.parameters()).dtype
    sigma = float(config["model_sigma"])
    times = config["model_times"]
    n_steps = int(config["generation"]["steps_per_unit_interval"])
    dt = 1.0 / n_steps
    guard = float(config["generation"]["endpoint_guard"])
    x = x0.clone()
    logw = torch.full((n_repeats, n_particles), -math.log(n_particles),
                      dtype=torch.float64, device=device)
    masses = [logw.exp().sum(-1)]
    for interval, start in enumerate(times[:-1]):
        for step in range(n_steps):
            time = start + step * dt
            local_time = min(max(time - start, guard), 1 - guard)
            t_input = torch.full((n_repeats * n_particles, 1), start + local_time,
                                 dtype=dtype, device=device)
            flat_x = x.reshape(n_repeats * n_particles, dimension)
            velocity, growth, score = model(t_input, flat_x)
            velocity = velocity.reshape(n_repeats, n_particles, dimension)
            growth = growth.reshape(n_repeats, n_particles).double()
            score = score.reshape(n_repeats, n_particles, dimension)
            drift = velocity + 0.5 * sigma**2 * score
            x = x + drift * dt + sigma * math.sqrt(dt) * noises[:, interval * n_steps + step]
            logw = logw + growth * dt
        masses.append(logw.exp().sum(-1))
    return torch.stack(masses, dim=-1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruot-run", required=True)
    parser.add_argument("--wfr-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args()
    runs = [Path(args.ruot_run), Path(args.wfr_run)]
    configs = [load_config(run / "config.yaml") for run in runs]
    config = configs[0]
    for key in ("dataset", "data_path", "dimension", "model_times", "model_sigma",
                "growth_penalty", "generation", "matching"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"paired runs differ at {key}")
    device = configure(config)
    if device.type != "cuda":
        raise RuntimeError("mass rollout diagnostic requires CUDA")
    checkpoints = [checked_checkpoint(run) for run in runs]
    truth = snapshots_for(config, full=True)
    first = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)
    seeds = list(range(args.seed_start, args.seed_start + 20))
    n_particles = int(config["generation"]["n_particles"])
    n_steps_total = (len(config["model_times"]) - 1) * int(config["generation"]["steps_per_unit_interval"])
    x0_rows, noise_rows = [], []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed + 300)
        ids = torch.randint(len(first), (n_particles,), device=device, generator=generator)
        x0_rows.append(first[ids])
        noise_rows.append(torch.randn((n_steps_total, n_particles, config["dimension"]),
                                      dtype=torch.float32, device=device, generator=generator))
    x0, noises = torch.stack(x0_rows), torch.stack(noise_rows)
    targets = np.asarray([float(snapshot.mass) for snapshot in truth], dtype=np.float64)
    method_traces = {}
    for name, run, checkpoint, cfg in zip(("RUOT", "WFR+Brownian bridge"), runs, checkpoints, configs):
        model = make_feedback(cfg, device)
        validate_feedback_architecture(checkpoint, cfg)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        trace = rollout_mass(model, x0, noises, cfg)
        method_traces[name] = {
            "run": str(run.resolve()),
            "feedback_sha256": file_hash(run / "feedback.pt"),
            "mass_by_seed": trace.tolist(),
            "mean": trace.mean(axis=0).tolist(),
            "sd": trace.std(axis=0, ddof=1).tolist(),
            "mean_relative_error": (trace / targets[None, :] - 1).mean(axis=0).tolist(),
            "mean_absolute_relative_error": (np.abs(trace / targets[None, :] - 1)).mean(axis=0).tolist(),
            "fraction_under_target": (trace < targets[None, :]).mean(axis=0).tolist(),
        }
        del model
        torch.cuda.empty_cache()
    ruot = np.asarray(method_traces["RUOT"]["mass_by_seed"])
    wfr = np.asarray(method_traces["WFR+Brownian bridge"]["mass_by_seed"])
    paired_rel_difference = (ruot - wfr) / targets[None, :]
    result = {
        "dataset": config["dataset"], "alpha": float(config["growth_penalty"]),
        "sigma": float(config["model_sigma"]), "seeds": seeds,
        "particles_per_seed": n_particles,
        "steps_per_unit_interval": int(config["generation"]["steps_per_unit_interval"]),
        "observation_times": list(config["model_times"]), "target_masses": targets.tolist(),
        "device": torch.cuda.get_device_name(device),
        "definition": "20 paired feedback rollouts using the same initial samples and Brownian increments as the Action evaluation; mass is sum_i exp(log_weight_i) at each observation time",
        "methods": method_traces,
        "paired_RUOT_minus_WFR_relative_mass_difference_mean": paired_rel_difference.mean(axis=0).tolist(),
        "paired_RUOT_minus_WFR_relative_mass_difference_sd": paired_rel_difference.std(axis=0, ddof=1).tolist(),
    }
    json_write(Path(args.output), result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "target_masses": targets.tolist(),
                      "methods": {name: {"mean": m["mean"], "sd": m["sd"],
                                         "mean_relative_error": m["mean_relative_error"],
                                         "mean_absolute_relative_error": m["mean_absolute_relative_error"],
                                         "fraction_under_target": m["fraction_under_target"]}
                                  for name, m in method_traces.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
