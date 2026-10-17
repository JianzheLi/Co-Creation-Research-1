"""Paired Monte Carlo estimates of post-training RUOT running action."""
from pathlib import Path
import argparse
import csv
import json
import math
import sys

import numpy as np
import torch
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.matching import make_feedback
from src.runtime import configure, file_hash, json_write, load_config, snapshots_for


def load_feedback(run, config):
    run = Path(run).resolve()
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("matching", {}).get("status") != "completed":
        raise RuntimeError(f"completed Matching is required: {run}")
    checkpoint = torch.load(run / "feedback.pt", map_location="cpu", weights_only=False)
    if (checkpoint["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt")
            or checkpoint["coupling_manifest_sha256"] != file_hash(run / "couplings" / "manifest.json")):
        raise RuntimeError(f"feedback checkpoint is stale: {run}")
    return checkpoint


@torch.no_grad()
def estimate_action(model, first_snapshot, config, seed):
    """Run one action-only rollout; state, weights, and accumulators stay on GPU."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    generation = config["generation"]
    n_particles = int(generation["n_particles"])
    steps = int(generation["steps_per_unit_interval"])
    times = config["model_times"]
    sigma = float(config["model_sigma"])
    alpha = float(config["growth_penalty"])

    generator = torch.Generator(device=device).manual_seed(seed + 300)
    ids = torch.randint(len(first_snapshot), (n_particles,), device=device, generator=generator)
    x = first_snapshot[ids].clone()
    logw = torch.full((n_particles,), -math.log(n_particles), dtype=torch.float64, device=device)
    interval_kinetic = torch.zeros((len(times) - 1, n_particles), dtype=torch.float64, device=device)
    interval_growth = torch.zeros_like(interval_kinetic)
    dt = 1 / steps

    for k in range(len(times) - 1):
        for local_step in range(steps):
            time = times[k] + local_step * dt
            local_time = min(max(time - times[k], generation["endpoint_guard"]),
                             1 - generation["endpoint_guard"])
            t_input = torch.full((n_particles, 1), times[k] + local_time,
                                 dtype=dtype, device=device)
            velocity, growth, score = model(t_input, x)
            drift = velocity + 0.5 * sigma**2 * score

            particle_mass = logw.exp()
            kinetic_rate = 0.5 * drift.double().square().sum(-1)
            growth_rate = 0.5 * alpha * growth.squeeze(-1).double().square()
            interval_kinetic[k] += particle_mass * kinetic_rate * dt
            interval_growth[k] += particle_mass * growth_rate * dt

            noise = torch.randn(x.shape, dtype=dtype, device=device, generator=generator) * math.sqrt(dt)
            x = x + drift * dt + sigma * noise
            logw = logw + growth.squeeze(-1).double() * dt
            if not torch.isfinite(x).all() or not torch.isfinite(logw).all():
                raise FloatingPointError(f"nonfinite weighted rollout at t={time + dt}")

    kinetic_by_interval = interval_kinetic.sum(-1)
    growth_by_interval = interval_growth.sum(-1)
    kinetic_particles = interval_kinetic.sum(0) * n_particles
    growth_particles = interval_growth.sum(0) * n_particles
    total_particles = kinetic_particles + growth_particles
    intervals = []
    for k, (kinetic, growth) in enumerate(zip(kinetic_by_interval.tolist(), growth_by_interval.tolist())):
        intervals.append({"start": float(times[k]), "end": float(times[k + 1]),
                          "kinetic": float(kinetic), "growth": float(growth),
                          "total": float(kinetic + growth)})
    kinetic = float(kinetic_by_interval.sum().item())
    growth = float(growth_by_interval.sum().item())
    return {
        "kinetic_action": kinetic,
        "growth_action": growth,
        "total_action": kinetic + growth,
        "particle_mc_standard_error": float(total_particles.std(unbiased=True).item() / math.sqrt(n_particles)),
        "kinetic_mc_standard_error": float(kinetic_particles.std(unbiased=True).item() / math.sqrt(n_particles)),
        "growth_mc_standard_error": float(growth_particles.std(unbiased=True).item() / math.sqrt(n_particles)),
        "intervals": intervals,
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if count > 1 else 0.0
    se = sd / math.sqrt(count) if count else float("nan")
    critical = float(student_t.ppf(0.975, count - 1)) if count > 1 else float("nan")
    return {"n": count, "mean": mean, "sd_between_replicates": sd,
            "standard_error_of_mean": se,
            "ci95": [mean - critical * se, mean + critical * se] if count > 1 else [None, None]}


def main():
    parser = argparse.ArgumentParser(description="Paired repeated Monte Carlo estimates of learned-feedback Action")
    parser.add_argument("--ruot-run", default="../Results/gene_steps_gpu_4000_seed42")
    parser.add_argument("--wfr-run", default="../Results/gene_wfr_bridge_alpha10_sigma025_seed42")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    run_specs = {
        "RUOT": Path(args.ruot_run),
        "WFR-FM bridge": Path(args.wfr_run),
    }
    configs = {name: load_config(run / "config.yaml") for name, run in run_specs.items()}
    ref = configs["RUOT"]
    for name, cfg in configs.items():
        for key in ("model_times", "model_sigma", "growth_penalty", "generation"):
            if cfg[key] != ref[key]:
                raise ValueError(f"the methods must share {key}; mismatch in {name}")

    device = configure(ref)
    if device.type != "cuda":
        raise RuntimeError("GPU is required for this requested Monte Carlo evaluation")
    truth = snapshots_for(ref, full=True)
    first_snapshot = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)
    models = {}
    for name, run in run_specs.items():
        model = make_feedback(configs[name], device)
        checkpoint = load_feedback(run, configs[name])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[name] = model
    torch.cuda.synchronize(device)

    seeds = list(range(args.seed_start, args.seed_start + args.repeats))
    replicates = []
    for index, seed in enumerate(seeds, start=1):
        paired = {}
        for name in run_specs:
            paired[name] = estimate_action(models[name], first_snapshot, configs[name], seed)
        replicates.append({"seed": seed, "methods": paired})
        print(f"[{index}/{args.repeats}] seed={seed}  RUOT={paired['RUOT']['total_action']:.6f}  "
              f"WFR-FM={paired['WFR-FM bridge']['total_action']:.6f}  "
              f"delta={paired['WFR-FM bridge']['total_action'] - paired['RUOT']['total_action']:+.6f}",
              flush=True)

    fields = ["kinetic_action", "growth_action", "total_action"]
    for k in range(len(ref["model_times"]) - 1):
        for component in ("kinetic", "growth", "total"):
            fields.append(f"interval_{k}_{component}")

    method_summaries = {}
    for name in run_specs:
        method_summaries[name] = {}
        for field in fields:
            if field.startswith("interval_"):
                _, k, component = field.split("_")
                vals = [row["methods"][name]["intervals"][int(k)][component] for row in replicates]
            else:
                vals = [row["methods"][name][field] for row in replicates]
            method_summaries[name][field] = summarize(vals)
        method_summaries[name]["mean_particle_mc_standard_error"] = float(np.mean([
            row["methods"][name]["particle_mc_standard_error"] for row in replicates]))

    paired_differences = {}
    for field in fields:
        differences = []
        for row in replicates:
            ruot, wfr = row["methods"]["RUOT"], row["methods"]["WFR-FM bridge"]
            if field.startswith("interval_"):
                _, k, component = field.split("_")
                delta = wfr["intervals"][int(k)][component] - ruot["intervals"][int(k)][component]
            else:
                delta = wfr[field] - ruot[field]
            differences.append(delta)
        stats = summarize(differences)
        stats["mean_percent_of_RUOT_mean"] = 100 * stats["mean"] / method_summaries["RUOT"][field]["mean"]
        paired_differences[field] = stats

    summary = {
        "definition": "learned_feedback_running_action",
        "formula": "integral sum_i w_i(t) * (0.5*|U(t,X_i)|^2 + alpha/2*G(t,X_i)^2) dt",
        "drift": "U = V + sigma^2/2 * Score",
        "psi": "g^2/2",
        "alpha": float(ref["growth_penalty"]),
        "sigma": float(ref["model_sigma"]),
        "steps_per_unit": int(ref["generation"]["steps_per_unit_interval"]),
        "particles_per_replicate": int(ref["generation"]["n_particles"]),
        "device": torch.cuda.get_device_name(device),
        "paired_common_random_numbers": True,
        "mc_seed_range": [seeds[0], seeds[-1]],
        "rollout_seed_offset": 300,
        "method_runs": {name: str(path.resolve()) for name, path in run_specs.items()},
        "feedback_checkpoint_sha256": {name: file_hash(path / "feedback.pt") for name, path in run_specs.items()},
        "method_summaries": method_summaries,
        "paired_wfr_minus_ruot": paired_differences,
        "replicates": replicates,
    }
    out_dir = Path(args.out_dir) if args.out_dir else Path("../Results") / f"action_monte_carlo_{args.repeats}_seed{args.seed_start}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_write(out_dir / "summary.json", summary)
    csv_rows = []
    for row in replicates:
        for name, result in row["methods"].items():
            flat = {"seed": row["seed"], "method": name,
                    "kinetic_action": result["kinetic_action"],
                    "growth_action": result["growth_action"],
                    "total_action": result["total_action"],
                    "particle_mc_standard_error": result["particle_mc_standard_error"],
                    "kinetic_mc_standard_error": result["kinetic_mc_standard_error"],
                    "growth_mc_standard_error": result["growth_mc_standard_error"]}
            for k, interval in enumerate(result["intervals"]):
                for component in ("kinetic", "growth", "total"):
                    flat[f"interval_{k}_{component}"] = interval[component]
            csv_rows.append(flat)
    with (out_dir / "replicates.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved repeated Action estimates to {out_dir.resolve()}")
    print(json.dumps({"RUOT total": method_summaries["RUOT"]["total_action"],
                      "WFR-FM total": method_summaries["WFR-FM bridge"]["total_action"],
                      "paired difference": paired_differences["total_action"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
