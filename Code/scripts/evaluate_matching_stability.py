"""Paired 20-seed rollout diagnostics for old and tanh-last-hidden Matching."""
from pathlib import Path
import argparse
import csv
import json
import math
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.matching import make_feedback
from src.runtime import configure, file_hash, json_write, load_config, snapshots_for


class LegacyMLP(nn.Module):
    """Original separate feedback MLP: hidden activation followed by linear output."""
    def __init__(self, dimension, n_output, width, layers, activation):
        super().__init__()
        act = nn.Tanh if activation == "tanh" else nn.SiLU
        blocks = []
        for layer in range(layers):
            blocks += [nn.Linear(dimension + 1 if layer == 0 else width, width), act()]
        blocks.append(nn.Linear(width, n_output))
        self.net = nn.Sequential(*blocks)

    def forward(self, t, x):
        return self.net(torch.cat((t, x), dim=-1))


class LegacyFeedback(nn.Module):
    def __init__(self, dimension, width, layers, activation):
        super().__init__()
        self.flow = LegacyMLP(dimension, dimension, width, layers, activation)
        self.growth = LegacyMLP(dimension, 1, width, layers, activation)
        self.score = LegacyMLP(dimension, dimension, width, layers, activation)

    def forward(self, t, x):
        return self.flow(t, x), self.growth(t, x), self.score(t, x)


def checked_checkpoint(run):
    run = Path(run)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("matching", {}).get("status") != "completed":
        raise RuntimeError(f"Matching is incomplete: {run}")
    cp = torch.load(run / "feedback.pt", map_location="cpu", weights_only=False)
    if (cp["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt")
            or cp["coupling_manifest_sha256"] != file_hash(run / "couplings" / "manifest.json")):
        raise RuntimeError(f"stale feedback checkpoint: {run}")
    return cp


def summarize(values):
    x = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if not len(x):
        return {"median": None, "p95": None, "maximum": None}
    return {"median": float(np.median(x)), "p95": float(np.quantile(x, 0.95)),
            "maximum": float(np.max(x))}


@torch.no_grad()
def simulate_batch(model, x0, noises, config, seeds):
    device = x0.device
    n_repeats, n_particles, dimension = x0.shape
    dtype = next(model.parameters()).dtype
    sigma = float(config["model_sigma"])
    alpha = float(config["growth_penalty"])
    times = config["model_times"]
    n_steps = int(config["generation"]["steps_per_unit_interval"])
    dt = 1.0 / n_steps
    guard = float(config["generation"]["endpoint_guard"])

    x = x0.clone()
    logw = torch.full((n_repeats, n_particles), -math.log(n_particles),
                      dtype=torch.float64, device=device)
    kinetic = torch.zeros(n_repeats, dtype=torch.float64, device=device)
    growth_action = torch.zeros_like(kinetic)
    maxima = {key: torch.zeros(n_repeats, dtype=torch.float64, device=device) for key in (
        "velocity_norm", "growth_abs", "score_norm", "drift_norm", "position_norm",
        "particle_log_weight", "particle_weight", "total_mass")}
    maxima["position_norm"] = x.double().norm(dim=-1).amax(dim=-1)
    maxima["total_mass"].fill_(1.0)
    finite = torch.ones(n_repeats, dtype=torch.bool, device=device)

    for interval, start in enumerate(times[:-1]):
        for step in range(n_steps):
            time = start + step * dt
            local_time = min(max(time - start, guard), 1 - guard)
            t_input = torch.full((n_repeats * n_particles, 1), start + local_time,
                                 dtype=dtype, device=device)
            flat_x = x.reshape(n_repeats * n_particles, dimension)
            velocity, growth, score = model(t_input, flat_x)
            velocity = velocity.reshape(n_repeats, n_particles, dimension)
            growth = growth.reshape(n_repeats, n_particles)
            score = score.reshape(n_repeats, n_particles, dimension)
            drift = velocity + 0.5 * sigma**2 * score

            maxima["velocity_norm"] = torch.maximum(maxima["velocity_norm"], velocity.double().norm(dim=-1).amax(-1))
            maxima["growth_abs"] = torch.maximum(maxima["growth_abs"], growth.double().abs().amax(-1))
            maxima["score_norm"] = torch.maximum(maxima["score_norm"], score.double().norm(dim=-1).amax(-1))
            maxima["drift_norm"] = torch.maximum(maxima["drift_norm"], drift.double().norm(dim=-1).amax(-1))
            particle_mass = logw.exp()
            maxima["particle_log_weight"] = torch.maximum(maxima["particle_log_weight"], logw.amax(-1))
            maxima["particle_weight"] = torch.maximum(maxima["particle_weight"], particle_mass.amax(-1))
            maxima["total_mass"] = torch.maximum(maxima["total_mass"], particle_mass.sum(-1))

            kinetic += (particle_mass * (0.5 * drift.double().square().sum(-1)) * dt).sum(-1)
            growth_action += (particle_mass * (0.5 * alpha * growth.double().square()) * dt).sum(-1)
            x = x + drift * dt + sigma * math.sqrt(dt) * noises[:, interval * n_steps + step]
            logw = logw + growth.double() * dt
            updated_mass = logw.exp()
            maxima["particle_log_weight"] = torch.maximum(maxima["particle_log_weight"], logw.amax(-1))
            maxima["particle_weight"] = torch.maximum(maxima["particle_weight"], updated_mass.amax(-1))
            maxima["total_mass"] = torch.maximum(maxima["total_mass"], updated_mass.sum(-1))
            maxima["position_norm"] = torch.maximum(
                maxima["position_norm"], x.double().norm(dim=-1).amax(-1))

        finite &= torch.isfinite(x).all(dim=-1).all(-1)
        finite &= torch.isfinite(logw).all(-1)

    finite &= torch.isfinite(kinetic) & torch.isfinite(growth_action)
    def safe_float(value):
        number = float(value.item())
        return number if math.isfinite(number) else None

    out = []
    for i, seed in enumerate(seeds):
        out.append({
            "seed": int(seed), "finite": bool(finite[i].item()),
            "kinetic_action": safe_float(kinetic[i]),
            "growth_action": safe_float(growth_action[i]),
            "total_action": safe_float(kinetic[i] + growth_action[i]),
            **{f"max_{key}": safe_float(value[i]) for key, value in maxima.items()},
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args()
    if args.repeats != 20:
        raise ValueError("this stability check is intentionally fixed to 20 seeds")

    baseline_run, candidate_run = Path(args.baseline_run), Path(args.candidate_run)
    baseline_config = load_config(baseline_run / "config.yaml")
    candidate_config = load_config(candidate_run / "config.yaml")
    for key in ("dimension", "model_times", "model_sigma", "growth_penalty", "generation", "data_path"):
        if baseline_config[key] != candidate_config[key]:
            raise ValueError(f"paired runs must share {key}")
    device = configure(candidate_config)
    if device.type != "cuda":
        raise RuntimeError("GPU required for the requested rollout diagnostics")
    baseline_cp = checked_checkpoint(baseline_run)
    candidate_cp = checked_checkpoint(candidate_run)
    baseline_model = LegacyFeedback(
        baseline_config["dimension"], baseline_config["matching"]["hidden_width"],
        baseline_config["matching"]["hidden_layers"],
        baseline_config["matching"].get("feedback_activation", "silu")).to(device=device, dtype=torch.float32)
    baseline_model.load_state_dict(baseline_cp["model"])
    candidate_model = make_feedback(candidate_config, device)
    candidate_model.load_state_dict(candidate_cp["model"])
    baseline_model.eval()
    candidate_model.eval()

    truth = snapshots_for(candidate_config, full=True)
    first_snapshot = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)
    seeds = list(range(args.seed_start, args.seed_start + args.repeats))
    n_particles = int(candidate_config["generation"]["n_particles"])
    steps_total = (len(candidate_config["model_times"]) - 1) * int(candidate_config["generation"]["steps_per_unit_interval"])
    dim = candidate_config["dimension"]
    initial, noise_rows = [], []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed + 300)
        ids = torch.randint(len(first_snapshot), (n_particles,), device=device, generator=generator)
        initial.append(first_snapshot[ids])
        noise_rows.append(torch.randn((steps_total, n_particles, dim), dtype=torch.float32,
                                      device=device, generator=generator))
    x0 = torch.stack(initial)
    noises = torch.stack(noise_rows)

    torch.cuda.synchronize(device)
    baseline_rows = simulate_batch(baseline_model, x0, noises, baseline_config, seeds)
    candidate_rows = simulate_batch(candidate_model, x0, noises, candidate_config, seeds)
    torch.cuda.synchronize(device)
    del noises

    summaries = {}
    for name, rows in (("baseline", baseline_rows), ("tanh_last_candidate", candidate_rows)):
        summaries[name] = {"finite_seeds": sum(row["finite"] for row in rows), "seed_count": len(rows)}
        for metric in ("total_action", "max_velocity_norm", "max_growth_abs", "max_score_norm",
                       "max_drift_norm", "max_position_norm", "max_particle_weight", "max_total_mass"):
            summaries[name][metric] = summarize([row[metric] for row in rows])

    invariants = {
        "last_hidden_activation_is_tanh": candidate_config["matching"].get("last_hidden_activation") == "tanh",
        "all_candidate_rollouts_finite": all(row["finite"] for row in candidate_rows),
    }

    result = {
        "definition": "20-seed paired rollout stability check; identical per-seed initial samples and Brownian increments",
        "seeds": seeds,
        "device": torch.cuda.get_device_name(device),
        "sigma": float(candidate_config["model_sigma"]),
        "alpha": float(candidate_config["growth_penalty"]),
        "particles_per_seed": n_particles,
        "steps_per_unit_interval": int(candidate_config["generation"]["steps_per_unit_interval"]),
        "baseline_run": str(baseline_run.resolve()),
        "candidate_run": str(candidate_run.resolve()),
        "baseline_checkpoint_sha256": file_hash(baseline_run / "feedback.pt"),
        "candidate_checkpoint_sha256": file_hash(candidate_run / "feedback.pt"),
        "invariants": invariants,
        "summaries": summaries,
        "replicates": [{"seed": seed, "baseline": old, "tanh_last_candidate": new}
                       for seed, old, new in zip(seeds, baseline_rows, candidate_rows)],
    }
    out_dir = candidate_run / "stability_20seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_write(out_dir / "summary.json", result)
    csv_rows = []
    for seed, old, new in zip(seeds, baseline_rows, candidate_rows):
        for method, row in (("baseline", old), ("tanh_last_candidate", new)):
            csv_rows.append({"method": method, **row})
    with (out_dir / "replicates.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"dataset": candidate_config["dataset"], "summaries": summaries,
                      "invariants": invariants, "output": str(out_dir.resolve())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
