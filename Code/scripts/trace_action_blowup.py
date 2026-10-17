"""Trace selected runaway particles through the original RUOT feedback field."""
from pathlib import Path
import argparse
import json
import math
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.matching import make_feedback
from src.runtime import configure, file_hash, load_config, snapshots_for


@torch.no_grad()
def trace(model, first, config, seed, particle):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    generation = config["generation"]
    n, steps = generation["n_particles"], generation["steps_per_unit_interval"]
    dt, sigma = 1 / steps, config["model_sigma"]
    times = config["model_times"]
    generator = torch.Generator(device=device).manual_seed(seed + 300)
    ids = torch.randint(len(first), (n,), device=device, generator=generator)
    x = first[ids].clone()
    initial_position = x[particle].detach().cpu().tolist()
    logw = torch.full((n,), -math.log(n), dtype=torch.float64, device=device)
    values = {key: [] for key in ("x_norm", "v_norm", "score_norm", "drift_norm", "growth", "log_weight")}
    for k in range(len(times) - 1):
        for local_step in range(steps):
            t = times[k] + local_step * dt
            local = min(max(t - times[k], generation["endpoint_guard"]), 1 - generation["endpoint_guard"])
            t_input = torch.full((n, 1), times[k] + local, dtype=dtype, device=device)
            v, g, score = model(t_input, x)
            drift = v + 0.5 * sigma**2 * score
            values["x_norm"].append(x[particle].norm())
            values["v_norm"].append(v[particle].norm())
            values["score_norm"].append(score[particle].norm())
            values["drift_norm"].append(drift[particle].norm())
            values["growth"].append(g[particle, 0])
            values["log_weight"].append(logw[particle])
            noise = torch.randn(x.shape, dtype=dtype, device=device, generator=generator) * math.sqrt(dt)
            x = x + drift * dt + sigma * noise
            logw = logw + g.squeeze(-1).double() * dt
    values = {key: torch.stack(value).cpu().tolist() for key, value in values.items()}
    thresholds = [2.55, 3.0, 4.0, 6.0, 8.0, 10.0, 20.0, 30.0]
    crossings = []
    for threshold in thresholds:
        index = next((i for i, radius in enumerate(values["x_norm"]) if radius >= threshold), None)
        if index is not None:
            crossings.append({"radius_threshold": threshold,
                              "time": index * dt,
                              **{key: values[key][index] for key in values}})
    maximum_index = max(range(len(values["drift_norm"])), key=values["drift_norm"].__getitem__)
    maximum = {"time": maximum_index * dt,
               **{key: values[key][maximum_index] for key in values}}
    return {"seed": seed, "particle": particle, "initial_position": initial_position,
            "crossings": crossings, "max_drift_step": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="../Results/gene_steps_gpu_4000_seed42")
    parser.add_argument("--seeds", default="46,51,52")
    parser.add_argument("--particles", default="173,1548,1245")
    parser.add_argument("--out", default="../Results/action_monte_carlo_20_seed42/blowup_trace.json")
    args = parser.parse_args()
    run = Path(args.run)
    config = load_config(run / "config.yaml")
    device = configure(config)
    truth = snapshots_for(config, full=True)
    first = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)
    checkpoint = torch.load(run / "feedback.pt", map_location=device, weights_only=False)
    if (checkpoint["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt")
            or checkpoint["coupling_manifest_sha256"] != file_hash(run / "couplings" / "manifest.json")):
        raise RuntimeError("stale feedback checkpoint")
    model = make_feedback(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    seeds = [int(x) for x in args.seeds.split(",")]
    particles = [int(x) for x in args.particles.split(",")]
    result = {"device": torch.cuda.get_device_name(device), "traces": [
        trace(model, first, config, seed, particle) for seed, particle in zip(seeds, particles)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
