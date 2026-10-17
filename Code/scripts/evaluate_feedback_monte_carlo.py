"""Paired action Monte Carlo for baseline and retrained feedback networks."""
from pathlib import Path
import argparse
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compute_action_monte_carlo import estimate_action, load_feedback, summarize
from src.matching import make_feedback
from src.runtime import configure, json_write, load_config, snapshots_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", default="../Results/gene_steps_gpu_4000_seed42")
    parser.add_argument("--candidate-run", default="../Results/gene_steps_gpu_tanh_matching_seed42")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--out", default="../Results/action_monte_carlo_tanh_matching_seed42/summary.json")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    runs = {"SiLU baseline": Path(args.reference_run), "tanh hidden": Path(args.candidate_run)}
    configs = {name: load_config(run / "config.yaml") for name, run in runs.items()}
    reference = configs["SiLU baseline"]
    for name, config in configs.items():
        for key in ("model_times", "model_sigma", "growth_penalty", "generation"):
            if config[key] != reference[key]:
                raise ValueError(f"runs must share {key}; mismatch in {name}")

    device = configure(reference)
    if device.type != "cuda":
        raise RuntimeError("GPU is required for this Monte Carlo comparison")
    truth = snapshots_for(reference, full=True)
    first = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)
    models = {}
    for name, run in runs.items():
        model = make_feedback(configs[name], device)
        model.load_state_dict(load_feedback(run, configs[name])["model"])
        model.eval()
        models[name] = model

    seeds = list(range(args.seed_start, args.seed_start + args.repeats))
    replicates = []
    for index, seed in enumerate(seeds, start=1):
        values = {name: estimate_action(model, first, configs[name], seed)
                  for name, model in models.items()}
        replicates.append({"seed": seed, "methods": values,
                           "paired_tanh_minus_silu": values["tanh hidden"]["total_action"]
                           - values["SiLU baseline"]["total_action"]})
        print(f"[{index}/{args.repeats}] seed={seed} "
              f"SiLU={values['SiLU baseline']['total_action']:.6f} "
              f"tanh={values['tanh hidden']['total_action']:.6f}", flush=True)

    totals = {name: summarize([row["methods"][name]["total_action"] for row in replicates])
              for name in runs}
    result = {
        "device": torch.cuda.get_device_name(device),
        "paired_common_random_numbers": True,
        "seed_range": [seeds[0], seeds[-1]],
        "reference_run": str(runs["SiLU baseline"].resolve()),
        "candidate_run": str(runs["tanh hidden"].resolve()),
        "total_action_summary": totals,
        "paired_tanh_minus_silu": summarize([row["paired_tanh_minus_silu"] for row in replicates]),
        "replicates": replicates,
    }
    json_write(args.out, result)
    print(f"Saved paired action comparison to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
