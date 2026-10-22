"""Run EB coupling/Matching candidates on a common learned conditional path."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.matching import train_matching
from src.runtime import configure, init_run, json_write, load_config
from src.solve_couplings import solve_couplings
from scripts.evaluate_architecture_w1_seeds import evaluate_sde_seed


RECIPES = {
    "affine_silu": ("plain_output_affine_norm", "silu", "silu", 512, 8),
    "nonaffine_silu": ("plain_output_norm", "silu", "silu", 512, 8),
    "nonaffine_leaky": ("plain_output_norm", "leaky_relu", "leaky_relu", 512, 8),
}


def prepare(args):
    source = args.source_run.resolve()
    destination = args.run_dir.resolve()
    config = load_config(source / "config.yaml")
    if config["dataset"] != "eb_noscale":
        raise ValueError("This experiment is for EB-Noscale.")
    source_status = json.loads((source / "status.json").read_text())
    if source_status.get("pair", {}).get("status") != "completed":
        raise ValueError("The shared conditional path is incomplete.")
    if args.stage == "couplings":
        config["coupling"]["minibatch_blocks"] = args.blocks
        config["coupling"]["parallel_workers"] = 2
    else:
        if source_status.get("couplings", {}).get("status") != "completed":
            raise ValueError("The source coupling is incomplete.")
        variant, activation, last_activation, width, depth = RECIPES[args.recipe]
        config["seed"] = args.seed
        config["matching"].update(
            network_variant=variant,
            growth_network_variant=variant,
            score_network_variant=variant,
            feedback_activation=activation,
            last_hidden_activation=last_activation,
            hidden_width=width,
            hidden_layers=depth,
            learning_rate=args.learning_rate,
            min_learning_rate=1e-5,
            steps=args.steps,
            validation_every=0,
        )
    configure(config)
    first_run = not (destination / "config.yaml").exists()
    init_run(destination, config)
    if first_run:
        names = ["pair_path.pt", "data_metadata.json"]
        if args.stage == "matching":
            names += ["distance_cost_table.pt", "distance_table_validation.json"]
        for name in names:
            path = source / name
            if path.exists():
                shutil.copy2(path, destination / name)
        status = {"pair": source_status["pair"]}
        if args.stage == "matching":
            shutil.copytree(source / "couplings", destination / "couplings")
            status["couplings"] = source_status["couplings"]
        json_write(destination / "status.json", status)
        json_write(destination / "tuning_setup.json", {
            "source_run": str(source),
            "stage": args.stage,
            "pair_path_reused": True,
            "couplings_reused": args.stage == "matching",
            "matching_initialization": "fresh; no feedback weights or optimizer state copied",
            "training_seed": config["seed"],
            "selection_metric": "mean W1 over noninitial observed times; report mass error separately",
            "evaluation": "all initial cells, all target cells, continuous SDE, exact empirical W1",
        })
    return config, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["couplings", "matching"])
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--recipe", choices=RECIPES, default="affine_silu")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[342, 343])
    args = parser.parse_args()
    config, run = prepare(args)
    if args.stage == "couplings":
        if not solve_couplings(config, run):
            raise RuntimeError("Coupling did not converge; inspect status.json.")
        return

    train_matching(config, run)
    frames = [evaluate_sde_seed(run, seed) for seed in args.evaluation_seeds]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(run / "tuning_metrics.csv", index=False)
    noninitial = metrics.loc[metrics.time > 0]
    per_seed = noninitial.groupby("seed").agg(
        W1=("W1", "mean"), mass_error=("mass_relative_error", "mean"))
    summary = {
        "status": "completed",
        "dataset": config["dataset"],
        "recipe": args.recipe,
        "learning_rate": args.learning_rate,
        "matching_steps": args.steps,
        "training_seed": args.seed,
        "coupling_blocks": config["coupling"]["minibatch_blocks"],
        "model_sigma": config["model_sigma"],
        "growth_penalty": config["growth_penalty"],
        "evaluation_seeds": args.evaluation_seeds,
        "mean_W1": float(per_seed.W1.mean()),
        "rollout_sd_W1": float(per_seed.W1.std(ddof=1)) if len(per_seed) > 1 else None,
        "mean_mass_relative_error": float(per_seed.mass_error.mean()),
        "max_mass_relative_error": float(noninitial.mass_relative_error.max()),
        "per_time": noninitial.groupby("time").agg(
            W1=("W1", "mean"), mass_relative_error=("mass_relative_error", "mean")
        ).reset_index().to_dict(orient="records"),
    }
    json_write(run / "tuning_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
