"""Short, sequential architecture screen for the Matching feedback networks.

All candidates reuse an identical trained pair path and coupling. Only the
feedback architecture changes. A fixed diagnostic set is sampled after
training and is not used for optimization.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DeepRUOT/Code: src/

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import torch
import yaml

from src.matching import (CouplingSampler, load_path, make_feedback,
                          matching_loss, matching_loss_weights,
                          sample_training_batch, train_matching)
from src.runtime import configure, file_hash
from scripts.evaluate_our_ode_sde import evaluate_run


ROOT = Path(__file__).resolve().parents[1]

ARCHITECTURES = {
    "plain_6x400": ("plain", 400, 6),
    "residual_6x400": ("residual", 400, 6),
    "gated_6x300": ("gated", 300, 6),
    "gated_residual_6x300": ("gated_residual", 300, 6),
    "fourier_6x420": ("fourier", 420, 6),
}
ENDPOINT_FOUR_NETWORK_ARCHITECTURES = {
    # Hold depth and width fixed so this screen compares network structure.
    "vanilla_mlp_6x400": ("plain", 400, 6),
    "residual_mlp_6x400": ("residual", 400, 6),
    "gated_mlp_6x400": ("gated", 400, 6),
    "gated_residual_mlp_6x400": ("gated_residual", 400, 6),
    "fourier_mlp_6x400": ("fourier", 400, 6),
}


def prepare_run(source: Path, destination: Path, base_config: dict, architecture: tuple,
                steps: int, endpoint_regularized_four_networks: bool = False) -> dict:
    if destination.exists() and not (destination / "pair_path.pt").exists():
        raise RuntimeError(f"refusing to use incomplete screen directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("pair_path.pt", "status.json", "distance_cost_table.pt", "distance_table_validation.json"):
        src = source / name
        if src.exists() and not (destination / name).exists():
            shutil.copy2(src, destination / name)
    src_couplings = source / "couplings"
    dst_couplings = destination / "couplings"
    if src_couplings.exists() and not dst_couplings.exists():
        shutil.copytree(src_couplings, dst_couplings)
    if file_hash(source / "pair_path.pt") != file_hash(destination / "pair_path.pt"):
        raise RuntimeError("screen run does not use the source pair-path checkpoint")
    src_manifest = source / "couplings" / "manifest.json"
    dst_manifest = destination / "couplings" / "manifest.json"
    if file_hash(src_manifest) != file_hash(dst_manifest):
        raise RuntimeError("screen run does not use the source coupling manifest")

    config = copy.deepcopy(base_config)
    variant, width, layers = architecture
    config["matching"]["steps"] = steps
    config["matching"]["hidden_width"] = width
    config["matching"]["hidden_layers"] = layers
    config["matching"]["network_variant"] = variant
    config["matching"]["validation_every"] = 0
    if endpoint_regularized_four_networks:
        config["matching"]["field_parameterization"] = "endpoint_regularized"
        config["matching"]["independent_velocity_networks"] = True
        config["matching"]["absorb_sigma_in_singular_velocity"] = True
        config["matching"]["loss_weighting"] = "endpoint_explicit_sigma"
    saved_config = destination / "config.yaml"
    if saved_config.exists():
        existing = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"screen directory already has another config: {destination}")
    return config


@torch.no_grad()
def diagnostic_loss(run: Path, config: dict, batches: int, batch_size: int) -> dict:
    device = configure(config)
    sampler = CouplingSampler(run, config, device)
    path = load_path(run, config, device)
    model = make_feedback(config, device)
    checkpoint = torch.load(run / "feedback.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(192837)
    totals = {}
    for _ in range(batches):
        global_t, targets, factor = sample_training_batch(sampler, path, batch_size, device, generator)
        loss, parts = matching_loss(model.matching_fields(global_t, targets["x"]), targets, factor,
                                    matching_loss_weights(config))
        totals["loss"] = totals.get("loss", 0.0) + loss.item()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value.item()
    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default="../Results/experiment2/our_method/eb_noscale")
    parser.add_argument("--output-root", default="../Results/experiment2/architecture_screen/eb_noscale")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--endpoint-regularized-four-networks", action="store_true",
                        help="fit finite flow, singular flow, growth, and score with four independent MLPs")
    parser.add_argument("--diagnostic-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--evaluate-w1", action="store_true",
                        help="evaluate ODE and SDE rollouts against all observed points and rank by SDE W1")
    parser.add_argument("--steps-per-unit", type=int,
                        help="override rollout evaluation resolution (defaults to the saved config)")
    args = parser.parse_args()
    source = (ROOT / args.source_run).resolve()
    output_root = (ROOT / args.output_root).resolve()
    base_config = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    source_status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if source_status.get("pair", {}).get("status") != "completed" or source_status.get("couplings", {}).get("status") != "completed":
        raise RuntimeError("source run must have completed pair-path and coupling stages")

    architectures = (ENDPOINT_FOUR_NETWORK_ARCHITECTURES if args.endpoint_regularized_four_networks
                     else ARCHITECTURES)
    results = []
    for name, architecture in architectures.items():
        run = output_root / name
        config = prepare_run(source, run, base_config, architecture, args.steps,
                             endpoint_regularized_four_networks=args.endpoint_regularized_four_networks)
        config_path = run / "screen_config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"\n=== {name}: {args.steps} Matching steps ===", flush=True)
        train_matching(config, run)
        run_status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        elapsed = float(run_status["matching"]["seconds"])
        metric = diagnostic_loss(run, config, args.diagnostic_batches, args.batch_size)
        w1_metric = {}
        if args.evaluate_w1:
            evaluated = evaluate_run(run, args.steps_per_unit, seed=342)
            noninitial = evaluated[evaluated.time > 0]
            for mode in ("ODE", "SDE"):
                values = noninitial[noninitial.inference == mode].W1
                w1_metric[f"{mode.lower()}_mean_W1"] = float(values.mean())
                w1_metric[f"{mode.lower()}_median_W1"] = float(values.median())
            mass = noninitial[noninitial.inference == "SDE"].mass_relative_error
            w1_metric["sde_mean_mass_relative_error"] = float(mass.mean())
        row = {"architecture": name, "variant": architecture[0], "width": architecture[1],
               "layers": architecture[2], "field_parameterization": config["matching"].get("field_parameterization", "raw"),
               "independent_velocity_networks": config["matching"].get("independent_velocity_networks", False),
               "loss_weighting": config["matching"].get("loss_weighting", "none"),
               "loss_weights": json.dumps(matching_loss_weights(config), sort_keys=True),
               "steps": args.steps, "seconds": elapsed, **metric, **w1_metric}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    rank_key = "sde_mean_W1" if args.evaluate_w1 else "loss"
    results.sort(key=lambda row: row[rank_key])
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "screen_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n=== Ranking (lower {rank_key} is better) ===", flush=True)
    for row in results:
        ranking = (f"SDE W1={row['sde_mean_W1']:.6g}, ODE W1={row['ode_mean_W1']:.6g}"
                   if args.evaluate_w1 else f"loss={row['loss']:.6g}")
        print(f"{row['architecture']:24s} {ranking}; weighted loss={row['loss']:.6g}; "
              f"flow={row['flow']:.6g}, growth={row['growth']:.6g}, score={row['score']:.6g}", flush=True)


if __name__ == "__main__":
    main()
