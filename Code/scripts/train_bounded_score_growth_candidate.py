"""Warm-start Matching with globally bounded Score/Growth readouts.

Velocity fields, pair path, coupling, objective, loss weights, and optimizer
schedule are inherited unchanged from the source run. The non-affine output
normalizers add no trainable parameters.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code: src/

import argparse
import json
import shutil
from pathlib import Path

import torch

from src.matching import (feedback_architecture, make_feedback, matching_version,
                          train_matching)
from src.runtime import configure, file_hash, init_run, json_write, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--bound-velocity", action="store_true",
                        help="also normalize the finite and singular velocity readouts")
    args = parser.parse_args()
    source, destination = args.source_run.resolve(), args.output_run.resolve()
    config = load_config(source / "config.yaml")
    source_status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if source_status.get("pair", {}).get("status") != "completed" or source_status.get("couplings", {}).get("status") != "completed":
        raise RuntimeError("source run must have completed pair path and coupling")
    source_checkpoint = torch.load(source / "feedback.pt", map_location="cpu", weights_only=False)
    if source_checkpoint.get("matching_version") != matching_version(config):
        raise RuntimeError("source checkpoint uses a different Matching objective")

    config["matching"]["steps"] = args.steps
    config["matching"]["network_variant"] = ("plain_output_norm" if args.bound_velocity else "plain")
    config["matching"]["growth_network_variant"] = "plain_output_norm"
    config["matching"]["score_network_variant"] = "plain_output_norm"
    config["matching"]["validation_every"] = 0
    configure(config)
    if destination.exists():
        if (destination / "feedback.pt").exists():
            raise FileExistsError(f"refusing to overwrite a candidate checkpoint: {destination}")
        saved_config = destination / "config.yaml"
        if saved_config.exists():
            import yaml
            if yaml.safe_load(saved_config.read_text(encoding="utf-8")) != config:
                raise RuntimeError(f"partial candidate has a different config: {destination}")
    destination = init_run(destination, config)
    if not (destination / "pair_path.pt").exists():
        shutil.copy2(source / "pair_path.pt", destination / "pair_path.pt")
    if (source / "data_metadata.json").exists() and not (destination / "data_metadata.json").exists():
        shutil.copy2(source / "data_metadata.json", destination / "data_metadata.json")
    if not (destination / "couplings").exists():
        shutil.copytree(source / "couplings", destination / "couplings")
    for name in ("distance_cost_table.pt", "distance_table_validation.json"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    if file_hash(source / "pair_path.pt") != file_hash(destination / "pair_path.pt"):
        raise RuntimeError("pair path copy failed hash verification")
    source_manifest = source / "couplings" / "manifest.json"
    destination_manifest = destination / "couplings" / "manifest.json"
    if file_hash(source_manifest) != file_hash(destination_manifest):
        raise RuntimeError("coupling manifest copy failed hash verification")

    model = make_feedback(config, torch.device(config.get("device", "cuda")))
    incompatible = model.load_state_dict(source_checkpoint["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"warm-start parameter mismatch: {incompatible}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["matching"]["learning_rate"])
    warmstart = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "completed_steps": 0, "matching_version": matching_version(config),
        "feedback_architecture": feedback_architecture(config), "seconds": 0.0,
        "torch_rng": source_checkpoint.get("torch_rng", torch.get_rng_state()),
        "cuda_rng": source_checkpoint.get("cuda_rng", torch.cuda.get_rng_state_all()),
        "path_checkpoint_sha256": file_hash(destination / "pair_path.pt"),
        "coupling_manifest_sha256": file_hash(destination_manifest), "config": config,
    }
    torch.save(warmstart, destination / "feedback.pt")
    json_write(destination / "status.json", {
        "pair": source_status["pair"], "couplings": source_status["couplings"],
        "warmstart": {
            "status": "completed", "source_run": str(source),
            "source_feedback_sha256": file_hash(source / "feedback.pt"),
            "intervention": ("non-affine LayerNorm on all four field readouts" if args.bound_velocity
                             else "non-affine LayerNorm on Score and Growth readouts only"),
        },
    })
    json_write(destination / "provenance.json", {
        "pair_stage_reused_from": str(source),
        "coupling_stage_reused_from": str(source),
        "pair_checkpoint_sha256": file_hash(destination / "pair_path.pt"),
        "coupling_manifest_sha256": file_hash(destination_manifest),
        "feedback_warmstart_sha256": file_hash(source / "feedback.pt"),
        "objective_unchanged": True,
        "velocity_network_unchanged_by_construction": not args.bound_velocity,
    })
    print(f"Training bounded Score/Growth candidate: {destination}", flush=True)
    train_matching(config, destination)
    print("Matching completed", flush=True)


if __name__ == "__main__":
    main()
