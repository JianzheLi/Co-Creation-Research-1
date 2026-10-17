"""Train affine-output-LayerNorm Matching models from random initialization.

The source run contributes only its already-trained pair path, couplings, and
data metadata. No feedback weights or optimizer state are loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code: src/

import argparse
import json
import shutil
from pathlib import Path

from src.matching import feedback_architecture, matching_version, train_matching
from src.runtime import configure, file_hash, init_run, json_write, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True,
                        help="completed non-affine run supplying only path and couplings")
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()

    source, destination = args.source_run.resolve(), args.output_run.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing candidate: {destination}")
    config = load_config(source / "config.yaml")
    source_status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if (source_status.get("pair", {}).get("status") != "completed"
            or source_status.get("couplings", {}).get("status") != "completed"):
        raise RuntimeError("source run must have completed pair path and coupling")

    # Change only the LayerNorm affine setting. The architecture variant is
    # explicit in checkpoints, while Matching's objective/version is unchanged.
    config["matching"]["steps"] = args.steps
    config["matching"]["network_variant"] = "plain_output_affine_norm"
    config["matching"]["growth_network_variant"] = "plain_output_affine_norm"
    config["matching"]["score_network_variant"] = "plain_output_affine_norm"
    config["matching"]["validation_every"] = 0
    configure(config)
    destination = init_run(destination, config)

    for name in ("pair_path.pt", "data_metadata.json", "distance_cost_table.pt",
                 "distance_table_validation.json"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "couplings", destination / "couplings")

    source_manifest = source / "couplings" / "manifest.json"
    destination_manifest = destination / "couplings" / "manifest.json"
    if file_hash(source / "pair_path.pt") != file_hash(destination / "pair_path.pt"):
        raise RuntimeError("pair path copy failed hash verification")
    if file_hash(source_manifest) != file_hash(destination_manifest):
        raise RuntimeError("coupling manifest copy failed hash verification")

    json_write(destination / "status.json", {
        "pair": source_status["pair"],
        "couplings": source_status["couplings"],
        "matching_initialization": {
            "status": "random_initialization",
            "warm_start": False,
            "seed": config["seed"],
        },
    })
    json_write(destination / "provenance.json", {
        "pair_stage_reused_from": str(source),
        "coupling_stage_reused_from": str(source),
        "pair_checkpoint_sha256": file_hash(destination / "pair_path.pt"),
        "coupling_manifest_sha256": file_hash(destination_manifest),
        "matching_initialization": "fresh random initialization using configured seed",
        "matching_warm_start": False,
        "intervention": "trainable affine LayerNorm on all four matching fields",
        "objective_unchanged": True,
    })
    json_write(destination / "matching_initialization.json", {
        "warm_start": False,
        "seed": config["seed"],
        "feedback_architecture": feedback_architecture(config),
        "matching_version": matching_version(config),
        "source_feedback_checkpoint_loaded": False,
        "optimizer_state_loaded": False,
    })

    print(f"Training affine LayerNorm Matching from scratch: {destination}", flush=True)
    train_matching(config, destination)
    print("Matching completed", flush=True)


if __name__ == "__main__":
    main()
