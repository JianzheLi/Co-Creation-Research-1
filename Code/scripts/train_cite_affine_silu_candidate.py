"""Run a controlled CITE affine-LayerNorm, SiLU Matching candidate.

Only the already-trained conditional path and Couplings are reused from the
source run. Matching starts with fresh random weights and Adam state.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.matching import feedback_architecture, matching_version, train_matching
from src.runtime import configure, file_hash, init_run, json_write, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--lr-schedule",
                        choices=("cosine", "cosine_2cycle", "cosine_2cycle_half"),
                        default="cosine")
    args = parser.parse_args()

    source, destination = args.source_run.resolve(), args.output_run.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing candidate: {destination}")
    source_status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if (source_status.get("pair", {}).get("status") != "completed"
            or source_status.get("couplings", {}).get("status") != "completed"):
        raise RuntimeError("source run must have completed pair path and Coupling")

    config = load_config(source / "config.yaml")
    matching = config["matching"]
    matching.update({
        "steps": args.steps,
        "feedback_activation": "silu",
        "last_hidden_activation": "silu",
        "network_variant": "plain_output_affine_norm",
        "growth_network_variant": "plain_output_affine_norm",
        "score_network_variant": "plain_output_affine_norm",
        "gradient_clip_norm": args.gradient_clip_norm,
        "lr_schedule": args.lr_schedule,
        "validation_every": 0,
    })
    matching.pop("warm_start", None)
    configure(config)
    init_run(destination, config)

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
            "status": "random_initialization", "warm_start": False,
            "seed": config["seed"],
        },
    })
    provenance = {
        "pair_stage_reused_from": str(source),
        "coupling_stage_reused_from": str(source),
        "pair_checkpoint_sha256": file_hash(destination / "pair_path.pt"),
        "coupling_manifest_sha256": file_hash(destination_manifest),
        "matching_initialization": "fresh random initialization using configured seed",
        "matching_warm_start": False,
        "source_feedback_checkpoint_loaded": False,
        "optimizer_state_loaded": False,
        "network_variant": "plain_output_affine_norm",
        "layernorm_affine": True,
        "activation": "silu",
        "gradient_clip_norm": args.gradient_clip_norm,
        "lr_schedule": args.lr_schedule,
        "objective_unchanged": True,
    }
    json_write(destination / "provenance.json", provenance)
    json_write(destination / "matching_initialization.json", {
        "warm_start": False,
        "seed": config["seed"],
        "feedback_architecture": feedback_architecture(config),
        "matching_version": matching_version(config),
        "source_feedback_checkpoint_loaded": False,
        "optimizer_state_loaded": False,
        "network_variant": "plain_output_affine_norm",
        "layernorm_affine": True,
        "activation": "silu",
    })

    print(json.dumps({"source": str(source), "destination": str(destination),
                      "steps": args.steps, "seed": config["seed"],
                      "gradient_clip_norm": args.gradient_clip_norm,
                      "lr_schedule": args.lr_schedule, "status": "training"},
                     ensure_ascii=False), flush=True)
    train_matching(config, destination)
    print(json.dumps({"destination": str(destination), "status": "completed"},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
