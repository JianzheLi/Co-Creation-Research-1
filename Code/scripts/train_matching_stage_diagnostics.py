"""Reproduce a Matching run and evaluate fixed SDE seeds at selected steps."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DeepRUOT/Code: src/

import argparse
import json
import shutil
from pathlib import Path

from src.matching import train_matching
from src.runtime import configure, file_hash, init_run, json_write, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+",
                        default=[1, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[342, 343])
    parser.add_argument("--stop-after-step", type=int,
                        help="stop after a selected diagnostic checkpoint, preserving the original LR schedule")
    args = parser.parse_args()
    source, destination = args.source_run.resolve(), args.output_run.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic run: {destination}")
    config = load_config(source / "config.yaml")
    source_status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if source_status.get("pair", {}).get("status") != "completed" or source_status.get("couplings", {}).get("status") != "completed":
        raise RuntimeError("source run must have completed pair path and coupling")
    config["matching"]["diagnostic_rollout_steps"] = args.steps
    config["matching"]["diagnostic_rollout_seeds"] = args.seeds
    if args.stop_after_step is not None:
        config["matching"]["diagnostic_stop_after_step"] = args.stop_after_step
    configure(config)
    destination = init_run(destination, config)
    shutil.copy2(source / "pair_path.pt", destination / "pair_path.pt")
    if (source / "data_metadata.json").exists():
        shutil.copy2(source / "data_metadata.json", destination / "data_metadata.json")
    shutil.copytree(source / "couplings", destination / "couplings")
    for name in ("distance_cost_table.pt", "distance_table_validation.json"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    manifest = destination / "couplings" / "manifest.json"
    if file_hash(source / "pair_path.pt") != file_hash(destination / "pair_path.pt"):
        raise RuntimeError("pair path copy failed hash verification")
    if file_hash(source / "couplings" / "manifest.json") != file_hash(manifest):
        raise RuntimeError("coupling manifest copy failed hash verification")
    json_write(destination / "status.json", {
        "pair": source_status["pair"], "couplings": source_status["couplings"],
        "diagnostic": {
            "status": "scheduled", "source_run": str(source),
            "steps": args.steps, "seeds": args.seeds,
            "objective_unchanged": True,
        },
    })
    json_write(destination / "provenance.json", {
        "pair_stage_reused_from": str(source), "coupling_stage_reused_from": str(source),
        "pair_checkpoint_sha256": file_hash(destination / "pair_path.pt"),
        "coupling_manifest_sha256": file_hash(manifest),
        "diagnostic_only_config_keys": ["matching.diagnostic_rollout_steps",
                                         "matching.diagnostic_rollout_seeds",
                                         "matching.diagnostic_stop_after_step"],
        "matching_objective_unchanged": True,
    })
    print(f"Replaying Matching with rollout checkpoints: {destination}", flush=True)
    train_matching(config, destination)
    print(f"Diagnostics saved to {destination / 'matching_rollout_diagnostics.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
