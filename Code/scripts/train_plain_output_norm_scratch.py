"""Train non-affine LayerNorm Matching models from random initialization.

The completed pair paths and couplings are reused from the prior non-affine
runs. No feedback weights or optimizer state are copied. Matching is trained
for 30,000 steps with the original dataset-specific configuration.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.matching import feedback_architecture, matching_version, train_matching
from src.runtime import configure, file_hash, init_run, json_write, load_config


OUTPUT_ROOT = ROOT / "../Results" / "experiment2" / "nonaffine_layernorm_scratch"
SOURCE_RUNS = {
    "cite": ROOT / "../Results" / "experiment2" / "cite_bounded_allfield_outputnorm_warmstart_leakyrelu_w512x8_growth4_lr3e3to1e5_30000_seed42",
    "eb_noscale": ROOT / "../Results" / "experiment2" / "eb_noscale_leakyrelu_allfield_outputnorm_warmstart_w512x8_a1225_sigma05_blocks8_fm30000_seed42",
    "weinreb": ROOT / "../Results" / "experiment2" / "weinreb_silu_allfield_outputnorm_warmstart_w512x8_a225_sigma05_blocks16_fm30000_seed42",
}


def train_one(dataset: str, source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing scratch run: {destination}")
    source_status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if (source_status.get("pair", {}).get("status") != "completed"
            or source_status.get("couplings", {}).get("status") != "completed"):
        raise RuntimeError(f"source pair path/coupling are incomplete: {source}")

    config = load_config(source / "config.yaml")
    matching = config["matching"]
    matching["network_variant"] = "plain_output_norm"
    matching["growth_network_variant"] = "plain_output_norm"
    matching["score_network_variant"] = "plain_output_norm"
    matching["steps"] = 30000
    matching["validation_every"] = 0
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
        "network_variant": "plain_output_norm",
        "layernorm_affine": False,
        "matching_steps": matching["steps"],
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
        "network_variant": "plain_output_norm",
        "layernorm_affine": False,
    })

    print(json.dumps({"dataset": dataset, "source_run": str(source),
                      "destination": str(destination), "status": "training",
                      "steps": matching["steps"], "seed": config["seed"]},
                     ensure_ascii=False), flush=True)
    train_matching(config, destination)
    print(json.dumps({"dataset": dataset, "destination": str(destination),
                      "status": "completed"}, ensure_ascii=False), flush=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset, source in SOURCE_RUNS.items():
        train_one(dataset, source, OUTPUT_ROOT / dataset)


if __name__ == "__main__":
    main()
