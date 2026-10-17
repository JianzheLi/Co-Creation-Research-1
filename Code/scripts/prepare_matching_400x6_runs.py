"""Prepare isolated Matching-only runs from completed original-method paths."""
from pathlib import Path
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import file_hash, json_write, load_config


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("gene", "mouse", "emt")


def main():
    for dataset in DATASETS:
        source = ROOT / "../Results" / f"{dataset}_distance_hermite_primal_seed42"
        config_path = ROOT / "configs" / f"{dataset}_matching_tanhlast_400x6_alpha10_sigma025.yaml"
        destination = ROOT / "../Results" / f"{dataset}_matching_tanhlast_400x6_alpha10_sigma025_seed42"
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty run directory: {destination}")
        status = json.loads((source / "status.json").read_text(encoding="utf-8"))
        if status.get("pair", {}).get("status") != "completed" or status.get("couplings", {}).get("status") != "completed":
            raise RuntimeError(f"completed path and couplings required: {source}")
        config = load_config(config_path)
        if config["pair"] != load_config(source / "config.yaml")["pair"]:
            raise RuntimeError(f"path architecture/settings differ for {dataset}")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "pair_path.pt", destination / "pair_path.pt")
        shutil.copytree(source / "couplings", destination / "couplings")
        table = source / "distance_cost_table.pt"
        if table.exists():
            shutil.copy2(table, destination / table.name)
        kept_status = {key: status[key] for key in ("pair", "couplings")}
        json_write(destination / "status.json", kept_status)
        json_write(destination / "matching_lineage.json", {
            "dataset": dataset,
            "source_run": str(source.resolve()),
            "source_pair_path_sha256": file_hash(source / "pair_path.pt"),
            "source_coupling_manifest_sha256": file_hash(source / "couplings" / "manifest.json"),
            "matching_architecture": {
                "separate_networks": ["velocity", "growth", "score"],
                "hidden_layers": config["matching"]["hidden_layers"],
                "hidden_width": config["matching"]["hidden_width"],
                "hidden_activation": config["matching"]["feedback_activation"],
                "last_hidden_activation": config["matching"]["last_hidden_activation"],
                "output_activation": "linear",
            },
        })
        print(f"Prepared {destination}")


if __name__ == "__main__":
    main()
