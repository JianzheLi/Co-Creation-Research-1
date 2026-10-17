from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config
from src.audit import audit_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gene_practical.yaml")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    audit_run(load_config(args.config), args.run_dir)
