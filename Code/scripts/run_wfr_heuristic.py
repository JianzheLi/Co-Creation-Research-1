import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config
from src.wfr_heuristic import run_wfr_heuristic


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the WFR-FM bridge heuristic")
    parser.add_argument("--config", default="configs/gene_wfr_bridge.yaml")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    metrics = run_wfr_heuristic(load_config(args.config), Path(args.run_dir))
    print(metrics[["time", "W1_normalized", "W2_normalized", "mass_pred", "mass_true",
                   "mass_relative_error"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
