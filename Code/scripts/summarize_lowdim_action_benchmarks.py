"""Summarize paired RUOT/WFR Action Monte Carlo results for this benchmark suite."""
from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "../Results" / "lowdim_action_benchmarks"
OUT = RUNS / "action_comparison_summary.csv"
MC_DIR = "action_monte_carlo_20_seed1000"
MIN_CLEAR_RELATIVE_DIFFERENCE_PERCENT = 1.0


def main():
    rows = []
    for summary_path in sorted(RUNS.glob(f"*/{MC_DIR}/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run = summary_path.parent.parent
        config = yaml.safe_load((run / "ruot" / "config.yaml").read_text(encoding="utf-8"))
        ruot = summary["method_summaries"]["RUOT"]["total_action"]["mean"]
        wfr = summary["method_summaries"]["WFR-FM bridge"]["total_action"]["mean"]
        paired = summary["paired_wfr_minus_ruot"]["total_action"]
        lower, upper = paired["ci95"]
        relative = float(paired["mean_percent_of_RUOT_mean"])
        rows.append({
            "setting": run.name,
            "dimension": int(config["dimension"]),
            "alpha": float(config["growth_penalty"]),
            "sigma": float(config["model_sigma"]),
            "n_paired_rollouts": int(paired["n"]),
            "ruot_action_mean": ruot,
            "wfr_action_mean": wfr,
            "wfr_minus_ruot_mean": float(paired["mean"]),
            "relative_difference_percent": relative,
            "paired_ci95_low": lower,
            "paired_ci95_high": upper,
            "clear_difference": bool((lower > 0 or upper < 0)
                                      and abs(relative) >= MIN_CLEAR_RELATIVE_DIFFERENCE_PERCENT),
            "mc_summary": str(summary_path.relative_to(ROOT)),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        print("No completed 20-replicate summaries found yet.")
        return
    result = result.sort_values("relative_difference_percent", key=lambda s: s.abs(), ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nClear-difference rule: paired 95% t interval excludes zero and |relative Action gap| >= "
          f"{MIN_CLEAR_RELATIVE_DIFFERENCE_PERCENT:g}%.")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
