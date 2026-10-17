"""Audit full-pipeline and paired-Monte-Carlo evidence for the benchmark suite."""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "../Results" / "lowdim_action_benchmarks"
MC_DIR = "action_monte_carlo_20_seed1000"
STAGES = ("pair", "couplings", "matching", "evaluation")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summaries = sorted(RUNS.glob(f"*/{MC_DIR}/summary.json"))
    failures = []
    clear_families = set()
    audited_settings = []
    for summary_path in summaries:
        run = summary_path.parent.parent
        ruot_dir, wfr_dir = run / "ruot", run / "wfr"
        cfg = yaml.safe_load((ruot_dir / "config.yaml").read_text(encoding="utf-8"))
        wfr_cfg = yaml.safe_load((wfr_dir / "config.yaml").read_text(encoding="utf-8"))
        summary = read_json(summary_path)
        for method, method_dir in (("RUOT", ruot_dir), ("WFR", wfr_dir)):
            status = read_json(method_dir / "status.json")
            for stage in STAGES:
                if status.get(stage, {}).get("status") != "completed":
                    failures.append(f"{run.name}/{method}: {stage} not completed")
            manifest = read_json(method_dir / "couplings" / "manifest.json")
            if manifest.get("status") not in {"converged", "audited"}:
                failures.append(f"{run.name}/{method}: Coupling manifest is not converged/audited")
            if len(manifest.get("intervals", [])) != len(cfg["model_times"]) - 1:
                failures.append(f"{run.name}/{method}: Coupling interval count mismatch")
            for interval in manifest.get("intervals", []):
                if method == "RUOT" and interval.get("status") != "converged":
                    failures.append(f"{run.name}/{method}: interval Coupling did not converge")
                if (method == "WFR" and interval.get("oet_semicoupling_relative_gap", 1.0)
                        > wfr_cfg["coupling"]["max_oet_semicoupling_relative_gap"]):
                    failures.append(f"{run.name}/{method}: interval semi-coupling gap too large")
                if (interval.get("row_relative_tv", 0.0) > 0.001
                        or interval.get("column_relative_tv", 0.0) > 0.001):
                    failures.append(f"{run.name}/{method}: interval marginal TV exceeds tolerance")
            if (method == "WFR" and
                    abs(float(manifest.get("alpha", -1)) - float(cfg["growth_penalty"])) > 1e-12):
                failures.append(f"{run.name}/{method}: Coupling alpha mismatch")
            environment = read_json(method_dir / "environment.json")
            if "5080" not in str(environment.get("gpu", "")):
                failures.append(f"{run.name}/{method}: run environment did not record RTX 5080")
            if not str(status.get("couplings", {}).get("device", "")).startswith("cuda"):
                failures.append(f"{run.name}/{method}: Coupling was not recorded on CUDA")
        ruot_pair = torch.load(ruot_dir / "pair_path.pt", map_location="cpu", weights_only=False)
        ruot_feedback = torch.load(ruot_dir / "feedback.pt", map_location="cpu", weights_only=False)
        wfr_pair = torch.load(wfr_dir / "pair_path.pt", map_location="cpu", weights_only=False)
        wfr_feedback = torch.load(wfr_dir / "feedback.pt", map_location="cpu", weights_only=False)
        if cfg["pair"].get("training_steps") != 6000 or ruot_pair.get("completed_steps") != 6000:
            failures.append(f"{run.name}: RUOT path did not complete exactly 6000 updates")
        if (ruot_pair.get("config", {}).get("growth_penalty") != cfg["growth_penalty"]
                or wfr_pair.get("alpha") != cfg["growth_penalty"]):
            failures.append(f"{run.name}: path checkpoint alpha mismatch")
        if cfg["matching"].get("steps") != 30000 or wfr_cfg["matching"].get("steps") != 30000:
            failures.append(f"{run.name}: Matching did not use 30000 steps")
        if (ruot_feedback.get("completed_steps") != 30000
                or wfr_feedback.get("completed_steps") != 30000):
            failures.append(f"{run.name}: one or both Matching models did not complete 30000 updates")
        if cfg["coupling"].get("mode") != "full" or wfr_cfg["coupling"].get("mode") != "full":
            failures.append(f"{run.name}: Couplings are not full matrices")
        for key in ("dimension", "model_times", "model_sigma", "growth_penalty", "generation"):
            if cfg.get(key) != wfr_cfg.get(key):
                failures.append(f"{run.name}: RUOT and WFR differ in shared config field {key}")
        if (wfr_pair.get("variance") != "sigma^2*t*(1-t) Brownian bridge heuristic"
                or summary.get("sigma") != cfg["model_sigma"]
                or summary.get("alpha") != cfg["growth_penalty"]
                or summary.get("psi") != "g^2/2"):
            failures.append(f"{run.name}: WFR Brownian-bridge / Action configuration audit failed")
        paired = summary.get("paired_wfr_minus_ruot", {}).get("total_action", {})
        if paired.get("n") != 20 or summary.get("paired_common_random_numbers") is not True:
            failures.append(f"{run.name}: Action comparison is not 20 paired rollouts")
        if summary.get("mc_seed_range") != [1000, 1019]:
            failures.append(f"{run.name}: MC seed range mismatch")
        relative = float(paired.get("mean_percent_of_RUOT_mean", np.nan))
        lo, hi = paired.get("ci95", [np.nan, np.nan])
        is_clear = abs(relative) >= 1 and (float(lo) > 0 or float(hi) < 0)
        if is_clear:
            clear_families.add(re.sub(r"_alpha[^_]+$", "", run.name))
        audited_settings.append((run.name, float(cfg["growth_penalty"]), relative, is_clear))

    summary_csv = pd.read_csv(RUNS / "action_comparison_summary.csv")
    csv_settings = set(summary_csv["setting"])
    audited_setting_names = {row[0] for row in audited_settings}
    if csv_settings != audited_setting_names:
        failures.append("summary CSV does not exactly cover the audited completed MC settings")
    if len(clear_families) < 5:
        failures.append(f"only {len(clear_families)} distinct data families meet the clear-difference rule")
    if failures:
        raise SystemExit("AUDIT FAILED:\n- " + "\n- ".join(failures))

    print(f"AUDIT PASS: {len(audited_settings)} complete dataset/alpha settings; "
          f"{len(clear_families)} distinct data families meet the clear-difference criterion.")
    for name, alpha, relative, is_clear in audited_settings:
        print(f"  {name}: alpha={alpha:g}, paired Action gap={relative:+.3f}%, clear={is_clear}")


if __name__ == "__main__":
    main()
