import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from scripts.screen_matching_architectures import prepare_run
from src.matching import matching_loss_weights, train_matching

source = ROOT / "../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42"
run = ROOT / "../Results/experiment2/eb_noscale_leakyrelu_residual_branch_scaled_w512x8_a1225_sigma05_blocks8_fm30000_seed42"
if run.exists():
    raise FileExistsError(f"refusing to overwrite an existing experiment: {run}")
base_config = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
config = prepare_run(source, run, base_config, ("residual_branch_scaled", 512, 8), 30000,
                     endpoint_regularized_four_networks=True)
config["matching"]["feedback_activation"] = "leaky_relu"
config["matching"]["last_hidden_activation"] = "leaky_relu"
(run / "screen_config.yaml").write_text(
    yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
if matching_loss_weights(config) != matching_loss_weights(base_config):
    raise RuntimeError("Matching loss weights changed in the architecture-only experiment")
print(json.dumps({"run": str(run), "config": str(run / "screen_config.yaml"),
                  "loss_weights": matching_loss_weights(config),
                  "reused_pair_from": str(source), "matching_steps": config["matching"]["steps"]},
                 ensure_ascii=False), flush=True)
train_matching(config, run)
status = json.loads((run / "status.json").read_text(encoding="utf-8"))
print(json.dumps(status["matching"], ensure_ascii=False), flush=True)
