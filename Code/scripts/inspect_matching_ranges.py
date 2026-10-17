"""Measure empirical Conditional Path target and fitted-field ranges on GPU."""
from pathlib import Path
import json
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.matching import CouplingSampler, make_feedback, sample_training_batch
from src.runtime import configure, file_hash, load_config, load_path, snapshots_for


def summarize(values):
    values = torch.cat(values).flatten().abs()
    probs = torch.tensor([0.99, 0.999, 0.9999, 1.0], device=values.device)
    return {f"abs_q{p:g}": float(v) for p, v in zip(probs.tolist(), torch.quantile(values, probs).tolist())}


@torch.no_grad()
def main():
    run = Path("../Results/gene_steps_gpu_4000_seed42")
    config = load_config(run / "config.yaml")
    device = configure(config)
    if device.type != "cuda":
        raise RuntimeError("GPU required")
    coupling = CouplingSampler(run, config, device)
    path = load_path(run, config, device)
    checkpoint = torch.load(run / "feedback.pt", map_location=device, weights_only=False)
    if checkpoint["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt"):
        raise RuntimeError("stale feedback checkpoint")
    model = make_feedback(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    names = ("state_position", "flow_target", "growth_target", "score_target", "flow_fitted", "growth_fitted",
             "score_fitted", "drift_fitted")
    components = {name: [] for name in names}
    norms = {name: [] for name in names}
    for _ in range(32):
        global_t, targets, _ = sample_training_batch(coupling, path, 2048, device)
        target_v = targets["target_v_scaled"] / targets["sqrt_wf"]
        target_g = targets["target_g"]
        target_s = targets["target_s_scaled"] / targets["sqrt_ws"]
        v, g, s = model(global_t, targets["x"])
        drift = v + 0.5 * config["model_sigma"]**2 * s
        for name, tensor in (("state_position", targets["x"]), ("flow_target", target_v), ("growth_target", target_g),
                             ("score_target", target_s), ("flow_fitted", v),
                             ("growth_fitted", g), ("score_fitted", s), ("drift_fitted", drift)):
            components[name].append(tensor.abs())
            if tensor.ndim > 1 and tensor.shape[-1] > 1:
                norms[name].append(tensor.norm(dim=-1))
            else:
                norms[name].append(tensor.reshape(-1).abs())
    result = {
        "device": torch.cuda.get_device_name(device),
        "samples": 32 * 2048,
        "component_abs_quantiles": {key: summarize(value) for key, value in components.items()},
        "norm_quantiles": {key: summarize(value) for key, value in norms.items()},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
