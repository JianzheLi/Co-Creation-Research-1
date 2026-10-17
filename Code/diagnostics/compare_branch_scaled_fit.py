import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure, load_config, load_path
from src.matching import (CouplingSampler, make_feedback, matching_loss, matching_loss_weights,
                          sample_training_batch)

base = Path(r"../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42").resolve()
cand = Path(r"../Results/experiment2/eb_noscale_leakyrelu_residual_branch_scaled_w512x8_a1225_sigma05_blocks8_fm30000_seed42").resolve()
c0, c1 = load_config(base / "config.yaml"), load_config(cand / "config.yaml")
device = configure(c0)
sampler, path = CouplingSampler(base, c0, device), load_path(base, c0, device)
models = []
for run, config in ((base, c0), (cand, c1)):
    checkpoint = torch.load(run / "feedback.pt", map_location=device, weights_only=False)
    model = make_feedback(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    models.append(model)
generator = torch.Generator(device=device).manual_seed(9472)
bins = [(0.00, .20), (.20, .40), (.40, .55), (.55, .70), (.70, .85), (.85, .98)]
fields = ("vreg", "vsing", "score", "growth")
values = {name: {b: {f: {k: [] for k in ("target", "pred", "err")} for f in fields}
                 for b in bins} for name in ("silu_residual", "leaky_residual_branch_scaled")}
with torch.no_grad():
    for _ in range(16):
        t, targets, factor = sample_training_batch(sampler, path, 2048, device, generator)
        local, interval = (t - t.floor()).squeeze(-1), t.floor().squeeze(-1)
        target = {"vreg": targets["target_v_regular"].norm(dim=-1),
                  "vsing": targets["target_v_singular_scaled"].norm(dim=-1),
                  "score": targets["target_s_scaled"].norm(dim=-1),
                  "growth": targets["target_g"].abs().squeeze(-1)}
        for name, model in zip(values, models):
            prediction = model.matching_fields(t, targets["x"])
            pred = {"vreg": prediction[0].norm(dim=-1), "vsing": prediction[1].norm(dim=-1),
                    "score": prediction[3].norm(dim=-1), "growth": prediction[2].abs().squeeze(-1)}
            error = {"vreg": (prediction[0] - targets["target_v_regular"]).norm(dim=-1),
                     "vsing": (prediction[1] - targets["target_v_singular_scaled"]).norm(dim=-1),
                     "score": (prediction[3] - targets["target_s_scaled"]).norm(dim=-1),
                     "growth": (prediction[2] - targets["target_g"]).abs().squeeze(-1)}
            for a, b in bins:
                mask = (interval == 1) & (local >= a) & (local < b)
                for field in fields:
                    values[name][(a, b)][field]["target"].append(target[field][mask].cpu())
                    values[name][(a, b)][field]["pred"].append(pred[field][mask].cpu())
                    values[name][(a, b)][field]["err"].append(error[field][mask].cpu())

def summarize(chunks):
    x = torch.cat(chunks).double()
    return {"n": len(x), "rms": float(x.square().mean().sqrt()),
            "p99": float(torch.quantile(x, .99)), "max": float(x.max())}

result = {}
for name, buckets in values.items():
    result[name] = {}
    for ab, by_field in buckets.items():
        key = f"{ab[0]:.2f}-{ab[1]:.2f}"
        result[name][key] = {}
        for field, chunks in by_field.items():
            m = {k: summarize(v) for k, v in chunks.items()}
            m["relative_rms_error"] = m["err"]["rms"] / max(m["target"]["rms"], 1e-12)
            result[name][key][field] = m
out = cand / "matching_timebin_fit_comparison.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
for name, bins_result in result.items():
    print(name)
    for timebin, field_result in bins_result.items():
        print(timebin, " ".join(f"{f}={field_result[f]['relative_rms_error']:.3f}" for f in fields),
              "score-pred-p99/max=", round(field_result["score"]["pred"]["p99"], 3),
              round(field_result["score"]["pred"]["max"], 3))
print("saved", out)
