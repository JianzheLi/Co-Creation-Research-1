import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure, load_config, snapshots_for, load_path
from src.experiment import make_feedback
from src.matching import CouplingSampler, sample_training_batch

run = Path(r"../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42").resolve()
cfg = load_config(run / "config.yaml")
dev = configure(cfg)
truth = snapshots_for(cfg, full=True)
cp = torch.load(run / "feedback.pt", map_location=dev, weights_only=False)
model = make_feedback(cfg, dev)
model.load_state_dict(cp["model"])
model.eval()
idx = torch.linspace(0, len(truth[0].x) - 1, 512, device=dev).long()
x = torch.as_tensor(truth[0].x, dtype=torch.float32, device=dev)[idx].clone()
gen = torch.Generator(device=dev).manual_seed(342)
capture = None
with torch.no_grad():
    for k in range(2):
        steps_here = 200 if k == 0 else 126
        for j in range(steps_here):
            s = j / 200
            sn = (j + 1) / 200
            u = math.sin(0.5 * math.pi * s) ** 2
            un = math.sin(0.5 * math.pi * sn) ** 2
            delta_t = un - u
            dt_ds = 0.5 * math.pi * math.sin(math.pi * s)
            t = torch.full((len(x), 1), k + u, dtype=x.dtype, device=dev)
            vr, vs, growth, score = model.matching_fields(t, x)
            if k == 1 and j == 120:
                ids = score.norm(dim=-1).topk(8).indices
                capture = {"time": k + u, "step": j, "ids": ids,
                           "x": x[ids].clone(), "score_norm": score[ids].norm(dim=-1),
                           "vreg_norm": vr[ids].norm(dim=-1), "growth": growth[ids].squeeze(-1)}
            drift_s = dt_ds * vr + math.pi * cfg["model_sigma"] * vs + 0.5 * math.pi * cfg["model_sigma"] * score
            x = x + drift_s * (1 / 200) + cfg["model_sigma"] * torch.randn(
                x.shape, dtype=x.dtype, device=dev, generator=gen) * math.sqrt(delta_t)

sampler = CouplingSampler(run, cfg, dev)
path = load_path(run, cfg, dev)
g = torch.Generator(device=dev).manual_seed(20261016)
samples, target_scores, pred_scores = [], [], []
for _ in range(100):
    t, targets, _ = sample_training_batch(sampler, path, 2048, dev, generator=g)
    mask = (t.floor().squeeze(-1) == 1) & ((t - t.floor()).squeeze(-1) >= 0.63) & ((t - t.floor()).squeeze(-1) <= 0.68)
    if mask.any():
        xx = targets["x"][mask]
        tt = t[mask]
        with torch.no_grad():
            ss = model.matching_fields(tt, xx)[3]
        samples.append(xx)
        target_scores.append(targets["target_s_scaled"][mask].norm(dim=-1))
        pred_scores.append(ss.norm(dim=-1))
train_x = torch.cat(samples)
target_norm = torch.cat(target_scores)
pred_norm = torch.cat(pred_scores)
std = train_x.std(dim=0).clamp_min(1e-5)
query = capture["x"]
dist = torch.cdist(query / std, train_x / std)
near_ix = dist.argmin(dim=1)
near_pred = pred_norm[near_ix]
query_pred = capture["score_norm"]

# Estimate the local Jacobian spectrum of the transformed SDE drift at the
# highest-score rollout state, with no stochastic noise in this local probe.
q = query[:1].detach().requires_grad_(True)
tq = torch.full((1, 1), capture["time"], device=dev, dtype=q.dtype)
s_local = capture["time"] - 1.0
s = (2.0 / math.pi) * math.asin(math.sqrt(s_local))
dt_ds = 0.5 * math.pi * math.sin(math.pi * s)
def drift(z):
    vr, vs, growth, score = model.matching_fields(tq, z)
    return dt_ds * vr + math.pi * cfg["model_sigma"] * vs + 0.5 * math.pi * cfg["model_sigma"] * score
jac = torch.autograd.functional.jacobian(drift, q, vectorize=True).reshape(cfg["dimension"], cfg["dimension"])
sym_eigs = torch.linalg.eigvalsh(0.5 * (jac + jac.T))

def summary(v):
    v = v.detach().double()
    return {"n": int(v.numel()), "median": float(v.median()), "p95": float(torch.quantile(v, .95)),
            "p99": float(torch.quantile(v, .99)), "max": float(v.max())}

out = {
    "rollout_time": capture["time"],
    "rollout_states": [{"particle": int(capture["ids"][i]), "xnorm": float(query[i].norm()),
                        "xabs": float(query[i].abs().max()), "score_scaled_norm": float(query_pred[i]),
                        "vreg_norm": float(capture["vreg_norm"][i]), "growth": float(capture["growth"][i]),
                        "nearest_train_standardized_distance": float(dist[i].min()),
                        "nearest_train_score_prediction": float(near_pred[i])} for i in range(len(query))],
    "training_bin_samples": int(train_x.shape[0]),
    "training_bin_xnorm": summary(train_x.norm(dim=-1)),
    "training_bin_max_abs_x": summary(train_x.abs().max(dim=-1).values),
    "training_bin_score_target_norm": summary(target_norm),
    "training_bin_score_prediction_norm": summary(pred_norm),
    "jacobian": {"spectral_norm": float(torch.linalg.matrix_norm(jac, ord=2)),
                 "symmetric_max_eigenvalue": float(sym_eigs.max()),
                 "symmetric_min_eigenvalue": float(sym_eigs.min())},
}
out_path = run / "evaluation_all_initial_points" / "sde_support_diagnostic.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
