import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure, load_config, snapshots_for
from src.experiment import make_feedback

base = Path(r"../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42").resolve()
cand = Path(r"../Results/experiment2/eb_noscale_leakyrelu_prenorm_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42").resolve()
c0 = load_config(base / "config.yaml")
device = configure(c0)
truth = snapshots_for(c0, full=True)
models = []
for run in (base, cand):
    c = load_config(run / "config.yaml")
    cp = torch.load(run / "feedback.pt", map_location=device, weights_only=False)
    model = make_feedback(c, device)
    model.load_state_dict(cp["model"])
    model.eval()
    models.append(model)

# Recreate the exact 512-particle seed-342 trajectory used by the original
# baseline trace and retain its highest-score state at t=1.6545.
idx = torch.linspace(0, len(truth[0].x) - 1, 512, device=device).long()
x = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)[idx].clone()
generator = torch.Generator(device=device).manual_seed(342)
ds, sigma, target_step = 1 / 200, float(c0["model_sigma"]), 120
captured = None
with torch.no_grad():
    for interval in range(2):
        nsteps = 200 if interval == 0 else target_step + 1
        for j in range(nsteps):
            s = j * ds
            sn = (j + 1) * ds
            local = math.sin(.5 * math.pi * s) ** 2
            local_next = math.sin(.5 * math.pi * sn) ** 2
            delta_t = local_next - local
            dt_ds = .5 * math.pi * math.sin(math.pi * s)
            t = torch.full((len(x), 1), interval + local, dtype=x.dtype, device=device)
            vr, vs, growth, score = models[0].matching_fields(t, x)
            if interval == 1 and j == target_step:
                ix = int(score.norm(dim=-1).argmax())
                captured = {"particle": ix, "time": interval + local, "x": x[ix].detach().clone(),
                            "baseline_score_norm": float(score[ix].norm())}
            singular_scaled = models[0].sigma * vs + .5 * sigma * score
            drift = dt_ds * vr + math.pi * singular_scaled
            x = x + drift * ds
            brownian = torch.randn(x.shape, dtype=x.dtype, device=device, generator=generator) * math.sqrt(delta_t)
            x = x + sigma * brownian

def diagnose(model):
    q = captured["x"].reshape(1, -1).detach().requires_grad_(True)
    t = torch.full((1, 1), captured["time"], device=device, dtype=q.dtype)
    s_local = captured["time"] - 1.0
    s = (2 / math.pi) * math.asin(math.sqrt(s_local))
    dt_ds = .5 * math.pi * math.sin(math.pi * s)
    with torch.no_grad():
        vr, vs, growth, score = model.matching_fields(t, q)
        norms = {"v_regular": float(vr.norm()), "v_singular_scaled": float(vs.norm()),
                 "score_scaled": float(score.norm()), "growth": float(growth.abs().max())}
    def drift(z):
        a, b, _, c = model.matching_fields(t, z)
        singular = model.sigma * b + .5 * sigma * c
        return dt_ds * a + math.pi * singular
    jac = torch.autograd.functional.jacobian(drift, q, vectorize=True).reshape(c0["dimension"], c0["dimension"])
    eigs = torch.linalg.eigvalsh(.5 * (jac + jac.T))
    norms.update({"drift_jacobian_spectral_norm": float(torch.linalg.matrix_norm(jac, ord=2)),
                  "drift_jacobian_max_symmetric_eigenvalue": float(eigs.max()),
                  "drift_jacobian_min_symmetric_eigenvalue": float(eigs.min())})
    if hasattr(model.score, "output_norm") and not isinstance(model.score.output_norm, torch.nn.Identity):
        head = model.score.net[-1]
        norms["score_global_output_norm_bound"] = float(torch.linalg.matrix_norm(head.weight, ord=2) * math.sqrt(head.in_features) + head.bias.norm())
    return norms

out = {"time": captured["time"], "particle": captured["particle"],
       "xnorm": float(captured["x"].norm()), "xabs": float(captured["x"].abs().max()),
       "baseline_replay_score_norm": captured["baseline_score_norm"],
       "silu_residual": diagnose(models[0]), "leaky_prenorm_residual": diagnose(models[1])}
path = cand / "evaluation_all_initial_points" / "same_state_field_jacobian_comparison.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
print("saved", path)
