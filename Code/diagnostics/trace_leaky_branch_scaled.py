import json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure, load_config, snapshots_for
from src.experiment import make_feedback

run = Path(r"../Results/experiment2/eb_noscale_leakyrelu_residual_branch_scaled_w512x8_a1225_sigma05_blocks8_fm30000_seed42").resolve()
cfg = load_config(run / "config.yaml")
device = configure(cfg)
truth = snapshots_for(cfg, full=True)
checkpoint = torch.load(run / "feedback.pt", map_location=device, weights_only=False)
model = make_feedback(cfg, device)
model.load_state_dict(checkpoint["model"])
model.eval()
n, ds, sigma = len(truth[0].x), 1 / cfg["generation"]["steps_per_unit_interval"], cfg["model_sigma"]
initial = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)

def stat_fields(vr, vs, growth, score, dt_ds, scale):
    pieces = {"vreg": dt_ds * vr,
              "vsing": math.pi * sigma * scale * vs,
              "score": 0.5 * math.pi * sigma * score,
              "growth": growth}
    norms = {k: v.reshape(n, -1).norm(dim=-1) if k != "growth" else v.reshape(n, -1).abs().max(-1).values
             for k, v in pieces.items()}
    return {k: float(v.max()) for k, v in norms.items()}

result = {}
with torch.no_grad():
    # Exact transformed-time Euler-Maruyama used by rollout().
    x = initial.clone()
    logw = torch.full((n,), -math.log(n), dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(342)
    rows, failure = [], None
    singular_scale = sigma if cfg["matching"].get("absorb_sigma_in_singular_velocity", False) else 1.0
    for k in range(len(cfg["model_times"]) - 1):
        for j in range(cfg["generation"]["steps_per_unit_interval"]):
            s = j * ds
            local = math.sin(0.5 * math.pi * s) ** 2
            local_next = math.sin(0.5 * math.pi * (s + ds)) ** 2
            delta_t = local_next - local
            dt_ds = 0.5 * math.pi * math.sin(math.pi * s)
            t = torch.full((n, 1), k + local, dtype=x.dtype, device=device)
            vr, vs, growth, score = model.matching_fields(t, x)
            values = stat_fields(vr, vs, growth, score, dt_ds, singular_scale)
            x_abs = float(x.abs().max())
            xnorm_max = float(x.norm(dim=-1).max())
            logw_abs = float(logw.abs().max())
            if j % 20 == 0 or max(values["vreg"], values["vsing"], values["score"]) > 50 or values["growth"] > 2 or x_abs > 100 or logw_abs > 8:
                rows.append({"interval": k, "step": j, "time": k + local, "x_abs": x_abs,
                             "xnorm_max": xnorm_max, "logw_abs": logw_abs, **values})
            singular_drift_scaled = singular_scale * vs + 0.5 * sigma * score
            drift_s = dt_ds * vr + math.pi * singular_drift_scaled
            x = x + drift_s * ds
            brownian = torch.randn(x.shape, dtype=x.dtype, device=device, generator=generator) * math.sqrt(delta_t)
            x = x + sigma * brownian
            logw = logw + (dt_ds * growth.squeeze(-1).double()) * ds
            if not torch.isfinite(x).all() or not torch.isfinite(logw).all():
                failure = {"interval": k, "step": j, "time": k + local_next,
                           "x_finite": bool(torch.isfinite(x).all()),
                           "logw_finite": bool(torch.isfinite(logw).all()),
                           "x_abs": float(x.abs().max()), "logw_abs": float(logw.abs().max()),
                           "fields_before_step": values}
                break
        if failure:
            break
    result["SDE"] = {"failure": failure, "rows": rows}

    # Exact transformed-time RK4 ODE used by rollout().
    x = initial.clone()
    logw = torch.full((n,), -math.log(n), dtype=torch.float64, device=device)
    rows, failure = [], None
    def rhs(s, z, lw, k):
        local = math.sin(0.5 * math.pi * s) ** 2
        dt_ds = 0.5 * math.pi * math.sin(math.pi * s)
        t = torch.full((n, 1), k + local, dtype=z.dtype, device=device)
        vr, vs, growth, _ = model.matching_fields(t, z)
        vals = stat_fields(vr, vs, growth, torch.zeros_like(vr), dt_ds, singular_scale)
        dx = dt_ds * vr + math.pi * singular_scale * vs
        dw = dt_ds * growth.squeeze(-1).double()
        return dx, dw, vals
    for k in range(len(cfg["model_times"]) - 1):
        for j in range(cfg["generation"]["steps_per_unit_interval"]):
            s = j * ds
            k1x, k1w, v1 = rhs(s, x, logw, k)
            k2x, k2w, v2 = rhs(s + .5*ds, x + .5*ds*k1x, logw + .5*ds*k1w, k)
            k3x, k3w, v3 = rhs(s + .5*ds, x + .5*ds*k2x, logw + .5*ds*k2w, k)
            k4x, k4w, v4 = rhs(s + ds, x + ds*k3x, logw + ds*k3w, k)
            vals = {f: max(v1[f], v2[f], v3[f], v4[f]) for f in v1}
            x = x + (ds / 6) * (k1x + 2*k2x + 2*k3x + k4x)
            logw = logw + (ds / 6) * (k1w + 2*k2w + 2*k3w + k4w)
            x_abs, xnorm_max, logw_abs = float(x.abs().max()), float(x.norm(dim=-1).max()), float(logw.abs().max())
            if j % 20 == 0 or max(vals["vreg"], vals["vsing"]) > 50 or vals["growth"] > 2 or x_abs > 100 or logw_abs > 8:
                rows.append({"interval": k, "step": j, "time": k + math.sin(0.5*math.pi*(s+ds))**2,
                             "x_abs": x_abs, "xnorm_max": xnorm_max, "logw_abs": logw_abs, **vals})
            if not torch.isfinite(x).all() or not torch.isfinite(logw).all():
                failure = {"interval": k, "step": j, "time": k + math.sin(0.5*math.pi*(s+ds))**2,
                           "x_finite": bool(torch.isfinite(x).all()),
                           "logw_finite": bool(torch.isfinite(logw).all()),
                           "x_abs": float(x.abs().max()), "logw_abs": float(logw.abs().max()),
                           "fields_at_rk_stages": vals}
                break
        if failure:
            break
    result["ODE"] = {"failure": failure, "rows": rows}

out = run / "evaluation_all_initial_points" / "branch_scaled_failure_trace_seed342.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
for mode in result:
    print(mode, json.dumps({"failure": result[mode]["failure"], "last_rows": result[mode]["rows"][-8:]}, indent=2), flush=True)
print("saved", out)
