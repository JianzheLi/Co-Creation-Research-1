import math
import numpy as np
import torch


def _failure_tensor_summary(name, value):
    """Compact numerical context for an already-detected rollout failure."""
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    max_abs_finite = (float(finite_values.abs().max().item()) if finite_values.numel() else None)
    return (f"{name}(max_abs_finite={max_abs_finite}, "
            f"nonfinite={int((~finite).sum().item())}/{tensor.numel()})")


def _diagnostic_tensor_stats(value):
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    return {
        "max_abs_finite": (float(finite_values.abs().max().item()) if finite_values.numel() else None),
        "nonfinite_count": int((~finite).sum().item()),
    }


@torch.no_grad()
def rollout(model, first_snapshot, times=(0, 1, 2, 3, 4), sigma=0.25,
            n_particles=2000, steps_per_unit=200, seed=42, endpoint_guard=1e-5, initial_x=None,
            normal_increments=None, action_alpha=None, trace_indices=None, stochastic=True,
            diagnostic_callback=None, capture_times=None):
    """One continuous rollout; stochastic=False integrates the learned flow ODE."""
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    if initial_x is None:
        first = torch.as_tensor(first_snapshot, device=device, dtype=dtype)
        ids = torch.randint(len(first), (n_particles,), device=device, generator=generator)
        x = first[ids].clone()
    else:
        x = torch.as_tensor(initial_x, device=device, dtype=dtype).clone()
        if len(x) != n_particles:
            raise ValueError("initial particle count mismatch")
    initial_positions = x
    if times[0] != 0 or any(b - a != 1 for a, b in zip(times[:-1], times[1:])):
        raise ValueError("rollout expects the specified unit time intervals")
    if not 0 < endpoint_guard < 0.5:
        raise ValueError("endpoint_guard must be in (0,0.5)")
    logw = torch.full((n_particles,), -math.log(n_particles), dtype=torch.float64, device=device)
    if trace_indices is not None:
        trace_indices = torch.as_tensor(trace_indices, dtype=torch.long, device=device)
        if trace_indices.ndim != 1 or bool(((trace_indices < 0) | (trace_indices >= n_particles)).any()):
            raise ValueError("trace indices must identify existing particles")
        trace_positions = [x[trace_indices].clone()]
    if action_alpha is not None:
        if action_alpha < 0:
            raise ValueError("action_alpha must be nonnegative")
        interval_kinetic = torch.zeros((len(times) - 1, n_particles), dtype=torch.float64, device=device)
        interval_growth = torch.zeros_like(interval_kinetic)
    positions, log_weights = [x.cpu().numpy().copy()], [logw.cpu().numpy().copy()]
    requested_capture_times = (None if capture_times is None else
                               tuple(float(value) for value in capture_times))
    if requested_capture_times is not None:
        if any(not math.isfinite(value) or value < times[0] or value > times[-1]
               for value in requested_capture_times):
            raise ValueError("capture_times must lie within the rollout time range")
        captures = {}
        capture_tolerance = 1e-10

        def maybe_capture(time_value):
            for requested in requested_capture_times:
                if abs(float(time_value) - requested) <= capture_tolerance:
                    captures[requested] = (x.detach().cpu().numpy().copy(),
                                           logw.detach().cpu().numpy().copy())

        maybe_capture(times[0])
    use_endpoint_clock = (action_alpha is None and
                          getattr(model, "field_parameterization", "raw") == "endpoint_regularized")
    use_transformed_ode = use_endpoint_clock and not stochastic
    use_transformed_sde = use_endpoint_clock and stochastic
    singular_velocity_scale = (
        float(getattr(model, "sigma", sigma))
        if getattr(model, "absorb_sigma_in_singular_velocity", False)
        else 1.0
    )
    transformed_trace_times = ([float(times[0])] if trace_indices is not None and use_endpoint_clock
                               else None)
    dt = 1 / steps_per_unit
    step = 0
    if not stochastic and normal_increments is not None:
        raise ValueError("normal increments are only valid for stochastic rollout")
    for k in range(len(times) - 1):
        endpoint_ode = use_transformed_ode
        endpoint_sde = use_transformed_sde
        if endpoint_ode:
            # The decoded probability-flow field contains B(t,x)/sqrt(t(1-t)).
            # This singularity is integrable, but a left-endpoint Euler step at
            # t=endpoint_guard overestimates its first-step integral by a factor
            # proportional to sqrt(dt / endpoint_guard). Reparameterize each interval by
            # t=sin^2(pi*s/2), for which dt/ds=pi*sqrt(t(1-t)); the transformed
            # ODE is regular and exactly equivalent to the decoded physical ODE.
            ds = dt

            def transformed_rhs(s, x_value, logw_value):
                local = math.sin(0.5 * math.pi * s) ** 2
                dt_ds = 0.5 * math.pi * math.sin(math.pi * s)
                t_value = torch.full((n_particles, 1), times[k] + local,
                                     dtype=dtype, device=device)
                v_regular, v_singular_scaled, growth, _ = model.matching_fields(t_value, x_value)
                dx_ds = (dt_ds * v_regular
                         + math.pi * singular_velocity_scale * v_singular_scaled)
                dlogw_ds = dt_ds * growth.squeeze(-1).double()
                return dx_ds, dlogw_ds

            for local_step in range(steps_per_unit):
                s = local_step * ds
                k1_x, k1_w = transformed_rhs(s, x, logw)
                k2_x, k2_w = transformed_rhs(s + 0.5 * ds,
                                             x + 0.5 * ds * k1_x,
                                             logw + 0.5 * ds * k1_w)
                k3_x, k3_w = transformed_rhs(s + 0.5 * ds,
                                             x + 0.5 * ds * k2_x,
                                             logw + 0.5 * ds * k2_w)
                k4_x, k4_w = transformed_rhs(s + ds,
                                             x + ds * k3_x,
                                             logw + ds * k3_w)
                x = x + (ds / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
                logw = logw + (ds / 6.0) * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)
                if trace_indices is not None:
                    trace_positions.append(x[trace_indices].clone())
                    s_next = (local_step + 1) * ds
                    local_next = math.sin(0.5 * math.pi * s_next) ** 2
                    transformed_trace_times.append(float(times[k] + local_next))
                if requested_capture_times is not None:
                    maybe_capture(times[k] + local_next)
                step += 1
                if not torch.isfinite(x).all() or not torch.isfinite(logw).all():
                    raise FloatingPointError(f"nonfinite weighted rollout at t={times[k + 1]}")
            positions.append(x.cpu().numpy().copy())
            log_weights.append(logw.cpu().numpy().copy())
            continue

        if endpoint_sde:
            # Apply the same deterministic time change to the SDE.  If
            # dX = (v_reg + C/sqrt(t(1-t)))dt + sigma dW_t, then for
            # t=sin^2(pi*s/2),
            # dX = (dt/ds*v_reg + pi*C)ds + sigma*sqrt(dt/ds)dW_s.
            # We use the exact physical-time increment between adjacent s
            # nodes for the Brownian variance, so the noise law is unchanged.
            ds = dt
            for local_step in range(steps_per_unit):
                s = local_step * ds
                s_next = (local_step + 1) * ds
                local = math.sin(0.5 * math.pi * s) ** 2
                local_next = math.sin(0.5 * math.pi * s_next) ** 2
                delta_t = local_next - local
                dt_ds = 0.5 * math.pi * math.sin(math.pi * s)
                t_input = torch.full((n_particles, 1), times[k] + local,
                                     dtype=dtype, device=device)
                v_regular, v_singular_scaled, growth, score_scaled = model.matching_fields(t_input, x)
                x_before, logw_before = x, logw
                singular_drift_scaled = (singular_velocity_scale * v_singular_scaled
                                         + 0.5 * sigma * score_scaled)
                drift_s = dt_ds * v_regular + math.pi * singular_drift_scaled
                x = x + drift_s * ds
                if normal_increments is None:
                    brownian = (torch.randn(x.shape, dtype=dtype, device=device, generator=generator)
                                * math.sqrt(delta_t))
                else:
                    # `normal_increments` keeps its existing uniform-physical-
                    # time convention; normalize it and rescale to this step's
                    # actual Brownian variance.
                    brownian = (torch.as_tensor(normal_increments[step], device=device, dtype=dtype)
                                * math.sqrt(delta_t / dt))
                x = x + sigma * brownian
                logw = logw + (dt_ds * growth.squeeze(-1).double()) * ds
                if trace_indices is not None:
                    trace_positions.append(x[trace_indices].clone())
                    transformed_trace_times.append(float(times[k] + local_next))
                if requested_capture_times is not None:
                    maybe_capture(times[k] + local_next)
                step += 1
                if diagnostic_callback is not None:
                    extreme_particle = int(v_singular_scaled.detach().abs().amax(dim=-1).argmax().item())
                    score_extreme_particle = int(score_scaled.detach().abs().amax(dim=-1).argmax().item())
                    diagnostic_callback({
                        "step": step,
                        "interval": k,
                        "time": float(times[k] + local_next),
                        "x_before": _diagnostic_tensor_stats(x_before),
                        "x_after": _diagnostic_tensor_stats(x),
                        "logw_before": _diagnostic_tensor_stats(logw_before),
                        "logw_after": _diagnostic_tensor_stats(logw),
                        "v_regular": _diagnostic_tensor_stats(v_regular),
                        "v_singular_scaled": _diagnostic_tensor_stats(v_singular_scaled),
                        "growth": _diagnostic_tensor_stats(growth),
                        "score_scaled": _diagnostic_tensor_stats(score_scaled),
                        "drift_s": _diagnostic_tensor_stats(drift_s),
                        "brownian": _diagnostic_tensor_stats(brownian),
                        "singular_velocity_extreme_particle": {
                            "index": extreme_particle,
                            "initial_position": initial_positions[extreme_particle].detach().cpu().tolist(),
                            "position_before": x_before[extreme_particle].detach().cpu().tolist(),
                            "v_regular": v_regular[extreme_particle].detach().cpu().tolist(),
                            "v_singular_scaled": v_singular_scaled[extreme_particle].detach().cpu().tolist(),
                            "score_scaled": score_scaled[extreme_particle].detach().cpu().tolist(),
                            "growth": growth[extreme_particle].detach().cpu().tolist(),
                            "drift_s": drift_s[extreme_particle].detach().cpu().tolist(),
                            "brownian": brownian[extreme_particle].detach().cpu().tolist(),
                        },
                        "score_extreme_particle": {
                            "index": score_extreme_particle,
                            "initial_position": initial_positions[score_extreme_particle].detach().cpu().tolist(),
                            "position_before": x_before[score_extreme_particle].detach().cpu().tolist(),
                            "v_regular": v_regular[score_extreme_particle].detach().cpu().tolist(),
                            "v_singular_scaled": v_singular_scaled[score_extreme_particle].detach().cpu().tolist(),
                            "score_scaled": score_scaled[score_extreme_particle].detach().cpu().tolist(),
                            "growth": growth[score_extreme_particle].detach().cpu().tolist(),
                            "drift_s": drift_s[score_extreme_particle].detach().cpu().tolist(),
                            "brownian": brownian[score_extreme_particle].detach().cpu().tolist(),
                        },
                    })
                if not torch.isfinite(x).all() or not torch.isfinite(logw).all():
                    context = "; ".join((
                        _failure_tensor_summary("x_before", x_before),
                        _failure_tensor_summary("logw_before", logw_before),
                        _failure_tensor_summary("x", x),
                        _failure_tensor_summary("logw", logw),
                        _failure_tensor_summary("v_regular", v_regular),
                        _failure_tensor_summary("v_singular_scaled", v_singular_scaled),
                        _failure_tensor_summary("growth", growth),
                        _failure_tensor_summary("score_scaled", score_scaled),
                        _failure_tensor_summary("drift_s", drift_s),
                        _failure_tensor_summary("brownian", brownian),
                    ))
                    raise FloatingPointError(
                        f"nonfinite weighted SDE rollout at t={times[k] + local_next}; {context}")
            positions.append(x.cpu().numpy().copy())
            log_weights.append(logw.cpu().numpy().copy())
            continue

        for local_step in range(steps_per_unit):
            t = times[k] + local_step * dt
            # Record the small one-sided protection at observation boundaries.
            local = min(max(t - times[k], endpoint_guard), 1 - endpoint_guard)
            t_input = torch.full((n_particles, 1), times[k] + local, dtype=dtype, device=device)
            if getattr(model, "field_parameterization", "raw") == "endpoint_regularized":
                v_regular, v_singular_scaled, growth, score_scaled = model.matching_fields(t_input, x)
                tau = local * (1.0 - local)
                sqrt_tau = math.sqrt(max(tau, endpoint_guard * (1.0 - endpoint_guard)))
                # Decode the singular bridge component only after multiplying
                # the SDE correction in its regularized coordinates.
                if stochastic:
                    drift = (v_regular + singular_velocity_scale * v_singular_scaled / sqrt_tau
                             + 0.5 * sigma * score_scaled / sqrt_tau)
                else:
                    drift = v_regular + singular_velocity_scale * v_singular_scaled / sqrt_tau
            else:
                v, growth, score = model(t_input, x)
                # V is the probability-flow velocity.  The corresponding SDE drift
                # is U = V + sigma^2/2 * score; the ODE uses V alone.
                drift = v + (0.5 * sigma ** 2 * score if stochastic else 0)
            if action_alpha is not None:
                particle_mass = logw.exp()
                kinetic_rate = 0.5 * drift.double().square().sum(-1)
                growth_rate = 0.5 * action_alpha * growth.squeeze(-1).double().square()
                interval_kinetic[k] += particle_mass * kinetic_rate * dt
                interval_growth[k] += particle_mass * growth_rate * dt
            if not stochastic:
                if normal_increments is not None:
                    raise ValueError("normal increments are only valid for stochastic rollout")
                noise = torch.zeros_like(x)
            elif normal_increments is None:
                noise = torch.randn(x.shape, dtype=dtype, device=device, generator=generator) * math.sqrt(dt)
            else:
                noise = torch.as_tensor(normal_increments[step], device=device, dtype=dtype)
            x = x + drift * dt + sigma * noise
            logw = logw + growth.squeeze(-1).double() * dt
            if trace_indices is not None:
                trace_positions.append(x[trace_indices].clone())
            if requested_capture_times is not None:
                maybe_capture(t + dt)
            step += 1
            if not torch.isfinite(x).all() or not torch.isfinite(logw).all():
                raise FloatingPointError(f"nonfinite weighted rollout at t={t + dt}")
        positions.append(x.cpu().numpy().copy())
        log_weights.append(logw.cpu().numpy().copy())
    result = dict(positions=np.asarray(positions), log_weights=np.asarray(log_weights), times=np.asarray(times),
                  sigma=sigma, seed=seed, steps_per_unit=steps_per_unit, endpoint_guard=endpoint_guard)
    if requested_capture_times is not None:
        missing = [value for value in requested_capture_times if value not in captures]
        if missing:
            raise ValueError(
                "capture_times must align with integration steps; "
                f"not captured: {missing}. Choose times on the 1/steps_per_unit grid."
            )
        result.update(
            capture_times=np.asarray(requested_capture_times, dtype=np.float64),
            capture_positions=np.stack([captures[value][0] for value in requested_capture_times]),
            capture_log_weights=np.stack([captures[value][1] for value in requested_capture_times]),
        )
    if trace_indices is not None:
        result.update(trace_positions=torch.stack(trace_positions).cpu().numpy(),
                      trace_indices=trace_indices.cpu().numpy(),
                      trace_times=(np.asarray(transformed_trace_times) if transformed_trace_times is not None
                                   else np.arange(step + 1) * dt))
    if action_alpha is not None:
        kinetic_particles = interval_kinetic.sum(0) * n_particles
        growth_particles = interval_growth.sum(0) * n_particles
        total_particles = kinetic_particles + growth_particles
        kinetic_by_interval = interval_kinetic.sum(-1)
        growth_by_interval = interval_growth.sum(-1)
        intervals = []
        for k, (kinetic, growth) in enumerate(zip(kinetic_by_interval.cpu().tolist(),
                                                   growth_by_interval.cpu().tolist())):
            intervals.append(dict(start=float(times[k]), end=float(times[k + 1]),
                                  kinetic=float(kinetic), growth=float(growth), total=float(kinetic + growth)))
        kinetic_total, growth_total = float(kinetic_by_interval.sum().item()), float(growth_by_interval.sum().item())
        result["action"] = dict(kinetic=kinetic_total, growth=growth_total,
                                total=kinetic_total + growth_total,
                                mc_standard_error=float(total_particles.std(unbiased=True).item() / math.sqrt(n_particles)),
                                kinetic_mc_standard_error=float(kinetic_particles.std(unbiased=True).item() / math.sqrt(n_particles)),
                                growth_mc_standard_error=float(growth_particles.std(unbiased=True).item() / math.sqrt(n_particles)),
                                intervals=intervals)
    return result
