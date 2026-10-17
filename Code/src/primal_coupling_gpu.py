"""Direct positive semi-couplings via KL-projected gradient and backtracking."""
import math
import time
import torch

from .coupling import CouplingResult, OracleError


@torch.no_grad()
def kl_project(log_p, log_q, a, b, ratio_min, ratio_max, tolerance=1e-9, max_iterations=2000,
               corrections=None, return_state=False, check_every=5):
    """KL-Dykstra projection onto row(P), column(Q), and the ratio cone.

    The cone projection couples P and Q; two independent normalizations alone
    do not enforce the ratio domain. Corrections stay on the input device.
    """
    p, q = log_p.clone(), log_q.clone()
    if corrections is None:
        row_correction, col_correction = torch.zeros_like(a), torch.zeros_like(b)
        cone_p, cone_q = torch.zeros_like(p), torch.zeros_like(q)
    else:
        row_correction, col_correction, cone_p, cone_q = corrections
    loga, logb = a.log(), b.log()
    log_min, log_max = math.log(ratio_min), math.log(ratio_max)
    for iteration in range(max_iterations):
        z = p+row_correction[:, None]
        shift = loga-torch.logsumexp(z, dim=1)
        p, row_correction = z+shift[:, None], -shift
        z = q+col_correction[None, :]
        shift = logb-torch.logsumexp(z, dim=0)
        q, col_correction = z+shift[None, :], -shift
        u, v = p+cone_p, q+cone_q
        log_r = v-u
        bound = torch.where(log_r < log_min, torch.full_like(log_r, ratio_min), ratio_max)
        log_bound = torch.where(log_r < log_min, torch.full_like(log_r, log_min), log_max)
        outside = (log_r < log_min) | (log_r > log_max)
        projected_p = (u+bound*v-bound*log_bound)/(1+bound)
        p = torch.where(outside, projected_p, u)
        q = torch.where(outside, projected_p+log_bound, v)
        cone_p, cone_q = u-p, v-q
        if (iteration + 1) % check_every == 0 or iteration + 1 == max_iterations:
            error = torch.maximum((torch.logsumexp(p, 1).exp()-a).abs().sum()/a.sum(),
                                  (torch.logsumexp(q, 0).exp()-b).abs().sum()/b.sum())
            converged = float(error) <= tolerance
            if converged or iteration + 1 == max_iterations:
                state = (row_correction, col_correction, cone_p, cone_q)
                if return_state:
                    return p, q, iteration + 1, state, converged, float(error)
                if converged:
                    return p, q, iteration + 1
    raise OracleError("primal_projection_not_converged", dict(relative_l1=float(error), iterations=max_iterations))


def perspective_value_gradient(p, q, log_p, cost, slope, epsilon, log_reference=None):
    ratio = q/p
    if log_reference is not None:
        log_relative = q.log()-log_reference
        objective = (p*cost+epsilon*(q*(log_relative-1)+log_reference.exp())).sum()
        return objective, cost-ratio*slope, slope+epsilon*log_relative
    objective = (p*(cost+epsilon*(log_p-1))).sum()
    return objective, cost-ratio*slope+epsilon*log_p, slope


@torch.no_grad()
def solve_primal_gpu(a, b, oracle, epsilon=.0625, ratio_min=.1, ratio_max=10.,
                     marginal_tolerance=.002, max_iterations=10000,
                     stationarity_tolerance=1e-4, projection_tolerance=None,
                     callback=None, device="cuda", entropy_side="source",
                     initial_P=None, initial_Q=None, initial_log_P=None, initial_log_Q=None,
                     initial_step=1.0, allow_nonmonotone_fallback=False,
                     nonmonotone_min_step_fraction=2**-12):
    started = time.perf_counter()
    a = torch.as_tensor(a, dtype=torch.float64, device=device)
    b = torch.as_tensor(b, dtype=torch.float64, device=device)
    if bool((a <= 0).any() | (b <= 0).any()) or epsilon <= 0:
        raise ValueError("positive weights and entropy required")
    if entropy_side not in {"source", "receiver", "receiver_conditional"}:
        raise ValueError(entropy_side)
    mass_a, mass_b = a.sum(), b.sum()
    log_reference = a.log()[:,None]+b.log()[None,:]-mass_b.log()
    def value(p, q, lp, lq, cost):
        if entropy_side == "receiver_conditional":
            log_conditional=lq-torch.logsumexp(lq,dim=1,keepdim=True)-b.log()[None,:]+mass_b.log()
            return (p*cost+epsilon*q*log_conditional).sum()
        if entropy_side == "receiver":
            return (p*cost+epsilon*(q*(lq-log_reference-1)+log_reference.exp())).sum()
        return (p*(cost+epsilon*(lp-1))).sum()
    mass_ratio = float(mass_b/mass_a)
    if not ratio_min < mass_ratio < ratio_max:
        raise OracleError("infeasible_ratio_range", dict(mass_ratio=mass_ratio))
    # Start from the independent coupling on a fresh solve.  A saved feasible
    # iterate can be supplied when continuing a run stopped only by its outer
    # iteration budget.
    if initial_log_P is not None or initial_log_Q is not None:
        if initial_log_P is None or initial_log_Q is None:
            raise ValueError("both initial log couplings must be supplied")
        log_p = torch.as_tensor(initial_log_P, dtype=torch.float64, device=device)
        log_q = torch.as_tensor(initial_log_Q, dtype=torch.float64, device=device)
        if log_p.shape != (a.numel(), b.numel()) or log_q.shape != log_p.shape:
            raise ValueError("initial log coupling has incompatible shape")
        if not bool(torch.isfinite(log_p).all() & torch.isfinite(log_q).all()):
            raise ValueError("initial log coupling must be finite")
    elif initial_P is None and initial_Q is None:
        log_p = a.log()[:, None]+b.log()[None, :]-mass_b.log()
        log_q = log_p+math.log(mass_ratio)
    else:
        if initial_P is None or initial_Q is None:
            raise ValueError("both initial couplings must be supplied")
        initial_P = torch.as_tensor(initial_P, dtype=torch.float64, device=device)
        initial_Q = torch.as_tensor(initial_Q, dtype=torch.float64, device=device)
        if initial_P.shape != (a.numel(), b.numel()) or initial_Q.shape != initial_P.shape:
            raise ValueError("initial coupling has incompatible shape")
        if bool((initial_P <= 0).any() | (initial_Q <= 0).any()):
            raise ValueError("initial coupling must be strictly positive")
        log_p, log_q = initial_P.log(), initial_Q.log()
    step = float(initial_step)
    history = []
    last_print = started
    status = "max_outer_iterations"
    # The KL projection is an inner subproblem.  Keep its relative marginal
    # residual at least 100x tighter than the requested outer tolerance, while
    # avoiding excessive Dykstra sweeps on large blocks.  Backtracking retries
    # with tighter tolerances if an approximate projection spoils descent.
    if projection_tolerance is None:
        # Preserve the historical behavior for configurations that do not
        # explicitly opt into an inner projection tolerance.
        projection_tolerance = min(1e-6, marginal_tolerance/100)
    projection_tolerance = float(projection_tolerance)
    if not math.isfinite(projection_tolerance) or projection_tolerance <= 0:
        raise ValueError("projection_tolerance must be finite and positive")
    allow_nonmonotone_fallback = bool(allow_nonmonotone_fallback)
    nonmonotone_min_step_fraction = float(nonmonotone_min_step_fraction)
    if (not math.isfinite(nonmonotone_min_step_fraction) or
            not 0 < nonmonotone_min_step_fraction <= 1):
        raise ValueError("nonmonotone_min_step_fraction must be in (0, 1]")
    # Keep the fallback's minimum step anchored to the step supplied at solver
    # entry.  Anchoring it to each iteration's trial step lets repeated
    # backtracking shrink the floor toward zero and accept numerically inert
    # updates as "monotone" because of the objective slack.
    minimum_fallback_step = (float(initial_step)*nonmonotone_min_step_fraction
                            if allow_nonmonotone_fallback else 0.0)
    for iteration in range(1, max_iterations+1):
        p, q = log_p.exp(), log_q.exp()
        # Ratio is computed from log weights to avoid 0/0 in negligible cells.
        ratio = (log_q-log_p).exp().clamp(ratio_min, ratio_max)
        cost, slope = oracle(ratio)[:2]
        objective = value(p, q, log_p, log_q, cost)
        if entropy_side == "receiver_conditional":
            gp=cost-ratio*slope
            gq=slope+epsilon*(log_q-torch.logsumexp(log_q,dim=1,keepdim=True)-b.log()[None,:]+mass_b.log())
        elif entropy_side == "receiver":
            gp, gq = cost-ratio*slope, slope+epsilon*(log_q-log_reference)
        else:
            gp, gq = cost-ratio*slope+epsilon*log_p, slope
        # Constants per constrained row/column vanish under KL projection.
        # Removing them improves numerical conditioning without changing a step.
        gp = gp-(p*gp).sum(1)[:, None]/a[:, None]
        gq = gq-(q*gq).sum(0)[None, :]/b[None, :]
        initial_trial_step = step*2 if iteration > 1 else step
        if allow_nonmonotone_fallback:
            initial_trial_step = max(initial_trial_step, minimum_fallback_step)
        accepted = False
        majorization_fallback = False
        small_increase_fallback = False
        nonmonotone_fallback = False
        total_backtracks = 0
        last_trial_detail = None
        meaningful_trials = []
        local_projection_tolerance = projection_tolerance
        for precision_attempt in range(3):
            trial_step = initial_trial_step
            max_backtracks = 12 if precision_attempt < 2 else 28
            for backtrack in range(max_backtracks):
                try:
                    next_log_p, next_log_q, projection_iterations = kl_project(
                        log_p-trial_step*gp, log_q-trial_step*gq, a, b, ratio_min, ratio_max,
                        local_projection_tolerance, check_every=10)
                except OracleError as exc:
                    if exc.status != "primal_projection_not_converged":
                        raise
                    trial_step *= .5
                    total_backtracks += 1
                    if backtrack >= 11:
                        break
                    continue
                next_p, next_q = next_log_p.exp(), next_log_q.exp()
                next_ratio = (next_log_q-next_log_p).exp().clamp(ratio_min, ratio_max)
                next_cost = oracle.value(next_ratio)
                next_objective = value(next_p, next_q, next_log_p, next_log_q, next_cost)
                linear = (gp*(next_p-p)+gq*(next_q-q)).sum()
                kl = (next_p*(next_log_p-log_p)-next_p+p+next_q*(next_log_q-log_q)-next_q+q).sum()
                # Standard relative-smoothness backtracking for mirror descent.
                slack = 1e-12*(1+objective.abs())
                majorization_margin = objective+linear+kl/trial_step+slack-next_objective
                monotonicity_margin = objective+slack-next_objective
                majorized = bool(next_objective <= objective+linear+kl/trial_step+slack)
                monotone = bool(next_objective <= objective+slack)
                relative_increase = (next_objective-objective)/torch.maximum(
                    torch.ones_like(next_objective), objective.abs()/(.5*(mass_a+mass_b)))
                small_increase = bool(relative_increase <= stationarity_tolerance**2)
                bounded_ascent = bool(relative_increase <= stationarity_tolerance)
                last_trial_detail = (trial_step, local_projection_tolerance, projection_iterations,
                                     next_objective-objective, majorization_margin,
                                     monotonicity_margin, majorized, monotone)
                if allow_nonmonotone_fallback and trial_step >= minimum_fallback_step:
                    meaningful_trials.append((float(trial_step), next_objective-objective,
                        relative_increase, local_projection_tolerance, majorized, monotone))
                # The second projection-precision pass is already a 10x tighter
                # feasibility solve.  If it yields a meaningful monotone step,
                # do not force another precision pass solely for majorization:
                # that last pass can fail to project even when this descent step
                # is sound.  The anchored step floor still rules out no-op steps.
                fallback_precision_ready = (precision_attempt >= 1 if allow_nonmonotone_fallback
                                            else precision_attempt == 2)
                # A successfully projected, majorized candidate with an
                # increase below tolerance**2 is already a negligible,
                # controlled ascent.  When the meaningful-step floor is met,
                # do not require a second projection pass: near active ratio
                # constraints that tighter Dykstra pass can fail even though
                # the current candidate satisfies the requested projection
                # tolerance.  Larger bounded ascents still require the tighter
                # fallback pass below.
                tiny_ascent_at_meaningful_step = (
                    allow_nonmonotone_fallback and majorized and small_increase and
                    trial_step >= minimum_fallback_step)
                small_increase_allowed = majorized and small_increase and (
                    fallback_precision_ready or tiny_ascent_at_meaningful_step)
                nonmonotone_allowed = (allow_nonmonotone_fallback and fallback_precision_ready and
                    not monotone and bounded_ascent and trial_step >= minimum_fallback_step)
                meaningful_step = (not allow_nonmonotone_fallback or
                                   trial_step >= minimum_fallback_step)
                can_accept = meaningful_step and (
                    (monotone and (majorized or fallback_precision_ready)) or
                    small_increase_allowed or nonmonotone_allowed)
                if bool(torch.isfinite(next_objective)) and can_accept:
                    accepted = True
                    majorization_fallback = not majorized
                    small_increase_fallback = not monotone
                    nonmonotone_fallback = nonmonotone_allowed
                    total_backtracks += backtrack
                    break
                trial_step *= .5
                total_backtracks += 1
                # If tiny steps are being rejected, the likely cause is
                # projection error rather than a bad descent direction.  Retry
                # at tighter projection accuracy before shrinking further.
                if precision_attempt < 2 and backtrack >= 11:
                    break
            if accepted:
                break
            local_projection_tolerance = max(local_projection_tolerance*.1, 1e-9)
        if not accepted:
            (last_step, last_tolerance, last_projection_iterations, objective_delta,
             majorization_margin, monotonicity_margin, last_majorized, last_monotone) = last_trial_detail
            last_trial_detail = dict(step_size=float(last_step), projection_tolerance=float(last_tolerance),
                projection_iterations=int(last_projection_iterations), objective_delta=float(objective_delta),
                majorization_margin=float(majorization_margin), monotonicity_margin=float(monotonicity_margin),
                majorized=last_majorized, monotone=last_monotone)
            details = dict(iteration=iteration, step_size=float(last_step),
                projection_tolerance=float(last_tolerance), minimum_fallback_step=float(minimum_fallback_step),
                last_trial=last_trial_detail)
            if meaningful_trials:
                values = torch.stack([torch.stack((delta, relative))
                                      for _, delta, relative, _, _, _ in meaningful_trials]).cpu().tolist()
                details["meaningful_trials"] = [dict(step_size=step, objective_delta=values[i][0],
                    relative_objective_change=values[i][1], projection_tolerance=tol,
                    majorized=majorized, monotone=monotone)
                    for i, (step, _, _, tol, majorized, monotone) in enumerate(meaningful_trials)]
            raise OracleError("primal_line_search_failed", details)
        step = trial_step
        mapping_absolute = ((next_p-p).abs().sum()/mass_a+(next_q-q).abs().sum()/mass_b)/step
        objective_scale = torch.maximum(torch.ones_like(next_objective),
                                        next_objective.abs()/(.5*(mass_a+mass_b)))
        # A projected-gradient residual has the units of the objective.  Use
        # a relative residual so that multiplying all action terms by a
        # constant does not change when the same tolerance is satisfied.
        mapping = mapping_absolute/objective_scale
        row_error = (next_p.sum(1)-a).abs().sum()/mass_a
        col_error = (next_q.sum(0)-b).abs().sum()/mass_b
        boundary_mass = (next_p*((next_ratio <= ratio_min*(1+1e-10)) |
                                (next_ratio >= ratio_max*(1-1e-10)))).sum()/mass_a
        # One device-to-host transfer for diagnostics and the stopping decision.
        # Preserve every-step checks and complete history without scalar syncs.
        (objective_value, row_value, col_value, mapping_value, mapping_absolute_value,
         objective_scale_value, boundary_value, objective_delta_value, relative_change_value) = torch.stack(
            (next_objective, row_error, col_error, mapping, mapping_absolute,
             objective_scale, boundary_mass, next_objective-objective, relative_increase)).cpu().tolist()
        log_p, log_q = next_log_p, next_log_q
        seconds = time.perf_counter()-started
        record = dict(iteration=iteration, seconds=seconds, algorithm="projected_mirror_descent",
                      entropy_side=entropy_side, objective=objective_value, row_relative_l1=row_value, column_relative_l1=col_value,
                      row_relative_tv=row_value/2, column_relative_tv=col_value/2,
                      projected_gradient_relative_l1=mapping_value,
                      projected_gradient_absolute_l1=mapping_absolute_value,
                      objective_scale=objective_scale_value, step_size=step,
                      line_search_backtracks=total_backtracks,
                      line_search_majorization_fallback=majorization_fallback,
                      line_search_small_increase_fallback=small_increase_fallback,
                      line_search_nonmonotone_fallback=nonmonotone_fallback,
                      line_search_nonmonotone_step_fraction=(step/initial_trial_step
                          if nonmonotone_fallback and initial_trial_step > 0 else None),
                      objective_change=objective_delta_value,
                      relative_objective_change=relative_change_value,
                      projection_iterations=projection_iterations,
                      projection_tolerance=local_projection_tolerance,
                      gap_is_certificate=False, boundary_P_mass_fraction=boundary_value)
        history.append(record)
        converged = max(row_value, col_value) < marginal_tolerance and mapping_value < stationarity_tolerance
        if callback and (converged or iteration == 1 or iteration % 100 == 0 or seconds-(last_print-started) >= 15):
            callback(record)
            last_print = time.perf_counter()
        if converged:
            status = "converged"
            break
    # No dual variables are solved in this primal method. Explicit NaNs avoid
    # passing diagnostic row/column offsets off as valid dual potentials.
    arrays = [v.cpu().numpy() for v in (log_p.exp(), log_q.exp(), (log_q-log_p).exp(), log_p,
              torch.full_like(a, float("nan")), torch.full_like(b, float("nan")))]
    return CouplingResult(*arrays, status, history, time.perf_counter()-started)
