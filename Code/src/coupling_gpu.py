"""GPU implementation of the same row/column dual updates as coupling.py."""
import time
import numpy as np
import torch
from .coupling import CouplingResult, OracleError


class TorchHermiteCost:
    """Validated Hermite coefficients stay on device throughout the dual solve."""
    def __init__(self, table, device="cuda"):
        self.device = torch.device(device)
        self.grid = torch.as_tensor(table.grid, dtype=torch.float64, device=self.device)
        self.derivatives = torch.as_tensor(table.derivatives, dtype=torch.float64, device=self.device).contiguous()
        self.coefficients = torch.as_tensor(table.coefficients, dtype=torch.float64, device=self.device).contiguous()
        self.shape = table.shape
        self.flat_coefficients = self.coefficients.reshape(-1, len(table.grid) - 1, 4)
        self.pair_ids = torch.arange(np.prod(self.shape), device=self.device)

    @classmethod
    def from_tensors(cls, grid, costs, derivatives, curvature_tolerance=1e-5):
        obj = cls.__new__(cls)
        obj.device = costs.device
        obj.grid = torch.as_tensor(grid, dtype=torch.float64, device=obj.device)
        obj.costs, obj.derivatives = costs, derivatives.contiguous()
        obj.shape = tuple(costs.shape[:-1])
        h = torch.diff(obj.grid)
        secant = torch.diff(costs, dim=-1) / h
        c0, c1 = costs[..., :-1], derivatives[..., :-1]
        c2 = (3 * secant - 2 * c1 - derivatives[..., 1:]) / h
        c3 = (c1 + derivatives[..., 1:] - 2 * secant) / h.square()
        obj.min_curvature = float(torch.minimum(2 * c2, 2 * c2 + 6 * c3 * h).min())
        obj.min_slope_increment = float(torch.diff(derivatives, dim=-1).min())
        if not bool(torch.isfinite(costs).all() & torch.isfinite(derivatives).all()):
            raise OracleError("cost_oracle_nonfinite", {})
        if obj.min_curvature < -curvature_tolerance or obj.min_slope_increment < 0:
            raise OracleError("cost_oracle_nonconvex", dict(min_curvature=obj.min_curvature,
                              min_slope_increment=obj.min_slope_increment))
        if obj.min_curvature < 0:
            raise OracleError("cost_table_needs_refinement", dict(min_curvature=obj.min_curvature))
        obj.coefficients = torch.stack((c0, c1, c2, c3), dim=-1)
        obj.flat_coefficients = obj.coefficients.reshape(-1, len(grid) - 1, 4)
        obj.pair_ids = torch.arange(np.prod(obj.shape), device=obj.device)
        return obj

    def _gather(self, index):
        return self.flat_coefficients[self.pair_ids, index.reshape(-1)].reshape(*self.shape, 4)

    def __call__(self, r):
        r = torch.broadcast_to(torch.as_tensor(r, dtype=torch.float64, device=self.device), self.shape)
        index = (torch.searchsorted(self.grid, r.contiguous(), right=True) - 1).clamp(0, len(self.grid) - 2)
        c = self._gather(index)
        z = r - self.grid[index]
        value = c[..., 0] + z * (c[..., 1] + z * (c[..., 2] + z * c[..., 3]))
        derivative = c[..., 1] + z * (2 * c[..., 2] + 3 * z * c[..., 3])
        curvature = 2 * c[..., 2] + 6 * z * c[..., 3]
        return value, derivative, curvature

    def conjugate(self, lam):
        lam = torch.broadcast_to(lam, self.shape)
        low = lam <= self.derivatives[..., 0]
        high = lam >= self.derivatives[..., -1]
        index = (torch.searchsorted(self.derivatives, lam.unsqueeze(-1).contiguous(), right=True).squeeze(-1) - 1).clamp(0, len(self.grid) - 2)
        c = self._gather(index)
        h = self.grid[index + 1] - self.grid[index]
        target = lam - c[..., 1]
        lo, hi = torch.zeros_like(h), h
        # The rational quadratic root is only an initial guess within the
        # derivative bracket. Safeguarded Newton then verifies/refines it.
        safe_target = target.clamp_min(0)
        discriminant = (4 * c[..., 2].square() + 12 * c[..., 3] * safe_target).clamp_min(0)
        denominator = 2 * c[..., 2] + discriminant.sqrt()
        z = torch.where(denominator > 1e-300, 2 * safe_target / denominator.clamp_min(1e-300), h / 2)
        z = torch.minimum(z.clamp_min(0), h)
        active = ~(low | high)
        for _ in range(3):
            residual = 2 * c[..., 2] * z + 3 * c[..., 3] * z.square() - target
            done = (residual.abs() <= 1e-11) | ~active
            lo = torch.where((residual < 0) & ~done, z, lo)
            hi = torch.where((residual >= 0) & ~done, z, hi)
            curvature = 2 * c[..., 2] + 6 * c[..., 3] * z
            proposal = z - residual / curvature.clamp_min(1e-300)
            valid = (curvature > 0) & (proposal > lo) & (proposal < hi)
            z = torch.where(done, z, torch.where(valid, proposal, (lo + hi) / 2))
        r = self.grid[index] + z
        r = torch.where(low, self.grid[0], torch.where(high, self.grid[-1], r))
        # Reuse the selected coefficients rather than search the same segment again.
        z = r - self.grid[index]
        value = c[..., 0] + z * (c[..., 1] + z * (c[..., 2] + z * c[..., 3]))
        derivative = c[..., 1] + z * (2 * c[..., 2] + 3 * z * c[..., 3])
        curvature = 2 * c[..., 2] + 6 * z * c[..., 3]
        rprime = torch.where(active, 1 / curvature.clamp_min(1e-300), 0)
        inverse_residual = torch.where(active, (derivative - lam).abs(), 0)
        return r, lam * r - value, rprime, inverse_residual


@torch.no_grad()
def solve_dual_gpu(a, b, table, epsilon=0.0625, ratio_min=0.1, ratio_max=10,
                   marginal_tolerance=1e-6, root_logmass_tolerance=1e-8,
                   max_outer_iterations=1000, max_bisection_iterations=60,
                   callback=None, device="cuda", root_method="newton", inner_callback=None):
    started = time.perf_counter()
    from .neural_coupling import NeuralPairCost
    oracle = table if isinstance(table, (TorchHermiteCost, NeuralPairCost)) else TorchHermiteCost(table, device)
    if root_method not in {"newton", "bisection"}:
        raise ValueError("unknown column root method")
    if isinstance(oracle, NeuralPairCost) and root_method != oracle.root_method:
        raise ValueError("pair and column root methods must agree")
    a = torch.as_tensor(a, dtype=torch.float64, device=device)
    b = torch.as_tensor(b, dtype=torch.float64, device=device)
    if bool(torch.any(a <= 0) | torch.any(b <= 0)) or epsilon <= 0:
        raise ValueError("positive masses and entropy coefficient required")
    mass_ratio = float(b.sum() / a.sum())
    if not ratio_min < mass_ratio < ratio_max:
        status = "infeasible_ratio_range" if mass_ratio < ratio_min or mass_ratio > ratio_max else "finite_dual_boundary_unsupported"
        raise OracleError(status, dict(mass_ratio=mass_ratio))
    loga, logb = a.log(), b.log()
    # torch.quantile reproduces numpy's even-sized median initialization.
    lam = torch.quantile(oracle(torch.ones(oracle.shape, dtype=torch.float64, device=device))[1], 0.5, dim=0)
    history = []
    last_inner_update = started

    def inner_progress(phase, iteration, inner_iteration=0):
        nonlocal last_inner_update
        now = time.perf_counter()
        if inner_callback is not None and now-last_inner_update >= 15:
            inner_callback(dict(phase=phase, iteration=iteration, inner_iteration=inner_iteration,
                                seconds=now-started, root_method=root_method))
            last_inner_update = now

    def column(value, logs):
        r, star, rprime, inverse_residual = oracle.conjugate(value[None, :])
        logp = logs[:, None] + star / epsilon
        logq = logp + r.log()
        logmass = torch.logsumexp(logq, dim=0)
        dlogmass = (((logq - logmass[None, :]).exp() * (r / epsilon + rprime / r)).sum(0)
                    if root_method == "newton" else None)
        return logmass - logb, dlogmass, r, star, logp, inverse_residual

    status = "max_outer_iterations"
    for iteration in range(1, max_outer_iterations + 1):
        r, star, _, _ = oracle.conjugate(lam[None, :])
        inner_progress("row_normalization", iteration)
        logs = loga - torch.logsumexp(star / epsilon, dim=1)
        width = torch.full_like(lam, max(epsilon, 0.1))
        lo, hi = lam - width, lam + width
        for bracket_step in range(80):
            flo, fhi = column(lo, logs)[0], column(hi, logs)[0]
            inner_progress("column_bracket", iteration, bracket_step+1)
            bad_lo, bad_hi = flo > 0, fhi < 0
            if not bool(torch.any(bad_lo | bad_hi)):
                break
            width = width * 2
            lo = torch.where(bad_lo, lam - width, lo)
            hi = torch.where(bad_hi, lam + width, hi)
        else:
            raise OracleError("column_root_bracket_failed", {})
        value = lam.clone()
        roots_converged = False
        for column_step in range(max_bisection_iterations):
            residual, dlogmass, r, star, logp, inverse_residual = column(value, logs)
            inner_progress("column_root", iteration, column_step+1)
            done = residual.abs() < root_logmass_tolerance
            if bool(done.all()):
                roots_converged = True
                break
            lo = torch.where((residual < 0) & ~done, value, lo)
            hi = torch.where((residual >= 0) & ~done, value, hi)
            if root_method == "bisection":
                value = torch.where(done, value, (lo + hi) / 2)
            else:
                proposal = value - residual / dlogmass
                valid = torch.isfinite(proposal) & (dlogmass > 0) & (proposal > lo) & (proposal < hi)
                value = torch.where(done, value, torch.where(valid, proposal, (lo + hi) / 2))
        if not roots_converged:
            raise OracleError("column_root_not_converged", dict(logmass_residual=float(residual.abs().max())))
        lam = value
        logq = logp + r.log()
        row_error = (torch.logsumexp(logp, dim=1).exp() - a).abs().sum() / a.sum()
        col_error = (torch.logsumexp(logq, dim=0).exp() - b).abs().sum() / b.sum()
        P, Q = logp.exp(), logq.exp()
        cost = lam[None, :] * r - star
        objective = (P * (cost + epsilon * (logp - 1))).sum()
        dual = epsilon * torch.dot(a, logs) + torch.dot(b, lam) - epsilon * P.sum()
        boundary = (r == ratio_min) | (r == ratio_max)
        # One host transfer supplies both stopping and logging diagnostics.
        scalar_values = torch.stack((row_error, col_error, inverse_residual.max(), objective, dual,
                                     objective - dual, residual.abs().max(), boundary.double().mean(),
                                     (P * boundary).sum() / P.sum())).cpu().tolist()
        errors, diagnostics = scalar_values[:3], scalar_values[3:]
        if errors[2] > 1e-7:
            raise OracleError("pair_root_not_converged", dict(max_residual=errors[2]))
        record = dict(iteration=iteration, seconds=time.perf_counter()-started, root_method=root_method,
                      row_relative_l1=errors[0], column_relative_l1=errors[1],
                      row_relative_tv=errors[0] / 2, column_relative_tv=errors[1] / 2,
                      objective=diagnostics[0], dual_objective=diagnostics[1],
                      primal_minus_dual_diagnostic=diagnostics[2], gap_is_certificate=False,
                      root_logmass_residual=diagnostics[3], boundary_ratio_fraction=diagnostics[4],
                      boundary_P_mass_fraction=diagnostics[5], inverse_derivative_residual=errors[2])
        history.append(record)
        if callback and (iteration == 1 or iteration % 10 == 0):
            callback(record)
        if max(errors[:2]) < marginal_tolerance:
            status = "converged"
            break
    # One final transfer of result matrices for archival and later stage reuse.
    arrays = [x.cpu().numpy() for x in (P, Q, r, logp, logs, lam)]
    return CouplingResult(*arrays, status, history, time.perf_counter() - started)
