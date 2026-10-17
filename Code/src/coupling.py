from dataclasses import dataclass
import time

import numpy as np
from scipy.special import logsumexp


class OracleError(RuntimeError):
    def __init__(self, status, details):
        super().__init__(status)
        self.status, self.details = status, details


class HermiteCost:
    """One C1 cubic in r supplies cost, both derivatives, and its conjugate.

    This is a numerical query approximation, never a convexification. Each
    cubic's curvature is checked at both endpoints (it is linear within it).
    """

    def __init__(self, grid, costs, derivatives, curvature_tolerance=1e-5):
        self.grid = np.asarray(grid, dtype=np.float64)
        self.costs = np.asarray(costs, dtype=np.float64)
        self.derivatives = np.asarray(derivatives, dtype=np.float64)
        self.shape = self.costs.shape[:-1]
        if self.costs.shape != self.derivatives.shape or self.costs.shape[-1] != len(self.grid):
            raise ValueError("invalid Hermite shapes")
        if not np.isfinite(self.costs).all() or not np.isfinite(self.derivatives).all():
            raise OracleError("cost_oracle_nonfinite", {})
        h = np.diff(self.grid)
        secant = np.diff(self.costs, axis=-1) / h
        c0 = self.costs[..., :-1]
        c1 = self.derivatives[..., :-1]
        c2 = (3 * secant - 2 * c1 - self.derivatives[..., 1:]) / h
        c3 = (c1 + self.derivatives[..., 1:] - 2 * secant) / h ** 2
        self.coefficients = np.stack((c0, c1, c2, c3), axis=-1)
        curvature = np.stack((2 * c2, 2 * c2 + 6 * c3 * h), axis=-1)
        self.min_curvature = float(curvature.min())
        self.min_slope_increment = float(np.diff(self.derivatives, axis=-1).min())
        if self.min_curvature < -curvature_tolerance or self.min_slope_increment < 0:
            index = np.unravel_index(np.argmin(curvature), curvature.shape)
            raise OracleError("cost_oracle_nonconvex", dict(min_curvature=self.min_curvature,
                              min_slope_increment=self.min_slope_increment, table_index=list(index)))
        # Even tiny negative curvature is not silently sorted or modified.
        if self.min_curvature < 0:
            raise OracleError("cost_table_needs_refinement", dict(min_curvature=self.min_curvature))

    def _gather(self, index):
        return np.take_along_axis(self.coefficients, index[..., None, None], axis=-2)[..., 0, :]

    def __call__(self, r):
        r = np.broadcast_to(np.asarray(r, dtype=np.float64), self.shape)
        if np.any(r < self.grid[0]) or np.any(r > self.grid[-1]):
            raise ValueError("ratio outside the trained domain")
        index = np.minimum(np.searchsorted(self.grid, r, side="right") - 1, len(self.grid) - 2)
        c = self._gather(index)
        z = r - self.grid[index]
        value = c[..., 0] + z * (c[..., 1] + z * (c[..., 2] + z * c[..., 3]))
        derivative = c[..., 1] + z * (2 * c[..., 2] + 3 * z * c[..., 3])
        curvature = 2 * c[..., 2] + 6 * z * c[..., 3]
        return value, derivative, curvature

    def conjugate(self, lam, ratio_min=None, ratio_max=None, max_iterations=60):
        lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), self.shape)
        low = lam <= self.derivatives[..., 0]
        high = lam >= self.derivatives[..., -1]
        left = np.zeros(self.shape, dtype=np.int64)
        right = np.full(self.shape, len(self.grid) - 1, dtype=np.int64)
        for _ in range(int(np.ceil(np.log2(len(self.grid)))) + 1):
            mid = (left + right) // 2
            slope = np.take_along_axis(self.derivatives, mid[..., None], axis=-1)[..., 0]
            move = slope <= lam
            left = np.where(move, mid, left)
            right = np.where(move, right, mid)
        index = np.minimum(left, len(self.grid) - 2)
        c = self._gather(index)
        h = self.grid[index + 1] - self.grid[index]
        # Safeguarded Newton within the located segment, with bisection fallback.
        lo = np.zeros(self.shape)
        hi = h.copy()
        z = h / 2
        target = lam - c[..., 1]
        active = ~(low | high)
        for _ in range(max_iterations):
            residual = 2 * c[..., 2] * z + 3 * c[..., 3] * z ** 2 - target
            if np.max(np.abs(residual[active]), initial=0.0) < 1e-11:
                break
            lo = np.where(residual < 0, z, lo)
            hi = np.where(residual >= 0, z, hi)
            curvature = 2 * c[..., 2] + 6 * c[..., 3] * z
            proposal = z - residual / np.where(curvature > 0, curvature, 1)
            valid = (curvature > 0) & (proposal > lo) & (proposal < hi)
            z = np.where(valid, proposal, (lo + hi) / 2)
        r = self.grid[index] + z
        r = np.where(low, self.grid[0], np.where(high, self.grid[-1], r))
        value, derivative, curvature = self(r)
        residual = np.abs(derivative - lam)
        if np.max(residual[active], initial=0.0) > 1e-7:
            raise OracleError("pair_root_not_converged", dict(max_residual=float(residual[active].max())))
        rprime = np.where(active, 1 / np.maximum(curvature, 1e-300), 0)
        return r, lam * r - value, rprime


def conjugate_bisection(oracle, lam, shape, ratio_min, ratio_max, max_iterations=60):
    """Generic reference implementation for smooth convex analytic/network costs."""
    lam = np.broadcast_to(lam, shape)
    lo = np.full(shape, ratio_min)
    hi = np.full(shape, ratio_max)
    _, dlo, _ = oracle(lo)
    _, dhi, _ = oracle(hi)
    low, high = lam <= dlo, lam >= dhi
    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        _, derivative, curvature = oracle(mid)
        if np.min(curvature) < -1e-8:
            raise OracleError("cost_oracle_nonconvex", dict(min_curvature=float(curvature.min())))
        lo = np.where(derivative < lam, mid, lo)
        hi = np.where(derivative >= lam, mid, hi)
    r = np.where(low, ratio_min, np.where(high, ratio_max, (lo + hi) / 2))
    value, derivative, curvature = oracle(r)
    rprime = np.where(~(low | high), 1 / np.maximum(curvature, 1e-300), 0)
    return r, lam * r - value, rprime


@dataclass
class CouplingResult:
    P: np.ndarray
    Q: np.ndarray
    r: np.ndarray
    log_P: np.ndarray
    log_s: np.ndarray
    lam: np.ndarray
    status: str
    history: list
    seconds: float


def solve_dual(a, b, oracle, epsilon=0.0625, ratio_min=0.1, ratio_max=10,
               marginal_tolerance=1e-6, root_logmass_tolerance=1e-8,
               max_outer_iterations=1000, max_bisection_iterations=60, callback=None):
    """Exactly the row-normalization / independent scalar column-root algorithm."""
    started = time.perf_counter()
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.any(a <= 0) or np.any(b <= 0) or epsilon <= 0:
        raise ValueError("positive masses and entropy coefficient required")
    mass_ratio = b.sum() / a.sum()
    if not ratio_min < mass_ratio < ratio_max:
        # Equality is feasible, but generally requires infinite limiting dual
        # potentials. This finite-dual implementation reports that case explicitly.
        status = "infeasible_ratio_range" if mass_ratio < ratio_min or mass_ratio > ratio_max else "finite_dual_boundary_unsupported"
        raise OracleError(status, dict(mass_ratio=float(mass_ratio)))
    shape = (len(a), len(b))
    _, derivative, curvature = oracle(np.ones(shape))
    if not np.isfinite(derivative).all() or np.min(curvature) < -1e-8:
        raise OracleError("cost_oracle_nonconvex", dict(min_curvature=float(curvature.min())))
    lam = np.median(derivative, axis=0)
    loga, logb = np.log(a), np.log(b)
    history = []

    def conjugate(value):
        if hasattr(oracle, "conjugate"):
            return oracle.conjugate(value[None, :], ratio_min, ratio_max, max_bisection_iterations)
        return conjugate_bisection(oracle, value[None, :], shape, ratio_min, ratio_max, max_bisection_iterations)

    def column(value, logs):
        r, star, rprime = conjugate(value)
        logp = logs[:, None] + star / epsilon
        logq = logp + np.log(r)
        logmass = logsumexp(logq, axis=0)
        normalized_q = np.exp(logq - logmass[None, :])
        derivative_logmass = (normalized_q * (r / epsilon + rprime / r)).sum(axis=0)
        return logmass - logb, derivative_logmass, r, star, logp

    status = "max_outer_iterations"
    for iteration in range(1, max_outer_iterations + 1):
        r, star, _ = conjugate(lam)
        logs = loga - logsumexp(star / epsilon, axis=1)
        # Expand independent left/right brackets while row coefficients stay fixed.
        width = np.full(len(b), max(epsilon, 0.1))
        lo, hi = lam - width, lam + width
        for _ in range(80):
            flo = column(lo, logs)[0]
            fhi = column(hi, logs)[0]
            bad_lo, bad_hi = flo > 0, fhi < 0
            if not (bad_lo.any() or bad_hi.any()):
                break
            width *= 2
            lo = np.where(bad_lo, lam - width, lo)
            hi = np.where(bad_hi, lam + width, hi)
        else:
            raise OracleError("column_root_bracket_failed", {})
        value = np.minimum(np.maximum(lam, lo), hi)
        roots_converged = False
        for _ in range(max_bisection_iterations):
            residual, dlogmass, r, star, logp = column(value, logs)
            done = np.abs(residual) < root_logmass_tolerance
            if done.all():
                roots_converged = True
                break
            lo = np.where((residual < 0) & ~done, value, lo)
            hi = np.where((residual >= 0) & ~done, value, hi)
            proposal = value - residual / dlogmass
            valid = np.isfinite(proposal) & (proposal > lo) & (proposal < hi)
            value = np.where(done, value, np.where(valid, proposal, (lo + hi) / 2))
        if not roots_converged:
            raise OracleError("column_root_not_converged", dict(logmass_residual=float(np.abs(residual).max())))
        lam = value
        logq = logp + np.log(r)
        # Residuals from log sums remain valid when tiny entries underflow on export.
        row_sums = np.exp(logsumexp(logp, axis=1))
        col_sums = np.exp(logsumexp(logq, axis=0))
        row_error = float(np.abs(row_sums - a).sum() / a.sum())
        col_error = float(np.abs(col_sums - b).sum() / b.sum())
        P, Q = np.exp(logp), np.exp(logq)
        cost = oracle(r)[0]
        objective = float(np.sum(P * (cost + epsilon * (logp - 1))))
        dual = float(epsilon * np.dot(a, logs) + np.dot(b, lam) - epsilon * P.sum())
        record = dict(iteration=iteration, row_relative_l1=row_error, column_relative_l1=col_error,
                      objective=objective, dual_objective=dual,
                      primal_minus_dual_diagnostic=objective - dual,
                      gap_is_certificate=False, root_logmass_residual=float(np.abs(residual).max()),
                      boundary_ratio_fraction=float(np.mean((r == ratio_min) | (r == ratio_max))),
                      boundary_P_mass_fraction=float(P[(r == ratio_min) | (r == ratio_max)].sum() / P.sum()))
        history.append(record)
        if callback and (iteration == 1 or iteration % 25 == 0):
            callback(record)
        if max(row_error, col_error) < marginal_tolerance:
            status = "converged"
            break
    return CouplingResult(P, Q, r, logp, logs, lam, status, history, time.perf_counter() - started)
