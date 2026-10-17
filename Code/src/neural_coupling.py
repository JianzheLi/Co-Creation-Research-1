"""Continuous cost-MLP queries and ratio inversion, without interpolation."""
import math
import warnings
import torch

from .coupling import OracleError


class NeuralPairCost:
    def __init__(self, oracle, delta, root_method="newton"):
        if root_method not in {"newton", "bisection"}:
            raise ValueError("unknown neural ratio root method")
        self.root_method = root_method
        self.model = oracle.model
        parameter = next(self.model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        delta = torch.as_tensor(delta, dtype=self.dtype, device=self.device)
        self.shape = tuple(delta.shape[:-1])
        self.distance = delta.norm(dim=-1).reshape(-1, 1)
        self.chunk = oracle.condition_chunk
        self.ratio_min, self.ratio_max = oracle.ratio_min, oracle.ratio_max
        self.warning_records = {}
        self.query_calls = [0, 0, 0]
        self.query_conditions = [0, 0, 0]
        if bool((self.distance > oracle.distance_max*(1+1e-12)).any()):
            raise ValueError("pair distance outside distilled domain")
        # Two endpoint evaluations are reused for boundary KKT conditions.
        # Interior evaluations always use the continuous MLP and autodiff.
        self.lower = self._query(self.distance, torch.full_like(self.distance, self.ratio_min))
        self.upper = self._query(self.distance, torch.full_like(self.distance, self.ratio_max))
        if bool((self.lower[1] >= self.upper[1]).any()):
            self._warn("endpoint_ordering", dict(reason="endpoint derivative ordering"))
        self._warm_ratio = None

    def _warn(self, phase, details):
        if phase not in self.warning_records:
            self.warning_records[phase] = dict(phase=phase, occurrences=0, first_details=details)
            warnings.warn(f"Cost monotonicity warning ({phase}): {details}; continuing without a convexity certificate",
                          RuntimeWarning, stacklevel=2)
        self.warning_records[phase]["occurrences"] += 1

    def _query(self, distance, ratio, derivatives=None, validate_ratio=False):
        derivatives = (1 if self.root_method == "bisection" else 2) if derivatives is None else derivatives
        self.query_calls[derivatives] += 1
        self.query_conditions[derivatives] += len(ratio)
        outputs = tuple(torch.empty_like(ratio) for _ in range(derivatives+1))
        for start in range(0, len(ratio), self.chunk):
            stop = start+self.chunk
            with torch.set_grad_enabled(derivatives > 0):
                r = ratio[start:stop].detach().requires_grad_(derivatives > 0)
                cost = self.model(distance[start:stop], r.log())
                values = (cost,)
                if derivatives:
                    slope = torch.autograd.grad(cost.sum(), r, create_graph=derivatives == 2)[0]
                    values += (slope,)
                if derivatives == 2:
                    values += (torch.autograd.grad(slope.sum(), r)[0],)
            for output, value in zip(outputs, values):
                output[start:stop] = value.detach()
        finite = torch.stack([torch.isfinite(value).all() for value in outputs]).all()
        if validate_ratio:
            outside = ((ratio < self.ratio_min) | (ratio > self.ratio_max)).any()
            # Combine domain and finite checks into a single host transfer.
            outside_value, finite_value = torch.stack((outside, finite)).cpu().tolist()
            if outside_value:
                raise ValueError("pair ratio outside distilled domain")
        else:
            finite_value = bool(finite)
        if not finite_value:
            raise OracleError("cost_oracle_nonfinite", {})
        return outputs

    def value(self, r):
        r = torch.broadcast_to(torch.as_tensor(r, dtype=self.dtype, device=self.device), self.shape)
        return self._query(self.distance, r.reshape(-1, 1), derivatives=0, validate_ratio=True)[0].reshape(self.shape)

    def __call__(self, r):
        r = torch.broadcast_to(torch.as_tensor(r, dtype=self.dtype, device=self.device), self.shape)
        return tuple(value.reshape(self.shape) for value in self._query(self.distance, r.reshape(-1, 1), validate_ratio=True))

    def check_curvature(self, ratio_nodes, curvature_tolerance):
        """Diagnostic probes only; these values are never used for cost queries.

        A finite check is not a convexity proof. Nonpositive curvature is a
        warning; it never modifies network values or stops the solver itself.
        """
        sorted_distance = self.distance.flatten().sort().values
        ids = torch.linspace(0, len(sorted_distance)-1, min(128, len(sorted_distance)),
                             device=self.device).long()
        d = sorted_distance[ids].repeat_interleave(ratio_nodes)[:, None]
        r = torch.linspace(math.log(self.ratio_min), math.log(self.ratio_max),
                           ratio_nodes, dtype=self.dtype, device=self.device).exp().repeat(len(ids))[:, None]
        curvature = self._query(d, r, derivatives=2)[2]
        index = int(curvature.argmin())
        report = dict(min_curvature=float(curvature.flatten()[index]),
                      distance=float(d.flatten()[index]), ratio=float(r.flatten()[index]),
                      samples=len(d), method="direct_mlp_autodiff", global_convexity_proven=False)
        if report["min_curvature"] < -curvature_tolerance:
            self._warn("domain_probe", report)
        return report

    @torch.no_grad()
    def conjugate(self, lam):
        lam = torch.broadcast_to(lam, self.shape).reshape(-1, 1)
        low, high = lam <= self.lower[1], lam >= self.upper[1]
        bad_order = self.lower[1] >= self.upper[1]
        prefer_lower = lam*self.ratio_min-self.lower[0] >= lam*self.ratio_max-self.upper[0]
        low = torch.where(bad_order, prefer_lower, low)
        high = torch.where(bad_order, ~prefer_lower, high)
        interior = ~(low | high)
        r = torch.full_like(lam, (self.ratio_min*self.ratio_max)**.5)
        if self._warm_ratio is not None:
            r.copy_(self._warm_ratio)
        r = r.clamp(self.ratio_min, self.ratio_max)
        r = torch.where(low, self.ratio_min, torch.where(high, self.ratio_max, r))
        cost = torch.where(low, self.lower[0], self.upper[0]).clone()
        curvature = torch.ones_like(r)
        inverse_residual = torch.zeros_like(r)
        lo, hi = torch.full_like(r, self.ratio_min), torch.full_like(r, self.ratio_max)
        active = interior.clone()
        for _ in range(64):
            ids = active.flatten().nonzero().flatten()
            if len(ids) == 0:
                break
            values = self._query(self.distance[ids], r[ids])
            c, slope = values[:2]
            curve = values[2] if self.root_method == "newton" else None
            if curve is not None and bool((curve <= 0).any()):
                worst = int(curve.argmin())
                pair = ids[worst]
                self._warn("continuous_ratio_inverse", dict(
                    distance=float(self.distance[pair]), ratio=float(r[pair]),
                    curvature=float(curve.flatten()[worst]), phase="continuous_ratio_inverse"))
            cost[ids] = c
            if curve is not None:
                curvature[ids] = curve
            residual = slope-lam[ids]
            inverse_residual[ids] = residual.abs()
            done = residual.abs() <= 1e-9
            active[ids] = ~done
            lo[ids] = torch.where((residual < 0) & ~done, r[ids], lo[ids])
            hi[ids] = torch.where((residual >= 0) & ~done, r[ids], hi[ids])
            if self.root_method == "bisection":
                r[ids] = torch.where(done, r[ids], (lo[ids]+hi[ids])/2)
            else:
                proposal = r[ids]-residual/curve
                valid = (curve > 0) & torch.isfinite(proposal) & (proposal > lo[ids]) & (proposal < hi[ids])
                r[ids] = torch.where(done, r[ids], torch.where(valid, proposal, (lo[ids]+hi[ids])/2))
        else:
            if bool(active.any()):
                raise OracleError("pair_root_not_converged", dict(max_residual=float(inverse_residual.max())))
        self._warm_ratio = r.clone()
        # A nonpositive curvature does not define the increasing inverse used
        # by column Newton. NaN signals an unavailable derivative so that the
        # column update falls back to its bracket midpoint; never clamp Crr.
        rprime = (torch.where(interior, torch.where(curvature > 0, 1/curvature, float("nan")), 0)
                  if self.root_method == "newton" else None)
        return (r.reshape(self.shape), (lam*r-cost).reshape(self.shape),
                rprime.reshape(self.shape) if rprime is not None else None, inverse_residual.reshape(self.shape))
