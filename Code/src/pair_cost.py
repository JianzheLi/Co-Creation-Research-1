import numpy as np
import torch
from .pair_path import exprel, spectral_apply


def finite_moments(s, sigma=0.25, alpha=10, mode="quadratic_minimal"):
    t, tau, Z = s["t"], s["tau"], s["Z"]
    wstar = -s["delta"] + t * s["h_mu"]
    Fstar = 0.5 * (1 - t) * wstar.square().sum(-1, keepdim=True) + 0.5 * sigma ** 2 * t * Z.exp().sum(-1, keepdim=True)
    Dunif = 0.5 * s["h_var"] * exprel(-Z) + 0.5 * s["dotZ"]
    if s.get("cost_objective") in {"tight_logmoment_upper_bound", "gaussian_reference_upper_bound", "diffusion_full_square_upper_bound"}:
        beta, zeta = s["growth_linear"], s["growth_quadratic"]
        sbeta = spectral_apply(s,s["var"],beta)
        Eres = (s["drift_mean"]+wstar).square().sum(-1,keepdim=True)+(s["var"]*(Dunif-.5*zeta).square()).sum(-1,keepdim=True)
        Eg2 = s["gamma"].square()+(beta*sbeta).sum(-1,keepdim=True)+.5*zeta.square().sum(-1,keepdim=True)
        EgF = s["gamma"]*Fstar+(sbeta*wstar).sum(-1,keepdim=True)+.5*(zeta*sigma**2*t*Z.exp()).sum(-1,keepdim=True)
    elif mode == "quadratic_minimal":
        A = -s["b"] + wstar
        D = (sigma ** 2 * t * Z.exp() + 2 * alpha * Dunif) / (2 * alpha + s["var"])
        Eres = (A.square() + s["var"] * D.square()).sum(-1, keepdim=True)
        covariance_b = spectral_apply(s, s["var"], s["b"])
        Eg2 = s["gamma"].square() + ((s["b"] * covariance_b).sum(-1, keepdim=True) + 0.5 * s["K"].square().sum(-1, keepdim=True)) / alpha ** 2
        EgF = s["gamma"] * Fstar - (covariance_b * wstar).sum(-1, keepdim=True) / alpha - (s["K"] * sigma ** 2 * t * Z.exp()).sum(-1, keepdim=True) / (2 * alpha)
    elif mode == "uniform_growth":
        A = s["dotxi"] + wstar
        Eres = (A.square() + s["var"] * Dunif.square()).sum(-1, keepdim=True)
        Eg2 = s["gamma"].square()
        EgF = s["gamma"] * Fstar
    else:
        raise ValueError(mode)
    return dict(E_res=Eres, E_g2=Eg2, E_gF=EgF)


def logmoment_bound_correction(s, sigma, alpha, mode):
    """Exact Gaussian moments for alpha/2 E(g + sigma^2/alpha log(phi/q))^2.

    q=N(x0,sigma^2*t*I). Only the open interval is used: the logarithmic
    terminal singularity is integrable, not a finite endpoint value.
    Compatible controls are unchanged; they need not minimize the new bound.
    """
    t = s["t"]
    displacement = s["mean_displacement"]
    relative_variance = (1-t)*s["Z"].exp()
    kl = .5*((-torch.log1p(-t)-s["Z"]+relative_variance-1).sum(-1,keepdim=True)
             + displacement.square().sum(-1,keepdim=True)/(sigma**2*t))
    mean = s["gamma"] + sigma**2/alpha*kl
    if mode == "quadratic_minimal":
        linear = (-s["b"]+displacement/t)/alpha
        quadratic = (sigma**2*(relative_variance-1)-s["K"])/alpha
    elif mode == "uniform_growth":
        linear = displacement/(alpha*t)
        quadratic = sigma**2*(relative_variance-1)/alpha
    else:
        raise ValueError(mode)
    variance = (linear*spectral_apply(s,s["var"],linear)).sum(-1,keepdim=True)
    variance = variance + .5*quadratic.square().sum(-1,keepdim=True)
    return .5*alpha*(mean.square()+variance)+2*sigma**4/(alpha*np.e**2)


def finite_pair_integrand(s, sigma=0.25, alpha=10, mode="quadratic_minimal", return_terms=False):
    terms = finite_moments(s, sigma, alpha, mode)
    q = -s["log_r"] + s["t"] * s["h_mass"]
    endpoint_mass = s["log_r"].exp() * q * exprel((1 - s["t"]) * q)
    d = s["delta"].shape[-1]
    parts = dict(residual=0.5 * s["mass"] * terms["E_res"],
                 growth=0.5 * alpha * s["mass"] * terms["E_g2"],
                 growth_F=s["mass"] * terms["E_gF"],
                 endpoint=d * sigma ** 2 / 2 * endpoint_mass)
    if s.get("cost_objective", "renormalized_surrogate") == "logmoment_upper_bound":
        parts["bound_correction"] = s["mass"] * logmoment_bound_correction(s, sigma, alpha, mode)
    elif s.get("cost_objective") == "tight_logmoment_upper_bound":
        from .tight_bound import positive_log_moment, log_ratio_moments
        kl, _ = log_ratio_moments(s,sigma)
        cross = s["gamma"]*kl+(s["growth_linear"]*spectral_apply(s,s["var"],s["mean_displacement"]/(sigma**2*s["t"]))).sum(-1,keepdim=True)
        cross = cross+.5*(s["growth_quadratic"]*((1-s["t"])*s["Z"].exp()-1)).sum(-1,keepdim=True)
        positive = positive_log_moment(s,sigma,nodes=s.get("logmoment_nodes",32))
        parts["bound_correction"] = s["mass"]*(.5*alpha*terms["E_g2"]+sigma**2*cross+sigma**4/(2*alpha)*(positive+4/np.e**2))
    elif s.get('cost_objective') in {'gaussian_reference_upper_bound', 'diffusion_full_square_upper_bound'}:
        from .gaussian_reference_bound import moments
        _, second, cross = moments(s, sigma)
        young_weight = sigma**2 if s['cost_objective'] == 'diffusion_full_square_upper_bound' else alpha
        parts['bound_correction'] = s['mass']*(.5*young_weight*terms['E_g2']+sigma**2*cross+sigma**4/(2*young_weight)*(second+4/np.e**2))
    value = sum(parts.values())
    return (value, parts) if return_terms else value


class CostOracle:
    """Frozen network plus fixed Gauss--Legendre quadrature; derivatives in r."""
    def __init__(self, path, nodes=32, condition_chunk=512):
        self.path = path.freeze()
        self.nodes, self.condition_chunk = nodes, condition_chunk
        x, w = np.polynomial.legendre.leggauss(nodes)
        p = next(path.parameters())
        self.t = torch.as_tensor((x + 1) / 2, dtype=p.dtype, device=p.device)
        self.w = torch.as_tensor(w / 2, dtype=p.dtype, device=p.device)
        if getattr(path, "cost_objective", "renormalized_surrogate").endswith("upper_bound"):
            u = self.t
            self.t = 1-(1-u)**2
            self.w = self.w*2*(1-u)
            if not torch.all((self.t > 0) & (self.t < 1)):
                raise ValueError("quadrature precision is insufficient for open-interval bound")

    def cost_tensor(self, delta, log_r):
        n, d = delta.shape
        t = self.t.repeat(n).reshape(-1, 1)
        s = self.path(t, delta.repeat_interleave(self.nodes, 0), log_r.repeat_interleave(self.nodes, 0))
        values = finite_pair_integrand(s, self.path.sigma, self.path.alpha, self.path.control_mode).reshape(n, self.nodes)
        return 0.5 * delta.square().sum(-1) - d * self.path.sigma ** 2 / 2 * log_r.exp().squeeze(-1) + (values * self.w).sum(-1)

    def tensor_query(self, delta, r, derivatives=2):
        """Evaluate on the path device, keeping all chunk outputs on that device.

        Chunking bounds the quadrature/autograd working set; it does not sample
        or approximate the conditions. Returned values and analytic derivatives
        with respect to r are detached, as in the NumPy compatibility interface.
        """
        if derivatives not in (0, 1, 2):
            raise ValueError("derivatives must be 0, 1, or 2")
        if self.condition_chunk <= 0:
            raise ValueError("condition_chunk must be positive")
        p = next(self.path.parameters())
        delta = torch.as_tensor(delta, dtype=p.dtype, device=p.device).reshape(-1, self.path.dimension)
        r = torch.as_tensor(r, dtype=p.dtype, device=p.device).reshape(-1, 1)
        if len(r) == 1:
            r = r.expand(len(delta), 1)
        if len(r) != len(delta) or torch.any(r <= 0):
            raise ValueError("positive ratio per condition required")
        # Preallocation avoids retaining chunk lists and a second complete copy
        # during concatenation for full coupling tables with millions of entries.
        collected = tuple(torch.empty(len(delta), dtype=p.dtype, device=p.device)
                          for _ in range(derivatives + 1))
        for start in range(0, len(delta), self.condition_chunk):
            dr = delta[start:start + self.condition_chunk]
            rr = r[start:start + self.condition_chunk]
            with torch.set_grad_enabled(derivatives > 0):
                ell = rr.log().detach().requires_grad_(derivatives > 0)
                cost = self.cost_tensor(dr, ell)
                values = [cost]
                if derivatives:
                    c_ell = torch.autograd.grad(cost.sum(), ell, create_graph=derivatives == 2)[0]
                    values.append((c_ell / rr).squeeze(-1))
                    if derivatives == 2:
                        c_ell_ell = torch.autograd.grad(c_ell.sum(), ell)[0]
                        values.append(((c_ell_ell - c_ell) / rr.square()).squeeze(-1))
            for result, value in zip(collected, values):
                result[start:start + len(dr)].copy_(value.detach())
        return collected[0] if derivatives == 0 else collected

    def __call__(self, delta, r, derivatives=2):
        """Compatibility API: transfer final results only, never per chunk."""
        result = self.tensor_query(delta, r, derivatives)
        if derivatives == 0:
            return result.cpu().numpy()
        return tuple(value.cpu().numpy() for value in result)
