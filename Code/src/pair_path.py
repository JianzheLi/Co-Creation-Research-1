import torch
from torch import nn
from torch.nn import functional as F


def exprel(z):
    """Both branches have safe denominators, including under autograd."""
    small = z.abs() < 1e-5
    safe_z = torch.where(small, torch.ones_like(z), z)
    series = 1 + z / 2 + z.square() / 6 + z.pow(3) / 24 + z.pow(4) / 120
    return torch.where(small, series, torch.expm1(z) / safe_z)


class SmoothCorrection(nn.Module):
    def __init__(self, n_input, n_output, width=128, layers=3, bound=4):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(n_input if k == 0 else width, width) for k in range(layers)])
        self.last = nn.Linear(width, n_output)
        self.bound = bound
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, z):
        for layer in self.hidden:
            z = torch.tanh(layer(z))
        return self.bound * torch.tanh(self.last(z))

    def value_and_time_jvp(self, z, squash=True):
        # Exact forward-mode chain rule. The tangent remains differentiable in
        # parameters, time, and log-ratio (including second ratio derivatives).
        tangent = torch.zeros_like(z)
        tangent[:, 0] = 1
        for layer in self.hidden:
            z = torch.tanh(layer(z))
            tangent = (1 - z.square()) * F.linear(tangent, layer.weight)
        raw = self.last(z)
        derivative = F.linear(tangent, self.last.weight)
        if not squash:
            return raw, derivative
        raw = torch.tanh(raw)
        derivative = (1 - raw.square()) * derivative
        return self.bound * raw, self.bound * derivative


def spectral_apply(state, eigenvalues, vector):
    """Apply a covariance-aligned operator without constructing dense matrices.

    In axial states eigenvalues are [parallel, perpendicular, ...], whereas
    vectors are always in world coordinates. Legacy states are coordinate-diagonal.
    """
    axis = state.get("axis")
    if axis is None:
        return eigenvalues * vector
    parallel = eigenvalues[..., :1]
    perpendicular = eigenvalues[..., 1:2] if eigenvalues.shape[-1] > 1 else parallel
    return perpendicular * vector + (parallel - perpendicular) * axis * (axis * vector).sum(-1, keepdim=True)


class PairPath(nn.Module):
    def __init__(self, dimension=2, sigma=0.25, alpha=10, width=128, layers=3,
                 bounds=(4, 4, 4), delta_scale=1, control_mode="quadratic_minimal",
                 parameterization="diagonal", cost_objective="renormalized_surrogate",
                 reference_second_moment=None):
        super().__init__()
        if sigma <= 0 or alpha <= 0 or delta_scale <= 0:
            raise ValueError("positive sigma, alpha and delta_scale required")
        if control_mode not in {"quadratic_minimal", "uniform_growth"}:
            raise ValueError(control_mode)
        if parameterization not in {"diagonal", "axial_distance"}:
            raise ValueError(f"unknown pair parameterization: {parameterization}")
        if dimension < 1 or any(bound <= 0 for bound in bounds):
            raise ValueError("positive dimension and correction bounds required")
        self.dimension, self.sigma, self.alpha = dimension, sigma, alpha
        self.delta_scale, self.control_mode = delta_scale, control_mode
        self.parameterization = parameterization
        if cost_objective not in {"renormalized_surrogate", "logmoment_upper_bound", "tight_logmoment_upper_bound", "gaussian_reference_upper_bound", "diffusion_full_square_upper_bound"}:
            raise ValueError(cost_objective)
        self.cost_objective = cost_objective
        self.reference_second_moment = reference_second_moment
        if cost_objective in {'gaussian_reference_upper_bound', 'diffusion_full_square_upper_bound'} and (reference_second_moment is None or reference_second_moment <= 0):
            raise ValueError('positive data-derived reference_second_moment required')
        self.logmoment_nodes = 32
        axial = parameterization == "axial_distance"
        n_input = 3 if axial else dimension + 2
        self.mu_net = SmoothCorrection(n_input, 1 if axial else dimension, width, layers, bounds[0])
        self.var_net = SmoothCorrection(n_input, 2 if axial else dimension, width, layers, bounds[1])
        self.mass_net = SmoothCorrection(n_input, 1, width, layers, bounds[2])

    def forward(self, t, delta, log_r, create_graph=True, x0=None):
        if t.ndim != 2 or t.shape[1] != 1 or log_r.shape != t.shape:
            raise ValueError("time/log_r must have shape [B,1]")
        if delta.shape != (len(t), self.dimension):
            raise ValueError("delta must have shape [B,dimension]")
        axial = self.parameterization == "axial_distance"
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        inputs = torch.cat((t, (distance if axial else delta) / self.delta_scale, log_r), dim=-1)
        hm, dhm = self.mu_net.value_and_time_jvp(inputs)
        if axial:
            # Mean progress is scalar; orientation enters only through geometry.
            hm, dhm = hm * delta, dhm * delta
            raw, draw = self.var_net.value_and_time_jvp(inputs, squash=False)
            # Isotropy at coincident endpoints, with anisotropy O(distance^2).
            # This uses the existing distance scale and introduces no new scale.
            q2 = (distance / self.delta_scale).square()
            parallel = torch.tanh(raw[:, :1] + q2 * raw[:, 1:2])
            perpendicular = torch.tanh(raw[:, :1])
            dparallel = (1 - parallel.square()) * (draw[:, :1] + q2 * draw[:, 1:2])
            dperpendicular = (1 - perpendicular.square()) * draw[:, :1]
            hs = self.var_net.bound * torch.cat((parallel, perpendicular.expand(-1, self.dimension - 1)), -1)
            dhs = self.var_net.bound * torch.cat((dparallel, dperpendicular.expand(-1, self.dimension - 1)), -1)
        else:
            hs, dhs = self.var_net.value_and_time_jvp(inputs)
        hw, dhw = self.mass_net.value_and_time_jvp(inputs)
        tau = t * (1 - t)
        dtau = 1 - 2 * t
        Z = tau * hs
        dotZ = dtau * hs + tau * dhs
        expZ = Z.exp()
        var = self.sigma ** 2 * tau * expZ
        dotvar = self.sigma ** 2 * expZ * (dtau + tau * dotZ)
        dvar_minus_sigma2 = self.sigma ** 2 * (dtau * torch.expm1(Z) - 2 * t + tau * expZ * dotZ)
        xi = t * delta + tau * hm
        if x0 is not None:
            xi = xi + x0
        dotxi = delta + dtau * hm + tau * dhm
        gamma = log_r + dtau * hw + tau * dhw
        mass = (t * log_r + tau * hw).exp()
        b = -self.alpha * dotxi / (self.alpha + (var[:, :1] if axial else var))
        K = -self.alpha * dvar_minus_sigma2 / (2 * self.alpha + var)
        state = dict(cost_objective=self.cost_objective, mean_displacement=t * delta + tau * hm, t=t, delta=delta, log_r=log_r, tau=tau, xi=xi, dotxi=dotxi,
                     var=var, dotvar=dotvar, Z=Z, dotZ=dotZ, mass=mass, gamma=gamma,
                     h_mu=hm, h_var=hs, h_mass=hw, b=b, K=K,
                     dvar_minus_sigma2=dvar_minus_sigma2)
        if axial:
            # Zero displacement has zero mean correction and isotropic covariance.
            state["axis"] = delta / torch.where(distance > 0, distance, torch.ones_like(distance))
        if self.cost_objective == "tight_logmoment_upper_bound":
            if not axial:
                raise ValueError("tight Gaussian moment bound requires axial_distance")
            beta = (-t*hm+tau*dhm)/(2*self.alpha+var[:, :1])
            zeta = self.sigma**2*(-torch.expm1(Z)+expZ*tau*dotZ)/(4*self.alpha+var)
            state.update(logmoment_nodes=self.logmoment_nodes, growth_linear=beta, growth_quadratic=zeta,
                         drift_mean=dotxi-spectral_apply(state,var,beta),
                         drift_times_var=.5*(dvar_minus_sigma2-var*zeta))
        elif self.cost_objective in {'gaussian_reference_upper_bound', 'diffusion_full_square_upper_bound'}:
            if not axial:
                raise ValueError('Gaussian reference requires axial_distance')
            scale = self.reference_second_moment
            denominator = self.sigma**2*(1-t)+scale*t
            # Cancellation-free form of dotxi - sigma^2*xi/reference_variance.
            numerator = t*(scale-self.sigma**2)/denominator*delta
            numerator = numerator+t*(scale-2*self.sigma**2+2*t*(self.sigma**2-scale))/denominator*hm+tau*dhm
            growth_weight = self.alpha + (self.sigma**2 if self.cost_objective == 'diffusion_full_square_upper_bound' else self.alpha)
            beta = numerator/(growth_weight+var[:, :1])
            one_minus_relative = (scale*t-self.sigma**2*(1-t)*torch.expm1(Z))/denominator
            zeta = (dvar_minus_sigma2+2*self.sigma**2*one_minus_relative)/(2*growth_weight+var)
            state.update(reference_variance=t*denominator, growth_linear=beta, growth_quadratic=zeta,
                         drift_mean=dotxi-spectral_apply(state, var, beta),
                         drift_times_var=.5*(dvar_minus_sigma2-var*zeta))
        # create_graph is kept for the public interface; forward-mode tangent
        # needs no reverse-mode grad(t). Callers decide when to detach labels.
        return state

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self


def controls(state, x, alpha=10, mode="quadratic_minimal"):
    if torch.any(state["var"] <= 0):
        raise ValueError("controls are defined on the open time interval")
    y = x - state["xi"]
    if state.get("cost_objective") in {"tight_logmoment_upper_bound", "gaussian_reference_upper_bound", "diffusion_full_square_upper_bound"}:
        u = state["drift_mean"]+spectral_apply(state,state["drift_times_var"]/state["var"],y)
        g = state["gamma"]+(state["growth_linear"]*y).sum(-1,keepdim=True)
        g = g+.5*((y*spectral_apply(state,state["growth_quadratic"]/state["var"],y)).sum(-1,keepdim=True)
                  -state["growth_quadratic"].sum(-1,keepdim=True))
        return u,g
    if mode == "quadratic_minimal":
        H = state["K"] / state["var"]
        Hy = spectral_apply(state, H, y)
        u = -state["b"] - Hy
        g = state["gamma"] - ((state["b"] * y).sum(-1, keepdim=True)
            + 0.5 * ((y * Hy).sum(-1, keepdim=True) - state["K"].sum(-1, keepdim=True))) / alpha
    elif mode == "uniform_growth":
        u = state["dotxi"] + spectral_apply(state, state["dvar_minus_sigma2"] / (2 * state["var"]), y)
        g = state["gamma"]
    else:
        raise ValueError(mode)
    return u, g
