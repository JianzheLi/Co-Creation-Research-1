"""Diagnose the existing bound, without training or changing the solver.

At fixed source/time, normalize the mixed mass to p=sum a_j phi_j.
L=log(p/q), ell_j=log(phi_j/q), q=N(x0,sigma^2*t I).
The spatial posterior correction is
 R= sigma^2 E[g(ell-L)] - Var(eta|X)/2 - alpha Var(g|X)/2.
The existing tight bound has exact gap
 B-R = alpha/2 E[(G+sigma^2 L/alpha)^2]
       +alpha E Var(g|X)+ E Var(eta|X)/2
       +sigma^4/(2 alpha) [sum a E ell_+^2 +4/e^2-E L^2].
All expectations here include X. The last bracket is nonnegative.
We also report the old full-log-square version, using closed Gaussian moments.
For the data-scaled reference and lambda=sigma^2, the full-square gap is
 sigma^2/2 E[(G+L)^2] + (alpha+sigma^2)/2 E Var(g|X)
 + E Var(eta|X)/2 + sigma^2/2 [sum a E ell^2+4/e^2-E L^2].
The gap identity is checked with common Monte Carlo moments; all reported
candidate bounds retain analytic Gaussian moments. The common-base F+R
diagnostic uses receiver conditional entropy, not the original source entropy.
This is a discrete mixture diagnostic, NOT an Action upper-bound certificate.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure, load_config, load_path, json_write, file_hash, snapshots_for
from src.tight_bound import log_ratio_moments, positive_log_moment
from src.pair_cost import finite_pair_integrand

ROOT = Path(__file__).resolve().parents[1]


def apply(axis, eigen, y):
    return eigen[..., -1:] * y + (eigen[..., :1]-eigen[..., -1:]) * axis * (axis*y).sum(-1, keepdim=True)


@torch.no_grad()
def audit(run, seed, time_nodes, rows, points):
    cfg = load_config(run/'config.yaml')
    device = configure(cfg)
    assert device.type == 'cuda'
    path = load_path(run, cfg, device)
    sigma, alpha, dim = path.sigma, path.alpha, path.dimension
    snapshots = snapshots_for(cfg)
    second_moments = []
    for source, target in zip(snapshots[:-1], snapshots[1:]):
        x, y = [torch.as_tensor(v.x, device=device, dtype=torch.float64) for v in (source, target)]
        a0, a1 = [torch.as_tensor(v.weights/v.mass, device=device, dtype=torch.float64) for v in (source, target)]
        second_moments.append(((a0*x.square().sum(-1)).sum()+(a1*y.square().sum(-1)).sum()
            -2*((a0[:, None]*x).sum(0)*(a1[:, None]*y).sum(0)).sum())/dim)
    reference_second_moment = torch.stack(second_moments).mean()
    gen = torch.Generator(device=device).manual_seed(seed)
    gx, gw = np.polynomial.legendre.leggauss(time_nodes)
    z = (gx+1)/2
    ts, tw = 1-(1-z)**2, gw*(1-z)
    names = ['correction_spatial', 'correction_eta', 'correction_growth',
             'true_spatial_remainder', 'bound_full', 'bound_tight',
             'slack_young', 'slack_growth_variance', 'slack_eta_variance',
             'slack_logmoment_full', 'slack_logmoment_tight',
             'extra_growth_penalty', 'cross_g_log', 'logmoment_penalty_full',
             'integrated_mass', 'identity_residual', 'analytic_mc_cross_error',
             'candidate_bound', 'candidate_young_slack', 'candidate_logmoment_slack',
             'bridge_reference_bound_alpha', 'bridge_reference_bound_diffusion',
             'renormalized_running', 'diffusion_young_slack',
             'diffusion_growth_variance_slack', 'diffusion_eta_variance_slack',
             'diffusion_logmoment_slack', 'diffusion_identity_residual',
             'diffusion_mc_cross_error']
    total = torch.zeros(len(names), device=device, dtype=torch.float64)
    endpoint_base = torch.zeros((), device=device, dtype=torch.float64)
    source_entropy = torch.zeros_like(endpoint_base)
    receiver_entropy = torch.zeros_like(endpoint_base)
    files = sorted((run/'couplings').glob('interval*.npz'))
    for file in files:
        with np.load(file) as a:
            D = {k: torch.as_tensor(a[k], dtype=torch.float64, device=device)
                 for k in ['P', 'Q', 'source_weights', 'source_positions', 'target_positions']}
        P, Q, a, x0, x1 = [D[k] for k in ['P', 'Q', 'source_weights', 'source_positions', 'target_positions']]
        if (P <= 0).any() or (Q <= 0).any():
            raise ValueError('Positive coupling support required for this diagnostic')
        endpoint_base += (.5*P*(x1[None]-x0[:, None]).square().sum(-1)).sum()-dim*sigma**2/2*Q.sum()
        source_entropy += sigma**2*(P*(P.log()-1)).sum()
        nu = Q.sum(0)/Q.sum()
        receiver_entropy += sigma**2*(Q*(Q.log()-Q.sum(1).log()[:, None]-nu.log()[None])).sum()
        ids = torch.multinomial(a, rows, replacement=True, generator=gen)
        uniform = torch.rand(rows, points, dtype=P.dtype, device=device, generator=gen)
        noise = torch.randn(rows, points, dim, dtype=P.dtype, device=device, generator=gen)
        # All arrays remain on the GPU, including reductions across sources/time.
        for t, weight in zip(ts, tw):
            for start in range(0, rows, 2):
                ii = ids[start:start+2]
                R, J = len(ii), P.shape[1]
                delta = x1[None]-x0[ii, None]
                lr = (Q[ii]/P[ii]).log()
                s = path(torch.full((R*J, 1), t, dtype=P.dtype, device=device),
                         delta.reshape(-1, dim), lr.reshape(-1, 1))
                vec = lambda key: s[key].reshape(R, J, dim)
                scalar = lambda key: s[key].reshape(R, J)
                mu, var, axis, gamma = vec('xi'), vec('var'), vec('axis'), scalar('gamma')
                if path.cost_objective in {'tight_logmoment_upper_bound', 'gaussian_reference_upper_bound', 'diffusion_full_square_upper_bound'}:
                    drift, dv, beta, zeta = [vec(k) for k in
                        ['drift_mean', 'drift_times_var', 'growth_linear', 'growth_quadratic']]
                else:
                    drift, dv, beta, zeta = -vec('b'), -vec('K'), -vec('b')/alpha, -vec('K')/alpha
                W = P[ii]*scalar('mass')
                mass = W.sum(1)
                prior = W/mass[:, None]
                cdf = prior.cumsum(1)
                cdf[:, -1] = 1
                ix = torch.searchsorted(cdf.contiguous(), uniform[start:start+R].contiguous()).clamp_max(J-1)
                row = torch.arange(R, device=device)[:, None]
                x = mu[row, ix] + apply(axis[row, ix], var[row, ix].sqrt(), noise[start:start+R])
                y = x[:, :, None]-mu[:, None]
                logphi = -.5*(var.log().sum(-1)[:, None]+dim*math.log(2*math.pi)
                              +(y*apply(axis[:, None], 1/var[:, None], y)).sum(-1))
                logmix = torch.logsumexp(prior.log()[:, None]+logphi, -1)
                post = (prior.log()[:, None]+logphi-logmix[:, :, None]).exp()
                logq = -.5*(dim*math.log(2*math.pi*sigma**2*t)+x.square().sum(-1)/(sigma**2*t))
                L = logmix-logq
                ell = logphi-logq[:, :, None]
                g = gamma[:, None]+(beta[:, None]*y).sum(-1)+.5*((y*apply(axis[:, None], zeta[:, None]/var[:, None], y)).sum(-1)-zeta.sum(-1)[:, None])
                eta = (drift-mu/t)[:, None]+apply(axis[:, None], (dv-var/t+sigma**2)[:, None]/var[:, None], y)
                G = (post*g).sum(-1)
                eta_bar = (post[..., None]*eta).sum(-2)
                vg = (post*(g-G[:, :, None]).square()).sum(-1).mean(-1)
                ve = (post*(eta-eta_bar[:, :, None]).square().sum(-1)).sum(-1).mean(-1)
                cross_mc = (post*g*ell).sum(-1).mean(-1)
                spatial = sigma**2*(post*g*(logphi-logmix[:, :, None])).sum(-1).mean(-1)
                rem = spatial-.5*ve-.5*alpha*vg
                kl, full = log_ratio_moments(s, sigma)
                pos = positive_log_moment(s, sigma, nodes=64)
                full, pos, kl = [v.reshape(R, J) for v in (full, pos, kl)]
                eg2 = gamma.square()+(beta*apply(axis, var, beta)).sum(-1)+.5*zeta.square().sum(-1)
                egell = gamma*kl+(beta*apply(axis, var, mu/(sigma**2*t))).sum(-1)+.5*(zeta*(var/(sigma**2*t)-1)).sum(-1)
                avg = lambda v: (prior*v).sum(-1)
                growth = .5*alpha*avg(eg2)
                cross = sigma**2*avg(egell)
                logfull = sigma**4/(2*alpha)*(avg(full)+4/math.e**2)
                logpos = sigma**4/(2*alpha)*(avg(pos)+4/math.e**2)
                bf, bt = growth+cross+logfull, growth+cross+logpos
                sy = .5*alpha*(G+sigma**2/alpha*L).square().mean(-1)
                sg, se = alpha*vg, .5*ve
                lf = logfull-sigma**4/(2*alpha)*L.square().mean(-1)
                lt = logpos-sigma**4/(2*alpha)*L.square().mean(-1)
                # Gap identity uses the MC pair cross and second moment for an
                # algebraic check; analytic/MC discrepancy is reported separately.
                bmc = .5*alpha*(post*g.square()).sum(-1).mean(-1)+sigma**2*cross_mc+logpos
                identity = bmc-rem-(sy+sg+se+lt)
                # Fixed diffusion-scale Young inequality, lambda=sigma^2 for
                # the unit interval. Reference = Brownian motion plus a random
                # ballistic displacement with observed second moment. No fit.
                qvar = sigma**2*t+reference_second_moment*t*t
                relative = var/qvar
                ckl = .5*((qvar.log()-var.log()+relative-1).sum(-1)+mu.square().sum(-1)/qvar)
                csecond = ckl.square()+(mu*apply(axis, var, mu)).sum(-1)/qvar.square()+.5*(relative-1).square().sum(-1)
                ccross = gamma*ckl+(beta*apply(axis, var, mu/qvar)).sum(-1)+.5*(zeta*(relative-1)).sum(-1)
                cb = .5*sigma**2*(avg(eg2)+2*avg(ccross)+avg(csecond)+4/math.e**2)
                cq = -.5*(dim*(math.log(2*math.pi)+qvar.log())+x.square().sum(-1)/qvar)
                cL = logmix-cq
                csy = .5*sigma**2*(G+cL).square().mean(-1)
                clm = .5*sigma**2*(avg(csecond)+4/math.e**2-cL.square().mean(-1))
                bq = sigma**2*t*(1-t)+reference_second_moment*t*t
                br = var/bq
                bk = .5*((bq.log()-var.log()+br-1).sum(-1)+mu.square().sum(-1)/bq)
                bs = bk.square()+(mu*apply(axis, var, mu)).sum(-1)/bq.square()+.5*(br-1).square().sum(-1)
                bc = gamma*bk+(beta*apply(axis, var, mu/bq)).sum(-1)+.5*(zeta*(br-1)).sum(-1)
                bba = .5*alpha*avg(eg2)+sigma**2*avg(bc)+sigma**4/(2*alpha)*(avg(bs)+4/math.e**2)
                bbd = .5*sigma**2*(avg(eg2)+2*avg(bc)+avg(bs)+4/math.e**2)
                b_logq = -.5*(dim*(math.log(2*math.pi)+bq.log())+x.square().sum(-1)/bq)
                bL = logmix-b_logq
                bell = logphi-b_logq[:, :, None]
                b_cross_mc = (post*g*bell).sum(-1).mean(-1)
                bsy = .5*sigma**2*(G+bL).square().mean(-1)
                bsg, bse = .5*(alpha+sigma**2)*vg, .5*ve
                blm = .5*sigma**2*(avg(bs)+4/math.e**2-bL.square().mean(-1))
                bmc = .5*sigma**2*((post*g.square()).sum(-1).mean(-1)+2*b_cross_mc+avg(bs)+4/math.e**2)
                bidentity = bmc-rem-(bsy+bsg+bse+blm)
                _, parts = finite_pair_integrand(s,sigma,alpha,path.control_mode,return_terms=True)
                ren = sum(value for key,value in parts.items() if key!='bound_correction')
                ren = avg((ren/s['mass']).reshape(R,J))
                values = torch.stack([spatial, -.5*ve, -.5*alpha*vg, rem,
                    bf, bt, sy, sg, se, lf, lt, growth, cross, logfull,
                    torch.ones_like(mass), identity, sigma**2*(avg(egell)-cross_mc), cb, csy, clm, bba, bbd,
                    ren,bsy,bsg,bse,blm,bidentity,sigma**2*(avg(bc)-b_cross_mc)], -1)
                factor = mass/a[ii]*a.sum()/rows*float(weight)
                total += (values*factor[:, None]).sum(0)
    ans = dict(zip(names, total.cpu().tolist()))
    ans.update(run=run.name, alpha=alpha, sigma=sigma, seed=seed,
               time_nodes=time_nodes, rows_per_block=rows, points=points,
               pair_checkpoint_sha256=file_hash(run/'pair_path.pt'),
               objective=path.cost_objective)
    ans['reference_second_moment'] = float(reference_second_moment)
    ans['pair_inner_product'] = float(endpoint_base)+ans['renormalized_running']
    ans['source_entropy'] = float(source_entropy)
    ans['receiver_conditional_entropy'] = float(receiver_entropy)
    ans['common_base_receiver'] = ans['pair_inner_product']+float(receiver_entropy)
    ans['diffusion_upper_objective'] = ans['common_base_receiver']+ans['bridge_reference_bound_diffusion']
    ans['posterior_corrected_proxy'] = ans['common_base_receiver']+ans['true_spatial_remainder']
    ans['diffusion_gap'] = ans['bridge_reference_bound_diffusion']-ans['true_spatial_remainder']
    ans['diagnostic_note'] = ('Discrete-label posterior estimate. The common-base proxy includes receiver conditional '
        'entropy and omits candidate-independent endpoint constants. It is not the learned rollout Action '
        'or a numerical certificate of the continuous-endpoint identity. Pair integral uses sampled source rows.')
    ans['candidate_gap'] = ans['candidate_bound']-ans['true_spatial_remainder']
    ans['gap_tight'] = ans['bound_tight']-ans['true_spatial_remainder']
    ans['gap_full'] = ans['bound_full']-ans['true_spatial_remainder']
    return ans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['gene', 'mouse', 'emt'])
    p.add_argument('--settings', nargs='+', default=['', '_alpha2_sigma1'])
    p.add_argument('--methods', nargs='+', default=['distance_hermite_primal', 'tight_bound'])
    p.add_argument('--time-nodes', type=int, default=16)
    p.add_argument('--rows', type=int, default=4)
    p.add_argument('--points', type=int, default=64)
    p.add_argument('--seed', type=int, default=6100)
    p.add_argument('--tag', default='screen')
    p.add_argument('--runs', nargs='+', help='Explicit run directories instead of the historical default list')
    args = p.parse_args()
    out = ROOT/'../Results/bound_slack_audit'
    out.mkdir(exist_ok=True)
    runs = [Path(r) for r in args.runs] if args.runs else [ROOT/'../Results'/f'{dataset}_{method}{setting}_seed42'
        for setting in args.settings for dataset in args.datasets for method in args.methods]
    for run in runs:
        begin = time.perf_counter()
        ans = audit(run, args.seed, args.time_nodes, args.rows, args.points)
        ans['seconds'] = time.perf_counter()-begin
        json_write(out/f'{run.name}_{args.tag}.json', ans)
        print(json.dumps(ans), flush=True)


if __name__ == '__main__':
    main()
