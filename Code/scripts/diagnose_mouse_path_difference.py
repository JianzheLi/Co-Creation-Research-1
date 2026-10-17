"""Read-only paired path diagnostic on conditions sampled from fitted RUOT P."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure, load_config, load_path, snapshots_for
from src.matching import CouplingSampler
from src.wfr_heuristic import WFRBridgePath
from src.pair_cost import finite_pair_integrand


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='../Results/mouse_distance_hermite_primal_seed42')
    args = parser.parse_args()
    run = Path(args.run_dir)
    cfg = load_config(run/'config.yaml')
    device = configure(cfg)
    path = load_path(run, cfg, device)
    sampler = CouplingSampler(run, cfg, device)
    generator = torch.Generator(device=device).manual_seed(8271)
    batch = sampler.sample(8192, device, generator)
    bridge = WFRBridgePath(path.alpha, path.sigma)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    tq = torch.tensor((nodes+1)/2, device=device, dtype=torch.float64)
    wq = torch.tensor(weights/2, device=device, dtype=torch.float64)
    totals = {k: [] for k in ('mean_shift', 'mass_relative', 'variance_relative',
                              'C_ruot', 'C_wfr_minimal', 'C_wfr_uniform', 'analytic_inner_gain')}
    for start in range(0,8192,256):
        delta = batch['delta'][start:start+256].repeat_interleave(64,0)
        ell = batch['log_r'][start:start+256].repeat_interleave(64,0)
        t = tq.repeat(256).reshape(-1,1)
        s = path(t,delta,ell)
        h = bridge(t,torch.zeros_like(delta),delta,ell)
        tau = t*(1-t)
        v = h['var'].expand_as(delta)
        z = torch.zeros_like(delta)
        w = dict(t=t,tau=tau,delta=delta,log_r=ell,xi=h['xi'],dotxi=h['dotxi'],
                 mass=h['mass'],gamma=h['gamma'],var=v,Z=z,dotZ=z,h_var=z,
                 h_mu=(h['xi']-t*delta)/tau,
                 h_mass=(h['mass'].log()-t*ell)/tau,
                 b=-path.alpha*h['dotxi']/(path.alpha+v),
                 K=-path.alpha*(h['dotvar']-path.sigma**2)/(2*path.alpha+v),
                 dvar_minus_sigma2=(h['dotvar']-path.sigma**2).expand_as(delta))
        def integrate(y):
            return (y.reshape(256,64)*wq).sum(-1)
        totals['mean_shift'].append(integrate((s['xi']-h['xi']).square().sum(-1)))
        totals['mass_relative'].append(integrate((s['mass']/h['mass']-1).square()))
        totals['variance_relative'].append(integrate((s['var']/v-1).square().mean(-1)))
        inner_gain = .5*h['mass']*((v/(path.alpha+v)*h['dotxi'].square()).sum(-1,keepdim=True)
                     + ((h['dotvar']-path.sigma**2).square()/(4*(2*path.alpha+v))).sum(-1,keepdim=True))
        totals['analytic_inner_gain'].append(integrate(inner_gain))
        constant = .5*delta[::64].square().sum(-1)-path.dimension*path.sigma**2/2*ell[::64].exp().squeeze(-1)
        for key,state,mode in [('C_ruot',s,'quadratic_minimal'),
                               ('C_wfr_minimal',w,'quadratic_minimal'),
                               ('C_wfr_uniform',w,'uniform_growth')]:
            totals[key].append(constant+integrate(finite_pair_integrand(state,path.sigma,path.alpha,mode)))
    values = {k: torch.cat(v) for k,v in totals.items()}
    gain_error = float((values['C_wfr_uniform']-values['C_wfr_minimal']-values['analytic_inner_gain']).abs().max())
    assert gain_error < 1e-10, gain_error
    result = {'sampling':'8192 conditions from RUOT P, equal probability per interval; same conditions for both paths',
              'time_quadrature_nodes':64,
              'alpha':path.alpha, 'sigma':path.sigma,
              'analytic_inner_gain_max_error':gain_error,
              'note':'C is the finite renormalized pair surrogate, not learned feedback rollout Action',
              'intervals':[]}
    truth = snapshots_for(cfg)
    n_intervals = len(cfg['model_times'])-1
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_intervals,3,figsize=(12,3*n_intervals),squeeze=False)
    for k in range(n_intervals):
        mask = batch['interval']==k
        means = {name: float(v[mask].mean()) for name,v in values.items()}
        for name in ('mean_shift','mass_relative','variance_relative'):
            means[name+'_rms'] = means.pop(name)**.5
        means['fraction_ruot_lower_than_wfr_minimal'] = float((values['C_ruot'][mask]<values['C_wfr_minimal'][mask]).double().mean())
        means['outer_path_gain'] = means['C_wfr_minimal']-means['C_ruot']
        means['total_pair_gain'] = means['C_wfr_uniform']-means['C_ruot']
        means['interval']=k
        quantiles = torch.tensor([.05,.5,.95], device=device, dtype=torch.float64)
        means['ratio_quantiles_05_50_95'] = torch.quantile(batch['log_r'][mask].exp(),quantiles).tolist()
        means['distance_quantiles_05_50_95'] = torch.quantile(batch['delta'][mask].norm(dim=-1),quantiles).tolist()
        means['growth_only_lower_bound_observed_masses']=2*path.alpha*(truth[k+1].mass**.5-truth[k].mass**.5)**2
        dd = batch['delta'][mask]
        rr = batch['log_r'][mask]
        distances = dd.norm(dim=-1)
        proximity = ((distances-distances.median())/distances.std().clamp_min(1e-8)).square()
        proximity += ((rr.flatten()-rr.median())/rr.std().clamp_min(1e-8)).square().flatten()
        index = proximity.argmin()
        delta0, ell0 = dd[index:index+1], rr[index:index+1]
        tt = torch.linspace(.001,.999,201,device=device,dtype=torch.float64)[:,None]
        sd = path(tt,delta0.expand(len(tt),-1),ell0.expand_as(tt))
        wd = bridge(tt,torch.zeros((len(tt),path.dimension),device=device,dtype=torch.float64),
                    delta0.expand(len(tt),-1),ell0.expand_as(tt))
        means['plotted_condition'] = {'distance':float(delta0.norm()),'ratio':float(ell0.exp().item())}
        time_cpu = tt.flatten().cpu().numpy()
        for col,key,title in [(0,'xi','Mean progress'),(1,'mass','Conditional mass'),(2,'var','Variance')]:
            a,b = sd[key],wd[key]
            if key == 'xi':
                a=(a*delta0).sum(-1)/delta0.square().sum()
                b=(b*delta0).sum(-1)/delta0.square().sum()
            if key == 'var':
                axes[k,col].plot(time_cpu,a[:,0].cpu(),label='Learned parallel')
                axes[k,col].plot(time_cpu,a[:,1].cpu(),label='Learned perpendicular')
            else:
                axes[k,col].plot(time_cpu,a.cpu(),label='Learned')
            axes[k,col].plot(time_cpu,b.cpu(),'--',label='WFR + bridge')
            axes[k,col].set_title(f'{k} to {k+1}: {title}')
            axes[k,col].set_xlabel('Local time')
            axes[k,col].legend(fontsize=8)
        axes[k,0].set_ylabel(f'd={float(delta0.norm()):.3f}, r={float(ell0.exp().item()):.3f}')
        result['intervals'].append(means)
    result['weighted_pair_surrogates_on_ruot_coupling'] = {
        key:sum(truth[k].mass*row[key] for k,row in enumerate(result['intervals']))
        for key in ('C_ruot','C_wfr_minimal','C_wfr_uniform','analytic_inner_gain','outer_path_gain','total_pair_gain')}
    fig.suptitle(f'Conditional paths at representative RUOT coupling pairs; alpha={path.alpha:g}')
    fig.tight_layout()
    fig.savefig(run/'conditional_path_comparison.png',dpi=150)
    plt.close(fig)
    (run/'conditional_path_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
