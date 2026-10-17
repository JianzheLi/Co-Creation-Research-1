"""Read-only comparison of saved RUOT/WFR semi-couplings on identical points."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure, load_config, file_hash
from src.distance_cost_table import DistanceHermiteGrid


def cluster(x):
    fitted = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x)
    # Spatial grouping only: cluster 0 is the center nearer the origin.
    near = np.linalg.norm(fitted.cluster_centers_, axis=1).argmin()
    return (fitted.labels_ != near).astype(int)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ruot-run', required=True)
    parser.add_argument('--wfr-run', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    runs = [Path(args.ruot_run), Path(args.wfr_run)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg, wcfg = [load_config(run/'config.yaml') for run in runs]
    for key in ('data_path', 'model_times', 'model_sigma', 'growth_penalty'):
        assert cfg[key] == wcfg[key], key
    assert cfg['dataset'] == 'official_deepruot_simulation_gene'
    device = configure(cfg)
    manifests = [json.loads((run/'couplings/manifest.json').read_text(encoding='utf-8')) for run in runs]
    blocks = [{(r['interval'],r['block']):r for r in m['intervals']} for m in manifests]
    assert blocks[0].keys() == blocks[1].keys()
    saved = torch.load(runs[0]/'distance_cost_table.pt',map_location=device,weights_only=False)
    assert saved['path_checkpoint_sha256'] == file_hash(runs[0]/'pair_path.pt')
    assert manifests[0]['table_sha256'] == file_hash(runs[0]/'distance_cost_table.pt')
    grid = DistanceHermiteGrid(saved['distances'],saved['ratios'],saved['cost'],saved['slope'])
    tensor = lambda x: torch.as_tensor(x,dtype=torch.float64,device=device)
    rows,details = [],[]
    fig,axes = plt.subplots(len(blocks[0]),3,figsize=(13,3.4*len(blocks[0])),squeeze=False)
    for k,key in enumerate(sorted(blocks[0])):
        arrays=[]
        for run,block in zip(runs,blocks):
            with np.load(run/'couplings'/block[key]['file']) as f:
                arrays.append({name:f[name] for name in f.files})
        for field in ('source_indices','target_indices','source_positions','target_positions','source_weights','target_weights'):
            np.testing.assert_array_equal(arrays[0][field],arrays[1][field])
        x,y = [tensor(arrays[0][name]) for name in ('source_positions','target_positions')]
        a,b = [tensor(arrays[0][name]) for name in ('source_weights','target_weights')]
        A,B = a.sum(),b.sum()
        ps,qs = [[tensor(data[name]) for data in arrays] for name in ('P','Q')]
        rs = [tensor(data['r']) for data in arrays]
        for p,q,r in zip(ps,qs,rs):
            assert torch.isfinite(p).all() and torch.isfinite(q).all() and torch.isfinite(r).all()
            assert (p>=0).all() and (q>=0).all() and (r>0).all()
            torch.testing.assert_close(p.sum(1),a,atol=1e-10,rtol=1e-8)
            torch.testing.assert_close(q.sum(0),b,atol=1e-10,rtol=1e-8)
            torch.testing.assert_close(p*r,q,atol=1e-12,rtol=1e-8)
        d = torch.cdist(x,y)
        query = grid.bind(d)
        row = dict(interval=key[0],block=key[1],source_points=len(a),target_points=len(b),
                   P_TV=float((ps[0]-ps[1]).abs().sum()/(2*A)),
                   Q_TV=float((qs[0]-qs[1]).abs().sum()/(2*B)))
        pb = [p@y/a[:,None] for p in ps]
        qb = [q.T@x/b[:,None] for q in qs]
        row['P_target_barycenter_rms']=float(((a/A)*(pb[0]-pb[1]).square().sum(-1)).sum().sqrt())
        row['Q_source_barycenter_rms']=float(((b/B)*(qb[0]-qb[1]).square().sum(-1)).sum().sqrt())
        row['row_growth_ratio_rms']=float(((a/A)*((qs[0].sum(1)-qs[1].sum(1))/a).square()).sum().sqrt())
        common=(ps[0]+ps[1])/(2*A)
        row['pair_log_ratio_rms_common_P']=float((common*(rs[0].log()-rs[1].log()).square()).sum().sqrt())
        sx,sy=[cluster(arrays[0][name]) for name in ('source_positions','target_positions')]
        coarse={}
        for method,p,q,r,bc in zip(('RUOT','WFR'),ps,qs,rs,pb):
            prob=p/a[:,None]
            entropy=-(torch.special.xlogy(prob,prob)).sum(1)
            row[method+'_effective_targets']=float(((a/A)*entropy).sum().exp())
            conditional_var=(p*y.square().sum(-1)[None,:]).sum(1)/a-bc.square().sum(-1)
            row[method+'_conditional_target_spread_rms']=float(((a/A)*conditional_var).sum().clamp_min(0).sqrt())
            row[method+'_mean_distance_P']=float((p*d).sum()/A)
            row[method+'_mean_distance_Q']=float((q*d).sum()/B)
            assert (r>=grid.ratios[0]).all() and (r<=grid.ratios[-1]).all()
            c=query.value(r)
            pair=(p*c).sum()
            entropy_term=cfg['model_sigma']**2*(torch.special.xlogy(p,p)-p).sum()
            kappa=(d/(2*cfg['growth_penalty']**.5)).clamp(max=np.pi/2).cos()
            row[method+'_RUOT_pair_cost']=float(pair)
            row[method+'_RUOT_entropy_term']=float(entropy_term)
            row[method+'_RUOT_objective']=float(pair+entropy_term)
            row[method+'_WFR_static_cost']=float((2*cfg['growth_penalty']*(p+q-2*kappa*(p*q).sqrt())).sum())
            coarse[method]={}
            for name,matrix,mass in [('P',p,A),('Q',q,B)]:
                summary=torch.zeros((2,2),device=device,dtype=torch.float64)
                for i in range(2):
                    for j in range(2):
                        summary[i,j]=matrix[tensor(sx==i).bool()][:,tensor(sy==j).bool()].sum()/mass
                coarse[method][name]=summary.cpu().tolist()
        for name in ('P','Q'):
            row[name+'_two_cluster_TV']=float(np.abs(np.array(coarse['RUOT'][name])-np.array(coarse['WFR'][name])).sum()/2)
        rows.append(row)
        details.append(dict(interval=key[0],block=key[1],coarse_normalized_couplings=coarse,
                            source_cluster_counts=np.bincount(sx).tolist(),target_cluster_counts=np.bincount(sy).tolist()))
        ix=np.lexsort((np.linalg.norm(arrays[0]['source_positions'],axis=1),sx))
        iy=np.lexsort((np.linalg.norm(arrays[0]['target_positions'],axis=1),sy))
        # Reordering is only for the plot; both methods use exactly the same order.
        for col,(method,matrix) in enumerate(zip(('RUOT','WFR'),ps)):
            plot=(matrix/a[:,None]).cpu().numpy()[ix][:,iy]
            im=axes[k,col].imshow(plot,aspect='auto',norm=LogNorm(vmin=1e-5,vmax=1),cmap='viridis')
            axes[k,col].set_title(f'{key[0]} to {key[0]+1}: {method} P(target | source)')
            axes[k,col].set_xlabel('Target (common spatial ordering)')
            axes[k,col].set_ylabel('Source (common spatial ordering)')
            fig.colorbar(im,ax=axes[k,col],fraction=.046,pad=.04)
        axes[k,2].axis('off')
        axes[k,2].text(.02,.95,
            f"P total variation: {row['P_TV']:.1%}\nQ total variation: {row['Q_TV']:.1%}\n\n"
            f"P barycenter RMS: {row['P_target_barycenter_rms']:.4f}\n"
            f"Effective targets: {row['RUOT_effective_targets']:.1f} / {row['WFR_effective_targets']:.1f}\n\n"
            f"Two-cluster TV (P): {row['P_two_cluster_TV']:.2%}\n"
            f"Two-cluster TV (Q): {row['Q_two_cluster_TV']:.2%}",va='top',fontsize=12)
    fig.suptitle(f"Saved semi-couplings, alpha={cfg['growth_penalty']:g}; colors are row-conditional probabilities",fontsize=14)
    fig.tight_layout()
    fig.savefig(out/'coupling_comparison.png',dpi=150)
    plt.close(fig)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(alpha=cfg['growth_penalty'],sigma=cfg['model_sigma'],runs=[str(run.resolve()) for run in runs],
                input_manifest_sha256=[file_hash(run/'couplings/manifest.json') for run in runs],
                definitions={'P_TV':'sum(abs(P_ruot-P_wfr))/(2*sum(a))',
                             'Q_TV':'sum(abs(Q_ruot-Q_wfr))/(2*sum(b))',
                             'effective_targets':'exp(source-mass-weighted mean conditional Shannon entropy)',
                             'two_cluster':'KMeans(2) on shared observed positions; label 0 is center closer to origin',
                             'cross_objectives':'same RUOT learned cost table/entropy, or same analytic WFR static cost; these are not rollout Actions'},
                caveats=['WFR saved solver hit its 1000-iteration cap; it passed marginal and OET/semi-coupling consistency audits, not a fixed-point convergence certificate.',
                         'RUOT includes sigma^2 joint entropy on P; WFR has no joint entropy penalty.'],
                intervals=rows,cluster_details=details)
    columns = ['RUOT_pair_cost','RUOT_entropy_term','RUOT_objective','WFR_static_cost']
    totals = {method:{name:sum(row[method+'_'+name] for row in rows) for name in columns}
              for method in ('RUOT','WFR')}
    totals['RUOT_minus_WFR'] = {name:totals['RUOT'][name]-totals['WFR'][name] for name in columns}
    result['sum_over_intervals_cross_objectives'] = totals
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
