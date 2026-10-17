"""Independent distance/ratio grid, no coupling weights or matching network."""
import sys,json,argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config,configure,load_path,json_write,file_hash
from src.wfr_heuristic import WFRBridgePath
from src.pair_cost import finite_pair_integrand
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='../Results/gene_distance_hermite_primal_seed42')
    parser.add_argument('--output-dir',default='../Results/standalone_path_diagnostic')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];run=(root/args.run_dir).resolve()
    out=(root/args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(run/'config.yaml');device=configure(cfg);path=load_path(run,cfg,device);bridge=WFRBridgePath(path.alpha,path.sigma)
    training=[json.loads(line) for line in (run/'pair_steps.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()][-1]
    max_distance=0.
    for f in (run/'couplings').glob('interval*.npz'):
        with np.load(f) as a:
            x=torch.as_tensor(a['source_positions'],dtype=torch.float64,device=device);y=torch.as_tensor(a['target_positions'],dtype=torch.float64,device=device)
            max_distance=max(max_distance,float(torch.cdist(x,y).max()))
    assert max_distance<np.pi*np.sqrt(path.alpha)
    distances=torch.linspace(0,max_distance,41,device=device,dtype=torch.float64)
    ratios=torch.linspace(.1,10,41,device=device,dtype=torch.float64)
    dd,rr=torch.meshgrid(distances,ratios,indexing='ij');flat_d=dd.flatten();flat_r=rr.flatten()
    results={}
    for nodes in (64,128):
        tx,tw=np.polynomial.legendre.leggauss(nodes)
        tq=torch.tensor((tx+1)/2,device=device);wq=torch.tensor(tw/2,device=device)
        batches={k:[] for k in ('mean_rms','mass_relative_rms','variance_relative_rms','C_learned','C_wfr_same_control','C_wfr_uniform')}
        for start in range(0,len(flat_d),128):
            ds=flat_d[start:start+128];rs=flat_r[start:start+128];n=len(ds)
            delta=torch.zeros(n,path.dimension,device=device,dtype=torch.float64);delta[:,0]=ds
            delta=delta.repeat_interleave(nodes,0);ell=rs.log().repeat_interleave(nodes)[:,None];t=tq.repeat(n)[:,None]
            s=path(t,delta,ell);h=bridge(t,torch.zeros_like(delta),delta,ell)
            v=h['var'].expand_as(delta);z=torch.zeros_like(delta);tau=t*(1-t)
            w=dict(t=t,tau=tau,delta=delta,log_r=ell,xi=h['xi'],dotxi=h['dotxi'],mass=h['mass'],gamma=h['gamma'],var=v,Z=z,dotZ=z,h_var=z,
                h_mu=(h['xi']-t*delta)/tau,h_mass=(h['mass'].log()-t*ell)/tau,b=-path.alpha*h['dotxi']/(path.alpha+v),
                K=-path.alpha*(h['dotvar']-path.sigma**2)/(2*path.alpha+v),dvar_minus_sigma2=(h['dotvar']-path.sigma**2).expand_as(delta))
            integrate=lambda a:(a.reshape(n,nodes)*wq).sum(-1)
            for k,a in [('mean_rms',(s['xi']-h['xi']).square().sum(-1)),('mass_relative_rms',(s['mass']/h['mass']-1).square()),('variance_relative_rms',(s['var']/v-1).square().mean(-1))]:
                batches[k].append(integrate(a).sqrt())
            const=.5*ds.square()-path.dimension*path.sigma**2/2*rs
            for k,state,mode in [('C_learned',s,'quadratic_minimal'),('C_wfr_same_control',w,'quadratic_minimal'),('C_wfr_uniform',w,'uniform_growth')]:
                batches[k].append(const+integrate(finite_pair_integrand(state,path.sigma,path.alpha,mode)))
        results[nodes]={k:torch.cat(v).cpu().numpy() for k,v in batches.items()}
    val=results[128];errors={k:float(np.max(np.abs(results[64][k]-v))) for k,v in val.items()}
    gain=val['C_wfr_same_control']-val['C_learned']
    summary=dict(run=str(run),checkpoint_sha256=file_hash(run/'pair_path.pt'),alpha=path.alpha,sigma=path.sigma,steps=training['step'],condition_time_samples=training['condition_time_samples'],
        grid=dict(distance=[0,max_distance],ratio=[.1,10],shape=[41,41],weighting='Uniform grid in distance and mass ratio; no coupling weighting'),
        quadrature_64_vs_128=errors,
        metrics={k:dict(rms=float(np.sqrt(np.mean(v*v))),median=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=float(v.max())) for k,v in val.items() if k.endswith('rms')},
        costs={k:float(val[k].mean()) for k in val if k.startswith('C_')},
        fraction_learned_lower_same_control=float(np.mean(gain>0)),mean_cost_gain_same_control=float(gain.mean()),
        gain_quantiles=np.quantile(gain,[0,.05,.5,.95,1]).tolist())
    np.savez_compressed(out/'grid.npz',distance=dd.cpu().numpy(),ratio=rr.cpu().numpy(),**{k:v.reshape(41,41) for k,v in val.items()})
    empirical=json.loads((run/'conditional_path_comparison.json').read_text(encoding='utf-8'))
    summary['empirical_diagnostic']=empirical
    json_write(out/'summary.json',summary)
    fig,axes=plt.subplots(1,4,figsize=(16,3.9),layout='constrained')
    for ax,v,title in zip(axes,[val['mean_rms'],100*val['mass_relative_rms'],100*val['variance_relative_rms'],gain],['Mean displacement RMS','Mass relative RMS (%)','Variance relative RMS (%)','Cost: WFR − learned (same control)']):
        im=ax.imshow(v.reshape(41,41),origin='lower',aspect='auto',extent=[.1,10,0,max_distance],cmap='coolwarm' if 'Cost' in title else 'viridis')
        ax.set_title(title,fontsize=10);ax.set_xlabel('Mass ratio r');ax.set_ylabel('Distance d');fig.colorbar(im,ax=ax,shrink=.85)
    fig.savefig(out/'grid.png',dpi=170);plt.close(fig)
    import shutil
    shutil.copyfile(run/'conditional_path_comparison.png',out/'representative_paths.png')
    rows=[]
    for row in empirical['intervals']:
        rows.append(f'<tr><td>{row["interval"]} → {row["interval"]+1}</td><td>{row["mean_shift_rms"]:.6f}</td><td>{row["mass_relative_rms"]:.2%}</td><td>{row["variance_relative_rms"]:.2%}</td><td>{row["C_ruot"]:.6f}</td><td>{row["C_wfr_minimal"]:.6f}</td></tr>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>单独诊断已训练条件路径</title><style>body{font:16px/1.8 system-ui;max-width:1350px;margin:30px auto;padding:0 20px;color:#223}img{width:100%}td,th{padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse;width:100%}</style><h1>已训练路径网络 vs WFR 均值、质量＋布朗桥方差</h1>
<p>MODEL_DESCRIPTION</p>
<p>独立网格覆盖数据端点对的完整距离范围及 r∈[0.1,10]，共 1681 个条件，使用 128 点时间积分，并与 64 点结果核对。两条路径在完全相同的条件上比较，不涉及不同 Coupling 或 Matching。WFR 路径同时评估与学习路径相同的兼容控制，以及原 WFR 的空间均匀生长控制，避免将控制差异误认为路径差异。比较的是有限成对 Cren，不是 rollout Action。</p><h2>完整距离–质量比网格</h2><img src="grid.png">
<h2>实际端点条件的补充诊断</h2><p>从已有 RUOT P 抽取 8192 个条件，时间区间等概率；对两条路径使用同一批条件，不重新求 Coupling。下表数值是对时间和条件的 RMS；质量、方差相对 WFR，均值为原数据坐标单位。</p><table><tr><th>区间</th><th>均值 RMS</th><th>质量相对 RMS</th><th>方差相对 RMS</th><th>学习路径 Cren</th><th>WFR 路径 Cren（同控制）</th></tr>'''+''.join(rows)+'''</table><img src="representative_paths.png"><p>图中每个区间选择接近该区间距离和质量比中位数的代表条件。协方差分别画出沿路径和垂直路径方向；WFR 使用各向同性布朗桥方差。</p><p><a href="summary.json">完整指标与模型哈希</a> · <a href="grid.npz">网格数值</a></p></html>'''
    page=page.replace('MODEL_DESCRIPTION',f'模型：{run.relative_to(root)}/pair_path.pt；原删项目标，α={path.alpha:g}、σ={path.sigma:g}。已完成 {training["step"]} 步预训练，累计 {training["condition_time_samples"]} 个条件–时间样本。达到既定训练步数不等同于证明全局最优。本次只作诊断，没有重训。')
    (out/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='empirical_diagnostic'},indent=2))


if __name__=='__main__':main()
