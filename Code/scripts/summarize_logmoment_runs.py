"""Compare the bound runs with archived alpha=10 RUOT and WFR baselines."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'../Results/logmoment_bound_comparison'
SPECS = {
    'gene': ('Simulation gene', 'gene_wfr_bridge_tanh_matching_seed42', 'action_monte_carlo_hermite_primal_vs_wfr_seed1000'),
    'mouse': ('Mouse', 'mouse_wfr_bridge_tanh_minibatch_seed42', 'action_monte_carlo_mouse_hermite_primal_vs_wfr_seed1000'),
    'emt': ('EMT', 'emt_wfr_bridge_seed42', 'action_monte_carlo_emt_seed1000'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]; detail={}
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    colors=['#5b6f96','#d97a35','#449e87']
    for ax,(dataset,(title,wfr,archive)) in zip(axes,SPECS.items()):
        runs=[f'{dataset}_distance_hermite_primal_seed42',f'{dataset}_logmoment_bound_seed42',wfr]
        old=read(ROOT/'../Results'/archive/'summary.json')
        new=read(ROOT/'../Results'/f'action_monte_carlo_{dataset}_logmoment_seed1000'/'summary.json')
        assert old['alpha']==new['alpha']==10 and old['sigma']==new['sigma']==.25
        assert old['mc_seed_range']==new['mc_seed_range']==[1000,1019]
        for key in ('steps_per_unit','particles_per_replicate','rollout_seed_offset'):
            assert old[key]==new[key],key
        assert old['feedback_checkpoint_sha256']['WFR-FM bridge']==new['feedback_checkpoint_sha256']['WFR-FM bridge']
        for key,run in [('RUOT',runs[0]),('WFR-FM bridge',wfr)]:
            assert hashlib.sha256((ROOT/'../Results'/run/'feedback.pt').read_bytes()).hexdigest()==old['feedback_checkpoint_sha256'][key]
        values=[np.array([x['methods'][key]['total_action'] for x in source['replicates']])
                for source,key in [(old,'RUOT'),(new,'RUOT'),(new,'WFR-FM bridge')]]
        for i,(label,run,v) in enumerate(zip(['Original RUOT','Bound RUOT','WFR bridge'],runs,values)):
            status=read(ROOT/'../Results'/run/'status.json')
            assert status['evaluation']['status']=='completed'
            metrics=pd.read_csv(ROOT/'../Results'/run/'metrics.csv')
            metrics=metrics[metrics.time>0]
            rows.append(dict(dataset=title,method=label,action_mean=v.mean(),action_std=v.std(ddof=1),
                mean_W1=metrics.W1_normalized.mean(),max_mass_relative_error=metrics.mass_relative_error.max(),
                run=run))
            bp=ax.boxplot(v,positions=[i+1],widths=.45,patch_artist=True,showfliers=False)
            bp['boxes'][0].set(facecolor=colors[i],alpha=.3)
            ax.scatter(i+1+np.linspace(-.12,.12,len(v)),v,s=14,color=colors[i],alpha=.8,zorder=3)
        diff=values[1]-values[0]
        detail[dataset]=dict(bound_minus_original_mean=float(diff.mean()),
            bound_minus_original_paired_standard_error=float(diff.std(ddof=1)/np.sqrt(len(diff))),
            bound_minus_original_percent=float(100*diff.mean()/values[0].mean()),
            seeds=[1000,1019])
        ax.set(title=title,xticks=[1,2,3],xticklabels=['Original','Bound','WFR'],ylabel='Feedback Action')
        ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'action_comparison.png',dpi=180)
    fig.savefig(OUT/'action_comparison.pdf')
    plt.close(fig)
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'comparison.csv',index=False)
    (OUT/'paired_differences.json').write_text(json.dumps(detail,indent=2),encoding='utf-8')
    intro='''<h1>Log-moment Bound：三个数据集对照</h1>
    <p>α=10，σ=0.25；新方法预训练3000步，Matching 30000步。每个已训练模型用20个相同Monte Carlo种子（1000–1019）评估反馈Action。这里的标准差是rollout随机性，不是重新训练的种子差异。</p>
    <p>Action采用相同的原始运行代价，未把新训练目标与旧目标直接比较。W1为非初始观测时刻的平均值，质量误差为这些时刻的最大相对误差；W1和质量来自单次标准rollout。</p>
    <p>新方法加入单路径log-moment修正，并将Coupling熵改为接收端KL。兼容控制、Matching损失和网络结构不变。该兼容控制对新目标可行，但不保证已最小化新目标的内层控制。连续上界推导不构成插值或神经网络rollout的数值上界认证。</p>'''
    html='<html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;line-height:1.6}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style>'+intro
    html+='<img src="action_comparison.png">'+table.drop(columns=['run']).to_html(index=False,float_format=lambda x:f'{x:.6g}')
    for dataset in SPECS:
        html+=f'<p><a href="../{dataset}_logmoment_bound_seed42/report.html">{dataset} 新方法完整报告与轨迹</a></p>'
    (OUT/'report.html').write_text(html+'</html>',encoding='utf-8')
    print(table.to_string(index=False))
    print(json.dumps(detail,indent=2))


if __name__=='__main__':
    main()
