"""Four-way comparison with checkpoint-verified paired MC archives."""
from summarize_logmoment_runs import ROOT,SPECS,read
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=ROOT/'../Results/tight_bound_comparison'; out.mkdir(exist_ok=True)
    rows=[]; differences={}
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,(dataset,(title,wfr,archive)) in zip(axes,SPECS.items()):
        sources=[read(ROOT/'../Results'/archive/'summary.json'),
                 read(ROOT/'../Results'/f'action_monte_carlo_{dataset}_logmoment_seed1000/summary.json'),
                 read(ROOT/'../Results'/f'action_monte_carlo_{dataset}_tight_bound_seed1000/summary.json')]
        runs=[f'{dataset}_distance_hermite_primal_seed42',f'{dataset}_logmoment_bound_seed42',f'{dataset}_tight_bound_seed42',wfr]
        vals=[]
        for i,(label,run,source,key,color) in enumerate(zip(
            ['Original RUOT','Previous bound','Tighter bound','WFR bridge'],runs,
            [sources[0],sources[1],sources[2],sources[2]],['RUOT','RUOT','RUOT','WFR-FM bridge'],
            ['#55749e','#d4914e','#ae5268','#489783'])):
            assert source['alpha']==10 and source['sigma']==.25 and source['mc_seed_range']==[1000,1019]
            assert source['steps_per_unit']==sources[0]['steps_per_unit']
            assert source['particles_per_replicate']==sources[0]['particles_per_replicate']
            assert source['feedback_checkpoint_sha256'][key]==hashlib.sha256((ROOT/'../Results'/run/'feedback.pt').read_bytes()).hexdigest()
            v=np.array([r['methods'][key]['total_action'] for r in source['replicates']]); vals.append(v)
            metrics=pd.read_csv(ROOT/'../Results'/run/'metrics.csv'); metrics=metrics[metrics.time>0]
            status=read(ROOT/'../Results'/run/'status.json'); assert status['evaluation']['status']=='completed'
            rows.append(dict(dataset=title,method=label,action_mean=v.mean(),action_std=v.std(ddof=1),
                mean_W1=metrics.W1_normalized.mean(),max_mass_relative_error=metrics.mass_relative_error.max(),
                pretraining_seconds=status.get('pair',{}).get('seconds'),coupling_seconds=status['couplings']['seconds'],
                matching_seconds=status['matching']['seconds'],run=run))
            bp=ax.boxplot(v,positions=[i+1],widths=.5,patch_artist=True,showfliers=False)
            bp['boxes'][0].set(facecolor=color,alpha=.3)
            ax.scatter(i+1+np.linspace(-.13,.13,len(v)),v,s=13,color=color,zorder=3)
        differences[dataset]={}
        for label,index in [('original',0),('previous_bound',1),('wfr',3)]:
            d=vals[2]-vals[index]
            differences[dataset][label]=dict(mean=float(d.mean()),percent=float(100*d.mean()/vals[index].mean()),
                paired_standard_error=float(d.std(ddof=1)/np.sqrt(len(d))))
        ax.set(title=title,xticks=[1,2,3,4],xticklabels=['Original','Prev. bound','Tight bound','WFR'],ylabel='Feedback Action')
        ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'action_comparison.png',dpi=170); fig.savefig(out/'action_comparison.pdf'); plt.close(fig)
    table=pd.DataFrame(rows); table.to_csv(out/'comparison.csv',index=False)
    (out/'paired_differences.json').write_text(json.dumps(differences,indent=2),encoding='utf-8')
    html='''<html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1250px;margin:35px auto;line-height:1.6}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style>
    <h1>更紧的 Bound：三个数据集完整实验</h1>
    <p>Young系数保持固定，没有自动调参。新版本保留负行KL，并用正部对数平方矩收紧界；路径预训练、成对代价及Matching均使用新目标下的解析最优控制。α=10、σ=0.25，预训练3000步，Matching 30000步。</p>
    <p>Action采用相同的原始反馈运行代价，20次配对Monte Carlo，种子1000–1019。标准差反映rollout随机性，不是重新训练的种子差异。原方法、上一版Bound和WFR使用已核验模型哈希的既有结果。W1为单次标准rollout的非初始观测时刻平均值。</p>
    <p>对同一路径、同一控制，新理论界不大于上一版；它仍不是所有上界中已知最紧的界，也不保证最终神经网络rollout的Action下降。数值积分、插值及Matching误差仍需区分。</p>
    <img src="action_comparison.png">'''
    html+=table.drop(columns=['run','pretraining_seconds','coupling_seconds','matching_seconds']).to_html(index=False,float_format=lambda v:f'{v:.6g}')
    html+='<h2>阶段耗时（并行运行的墙钟时间）</h2>'+table[['dataset','method','pretraining_seconds','coupling_seconds','matching_seconds']].to_html(index=False,float_format=lambda v:f'{v:.1f}')
    for dataset in SPECS:
        html+=f'<p><a href="../{dataset}_tight_bound_seed42/report.html">{dataset} 新版本完整报告与轨迹</a></p>'
    (out/'report.html').write_text(html+'</html>',encoding='utf-8')
    print(table.to_string(index=False)); print(json.dumps(differences,indent=2))


if __name__=='__main__': main()
