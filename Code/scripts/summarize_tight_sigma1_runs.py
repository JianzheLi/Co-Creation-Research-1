"""Compare the alpha=2, sigma=1 reruns using verified, paired MC estimates."""
from pathlib import Path
import hashlib
import html
import json
import os

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '../Results/tight_bound_alpha2_sigma1_comparison'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    rows, differences = [], {}
    fig, axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    links = []
    for ax, (dataset, title) in zip(axes, [('gene','Simulation gene'),('mouse','Mouse'),('emt','EMT')]):
        tight = read(ROOT/f'../Results/action_monte_carlo_{dataset}_tight_alpha2_sigma1_seed1000/summary.json')
        old = read(ROOT/f'../Results/action_monte_carlo_{dataset}_original_alpha2_sigma1_seed1000/summary.json')
        for source in [tight, old]:
            assert source['alpha']==2 and source['sigma']==1
            assert source['mc_seed_range']==[1000,1019]
            assert source['particles_per_replicate']==2000 and source['steps_per_unit']==200
        values = {}
        specs = [('Original RUOT','distance_hermite_primal',old,'RUOT','#55749e'),
                 ('Tighter bound','tight_bound',tight,'RUOT','#ae5268'),
                 ('WFR + BB','wfr_bridge',tight,'WFR-FM bridge','#489783')]
        for pos,(label,stem,source,key,color) in enumerate(specs,1):
            run=ROOT/f'../Results/{dataset}_{stem}_alpha2_sigma1_seed42'
            assert source['feedback_checkpoint_sha256'][key]==hashlib.sha256((run/'feedback.pt').read_bytes()).hexdigest()
            status=read(run/'status.json')
            assert status['evaluation']['status']=='completed'
            v=np.array([r['methods'][key]['total_action'] for r in source['replicates']])
            assert len(v)==20 and np.isfinite(v).all()
            values[label]=v
            metrics=pd.read_csv(run/'metrics.csv')
            metrics=metrics[metrics.time>0]
            summary=source['method_summaries'][key]
            rows.append(dict(dataset=title,method=label,action_mean=float(v.mean()),
                action_sd=float(v.std(ddof=1)),kinetic_mean=summary['kinetic_action']['mean'],
                growth_mean=summary['growth_action']['mean'],mean_W1=float(metrics.W1_normalized.mean()),
                mean_W2=float(metrics.W2_normalized.mean()),
                mean_mass_relative_error=float(metrics.mass_relative_error.mean()),
                max_mass_relative_error=float(metrics.mass_relative_error.max()),
                pretraining_seconds=status.get('pair',{}).get('seconds'),
                coupling_seconds=status['couplings']['seconds'],matching_seconds=status['matching']['seconds'],run=run.name))
            box=ax.boxplot(v,positions=[pos],widths=.5,patch_artist=True,showfliers=False)
            box['boxes'][0].set(facecolor=color,alpha=.3)
            ax.scatter(pos+np.linspace(-.13,.13,len(v)),v,s=13,color=color,zorder=3)
            links.append(f'<a href="../{run.name}/report.html">{title}: {html.escape(label)}</a>')
        # WFR is independently rerun in both comparisons using identical RNG seeds.
        old_wfr=np.array([r['methods']['WFR-FM bridge']['total_action'] for r in old['replicates']])
        np.testing.assert_allclose(old_wfr,values['WFR + BB'],rtol=1e-7,atol=1e-8)
        differences[dataset]={}
        for label in ['Original RUOT','WFR + BB']:
            delta=values['Tighter bound']-values[label]
            differences[dataset][label]=dict(mean=float(delta.mean()),
                percent=float(100*delta.mean()/values[label].mean()),
                paired_standard_error=float(delta.std(ddof=1)/np.sqrt(20)))
        ax.set(title=title,xticks=[1,2,3],xticklabels=['Original','Tighter bound','WFR + BB'],ylabel='Feedback RUOT Action')
        ax.grid(axis='y',alpha=.2)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/'action_comparison.png',dpi=180)
    fig.savefig(OUT/'action_comparison.pdf')
    plt.close(fig)
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'comparison.csv',index=False)
    (OUT/'comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'paired_differences.json').write_text(json.dumps(differences,indent=2),encoding='utf8')
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>更紧上界：α=2，σ=1</title>
    <style>body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;max-width:1350px;margin:30px auto;padding:0 20px;color:#24303b}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef2f7}img{width:100%}a{color:#1769aa}</style>
    <h1>更紧上界代价：α=2，σ=1，三个数据集完整重跑</h1>
    <p>使用 tight_logmoment_upper_bound：保留负行 KL、使用正部对数平方矩，Young 系数固定为 α。预训练和 Matching 使用该目标下的解析最优兼容控制。路径预训练 3000 步，Matching 30000 步，训练种子 42。Mouse 按数据默认顺序分 8 块；gene 与 EMT 使用完整矩阵。</p>
    <p>原方法与 WFR + 布朗桥使用之前相同参数的已训练模型，上界分支从路径预训练开始重跑。所有方法重新做 20 次配对 Monte Carlo（种子 1000–1019），每次 2000 粒子、每单位时间 200 步。Action 是反馈 SDE 的真实运行代价，不是上界训练目标；图和标准差反映 rollout 随机性，不是重新训练的种子差异。W1 与质量误差来自标准评测 rollout，对非初始观测时刻取平均。</p>
    <img src="action_comparison.png"><h2>Action、分布和质量</h2>'''
    document+=table[['dataset','method','action_mean','action_sd','kinetic_mean','growth_mean','mean_W1','mean_W2','mean_mass_relative_error','max_mass_relative_error']].to_html(index=False,float_format=lambda x:f'{x:.6g}')
    document+='<h2>阶段耗时（秒）</h2>'+table[['dataset','method','pretraining_seconds','coupling_seconds','matching_seconds']].to_html(index=False,float_format=lambda x:f'{x:.1f}')
    document+='<h2>轨迹与详细结果</h2><p>'+' · '.join(links)+'</p>'
    document+='<p><a href="comparison.csv">汇总 CSV</a> · <a href="paired_differences.json">配对 Action 差异</a></p></html>'
    (OUT/'report.html').write_text(document,encoding='utf8')
    print(table.to_string(index=False))
    print(json.dumps(differences,indent=2))


if __name__=='__main__':
    main()
