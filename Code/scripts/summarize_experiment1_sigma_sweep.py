"""Aggregate paired Monte Carlo and distribution-quality results."""
from pathlib import Path
import argparse
import csv
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'../Results/experiment1_alpha2_sigma_sweep'
DATASETS=('gene','mouse','emt','dygen')
SIGMAS=(0.1,0.25,0.5,1.0)


def main(output_root=OUT,datasets=DATASETS):
    output_root=Path(output_root)
    datasets=tuple(datasets)
    focused_ids={'gene','emt','dygen'}
    suffix='' if set(datasets)==set(DATASETS) else ('_three_datasets' if set(datasets)==focused_ids else '_focused')
    summary_stem='summary' if not suffix else f'summary{suffix}'
    plot_suffix=suffix
    report_name='report.html' if not suffix else f'report{suffix}.html'
    records=[]
    for dataset in datasets:
        for sigma in SIGMAS:
            path=output_root/dataset/f'sigma_{str(sigma).replace(".","p")}'/'comparison_20seed.json'
            if path.exists():
                records.append(json.loads(path.read_text(encoding='utf-8')))
    if not records:
        raise SystemExit('No completed paired comparisons yet.')
    if len(records) != len(datasets)*len(SIGMAS):
        raise SystemExit(f'Expected {len(datasets)*len(SIGMAS)} completed dataset/sigma comparisons, found {len(records)}.')
    fig,axes=plt.subplots(2,len(datasets),figsize=(4.7*len(datasets),8),squeeze=False,constrained_layout=True)
    for col,dataset in enumerate(datasets):
        selected=sorted([r for r in records if r['dataset']==dataset],key=lambda r:r['sigma'])
        if not selected: continue
        x=[r['sigma'] for r in selected]
        methods=[('RUOT','#1776a5'),('WFR+Brownian bridge','#d16a35'),
                 ('WFR coupling + Brownian bridge (log-linear mass)','#2a9d72')]
        for method,color in methods:
            if not all(method in r['methods'] for r in selected):
                continue
            action=[r['methods'][method]['action_mean'] for r in selected]
            sd=[r['methods'][method]['action_sd'] for r in selected]
            axes[0,col].errorbar(x,action,yerr=sd,marker='o',capsize=3,label=method,color=color)
            axes[1,col].plot(x,[r['methods'][method]['W1_mean'] for r in selected],marker='o',label=method,color=color)
        display_name={'gene':'Simulation Gene','mouse':'Mouse hematopoiesis','emt':'EMT','dygen':'WFR-FM Dygen'}
        axes[0,col].set(title=display_name.get(dataset,dataset.upper()),xlabel='σ',ylabel='Rollout Action (mean ± SD)',xscale='log')
        axes[1,col].set(xlabel='σ',ylabel='Mean normalized W1',xscale='log')
        for ax in axes[:,col]:ax.grid(alpha=.22)
        axes[0,col].legend(fontsize=8)
    output_root.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_root/f'{summary_stem}.png',dpi=180);plt.close(fig)
    rows=[]
    body=''
    for r in records:
        ru,wf=r['methods']['RUOT'],r['methods']['WFR+Brownian bridge']
        bb=r['methods'].get('WFR coupling + Brownian bridge (log-linear mass)')
        paired=r['paired_action_difference']
        row=dict(dataset=r['dataset'],alpha=r['alpha'],sigma=r['sigma'],
            RUOT_action_mean=ru['action_mean'],RUOT_action_sd=ru['action_sd'],RUOT_finite=ru['finite_seeds'],
            WFR_BB_action_mean=wf['action_mean'],WFR_BB_action_sd=wf['action_sd'],WFR_BB_finite=wf['finite_seeds'],
            WFR_minus_RUOT=paired['WFR_minus_RUOT_mean'],paired_SE=paired['standard_error'],
            RUOT_lower_count=paired['seeds_RUOT_lower'],WFR_lower_count=paired['seeds_WFR_lower'],
            RUOT_W1=ru['W1_mean'],WFR_BB_W1=wf['W1_mean'],
            RUOT_W2=ru['W2_mean'],WFR_BB_W2=wf['W2_mean'],
            RUOT_mass_error=ru['mass_relative_error_mean'],WFR_BB_mass_error=wf['mass_relative_error_mean'],
            WFR_BB_exp_mass_action_mean=bb['action_mean'] if bb else None,
            WFR_BB_exp_mass_action_sd=bb['action_sd'] if bb else None,
            WFR_BB_exp_mass_finite=bb['finite_seeds'] if bb else None,
            WFR_BB_exp_mass_minus_RUOT=(bb['action_mean']-ru['action_mean']) if bb else None,
            WFR_BB_exp_mass_W1=bb['W1_mean'] if bb else None,
            WFR_BB_exp_mass_W2=bb['W2_mean'] if bb else None,
            WFR_BB_exp_mass_error=bb['mass_relative_error_mean'] if bb else None)
        rows.append(row)
        fmt=lambda v: 'NA' if v is None else f'{v:.6g}'
        body+='<tr>'+''.join(f'<td>{html.escape(fmt(row[k]) if isinstance(row[k],(int,float)) or row[k] is None else str(row[k]))}</td>' for k in (
            'dataset','sigma','RUOT_action_mean','RUOT_action_sd','WFR_BB_action_mean','WFR_BB_action_sd',
            'WFR_minus_RUOT','paired_SE','RUOT_lower_count','WFR_lower_count',
            'RUOT_W1','WFR_BB_W1','RUOT_W2','WFR_BB_W2','RUOT_mass_error','WFR_BB_mass_error',
            'WFR_BB_exp_mass_action_mean','WFR_BB_exp_mass_action_sd','WFR_BB_exp_mass_finite',
            'WFR_BB_exp_mass_minus_RUOT','WFR_BB_exp_mass_W1','WFR_BB_exp_mass_W2','WFR_BB_exp_mass_error'))+'</tr>'
    with (output_root/f'{summary_stem}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output_root/f'{summary_stem}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    is_dygen=tuple(datasets)==('dygen',)
    display_name={'gene':'Simulation Gene','mouse':'Mouse hematopoiesis','emt':'EMT','dygen':'WFR-FM Dygen'}
    selected_names=[display_name.get(d,d.upper()) for d in datasets]
    title='WFR-FM Dygen 数据上的三种方法比较' if is_dygen else f"实验一：{' / '.join(selected_names)}（α=2）"
    data_text='、'.join(selected_names)
    if 'mouse' in datasets:
        data_text += '。Mouse 的源、目标均按数据默认顺序分成 4 块，所有方法使用相同块边界'
    if 'dygen' in datasets:
        data_text += ('。Dygen 为 WFR-FM 官方 5 维、5 时刻数据（t=0,…,4），来自 '
                      '<a href="https://github.com/QiangweiPeng/WFR-FM/blob/main/data/dygen.csv">WFR-FM 仓库</a>')
    data_text += '。'
    heading='α=2：WFR-FM Dygen 数据集' if is_dygen else f"实验一：{'、'.join(selected_names)}（α=2）"
    summary_image=f'{summary_stem}.png'
    scatter_image=f'all_datasets_action_scatter{plot_suffix}.png'
    boxplot_image=f'all_datasets_action_boxplot{plot_suffix}.png'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1450px;margin:36px auto;padding:0 22px;color:#263344}}
table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{padding:7px;border-bottom:1px solid #d8dee6;text-align:right}}
td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}</style>
<h1>{heading}</h1>
<p>固定 α=2，比较 σ=0.1、0.25、0.5、1.0。数据：{data_text}</p>
<p>RUOT 使用 ψ(g)=g²/2、成对重整化代价及 σ² 熵正则；路径网络为 64×3、3000 步。
WFR+布朗桥使用 α 缩放的 WFR-FM OET 静态代价（δ²=α，含 2α 因子），条件均值与质量沿 WFR geodesic，方差取 σ²t(1−t)。
新增 Baseline 固定使用完全相同的 WFR Coupling，只将中间条件路径换成布朗桥：线性均值、方差 σ²t(1−t)，质量 m_t=P exp(t log(Q/P))。
三个方法使用相同的独立反馈网络 V/G/Score（400 宽、6 个隐藏层，倒数第二隐藏层 Tanh）、30,000 Matching 步、数据、训练种子和 rollout 设置。</p>
<p>Action 使用每组一个已训练模型，基于 20 个配对随机 rollout 种子估计；每个种子三方法共用初始粒子与 Brownian 增量。
图中误差条为 rollout 标准差。W1/W2 和质量误差为该模型标准评测 rollout 的各时刻均值。
同一训练设置仅训练种子 42 一次，因此 rollout 变异不包含重新训练的变异。有限实验比较不能单独证明全局最优性。</p>
<img src="{summary_image}" alt="Action 与 W1 随噪声强度变化">
<h2>20 次配对 rollout 的 Action 分布</h2>
<p>散点图中的灰线从 RUOT 连接到同一随机种子的两个 Baseline；箱线图汇总每个数据集、σ 和方法下的 20 个 Action 值。</p>
<img src="{scatter_image}" alt="所选数据集的配对 Action 散点图">
<img src="{boxplot_image}" alt="所选数据集的配对 Action 箱线图">'''
    page+='<h2>数值摘要</h2><table><tr>'+''.join(f'<th>{v}</th>' for v in (
        '数据','σ','RUOT Action 均值','RUOT SD','WFR+BB Action 均值','WFR+BB SD','WFR−RUOT','配对 SE',
        'RUOT 更低数','WFR 更低数','RUOT W1','WFR+BB W1','RUOT W2','WFR+BB W2','RUOT 质量误差','WFR+BB 质量误差',
        'BB 指数质量 Action 均值','BB 指数质量 SD','配对 Action 差 (BB−RUOT)','BB 指数质量 W1','BB 指数质量 W2','BB 指数质量误差'))+'</tr>'+body+'</table>'
    page+=f'<p><a href="{summary_stem}.csv">CSV</a> · <a href="{summary_stem}.json">完整 JSON（含每个 rollout seed）</a></p></html>'
    (output_root/report_name).write_text(page,encoding='utf-8')
    print(json.dumps(dict(completed_comparisons=len(records),report=str((output_root/report_name).resolve())),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,default=OUT)
    parser.add_argument('--datasets',nargs='+',default=list(DATASETS))
    args=parser.parse_args()
    main(args.output_root,args.datasets)
