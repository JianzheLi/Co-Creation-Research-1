"""Summarize completed samples only; no new paths, candidates or MC draws."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.runtime import configure,load_config,json_write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'../Results/posterior_candidate_sweep'
KEYS=['pair_inner_product','coupling_entropy','retained','entropy_correction','eta_correction','growth_correction','dropped']


def entropy_values(dataset,records):
    config=load_config(ROOT/f'configs/{dataset}_tight_bound.yaml')
    device=configure(config)
    old=ROOT/f'../Results/{dataset}_distance_hermite_primal_seed42/couplings'
    new=ROOT/f'../Results/{dataset}_tight_bound_seed42/couplings'
    pairs=[]
    for file in sorted(old.glob('interval*.npz')):
        with np.load(file) as a,np.load(new/file.name) as b:
            pairs.append((torch.as_tensor(a['P'],dtype=torch.float64,device=device),torch.as_tensor(b['P'],dtype=torch.float64,device=device)))
    result={}
    with torch.no_grad():
        for record in records:
            c=record['coordinates'][0]
            if c not in result:
                value=torch.zeros((),dtype=torch.float64,device=device)
                for p0,p1 in pairs:
                    p=(1-c)*p0+c*p1
                    value+=(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum()
                result[c]=float(value*config['model_sigma']**2)
    return [result[r['coordinates'][0]] for r in records]


def main():
    summaries={}; rows=[]
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.6),layout='constrained')
    for col,dataset in enumerate(['gene','mouse','emt']):
        records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/dataset).glob('c*_screen.json'))]
        entropies=entropy_values(dataset,records)
        samples=[]
        for record,entropy in zip(records,entropies):
            sample=[]
            for rep in record['replicates']:
                sample.append([rep['surrogate']-entropy,entropy,rep['surrogate'],rep['entropy_correction'],rep['eta_correction'],rep['growth_correction'],rep['total_correction']])
            samples.append(sample)
        samples=np.array(samples); data=samples.mean(axis=1)
        ranges=np.ptp(data,axis=0); sd=data.std(axis=0,ddof=1)
        components={key:dict(minimum=float(data[:,j].min()),maximum=float(data[:,j].max()),range=float(ranges[j]),
            range_over_retained=float(ranges[j]/ranges[2]),candidate_standard_deviation=float(sd[j]),sd_over_retained=float(sd[j]/sd[2])) for j,key in enumerate(KEYS)}
        per_rep=[float(np.ptp(samples[:,i,6])/np.ptp(samples[:,i,2])) for i in range(samples.shape[1])]
        best_f=int(data[:,2].argmin()); best_j=int((data[:,2]+data[:,6]).argmin())
        low=np.argsort(data[:,2])[:int(np.ceil(len(data)/4))]
        summary=dict(candidate_count=len(records),monte_carlo_seeds=records[0]['seeds'],components=components,
            range_ratio_per_mc_seed=per_rep,sum_individual_correction_ranges_over_retained=float(ranges[3:6].sum()/ranges[2]),
            lowest_F_quartile=dict(count=len(low),retained_range=float(np.ptp(data[low,2])),dropped_range=float(np.ptp(data[low,6]))),
            best_F=records[best_f]['id'],best_discrete_F_plus_R=records[best_j]['id'],
            discrete_F_plus_R_regret=float(data[best_f,2]+data[best_f,6]-data[best_j,2]-data[best_j,6]),
            note='Completed candidates only. Entropy recomputed exactly from saved P; pair inner product recovered as saved F minus entropy. No new posterior evaluations.')
        summaries[dataset]=summary
        for record,values in zip(records,data):
            rows.append(dict(dataset=dataset,candidate=record['id'],**dict(zip(KEYS,map(float,values)))))
        labels=['Pair cost','Entropy','Retained sum','Growth–entropy','Drift variance','Growth variance','Dropped sum']
        axes[col].barh(np.arange(7),ranges,color=['#4275b5','#85a2c9','#173d70','#dc9b40','#e2b87c','#efd2ab','#b96913'])
        axes[col].set_yticks(np.arange(7),labels=labels,fontsize=9)
        axes[col].invert_yaxis(); axes[col].set_xlabel('Candidate range: maximum − minimum')
        axes[col].set_title(f'{dataset.upper()} · {len(records)} candidates\nDropped / retained = {ranges[6]/ranges[2]:.1%}')
        axes[col].spines[['top','right']].set_visible(False)
    fig.suptitle('Variation of retained and dropped terms on the same completed candidates',fontsize=14)
    fig.savefig(OUT/'retained_vs_dropped.png',dpi=180);fig.savefig(OUT/'retained_vs_dropped.pdf');plt.close(fig)
    json_write(OUT/'retained_vs_dropped.json',summaries)
    with (OUT/'retained_vs_dropped.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=[]; component_table=[]
    for dataset,s in summaries.items():
        c=s['components']
        table.append(f'<tr><th>{dataset}</th><td>{s["candidate_count"]}</td>'+''.join(f'<td>{c[k]["range"]:.6f}</td>' for k in ['pair_inner_product','coupling_entropy','retained','dropped'])+f'<td>{c["dropped"]["range_over_retained"]:.1%}</td><td>{c["dropped"]["sd_over_retained"]:.1%}</td></tr>')
        component_table.append(f'<tr><th>{dataset}</th>'+''.join(f'<td>{c[k]["range"]:.6f} ({c[k]["range_over_retained"]:.1%})</td>' for k in ['entropy_correction','eta_correction','growth_correction'])+'</tr>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>丢掉部分与保留部分的变化幅度</title>
<style>body{font:16px/1.75 system-ui,sans-serif;max-width:1300px;margin:35px auto;padding:0 25px;color:#223}h1{font-size:28px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}img{width:100%}.note{padding:18px;background:#f3f5f8;border-radius:8px}</style>
<h1>丢掉部分与保留部分：比较候选间的变化幅度</h1>
<p><b>保留部分：</b>A = ⟨P,C<sub>ren</sub>⟩，E = σ²ΣP log P，F = A + E。<br><b>丢掉部分：</b>R = R₁ + R₂ + R₃，依次为生长–后验熵、漂移后验方差和生长后验方差修正，均包含原公式的系数与符号。</p>
<p>使用同一批候选计算 osc(H) = max H − min H，核心指标为 osc(R)/osc(F)。同时列出 A、E 的振幅，避免其相互抵消被隐藏。标准差之比反映候选中的整体变化，较少受极值影响；它等价于所有候选两两差值的均方根之比。</p>
<h2>已有结果</h2><table><tr><th>数据</th><th>候选数</th><th>⟨P,Cren⟩ 振幅</th><th>熵项振幅</th><th>保留合计振幅</th><th>丢掉合计振幅</th><th>振幅之比</th><th>标准差之比</th></tr>'''+''.join(table)+'''</table>
<h2>三项修正分别比较</h2><p>括号内为该项振幅 / 保留合计 F 的振幅。</p><table><tr><th>数据</th><th>生长–后验熵项</th><th>漂移方差项</th><th>生长方差项</th></tr>'''+''.join(component_table)+'''</table>
<img src="retained_vs_dropped.png" alt="相同候选集上的保留项与删除项振幅">
<h2>计算范围</h2><p>已按要求停止候选扫描。本页只使用停止前完整保存的候选，不做新训练、新候选或新的后验采样。候选在原 RUOT 与紧 Bound 结果之间，分别改变 Coupling、均值路径、方差路径、质量路径和兼容控制；各数据集保留各自已完成的全部候选。</p>
<p>后验评估使用两个 Monte Carlo seed（6000、6001），相同 seed 在候选间共享随机数；每区间抽样 32 个源点，每源点 16 个高斯样本，时间为 16 点积分。固定源点时，对所在块的全部目标计算后验。熵项 E 由保存的 Coupling 精确重算；A 从已有 F−E 恢复，没有再次访问路径网络。这是初步数值估计，尚未作独立极值复核或时间积分加密。</p>
<h2>如何解读</h2><p>若同一候选集合上 osc(R) ≤ ε，且 θ̂ 是 F 的 δ-近似最优解，则 (F+R)(θ̂) − min(F+R) ≤ δ+ε。候选间变化小才是修正不明显改变目标值的理由，修正的绝对值小并非必要条件。振幅比也不能保证几乎等价的候选排序完全不变。</p>
<div class="note">这里评估的是固定离散端点标签下的后验修正诊断。手稿的连续端点重整化恒等式不能直接等同于有限原子端点或训练后反馈网络的 rollout Action。因此本页对比说明这些已完成候选中的相对变化，不是整个可行空间的严格上界，也不是 rollout Action 的误差证书。</div>
<p><a href="retained_vs_dropped.csv">逐候选数值 CSV</a> · <a href="retained_vs_dropped.json">详细统计 JSON</a> · <a href="retained_vs_dropped.pdf">图 PDF</a></p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
