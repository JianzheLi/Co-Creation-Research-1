"""Report every candidate and paired variation, including unfavorable cases."""
from pathlib import Path
import json,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'../Results/posterior_candidate_sweep'
FIELDS=['entropy_correction','eta_correction','growth_correction','total_correction','surrogate']
LABELS=['Growth–posterior entropy','Drift variance','Growth variance','Sum of corrections','Dropped-term objective']


def values(record,key):
    return np.array([r[key] for r in record['replicates']])


def paired(a,b,key):
    va={r['seed']:r[key] for r in a['replicates']}
    vb={r['seed']:r[key] for r in b['replicates']}
    d=np.array([va[k]-vb[k] for k in sorted(va.keys()&vb.keys())])
    se=float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None
    half=float(student_t.ppf(.975,len(d)-1)*se) if se is not None else None
    return dict(mean=float(d.mean()),standard_error=se,ci95_half_width=half,replicates=len(d))


def main():
    summary={}; csv_rows=[]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    palette=['#d48a00','#3569ae','#a74673','#171717']
    for column,dataset in enumerate(['gene','mouse','emt']):
        folder=OUT/dataset
        records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('c*_screen.json'))]
        if not records: continue
        record_by_id={r['id']:r for r in records}
        base=record_by_id['c000']
        data=np.array([[values(r,key).mean() for key in FIELDS] for r in records])
        by_field={}
        for j,key in enumerate(FIELDS):
            imin=int(data[:,j].argmin()); imax=int(data[:,j].argmax())
            lo,hi=records[imin],records[imax]
            item=dict(minimum=float(data[imin,j]),maximum=float(data[imax,j]),range=float(data[imax,j]-data[imin,j]),
                      minimum_candidate=lo['id'],maximum_candidate=hi['id'],paired_screen_extremes=paired(hi,lo,key))
            f_lo=folder/f"{lo['id']}_confirm.json"; f_hi=folder/f"{hi['id']}_confirm.json"
            if f_lo.exists() and f_hi.exists():
                item['independent_confirmation']=paired(json.loads(f_hi.read_text(encoding='utf-8')),json.loads(f_lo.read_text(encoding='utf-8')),key)
            by_field[key]=item
        for r in records:
            csv_rows.append(dict(dataset=dataset,id=r['id'],kind=r['kind'],**dict(zip(['coupling','mean','covariance','mass','control'],r['coordinates'])),
                **{key:float(values(r,key).mean()) for key in FIELDS},
                **{f'{key}_shift':paired(r,base,key)['mean'] for key in FIELDS}))
        f_range=by_field['surrogate']['range']
        sum_ranges=sum(by_field[key]['range'] for key in FIELDS[:3])
        low_indices=np.argsort(data[:,-1])[:max(1,int(np.ceil(len(records)/4))) ]
        low=data[low_indices]
        low_ranges={key:float(np.ptp(low[:,j])) for j,key in enumerate(FIELDS)}
        # This is an empirical ranking diagnostic, never called feedback Action.
        fbest=int(data[:,-1].argmin()); jbest=int((data[:,-1]+data[:,3]).argmin())
        differences=data-data[0]
        for j,key in enumerate(FIELDS[:4]):
            axes[0,column].scatter(differences[:,-1],differences[:,j],s=12,alpha=.65,color=palette[j],label=LABELS[j])
        axes[0,column].axhline(0,color='.65',lw=.6)
        axes[0,column].set_title(f'{dataset.upper()} · {len(records)} candidates')
        axes[0,column].set_xlabel('Change in dropped-term objective F')
        axes[0,column].set_ylabel('Change in correction from original candidate')
        # Sorted candidate values show their spread without pretending that
        # these Monte Carlo estimates are independent training experiments.
        for j in range(4):
            axes[1,column].plot(np.arange(len(records)),np.sort(differences[:,j]),color=palette[j],lw=1.6,label=LABELS[j])
        axes[1,column].axhline(0,color='.65',lw=.6)
        axes[1,column].set_xlabel('Candidate rank within each correction')
        axes[1,column].set_ylabel('Sorted change in correction')
        total_bias=values(base,'entropy_raw_direct')-values(base,'entropy_correction')
        summary[dataset]=dict(candidate_count=len(records),fields=by_field,sum_component_ranges=sum_ranges,
            lowest_objective_quartile=dict(count=len(low_indices),candidate_ids=[records[i]['id'] for i in low_indices],ranges=low_ranges),
            sum_component_ranges_over_objective_range=sum_ranges/f_range if f_range else None,
            total_range_over_objective_range=by_field['total_correction']['range']/f_range if f_range else None,
            best_surrogate_candidate=records[fbest]['id'],best_discrete_corrected_candidate=records[jbest]['id'],
            sampled_discrete_corrected_regret=float((data[fbest,-1]+data[fbest,3])-(data[jbest,-1]+data[jbest,3])),
            entropy_direct_minus_control_variate_mean=float(total_bias.mean()))
    axes[0,0].legend(fontsize=8,loc='best')
    fig.suptitle('Posterior corrections across a fixed five-dimensional feasible candidate set',fontsize=15)
    fig.savefig(OUT/'candidate_variation.png',dpi=180)
    fig.savefig(OUT/'candidate_variation.pdf')
    plt.close(fig)
    with (OUT/'candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    table=[]
    for dataset,s in summary.items():
        table.append('<tr><th>'+dataset+'</th><td>'+str(s['candidate_count'])+'</td>'+''.join(f'<td>{s["fields"][k]["range"]:.6g}</td>' for k in FIELDS)+f'<td>{s["sum_component_ranges_over_objective_range"]:.2%}</td></tr>')
    details=[]
    for dataset,s in summary.items():
        rows=[]
        for key,item in s['fields'].items():
            conf=item.get('independent_confirmation')
            value='尚未复核' if conf is None else f'{conf["mean"]:.6g} ± {conf["ci95_half_width"]:.2g}'
            rows.append(f'<tr><td>{html.escape(key)}</td><td>{item["minimum_candidate"]} → {item["maximum_candidate"]}</td><td>{value}</td></tr>')
        details.append(f'<h3>{dataset}：筛选极值的独立复核</h3><table><tr><th>项</th><th>最小 → 最大候选</th><th>差值 ± 点态 95% MC 区间半宽</th></tr>'+''.join(rows)+'</table>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>候选解上的后验修正变化</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1350px;margin:35px auto;padding:0 25px;color:#223}h1{font-size:29px}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}img{width:100%}.note{background:#f3f5f8;padding:18px;border-radius:8px}</style>
<h1>候选解上的后验修正变化</h1><p>记完整的起点条件混合目标为 J = F + R₁ + R₂ + R₃ + 固定常数。若在共同候选集合 S 上，各项的振幅 osc(Rₖ) ≤ εₖ，则 δ-近似最小化 F 所带来的 J 次优误差至多 δ + ε₁ + ε₂ + ε₃。这是关于目标值的结论，不保证参数距离或梯度也小。</p>
<h2>候选设计与评估</h2><p>每个数据集预先固定 163 个候选：五维立方体的 32 个顶点、128 个 Sobol 点及 3 个对角内点。五个坐标分别改变原 RUOT 与紧 Bound 解之间的 Coupling、均值网络、方差网络、质量网络及兼容控制。P、Q 在同一可行集合内线性混合；三个路径网络分别插值；控制在同一候选密度对应的两组兼容控制之间插值。保留训练时的端点性质及分块，使用全部候选，未按结果筛除。</p>
<p>筛选使用 2 个独立 Monte Carlo seed、每区间 32 个源点抽样、每源点 16 个高斯样本及 16 点时间积分；同一 seed 的候选共享随机数。固定源点时，对其所在块的所有目标标签计算精确后验。筛选得到各项极值后，使用新的 4 个 seed、128 个源点、32 个高斯样本及 32 点时间积分独立复核。区间反映 Monte Carlo 误差，不涵盖候选覆盖误差和时间离散误差。</p>
<h2>各项候选间的变化范围：最大值 − 最小值</h2><table><tr><th>数据</th><th>候选数</th><th>生长–熵项</th><th>漂移方差项</th><th>生长方差项</th><th>三项合计</th><th>F</th><th>三项振幅之和 / F 振幅</th></tr>'''+''.join(table)+'''</table><img src="candidate_variation.png" alt="三个数据集的候选变化图">'''+''.join(details)+'''
<h2>理论含义与适用范围</h2><p>令 θ̂ 满足 F(θ̂) ≤ inf<sub>S</sub> F + δ，θ* 最小化 J，则 J(θ̂) − J(θ*) ≤ δ + R(θ̂) − R(θ*) ≤ δ + osc<sub>S</sub>R ≤ δ + Σ osc<sub>S</sub>Rₖ。修正即使绝对值较大，只要在候选间接近常数，也几乎不改变优化目标值。三项分别变化小比仅观察总和抵消更强。</p>
<p>这一性质不可能无条件覆盖所有可行控制。即使密度路径不变，也能给标签控制加入互相抵消的兼容扰动，使分量方差任意增大。因此实验只检验当前训练解附近、上述预先固定的候选区域。163 个点也不能给整个连续立方体提供严格的上确界证明。</p>
<div class="note"><b>计算对象：</b>本实验使用固定的离散端点标签与区间内高斯后验，报告三项修正的离散诊断值。手稿的精确重整化恒等式需要连续终点及相应熵渐近条件；有限原子端点的原始条件漂移 Action 存在端点奇性。因此此处的 F + R 不能称为训练后无标签反馈网络的精确 RUOT Action，也不能由候选间振幅直接给此前 rollout Action 提供严格误差界。不同数据集或标签数下修正的绝对值也不宜直接比较。</div>
<p>熵项使用边界控制变量：σ²Σ(Q−P) log(P/a) + σ²∫ΣPm E[g log(ω/(P/a))]dt，第一项精确计算，以降低采样误差。漂移和生长项分别为 −½∫p Varω(η) 与 −α/2∫p Varω(g)。全部候选均保持 α=10、σ=0.25；未改变训练结果。</p>
<p><a href="candidates.csv">全部候选数值 CSV</a> · <a href="summary.json">统计结果 JSON</a> · <a href="candidate_variation.pdf">图 PDF</a></p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
