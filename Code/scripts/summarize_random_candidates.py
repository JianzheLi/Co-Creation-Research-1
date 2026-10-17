from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'../Results/random_posterior_candidates'
FIELDS=['pair_inner_product','coupling_entropy','surrogate','entropy_correction','eta_correction','growth_correction','total_correction']
GROUPS=['random_coupling','random_path','both_random']
NAMES=['Random coupling','Random path','Both random']


def stats(records):
    sample=np.array([[[r[k] for k in FIELDS] for r in rec['replicates']] for rec in records])
    values=sample.mean(1); span=np.ptp(values,axis=0)
    return dict(count=len(records),ranges=dict(zip(FIELDS,map(float,span))),
        dropped_retained_range_ratio=float(span[-1]/span[2]),
        component_range_ratios=dict(zip(FIELDS[3:6],map(float,span[3:6]/span[2]))),
        per_seed_range_ratios=[float(np.ptp(sample[:,i,-1])/np.ptp(sample[:,i,2])) for i in range(sample.shape[1])])


def main():
    summary={}; csvrows=[]; tab=[]; group_rows=[]
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for i,dataset in enumerate(['gene','mouse','emt']):
        records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/dataset).glob('*.json')) if p.name!='design.json']
        assert len(records)==13,(dataset,len(records))
        summary[dataset]=stats(records)
        summary[dataset]['groups']={group:stats([r for r in records if r['group']==group]) for group in GROUPS}
        for rec in records:
            csvrows.append(dict(dataset=dataset,id=rec['id'],group=rec['group'],**{k:float(np.mean([v[k] for v in rec['replicates']])) for k in FIELDS}))
        baseline=next(r for r in csvrows if r['dataset']==dataset and r['group']=='baseline')
        for group,label in zip(GROUPS,NAMES):
            rows=[r for r in csvrows if r['dataset']==dataset and r['group']==group]
            axes[i].scatter([r['surrogate']-baseline['surrogate'] for r in rows],[r['total_correction']-baseline['total_correction'] for r in rows],label=label,s=42)
        axes[i].scatter([0],[0],color='black',marker='x',label='Trained baseline')
        axes[i].axhline(0,color='.7',lw=.6);axes[i].set_title(dataset.upper());axes[i].set_xlabel('Change in retained F');axes[i].set_ylabel('Change in dropped R')
        s=summary[dataset];r=s['ranges']
        tab.append(f'<tr><th>{dataset}</th>'+''.join(f'<td>{r[k]:.5g}</td>' for k in [FIELDS[0],FIELDS[1],FIELDS[2],FIELDS[-1]])+f'<td>{s["dropped_retained_range_ratio"]:.1%}</td></tr>')
        group_rows.append(f'<tr><th>{dataset}</th>'+''.join(f'<td>{s["groups"][g]["dropped_retained_range_ratio"]:.1%}</td>' for g in GROUPS)+''.join(f'<td>{s["component_range_ratios"][k]:.1%}</td>' for k in FIELDS[3:6])+'</tr>')
    axes[0].legend(fontsize=8);fig.suptitle('Broad random feasible candidates: retained versus dropped variation')
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (OUT/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>广泛随机候选的后验修正</title><style>body{font:16px/1.8 system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#223}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}table{border-collapse:collapse;width:100%}img{width:100%}</style>
<h1>随机 Coupling 与随机条件路径：丢掉项和保留项的变化</h1>
<p>每个数据集 12 个随机候选，加 1 个原 RUOT 基准。三组各 4 个：只随机 Coupling、只随机路径、两者都随机。所有候选保留 α=10、σ=0.25、原数据分块与端点兼容参数化，不做训练或按结果筛选。</p>
<p>Coupling：对 P、Q 分别施加独立高斯对数权重，标准差为 1 或 3，再进行 KL 投影，仅满足 row(P)=a、column(Q)=b、0.1≤Q/P≤10；相对边缘误差 ≤10⁻⁸。未使用位置距离或成对代价构造随机权重。</p>
<p>路径：三个网络全部重新随机，包括输出层。每层权重分布 N(0,gain²/fan_in)，偏置 N(0,(0.2gain)²)，gain 为 0.5 或 1.5。四个随机种子与两个尺度交叉组合。仍使用原来的有界 tanh 修正，均值、协方差与质量满足原端点结构。每条候选路径使用原兼容控制律，避免额外改变控制选择。</p>
<h2>整个候选集合的振幅：最大值减最小值</h2><p>F=⟨P,Cren⟩+σ²ΣP log P；R 为三项修正之和。各项在同一候选集上比较。</p><table><tr><th>数据</th><th>成对代价内积</th><th>熵项</th><th>保留 F</th><th>丢掉 R</th><th>osc(R)/osc(F)</th></tr>'''+''.join(tab)+'''</table>
<h2>分组与单项振幅比</h2><p>前三列在各组的四个随机候选内计算 osc(R)/osc(F)，不含基准；后三列在完整 13 个候选上计算各修正项振幅 / F 振幅。</p><table><tr><th>数据</th><th>只随机 Coupling</th><th>只随机路径</th><th>两者随机</th><th>生长–后验熵</th><th>漂移方差</th><th>生长方差</th></tr>'''+''.join(group_rows)+'''</table><img src="comparison.png">
<p>评估使用 GPU float64、两个共同随机数 seed（8100、8101），每区间 32 个源点抽样、每源点 16 个高斯样本、16 个时间积分节点。属于小规模广域压力测试，未经时间积分加密或独立精度复核。</p>
<p><b>解释：</b>随机候选往往具有很高的保留代价；其中修正的相对变化小，不能推出最优解附近也小。反过来，若随机候选中修正变化明显，说明不能把“修正普遍近似常数”作为全局理论性质。它们依旧只覆盖现有高斯路径参数化，不是所有概率路径。</p>
<p>这里是固定离散端点标签的修正诊断，并非训练后反馈网络的精确 rollout Action 分解。连续终点重整化恒等式的适用条件仍须区别对待。</p>
<p><a href="candidates.csv">逐候选 CSV</a> · <a href="summary.json">统计 JSON</a> · <a href="comparison.pdf">图 PDF</a></p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
