"""Report 20 feasible random perturbations at each scale, for gene and EMT."""
from pathlib import Path
import csv
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('../Results/coupling_neighborhood_20')
    records=[json.loads((out/ds/'results.json').read_text(encoding='utf-8')) for ds in ['gene','emt']]
    assert all(r['status']=='completed' for r in records)
    summaries=[];rows=[];sections=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,r in zip(axes,records):
        validation={}
        for check in r['validation']:
            for c in check['candidates']:
                validation.setdefault(c['id'],[]).append(c['delta_corrected'])
        groups=[];body=''
        for i,radius in enumerate([.001,.01,.05]):
            candidates=[c for c in r['candidates'] if c['requested_max_TV']==radius]
            assert len(candidates)==20
            vals=[c['delta_corrected'] for c in candidates];groups.append(vals)
            verified=[c for c in candidates if c['id'] in validation]
            best=min(verified,key=lambda c:np.mean(validation[c['id']]))
            bv=validation[best['id']]
            summary=dict(dataset=r['dataset'],radius=radius,count=len(candidates),
                screen_min=min(vals),screen_median=float(np.median(vals)),screen_max=max(vals),
                screen_improved=sum(x<0 for x in vals),best_verified=best['id'],
                best_verified_delta_mean=float(np.mean(bv)),best_verified_deltas=bv,
                verified_improved=sum(all(x<0 for x in validation[c['id']]) for c in verified),
                max_marginal_error=max(max(c['row_relative_l1'],c['column_relative_l1']) for c in candidates))
            summaries.append(summary)
            body+='<tr>'+''.join(f'<td>{v}</td>' for v in [f'{radius:.1%}',20,
                f'{min(vals):+.6g}',f'{np.median(vals):+.6g}',f'{max(vals):+.6g}',
                summary['screen_improved'],', '.join(f'{v:+.6g}' for v in bv)])+'</tr>'
            ax.scatter(i+1+np.linspace(-.14,.14,20),vals,s=19,alpha=.7,color='#3485ae')
            for c in verified:
                ax.scatter([i+1]*2,validation[c['id']],marker='x',s=38,color='#ce4936',zorder=5)
        ax.boxplot(groups,positions=[1,2,3],widths=.45,showfliers=False)
        ax.axhline(0,color='gray',lw=.7);ax.grid(axis='y',alpha=.2)
        ax.set_xticks([1,2,3],['0.1%','1%','5%'])
        ax.set(xlabel='max relative TV of P/Q',ylabel='Change in corrected objective',title=r['dataset'].upper())
        ax.set_yscale('symlog',linthresh=1e-6)
        ax.set_ylim(-1e-6,2*max(max(g) for g in groups))
        detail=''
        for c in r['candidates']:
            vv=validation.get(c['id'],[])
            row=dict(dataset=r['dataset'],id=c['id'],scale=c['requested_max_TV'],
                P_TV=c['P_relative_TV'],Q_TV=c['Q_relative_TV'],
                delta_F=c['delta_retained'],delta_R=c['delta_dropped'],delta_J=c['delta_corrected'],
                validation_7400=vv[0] if vv else '',validation_7401=vv[1] if vv else '')
            rows.append(row)
            detail+='<tr>'+''.join(f'<td>{v}</td>' for v in [html.escape(c['id']),
                f'{row["P_TV"]:.4%}',f'{row["Q_TV"]:.4%}',f'{row["delta_F"]:+.7g}',
                f'{row["delta_R"]:+.7g}',f'{row["delta_J"]:+.7g}',
                ', '.join(f'{x:+.7g}' for x in vv) or '—'])+'</tr>'
        sections.append(f'<h2>{r["dataset"].upper()}：{r["shape"][0]} × {r["shape"][1]}</h2>'
            '<table><tr><th>TV 尺度</th><th>候选数</th><th>筛查最小 ΔJ</th><th>中位数</th><th>最大值</th>'
            '<th>筛查改善数</th><th>最佳复核候选的两次 ΔJ</th></tr>'+body+'</table>'
            '<details><summary>全部候选：保留项与修正项的变化</summary><table><tr><th>ID</th><th>P TV</th><th>Q TV</th>'
            '<th>ΔF</th><th>ΔR</th><th>ΔJ 筛查</th><th>ΔJ 复核</th></tr>'+detail+'</table></details>')
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    with (out/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>每尺度 20 个 Coupling 邻域候选</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#243040}
table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}
td:first-child,th:first-child{text-align:left}img{max-width:100%}details{margin-top:20px}</style>
<h1>Gene / EMT：每个尺度 20 个可行随机 Coupling</h1>
<p>α = 10，σ = 0.25，原始模型训练种子 42。每个数据集 60 个候选；
使用新的 20 个随机方向（种子 8200–8219），每个方向分别构造相对 TV 为 0.1%、1%、5% 的候选。
同一尺度内方向独立，三个尺度共用方向以比较幅度影响；这不是对可行域的均匀抽样。</p>
<p>冻结条件路径网络参数，但按新的 Q/P 重新计算路径；仅扰动第一张 Coupling，其他时间段固定。
P 的行边缘、Q 的列边缘、非负性和质量比范围均保留。未重新训练 Matching。</p>
<p>J = F + R。F 为原始成对代价内积与发射熵项，R 为三项后验修正。
直接计算离散后验修正目标，使用等价的接收条件熵和空间余项形式，不使用 Bound。
该诊断目标不等同于重新 rollout 测量的 Action。</p>
<p>筛查枚举所有源点与终点分量，8 个时间节点，每分量 1 个高斯样本。
每个尺度选择筛查最好的 2 个候选，分别用独立种子 7400、7401、16 个时间节点和每分量 2 个样本复核。
每次评价中候选和原解使用共同随机数；两次复核不构成置信区间。</p>
<p>蓝点和箱线图为 20 个候选的筛查值，红叉为最好的两个候选的独立复核值。
负值表示优于原解。有限随机方向的结果不能证明局部最优，更不能排除未测试方向的改善。
ΔF 与 ΔR 可能抵消，目标变化小不代表修正项本身变化小。</p>
<p>本次仅增加随机方向。上一轮 Gene 沿旧 Bound Coupling 方向找到的微小改善
（两次复核平均 −4.265×10⁻⁶）仍然成立，未包含在本轮每尺度 20 个随机候选中。
<a href="../coupling_neighborhood/report.html">上一轮结果</a></p>
<img src="comparison.png" alt="每个尺度 20 个候选的修正目标变化">'''
    page+=''.join(sections)+'<p><a href="candidates.csv">全部候选 CSV</a> · <a href="summary.json">逐尺度摘要 JSON</a></p></html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
