"""Summarize the gene/EMT feasible-neighborhood diagnostic, without retraining."""
from pathlib import Path
import csv
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out = Path('../Results/coupling_neighborhood')
    records = [json.loads((out / ds / 'results.json').read_text(encoding='utf-8'))
               for ds in ('gene', 'emt')]
    assert all(r['status'] == 'completed' for r in records)
    rows, summaries, sections = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, r in zip(axes, records):
        ds = r['dataset']
        validation = {}
        for check in r['validation']:
            for c in check['candidates']:
                validation.setdefault(c['id'], []).append(c['delta_corrected'])
        for c in r['candidates']:
            rows.append(dict(dataset=ds, candidate=c['id'], P_TV=c['P_relative_TV'],
                Q_TV=c['Q_relative_TV'], delta_retained=c['delta_retained'],
                delta_dropped=c['delta_dropped'], delta_corrected_screen=c['delta_corrected'],
                validation_seed7400=validation.get(c['id'], ['', ''])[0],
                validation_seed7401=validation.get(c['id'], ['', ''])[1]))
        best = min(validation, key=lambda k: np.mean(validation[k]))
        c = next(c for c in r['candidates'] if c['id'] == best)
        v = validation[best]
        summaries.append(dict(dataset=ds, shape=r['shape'], candidates=len(r['candidates']),
            best_verified_candidate=best, P_TV=c['P_relative_TV'], Q_TV=c['Q_relative_TV'],
            verified_delta_mean=float(np.mean(v)), verified_delta_range=[min(v), max(v)],
            verified_improvement=all(x < 0 for x in v), seconds=r['seconds']))
        for direction in dict.fromkeys(c['direction'] for c in r['candidates']):
            group = [c for c in r['candidates'] if c['direction'] == direction]
            ax.plot([100*max(c['P_relative_TV'], c['Q_relative_TV']) for c in group],
                    [c['delta_corrected'] for c in group], '.-', alpha=.65, label=direction)
        for key, values in validation.items():
            c = next(c for c in r['candidates'] if c['id'] == key)
            ax.scatter([100*max(c['P_relative_TV'], c['Q_relative_TV'])]*len(values),
                       values, marker='x', s=45, color='black', zorder=5)
        ax.axhline(0, color='gray', lw=.7)
        ax.set(xlabel='max relative TV of P/Q (%)', ylabel='Corrected objective change', title=ds.upper())
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1e-5)
        ax.grid(alpha=.18)
        body = ''
        for c in r['candidates']:
            vals = validation.get(c['id'])
            validation_text = ', '.join(f'{x:+.8g}' for x in vals) if vals else '—'
            body += '<tr>' + ''.join(f'<td>{x}</td>' for x in (
                html.escape(c['id']), f"{c['P_relative_TV']:.4%}", f"{c['Q_relative_TV']:.4%}",
                f"{c['delta_retained']:+.8g}", f"{c['delta_dropped']:+.8g}",
                f"{c['delta_corrected']:+.8g}", validation_text)) + '</tr>'
        sections.append(f'<h2>{ds.upper()}：{r["shape"][0]} × {r["shape"][1]}，{len(r["candidates"])} 个候选</h2>'
            '<table><thead><tr><th>扰动方向 / 请求半径</th><th>P TV</th><th>Q TV</th>'
            '<th>ΔF（保留项）</th><th>ΔR（三项修正）</th><th>Δ(F+R) 筛查</th>'
            '<th>独立复核 7400 / 7401</th></tr></thead><tbody>' + body + '</tbody></table>')
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)
    with (out/'candidates.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding='utf-8')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Coupling 邻域扰动诊断</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1350px;margin:40px auto;padding:0 24px;color:#243040}
table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #dde3e9;text-align:right}
th:first-child,td:first-child{text-align:left}img{max-width:100%}.note{padding:18px;background:#f0f5fa;border-radius:8px}</style>
<h1>最终 Coupling 附近：加入真实后验修正后是否有更优候选？</h1>
<p>仅 Gene 与 EMT；α = 10，σ = 0.25，训练种子 42。保持原始路径网络参数固定，
只扰动第一张已保存 Coupling，其余时间段保持不变。不重新训练 Matching。</p>
<p class="note">Gene 找到经两次独立采样复核的微小改善（约 −4.27×10⁻⁶）；
EMT 筛查最优的两个候选在两次独立复核中均未改善，较好的候选约为 +1.62×10⁻⁵。有限候选测试不能证明局部最优或全局最优，
也不能证明最优 Coupling 不变。这里评价的是离散后验修正目标，不是重新 rollout 得到的 Action。</p>
<p>F = 成对重整化代价与 P 的内积 + 原始发射熵项；R 包含三项被删掉的修正。
计算 F+R 时采用等价的接收条件熵 + 空间后验余项形式，端点边缘固定带来的常数不影响目标差。
直接计算后验，不使用上界。保留项和修正项的变化可能明显抵消，不能据此称 R 在邻域内不变。</p>
<p>四个随机可行方向、一个梯度构造方向、一个指向旧 Bound Coupling 的方向；
在相对 TV 0.1%、1%、5% 处测试。凸组合保持 P 的行边缘、Q 的列边缘及质量比约束。
固定原始网络参数，但按候选 Q/P 重新计算条件路径。</p>
<p>筛查枚举该矩阵的所有源点与终点分量，8 个时间节点，每分量一个高斯样本；
筛查最优的两个候选以 16 个时间节点、每分量两个高斯样本，使用独立种子 7400、7401 复核。
候选与基准共享随机数。两个复核数值不是置信区间。黑色叉号为复核结果，彩线为筛查结果。</p>
<img src="comparison.png" alt="两数据集的目标变化与 Coupling 扰动幅度">'''
    page += ''.join(sections) + '<p><a href="candidates.csv">全部候选 CSV</a> · <a href="summary.json">摘要 JSON</a></p></html>'
    (out/'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
