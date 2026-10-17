from pathlib import Path
import json, html
import pandas as pd
root=Path.cwd(); outputs=(root/'../Results').resolve(); out=outputs/'alpha2_sigma1_comparison'; out.mkdir(parents=True,exist_ok=True)
rows=[]
for name,label in [('gene','Simulation gene'),('mouse','Mouse hematopoiesis'),('emt','EMT')]:
 ru=outputs/f'{name}_distance_hermite_primal_alpha2_sigma1_seed42'
 wf=outputs/f'{name}_wfr_bridge_alpha2_sigma1_seed42'
 ra=json.loads((ru/'action.json').read_text(encoding='utf-8')); wa=json.loads((wf/'action.json').read_text(encoding='utf-8'))
 rm=pd.read_csv(ru/'metrics.csv'); wm=pd.read_csv(wf/'metrics.csv')
 rp=json.loads((ru/'conditional_path_comparison.json').read_text(encoding='utf-8'))
 def mean_after0(df,col): return float(df.loc[df.time>0,col].mean())
 row={
  'dataset':label,'ruot_action':ra['total_action'],'ruot_action_se':ra['particle_mc_standard_error'],
  'wfr_action':wa['total_action'],'wfr_action_se':wa['particle_mc_standard_error'],
  'action_diff':ra['total_action']-wa['total_action'],
  'action_diff_pct_wfr':100*(ra['total_action']-wa['total_action'])/wa['total_action'],
  'ruot_w1_mean':mean_after0(rm,'W1_normalized'),'wfr_w1_mean':mean_after0(wm,'W1_normalized'),
  'ruot_w2_mean':mean_after0(rm,'W2_normalized'),'wfr_w2_mean':mean_after0(wm,'W2_normalized'),
  'ruot_masserr_mean_pct':100*mean_after0(rm,'mass_relative_error'),
  'wfr_masserr_mean_pct':100*mean_after0(wm,'mass_relative_error'),
  'mean_path_rms':sum(x['mean_shift_rms'] for x in rp['intervals'])/len(rp['intervals']),
  'mass_path_rms_pct':100*sum(x['mass_relative_rms'] for x in rp['intervals'])/len(rp['intervals']),
  'variance_path_rms_pct':100*sum(x['variance_relative_rms'] for x in rp['intervals'])/len(rp['intervals']),
  'pair_cost_gain':rp['weighted_pair_surrogates_on_ruot_coupling']['outer_path_gain'],
  'ruot_dir':ru.name,'wfr_dir':wf.name,
 }
 rows.append(row)
pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
(out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
head='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>α=2, σ=1 三数据集实验</title><style>body{font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;max-width:1500px;margin:28px auto;padding:0 22px;color:#233}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf2f7}section{margin:28px 0}img{max-width:100%;height:auto}a{color:#1769aa}</style><h1>α=2，σ=1：三个数据集 RUOT 与 WFR 对照</h1><p>三组均使用 ψ(g)=g²/2、3000 步路径预训练、30000 步 Matching、2000 粒子连续 rollout、200/单位时间步。RUOT 与 WFR 使用相同的种子和匹配网络设置；Mouse 两边均按数据默认顺序分成8块，Gene 与 EMT 使用完整矩阵。此时 β=σ²/α=0.5，布朗桥最大方差与 α 之比为 σ²/(4α)=12.5%。</p>'''
summary='''<section><h2>连续 rollout 结果</h2><p>Action 是训练后反馈模型在同一离散 rollout 上估计的 RUOT running Action，括号是粒子 Monte Carlo 标准误。W1/W2 和质量误差对观测时刻 t&gt;0 取平均。每个数据集目前是一个固定训练种子，不是多种子统计。</p><table><tr><th>数据集</th><th>RUOT Action ± SE</th><th>WFR Action ± SE</th><th>RUOT−WFR</th><th>平均 W1 RUOT / WFR</th><th>平均 W2 RUOT / WFR</th><th>平均质量误差 RUOT / WFR</th></tr>'''
for r in rows:
 summary+=f"<tr><td>{html.escape(r['dataset'])}</td><td>{r['ruot_action']:.4f} ± {r['ruot_action_se']:.3f}</td><td>{r['wfr_action']:.4f} ± {r['wfr_action_se']:.3f}</td><td>{r['action_diff']:+.4f} ({r['action_diff_pct_wfr']:+.1f}%)</td><td>{r['ruot_w1_mean']:.4f} / {r['wfr_w1_mean']:.4f}</td><td>{r['ruot_w2_mean']:.4f} / {r['wfr_w2_mean']:.4f}</td><td>{r['ruot_masserr_mean_pct']:.2f}% / {r['wfr_masserr_mean_pct']:.2f}%</td></tr>"
summary+='</table></section><section><h2>相同 RUOT Coupling 条件上的条件路径差异</h2><p>从 RUOT 的 P 中按概率抽取8192组成对条件，64点时间积分。质量与方差误差为相对 WFR+布朗桥的时空 RMS；均值为数据坐标单位。成对代价差是同一 RUOT Coupling 下 WFR 条件路径（使用同一最小兼容控制）减去学习到的高斯路径，不能当作反馈 rollout Action。</p><table><tr><th>数据集</th><th>均值位移 RMS</th><th>质量相对 RMS</th><th>方差相对 RMS</th><th>WFR−RUOT 成对 Cren</th></tr>'
for r in rows:
 summary+=f"<tr><td>{html.escape(r['dataset'])}</td><td>{r['mean_path_rms']:.4f}</td><td>{r['mass_path_rms_pct']:.2f}%</td><td>{r['variance_path_rms_pct']:.2f}%</td><td>{r['pair_cost_gain']:.4f}</td></tr>"
summary+='</table>'
for r in rows:
 summary+=f"<h3>{html.escape(r['dataset'])}</h3><img src='../{r['ruot_dir']}/conditional_path_comparison.png'><p><a href='../{r['ruot_dir']}/report.html'>RUOT 完整结果</a> · <a href='../{r['wfr_dir']}/report.html'>WFR 完整结果</a> · <a href='../{r['ruot_dir']}/conditional_path_comparison.json'>条件路径指标</a></p>"
summary+='</section><p>数值结果：<a href="summary.csv">CSV</a> · <a href="summary.json">JSON</a></p></html>'
(out/'report.html').write_text(head+summary,encoding='utf-8')
print(out/'report.html')
for r in rows: print(r['dataset'], 'Action Δ',r['action_diff'], 'W1',r['ruot_w1_mean'],r['wfr_w1_mean'],'mass RMS %',r['mass_path_rms_pct'],'var RMS %',r['variance_path_rms_pct'])
