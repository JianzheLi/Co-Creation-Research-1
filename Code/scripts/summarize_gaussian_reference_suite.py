"""Summarize complete comparisons without treating in-progress runs as results."""
from pathlib import Path
import argparse
import csv
import html
import json
import math
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = 'gaussian_reference'
OUT = ROOT/'../Results/gaussian_reference_comparison'


def diffusion_diagnostics():
    """Summarize matched diagnostic seeds, retaining the finest time rule."""
    records = {}
    for file in (ROOT/'../Results/bound_slack_audit').glob('*_diffusion_decomposition*.json'):
        r = json.loads(file.read_text(encoding='utf-8'))
        key = (r['run'],r['seed'])
        if key not in records or r['time_nodes'] > records[key]['time_nodes']:
            records[key] = r
    grouped = {}
    for (run,seed),r in records.items():
        grouped.setdefault(run,[]).append(r)
    if not grouped:
        return ''
    fields = ['pair_inner_product','receiver_conditional_entropy','true_spatial_remainder',
        'bridge_reference_bound_diffusion','diffusion_gap','diffusion_upper_objective',
        'posterior_corrected_proxy','diffusion_young_slack','diffusion_logmoment_slack']
    summaries = {}
    lines = ['<h2>上界松弛诊断</h2>',
        '<p>读取训练好的路径和 Coupling，以共同采样种子估计空间后验修正 R；B 为扩散尺度完整平方矩修正。F 为成对重整化代价与接收条件熵之和。这里的 F+R 是离散标签诊断量，省略固定端点常数，不能等同于最终网络 rollout Action，也不是连续端点恒等式的数值认证。每块抽样 8 个源、每源 256 个高斯样本；表中为诊断种子均值，详细文件保留各种子的值及时间积分节点。</p>',
        '<table><tr><th>候选路径</th><th>诊断次数</th><th>R</th><th>B</th><th>B−R</th><th>F+B</th><th>F+R</th></tr>']
    for run,rs in sorted(grouped.items()):
        means = {k:float(np.mean([r[k] for r in rs])) for k in fields}
        summaries[run] = dict(mean=means,seeds=[r['seed'] for r in rs],
            time_nodes=[r['time_nodes'] for r in rs],replicates=rs)
        lines.append(f'<tr><td>{html.escape(run)}</td><td>{len(rs)}</td>'+''.join(
            f'<td>{means[k]:.6g}</td>' for k in ['true_spatial_remainder','bridge_reference_bound_diffusion',
                'diffusion_gap','diffusion_upper_objective','posterior_corrected_proxy'])+'</tr>')
    lines.append('</table><p>Young 配方与矩估计的松弛分解已独立核验；分解恒等式误差最大为 '+
        f'{max(abs(r["diffusion_identity_residual"]) for r in records.values()):.3g}。'
        '解析矩与有限样本估计仍存在 Monte Carlo 差异，另存于详细数据。'
        '<a href="diagnostics.json">诊断详细数据</a></p>')
    (OUT/'diagnostics.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    return '\n'.join(lines)


def main():
    global CANDIDATE, OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',choices=['gaussian_reference','diffusion_full_square'],default='gaussian_reference')
    CANDIDATE=parser.parse_args().candidate
    OUT=ROOT/'../Results'/f'{CANDIDATE}_comparison'
    OUT.mkdir(exist_ok=True)
    rows = []
    for file in sorted(OUT.glob('*_20seed.json')):
        r = json.loads(file.read_text(encoding='utf-8'))
        methods = r['methods']
        for name, m in methods.items():
            status = json.loads((Path(m['run'])/'status.json').read_text(encoding='utf-8'))
            rows.append(dict(dataset=r['dataset'],alpha=r['alpha'],sigma=r['sigma'],method=name,
                action_mean=m['action_mean'],action_sd=m['action_sd'],finite_seeds=m['finite_seeds'],
                mean_W1=m['mean_W1'],mean_mass_error=m['mean_mass_error'],
                pair_seconds=status['pair'].get('seconds'),coupling_seconds=status['couplings'].get('seconds'),
                matching_seconds=status['matching'].get('seconds'),
                action_difference=r.get('paired_difference',{}).get('mean'),
                difference_ci95_halfwidth=(student_t.ppf(.975,19)*r['paired_difference']['standard_error']
                    if 'paired_difference' in r else None),
                run=m['run']))
    if rows:
        with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w = csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>高斯参考上界实验</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:right}th:first-child,td:first-child{text-align:left}code{background:#eee;padding:2px 5px}</style>',
        '<h1>数据确定的高斯参考上界：完整流程对照</h1>',
        '<p>这是实验候选，尚未保证降低 Action。Young 系数保持 α；参考方差为 σ²t(1−t)+s_data t²，s_data 从观测位移二阶矩确定。没有新增可调系数。</p>',
        '<p>两方法均从头训练：路径 3 层×64、3000 步；独立 Score/Velocity/Growth 网络 6 层×400、30000 步；倒数第二层 Tanh，输出层线性。保持原有损失和分块。每个比较使用 20 个共同 rollout 种子（不是 20 次独立训练）。W1 和质量误差来自实验配置的最终评测，排除初始时刻。</p>',
        '<table><tr><th>数据</th><th>α / σ</th><th>方法</th><th>Action 均值 ± SD</th><th>W1</th><th>质量相对误差</th><th>有限轨迹</th><th>训练/求解秒数</th></tr>']
    for r in rows:
        action = (f"{r['action_mean']:.6g} ± {r['action_sd']:.3g}" if r['finite_seeds']==20
                  else '存在非有限结果；不据此判断改善')
        body.append(f"<tr><td>{r['dataset']}</td><td>{r['alpha']:g} / {r['sigma']:g}</td><td>{r['method']}</td><td>{action}</td><td>{r['mean_W1']:.5g}</td><td>{r['mean_mass_error']:.3g}</td><td>{r['finite_seeds']}/20</td><td>{r['pair_seconds']:.1f} / {r['coupling_seconds']:.1f} / {r['matching_seconds']:.1f}</td></tr>")
    body.append('</table><h2>配对 Action 差值与分解</h2><p>差值均为新上界减原方法；区间只反映 rollout Monte Carlo 不确定性，不包含训练随机性。</p><table><tr><th>数据 / 参数</th><th>ΔAction（95% 区间）</th><th>Δ漂移代价</th><th>Δ生长代价</th></tr>')
    for file in sorted(OUT.glob('*_20seed.json')):
        result=json.loads(file.read_text(encoding='utf-8'))
        if 'paired_difference' not in result:continue
        d=result['paired_difference'];half=student_t.ppf(.975,19)*d['standard_error']
        old,new=[result['methods'][k]['replicates'] for k in ['original',CANDIDATE]]
        kinetic=np.mean([b['kinetic_action']-a['kinetic_action'] for a,b in zip(old,new)])
        growth=np.mean([b['growth_action']-a['growth_action'] for a,b in zip(old,new)])
        body.append(f"<tr><td>{result['dataset']} / {result['setting']}</td><td>{d['mean']:.5g} [{d['mean']-half:.5g}, {d['mean']+half:.5g}]</td><td>{kinetic:.5g}</td><td>{growth:.5g}</td></tr>")
    body.append('</table><h2>候选路径上的紧度诊断</h2><p>下表是离散标签混合的空间后验余项估计，未包含端点熵转换；不是最终反馈 Action 的严格数值上界。两列在同一路径、同一控制、同一 Coupling 上计算。</p><table><tr><th>候选</th><th>实际空间后验余项估计</th><th>新上界修正</th></tr>')
    for file in sorted((ROOT/'../Results/bound_slack_audit').glob('*_trained_comparison.json')):
        a=json.loads(file.read_text(encoding='utf-8'))
        body.append(f"<tr><td>{html.escape(a['run'])}</td><td>{a['true_spatial_remainder']:.6g}</td><td>{a['bridge_reference_bound_alpha']:.6g}</td></tr>")
    body.append('</table><h2>尚未完成或未通过的比较</h2><ul>')
    for dataset in ['gene','mouse','emt']:
        for setting in ['a2_s1','a10_s025']:
            if not (OUT/f'{dataset}_{setting}_20seed.json').exists():
                states=[]
                failure_file=OUT/f'{dataset}_{setting}_failure.json'
                if failure_file.exists():
                    failure=json.loads(failure_file.read_text(encoding='utf-8-sig'))
                    label='已停止：' if failure.get('status')=='stopped_by_user' else '候选未通过：'
                    body.append('<li>'+html.escape(dataset+' '+setting+' — '+label+failure['reason'])+'</li>')
                    continue
                for method in ['original',CANDIDATE]:
                    run=ROOT/'../Results'/f'{dataset}_{method}_64x3_400x6_{setting}_seed42'
                    if (run/'status.json').exists():
                        s=json.loads((run/'status.json').read_text(encoding='utf-8'))
                        states.append(method+': '+', '.join(f'{k}={v["status"]}' for k,v in s.items()))
                    else:states.append(method+': 未开始')
                body.append('<li>'+html.escape(dataset+' '+setting+' — '+'; '.join(states))+'</li>')
    body.append('</ul><p>严格区分：理论上界、离散后验紧度诊断、最终神经网络 SDE rollout Action。表中报告的是最后一项；W1 与质量误差用于核对较低 Action 是否伴随拟合退化。运行时间是共享 GPU 下的墙钟时间，不能据此直接推断孤立吞吐量。</p></html>')
    if CANDIDATE=='diffusion_full_square':
        body=[line.replace('高斯参考上界','扩散尺度完整平方矩上界').replace('Young 系数保持 α','Young 系数固定为 σ²；内层增长权重为 α+σ²').replace('两方法均从头训练','原方法复用已完成的同配置训练，新目标从头训练') for line in body]
        start=next((i for i,x in enumerate(body) if '<h2>候选路径上的紧度诊断' in x),None)
        end=next((i for i,x in enumerate(body) if '<h2>尚未完成或未通过的比较' in x),None)
        if start is not None and end is not None:
            body[start:end]=[]
        body.insert(len(body)-1,diffusion_diagnostics())
        if rows:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes=plt.subplots(2,3,figsize=(12,7),squeeze=False)
            for row,setting in enumerate(['a10_s025','a2_s1']):
                for col,dataset in enumerate(['gene','mouse','emt']):
                    ax=axes[row,col]
                    file=OUT/f'{dataset}_{setting}_20seed.json'
                    ax.set_title(f'{dataset}: '+('alpha=10, sigma=0.25' if row==0 else 'alpha=2, sigma=1'))
                    if not file.exists():
                        failure=OUT/f'{dataset}_{setting}_failure.json'
                        label=('Stopped by user' if json.loads(failure.read_text(encoding='utf-8')).get('status')=='stopped_by_user'
                            else 'Numerical check failed') if failure.exists() else 'Not completed'
                        ax.text(.5,.5,label,ha='center',transform=ax.transAxes)
                        ax.set_axis_off()
                        continue
                    result=json.loads(file.read_text(encoding='utf-8'))
                    values=[[r['total_action'] for r in result['methods'][method]['replicates'] if r['finite']]
                            for method in ['original',CANDIDATE]]
                    ax.boxplot(values,tick_labels=['Original','Full-square'],showfliers=False)
                    for k,data in enumerate(values):
                        ax.scatter(k+1+np.linspace(-.08,.08,len(data)),data,s=15,alpha=.65)
                    ax.set_ylabel('Raw weighted SDE Action')
                    ax.grid(axis='y',alpha=.2)
            fig.tight_layout()
            fig.savefig(OUT/'comparison.png',dpi=160)
            plt.close(fig)
            body.insert(4,'<img src="comparison.png" alt="20-seed Action comparison" style="max-width:100%">')
    (OUT/'report.html').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps(dict(completed_comparisons=len(rows)//2,report=str(OUT/'report.html')),ensure_ascii=False))


if __name__=='__main__':main()
