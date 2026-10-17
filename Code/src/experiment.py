from pathlib import Path
import html
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from .evaluate import evaluate_snapshot
from .matching import make_feedback, validate_feedback_architecture
from .runtime import configure, file_hash, init_run, json_write, snapshots_for, update_stage
from .sample import rollout


def plot_results(run, truth, generated=None, metrics=None):
    run = Path(run)
    if generated is not None:
        n_times = len(truth)
        fig, axes = plt.subplots(2, n_times, figsize=(3.2 * n_times, 6), sharex=True, sharey=True,
                                 squeeze=False)
        for k in range(n_times):
            axes[0, k].scatter(truth[k].x[:, 0], truth[k].x[:, 1], s=5, alpha=0.5)
            a = np.exp(generated["log_weights"][k] - logsumexp(generated["log_weights"][k]))
            x = generated["positions"][k]
            axes[1, k].scatter(x[:, 0], x[:, 1], s=np.minimum(60, 4 + 3000 * a), c=a, cmap="viridis", alpha=0.55)
            axes[0, k].set_title(f"t={k}; observed M={truth[k].mass:.4g}")
            axes[1, k].set_title(f"generated M={metrics.iloc[k].mass_pred:.4g}")
            axes[1, k].set_xlabel("x1")
        axes[0, 0].set_ylabel("Observed x2")
        axes[1, 0].set_ylabel("Generated x2")
        if truth[0].x.shape[1] > 2:
            fig.suptitle(f"x1/x2 projection of {truth[0].x.shape[1]}D data; metrics use all coordinates")
        fig.tight_layout()
        fig.savefig(run / "snapshots.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(metrics.time, metrics.mass_true, "o-", label="Observed mass")
        axes[0].plot(metrics.time, metrics.mass_pred, "o-", label="Generated mass")
        axes[0].set(xlabel="Model time", ylabel="Unnormalized total mass")
        axes[1].plot(metrics.time, metrics.W1_normalized, "o-", label="W1")
        axes[1].plot(metrics.time, metrics.W2_normalized, "o-", label="W2")
        axes[1].set(xlabel="Model time", ylabel="Probability Wasserstein distance")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run / "mass_and_distance.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pair_steps = (run / "pair_steps.jsonl").exists()
    pair_history = "pair_steps.jsonl" if pair_steps else "pair_history.jsonl"
    for ax, name, xlabel in zip(axes, (pair_history, "matching_history.jsonl"), ("Pair step" if pair_steps else "Coverage epoch", "Matching step")):
        file = run / name
        if file.exists():
            rows = []
            invalid_rows = 0
            for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Interrupted runs can leave partial JSONL records. These
                    # logs are diagnostic only; keep valid history for plotting.
                    invalid_rows += 1
            if invalid_rows:
                print(f"Warning: skipped {invalid_rows} invalid training-history rows in {file}", flush=True)
            ax.plot([r.get("epoch", r.get("step")) for r in rows], [r["loss"] for r in rows])
            ax.set_yscale("symlog", linthresh=0.1)
        ax.set(xlabel=xlabel, ylabel="Training loss")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "training.png", dpi=150)
    plt.close(fig)


def write_report(run, config):
    run = Path(run)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    completed = status.get("evaluation", {}).get("status") == "completed"
    env = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    pair_cfg = config["pair"]
    title = "实验结果" if completed else "实验阶段诊断"
    if config.get("experiment_method") in {"wfr_fm_bridge", "wfr_fm_bridge_exp_mass"}:
        is_exp_mass_bridge = config.get("experiment_method") == "wfr_fm_bridge_exp_mass"
        if is_exp_mass_bridge:
            pair_training = "WFR Coupling 固定端点配对；条件路径采用线性均值、布朗桥方差與指数质量插值"
            heading = f"WFR Coupling + 布朗桥条件路径：{title}"
            path_description = (
                f"<p>Coupling 使用 α={config['growth_penalty']:g} 的 WFR-FM OET；只用它确定端点半耦合 P/Q。"
                f"中间条件路径改为真正的布朗桥：均值 ξ_t=(1−t)x_0+tx_1，方差 σ²t(1−t)I（σ={config['model_sigma']:g}）；"
                "质量按 m_t=P exp(t log(Q/P)) 插值，因此对数质量随时间线性变化。</p>")
        else:
            pair_training = "解析 WFR traveling-Dirac 均值与二次 WFR 质量曲线；未训练成对路径网络"
            heading = f"WFR-FM 启发式路径：{title}"
            path_description = (
                f"<p>条件均值和质量取解析 WFR geodesic，方差按本次指定的 σ²t(1−t) 布朗桥平滑，σ={config['model_sigma']:g}。"
                "论文的 traveling Gaussian 使用常方差；这里的布朗桥方差是与 RUOT 匹配的启发式变体。</p>")
        method_description = (
            f"<p>增长惩罚为 αψ(g)，ψ(g)=g²/2，α={config['growth_penalty']:g}，因此 WFR δ²=α。Dirac 半耦合静态代价为 "
            "2α(P+Q−2κ√(PQ))，κ=cos(min(‖x−y‖/(2√α),π/2))；等价 OET 目标整体含同一 2α 因子，"
            "κ 内也使用 α。OET 按 WFR-FM 设两端 KL 系数为 1，不额外加入 RUOT 的 σ² 联合熵项。</p>"
            + path_description +
            "<p>反馈网络结构、批量、优化器、步数、数据、种子和生成评测设置沿用 RUOT 实验。</p>")
    else:
        pair_training = (f"从头训练 {pair_cfg['training_steps']} 步" if pair_cfg.get("training_mode") == "steps"
                         else f"覆盖训练 {pair_cfg['coverage_epochs']} 轮")
        heading = f"DeepRUOT Matching：{title}"
        method_description = (
            f"<p>增长惩罚为 α ψ(g)，α={config['growth_penalty']:g}、ψ(g)=g²/2；σ={config['model_sigma']:g}。优化对象是删除群体修正后的重整化近似目标，"
            "进一步受对角高斯与网络参数化限制。这里没有声明精确求解原连续 RUOT。</p>")
        if pair_cfg.get("cost_objective") == "logmoment_upper_bound":
            method_description = ("<p>Log-moment upper-bound objective: single-path Gaussian moment correction; receiving coupling KL. "
                                  "Compatible controls and Matching losses are unchanged. Numerical quadrature/interpolation and "
                                  "approximate Matching do not constitute a certified rollout Action bound.</p>")
        if pair_cfg.get("cost_objective") == "tight_logmoment_upper_bound":
            method_description = ("<p>Tighter log-moment bound: positive-part log moment, receiving conditional entropy, "
                                  "and analytic optimal compatible controls for the new objective. Young coefficient remains alpha. "
                                  "Numerical interpolation and Matching are not a certified rollout Action bound.</p>")
        if pair_cfg.get('cost_objective') == 'gaussian_reference_upper_bound':
            method_description = (
                '<p>Gaussian bridge-mixture reference upper bound: reference variance is '
                'sigma² t(1-t)+s_data t², with s_data derived from observed endpoint second moments. '
                'Young coefficient stays alpha; all log moments and optimal compatible controls are analytic. '
                'Coupling uses receiving conditional entropy. This is an experimental candidate, '
                'not a certified numerical bound on learned rollout Action.</p>')
        if pair_cfg.get('cost_objective') == 'diffusion_full_square_upper_bound':
            method_description = (
                '<p>Diffusion-scale full-square bound: the pair correction is '
                'sigma²/2 E[(g+log(phi/q))²]+2 sigma²/e². The common reference has variance '
                'sigma² t(1-t)+s_data t², with s_data determined by the observations. '
                'The analytic inner controls use growth weight alpha+sigma²; Coupling uses '
                'receiving conditional entropy. Matching losses and training budgets are unchanged. '
                'No positive-part quadrature, Hellinger term, or additional endpoint mass correction is used. '
                'The final learned rollout Action is evaluated separately from the theoretical bound.</p>')
        if pair_cfg.get("parameterization") == "axial_distance":
            method_description = method_description.replace("对角高斯", "沿位移/垂直位移两类方差的轴对称高斯")
            method_description += "<p>路径网络仅输入时间、距离、对数质量比；均值沿端点连线，两个协方差特征值保留布朗型端点斜率。</p>"
        if config.get("distillation", {}).get("enabled"):
            method_description += "<p>普通 tanh MLP 在距离/质量比网格上蒸馏路径代价，按配置步数结束，独立误差仅作报告。Coupling 直接查询连续蒸馏网络并自动微分，不使用 Hermite 插值，也不调用条件路径时间求积。</p>"
        if config['coupling'].get('cost_evaluation') == 'shared_distance_hermite':
            method_description += "<p>直接从冻结路径网络批量计算共享距离/质量比网格上的 C 和 C_r，不进行蒸馏。距离方向线性插值、质量比方向三次 Hermite 插值，C_r 为同一插值函数的导数。投影梯度迭代仅查询 GPU 驻留表；插值误差及建表耗时单独记录。</p>"
    parts = [f"<h1>{heading}</h1>", method_description,
             f"<p>设备：{html.escape(str(env['gpu']))}。种子：{config['seed']}。Coupling 模式：{config['coupling']['mode']}，"
             f"方法：{html.escape(str(config['coupling'].get('method', config['coupling'].get('algorithm', 'unspecified'))))}。</p>",
             f"<p>路径：{pair_training}。反馈网络训练配置：{config['matching']['steps']} 步。实际完成量见下表。</p>",
             "<p>Flow 损失使用 √(t(1−t)) 加权速度误差。Matching 不构造验证集，最终以生成评估为准。</p>"]
    if config.get("max_observed_points"):
        parts.append(f"<p>这是工程验证配置：每快照最多保留 {config['max_observed_points']} 个训练点，保留原始总质量。POT 评估仍使用全部官方观测点。</p>")
    parts.append(f"<p>训练与评测使用全部 {config['dimension']} 个坐标；快照散点图仅展示 x1/x2 投影。</p>")
    if config["coupling"]["mode"] == "minibatch":
        blocks = config["coupling"].get("minibatch_blocks", "按块大小提示自动确定")
        order = config["coupling"].get("partition_order", "dataset_default")
        parts.append(f"<p>使用 {blocks} 个按数据集默认顺序切出的源/目标块，并逐块配对求解；"
                     f"RUOT 与 WFR-FM 使用相同分块规则（顺序：{html.escape(str(order))}）。"
                     "这是受限块对角支持的 minibatch-OT 近似，不等于全局 OT。每个观测点保留原质量，不平均条件质量比。</p>")
    parts.append("<h2>完成状态与耗时</h2><pre>" + html.escape(json.dumps(status, ensure_ascii=False, indent=2)) + "</pre>")
    failure = run / "coupling_failure.json"
    if not completed and failure.exists():
        parts.append("<h2>未满足的条件</h2><p>以下失败已保存；失败 Coupling 不会用于正式 Matching，也不生成伪造的最终指标。</p><pre>" + html.escape(failure.read_text(encoding="utf-8")) + "</pre>")
    if completed:
        metrics = pd.read_csv(run / "metrics.csv")
        parts.append("<h2>逐时刻指标</h2>" + metrics.to_html(index=False, float_format=lambda x: f"{x:.6g}"))
        parts.append("<p>W1、W2 分别对欧氏距离与平方欧氏距离调用 POT emd2。质量误差从未归一化的 log_weights 计算；归一化只用于形状评估。第一个时刻使用随机抽取的初始粒子，因此形状误差可以非零。</p>")
        parts.append(f"<p>从 t={config['model_times'][0]} 一次连续 rollout 到 t={config['model_times'][-1]}，没有中间快照重启，没有分裂、复制或删除粒子。所有粒子的原始对数权重和 ESS 均保留。</p>")
        for file in ("snapshots.png", "mass_and_distance.png"):
            parts.append(f'<img src="{file}" style="max-width:100%">')
        sensitivity = run / "step_sensitivity.csv"
        if sensitivity.exists():
            parts.append("<h2>步长与端点保护检查</h2><p>使用相同初始粒子和耦合布朗增量，比较配置步长与半步长；另检查保护大小乘 10 的影响。</p>" + pd.read_csv(sensitivity).to_html(index=False))
        comparison = run / "comparison_to_ruot.csv"
        if comparison.exists():
            parts.append("<h2>与相同配置 RUOT 结果对照</h2>" + pd.read_csv(comparison).to_html(index=False, float_format=lambda x: f"{x:.6g}"))
    action_file = run / "action.json"
    if action_file.exists():
        action = json.loads(action_file.read_text(encoding="utf-8"))
        action_rows = pd.DataFrame(action["intervals"])
        parts.append("<h2>训练后反馈模型的 RUOT 运行作用量</h2>"
                     "<p>按加权粒子估计 ∫ρ[|U|²/2 + αG²/2]dt，其中 U=V+σ²Score/2、ψ(g)=g²/2。"
                     "这是训练后近似反馈动力学在当前 Euler 网格上的数值作用量，不代表原 RUOT 问题的精确最优值。</p>"
                     + action_rows.to_html(index=False, float_format=lambda x: f"{x:.8g}")
                     + f"<p>总 Action：{action['total_action']:.8g}；粒子 Monte Carlo 标准误："
                     f"{action['particle_mc_standard_error']:.3g}；每单位时间 {action['steps_per_unit']} 步。</p>")
        check = action.get("discretization_check")
        if check:
            parts.append(f"<p>相同布朗增量下，{check['coarse_steps_per_unit']} 步与 "
                         f"{check['fine_steps_per_unit']} 步 Action 的相对差为 "
                         f"{check['relative_difference']:.3%}。</p>")
    parts.append('<h2>训练曲线</h2><img src="training.png" style="max-width:100%">')
    parts.append("<p>配置、锁定版本、路径 checkpoint、反馈 checkpoint、成对代价缓存、P/Q/r、约束残差、粒子位置和对数权重保存在本目录。配置停止量不等同于训练已收敛；没有已有方法对照时不宣称速度或精度优越。</p>")
    (run / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>DeepRUOT experiment</title><style>body{font:16px sans-serif;margin:40px;max-width:1400px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}pre{white-space:pre-wrap}</style>' + "\n".join(parts), encoding="utf-8")


def sensitivity(model, truth, config):
    cfg = config["generation"]
    n = min(cfg["sensitivity_particles"], cfg["n_particles"])
    steps = cfg["steps_per_unit_interval"]
    rng = np.random.default_rng(config["seed"] + 202)
    initial = truth[0].x[rng.integers(len(truth[0].x), size=n)]
    n_intervals = len(config["model_times"]) - 1
    dimension = truth[0].x.shape[1]
    fine_noise = rng.normal(size=(n_intervals * 2 * steps, n, dimension)) / np.sqrt(2 * steps)
    coarse_noise = fine_noise.reshape(n_intervals * steps, 2, n, dimension).sum(1)
    kwargs = dict(model=model, first_snapshot=truth[0].x, times=config["model_times"], n_particles=n, sigma=config["model_sigma"],
                  seed=config["seed"] + 202, initial_x=initial)
    coarse = rollout(**kwargs, steps_per_unit=steps, endpoint_guard=cfg["endpoint_guard"], normal_increments=coarse_noise)
    fine = rollout(**kwargs, steps_per_unit=2 * steps, endpoint_guard=cfg["endpoint_guard"], normal_increments=fine_noise)
    guard = rollout(**kwargs, steps_per_unit=steps, endpoint_guard=cfg["endpoint_guard"] * 10, normal_increments=coarse_noise)
    rows = []
    for k in range(len(truth)):
        mc, mf, mg = [float(np.exp(logsumexp(result["log_weights"][k]))) for result in (coarse, fine, guard)]
        rows.append(dict(time=k, N_particles=n, coarse_steps=steps, fine_steps=2 * steps,
                         coarse_mass=mc, fine_mass=mf, guarded_mass=mg,
                         step_position_rms=float(np.sqrt(np.mean(np.square(coarse["positions"][k] - fine["positions"][k])))),
                         step_mass_relative_difference=abs(mc - mf) / mf,
                         guard_position_rms=float(np.sqrt(np.mean(np.square(coarse["positions"][k] - guard["positions"][k])))),
                         guard_mass_relative_difference=abs(mc - mg) / mc))
    return pd.DataFrame(rows)


def generate_and_evaluate(config, run):
    device = configure(config)
    run = init_run(run, config)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if status.get("matching", {}).get("status") != "completed":
        raise RuntimeError("completed Matching is required for evaluation")
    truth = snapshots_for(config, full=True)
    model = make_feedback(config, device)
    cp = torch.load(run / "feedback.pt", map_location=device, weights_only=False)
    validate_feedback_architecture(cp, config)
    if cp["path_checkpoint_sha256"] != file_hash(run / "pair_path.pt") or cp["coupling_manifest_sha256"] != file_hash(run / "couplings" / "manifest.json"):
        raise RuntimeError("feedback checkpoint is stale")
    model.load_state_dict(cp["model"])
    model.eval()
    cfg = config["generation"]
    update_stage(run, "generation", "running")
    start = time.perf_counter()
    generated = rollout(model, truth[0].x, config["model_times"], config["model_sigma"], cfg["n_particles"],
                        cfg["steps_per_unit_interval"], config["seed"] + 300, cfg["endpoint_guard"])
    np.savez_compressed(run / "generated.npz", **generated, feedback_checkpoint_sha256=file_hash(run / "feedback.pt"))
    update_stage(run, "generation", "completed", seconds=time.perf_counter() - start, particles=cfg["n_particles"],
                 steps_per_unit=cfg["steps_per_unit_interval"], endpoint_guard=cfg["endpoint_guard"])
    update_stage(run, "evaluation", "running")
    start = time.perf_counter()
    rows = []
    for k, snapshot in enumerate(truth):
        row = evaluate_snapshot(generated["positions"][k], generated["log_weights"][k], snapshot.x,
                                n_initial_observed=truth[0].original_count, mass_true=snapshot.mass)
        rows.append(dict(time=k, N_observed=len(snapshot.x), N_generated=cfg["n_particles"], seed=config["seed"] + 300, **row))
        print(f"t={k}: W1={row['W1_normalized']:.5g}, W2={row['W2_normalized']:.5g}, mass={row['mass_pred']:.5g}/{row['mass_true']:.5g}", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run / "metrics.csv", index=False)
    update_stage(run, "evaluation", "completed", seconds=time.perf_counter() - start, solver="POT emd2", exact_all_observations=True)
    start = time.perf_counter()
    sensitivity(model, truth, config).to_csv(run / "step_sensitivity.csv", index=False)
    update_stage(run, "sensitivity", "completed", seconds=time.perf_counter() - start)
    plot_results(run, truth, generated, metrics)
    write_report(run, config)
    return metrics
