# EB / Weinreb Matching 与 SDE 稳定性复跑

## 设置

两组均复用原实验的 Conditional Path、Coupling 和数据划分，仅重新训练 Matching 30,000 步。沿用原学习率计划（cosine，3e-3 到 1e-5）、batch size 2048、端点参数化、sigma=0.5，以及各数据集原有生长惩罚和 Growth loss 权重。四个回归场（regular velocity、singular velocity、growth、score）均使用非仿射 LayerNorm 输出归一化；从原 Matching 权重 warm-start。训练目标未改，旧实验目录保持原样。

## 结果

| 数据集 | alpha / Coupling 块数 | 20 个 SDE 种子 | 平均质量相对误差（新模型） | 原模型平均质量相对误差 | 新模型 post-t0 最大坐标绝对值 |
|---|---:|---:|---:|---:|---:|
| EB | 1225 / 8 | 20/20 完成，无发散 | 0.735% | 0.791% | 58.13 |
| Weinreb | 225 / 16 | 20/20 完成，无发散 | 0.987% | 1.381% | 87.04 |

EB 各时间点的平均质量误差（新模型）为 1.089%、0.263%、1.156%、0.430%；Weinreb 为 1.519%、0.456%。两者所有种子都完成了全量初始点 rollout，未出现非有限值、rollout 失败或位置/权重爆炸。原模型在这两个数据集的既有 20 种子检查中也均未发散，因此这次实验说明新结构没有破坏稳定性，但不能据此说它修复了这两个数据集上的发散问题。

空间拟合方面，新结构的 W1 略有退化。EB 对相同的两个固定种子，终点 W1 从 13.498/13.510 变为 13.889/13.952（约高 3.1%）；Weinreb seed 342 的 W1 在 t=1、2 分别从 4.858、5.626 变为 4.946、5.796（约高 1.8%、3.0%）。这些 W1 对照只用了固定种子；20 个种子的完整 rollout 和质量误差均已检查。全量点上的 W1 计算耗时明显更长，所以没有对 20 个种子全部重复求 W1。

## 文件

- EB 新模型：`eb_noscale_leakyrelu_allfield_outputnorm_warmstart_w512x8_a1225_sigma05_blocks8_fm30000_seed42`
- Weinreb 新模型：`weinreb_silu_allfield_outputnorm_warmstart_w512x8_a225_sigma05_blocks16_fm30000_seed42`
- 两个新模型的完整逐种子 SDE 记录均在各自 `evaluation_all_initial_points/checkpoint_architecture_20seeds_stability.json`。
