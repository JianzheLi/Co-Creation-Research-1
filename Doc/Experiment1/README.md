# 实验 1：Action 与 rollout

本目录收录两组便于审阅的结果摘要：

- Gene、EMT、Dygen：`alpha=2`，`sigma=0.1, 0.25, 0.5, 1.0`，RUOT 与 WFR+Brownian Bridge 的 Action/分布指标；每个固定模型有 20 次 Monte Carlo 评测。见 `summary_three_datasets.csv`、`summary_three_datasets.json` 和配对样本表。
- Gene、Mouse、EMT：`alpha=10` 的 20 次 Action 比较及 RUOT 连续 rollout 图，见 `Figures/action_alpha10_three_datasets.pdf` 与 `Figures/ruot_rollouts_three_datasets.pdf`。

网页报告为 `report_three_datasets.html`。原始模型、全部粒子样本与运行日志保存在 `../../Results/` 对应实验目录；summary 中的 `run` 字段使用相对路径，便于移动仓库。
