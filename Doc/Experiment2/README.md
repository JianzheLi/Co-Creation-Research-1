# 实验 2：分布匹配

本目录记录 Cite、EB-Noscale、Weinreb 真实数据集上各方法的重建汇总。`all_timepoint_metrics.csv` 保留逐时间点指标，`method_dataset_summary.csv` 汇总各方法的 W1 和质量误差；`Figures/` 提供跨数据集的 W1 图。诊断说明记录 LayerNorm、匹配稳定性和输出范数相关检查。

完整模型、各时间点训练目录与评测细节保存在 `../../Results/experiment2/`。这些完整运行产物不纳入提交；更新实验时应保留独立运行目录并刷新本目录的轻量汇总。

[2026-10-23 EB 调优记录](eb_tuning_20261023.md)单独记录固定 σ=0.5、α=1225 下的 13 组配置比较、独立 Matching 重训、留出预测及已有 baseline 清单。对应的新运行目录为 `../../Results/eb_tuning_20261023/`，历史汇总表保持原样。
