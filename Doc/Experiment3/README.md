# 实验 3：Hold-One-Out

本目录收录基线阶段状态和留出中间时间点的 W1 结果。`baseline_holdout_w1.csv` 与 `holdout_baseline_comparison.csv` 汇总基线，`ours_holdout_w1.csv` 汇总 RUOT Matching。任务在每个数据集上逐个留出内部观测时间点，只用其余时刻训练，再在压缩后相应时刻评测预测 W1。

完整训练和预测产物保存在 `../../Results/experiment3/` 与 `../../Results/experiment3_affine_silu/`。CSV 的运行目录使用相对 `Results/` 的路径；原始输出不纳入提交。
