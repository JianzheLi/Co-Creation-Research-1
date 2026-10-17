# Simulation-Free DeepRUOT

本仓库研究如何从离散时刻的观测分布求解带扩散和质量变化的随机最优控制问题。给定端点分布 \(\rho_0,\rho_1\)，控制位置漂移 \(U_t(x)\) 和增长率 \(G_t(x)\)，等价地令 \(dX_t=U_t(X_t)dt+\sigma dB_t\)、\(d\log w_t=G_t(X_t)dt\)，求解

\[
\min_{\rho,U,G}\int_0^1\!\int \rho_t(x)\left(\frac12\lVert U_t(x)\rVert^2+\frac{\alpha}{2}G_t(x)^2\right)\,dx\,dt,
\quad \partial_t\rho=-\nabla\!\cdot(\rho U)+\frac{\sigma^2}{2}\Delta\rho+G\rho,
\quad \rho_{t=0}=\rho_0,\;\rho_{t=1}=\rho_1.
\]

目标是构造 Simulation-Free 求解流程：学习端点条件路径，计算成对代价并求解 Coupling，再用 Flow Matching 拟合反馈控制，最后通过带权粒子 SDE rollout 评估。完整目标中难以计算的三项修正分别涉及后验对数项、漂移残差的后验均值和生长率的后验均值；它们使目标难以写成独立的成对代价。实验主线暂时删去这三项，优化可分解的重整化代理目标。因此，当前方法是原 RUOT 的近似，尚未证明删项不改变最优解，也不把代理目标称作 RUOT Action 的上界。

## 当前实验采用的默认路线

- 取 \(\psi(g)=g^2/2\)；\(\alpha\)、\(\sigma\) 和训练预算由各实验 YAML 明确设置。
- 条件路径采用轴对称高斯参数化。三个路径网络分别拟合均值、质量和沿路径/垂直路径的方差，输入时间、端点距离和质量比；Matching 从完整条件高斯分布采样。
- 不蒸馏成对代价网络。冻结路径后直接计算距离—质量比网格上的代价及所需导数，使用距离方向线性插值、质量比方向分段三次 Hermite 插值查询代价表。
- Coupling 默认使用投影镜像下降（Projected Mirror Descent）求两张 Coupling；全矩阵或按块求解由具体数据集配置决定。
- 当前端点正则化 Matching 配置使用四个相互独立的网络，分别拟合有限速度、奇异速度、生长和 Score；推理时按公式还原完整场。网络宽度、层数、激活和步数以所用 YAML 为准。
- 最终从初始时刻进行连续带权粒子 SDE rollout，并在仓库根目录 `Results/` 保存轨迹和评测结果。

上述是当前实验默认路线，不代表其他配置或研究分支也采用相同设置。三个后验修正项的处理仍是主要理论问题；相关推导和实验候选见 `Doc/`，不要据此把当前代理目标解释成已证明的原问题上界。

## 环境与运行

仓库根目录包含 `Code/`、`Doc/` 和 `Manuscript/`。推荐使用 Python 3.11；从仓库根目录进入 `Code/` 后安装依赖并运行。数据 CSV 已包含在仓库中，普通训练不需要先下载 Baselines 或 WFR-FM 仓库。

```powershell
conda create -n ruotmatching python=3.11 -y
conda activate ruotmatching
python -m pip install --upgrade pip
Set-Location Code
python -m pip install -r requirements.txt
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python -m pytest -q
python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

如果有 NVIDIA GPU，请安装与本机驱动匹配的 CUDA 版 PyTorch，并确认 `torch.cuda.is_available()` 为 `True`。配置中的 `device: cuda` 会要求可用 GPU；没有 GPU 时可将所用 YAML 中的 `device` 改为 `cpu`，完整实验会明显变慢。

小型 smoke 配置用于快速检查流水线；完整实验请使用对应数据集的实验 YAML。若在仓库根目录运行，`--run-dir` 使用 `Results/...`；若先进入 `Code/` 运行，同一个输出位置应写成 `../Results/...`：

```powershell
# 从仓库根目录运行
python Code/scripts/run_experiment.py --config Code/configs/smoke_verified.yaml --run-dir Results/smoke_verified

# 或从 Code/ 目录运行
Set-Location Code
python scripts/run_experiment.py --config configs/experiment1_gene_alpha2_sigma0p25_ruot.yaml --run-dir ../Results/experiment1_gene_alpha2_sigma0p25_ruot
```

所有数据集及来源见 [`Code/data/README.md`](Code/data/README.md)。实验 2 的 CITE、EB 和 Weinreb 数据已放在 `Code/data/`。rollout 评估脚本会自动创建 `--run-dir`，并将连续轨迹 `generated.npz`、指标和报告写入该运行目录。因此从仓库根目录传 `Results/<实验名>`，或从 `Code/` 传 `../Results/<实验名>`，都会在仓库根目录生成 `Results/`。评估脚本读取对应运行目录，因此应先完成训练，再运行评估。`Results/` 已加入 `.gitignore`，不会随代码和数据提交。

## 协作开发（Fork + Pull Request）

协作者在自己的 Fork 中开发，通过 Pull Request（PR）提交改动；维护者审查后再合并到主仓库的 `main`。协作者不需要主仓库写权限，也不要直接在 `main` 上开发。

1. 在 GitHub 打开主仓库，点击 **Fork**，把项目复制到自己的账号。
2. 克隆自己的 Fork，并添加主仓库为 `upstream`：

   ```powershell
   git clone https://github.com/<GitHub用户名>/Co-Creation-Research-1.git
   cd Co-Creation-Research-1
   git remote add upstream https://github.com/ZerooVector/Co-Creation-Research-1.git
   git remote -v
   ```

3. 开始新工作前同步主仓库：

   ```powershell
   git switch main
   git fetch upstream
   git merge --ff-only upstream/main
   git push origin main
   ```

4. 为每项工作创建独立分支：

   ```powershell
   git switch -c research/<简短主题>
   ```

5. 按本 README 配置环境、运行相关测试和实验。数据在 `Code/data/`；运行结果写入根目录 `Results/`，该目录已被忽略，不要提交模型、日志或结果文件。
6. 只暂存本次修改的代码、配置和文档，检查状态后提交并推送：

   ```powershell
   git add <本次修改的文件>
   git status
   git commit -m "描述本次改动"
   git push -u origin research/<简短主题>
   ```

7. 在 GitHub 创建 PR，目标仓库为 `ZerooVector/Co-Creation-Research-1`，目标分支为 `main`。说明研究目的、改动、复现命令和测试结果；实验结果生成后写明本地 `Results/` 路径。若审查后需要修改，继续推送同一分支，PR 会自动更新。
8. PR 合并后，用第 3 步的命令同步主仓库更新。维护者可为 `main` 设置 PR 审查和分支保护规则，详见 [GitHub 分支保护文档](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule)。

## WFR-FM 启发式对照

`configs/gene_wfr_bridge.yaml` 保持同一基因模拟数据、α=10、σ=0.25、反馈网络、2048 批量、30000 步 Matching 和 2000 粒子生成评测。成对路径改用解析 WFR 均值与质量，不做路径网络预训练；条件方差按本实验指定的布朗桥 `σ²t(1−t)` 设置。WFR-FM 原论文采用常方差 traveling Gaussian，因此这是用布朗桥方差构造的启发式比较。

这里 `ψ(g)=g²/2` 对应 WFR 参数 `δ²=α`。静态 Dirac 半耦合代价为 `2α(P+Q−2κ√(PQ))`，其中 `κ=cos(min(||x−y||/(2√α), π/2))`。等价 OET 耦合按 WFR-FM 的两端 KL 系数 `[1,1]` 求解；该方法没有 RUOT 中 `ε=σ²` 的联合熵项，因此不把 RUOT 的数值熵正则误当成 WFR 生长惩罚。

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python scripts/run_wfr_heuristic.py --config configs/gene_wfr_bridge.yaml --run-dir ../Results/gene_wfr_bridge_alpha10_sigma025_seed42
```

运行会保存 WFR coupling、反馈模型、粒子评测、与 RUOT 指标对照表和训练后作用量。

Windows 下 NumPy/MKL 与 PyTorch 各自携带的 Intel OpenMP DLL 曾发生冲突。工程在数值库导入前设置 `MKL_THREADING_LAYER=SEQUENTIAL`，避免加载第二套 OpenMP；独立测试和性能测量也应按上方命令设置该变量。不要设置 `KMP_DUPLICATE_LIB_OK`。GPU 训练仍正常工作，PyTorch CPU 线程数由配置决定。

## Simulation-Gene 历史配置记录

| 配置 | 观测点 | 路径网络 | 路径预训练 | Coupling | Matching |
|---|---|---|---|---|---|
| `gene_steps_gpu.yaml`（早期全矩阵配置） | 全部 3031 点 | 32 宽、2 层 | 从头训练 4000 步，每条件 8 个时间 | 4 个全矩阵、GPU、最多 10000 轮 | 30000 步、余弦学习率、无验证集 |
| `gene_full_gpu.yaml`（已停止的历史运行） | 全部 3031 点 | 32 宽、2 层 | 复用已完成的 10 轮覆盖训练 | 全矩阵、GPU，流程未完成 | 未完成实验 |
| `gene.yaml` | 全部 3031 点 | 128 宽、3 层 | 10 轮 | 全连接、最多 1000 轮 | 50000 步 |
| `gene_practical.yaml`（已停止的分块诊断） | 全部 3031 点 | 32 宽、2 层 | 已完成 10 轮 | 约 64 点随机分块 | 原计划 10000 步，未完成实验 |
| `smoke.yaml` | 每时刻 16 点 | 16 宽、2 层 | 小规模 4 轮 | 全连接、最多 1000 轮 | 300 步 |
| `smoke_verified.yaml` | 每时刻 16 点 | 16 宽、2 层 | 小规模 4 轮 | 全连接、最多 10000 轮 | 300 步 |

`smoke.yaml` 的最初实验保留为失败诊断；其 Coupling 在 1000 轮时未满足 `1e-6` 容差。增加迭代上限后的工程检查保存在独立目录，避免改变已有实验的配置。

该配置的四个 Coupling 矩阵尺寸为 `400×442`、`442×530`、`530×690`、`690×969`，每个相邻区间使用全部源点和目标点。收敛阈值 `marginal_tolerance=0.002` 表示相对 L1 边缘残差，对应相对 Total Variation `0.001`：`relative_TV = sum(abs(residual)) / (2 * sum(target_mass))`。P 的行边缘和 Q 的列边缘均须满足该要求。

当前条件路径使用 `pair.training_mode: steps`，更新 **4000 步**。每步从全部 **1445370 个**相邻候选位移中均匀抽取 **2048 个**条件，独立采样 `r ~ Uniform(0.1,10)` 后计算 `log(r)`，每个条件均匀采样 **8 个**开区间时间。位移通过 `(k,i,j)` 索引按需构造；这不要求每轮无遗漏遍历全部位移，日志分别记录累计条件访问数与实际见到的唯一候选对数。`pair.training_steps` 控制更新次数，`pair.times_per_condition` 控制时间采样量。

Matching 使用 **30000 步**，学习率按余弦计划从 `1e-3` 降至 `1e-4`。设置 `matching.validation_every=0`，不构造 Matching 验证集，通过最终生成效果和评估指标判断结果。Flow 恢复使用原加权误差 `||sqrt(t(1-t)) V - target_scaled||²`，其加权标签在端点附近保持有限。

历史 `gene_practical` 已完成 10 轮全位移覆盖预训练，耗时约 **290.58 秒**：前三轮质量比分别固定为 0.1、1、10，随后对数均匀采样，每条件 4 个时间。`gene_full_gpu` 曾复用该 checkpoint，但两个历史运行均在完整流程完成前停止。它们保留作诊断；`gene_steps_gpu` 使用步数模式重新训练。

## 分阶段运行与恢复

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python scripts/train_pairs.py --config configs/gene_steps_gpu.yaml --run-dir ../Results/gene_steps_gpu_4000_seed42
python scripts/solve_couplings.py --config configs/gene_steps_gpu.yaml --run-dir ../Results/gene_steps_gpu_4000_seed42
python scripts/train_matching.py --config configs/gene_steps_gpu.yaml --run-dir ../Results/gene_steps_gpu_4000_seed42
python scripts/generate_and_evaluate.py --config configs/gene_steps_gpu.yaml --run-dir ../Results/gene_steps_gpu_4000_seed42
python scripts/compute_action.py --config configs/gene_steps_gpu.yaml --run-dir ../Results/gene_steps_gpu_4000_seed42
```

复用路径的辅助工具仍保留，适用于另建诊断实验；当前从头训练的主实验不执行该步骤。以下历史 coverage 示例要求目标目录**尚不存在**，工具核验路径阶段配置及来源，保留原预训练耗时和 checkpoint 哈希；它不复用旧分块 Coupling。复用实验的端到端计时仍包含来源预训练耗时。

```powershell
python scripts/reuse_pair_checkpoint.py --source ../Results/gene_practical_seed42 --config configs/gene_full_gpu.yaml --run-dir ../Results/gene_full_gpu_reuse_seed42
python scripts/run_experiment.py --config configs/gene_full_gpu.yaml --run-dir ../Results/gene_full_gpu_reuse_seed42
```

当前 steps 模式的路径每 500 步及最终保存 checkpoint；Matching 每 1000 步及最终保存。重新执行同一配置、同一输出目录，会恢复最近的 checkpoint，包括优化器和随机数状态。历史 coverage 模式每个完整覆盖轮次保存，中断时从该轮开头重跑。已有输出目录拒绝接受不同的配置，参数变更应使用新目录。

Coupling 必须成功收敛才能训练反馈网络；反馈 checkpoint 必须匹配路径与 Coupling 的哈希。可以只重做生成评估，不需要再训练。

为检查反馈在训练分布之外的外推稳定性，`gene_steps_gpu_tanh_matching.yaml` 与 `gene_wfr_bridge_tanh_matching.yaml` 只把反馈网络隐藏层激活从 SiLU 换成 tanh，线性输出层、Matching 损失和其余训练参数保持不变。tanh 隐藏表征使输出对极端时空输入有界，同时保留线性输出层拟合不同目标尺度。两个方法各自复用原成对路径与 Coupling，在新目录单独重训 Matching；可用 `scripts/compute_action_monte_carlo.py` 做两种方法的配对 Action Monte Carlo。

## 实现中的数值约定

### 距离条件路径与代价蒸馏

新配置 `configs/gene_distance_distilled.yaml` 在模拟基因数据上启用完整流程：

```powershell
python scripts/run_experiment.py --config configs/gene_distance_distilled.yaml --run-dir ../Results/gene_ruot_distance_distilled_seed42
```

`pair.parameterization: axial_distance` 使三个路径网络只输入 `(t, d/delta_scale, log(r))`。
设 `Delta=x1-x0`、`d=|Delta|`、`e=Delta/d`，均值为 `x0+[t+t(1-t)h_mu]Delta`，
协方差为 `s_parallel ee^T+s_perpendicular(I-ee^T)`。两类方差都使用
`s_j=sigma^2 t(1-t) exp(t(1-t)h_j)`，保留两端零协方差及 `+sigma^2 I`、`-sigma^2 I` 的时间导数。
方差网络输出两个原始标量 `a,b`，取 `h_perpendicular=B tanh(a)`、
`h_parallel=B tanh(a+(d/delta_scale)^2 b)`，因此零距离时自动各向同性，方差差异为 `O(d^2)`，没有新增尺度参数。
状态中的 `var/Z/K` 存储平行特征值及重复的垂直特征值；向量仍在实际坐标系中，
通过投影作用协方差、平方根和逆矩阵。Matching 从完整的旋转高斯分布采样，损失保持原定义。

路径预训练 3000 步后，在 `129 x 257` 的距离/质量比网格（质量比轴按对数取点）上生成代价及导数监督。
代价网络是普通三层隐藏层、每层 128 单元的 tanh MLP，无凸性结构或凸性惩罚。
仅使用 Adam，训练监督同时包括归一化代价和质量比导数 `dC/dr`；所有网格中点及独立随机点用于误差检查。
蒸馏固定训练 20,000 步（由 `distillation.training_steps` 设置），完成后直接保存 `cost_network.pt`，再次运行时复用。
独立误差仅作报告，不触发续训，也不作为蒸馏阶段的退出门槛。可用 `--stage distill` 单独运行此阶段。

`configs/gene_distance_distilled_strict.yaml` 使用更严格的蒸馏损失：对归一化代价误差和
`dC/dr` 误差分别计算 `mean(error^2) + max(error^2)` 后相加。
最大值取当前训练批次；每批 4096 点固定包含完整网格的 768 个边界点，其余均匀随机抽样。
因此少量边界大误差有单独的损失贡献，不会仅被均方平均稀释。没有新增监督目标、曲率惩罚或网络结构。
训练仍固定 20,000 步；该损失本身不是连续域凸性的数学保证，部署前和求根时仍需检查曲率。

`configs/gene_distance_distilled_continuous.yaml` 在同样的严格损失下设置
`sampling: continuous`、`max_error_weight: 1.0`。每批保留同样的 768 个边界监督点，
其余 3328 个条件每步独立从物理距离区间和物理质量比区间均匀采样（不是对 log(r) 均匀采样）。
这些点的 C 和 dC/dr 由冻结教师在 GPU 上即时求积和求导，不从网格插值；训练仍只拟合这两个量。
原网格保留用于相同的输出归一化、边界监督和独立比较，正式 Coupling 仍直接查询代价网络。

当前细网格实验使用 `configs/gene_distance_distilled_dense.yaml`：513×1025=525825 点，
距离轴均匀、质量比轴沿用对数均匀网格。C 与 dC/dr 在训练前一次性预计算，
归一化输入与两类标签全程驻留 GPU；优化步只抽取已存标签，不重复计算教师。
每步 4096 点含 768 个边界样本（始终包含四角），其余从完整细网格抽样。
最大误差权重仍为 1，固定训练 20000 步。连续采样作为可选实验保留，当前配置不启用它。

开启蒸馏后，Coupling 的代价、质量比一阶及二阶导数均直接查询 `cost_network.pt` 并自动微分，
质量比反解使用连续网络的导数进行带括区间的 Newton/二分求根，不构建 Hermite 表，也不运行路径网络或时间求积。
Coupling manifest 记录代价网络哈希，Matching 检查该哈希以拒绝过期 Coupling。
连续单调反解的理论前提仍是质量比方向凸性。按当前实验策略，蒸馏模型的非正曲率只报 warning 并记录，
不会自动续训或修改网络；Newton 步遇到非正曲率或越出括区间时回退二分。
列 Newton 的导数不可用时也回退二分。非有限查询、括区间失败或最终残差不收敛仍作为失败处理。
有非凸警告的结果不具有全局共轭最优性证明，边缘残差收敛也不提供该证明。
路径网络保留用于条件分布采样与 Matching 监督。旧配置默认保留原坐标对角参数化，以兼容历史 checkpoint。

`distillation_grid.pt` 保存教师网格，`distillation_history.jsonl` 保存蒸馏损失，
`distillation_validation.json` 报告独立样本上的代价/导数误差、实测曲率和 GPU 查询计时。

- 时间导数使用逐层传播的精确前向 JVP，避免对多个输出求和后混淆分量。JVP 仍可对网络参数和质量比求导；测试核验了与 PyTorch 自动微分、有限差分的对应关系。
- 使用稳定 `exprel`、有界的 `K=H·var` 和解析高斯空间期望训练有限被积函数。训练中不模拟条件 SDE，不直接累计发散能量再减大对数，不截断负的重整化代价。
- 冻结路径后使用固定 Gauss–Legendre 节点，并验证 32 与 64 节点的代价和两阶导数；必要时统一增加节点。
- 仅未开启蒸馏的旧配置继续使用关于 r 的分段三次 Hermite 代价缓存，并核验代价、导数及片段曲率。蒸馏配置直接查询连续网络，无插值、导数排序、凸包或额外凸惩罚。
- 未蒸馏的旧 Hermite 路径保留原凸性失败检查；连续蒸馏网络使用上述 warning 策略。有限点的曲率探测不构成全域凸性证明。
- Coupling 直接使用行归一化和固定行系数的逐列标量求根。质量比反解与列求根均有括区间保护，Newton 仅用作加速，失效时二分。核 `K` 随列变量变化，不是普通固定核 Sinkhorn。
- 当前 GPU 求解器执行与正文相同的双对偶算法，并行处理完整矩阵。代价查询的 `condition_chunk=4096` 仅控制网络求值和自动微分的显存工作集；结果保留在 GPU，不改变 Coupling 支持，也不构成 minibatch OT。
- 每次外层更新后同时检查 `P` 的行边缘和 `Q` 的列边缘。当前残差下的原始减对偶差只作诊断，不当作严格最优性证书。
- 质量比求解范围严格限定在 `[0.1,10]`。有限对偶求解器对总质量比恰好等于边界的情形明确返回 `finite_dual_boundary_unsupported`，不假装得到有限对偶解。
- Matching 采用 `M_k/p_k * m(t)` 的质量权重；Growth 标签为相容的空间相关增长，不替换成平均增长 γ。Flow 表示概率流速度，Score 输出原始 score；生成漂移只加一次 `σ²·Score/2`。
- Matching 从完整条件高斯分布采样 `x = xi(t) + sqrt(var(t)) * epsilon`，其中 `epsilon ~ N(0,I)`；标签在实际采样位置计算，因此包含偏离均值曲线的样本。

主要计算张量在 GPU 上驻留，CPU/GPU 传输集中在阶段读档、归档时的一次性批量传输，以及少量收敛判断和日志标量同步；最终 POT 评估使用 CPU。代价查询不会逐块将结果搬回 CPU，Coupling 不会在每次迭代中搬运完整矩阵。

旧 `gene_practical` 的分块 OT 将同一时刻的点随机分成约 64 点的源、目标块，会限制 Coupling 支持。该运行已停止，保留作诊断和历史路径预训练来源。当前 `gene_steps_gpu` 使用上述四个全矩阵。

## 数据与评估

官方 CSV 固定为 commit `e9b06bca530b4f442c829692bf43b656d0a1af4e`，下载后核验 SHA256 `d41e3bcd94ff55383264461f2eed048780dbea8faffb77011cff9b8dce231d10`。没有逐时刻坐标标准化，也没有将快照总质量归一化后送入 Coupling。

五个模型时刻为 0、1、2、3、4，点数分别 400、442、530、690、969，总质量分别 1、1.105、1.325、1.725、2.4225。小规模配置下采样后仍保留这些总质量。

生成从第一个官方快照均匀抽取固定数量粒子、赋初始权重 `1/N_gen`，一次连续演化到最后时刻，中途不重启、不分裂、不删除、不重采样、不强制归一化。`log_weights` 始终保留原始质量。首次形状误差可能因为有限初始抽样而非零；相同点云零误差的评估器测试单独通过。

评估对**全部官方观测点和全部生成粒子**调用 POT 精确 EMD。W1 以欧氏距离为代价；W2 以平方欧氏距离为代价并在最后开方。双方概率归一化仅用于这两个形状距离，质量误差由原始对数权重单独计算。检查 POT 求解状态并报告 ESS。

## 实验输出

EMT 官方数据实验使用 `configs/emt_distance_hermite_primal.yaml` 和
`configs/emt_wfr_bridge.yaml`，均为 α=10、σ=0.25。保留 CSV 的 x1 至 x10 全部坐标，
时刻 0、1、2、3 的样本数为 577、885、788、883；没有降维或额外标准化。
两种方法均使用完整 Coupling，RUOT 预训练 3000 步，Matching 均为 30000 步。
通用 CSV 读取器核验坐标数与配置一致，避免将高维数据静默截为二维。
W1、W2 和 Action 在十维空间计算，快照散点图仅显示 x1/x2 投影。
下载字节由配置中的 SHA256 固定，训练及最终评测使用相同文件。

`configs/gene_distance_hermite_primal.yaml` 不做蒸馏：直接批量查询冻结路径，建立所有区间共享的
距离/质量比表（初始 513×257），以距离线性、质量比三次 Hermite 插值提供 C 和同一函数的 C_r。
在全部单元中心、边界和随机点验证插值误差；必要时只加密网格，不重训路径。
投影梯度只查询 GPU 表，后续运行停止阈值为驻点残差 `1e-3` 和边缘相对 TV `1e-3`。
`distance_table_validation.json` 单独报告建表耗时和误差，`couplings` 报告各段优化耗时。

`configs/gene_distance_dense_newton_curvature.yaml` 复用距离条件路径，用稠密网格预计算
`C`、`C_r`、`C_rr` 三个教师目标，并按同一代价尺度归一化。
普通 MLP 的一阶、二阶自动微分同时参与 mean+max 损失，三项权重均为 1，固定 20000 步。
部署仍查询同一个连续代价网络；Coupling 使用带二分保护的 Newton 行列对偶更新，
按边缘相对 TV 小于 `1e-3` 停止，不使用原始变量算法的驻点残差阈值。
代价查询每批 32768 对，教师路径求积每批 4096 对以控制高阶自动微分显存。

`configs/gene_distance_dense_primal.yaml` 直接优化两个 semi-coupling：
`sum P*C(d,Q/P) + epsilon*P*(log(P)-1)`，其中 `epsilon=sigma^2`。
对 P 和 Q 的梯度分别是 `C-r*C_r+epsilon*log(P)` 和 `C_r`。
使用保持正值的 KL 投影梯度（镜像梯度），不做质量比反解或列变量求根，也不使用 Crr 更新。
KL-Dykstra 联合投影强制 `P*1=a`、`Q^T*1=b` 和 `ratio_min*P <= Q <= ratio_max*P`；
边缘单独归一化不被当作联合投影。目标函数的回溯检查自适应选择步长。
当前配置停止同时要求边缘相对 TV 小于 `1e-3` 以及归一化投影梯度残差小于 `1e-3`，因此不会把首次投影可行当作收敛。
代价查询每批 `32768` 对（仅计算分批，仍优化完整耦合矩阵）；每步诊断与停止量合并传回 CPU，
域检查和有限值检查也合并同步，保留逐步收敛判断、异常检查和完整历史。
原始目标和所有优化残差保存于各区间历史。此方法不求对偶变量，归档中 `log_s`、`lam` 明确记为 NaN；
Matching 只读取 P、Q、r，并核验边缘、代价网络哈希和收敛状态。

每个输出目录包含：

- `config.yaml`、`environment.json`、`environment-lock.txt`、`status.json`：配置、环境、种子、GPU 和完成状态。
- `pair_path.pt`、`pair_steps.jsonl`、`pair_validation.jsonl`：当前 steps 模式的路径模型、实际候选对覆盖量、损失分项、梯度和路径检查；历史 coverage 模式使用 `pair_history.jsonl`、`pair_batches.jsonl`。
- `quadrature_validation.json`、`cost_cache/`：确定性查询精度、模型哈希、位移哈希和统一查询缓存。
- `couplings/`：每个区间/块的 P、Q、r、log_P、行/列索引、边缘质量、对偶变量、收敛历史和目标值。
- `feedback.pt`、`matching_history.jsonl`：统一反馈模型和回归训练记录。
- `generated.npz`、`metrics.csv`：连续 rollout 的位置、真实对数权重、时间、步长与逐时刻指标。
- `action.json`、`action_intervals.csv`：用已训练的 (V,G,S) 计算的加权粒子运行作用量，漂移取 (U=V+\sigma^2S/2)，增长成本取 \(\alpha G^2/2\)；记录动能项、增长项、区间拆分及粒子 Monte Carlo 误差。
- `step_sensitivity.csv`：相同初始位置和耦合布朗增量下，原步长/半步长和端点保护变更的影响。
- `snapshots.png`、`mass_and_distance.png`、`training.png`、`report.html`：图表和结果说明。
- 失败时的 `coupling_failure.json`：适用条件或迭代失败的具体诊断，失败结果不会用于正式 Matching。

实际训练达到配置轮数不等同于已达到最优；优劣判断需要进一步基线对照。报告分别记录路径预训练、查询建表、Coupling、Matching、生成、POT 评估时间，端到端时间应包括前处理与训练全部阶段。

## 核验与来源

### 扩散尺度完整平方矩上界：独立实验分支

`pair.cost_objective: diffusion_full_square_upper_bound` 使用相同的数据确定参考
`q=N(x0,[sigma² t(1-t)+s_data t²] I)`，将 Young 系数从 α 改为 σ²。
新增成对修正是 `integral m [sigma²/2 E(g+log(phi/q))²+2 sigma²/e²] dt`。
固定路径的控制目标为 `E[|u|²/2+(alpha+sigma²)g²/2+sigma² g log(phi/q)]`；
解析控制中的增长权重因此是 `A=alpha+sigma²`，而不是旧分支的 `2 alpha`。
新控制贯穿预训练、代价建表和 Matching 标签；Coupling 使用接收端条件熵。
未叠加 Hellinger、正部矩或端点质量改进项；不改变 Matching 损失、训练预算和分块。
在一致正则的高斯路径族及适当参考下，修正为 O(sigma² log² sigma) 并趋零；
该结论不自动适用于随 sigma 发散的网络参数族，也不构成最终反馈 Action 的数值认证。

运行 `scripts/run_gaussian_reference_suite.py --candidate diffusion_full_square --dataset gene`
（也可选 mouse、emt），复用同架构、同配置的已完成原方法结果，对新目标从头训练完整流程，
以 20 个共同 rollout 种子比较 Action、漂移/生长分解、W1 和质量误差。
输出独立保存在 `../Results/diffusion_full_square_comparison`；此前算法、模型不覆盖。
修改前源代码备份在 `../Results/algorithm_backups/before_diffusion_full_square/src`。

### 数据确定的高斯参考上界：实验候选，尚未确认改善 Action

`gaussian_reference_upper_bound` 保留 Young 系数 α，不拟合、不搜索这个系数。
参考密度取 `q_t=N(x0,vq(t) I)`，其中
`vq(t)=sigma² t(1-t)+s_data t²`。
`s_data` 是各相邻时间区间的独立源/目标抽样位移平方的期望除以维数，再对区间等权平均。
它从观测数据直接计算，配置中保存其数值用于复现；加载配置会核验该值，不能把它当作可调超参数。
这是随机高斯终点的布朗桥混合参考，不需要查询目标混合密度或后验。

令 `ell=log(phi/q)`。成对代价使用完整的解析高斯对数平方矩：

    C_ref = C_ren + integral m [alpha/2 E g² + sigma² E(g ell)
                               +sigma⁴/(2 alpha)(E ell²+4/e²)] dt.

对任意归一化参考 q，设 `p=sum a_j phi_j`、`L=log(p/q)`。
由 `r(log r)_+²` 的凸性和 `r(log r)_-² <= 4/e²`，
`E_p L² <= sum a_j E_phi_j ell_j²+4/e²`。
对 `-sigma² G L` 使用系数 α 的 Young 不等式，再使用条件 Jensen 不等式，得到上述逐对上界。
接收端条件熵与此前 tighter-bound 分支相同。
改变参考分布不改变这个证明；但新界并不逐点支配旧的正部矩界。

固定条件路径时，控制有关的目标为 `E[|u|²/2+alpha g²+sigma² g ell]`。
记 `y=x-mu`、`gamma=dot m/m`，在协方差特征坐标中，其唯一最优兼容控制为

    beta = (2 alpha I+Sigma)^(-1) [dot mu-sigma²(mu-x0)/vq],
    zeta_i = [dot s_i+sigma²-2 sigma² s_i/vq]/(4 alpha+s_i),
    g = gamma+beta^T y+1/2 sum zeta_i(y_i²/s_i-1),
    u_i = dot mu_i-s_i beta_i+(dot s_i-sigma²-s_i zeta_i)y_i/(2 s_i).

这些控制满足 `u=gradient(2 alpha g+sigma² ell)` 及条件 FP 方程。
预训练、建表、Coupling 和 Matching 使用同一新目标/控制；端点方差条件不变。
该分支不增加径向求积、网络、训练轮数或路径间交互；解析矩仍为 O(d)。
实际总耗时需要与对照实验一起报告，不能把相同复杂度误称为零额外运算。

`scripts/audit_bound_slack.py` 对已有模型计算同一候选上的松弛来源；它是离散后验诊断，
不是连续 Action 恒等式的数值认证。`scripts/run_gaussian_reference_suite.py` 完整重跑
原方法和新候选，统一 64×3 路径网络及 400×6 反馈网络，在三个数据集、两组参数上
用 20 个共同 rollout 种子评测。20 个种子不代表 20 次独立训练。
数值插值、离散端点和近似 Matching 仍不提供最终反馈 Action 的严格上界保证。
只有实验结果才能说明此候选是否降低实际 Action；未确认前不替换原方法。

### 更紧的 Bound 与其解析最优控制（Young 系数固定）

新配置 `configs/{gene,mouse,emt}_tight_bound.yaml` 使用
`pair.cost_objective: tight_logmoment_upper_bound`。前两版均保留；本次修改前
源文件另存于 `../Results/algorithm_backups/before_tight_bound/src/`。
Young 系数仍为 α，不做学习或自动调参。路径网络输入仍是距离、质量比、时间。

固定起点和时间，令混合密度 `p=sum a_j phi_j`，共同参考
`q=N(x0,sigma² t I)`，`ell_j=log(phi_j/q)`。因为
`h(r)=r(log r)_+²` 凸且 `r(log r)_-² <= 4/e²`，Jensen 给出

    E_p log²(p/q) <= sum_j a_j E_phi_j (ell_j)_+² + 4/e².

这比上一版右侧使用完整 `E ell_j²` 更紧。新成对代价为

    C_tight = C_ren + integral m [alpha/2 E g² + sigma² E(g ell)
                                 + sigma⁴/(2 alpha)(E ell_+² + 4/e²)] dt.

保留接收端熵分解中的负行 KL 后，Coupling 目标（省略固定端点常数）为

    sum_ij P_ij C_tight(d_ij,Q_ij/P_ij)
      + sigma² sum_ij Q_ij log[Q_ij / (s_i nu_j)],
    s_i=sum_j Q_ij, nu_j=b_j/sum_j b_j, row(P)=a, col(Q)=b.

它等于 `sigma²[KL_m(Q || a tensor nu)-KL_m(s || a)]` 的熵部分。
其 Q 梯度是 `C_r+sigma² log[Q/(s nu)]`，P 梯度是 `C-r C_r`。
只增加行归约，不需要后验或路径两两交互；分块时每块使用相应归一化 nu。

对于相同路径和控制，旧 Bound 减去新 Bound 精确等于

    sigma² KL_m(s || a)
      + sigma⁴/(2 alpha) sum_ij P_ij integral m_ij E(ell_ij)_-² dt >= 0.

仍然丢掉的负速度/生长混合方差没有在这里被宣称已精确计算；也不宣称这是所有可选上界中最紧的一个。

固定高斯路径时，控制有关的运行代价是
`E[|u|²/2 + alpha g² + sigma² g ell]`。它在 FP 线性约束下严格凸。
令 y=x-mu、gamma=dot m/m，在协方差特征坐标中 Sigma=diag(s_i)，定义

    beta=(2 alpha I+Sigma)^(-1)(dot mu-(mu-x0)/t),
    zeta_i=(dot s_i+sigma²-2 s_i/t)/(4 alpha+s_i).

唯一最优控制为

    g*=gamma+beta^T y+1/2 sum_i zeta_i(y_i²/s_i-1),
    u*_i=dot mu_i-s_i beta_i+(dot s_i-sigma²-s_i zeta_i)y_i/(2 s_i).

这同时满足 FP 方程及 KKT 条件 `u*=gradient(2 alpha g*+sigma² ell)`。
原端点方差条件保持不变。代码通过无消去误差的表达式计算 beta、zeta，
Matching 直接使用相同控制，保留 t(1-t) 的 flow 权重。

正部矩在轴向使用截断正态的四阶矩解析式，在垂直方向只剩一个卡方积分。
训练采用32点广义 Gauss-Laguerre；Coupling 前独立比较32/64、必要时64/128、128/256点，
同时核验 C、C_r、C_rr，再独立验证时间积分和 Hermite 表。
理论不等式对精确积分成立；这些数值容差及近似 Matching 仍不构成 rollout Action 的严格数值认证。

### Log-moment Bound 实验分支

`configs/{gene,mouse,emt}_logmoment_bound.yaml` 启用新目标；原配置仍默认
`pair.cost_objective: renormalized_surrogate`。修改前源文件另保存在
`../Results/algorithm_backups/before_logmoment_bound/src/`，旧模型和结果不覆盖。

新分支使用共同参考 `q=N(x0,sigma² t I)`，在成对重整化代价上加入
`integral m [alpha/2 E(g+sigma²/alpha log(phi/q))²+2 sigma⁴/(alpha e²)] dt`。
高斯矩解析计算；条件路径仍只输入距离、质量比、时间。建表的 C、C_r
及 Coupling 查询均来自这个新代价，没有沿用旧表。积分采用开区间变量变换，
允许可积的终端对数奇异性，不把它误称为端点处有限。

Coupling 优化 `sum P C_up(Q/P)+sigma² KL_m(Q || a bᵀ/M_b)`，
其中 row(P)=a、col(Q)=b。接收端熵同时用于梯度、回溯目标和收敛过程。
固定端点常数不参与优化；去掉的负行 KL 使界变松。Matching 保留原兼容
控制与回归损失：这些控制仍满足条件路径方程，但不声称是新上界目标的内层最优控制。

连续模型上的上界推导不等于离散表或最终神经网络 rollout 的数值上界认证。
比较效果应使用同样定义的真实反馈 Action、W1、质量误差，并检查积分和插值精度。

新增核验 `tests/test_logmoment_bound.py` 包括独立高斯求积、接收 KL 梯度、
开区间积分收敛与质量比导数。新运行输出在独立的 `*_logmoment_bound_seed42` 目录。

测试涵盖端点与方差斜率、时间和参数导数、空间 MC 矩、二次最小控制 PDE、与均匀增长的能量比较、布朗桥调试参考、有限重整化表达式的截断极限、Coupling 双边缘与质量比/Fenchel 条件、非凸拒绝、Matching 标签与混合后验梯度、POT 概率/质量分离、连续带权生成及官方数据索引。

- [DeepRUOT 官方仓库](https://github.com/zhenyiizhang/DeepRUOT)
- [固定版本官方 CSV](https://raw.githubusercontent.com/zhenyiizhang/DeepRUOT/e9b06bca530b4f442c829692bf43b656d0a1af4e/data/simulation_gene_data.csv)
- [POT 官方 API](https://pythonot.github.io/all.html#ot.emd2)
- 本目录的实验交接说明与组会理论草稿。
