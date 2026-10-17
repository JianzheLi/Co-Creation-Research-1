# DeepRUOT：实验与代码实现交接说明

> 本文记录 2026-10-12 阶段的实验实现交接约定，保留作历史说明；后续实验设置可能已演进。
> 理论推导见[组会讨论草稿](DeepRUOT组会讨论草稿.tex)及其[PDF](DeepRUOT组会讨论草稿.pdf)，已完成实验的汇总见 [实验 1](Experiment1/README.md)、[实验 2](Experiment2/README.md) 和 [实验 3](Experiment3/README.md)。
> 复现实验时以当前 `Code/configs/` 配置和仓库根目录 README 为准；本文件中的早期计划不代表所有实验均尚未完成。

## 0. 本轮固定要求

1. 增长惩罚固定为 **growth_penalty = 10**，对应正文的 \(\alpha=10\)、\(\psi(g)=g^2/2\)。本轮不做该参数的比较。
2. 三个路径网络输入 **时间、位移向量、对数质量比**，输出均值、对角协方差、质量的修正。
3. 先预训练条件路径，再对相邻观测时刻分别求两张 Coupling，最后统一训练 Flow、Growth、Score。不要每次 Matching 更新都重新训练条件路径。
4. 路径预训练覆盖观测数据中相邻快照的**全部候选位移**，质量比覆盖 **0.1–10**，而非只用最近邻或最终 Coupling 上的位移。
5. Coupling **直接采用正文的行归一化＋逐列标量求根算法**。不自行替换成另一套优化问题。
6. 数据大时允许 minibatch-OT，仍然是先固定路径网络，再求批次 Coupling。
7. 生成时使用**固定数量的带权重粒子**，不实现分裂、复制、凋亡或删除粒子。
8. 第 \(k\) 个观测时刻有 \(N_k\) 个数据点，其总质量是 \(M_k=N_k/N_0\)。观测点在每个时刻内部等权，因而每个观测点的未归一化质量都是 \(1/N_0\)。
9. 生成后用 **POT** 计算双方归一化概率分布的 \(W_1,W_2\)，另报未归一化总质量误差。不能用归一化后的权重计算质量误差。
10. 第一轮在 DeepRUOT 官方 synthetic gene 数据上运行。条件训练和 Matching 不模拟 SDE；训练后的生成、评估允许模拟 SDE。

## 1. 实现顺序与文件组织

建议建立独立实验代码目录，至少包含：

~~~text
../Code/configs/gene.yaml
../Code/src/data.py                  # 官方数据下载、校验、时间分组与质量
../Code/src/pair_path.py             # 三个网络、时间导数、兼容控制
../Code/src/pair_cost.py             # 有限被积函数、MC 训练、固定节点查询
../Code/src/coupling.py              # 正文对偶算法、标量求根、残差
../Code/src/matching.py             # 多时间点 Flow/Growth/Score 回归
../Code/src/sample.py                # 带权粒子 SDE 生成
../Code/src/evaluate.py              # POT W1、W2、质量误差
../Code/scripts/train_pairs.py
../Code/scripts/solve_couplings.py
../Code/scripts/train_matching.py
../Code/scripts/generate_and_evaluate.py
tests/test_math.py
../Results/<run_id>/
~~~

执行顺序：

~~~text
准备全部观测快照及其质量
→ 构建相邻快照的位移索引
→ 对所有位移、质量比、时间预训练条件路径
→ 冻结路径网络，建立确定性的成对代价查询
→ 逐个相邻区间求 P、Q
→ 固定全部区间的 P、Q，统一训练 V、G、S
→ 从第一个时间点一次性向后生成
→ 在全部观测时刻计算 POT W1、W2 和质量误差
~~~

不要在生成到中间时刻后重新使用该时刻真实数据初始化。逐段重启可以作为调试，但不能代替完整 rollout。

推荐依赖：PyTorch、NumPy、SciPy、POT、pandas、PyYAML、matplotlib、tqdm。环境安装成功后保存锁定的版本清单和随机种子。算法无需引入 ODE/SDE 反向传播依赖。

## 2. 数据、质量及时间约定

### 2.1 主实验直接使用已核验的官方数据

DeepRUOT 原文是 Zhang, Li, Zhou，*Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport*，ICLR 2025。

- [论文，附录 B.1](https://arxiv.org/html/2410.00844v2#A2.SS1)
- [官方仓库](https://github.com/zhenyiizhang/DeepRUOT)
- [固定版本的 CSV](https://raw.githubusercontent.com/zhenyiizhang/DeepRUOT/e9b06bca530b4f442c829692bf43b656d0a1af4e/data/simulation_gene_data.csv)
- [固定版本的实验 Notebook](https://github.com/zhenyiizhang/DeepRUOT/blob/e9b06bca530b4f442c829692bf43b656d0a1af4e/notebook/simulation_gene.ipynb)

已核验 commit：
e9b06bca530b4f442c829692bf43b656d0a1af4e。

CSV 的 SHA256：
d41e3bcd94ff55383264461f2eed048780dbea8faffb77011cff9b8dce231d10。

CSV 列为 samples、x1、x2，共 3031 行：

| samples／模型时间 | 物理记录时间 | 点数 \(N_k\) | 总质量 \(N_k/400\) |
|---|---:|---:|---:|
| 0 | 0 | 400 | 1 |
| 1 | 8 | 442 | 1.105 |
| 2 | 16 | 530 | 1.325 |
| 3 | 24 | 690 | 1.725 |
| 4 | 32 | 969 | 2.4225 |

本轮在 CSV 的模型时间 \(0,1,2,3,4\) 上训练；保留原始二维坐标，不做各时刻单独标准化。每个观测点赋质量 \(1/400\)。如果以后抽样减少观测点，必须保留原始总质量，再把它均匀分给保留的点，不能让下采样改变质量。

不要把五个时刻分别归一化成总质量 1 后送入 Coupling；归一化仅在评估形状距离时使用。

三个网络只需覆盖相邻区间的位移集合：
\[
\mathcal D=\bigcup_{k=0}^{3}
\{x_j^{k+1}-x_i^k:1\le i\le N_k,\ 1\le j\le N_{k+1}\}.
\]
四个区间分别有 176800、234260、365700、668610 对，共 1445370 对。通过 \((k,i,j)\) 索引按需构造位移，不必常驻一个巨大的三维张量。

### 2.2 原文的数据生成动力学，以及本项目的带权实现

原文附录 B.1 给出三基因系统。写为 Itô 形式：
\[
dX_i=b_i(X)\,dt+\eta_i\,dW_i,
\]
\[
b_1=\frac{\alpha_1X_1^2+\beta}
{1+\alpha_1X_1^2+\gamma_2X_2^2+\gamma_3X_3^2+\beta}
-\delta_1X_1,
\]
\[
b_2=\frac{\alpha_2X_2^2+\beta}
{1+\gamma_1X_1^2+\alpha_2X_2^2+\gamma_3X_3^2+\beta}
-\delta_2X_2,
\qquad
b_3=\frac{\alpha_3X_3^2}{1+\alpha_3X_3^2}-\delta_3X_3.
\]

原文参数为：
\[
(\alpha_1,\alpha_2,\alpha_3)=(0.5,1,1),\quad
(\gamma_1,\gamma_2,\gamma_3)=(0.5,1,10),
\]
\[
\delta_1=\delta_2=\delta_3=0.4,\quad
(\eta_1,\eta_2,\eta_3)=(0.05,0.05,0.01),\quad \beta=1.
\]
初始来自均值 \((2,0.2,0)\)、\((0,0,2)\) 的两个高斯，协方差按原文 \(0.01I\) 解释。原文使用步长 1、记录时间 \([0,8,16,24,32]\)，每步把负表达量改为 0，最后研究前两维投影。

**原文通过随机分裂产生不同点数；本项目按用户要求不编写分裂／凋亡实现。** 若需要重新生成补充数据，则保留固定数量粒子，采用上述空间 SDE，并令
\[
d\log w_t=\Gamma_{\rm data}(X_t)\,dt,\qquad
\Gamma_{\rm data}(x)=\frac{A_{\rm data}}{100}\frac{x_2^2}{1+x_2^2}.
\]
这里 \(A_{\rm data}\) 是数据生成的增长幅度，**与控制惩罚 growth_penalty=10 完全不同**。原文把相应量写为分裂百分比，但参数表没有列出其幅度；本次检查的官方仓库只有成品数据与训练 Notebook，没有独立生成脚本。因此，不能把一个自行选择的 \(A_{\rm data}\) 宣称为原作者参数。

重新生成时的伪代码：

~~~python
# A_data 必须由补充实验配置明确给出；主实验加载官方 CSV，不需要它。
x = sample_initial_cells(n_particles)       # 三维；固定粒子数
logw = full(n_particles, -log(n_particles))
for step in range(n_steps):
    growth = (A_data / 100.0) * x[:, 1]**2 / (1.0 + x[:, 1]**2)
    logw = logw + growth * dt
    x = x + drift(x) * dt + eta * sqrt(dt) * normal_like(x)
    x = maximum(x, 0.0)                     # 沿用原文的非负修正
    save_if_needed(x, logw)
~~~

该带权连续增长模型是对原分裂模拟的改写，不是逐样本复现原 CSV；没有分裂就不使用原文的 daughter perturbation。若需从带权真值产生“各时刻内部等权”的观测快照，在记录时独立按 \(w/\sum w\) 采样观测位置，取 \(N_k\) 与总质量成比例。这只是快照观测抽样，不改变用于演化的固定粒子集合。主实验不需要重新生成：直接用官方 CSV 的实际点数确定观测质量。

原文图注、附录及 Notebook 对模型扩散参数有 0.05 和 0.25 的不同设定。本轮建议明确固定 **model_sigma=0.25**，沿用已核验 Notebook 的设定；它是本方法的扩散超参数，不把它说成三维真值扩散的精确时间换算。增长惩罚始终为 10。

## 3. 优化目标与约定

在一个长度为 1 的相邻区间内，单位初始质量的条件路径满足
\[
\partial_tf=-\nabla\cdot(fu)+\frac{\sigma^2}{2}\Delta f+gf,\quad
f_0=\delta_{x_0},\quad f_1=r\delta_{x_1}.
\]
运行能量为
\[
\int_0^1\int f\left(\frac12\|u\|^2+\frac{\alpha}{2}g^2\right)dx\,dt,
\qquad \alpha=10.
\]
\(u\) 是 SDE 漂移，不是后文的概率流速度。记 \(\Delta=x_1-x_0\)、\(\ell=\log r\)，成对代价采用正文的有限重整化表达式。

当前算法优化的是**删除群体修正后的近似目标**，再叠加对角高斯与网络限制。不把它写成已经精确求解原 RUOT，也不把删项目标自动称为原目标的上界。这里无需重新推导理论，但实验报告要保留这个范围。

## 4. 三个条件路径网络

### 4.1 参数化

三个平滑 MLP：
\[
h_\mu=B_\mu\tanh N_\mu(t,\Delta,\ell)\in\mathbb R^d,\quad
h_\Sigma=B_\Sigma\tanh N_\Sigma(t,\Delta,\ell)\in\mathbb R^d,\quad
h_m=B_m\tanh N_m(t,\Delta,\ell)\in\mathbb R.
\]
初版可用各 3 个隐藏层、宽 128、Tanh；\(B_\mu=B_\Sigma=B_m=4\)。这些是实现初值，不是论文报告的超参数。最后一层权重和偏置初始化为 0。不要使用 BatchNorm、Dropout 或 \(1/t\) 等奇异输入。

设 \(\tau=t(1-t)\)、\(Z_j=\tau h_{\Sigma,j}\)，定义
\[
\xi=x_0+t\Delta+\tau h_\mu,\qquad
s_j=\sigma^2\tau e^{Z_j},\qquad
m=e^{t\ell+\tau h_m},\qquad
f=m\,\mathcal N(\xi,\operatorname{diag}s).
\]
网络先在 \(x_0=0,m_0=1\) 下工作，查询时平移回起点。输入位移可用固定尺度标准化，但公式中的 \(\Delta\) 使用真实位移，不能把标准化尺度遗漏在输出上。

这保证
\[
\xi_0=x_0,\ \xi_1=x_1,\quad m_0=1,\ m_1=r,\quad
s_j(t)=\sigma^2t+O(t^2),\
s_j(1-\delta)=\sigma^2\delta+O(\delta^2).
\]
端点按测度极限解释。不能随意换成任意归零方差，否则原来扣除的对数发散不一定正确。

### 4.2 时间导数及自动微分

\[
\dot\xi=\Delta+(1-2t)h_\mu+\tau\dot h_\mu,\quad
\gamma=\partial_t\log m=\ell+(1-2t)h_m+\tau\dot h_m,
\]
\[
\dot Z=(1-2t)h_\Sigma+\tau\dot h_\Sigma,\quad
\dot s_j=\sigma^2e^{Z_j}[(1-2t)+\tau\dot Z_j].
\]

时间张量形状为 [B,1]，位移为 [B,d]，对数质量比为 [B,1]。逐输出分量计算时间导数，或使用 JVP；不能用一次 sum 把多个输出维的导数混在一起。

~~~python
def time_derivative(y, t, create_graph):
    # 样本间独立的 MLP；y: [B, D], t: [B, 1]
    out = []
    for j in range(y.shape[-1]):
        dy = torch.autograd.grad(
            y[:, j].sum(), t,
            create_graph=create_graph, retain_graph=True
        )[0]
        out.append(dy)
    return torch.cat(out, dim=-1)
~~~

路径训练使用 create_graph=True，因为还需对时间导数求参数梯度。Coupling 查询／Matching 标签生成时冻结路径参数，但计算时间导数的部分仍需 enable_grad，不能把整个查询包在 no_grad 中。

## 5. 兼容控制和有限的即时损失

主实验采用正文中**二次惩罚下固定高斯密度的最小兼容控制**。保留均匀增长版本用于公式调试，但不要在训练代价和 Matching 标签之间混用两个版本。

### 5.1 最小兼容控制

以下所有运算逐样本进行，求和只对空间维度。定义
\[
b_j=-\frac{\alpha\dot\xi_j}{\alpha+s_j},\qquad
H_j=\frac{\alpha(\sigma^2-\dot s_j)}{s_j(2\alpha+s_j)},\qquad
K_j=H_js_j.
\]
对于 \(y=x-\xi\)：
\[
u_j(x)=-b_j-H_jy_j,
\]
\[
g(x)=\gamma-\frac1\alpha
\left[\sum_jb_jy_j+\frac12\sum_jH_j(y_j^2-s_j)\right].
\]
这里 \(g(x)\) 一般不是常数，不能换成 \(\gamma\)。给定该高斯密度路径，这组控制最小化兼容控制能量。对应的原始瞬时能量
\[
\mathcal E_t=\frac m2\left[
\alpha\gamma^2+
\sum_j\frac{\dot\xi_j^2}{1+s_j/\alpha}+
\sum_j\frac{(\dot s_j-\sigma^2)^2}{4s_j+2s_j^2/\alpha}
\right]
\]
在终点发散，因此**不能直接用 \(\mathcal E_t\) 做整个区间的 MC 损失**。

### 5.2 无需空间采样的有限表达式

令 \(e=1-t\)，定义
\[
w_*=-\Delta+t h_\mu=\frac{\xi-x_1}{1-t},\qquad
F_*=\frac{e}{2}\|w_*\|^2+\frac{\sigma^2t}{2}\sum_je^{Z_j}.
\]
再定义
\[
D_j^{\rm unif}=\frac{1-e^{-Z_j}}{2\tau}+\frac12\dot Z_j,
\quad
A_j=-b_j+w_{*,j},
\]
\[
D_j=\frac{\sigma^2t e^{Z_j}+2\alpha D_j^{\rm unif}}{2\alpha+s_j}.
\]
则
\[
E_{\rm res}=\|A\|^2+\sum_js_jD_j^2
=\mathbb E\|u-v\|^2,\qquad v(x)=\frac{x_1-x}{1-t},
\]
\[
E_{g^2}=\gamma^2+
\frac{\sum_jb_j^2s_j+\frac12\sum_jK_j^2}{\alpha^2},
\]
\[
E_{gF}=\gamma F_*
-\frac1\alpha\sum_jb_js_jw_{*,j}
-\frac1{2\alpha}\sum_jK_j\,\sigma^2t e^{Z_j}.
\]
有限被积函数及完整查询代价为
\[
\boxed{\mathcal L_\theta
=m\left(\frac12E_{\rm res}+\frac{\alpha}{2}E_{g^2}+E_{gF}\right)
+\frac{d\sigma^2}{2}\frac{m-r}{1-t}},
\]
\[
\boxed{c_\theta(\Delta,r)=
\frac12\|\Delta\|^2-\frac{d\sigma^2r}{2}
+\int_0^1\mathcal L_\theta(t;\Delta,r)\,dt.}
\]
该式已经把高斯空间期望解析积分掉，训练只采样位移、质量比、时间。它是有限量，不需要模拟条件 SDE。

### 5.3 数值稳定写法

定义 \(\operatorname{exprel}(z)=(e^z-1)/z\)，在零附近使用
\(1+z/2+z^2/6+z^3/24\)。实现分支时也要让未选分支的分母安全，避免 autograd 中产生 NaN。

使用
\[
D_j^{\rm unif}
=\tfrac12 h_{\Sigma,j}\operatorname{exprel}(-Z_j)+\tfrac12\dot Z_j.
\]
令 \(q=-\ell+t h_m\)，则
\[
\frac{m-r}{1-t}=r q\,\operatorname{exprel}((1-t)q).
\]
计算 \(K_j\) 时不要先计算发散的 \(H_j\)：
\[
K_j=-\frac{\alpha(\dot s_j-\sigma^2)}{2\alpha+s_j},
\]
\[
\dot s_j-\sigma^2
=\sigma^2[(1-2t)\operatorname{expm1}(Z_j)-2t+\tau e^{Z_j}\dot Z_j].
\]
这些式子消除了主要的端点相消。代价计算、求积、Coupling 初版用 float64；Matching 网络可用 float32。不要在原始累计发散能量上减去一个大对数来实现训练目标。

### 5.4 均匀增长版本仅作调试参考

若配置 control_mode=uniform_growth，则
\[
g=\gamma,\quad u_j=\dot\xi_j+\frac{\dot s_j-\sigma^2}{2s_j}(x_j-\xi_j).
\]
定义 \(A^{\rm unif}=(1-t)h_\mu+\tau\dot h_\mu\)，对应有限损失为
\[
\mathcal L^{\rm unif}
=m\left[
\tfrac12(\|A^{\rm unif}\|^2+\sum_js_j(D_j^{\rm unif})^2)
+\tfrac\alpha2\gamma^2+\gamma F_*
\right]+\frac{d\sigma^2}{2}\frac{m-r}{1-t}.
\]
主实验始终使用 quadratic_minimal，不自动切换。

## 6. 通过 Monte Carlo 预训练成对路径

### 6.1 覆盖全部位移和质量比

对每个区间展平 pair_id，令
\(i=\lfloor{\rm pair\_id}/N_{k+1}\rfloor\)、\(j={\rm pair\_id}\bmod N_{k+1}\)。
每个 coverage epoch 随机打乱索引，遍历全部候选对；位移重复可保留，不需要浮点去重。

每个候选位移每次出现时，独立采样
\[
\ell\sim{\rm Uniform}(\log0.1,\log10),\qquad r=e^\ell.
\]
额外让每个位移在前几个 coverage epoch 分别见到 \(r=0.1,1,10\)，随后使用连续采样。不要只抽一组质量比并永久固定。连续区间无法有限遍历，要求的是持续覆盖和验证，不是声称见过每个实数质量比。

每个条件取 4 个独立时间 \(t\sim{\rm Uniform}(0,1)\)。可用开区间随机数或带随机抖动的分层采样；不要固定截掉端点附近一段而不记录目标变化。

本轮可从 10 个完整 coverage epoch 起步：前 3 轮覆盖上述特定质量比，其余轮连续采样。把完成了多少位移遍历写入日志。大数据下若只能随机访问，所有候选对仍须有非零采样概率，并报告实际覆盖率。

### 6.2 损失和更新

\[
\min_\theta\mathbb E_{\Delta,\ell,t}\mathcal L_\theta(t;\Delta,e^\ell).
\]
端点常数不依赖路径网络参数，训练梯度可省略；验证完整代价及 Coupling 查询必须恢复它。

~~~python
for delta, ell in condition_batches:
    # 每个条件复制多个时间；t 是独立的可求导叶子张量
    t = sample_open_uniform_times(...)
    t.requires_grad_(True)
    state = pair_path(t, delta, ell, create_graph=True)
    integrand = finite_pair_integrand(state, alpha=10, sigma=model_sigma)
    loss = integrand.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(pair_path.parameters(), max_norm=10.0)
    optimizer.step()
~~~

建议初始 Adam，学习率 \(10^{-3}\)，每批 512 个条件、每条件 4 个时间。不是已调好的最优设置。记录各项损失、梯度范数、质量范围、端点有限性、未出现 NaN，以及按位移长度／质量比分层的验证误差。

不要额外把每个样本损失除以当前质量 \(m\)，否则会改变所优化的成对代价。不要为了让数值好看把负的重整化代价截为 0。

### 6.3 冻结后的确定性查询

Coupling 不使用每次随机变化的 MC 代价。用固定 Gauss–Legendre 时间节点计算 \(c_\theta\)，先用 32 节点，并在固定验证条件上与 64 节点比较；若误差不满足查询容差，再统一增加节点。

查询接口：
~~~python
cost, dcost_dr, d2cost_dr2 = oracle(delta, r)
~~~

若先对 \(\ell=\log r\) 求导：
\[
c_r=\frac{c_\ell}{r},\qquad
c_{rr}=\frac{c_{\ell\ell}-c_\ell}{r^2}.
\]
不能把 \(c_\ell\) 直接送进正文的 \(c'(r)=\lambda\)。

大矩阵按端点块查询、缓存。缓存包含路径 checkpoint 哈希、sigma、alpha、control_mode、时间节点、位移和质量比范围，防止换模型后继续读取旧代价。可使用经过精度核验的一维表加速，但代价、导数和共轭必须来自同一个查询函数；不要分别拟合互不一致的三个表。

## 7. 直接实现正文的 Coupling 算法

### 7.1 离散问题

在区间 \(k\)：
\[
a_i=\frac1{N_0},\quad b_j=\frac1{N_0},\quad
\sum_i a_i=M_k,\quad\sum_jb_j=M_{k+1}.
\]
保留两张矩阵 \(P,Q\)，求
\[
\min_{P,Q}\sum_{ij}P_{ij}c_{ij}(Q_{ij}/P_{ij})
+\varepsilon\sum_{ij}P_{ij}\left(\log(P_{ij}/w_{ij})-1\right),
\]
\[
P\mathbf1=a,\qquad Q^\top\mathbf1=b,\qquad
\varepsilon=\sigma^2,\quad w_{ij}=1.
\]
\(c_{ij}(r)=c_\theta(x_j^{k+1}-x_i^k,r)\)。本轮不要额外调整熵强度，也不要把它写成 \(2\sigma^2\)：当前成本中已采用 \(1/2\) 的控制能量约定。

由于只训练 \(0.1\le r\le10\)，第一版明确求解这个**带质量比范围限制**的问题。不得把求得的范围外 \(r\) 截断后继续声称满足约束。先检查必要条件 \(0.1\le M_{k+1}/M_k\le10\)；在全连接支持、正质量下，常数质量比加普通耦合也证明该条件足以构造可行解。稀疏支持还需另外检查。

### 7.2 成对一维查询

定义
\[
R_{ij}(\lambda)\in\arg\max_{r\in[0.1,10]}
\{\lambda r-c_{ij}(r)\},\quad
c_{ij}^*(\lambda)=\lambda R_{ij}(\lambda)-c_{ij}(R_{ij}(\lambda)).
\]
在正文的光滑凸情形：
- 若 \(\lambda\le c'_{ij}(0.1)\)，取 \(R=0.1\)；
- 若 \(\lambda\ge c'_{ij}(10)\)，取 \(R=10\)；
- 否则在区间内解 \(c'_{ij}(R)=\lambda\)。

使用带括区间的二分，Newton 只能作为受保护的加速。不要用未经括区间的 Newton 跳出网络训练域。

记
\[
\log K_{ij}(\lambda_j)=\log w_{ij}+c_{ij}^*(\lambda_j)/\varepsilon.
\]
\(K\) 会随列变量改变，不是预先固定的普通 Sinkhorn 核。

### 7.3 交替更新

初始化 \(\lambda_j\)；可用该列 \(c'_{ij}(1)\) 的中位数。以下全部使用 log-sum-exp：

1. **行归一化**
\[
\log s_i=\log a_i-\operatorname{LSE}_j\log K_{ij}(\lambda_j).
\]

2. **逐列求根**，固定当前 \(s_i\)，对每个 \(j\) 解
\[
\operatorname{LSE}_i[
\log s_i+\log K_{ij}(\lambda)+\log R_{ij}(\lambda)]
=\log b_j.
\]
逐列扩张 \(\lambda\) 的左右括区间，直到函数值夹住目标。各列可并行。

3. **恢复矩阵**
\[
\log P_{ij}=\log s_i+\log K_{ij}(\lambda_j),\qquad
Q_{ij}=P_{ij}R_{ij}(\lambda_j).
\]

4. **同时检查两边**
\[
e=\max\left\{
\frac{\|P\mathbf1-a\|_1}{M_k},
\frac{\|Q^\top\mathbf1-b\|_1}{M_{k+1}}
\right\}.
\]
若 \(e<10^{-6}\) 则结束，否则继续。初始最大外层迭代 1000，标量二分最多 60 步，内部对数质量残差目标 \(10^{-8}\)。

列更新会再次改变行和，所以不能只做一轮，也不能只检查列约束。保存两组残差、目标值、边界质量比占比、迭代次数与耗时。达到迭代上限时返回未收敛状态，不把失败结果悄悄用于主实验。

### 7.4 凸性与数值范围的检查

正文的单调求根保证依赖 \(c(r)\) 的凸性。**高斯网络查询值不自动保留该性质。** 本轮仍直接采用正文算法，不增加替代 Coupling 方法，但必须检查查询曲线的割线斜率／\(c_{rr}\)、导数一致性和求根单调性。

少量负曲率若能由求积误差解释，先提高确定性查询精度；若存在明显非凸，不要偷偷排序导数或换成凸包。保存失败样例、返回 cost_oracle_nonconvex 状态，在实验报告中说明正文求解器的适用条件尚未满足。任何修正目标的方案需单独讨论。

Coupling 求解期间冻结路径网络；不得为了让标量求根成功而在线更新路径。

### 7.5 minibatch-OT

若全矩阵查询或存储过大，把每个时间区间的数据随机分成相同数量的源、目标块，在每对块内用上述相同算法求解，拼成块对角 \(P,Q\)。各真实数据点保留原质量 \(1/N_0\)，因此拼接后仍满足原来的行／列约束。

对于大小不同的两端，按同一块数分块，使每块的目标／源总质量比接近全局质量比；不要求每个块两端总质量相等。每块仍检查质量比范围可行性。

块内 OT 是受限支持的近似，不等于全局 OT。记录批大小和划分种子。若对多次随机划分求得的 Coupling 做平均，Matching 时保留“划分编号＋原端点对”的路径分量；不要平均 \(P,Q\) 后只用平均质量比替代所有路径，那会改变原来的路径混合。

第一版优先离线预计算并缓存块 Coupling。Matching 阶段只采样缓存，不重新训练路径，也无需每步重做 OT。

## 8. 多个时间点：统一 Flow、Growth、Score Matching

### 8.1 所有区间共用一组反馈网络

训练三个网络
\[
V_\vartheta(t_{\rm global},x),\quad
G_\vartheta(t_{\rm global},x),\quad
S_\vartheta(t_{\rm global},x).
\]
输入只有全局时间与当前位置；不输入起点、终点、质量比或区间 Coupling。可各用 3 层宽 128 的 MLP、SiLU。路径网络保持 Tanh。

这里的 Flow \(V\) 是概率流速度，Score \(S\) 是**原始 score**，不是 \(\sigma^2S/2\)。输出接口必须写清楚，防止重复乘扩散系数。

### 8.2 采样与监督标签

选择区间 \(k\)，再采样
\[
(i,j)\sim P^{(k)}/M_k,\quad
r=Q_{ij}^{(k)}/P_{ij}^{(k)},\quad
s\sim{\rm Uniform}(0,1),\quad \epsilon\sim\mathcal N(0,I).
\]
本轮区间长度均为 1，令 \(t_{\rm global}=k+s\)。查询冻结路径，平移均值后生成
\[
X=\xi_s+\sqrt{s_{\rm var}}\odot\epsilon.
\]
这里 \(s_{\rm var}\) 表示方差向量，代码中用 var，避免与局部时间 s 混淆。

标签为
\[
S^{ij}(X)=-\frac{X-\xi}{s_{\rm var}},\quad
G^{ij}(X)=g^{ij}(X),\quad
V^{ij}(X)=u^{ij}(X)-\frac{\sigma^2}{2}S^{ij}(X).
\]
必须使用第 5 节与代价优化一致的最小兼容控制。特别地，Growth 标签不能用 \(\gamma\) 替代空间相关的 \(g^{ij}(X)\)。

用 \(\epsilon\) 可直接计算
\[
g^{ij}(X)=\gamma-\frac1\alpha
\left[\sum_j b_j\sqrt{s_j}\epsilon_j+
\frac12\sum_jK_j(\epsilon_j^2-1)\right].
\]

### 8.3 质量加权损失

取仅依赖局部时间的权重
\[
w_F(s)=s(1-s),\qquad w_G(s)=1,\qquad
w_S(s)=\sigma^2s(1-s).
\]
若区间以概率 \(p_k\) 被抽到，一个对应全时间质量积分的估计为
\[
\mathcal L_{\rm match}=
\mathbb E\frac{M_k}{p_k}\,m_s^{ij}
\left[
w_F\|V_\vartheta-V^{ij}\|^2+
w_G|G_\vartheta-g^{ij}|^2+
w_S\|S_\vartheta-S^{ij}\|^2
\right].
\]
整体固定倍数可统一除掉。默认区间均匀采样；保留 \(M_k\)，避免各区间真实质量差异被无意消除。

**当前条件质量 \(m_s^{ij}\) 必须乘入损失。** 按 \(P/M_k\) 采样后不能再乘一次 \(P\)，也不能忘掉 \(m\)。不用显式计算任何混合后验。

不要把各条件路径自身的方差 \(s_j^{ij}\) 作为额外损失权重，因为它依赖端点对，会改变回归目标；这里使用的是统一时间权重。

### 8.4 稳定计算标签

设 \(\tau=s(1-s)\)。加权 score 标签为
\[
\sqrt{w_S}\,S_j^{ij}(X)=-e^{-Z_j/2}\epsilon_j.
\]
最小兼容控制对应的加权 flow 标签为
\[
\sqrt{w_F}\,V_j^{ij}(X)
=-\sqrt\tau\,b_j+
\frac{\sigma^2/2-K_j}{\sigma e^{Z_j/2}}\epsilon_j.
\]
直接计算“加权预测减加权标签”，避免先构造很大的原始标签再平方。模型最终输出仍是原始 \(V,G,S\)。

~~~python
# 路径标签需要时间自动微分，但不更新路径参数。
with torch.enable_grad():
    state = frozen_pair_path(local_t, delta, log_r, create_graph=False)
    x, target_v_scaled, target_g, target_s_scaled, mass = make_targets(state)

x = x.detach()
mass = mass.detach()
target_v_scaled = target_v_scaled.detach()
target_g = target_g.detach()
target_s_scaled = target_s_scaled.detach()

v = flow_net(global_t, x)
g = growth_net(global_t, x)
score = score_net(global_t, x)
lf = ((sqrt_wf * v - target_v_scaled)**2).sum(-1)
lg = (g.squeeze(-1) - target_g.squeeze(-1))**2
ls = ((sqrt_ws * score - target_s_scaled)**2).sum(-1)
loss = (interval_factor * mass.squeeze(-1) * (lf + lg + ls)).mean()
~~~

参考来源：
- [WFR-FM，第 4 节，质量加权的 Flow/Growth Matching 及多时间点流程](https://arxiv.org/html/2601.06810)
- [Tong et al., AISTATS 2024，Simulation-free Schrödinger bridges via score and flow matching](https://proceedings.mlr.press/v238/tong24a.html)
- [官方 TorchCFM 代码](https://github.com/atong01/conditional-flow-matching)

用户所说 S²FM 对应这里的 \([SF]^2M\)。沿用其条件 Flow/Score 回归思想，扩展到当前带权路径；不要把 balanced Brownian bridge 的固定标签直接替换进我们的已优化路径。

### 8.5 时间间隔不等时

本轮官方数据按等间隔模型时间实现即可。以后若相邻间隔为 \(h_k\)，不能只改全局时间输入却不改动力学。

局部时间 \(s=(t-t_k)/h_k\) 下：
\[
\sigma_{\rm local}=\sqrt{h_k}\sigma,\quad
u_{\rm physical}=u_{\rm local}/h_k,\quad
g_{\rm physical}=g_{\rm local}/h_k,\quad S_{\rm physical}=S_{\rm local}.
\]
二次惩罚下
\[
c_{\rm physical}(\Delta,r)
=\frac1{h_k}c_{\rm unit}(\Delta,r;\sqrt{h_k}\sigma,\alpha),
\]
这里采用局部相对终点截断的重整化约定；换成绝对时间截断还差一个对固定末端总质量无影响的常数。物理成本对应的 Coupling 熵系数仍为 \(\sigma^2\)。

不同 \(h_k\) 可分别预训练路径网络，每个仍只输入 \((s,\Delta,\log r)\)。若要共用网络，须显式增加区间长度／扩散条件，不能把只在固定 sigma 下训练的网络直接用于任意长度。Matching 使用物理 flow、growth，积分采样权重相应为 \(h_kM_k/p_k\)。

## 9. 训练后的带权 SDE 生成

从第一个快照按均匀概率采样 \(N_{\rm gen}\) 个初始位置，每个初始权重 \(1/N_{\rm gen}\)。使用
\[
U(t,x)=V_\vartheta(t,x)+\frac{\sigma^2}{2}S_\vartheta(t,x),
\]
\[
dX_t=U(t,X_t)\,dt+\sigma\,dW_t,\qquad
d\log w_t=G_\vartheta(t,X_t)\,dt.
\]
固定粒子数；权重可增可减，始终不需要分裂、死亡或重新采样。

~~~python
x = sample_uniform_from_first_snapshot(n_gen)
logw = torch.full((n_gen,), -math.log(n_gen), device=device)
for each small step t -> t + dt:
    u = flow(t, x) + 0.5 * sigma**2 * score(t, x)
    g = growth(t, x).squeeze(-1)
    x_next = x + u * dt + sigma * math.sqrt(dt) * torch.randn_like(x)
    logw_next = logw + g * dt
    x, logw = x_next, logw_next
    save_at_observation_times(x, logw)
~~~

初版每个单位时间区间 200 步，并用更细步长在少量样本上核验离散误差。最后一步准确落在观测时刻。生成不使用真值后验、不把粒子吸附到真实数据点，也不在每个时刻强制归一化总质量。

网络在开区间训练；实现应避免把条件公式直接代入 \(s=0,1\)。rollout 的时间输入可做明确记录的小量端点保护，保护大小与步长敏感性需要检查。不要把这种数值保护当成严格恢复硬 Dirac 端点的证明。

保存 positions、log_weights、模型时间、sigma、checkpoint、种子、步长。总质量通过 logsumexp 计算。不要对每一步 log_weights 做无补偿的减均值，这会丢失总质量。

若为稳定性存储归一化权重，必须同时保存独立的 log_total_mass，最后恢复真实权重。记录有效样本量 ESS，禁止静默丢掉大权重粒子。

## 10. POT 评估：形状与质量分开

观测快照 \(Y_k\) 有 \(N_k\) 点，目标总质量
\[
M_k=N_k/N_0,\qquad b_j^{\rm raw}=1/N_0,\qquad
b_j^{\rm prob}=1/N_k.
\]
生成粒子 \(X_k\) 带权 \(w_n=e^{\ell_n}\)，令
\[
\widehat M_k=\sum_nw_n,\qquad
a_n^{\rm prob}=w_n/\widehat M_k.
\]
用户要求的归一化 \(W_1,W_2\) 指**将两边质量归一化为概率分布**，不是额外除以坐标尺度：
\[
W_1=\min_{\pi\in\Pi(a^{\rm prob},b^{\rm prob})}
\sum_{nj}\pi_{nj}\|X_n-Y_j\|,
\]
\[
W_2=\left[
\min_{\pi\in\Pi(a^{\rm prob},b^{\rm prob})}
\sum_{nj}\pi_{nj}\|X_n-Y_j\|^2
\right]^{1/2}.
\]
分别用欧氏距离、平方欧氏距离做代价，分别求 OT。POT 的 emd2 返回传输目标值；它的名字不意味着输入什么成本都自动变成 \(W_2\)。

~~~python
import numpy as np
import ot
from scipy.special import logsumexp

def evaluate_snapshot(x_gen, logw_gen, y_obs, n_initial_observed):
    x = np.asarray(x_gen, dtype=np.float64)
    y = np.asarray(y_obs, dtype=np.float64)
    logw = np.asarray(logw_gen, dtype=np.float64).reshape(-1)
    assert len(x) == len(logw) and len(y) > 0
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.isfinite(logw).all()

    log_mass = logsumexp(logw)
    mass_pred = float(np.exp(log_mass))
    if not np.isfinite(mass_pred):
        raise FloatingPointError("predicted mass overflow")
    mass_true = len(y) / float(n_initial_observed)

    a = np.exp(logw - log_mass)
    a /= a.sum()
    b = np.full(len(y), 1.0 / len(y), dtype=np.float64)

    d2 = ot.dist(x, y, metric="sqeuclidean")
    d2 = np.maximum(d2, 0.0)       # 仅去掉距离的浮点负零
    d1 = np.sqrt(d2)
    w1, info1 = ot.emd2(a, b, d1, numItermax=1_000_000, log=True)
    w2_squared, info2 = ot.emd2(a, b, d2, numItermax=1_000_000, log=True)
    # 检查当前 POT 版本返回的 warning/result_code，失败时不报正式指标。
    if info1.get("warning") or info2.get("warning"):
        raise RuntimeError((info1, info2))

    return {
        "W1_normalized": float(w1),
        "W2_normalized": float(np.sqrt(max(w2_squared, 0.0))),
        "mass_pred": mass_pred,
        "mass_true": mass_true,
        "mass_signed_error": mass_pred - mass_true,
        "mass_abs_error": abs(mass_pred - mass_true),
        "mass_relative_error": abs(mass_pred - mass_true) / mass_true,
        "ESS": float(1.0 / np.square(a).sum()),
    }
~~~

来源：[POT 官方 API](https://pythonot.github.io/all.html)。保存 POT 版本与求解状态。

主实验点数不大，优先对全部观测点及固定的生成样本计算精确 EMD。若生成数过大，可以额外按归一化生成权重抽取评估样本，报告这是抽样近似及样本数；总质量仍使用全部带权粒子计算，不能从重采样后的等权点数估计。

逐时刻输出 metrics.csv，列至少为：
time、N_observed、N_generated、W1_normalized、W2_normalized、mass_true、mass_pred、mass_abs_error、mass_relative_error、ESS、seed。

第一个时刻若使用观测点本身、每点 \(1/N_0\)，三项误差应为 0（浮点容差内），作为评估器测试。

## 11. 初版配置和实验输出

~~~yaml
seed: 42
dataset: official_deepruot_simulation_gene
dimension: 2
model_times: [0, 1, 2, 3, 4]
model_sigma: 0.25
growth_penalty: 10.0
growth_penalty_type: quadratic_half
control_mode: quadratic_minimal

pair:
  ratio_min: 0.1
  ratio_max: 10.0
  coverage_epochs: 10
  conditions_per_batch: 512
  times_per_condition: 4
  hidden_width: 128
  hidden_layers: 3
  activation: tanh
  correction_bounds: [4.0, 4.0, 4.0]
  learning_rate: 0.001
  dtype: float64
  quadrature_nodes: 32

coupling:
  algorithm: row_normalization_column_root
  entropy_coefficient: sigma_squared
  reference_weights: ones
  marginal_tolerance: 0.000001
  root_logmass_tolerance: 0.00000001
  max_outer_iterations: 1000
  max_bisection_iterations: 60
  mode: full             # 显存／耗时不足时改为 minibatch，并记录
  minibatch_size_hint: 256

matching:
  hidden_width: 128
  hidden_layers: 3
  batch_size: 2048
  learning_rate: 0.001
  steps: 50000
  dtype: float32
  interval_sampling: uniform

generation:
  n_particles: 2000
  steps_per_unit_interval: 200
  particle_birth_death: false
  carry_log_weights: true

evaluation:
  library: POT
  transport_solver: emd2
  normalize_for_W1_W2: true
  report_mass_separately: true
~~~

先小批量跑通，再执行完整配置。配置是启动设置，不保证训练收敛；必须记录停止条件与实际完成训练量。growth_penalty 固定 10，不增加惩罚参数扫描。

分别报告路径预训练、代价查询／建表、Coupling、Matching、生成、POT 评估的时间。报告端到端时间时必须包括预训练和 Coupling，不能只给最后 Matching 的速度。缓存复用后的摊销时间可另列。

至少输出：
- 路径和反馈网络的 checkpoint、完整配置、环境版本；
- 每个区间的 P、Q、r、两组约束残差和收敛日志；
- 带权生成位置及 log_weights；
- 各时刻指标 CSV；
- 真值／生成散点图、质量随时间曲线、训练曲线；
- 一份结果说明，明确是否使用 minibatch-OT、是否出现非凸查询／边界比例集中。

## 12. 开始完整训练前必须通过的检查

1. 参数化在 \(t=0,1\) 给出正确的均值和质量；方差斜率符合 \(\pm\sigma^2\)。
2. 时间导数与中心差分在内部时刻一致；参数梯度通过小规模 gradcheck 或有限差分。
3. 在随机条件下，解析 \(E_{\rm res},E_{g^2},E_{gF}\) 与高斯空间 MC 均值一致。
4. 对同一高斯路径，最小兼容控制满足反应扩散 PDE；其瞬时能量不高于均匀增长实现。
5. **仅在 uniform_growth 调试模式**，令所有修正为 0、\(r=1\)，应得到布朗桥及
\[
c(\Delta,1)=\frac12\|\Delta\|^2-\frac{d\sigma^2}{2}.
\]
quadratic_minimal 允许额外的空间相关增长，即使 \(m_t=1\)，也不能误要求它与这个受限结果相等。
6. 比较有限表达式与多个末端截断下的“原始能量减对数”极限。该测试只用于核验，训练仍使用有限式。
7. Coupling 在小规模已知凸代价例子上满足两组边缘、质量比范围、Fenchel 取等；再接入网络查询。
8. 不把仍有边缘残差时的负数“原始－对偶差”当成最优性证书。
9. Matching 在少量高斯混合上与显式后验加权 \(V,G,S\) 一致；显式后验仅用于小规模测试。
10. POT 测试：相同点云、相同概率权重给出零 \(W_1,W_2\)；生成权重整体乘 2 后形状距离不变，但总质量翻倍。
11. 多时刻生成不断点重启，质量一直携带；评估归一化不能反向覆盖保存的真实权重。
12. 有限经验点端点在非零扩散下可能引起原始控制能量发散。因此不要把有限的成对重整化目标或有限步长 rollout 能量宣称为连续原 SOC 的精确最优值。主要报告本文件规定的形状、质量、耗时指标。

## 13. 交接给实验 Agent 的具体任务

按上面的接口实现完整流水线，先通过小规模公式测试，再在官方五快照二维数据上训练。不要修改增长惩罚 10，不增加分裂／凋亡，不引入新的 Coupling 求解问题。

交付代码、可复现配置、checkpoint、逐时刻 POT 指标和总质量误差，并明确记录未收敛或适用条件失败的环节。当前研究的目标是检验“成对预训练＋双 Coupling＋统一 Matching”的实用性；实验结果出来后再判断速度和精度，不提前写结论。

