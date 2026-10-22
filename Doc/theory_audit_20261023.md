# 理论问题记录：删项反例、端点极限与结论边界

本文对照 2026-10-23 的主分支文稿，记录此前讨论的理论问题，并给出可以独立核验的完整反例。这里需要区分已经成立的恒等式、主动引入的近似，以及超出证据范围的结论。

**主要结论：删除后验修正可能使代理目标下降、真实 RUOT Action 上升。** 下文在光滑高斯端点上证明这一点，逐项验证文稿假设，并给出原问题的精确最优值。反例使用一般理论允许的均匀增长兼容控制；它不是当前实验的 `quadratic_minimal` 控制律，不能直接据此断言该实验配置的最终解无效。

文稿已经明确承认删项与单高斯路径限制属于近似，因此不把这些声明本身认定为推导错误。真正需要避免的是把完整目标的等价性、给定密度下的控制最小性，继续用于证明删项算法的原问题最优性。另一个具体问题是摘要中“较精确近似最优控制律”的表述，尚缺相应的最优性或控制误差证据。

## 1. 哪些目标必须分开

[理论草稿](DeepRUOT组会讨论草稿.tex)的 `eq:finite-mixed-action`、`eq:source-energy-gap` 和 `eq:surrogate-decomposition` 给出

$$
\mathcal T(\Gamma)
=\mathcal S(\Gamma)+C_{\mathrm{fixed}}+\mathcal R(\Gamma),
\qquad
J[\rho,\bar U,\bar G]
=\mathcal T(\Gamma)-\mathcal D_{\mathrm{start}}(\Gamma),
$$

其中

$$
\mathcal R
=\sigma^2\int_0^1\widetilde{\mathcal K}_t\,dt
-\frac12\int_0^1\mathcal V_{\eta,t}\,dt
-\alpha\int_0^1\mathcal J_{\psi,t}\,dt,
\qquad \mathcal D_{\mathrm{start}}\geq0.
$$

- $\mathcal T$ 是保留全部修正、混合终点后的真实控制代价。
- $\mathcal S$ 是删除三个动态修正后的可分解目标。
- $J$ 是进一步消去起点标签后，无标签反馈的真实 RUOT Action。
- $C_{\mathrm{fixed}}$ 只依赖固定端点数据，不影响同一问题内的优化。

文稿关于 $\inf\mathcal T=\inf J$ 的证明要求一般条件路径与反馈恢复性质。该证明本身不提供 $\mathcal S$ 的最优性保证。下面的构造甚至令 $\mathcal D_{\mathrm{start}}=0$，从而把问题完全隔离在删除 $\mathcal R$ 这一步。

## 2. 反例的全部数据与控制

### 2.1 端点、扩散与两张 coupling

取 $d=2$、$\alpha=2$、$\psi(g)=g^2/2$，先允许任意 $\sigma>0$。主反例随后取 $\sigma=1$。记扩散热核

$$
k_s(y)=(2\pi\sigma^2s)^{-d/2}
\exp\!\left(-\frac{\|y\|^2}{2\sigma^2s}\right).
$$

取光滑、严格正的端点密度

$$
\mu_0=\mathcal N(0,I_2),
\qquad
\mu_1=4\mathcal N\bigl(0,(1+\sigma^2)I_2\bigr),
$$

以及固定的半耦合

$$
\pi^+(x_0,x_1)=\mu_0(x_0)k_1(x_1-x_0),
\qquad
\pi^-(x_0,x_1)=4\pi^+(x_0,x_1).
$$

显然 $\int\pi^+\,dx_1=\mu_0$；由高斯卷积，$\int\pi^-\,dx_0=\mu_1$。每对端点的质量比都为 $r_z=4$，发射条件密度为 $q^+(x_1\mid x_0)=k_1(x_1-x_0)$。这些 coupling 在后面的路径扰动中保持不变。

### 2.2 条件路径与兼容控制

任取光滑正质量曲线 $m(t)$，满足 $m(0)=1,m(1)=4$。定义

$$
\begin{aligned}
b_t^{x_0,x_1}(x)
&=\frac{k_t(x-x_0)k_{1-t}(x_1-x)}{k_1(x_1-x_0)}\\
&=\mathcal N\bigl(x;(1-t)x_0+tx_1,\sigma^2t(1-t)I_2\bigr),\\
f_t^{x_0,x_1}(x)&=m(t)b_t^{x_0,x_1}(x),\\
u_t^{x_0,x_1}(x)&=\frac{x_1-x}{1-t},
\qquad g_t^{x_0,x_1}(x)=\gamma(t):=\frac{m'(t)}{m(t)}.
\end{aligned}
$$

这是均匀增长控制，不是对给定高斯密度联合优化漂移与空间相关增长后的 `quadratic_minimal` 控制。

兼容性可以直接验证：设 $a=k_t(x-x_0)$、$h=k_{1-t}(x_1-x)$，则

$$
\partial_t a=\frac{\sigma^2}{2}\Delta a,
\qquad
\partial_t h=-\frac{\sigma^2}{2}\Delta h,
\qquad u=\sigma^2\nabla\log h.
$$

因此

$$
\partial_t(ah)
=\frac{\sigma^2}{2}(h\Delta a-a\Delta h)
=-\nabla\cdot(ah\,u)+\frac{\sigma^2}{2}\Delta(ah).
$$

除以不依赖 $t,x$ 的 $k_1(x_1-x_0)$，再乘以 $m(t)$，正好得到

$$
\partial_t f=-\nabla\cdot(fu)+\frac{\sigma^2}{2}\Delta f+\gamma f.
$$

两端按测度收敛为 $f_0=\delta_{x_0}$、$f_1=4\delta_{x_1}$。

### 2.3 混合后得到纯扩散与均匀增长

终点混合可以显式积分：

$$
\begin{aligned}
p_t(x\mid x_0)
&=\int q^+(x_1\mid x_0)f_t^{x_0,x_1}(x)\,dx_1
=m(t)k_t(x-x_0),\\
\omega_t(x_1\mid x,x_0)&=k_{1-t}(x_1-x).
\end{aligned}
$$

由该后验的均值为 $x$，有

$$
U_t^{x_0}(x)=\mathbb E_{\omega_t}\frac{x_1-x}{1-t}=0,
\qquad G_t^{x_0}(x)=\gamma(t).
$$

二者已经不依赖起点标签，所以 $\mathcal D_{\mathrm{start}}=0$。总密度与真实代价为

$$
\rho_t=m(t)\mathcal N\bigl(0,(1+\sigma^2t)I_2\bigr),
\qquad
\mathcal T[m]=J[m]=\int_0^1\frac{m'(t)^2}{m(t)}\,dt.
$$

### 2.4 原问题的全局最优值恰为 4

对任意满足质量守恒积分关系的可行反馈，记总质量 $M(t)=\int\rho_t$。积分密度方程并用 Cauchy–Schwarz，有

$$
M'(t)=\int\rho_tG_t,
\qquad
\int\rho_tG_t^2\geq\frac{M'(t)^2}{M(t)}.
$$

故在 $\alpha=2$ 时

$$
\begin{aligned}
J[\rho,U,G]
&\geq\int_0^1\frac{M'(t)^2}{M(t)}\,dt\\
&=4\int_0^1\left|\frac{d}{dt}\sqrt{M(t)}\right|^2dt
\geq4\bigl(\sqrt{M(1)}-\sqrt{M(0)}\bigr)^2=4.
\end{aligned}
$$

若质量可能触及零，可先用 $\sqrt{M+\eta}$ 建立同一下界再令 $\eta\downarrow0$；这不要求所有比较路径的质量都有统一正下界。

本构造取 $m_*(t)=(1+t)^2$，则 $U_*=0$、$G_*=2/(1+t)$，恰好达到下界。因此 $J_*=4$，且相应混合表示也达到 $\mathcal T_*=4$。这不是把某个启发式解当作最优参照。

## 3. 为什么它满足文稿的假设

下面针对 $m_*$ 及第 5 节的 $m_\varepsilon$ 核对。两者均光滑、严格正，$m,m',\gamma$ 在闭区间有界。文稿完整混合可行类明确允许任意兼容控制，不要求每条条件路径独立最优；参见“附录：混合目标的等价性与无标签最优反馈”的开头，以及“受限对照：固定为空间均匀增长”。

**端点二阶矩。** 设 $\Delta=x_1-x_0$，$F_t=\|x-x_1\|^2/[2(1-t)]$。直接计算

$$
\int f_tF_t\,dx
=\frac{m(t)}2\bigl((1-t)\|\Delta\|^2+d\sigma^2t\bigr)
\longrightarrow\frac12\|\Delta\|^2\quad(t\downarrow0),
$$

以及

$$
\frac1\delta\int f_{1-\delta}\|x-x_1\|^2dx
=m(1-\delta)\bigl(\delta\|\Delta\|^2+d\sigma^2(1-\delta)\bigr)
\longrightarrow4d\sigma^2.
$$

这正是 `eq:initial-boundary` 与 `eq:terminal-second-moment`，不只是弱端点收敛。

**成对积分条件。** 这里 $u=v$，所以残差动能为零；$m\psi(\gamma)$ 有界；$\int f|gF|=|\gamma|\int fF$ 对时间可积；由 $m(1)=4$ 与有界导数，$|m(t)-4|/(1-t)$ 有界。因此 `eq:integrability` 的四项均满足。高斯尾部使空间分部积分边界项消失。

**核与熵渐近。** 条件均值偏离端点为 $O(t)$ 或 $O(\delta)$，协方差为 $\sigma^2tI+O(t^2)$ 或 $\sigma^2\delta I+O(\delta^2)$。而且有精确熵公式

$$
\int f_t\log f_t\,dx
=m(t)\log m(t)-\frac{d\,m(t)}2
\log\bigl(2\pi e\sigma^2t(1-t)\bigr).
$$

初端据此得到共同热核熵，末端得到

$$
\int f_{1-\delta}\log f_{1-\delta}\,dx
=4\log4-\frac{4d}{2}\log(2\pi e\sigma^2\delta)
+O(\delta|\log\delta|).
$$

混合密度 $h_t^{x_0}=\mu_0(x_0)m(t)k_t(x-x_0)$ 在末端收敛到连续高斯密度 $\pi^-(x_0,x)$，对应熵也收敛。因此不涉及把有限个终点原子误当成连续密度。

**配对积分和修正可积性。** $\pi^+$ 是联合高斯密度，$\log\pi^+$ 是二次多项式加常数；后文显式的 $C_{\mathrm{ren}}^z$ 至多按 $\|\Delta\|^2$ 增长。因此 $\int\pi^+|C_{\mathrm{ren}}^z|$、所有相应带权对数积分均有限，端点余项可在高斯配对积分下控制。第 4 节还将给出 $\eta=0$、$\mathcal J_\psi=0$ 及 $|\widetilde{\mathcal K}_t|\leq C(1+|\log(1-t)|)$，满足 `eq:mixing-integrability`。

**反馈恢复性质。** 混合后的 $U=0$、$G=\gamma(t)$ 光滑有界；保守传播核是 $k$，带反应的源传播核是 $m(t)k_t$。代入文稿的 `eq:feedback-pair-lift`，恰好恢复上述 $\pi^+,\pi^-,f,u,g$。所以这个例子也符合完整目标等价性证明采用的恢复框架。

## 4. 被删除的修正与代理目标的精确值

### 4.1 三个修正中只有后验对数项非零

由条件高斯 score，直接得到

$$
\eta_t^z
=u_t^z-\frac{x-x_0}{t}-\sigma^2\nabla\log f_t^z=0.
$$

增长率也对所有终点标签相同，故 $\mathcal V_{\eta,t}=\mathcal J_{\psi,t}=0$。后验则是协方差 $\sigma^2(1-t)I_d$ 的高斯密度，所以

$$
\mathbb E_{\omega_t}[\log\omega_t]
=-\frac d2\log\bigl(2\pi e\sigma^2(1-t)\bigr),
$$

$$
\widetilde{\mathcal K}_t
=-\frac d2m'(t)\log\bigl(2\pi e\sigma^2(1-t)\bigr).
$$

注意这里 $\omega_t$ 是连续密度，不是有限标签概率；其对数期望没有固定符号。于是

$$
\boxed{
\mathcal R[m]
=-\frac{d\sigma^2}{2}\int_0^1m'(t)
\log\bigl(2\pi e\sigma^2(1-t)\bigr)\,dt
}
$$

且

$$
\boxed{
\mathcal S[m]+C_{\mathrm{fixed}}
=J[m]+\frac{d\sigma^2}{2}\int_0^1m'(t)
\log\bigl(2\pi e\sigma^2(1-t)\bigr)\,dt.
}
$$

### 4.2 直接从成对代价计算，得到同一结果

为避免仅依赖混合恒等式，单独计算截断能量：

$$
C_\delta^z
=\int_0^{1-\delta}\left[
\frac{m(t)}2\|\Delta\|^2
+\frac{d\sigma^2}{2}\frac{m(t)t}{1-t}
+\frac{m'(t)^2}{m(t)}\right]dt.
$$

减去 $4d\sigma^2\log(1/\delta)/2$ 后取极限，得到

$$
C_{\mathrm{ren}}^z
=\frac{\|\Delta\|^2-d\sigma^2}{2}\int_0^1m(t)\,dt
+\frac{d\sigma^2}{2}\int_0^1\frac{m(t)-4}{1-t}\,dt+J[m].
$$

因为在 $\pi^+$ 下 $\mathbb E\|\Delta\|^2=d\sigma^2$，第一项积分为零。再加 coupling 熵和 $C_{\mathrm{fixed}}$，初始密度熵抵消，留下

$$
\mathcal S[m]+C_{\mathrm{fixed}}
=J[m]+\frac{d\sigma^2}{2}
\left[\int_0^1\frac{m(t)-4}{1-t}\,dt
+3\log(2\pi e\sigma^2)\right].
$$

用分部积分

$$
\int_0^1m'(t)\log(1-t)\,dt
=\int_0^1\frac{m(t)-4}{1-t}\,dt
$$

可见两种计算完全一致。

## 5. 保持端点不变，两个目标的排序严格相反

现在固定 $\sigma=1$，令

$$
h(t)=t^2(1-t)^2,
\qquad m_\varepsilon(t)=(1+t)^2-\varepsilon h(t),
\qquad 0<\varepsilon\leq1.
$$

由于 $h(0)=h(1)=h'(0)=h'(1)=0$，端点质量及其一阶变化均保持不变。又有 $m_\varepsilon(t)\geq1$，所以第 3 节的正则性和积分条件仍全部成立。

记 $q_\varepsilon=\sqrt{m_\varepsilon}$。因为 $q_\varepsilon(0)=1,q_\varepsilon(1)=2$，

$$
\boxed{
J[m_\varepsilon]-4
=4\int_0^1(q_\varepsilon'(t)-1)^2\,dt>0.
}
$$

严格不等式来自 $\varepsilon>0$ 时 $q_\varepsilon\neq1+t$。因此扰动确实让真实 Action 变差。

另一方面，边界项为零，故

$$
\int_0^1h'(t)\log\bigl(2\pi e(1-t)\bigr)\,dt
=\int_0^1\frac{h(t)}{1-t}\,dt
=\int_0^1t^2(1-t)\,dt=\frac1{12}.
$$

由于 $d\sigma^2/2=1$，得到精确差值

$$
\boxed{
\mathcal S[m_\varepsilon]-\mathcal S[m_*]
=J[m_\varepsilon]-4-\frac{\varepsilon}{12}.
}
$$

这里真实代价只有二阶增量，而被删除的修正发生一阶变化。还可以给出不依赖数值求积的显式范围：利用 $q_\varepsilon\geq1$、$h\leq1/16$、$|h'|\leq1/2$，

$$
q_\varepsilon'-1
=\frac{\varepsilon}{q_\varepsilon}
\left(\frac{h}{1+t+q_\varepsilon}-\frac{h'}2\right),
\qquad
|q_\varepsilon'-1|\leq\frac{9\varepsilon}{32}.
$$

因此

$$
0<J[m_\varepsilon]-4\leq\frac{81}{256}\varepsilon^2,
\qquad
\mathcal S[m_\varepsilon]-\mathcal S[m_*]<0
\quad\text{当 }0<\varepsilon<\frac{64}{243}.
$$

特别地，$\varepsilon=0.1$ 严格落在这个区间内。数值仅用于展示量级：

| 质量路径 | 真实 $J=\mathcal T$ | $\mathcal S+C_{\mathrm{fixed}}$ | 被删除的 $\mathcal R$ |
|---|---:|---:|---:|
| $m_*(t)=(1+t)^2$ | 4.000000000000 | 9.013631199228 | −5.013631199228 |
| $m_{0.1}(t)=m_*(t)-0.1t^2(1-t)^2$ | 4.000088084812 | 9.005385950707 | −5.005297865895 |

两个路径具有完全相同的端点、coupling 和质量比。差异不是改变任务、改变端点熵常数、起点标签平均，或者数值训练未收敛造成的。

### 同一构造也排除未经修正的一般上界断言

这一段改变扩散系数并相应改变目标端点，属于另一个明确指定的 RUOT 问题。在第 2 节的一般 $\sigma$ 构造中，仍取 $m_*=(1+t)^2$，则 $J_*=4$，而

$$
\mathcal S[m_*]+C_{\mathrm{fixed}}
=4+\sigma^2\left[3\log(2\pi e\sigma^2)-\frac72\right].
$$

取 $\sigma=1/4$，端点为 $\mu_1=4\mathcal N(0,17I_2/16)$，上述值约为 $3.793491564532<4$。所以原样删除修正后的 $\mathcal S+C_{\mathrm{fixed}}$ 也不是一般性的真实 Action 上界。这个结论不针对另外添加 Young/Jensen 修正的上界候选目标。

## 6. 反例到底否定了什么

**已经否定的结论：** 对文稿的一般混合可行类，降低 $\mathcal S$ 不保证降低真实 Action；完整目标与删项目标在混合表示 $\Gamma$ 上也不具有相同的全局最优解集合，因为上面给出的一个完整目标全局最优表示，甚至不是 $\mathcal S$ 的局部极小点。修正项也不必在原问题最优处消失：主反例中 $\mathcal R[m_*]\approx-5.0136$。

**没有被否定的结论：** 完整混合恒等式、其正则条件下的反馈恢复论证、给定高斯密度路径的最小兼容控制公式，以及质量加权 Matching 的理想条件均值关系。上述构造与这些结论相容。

**尚未证明的事情：** 这不是当前有限网络、`quadratic_minimal` 控制和具体 coupling 求解器的反例，也没有证明所有 $\mathcal S$ 的全局最小解经投影后都不可能达到原问题最优值。一个原问题最优反馈可能有多个混合表示；本反例只需两个明确的可行表示，就足以推翻目标排序一致与最优解集合相同的普遍断言。它也不否定已有 EB 等数据上的数值效果。

文稿 `eq:surrogate-objective` 后已经指出这些结论不能直接继承。建议保留这一限定，并用本反例把“缺少证明”具体化为“无附加条件时确实可能失败”。

## 7. 其他理论问题及其影响范围

### 7.1 连续终点的发散抵消不能直接用于固定有限样本

文稿“附录：终点混合与对数发散的抵消”要求混合末端具有连续密度和熵收敛性，并在“连续终点与有限样本”一段明确指出有限 Dirac 终点的区别。这一限定是正确的。

一个直接的计算说明区别：固定起点，取 $N$ 个终点、权重 $q_j>0$ 且 $\sum_jq_j=1$，每个分量为单位质量布朗桥，增长率为零。离散终点后验 $\omega_j$ 的质量加权对数量满足

$$
-\log N\leq\mathcal H_t
=\int p_t(x)\sum_{j=1}^N\omega_j(t,x)\log\omega_j(t,x)\,dx\leq0.
$$

同样的截断混合计算给出

$$
\mathcal T_\delta
=\mathcal A_\delta-\sigma^2(\mathcal H_{1-\delta}-\mathcal H_0).
$$

后一个差在固定 $N$ 下有界，而

$$
\mathcal A_\delta=\frac{d\sigma^2}{2}\log\frac1\delta+O(1).
$$

所以这个固定有限混合的 $\mathcal T_\delta$ 仍对数发散。连续终点后验的微分熵渐近与有限标签熵不是同一个极限，不能互换。

实际 [条件路径参数化](../Code/src/pair_path.py)在末端令方差收缩；[采样器](../Code/src/sample.py)和 [Action 计算](../Code/src/action.py)则使用有限步长、端点保护与学习到的反馈。因此已报告的有限数值 Action 可以作为该数值过程的指标，但不能仅据此证明精确匹配固定 Dirac 终点的理想混合具有同一个有限 Action。需要明确解释为平滑端点问题，或给出样本数、平滑尺度与时间截断的连续极限关系。这里不把所有有限反馈数值结果本身判为错误。

### 7.2 给定高斯密度的最小控制，不等于一般路径最优

文稿的 `eq:quadratic-minimal-gaussian-controls` 消去的是固定 $f_t$ 后的兼容控制。令 $\mathcal G$ 表示允许的单高斯路径类，$\mathcal A$ 表示一般允许路径类，则仅有

$$
\inf_{(f,u,g)\in\mathcal A}C_{\mathrm{ren}}[f,u,g]
\leq
\inf_{f\in\mathcal G}C_{\mathrm{ren}}[f,u_f^{\min},g_f^{\min}].
$$

控制的解析消元不提供反向不等式；旋转和平移对称性也不意味着最优条件密度必须为单高斯。文稿已经承认这一点，属于建模限制，不是该控制公式的代数错误。仓库有 [非高斯可行下降方向诊断](../Code/scripts/check_nongaussian_direction.py)，但本记录没有重跑它，也不把脚本存在本身当作所有配置的实验结论。

### 7.3 摘要中的“较精确近似最优控制律”超出了现有论证

草稿 TL;DR 写道：“实验结果表明，使用 WFR Coupling + WFR 均值 + WFR 质量 + 布朗桥方差即可实现对 DeepRUOT 问题最优控制律的较精确近似。”

比较几个近似方法的 Action 或边缘分布误差，不能单独推出控制律接近原问题最优控制；即使得到最优值差的估计，也还需要稳定性或可识别性等条件才能进一步控制反馈距离。现有完整目标定理没有提供删项算法、单高斯限制和神经网络回归后的这一整套误差保证。

建议将该句改为：“实验评估了 WFR Coupling、WFR 均值与质量及布朗桥方差构成的 RUOT 启发式近似；其控制律与原问题最优控制的距离仍待验证。”这属于收紧结论，不需要撤回已有数值结果。

## 8. 数值复核方法

以下标准库代码仅复核第 5 节的数值和两种代理目标计算的一致性；严格排序由上面的解析不等式保证。可保存到任意临时 Python 文件运行，不需要训练模型或安装科学计算依赖。

```python
from math import e, log, pi


def simpson(f, n=20000):
    return (
        f(0.0) + f(1.0)
        + 4 * sum(f(i / n) for i in range(1, n, 2))
        + 2 * sum(f(i / n) for i in range(2, n, 2))
    ) / (3 * n)


for eps in (0.0, 0.1):
    def m(t):
        return (1 + t)**2 - eps * t**2 * (1 - t)**2

    def dm(t):
        return 2 * (1 + t) - 2 * eps * t * (1 - t) * (1 - 2 * t)

    J = simpson(lambda t: dm(t)**2 / m(t))
    minus_R = 3 * log(2 * pi * e) - 3.5 - eps / 12
    # (m(t) - 4) / (1 - t) 的连续延拓，避免在 t=1 作 0/0。
    pair_term = simpson(lambda t: -(3 + t) - eps * t**2 * (1 - t))
    direct = J + pair_term + 3 * log(2 * pi * e)
    print(f"eps={eps:.1f}: J={J:.12f}, S+C={J + minus_R:.12f}, "
          f"R={-minus_R:.12f}, direct={direct:.12f}")

sigma = 0.25
value = 4 + sigma**2 * (3 * log(2 * pi * e * sigma**2) - 3.5)
print(f"sigma={sigma}: J=4.000000000000, S+C={value:.12f}")
```

该代码已运行；输出与表中数字一致，两条独立公式计算的代理目标在浮点精度内一致。本记录没有新增或重跑任何学习实验。
