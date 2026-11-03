# Young-Laplace 压差测量协议的复核与 p0 偏置反证（TensorSPH / CSF 表面张力）

**任务**: m2545 ｜ **范围**: `benchmarks/laplace_equilibrium/`、`tests/test_laplace_equilibrium.py`
（**未改 `src/`**：本单只做证据链与测量协议，物理修复留给下一轮） ｜ 上游: `csf_consistent.py`（m2447）

本文所有数字由 `benchmarks/laplace_equilibrium/run.py` 写出的 `equilibrium_report.json`
经 `make_doc.py` 程序化注入，无手抄。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

半径 `R = 0.02 m` 的二维单相静止液滴，六方密排晶格初始化，四周真空，无重力。
表面张力使液滴内部建立过压，Young-Laplace 定律给出

```
Delta p = p_in - p_out = sigma / R = 360 Pa      (p_out = 0, 真空)
```

| 参数 | 值 |
|---|---|
| 半径 R | 0.02 m |
| 表面张力 sigma | 7.2 N/m |
| 参考密度 rho0 | 1000 kg/m^3 |
| 声速 c0 | 20 m/s（Mach ~ 0.03） |
| Tait 指数 gamma | 7（B = rho0 c0^2/gamma = 57142.9 Pa） |
| 黏性 | 0.01（`Config.viscosity`） |
| 分辨率 | dx = 0.003 / 0.002 / 0.0015 m，h = 1.3 dx |
| 毛细时间 t_sigma = sqrt(rho0 R^3/sigma) | 3.333e-02 s |
| 积分窗 t_end | 0.0936 s = 3 t_sigma，dt = 3.900e-05 s |
| 判据 | 平台标准 15 %（<3 % 为优秀） |

## 2. 怎么做的（方法 / 流程 / 关键设置）

前两轮（`laplace_law` m2174：−118…−77 %；`csf_consistent` m2447：−75.6 %）把压差判为 FAIL，
但两者的**测量协议**本身有两个缺陷。本单逐条拆开做对照实验：

**(A) 初始条件对照。** 出厂六方晶格（`hexagonal_droplet`，质量 `rho0 dx dy`）的核和密度是
`rho/rho0 = 1.00588`，对应初始内压 **2395 Pa**，
是待测 Laplace 压差 360 Pa 的 **6.6 倍**；液滴因此先被一个巨大的膨胀波驱动，整个测量窗都被瞬态污染。
对照组把质量整体乘 `0.994151`（在同晶格上实测 `rho0/<rho>`），使芯部
`rho/rho0 = 1.000000000000`、初始内压 `0.00e+00` Pa，唯一驱动力就是表面张力。

**(B) 估计量对照。** 出厂估计量 `p_shell = <p>_(0.80R<r<0.95R)` 的取样带落在**核亏损层**：
该带内 `rho/rho0 = 0.914`（初始）/ 0.950（末态），Tait 压强被
`clamp_min(0)` 截零，实际不是压力。Young-Laplace 在真空外侧的正确提法是绝对内压：
`Delta p = <p>_(r<0.7R) - 0`。

**(C) p0 偏置反证（backlog P0 第 1 条）。** `backlog_update_m2174.md` 的 P0 断言：
"Tait 压强必须加 Adami(2012) 背景压强 `p0 = sigma/R`，否则无法建立过压"，并称这是 −77…−118 % 的主因。
本单把该修法**原样实现**（monkeypatch `droplet_impact.tait_pressure` 为
`p = p0 + B[(rho/rho0)^gamma - 1]`，`p0 = 360 Pa`），与 `p0 = 0` 做同一测量窗的对照。

**(D) 准静态控制。** 附加速度阻尼（`tau_damp = 5e-4 s`）版本，用来判断测到的压差是不是真平衡态。

诊断量：离散 delta 函数归一 `N_delta = sum_i delta_s,i A_i / (2 pi R)`（`A_i = m_i/rho0`，
Brackbill 1992 要求 `N_delta = 1`）、`max|v|/sqrt(sigma/(rho0 R))`、以及**被零截断的粒子比例**
（Tait 原始值为负的粒子占比）。

工具链：`benchmarks/laplace_equilibrium/run.py`（全部实验 + matplotlib 出图）、
`make_doc.py`（本文注入）、`tests/test_laplace_equilibrium.py`（8 项回归，35 s）。

## 3. 计算结果

### 3.1 协议分解（dx = 2 mm，362 粒子，同一积分窗）

| 协议 | Delta p [Pa] | 相对误差 |
|---|---|---|
| shipped init + shipped shell-band estimator | 37.4 | -89.6 % |
| shipped init + vacuum-referenced estimator | 160.9 | -55.3 % |
| density-consistent init + vacuum reference | 162.4 | -54.9 % |

**结论**：−89.6 % 的旧数字里，**约 34 个百分点来自估计量缺陷**（shell 带取在截零层），
初始条件缺陷对末态影响很小（−55.3 % vs −54.9 %，两初始条件在长时收敛到同一值），
**残余 −55 % 是模型的**，不是初始化的。

### 3.2 分辨率扫描（密度一致初始化 + 真空参考估计量）

| dx [m] | N | Delta p [Pa] | 相对误差 | N_delta 初 -> 末 | max\|v\|/v_ref | rho/rho0 芯部 |
|---|---|---|---|---|---|---|
| 0.0030 | 161 | 156.8 | -56.4 % | 0.665 -> 0.291 | 0.098 | 0.998943 |
| 0.0020 | 362 | 162.4 | -54.9 % | 0.691 -> 0.323 | 0.065 | 1.000118 |
| 0.0015 | 642 | 166.1 | -53.9 % | 0.697 -> 0.348 | 0.024 | 1.000314 |

误差 -56.4 % .. -53.9 %，**随分辨率不收敛**（2 倍加密只改变 2.5 个百分点），
且初态 `N_delta` 只有 0.665 .. 0.697（应为 1），运行中进一步衰减到 0.29–0.35。

### 3.3 p0 偏置反证（dx = 2 mm）

| p0 [Pa] | Delta p [Pa] | 相对误差 | p_shell 带 [Pa] | rho_shell/rho0 | 截零粒子占比（shell 带） |
|---|---|---|---|---|---|
| 0 | 162.4 | -54.9 % | 122.1 | 0.950 | 47 % |
| 360 | 162.5 | -54.9 % | 121.8 | 0.949 | 49 % |

**`p0 = 0` 与 `p0 = 360 Pa` 的测压差相差 0.15 Pa（162.4 vs 162.5，0.04 %）** ——
backlog P0 第 1 条对本判据是**空操作**，该假设被直接证伪。机制清楚：核亏损层里
`B[(rho/rho0)^gamma - 1]` 约 −3e4 Pa，加上 p0 = 360 仍为负，被 `clamp_min(0)` 吃掉；
而流体内部自己会松弛到同一个力平衡压力，所以总压不变。全粒子中
**32 % 的粒子（末态）处于零截断**，即根本不携带压强。

### 3.4 压差从哪来：N_delta 与 Delta p 的线性耦合

准静态段（t > 0.3 t_end，34 个采样）内 `Delta p/(sigma/R)` 与 `N_delta` 的
最小二乘斜率 **2.08**，相关系数 **r = 0.983**：压差不足与离散 delta 函数归一不足是
**同一个量**。同时 spurious current 末态仍为 `0.065 v_ref`（Adami 2012 的目标 < 0.1，
不算离谱），它持续把界面壳层搅散 —— 这就是 `N_delta` 在 0.0936 s 内从 0.691 衰减到
0.323 的驱动。

准静态窗口（末 25 %）均值：无阻尼 `183.0 Pa`（-49.2 %），
加阻尼 `211.3 Pa`（-41.3 %）。

![Young-Laplace 协议复核](../../benchmarks/laplace_equilibrium/laplace_equilibrium.png)

图：(a) 两初始条件下的内压历史（虚线 = 解析 360 Pa）；(b) `N_delta` 随时间衰减；
(c) `Delta p/(sigma/R)` 对 `N_delta`（斜率 2.08，r = 0.983）；(d) 径向压力剖面，
灰区 `r/R > 0.85` 即核亏损层（shell 带估计量的取样区）。

## 4. 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 实测 | 相对误差 | < 3 % ? | < 15 % ? |
|---|---|---|---|---|---|
| Delta p（协议修正后，dx=2 mm） | 360 Pa | 162.4 Pa | -54.9 % | 否 | 否 |
| Delta p（旧协议 shell 带） | 360 Pa | 37.4 Pa | -89.6 % | 否 | 否 |
| Delta p 分辨率收敛性 | 0 % 趋势 | -56.4 % .. -53.9 % | 不收敛 | 否 | 否 |
| 初始晶格内压 | 0 Pa | 2395 Pa | 6.6x | 否（协议缺陷，已修正） | 否 |
| p0 偏置是否改变判据 | 应显著改善 | 改变 0.15 Pa | 0.04 % | 是（空操作） | 是 |
| N_delta（初始，应为 1） | 1.0 | 0.691 | 30.9 % | 否 | 否 |

**判定**：Young-Laplace 压差仍 **FAIL（−55 %，非收敛）**，证据等级维持 **WITHHELD**：
实现存在、解析判据不通过。但本单把误差从「−75.6 %（协议混杂）」重定为「−55 %（模型结构性）」，
并证伪了 backlog 里被当作主因的 p0 偏置假设。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **不要把 `p_shell` 取样带放在 `r > 0.85 R`**：核和密度在自由面亏损 ~9 %，
   Tait 压强被零截断，那里的"压力"没有物理意义（旧两轮的 −75.6 %/−118 % 主要栽在这里）。
2. **初始晶格必须做密度一致性标定**：出厂 `rho/rho0 = 1.0059` 造成 2395 Pa 初始过压（6.6 倍判据），
   长时虽收敛到同一值，但会把任何短窗测量（含全部"准静态"细扫）污染。
3. **p0 偏置不是解**：Adami 的背景压强是为**多相**（两相都有粒子）设计的；单相真空自由面上
   零截断把它吃掉。要动就得动截断/自由面处理（例如真空侧补粒子、密度扩散 δ-SPH），
   而不是往 EOS 里加常数。
4. **本模型不存在静态平衡**：`max|v|/v_ref` 长时停在 0.02–0.1，界面持续被搅散，
   所以"某一个时刻的 Delta p"不是唯一值 —— 这是判据必须写明积分窗的原因。
5. 局限：2-D 单相、`c0 = 20 m/s`（压缩率大，所需压缩 9e-4）、`h = 1.3 dx`；
   三维/低 Mach 下的结论需重测。

## 6. 复现

```bash
PYTHONPATH=src python benchmarks/laplace_equilibrium/run.py       # ~7 min, 产出 JSON + PNG
PYTHONPATH=src python benchmarks/laplace_equilibrium/make_doc.py  # 重生成本文
PYTHONPATH=src python -m pytest tests/test_laplace_equilibrium.py -q   # 8 passed, ~35 s
```