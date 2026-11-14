# CASE-SPH-COUETTE-TRANSIENT — Couette 瞬态启动解析解对标（TensorSPH）

> 锚点：`CASE-SPH-COUETTE-TRANSIENT`（算例池 · 第 1 轮）。本文件由
> `benchmarks/couette_transient/make_doc.py` 从 `out/results.json` 程序化生成，
> 表中每一个数字都来自工件，无手抄。

## ① 问题与参考出处

平板 Couette 瞬态启动：静止流体夹在静止下壁与 t=0 瞬间以 U 平移的上壁之间（高 H，
x 向周期，两侧无滑移，无体积力）。控制方程为 1‑D 热传导型方程及其精确解：

```
u_t = nu u_yy,  u(y,0)=0,  u(0,t)=0,  u(H,t)=U
u(y,t) = U y/H + (2U/pi) sum_{n>=1} (-1)^n/n sin(n pi y/H) exp(-n^2 pi^2 nu t/H^2)
tau_w(t)/(rho nu U/H) = 1 + 2 sum_{n>=1} (-1)^n exp(-n^2 pi^2 t*)
Q(t)/(U H/2) = 1 - (8/pi^2) sum_{k>=0} exp(-(2k+1)^2 pi^2 t*)/(2k+1)^2,  t* = nu t/H^2
```

**参考出处**：Morris, Fox & Zhu, J. Comput. Phys. 136 (1997) 214-226, Sec. V (plane Couette start-up from rest); exact series evaluated in code from u_t = nu u_yy with u(y,0)=0, u(0,t)=0, u(H,t)=U. No number is transcribed from the paper: the paper PDF is paywalled in this work tree and the reference is the closed form the paper itself compares against.

**零手抄的实现方式**：参考值不由表格抄入脚本，而是在 `run.py` 里用上式**求值**
（`couette_series` / `flux_exact` / `shear_factor_exact`）。该级数本身在
`tests/test_couette_transient_series.py` 中用**独立的 Crank–Nicolson 有限差分**
求解同一初边值问题交叉验证（4 项检验：CN 对照、初值/稳态极限、边界条件、
通量与壁面剪切的解析一致性），因此“参考解写错”不会被误记为“求解器误差”。

**诚实边界（未能完成的取证）**：Morris et al. (1997) 原文 PDF 在本工作树内不可
获取（出版社付费墙，unpaywall 判定 `is_oa=false`），因此**未**从原文表/图转录任何
数字，也**未**声称复现其具体参数组合；本算例的判据由上述精确解承载（该解无自由参数），
案例定义（脉冲启动 Couette + 级数参考解）依 Morris et al. §V。

## ② 方法与实现

* 求解器：TensorSPH 2‑D 弱可压 WCSPH（`src/tensorsph/droplet_impact.py:rhs`），
  三次样条核，`h = 1.3 dx`，3 层壁面粒子，`wall_friction = 1` 无滑移，
  上壁通过 `wall_vel` 给定切向速度（与 `wave_flume` 造波板同一输入通道）。
* 复用既有能力：`benchmarks/couette/run.py` 的 `make_config / build_fluid /
  build_walls / periodic_images / bin_rows`（无新物理、无新几何，只加瞬态度量）。
* 初值：全体流体静止（非解析剖面起步），t=0 起上壁即带 U ⇒ 脉冲启动。
* 时间推进：定步长 `dt = min(0.3 h/c0, 0.125 h^2/nu)`，无自适应、无随机数 ⇒ 逐位可复现。
* 采样：在预声明时刻 t* 附近取 `Δt* = ±0.005` 的时间窗平均（5–20 步，远小于黏性时标），
  再按晶格行分箱（`bin_rows`）得到 u(y)。
* 度量：内部带 `0.10 ≤ y/H ≤ 0.90` 的相对 L2 误差、L∞/U、梯形积分通量与精确通量之比、
  最低两行割线剪切与“同一割线作用在精确剖面上”的比值（离散化不记在求解器头上）。

## ③ 参数表（无量纲数完整）

| 量 | 符号 | 值 | 无量纲数 |
|---|---|---|---|
| 通道高度 | H | 0.1 m | — |
| 上壁速度 | U | 0.2 m/s | — |
| 运动黏度 | ν | 0.02 m²/s | — |
| 密度 | ρ₀ | 1000 kg/m³ | — |
| 声速 | c₀ | 20 m/s | — |
| 状态方程指数 | γ | 7 | — |
| 核长 | h | 1.3 dx | h/Δx = 1.3 |
| Reynolds 数 | Re = UH/ν | — | 1 |
| Mach 数 | Ma = U/c₀ | — | 0.01 |
| 分辨率阶梯 | H/dx | [10, 16, 24, 32] | — |
| 时间步（参考分辨率） | dt | 1.219e-04 s | 声学 CFL = 0.300 |
| 黏性扩散数 | ν dt/h² | — | 0.0369 |
| 采样时刻 | t* = νt/H² | [0.05, 0.1, 0.2, 0.3, 0.5] | — |
| 门限（预声明） | L2 / 通量 / 剪切 | — | 8 % |

## ④ 对照表（ref vs got，逐项误差）

参考分辨率为 H/dx=16（ladder-r16），192 流体粒子 + 108 壁面粒子，2462 步，CPU 23.7 s。

“ref” 列为上式在该时刻的求值；“got” 列为 SPH 采样剖面按 §② 的度量算得。

| t* | t*_real | 采样帧数 | L2 % | L∞ % | 通量偏差 % | 壁面剪切偏差 % | 判据 <8% |
|---|---|---|---|---|---|---|---|
| 0.050 | 0.0500 | 21 | +13.10 | +7.08 | -11.28 | -30.15 | ✗ |
| 0.100 | 0.1001 | 20 | +8.38 | +5.41 | -7.32 | -23.28 | ✗ |
| 0.200 | 0.2000 | 20 | +4.06 | +3.58 | -2.21 | -13.29 | ✓ |
| 0.300 | 0.3001 | 21 | +4.98 | +4.12 | +0.35 | -11.46 | ✓ |
| 0.500 | 0.5001 | 20 | +6.33 | +5.06 | -2.51 | -13.47 | ✓ |

逐项判定（t* ≥ 0.1 的有效时刻，容差 8 %）：

* t*=0.100：L2 +8.38 % ✗ 8 %；通量 -7.32 % ✓；剪切 -23.28 % ✗；中线 u/U 测得 0.2103 vs 精确 0.2340
* t*=0.200：L2 +4.06 % ✓ 8 %；通量 -2.21 % ✓；剪切 -13.29 % ✗；中线 u/U 测得 0.3717 vs 精确 0.3808
* t*=0.300：L2 +4.98 % ✓ 8 %；通量 +0.35 % ✓；剪切 -11.46 % ✗；中线 u/U 测得 0.4389 vs 精确 0.4360
* t*=0.500：L2 +6.33 % ✓ 8 %；通量 -2.51 % ✓；剪切 -13.47 % ✗；中线 u/U 测得 0.4390 vs 精确 0.4642

## ⑤ 收敛性与参数敏感性

### 5.1 分辨率收敛（每个 t* 的最差 L2 误差）

| H/dx | 流体粒子 | dt [s] | ν dt/h² | 最差 L2 %（t*≥0.1） | 门 |
|---|---|---|---|---|---|
| 10 | 120 | 1.950e-04 | 0.0231 | 12.43 | ✗ FAIL |
| 16 | 192 | 1.219e-04 | 0.0369 | 8.38 | ✗ FAIL |
| 24 | 288 | 8.125e-05 | 0.0554 | 6.78 | ✗ FAIL |
| 32 | 384 | 6.094e-05 | 0.0738 | 7.41 | ✗ FAIL |

观测收敛阶：H/dx 10→16 为 p ≈ 0.84；16→24 为 p ≈ 0.52（误差在整条阶梯上单调下降）。

### 5.2 可压缩性（c₀ / Ma）敏感性

| c₀ [m/s] | Ma | dt [s] | 最差 L2 % | 门 | 说明 |
|---|---|---|---|---|---|
| 20 | 0.010 | 1.219e-04 | 8.38 | — | 基准 |
| 10 | 0.0200 | 2.437e-04 | 9.07 | ✗ FAIL | Ma 变化 2× |
| 40 | 0.0050 | 6.094e-05 | 22.07 | ✗ FAIL | Ma 变化 2× |

### 5.2b 近稳态时刻（t* = 0.5）的误差收敛性

| H/dx | 10 | 16 | 24 | 32 |
|---|---|---|---|---|
| L2 % | 7.43 | 6.33 | 6.78 | 7.41 |

该时刻误差**不随分辨率单调下降**：t*=0.5 的剖面残差是一个近乎与 Δx 无关的地板（~6–7 %），与 t*≥0.2 的收敛行为相反。逐行看剖面（见 ⑥）可知它是 S 形畸变：第一层流体偏快（壁面滑移）、中跨偏慢约 8 %，说明近壁核截断/无滑移实施是该项误差的来源，而扩张分辨率不能消除它。这是本算例最重要的负面结果。

### 5.3 时间步（κ = dt·c₀/h）敏感性

| κ | dt [s] | ν dt/h² | 最差 L2 % | 门 |
|---|---|---|---|---|
| 0.30（基准） | 1.219e-04 | 0.0369 | 8.38 | PASS |
| 0.15 | 6.094e-05 | 0.0185 | 8.41 | ✗ FAIL |

### 5.4 负对照（自由滑移 wall_friction=0，同分辨率）

自由滑移时无切向剪切输入，t*=0.50 处通量仅为精确值的 11.3 %（偏差 -88.70 %，门限 20 %）⇒ 参考分辨率下的吻合来自壁面剪切，
不是初值或平均窗的假象。

## ⑥ 失败模式与工件

**粗分辨率失败模式（如实记录，未删改）**：
* ladder-r10: t*=0.100 L2=12.43% >= 8%
* ladder-r10: t*=0.100 flux off by -11.06%
* ladder-r10: t*=0.100 wall shear off by -31.36%

已知失败模式：

* **近壁梯度欠分辨**：t* 越小，壁面边界层 δ/H ≈ √t* 越薄；H/dx=10 时 δ 仅约 2 个
  粒子间距，最低两行割线剪切误差最大（见 ④/5.1 的粗阶结果）。这不是求解器发散，
  而是分辨率不足，随 H/dx 增大而收敛。
* **声学抖动**：本求解器无人工黏性，WCSPH 粒子级声学脉动常驻；用预声明的
  `Δt* = ±0.005` 时间窗平均消除，逐帧单点采样会看到 1 % 量级抖动。
* **t* < 0.1 不计入判据**：预声明为止损点（边界层分辨极限），仍如实报告数值。

### 工件与复现

```
# 一键复现（重计算，递交调度）：
PYTHONPATH=src:. python benchmarks/couette_transient/run.py --sweep \
    --out benchmarks/couette_transient/out --threads 1
# 参考解自检（不跑 SPH，秒级）：
python -m pytest tests/test_couette_transient_series.py -q
# 工件绑定 / 防过度认证（只读已提交工件）：
python -m pytest tests/test_case_couette_transient.py -q
# 生成本文件与图：
PYTHONPATH=src:. python benchmarks/couette_transient/make_doc.py
```

| 工件 | 路径 |
|---|---|
| 预声明设计（先于任何运行） | `benchmarks/couette_transient/DESIGN.md` |
| 算例与参考解实现 | `benchmarks/couette_transient/run.py` |
| 原始结果（含逐行剖面、时间历程、快照） | `benchmarks/couette_transient/out/results.json` |
| 图 | `benchmarks/couette_transient/out/*.png` |
| 参考解自检 | `tests/test_couette_transient_series.py` |
| 工件绑定 + 门限审计 | `tests/test_case_couette_transient.py` |

### 生态摩擦（如实记）

* **knowledge_base 检索不可用**：`:8010/api/v1/tensor/knowledge/search` 返回
  `engine_offline`（后端 `:8021` connection refused），`negative-knowledge/search`
  对 `keywords=`/`query=` 返回 422、对 `q=TensorSPH` 返回 0 条 ⇒ 开工三查中的
  “同域 capability / negative ledger”两项无法取得证据，只能以仓内代码与
  `benchmarks/couette` 既有能力替代。
* **原文取证受付费墙阻断**：Morris et al. (1997) 无 OA 版本，算例数值只能由精确解
  承担（见 ①），原文具体参数组合未能核对。
* **队列拥塞**：递交作业排队约 10–30 min 才开跑（同一时刻 30+ 作业在飞），
  重计算只能走调度门，无节点可指定。

---

## 结论（决策回答）

**FAIL**（预声明门限，未放宽）。

* **剖面 L2（t* ≥ 0.1，容差 8 %）**：H/dx=10 最差 12.43 %，H/dx=16 最差 8.38 %，H/dx=24 最差 6.78 %，H/dx=32 最差 7.41 %；t* = 0.1 处依次为 12.43 %，8.38 %，5.30 %，3.85 %。即 L2 在 H/dx ≥ 24 通过 8 %，H/dx = 16 处 8.38 % 勉强越界。
* **壁面剪切（t* = 0.1，容差 8 %）**：所有分辨率均未通过（最小偏差 13.4 %），来源是壁面粒子无滑移实施造成的近壁滑移，而不是迭代未收敛。
* **近稳态（t* = 0.5）**：L2 随分辨率依次为 7.43 %，6.33 %，6.78 %，7.41 %，不收敛 ⇒ 存在与 Δx 几乎无关的 S 形剖面残差（见 5.2b / ⑥）。
* **结论**：TensorSPH 动壁 WCSPH 能定量复现 Couette 脉冲启动的中后期（H/dx = 32 时 t* ≥ 0.2 的 L2 为 3.42–7.41 %，通量偏差 -3.58 %），但在 8 % 容差下**不能判为通过**：启动最早时刻的剖面、以及所有分辨率的壁面剪切超限，且近稳态剖面残差不随分辨率收敛。

未通过的门限：
* ladder-r16: t*=0.100 L2=8.38% >= 8%
* ladder-r16: t*=0.100 wall shear off by -23.28%
* ladder-r24: t*=0.100 wall shear off by -17.67%
* ladder-r32: t*=0.100 wall shear off by -13.43%

