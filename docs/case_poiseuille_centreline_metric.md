# 泊肃叶流镜像壁算例：Gate A 中心线度量的采样偏置与其修正

**锚点** CASE-POISEUILLE-SPH（算例池·泊肃叶流，ref: 解析解，容差 <5 %）
**仓库** TensorSPH ｜ **轮次** selfstart m20930 ｜ **作业** cj3057（loong64 n315，8 核）

---

## 1. 案例与问题

平面泊肃叶流：`y = 0, H` 两壁无滑移，`x` 方向周期，均匀体积力 `g_x`。定常解析解

```
u(y) = g (H - y) y / (2 nu),  u_max = g H^2 / (8 nu),  tau_w = rho g H / 2
```

（White, *Viscous Fluid Flow* 3rd ed. §3-2.2；Batchelor §4.2。仓库内
`check_analytic_symbolically()` 用 sympy 复推 ODE 残差，残差恒等于 0。）

参数表（与既有 `benchmarks/poiseuille_mirror/run.py` 完全一致，本次未改物理设置）：

| 量 | 值 |
|---|---|
| H（槽高） | 0.1 m |
| g_x | 9.81 m/s² |
| nu | 0.01 m²/s |
| rho0 | 1000 kg/m³ |
| 声速 | 20 m/s（max Mach ≈ 0.061） |
| h/dx | 1.3（支撑半径 2h） |
| 壁面处理 | 镜像鬼粒子（`tensorsph.wall_mirror.plane_walls`），无滑移 |
| x 方向 | 12 列、周期镜像 ±1 周期 |
| t_end / 平均窗 | 1.0 s / 0.2 s |

**本单要回答的问题**：该算例的 Gate A 指标是

```
err_A = u_row_max / u_max(continuum) - 1
```

即把"分箱后的最大行速度"与"连续解的中心线值"相比。而网格行位于
`y_k = dx/2 + k dx`，当 `H/dx` 为偶数时**没有任何一行落在 `y = H/2`**；对抛物线而言，
最近行相对中心线的亏欠恰好是 `1 - (dx/H)^2`。若确实如此，则已发表报告里
"中心线精度 0.017 %（H/dx=12）"这一头号数字是**度量伪影与离散误差相消**的巧合，
而"误差随分辨率单调下降"的说法也不成立。

---

## 2. 方法

1. **解析侧**：不靠手抄，用闭式抛物线在网格行上采样，推出偶数网格的采样因子
   `u(y_nearest)/u(H/2) = 1 - (dx/H)^2`（§3.1 表格即该式的数值验证）。
2. **数值侧（新增）** `benchmarks/poiseuille_mirror/study_centreline_sampling.py`：
   在**未修改的**原算例上跑奇/偶配对阶梯 `H/dx = 15, 16, 17, 24, 25`，每档给出四个互相独立的
   中心线估计量：
   - `A_legacy` = 行最大 / 连续中心线（现行 Gate A）
   - `A_same_row` = 行最大 / 解析值在**该行自身 y** 处的值
   - `A_fit_full` = 全行抛物线最小二乘拟合，在 `y = H/2` 求值
   - `A_fit_mid` = `0.25 ≤ y/H ≤ 0.75` 带内拟合，在 `y = H/2` 求值
   奇数档在 `y = H/2` 上有真实行（无采样惩罚），偶数档没有 —— 这是天然的对照实验。
3. **复核既有阶梯（新增）** `benchmarks/poiseuille_mirror/analyse_centreline_ladder.py`：
   对 `out/adversarial_recheck.json` 里已发表的 8/12/16/24/32 五档，用其存档的逐行剖面
   重算无采样惩罚的中心线估计，量化"已发表趋势里有多少是度量伪影"。
4. **工具链**：PyTorch(CPU) + 既有 `rhs`/`plane_walls`；作业经 `run_compute` 递交外协方 所 loong64 节点（cj3057）。图示 matplotlib。

---

## 3. 计算结果

### 3.1 采样因子（解析，可为任意网格复算）

| H/dx | 最近行 y/H | `1-(dx/H)²` |
|---|---|---|
| 8 | 0.5625 | 0.984375 |
| 12 | 0.45833 | 0.993056 |
| 16 | 0.46875 | 0.996094 |
| 24 | 0.47917 | 0.998264 |
| 32 | 0.48438 | 0.999023 |

### 3.2 奇/偶配对实验（cj3057，t_end = 1.0 s）

| H/dx | 行在 y=H/2? | A_legacy (%) | A_same_row (%) | A_fit_full (%) | legacy−same_row (pp) | 预测 −(dx/H)² (pp) |
|---|---|---|---|---|---|---|
| 15 | 是 | **+0.063** | +0.063 | +0.239 | 0.000 | 0（无惩罚） |
| 16 | 否 | **−0.254** | +0.137 | +0.192 | **−0.391** | **−0.391** |
| 17 | 是 | **−0.014** | −0.014 | +0.116 | 0.000 | 0（无惩罚） |
| 24 | 否 | **−0.341** | −0.168 | −0.146 | **−0.173** | **−0.174** |
| 25 | 否→* | — | — | — | — | −0.160 |

**关键数字**：偶数档上 `A_legacy − A_same_row` 与闭式采样因子相差
`0.0005 pp`（H/dx=16）与 `0.0004 pp`（H/dx=24）。这是对"采样伪影"假设的直接、定量确认，
不是拟合出来的。相反，`A_legacy` 在 15→16→17 之间呈 **−0.317 / +0.240 pp 的锯齿**，
而 `A_fit_full` 同区间平滑单调（+0.239 → +0.192 → +0.116，步长 −0.047/−0.076 pp）。

\* `H/dx = 25` 档在节点上以 `free(): corrupted unsorted chunks` 异常中止（首次递交
cj3057 时 24 档已完成、25 档启动即崩），**该档数据缺失，未插值、未伪造**；详见 §5.1。

### 3.3 已发表阶梯的重判（8 → 32，无采样惩罚度量）

| H/dx | A_legacy (%) | **A_fit at H/2 (%)** | 采样因子 | 校正后残差 (pp) | 内部行 RMS (%) |
|---|---|---|---|---|---|
| 8 | +0.384 | **+1.957** | 0.984375 | +0.02 | 2.723 |
| 12 | +0.017 | **+0.666** | 0.993056 | +0.05 | 1.234 |
| 16 | −0.254 | **+0.192** | 0.996094 | −0.06 | 0.424 |
| 24 | −0.341 | **−0.146** | 0.998264 | −0.02 | 0.148 |
| 32 | −0.513 | **−0.275** | 0.999023 | −0.14 | 0.219 |

"校正后残差"= `A_legacy − [(1+A_fit)·(1−(dx/H)²) − 1]`，五档全部 ≤ 0.14 pp，说明
`A_legacy` 的取值几乎完全由「无采样惩罚的求解器误差 × 采样因子」解释。

---

## 4. 图

![centreline estimators](benchmarks/poiseuille_mirror/figures/centreline_estimators.png)

*四个中心线估计量随 H/dx 的变化。红色（现行 Gate A）随网格奇偶锯齿；三条无采样惩罚的
估计量平滑。灰色带为 ±5 % 判据带。*

![sampling factor](benchmarks/poiseuille_mirror/figures/centreline_sampling_factor.png)

*双对数：`|A_legacy|`、`|A_fit|` 与闭式采样因子 `(dx/H)²`。粗网格档 `A_legacy` 被
采样因子抬到 `A_fit` 之上（H/dx=8 时 0.38 % 的真误差被记作"几乎为零"）。*

![compensation](benchmarks/poiseuille_mirror/figures/centreline_compensated.png)

*假设的一次性检验：`(1+A_legacy)/(1−(dx/H)²)` 对 `1+A_fit`，全部点落在 y=x 上（标注为 H/dx）。*

---

## 5. 注意点与局限

### 5.1 未完成项（如实记录）

- `H/dx = 25` 档缺失：节点上 `free(): corrupted unsorted chunks`（glibc 堆破坏）中止进程，
  非 Python 异常。复算作业 cj3070（15/16/17）、cj3071（24/25）递交时排队 7–8 位，
  在预算内未到终态，故本报告不含其数据。**该堆破坏是否可复现尚未确定**，
  已作为独立风险登记（见 §6 建议），当前结论不依赖该档：奇偶对照在 15/16/17 与 24 上已闭合。
- 数据通道：`run_compute fetch` 返回的是**递交前**的 stage 快照，节点侧写出的 JSON 取不回来。
  因此 `out/centreline_sampling.json` 由
  `benchmarks/poiseuille_mirror/transcribe_centreline_study.py` 从作业 stdout **转录**
  （3 位小数百分数精度），`provenance.transcribed = true` 写在文件里。这是流程缺陷，
  建议平台修 `fetch` 语义（回传运行后目录）。
- `A_fit_mid` 在转录层没有对应日志字段，留 `null`；重判层（§3.3）有完整值。

### 5.2 物理/数值局限

- 校正后的中心线误差在 H/dx ≥ 16 后变号为负并缓慢增大（+0.19 → −0.15 → −0.28 %）：
  这是**残留的离散/可压缩性偏置**（Mach ≈ 0.061，O(M²) ∼ 0.4 % 量级），
  不是采样伪影；本文不主张它是收敛阶，因为误差过零，`|err|` 的幂律拟合在过零点附近无意义。
- 内部行 RMS 在 24 → 32 出现非单调（0.148 → 0.219 %），与已发表报告"All metrics improve
  monotonically"的措辞直接冲突；该措辞应视为**被证伪**（见 §6）。
- 算例本身仍是 2D、层流、固定 nu/g/H 的单点验证；不覆盖湍流、可变物性、粗糙壁。

---

## 6. 误差判定（逐项 vs 解析基准）

| 关键量 | 最差档 | 值 | 基准 | 相对误差 | <3 %？ | <5 %（算例判据）？ |
|---|---|---|---|---|---|---|
| 中心线 u_max（**校正后**） | H/dx=8 | 1.25025 m/s | 1.22625 m/s | **+1.96 %** | 是 | 是 |
| 中心线 u_max（校正后，H/dx≥12） | H/dx=12 | 1.23442 m/s | 1.22625 m/s | **+0.67 %** | 是 | 是 |
| 中心线 u_max（校正后，H/dx≥16） | H/dx=32 | 1.22288 m/s | 1.22625 m/s | **−0.28 %** | 是 | 是 |
| 中心线 u_max（**现行 Gate A 读数**） | H/dx=32 | 1.21996 m/s | 1.22625 m/s | −0.51 % | 是（但含 0.10 % 伪影） | 是 |
| 内部行 RMS | H/dx=12 | 1.234 % | 0（逐行解析） | 1.23 % | 是 | 是 |
| 内部行 RMS 最好档 | H/dx=24 | 0.148 % | 0 | 0.15 % | 是 | 是 |
| 壁面剪切（G4 拟合法） | H/dx=32 | 490.5·(1−0.00215) | 490.5 Pa | −0.22 % | 是 | 是 |

**是否 <3 %：是**（全部关键量；最差为 H/dx=8 的 +1.96 %）。
**是否 <5 %（算例池判据）：是**，且**校正后**度量下同样成立——即判据不是靠度量伪影维持的。

### 对既有报告的两处勘误（诚实性要求）

1. `docs/case_poiseuille_sph_report.md` §6 "u_max +0.017 %（H/dx=12）"：该值是采样伪影
   （−0.69 %）与求解器偏置（+0.67 %）相消的结果；**真实中心线误差为 +0.67 %**，
   仍满足判据，但精度声明被夸大约 40 倍。
2. 同文 §4.2 "All metrics improve monotonically with resolution" **不成立**：
   24 → 32 档内部行 RMS 由 0.148 % 升到 0.219 %，中心线原始读数由 −0.341 % 降到 −0.513 %。

---

## 7. 复现

```bash
# 1) 奇/偶配对实验（重计算，需递交算力）
PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_centreline_sampling.py \
    --resolutions 15 16 17 24 25 --threads 8 \
    --out benchmarks/poiseuille_mirror/out

# 2) 从作业日志转录（fetch 不回传运行后目录时的通道）
python benchmarks/poiseuille_mirror/transcribe_centreline_study.py \
    --job cj3057 --log /root/tp-agent/cluster/jobs/cj3057.log

# 3) 既有阶梯重判 + 图 + 测试（纯后处理，轻量）
PYTHONPATH=src:. python benchmarks/poiseuille_mirror/analyse_centreline_ladder.py
PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_centreline_figure.py
PYTHONPATH=src:. python -m pytest tests/test_case_poiseuille_centreline_sampling.py -q
```

产物：
- `benchmarks/poiseuille_mirror/out/centreline_sampling.json`（配对实验，含 provenance/缺失档）
- `benchmarks/poiseuille_mirror/out/centreline_ladder_analysis.json`（8–32 重判）
- `benchmarks/poiseuille_mirror/figures/centreline_{estimators,sampling_factor,compensated}.png`
- `tests/test_case_poiseuille_centreline_sampling.py`（11 项：合成抛物线代数 + 归档一致性）

## 8. 结论

Gate A 现行度量把"分箱行速度"与"连续中心线速度"相比，在偶数网格上结构性携带
`1-(dx/H)²` 的惩罚（H/dx=8 时达 1.56 %，超过 5 % 判据的 1/3）。本文用奇/偶配对数值实验
把该惩罚定量确认为与闭式因子相差 ≤ 0.0005 pp，给出三种无惩罚估计量，并把已发表阶梯重判：
算例的**真实**中心线精度在 H/dx = 8/12/16/24/32 上为 +1.96/+0.67/+0.19/−0.15/−0.28 %，
全部满足 <5 %（亦满足 <3 %）；同时证伪"全部指标随分辨率单调改善"的表述。算例池判据维持
通过，但中心线精度声明与单调性声明必须按 §6 勘误。