# 验证声明台帐：Young–Laplace 压差的六条读数与唯一裁定

**仓库**: TensorSPH ｜ **任务**: m3544 ｜ **台账**: `tensorsph-verification-claims` ｜ **机器门禁**: `tests/test_verification_ledger.py`

> 本文由 `benchmarks/verification_ledger/make_doc.py` 从 `ledger.json` 生成，全部数字程序化注入，无手抄。

## 0. 一句话结论

同一个物理量（二维液滴 `$\Delta p=\sigma/R=360.0$ Pa）在本仓被 6 篇验证文档引用了 6 个互不相同的误差值，跨度 **140.9 个百分点**（-93.6% … +47.3%）。台帐逐条溯源后裁定：

- 唯一**权威**读数为 **-34.0%**（C06，absolute_time_avg），证据等级 **WITHHELD**；
- 与平台 15% 判据的关系：**不通过**；与论文 3% 判据的关系：**不通过**；
- 全仓唯一的 `✅ PASS` 声明（C03）在本台帐中被**撤下**，理由见 §4。

## 1. 为什么需要这台帐

表面张力 CSF 模块（m277 合入）的验证欠账已经陆续被多张单子清理过（m2174 / m2592 / m2971 / m3028 …）。每张单子都诚实地报了自己的误差，但它们**各用各的测量口径、各用各的网格**，而且 `benchmarks/<name>/validation_report.json` 这类工件路径会被后来的单子**原地覆盖**。结果是：

1. 下游读者（含平台能力审计）无法判断该引哪个数；
2. 一个「+5.6%，15% 门内」的读数可以与「−93.6%」并存而无人指出二者不能同时成立；
3. 历史声明的证据可以在文档一字未改的情况下**从仓库里消失**。

台帐的做法：不重写任何人的报告，而是给每条声明补上「原文引语 → 引用工件 → 工件现值 → 时间序列自洽性 → 判定」的完整链条，并把它变成一个可执行的测试门禁。

## 2. 算例与口径分类

| 项 | 值 |
|---|---|
| 几何 | 二维静止自由液滴（无限长柱截面），六角密排，单相，无重力 |
| 半径 R | 0.02 m |
| 表面张力 σ | 7.2 N/m（故意取 10× 水，压缩毛细时间） |
| 参考密度 ρ0 | 1000.0 kg/m³ |
| 解析基准 σ/R | **360.0 Pa** |

四种口径互不等价，混用是误差跨度大的主要原因：

| 口径 | 定义 | 台帐评价 |
|---|---|---|
| `absolute_instant` | p_core read as absolute pressure at a single time snapshot; ambient reference = 0 Pa. Forbidden by m2971 P2 (snapshot sits inside the breathing-mode noise band). | |
| `absolute_time_avg` | p_core absolute, averaged over the tail window of the run. The definition adopted by m2971 P2 as the only publishable reading. | |
| `band` | p_core minus a shell/interface band value (p_core - p_shell, p_center - p(0.80-0.95R), bandB). Confounded: the shell band lies in the free-surface density-deficit region, so the measured value is dominated by an artefact the law does not contain. | |
| `plateau` | p_plateau(r<0.6R) - p_ambient(r>1.0R). Correct in principle, but in this configuration the plateau carries a large fluctuation (see plateau_std) because the interior is on the EOS pressure floor. | |

## 3. 六条声明逐条溯源

| ID | 文档 | 引语 | 口径 | 声明误差 | 文档判定 | 工件现值 | 溯源 | 时序自洽 |
|---|---|---|---|---|---|---|---|---|
| C01 | `docs/verification_surface_tension_laplace.md` | `-93.6%` | band | -93.60% | WITHHELD | +5.60% | OVERWRITTEN | n/a |
| C02 | `docs/verification_surface_tension_consistent.md` | `-75.6%` | band | -75.60% | WITHHELD | -78.56% | OVERWRITTEN | n/a |
| C03 | `docs/verification_csf_laplace_consistent.md` | `+5.6%` | absolute_instant | +5.60% | PASS | +5.60% | RESOLVED | FLAGGED |
| C04 | `docs/verification_csf_curvature_bias.md` | `409.8` | band | -78.56% | WITHHELD | -78.56% | RESOLVED | FLAGGED |
| C05 | `docs/verification_laplace_plateau.md` | `47.3%` | plateau | +47.31% | WITHHELD | +47.31% | RESOLVED | n/a |
| C06 **★** | `docs/verification_laplace_pressure_floor.md` | `-34.0%` | absolute_time_avg | -33.99% | WITHHELD | -33.99% | RESOLVED | n/a |

`★` = 台帐裁定的权威行。`溯源` 状态含义：`RESOLVED` 工件仍持有该数；`OVERWRITTEN` 该路径已被后续算例覆盖，原始证据不再存在于仓库。

### 3.1 漂移与噪声的直接证据（由工件自身的时间序列重算）

| 声明 | 工件 | 样本 | 原始范围 [Pa] | 尾窗均值 ± σ [Pa] | 尾窗均值误差 | 最优绝对误差 |
|---|---|---|---|---|---|---|
| C03 | `benchmarks/laplace_law/validation_report.json` | 5 | 137.9 … 416.7 | 143.4 ± 7.8 | -60.2% | 15.76% |
| C04 | `benchmarks/csf_consistent/validation_report.json` | 13 | 0.0 … 2395.4 | 276.4 ± 163.8 | -23.2% | 10.19% |

这两行是台帐自己从工件算出来的（不是任何文档的声明），它们给出 §4 的核心证据。

## 4. 三个发现

### F1　唯一的 `✅ PASS` 不支持（C03）

`docs/verification_csf_laplace_consistent.md` §3.2 记 `Δp = 380.2 Pa (+5.6%) ✅ Within 15% platform bar`。该读数的工件 `benchmarks/laplace_law/validation_report.json` 自带 `history.p_center`：

- 记录窗口内 `p_center` 由 **416.7 Pa 单调衰减到 137.9 Pa**，尾窗均值 **143.4 Pa（-60.2%）**；
- 也就是说这条轨迹里**最优的绝对误差也只有 15.76%**，从未进入 15% 门；
- 而 `+5.6%` 对应的 380.2 Pa 落在轨迹记录窗口**之外**（更早的瞬态），即该读数是在液滴尚未平衡时取的；
- 该算例还是全仓最粗的一档：dx = 4 mm、N = 88、380 步、无质量标定、c0 = 20（m2971 把此档的初始核和偏置造成的虚假核心压强记为 2395 Pa）。

**裁定：撤下该 PASS**。原文不删（删掉一个错误判据比标注它更糟），改为在文档头部挂台帐指针块；`audit_allowlists.withdrawn_pass_claims` 记录这次撤下，门禁会保证「撤下的 PASS 不得悄悄复活」。

### F2　两条声明的证据已被原地覆盖（C01、C02）

- **C01** `docs/verification_surface_tension_laplace.md` 引 `-93.60%`，但 `benchmarks/laplace_law/validation_report.json` 现值 `+5.60%`。
- **C02** `docs/verification_surface_tension_consistent.md` 引 `-75.60%`，但 `benchmarks/csf_consistent/validation_report.json` 现值 `-78.56%`。

根因是**工件路径按名字复用**：后来的单子把 `benchmarks/<name>/validation_report.json` 整体覆盖，于是早期单子的证据消失而文档一字未改。这不是谁的失误，而是缺少约定；建议见 §6 P1。

### F3　没有单一权威读数，跨度 140.9 个百分点

六条声明从 +47.3% 到 -93.6%。台帐裁定权威行为 **C06（-33.99%，absolute_time_avg）**，依据：它是唯一满足 P2 所定「绝对压强 + 时间平均」口径、且工件可完全复算（`benchmarks/laplace_pressure_floor/report.json` 的 `runs[2]`）的读数。

## 5. 权威裁定与误差判定

| 判据 | 门限 | 权威读数 | 是否通过 |
|---|---|---|---|
| 平台精度标准 | 15% | -33.99% | **否** |
| 论文级标准 | 3% | -33.99% | **否** |

**显式回答「是否 < 3%」：否。** 是否 < 15%：否。证据等级 **WITHHELD**。

裁定理由（沿用 m2971 的归因，本台帐未改动物理结论）：Authoritative reading is -34.0% against sigma/R = 360 Pa, far outside the 15% platform bar and outside the 3% publication bar. The residual is a free-surface pressure closure defect (Tait clamp p>=0 cannot carry sigma/R on a -6%..-31% density-deficit shell), not a surface-tension discretisation defect: the cut-force identity (global force) is inside 15%.

## 6. 机器门禁与交接

`tests/test_verification_ledger.py` 共 13 项，全部绿：

1. 每条引语必须在它引用的文档里**逐字存在**（防手抄漂移）；
2. 每个工件 + JSON 指针必须仍可解析（防工件消失）；
3. 权威行必须 `RESOLVED` 且 **(measured−ref)/ref 的重算值 = 工件存值**（算术自洽）；
4. 证据等级必须由权威读数推出（15% 门），不能手写；
5. 不得存在未登记的「不支持的 PASS」；撤下的 PASS 必须仍在原件里挂指针块；
6. 被覆盖工件集合必须与声明清单一致（**新出现的孤儿会让门禁红**）；
7. 单快照声明若与自身轨迹尾段矛盾必须被标记；
8. 派生轨迹必须证明唯一 PASS **在其自身算例的稳态下不可达**；
9. 文档间矛盾必须被显式记录，且跨度数值必须与声明一致；
10. 台帐涉及的每篇文档都必须挂台帐指针块；
11. 提交的 `ledger.json` 必须可由 `audit.py` 复现（禁止手改）。

### 交接

- **P0（不变，来自 m2971）**：自由面压强闭合（shell-pressure 外推）。验收判据不变：`benchmarks/laplace_pressure_floor/run.py` 的 `absolute_time_average_tail` 在 c0=20/5 × dx=0.002/0.0015 四点全部 <15%。修好后 **重新跑 `audit.py`**：`test_evidence_grade_follows_from_the_authoritative_reading` 会自动接受新的等级，而 `test_laplace_dp_is_not_claimed_within_3pct` 会红，提醒更新本文。
- **P1（新增，本台帐）**：验证工件路径加任务号（`benchmarks/laplace_law/m3544/…`），禁止跨单原地覆盖。C01/C02 的孤儿化正是这条缺失的直接后果。
- **P2（新增，本台帐）**：`p_center` 报告必须同时落盘时间序列（C03/C04 能被抓出来，正是因为它们的工件恰好带了 `history`）。没有轨迹的读数一律按 `absolute_instant` 处理，不得用于判定门限。

## 7. 注意点与适用范围

- 本台帐**只覆盖算例 A**（σ = 7.2 N/m, R = 0.02 m, σ/R = 360 Pa）。`docs/validation/surface-tension-laplace.md` 与 `docs/validation/young_laplace_csf.md` 用的是另一套参数（σ = 0.072 N/m, R = 0.1 m / σ/R = 0.72 Pa），误差在 4.37%…84% 之间，**未纳入本次裁定**——它们的口径与网格均不同，混入会再次制造不可比数字。这是明确的未完成项。
- 台帐不评价曲率、切割力、伪流三项：这些量在多张单子里已经进入 15% 门（切割力恒等式、法向对齐、伪流 < 0.1 毛细速度），本台帐的矛头只对准**端到端压差**。
- 引语是**逐字子串**匹配，因此文档若改写措辞，门禁会红并要求同步更新 manifest——这是有意的，不是误报。

## 8. 结果图

![ledger](benchmarks/verification_ledger/ledger.png)

左：六条声明的误差与口径（红=权威行，蓝线=15%/3% 门）。右：C03 所引工件自身的 `p_center` 轨迹在记录窗口内由 416.7 Pa 衰减至 137.9 Pa，而 `+5.6%` 头条值 380.2 Pa 不在该衰减段上——这是撤下那个 PASS 的直接证据。

