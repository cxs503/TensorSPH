# m4802（重开 · m4887）· TensorSPH main CI 红清除：模态 2 频率钉子按平台记录 + 判定改判平台无关物证

**结论先行**：本单要交付的功能是「把 main 上 4 项常红测试修成**既紧又真**的钉子，同时不放松任何判据」。
代码与测试层的处置**已先期落进 `main`**（PR #330–#334，m4795 / m4880 的合并序列），本单（PR #324 关闭后重开）
在**当前 main** 上复核该功能并补上此前从未入库的**交付账本**：本报告 + 其机器校验器
`docs/reports/m4802_make_report.py` + 守护测试 `tests/test_report_m4802.py`。
报告里每个数字都由脚本从已入库工件注入（`--check` 可核，手改即变红），因此"文档 vs 证据"不再是自述。

**判定口径一个字没改**：m = 2 毛细频率仍是 honest **FAIL**（基准算例 −43.53 %，CI 廉价算例 −31.78 %，
一周期守卫算例 −61.69 %），重叠判定 `within_15pct = false`、`within_3pct = false`。
本单修正的是"**用哪条证据去断言这些判定**"，不是判定本身。

## 0. 落地路线（诚实记账：这段代码现在在 main 的哪里）

| 交付物 | 状态 | 位置 |
|---|---|---|
| 模态 2 伪频率「按平台记录 + 未记录平台按包络兜底」 | 已入库 | `tests/test_oscillating_droplet.py::RECORDED_MODE2_ERR_PCT` |
| 15 % 门判定改判平台无关物证（入库 verdict + 现场守恒） | 已入库 | `tests/test_oscillating_droplet.py::test_oscillation_frequency_gate_is_not_met` |
| 一周期守卫算例：删除"永不可能通过"的 `|err| < 30 %` | 已入库 | `tests/test_oscillating_droplet_lamb.py::test_reproduces_the_recorded_capillary_frequency_deficiency` |
| 图件从"断言 png 存在"改判"可由入库 report 确定性重绘" | 已入库 | `tests/test_laplace_pressure_jump.py::test_figure_exists` + `benchmarks/laplace_pressure_jump/make_figure.py` |
| 证据账本 + 生成器 + `--check` | 已入库 | `benchmarks/oscillating_droplet/ci_pin_evidence.{py,json}` |
| 文档口径 | 已入库 | `docs/verification_oscillating_droplet.md` 附录 B / C |
| **交付账本（本报告）+ 其机器校验器 + 守护测试** | **本 PR 新增** | `docs/reports/m4802-*.md` / `m4802_make_report.py` / `tests/test_report_m4802.py` |

原 PR #324 的 8 个文件里 7 个的内容在 #330–#334 期间分片进了 main（本单以 `git diff` 逐文件核对，
见 §2.6），但**报告本身与"报告不得漂移"的守护从未入库**——于是 main 里存在一组按平台记录的钉子，
却没有一份能复算的说明解释"为什么必须按平台记录、为什么判定可以平台无关"。本单补的就是这个洞。

## ① 案例介绍（物理背景 / 几何 / 参数）

二维液滴（无限长柱截面）在表面张力作用下做自由振荡；线性理论给出 Rayleigh–Lamb 频率
$\omega_m^2=\sigma m(m^2-1)/(\rho R^3)$，$m=2$ 时 $\omega_2=\sqrt{6\sigma/(\rho R^3)}$，
本仓三个算例的 $\omega_2$ 分别是 207.85 / 73.48 / 73.48 rad/s。求解器是 PyTorch 2-D 弱可压缩 SPH +
consistent CSF 表面张力（`tensorsph.droplet_impact`），CPU、无 RNG。

| 量 | CI 廉价算例（测试主体） | 基准算例（文档/判定口径） | 一周期守卫算例（lamb） |
|---|---|---|---|
| 半径 R | 10.0 mm | 20.0 mm | 20.0 mm |
| 粒子间距 dx | 1.00 mm | 2.00 mm | 2.50 mm |
| 初始扰动 ε | 0.10 | 0.10 | 0.05 |
| 粒子数 N | 357 | 357 | 639 |
| 步数 / 时长 | 2791 / 1.80 T | 3947 / 1.80 T | 3508 / 1.00 T |
| σ / ρ₀ / c₀ | 7.2 N·m⁻¹ / 1000 kg·m⁻³ / 20 m·s⁻¹ | 同左 / 20 | 同左 / 10 |
| ν | 5e-05 m²·s⁻¹ | 5e-05 | 1e-03 |
| 解析 ω₂ | 207.85 rad/s | 73.48 rad/s | 73.48 rad/s |

（表中量取自 `ci_pin_evidence.json::case` 与两份 `validation_report*.json`，与本报告 §③ 同源。）

被修的四项红测（`main@799310b`）：
1. `test_oscillating_droplet.py` 的两条物理层钉子（`xfail(strict=True)` 在 loong64 runner 上恰好复现录得值 → XPASS → 红）；
2. `test_oscillating_droplet_lamb.py::test_mode_oscillates_near_the_capillary_frequency`（断言 `|err| < 30 %`，
   而它自己的文档记录 −61.69 %，从落地之日起就不可能通过）；
3. `test_laplace_pressure_jump.py::test_figure_exists`（断言一个二进制 png 存在，而该目录只跟踪 .py/.json，CI 上恒假）。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **先定因、再改测**：CI 廉价算例在 1.80 个解析周期内**发散**——`max|v|/c₀ = 1.63`（$c_0$ 是数值声速，
   超过 1 意味着有粒子在跑得比声速还快），拟合出的 $r^2$ 只有 0.79。即被测"频率"是**发散轨迹的伪频率**，
   其形态由浮点归约顺序决定 → 同代码同输入 loongarch64 −31.78 %、x86_64 +8.64 %。
2. **钉子改形（不是放松）**：`xfail(strict=True)` 的语义是"要求永不匹配"，当 CI runner 换成钉子取样平台时
   必然 XPASS。改为**按平台记录**（每平台 2 pp 容差，仍是紧的漂移检测），未记录平台只按已记录包络
   (−60 %, +30 %) 兜底并 xfail/skip——**不编造钉子，也不给静默通过**。
3. **门判定改判平台无关物证**：读已入库 `validation_report.json::verdict`（−43.53 %，`within_15pct=false`、
   `within_3pct=false`）+ 现场 run 的质量守恒（`mass_relative_drift < 1e-12`）。仅在"该平台已记录且记录值本身
   越门"时额外要求现场值也越门，因此**真改进会先在这里变红**，逼着先改记录再改判定。
4. **删掉永不可能通过的控制**：lamb 守卫算例的 −61.69 % 按平台记录（容差 1 pp），并保留平台无关断言
   `|err| > 15 %`——该量在任何平台都远未达门，且该 run 在 0.44 T 就有粒子离开钳位盒，本身无效。
5. **图件改判可复现性**：图是已入库 `validation_report.json` 的确定性函数，测试现场重绘并校验 PNG 魔数与
   >10 kB，缺 blob 时自动重建，既覆盖仓库内容也覆盖绘图路径本身。
6. **本单新增的守护**：`m4802_make_report.py` 把 §③ 三张表从三个 JSON 渲染后**就地注入**本文件
   （`<!-- BEGIN GENERATED:… -->` 标记区），`--check` 逐块比对；`tests/test_report_m4802.py` 调用它，
   并额外断言"honest FAIL 的口径没被改写"（verdict 里两个 within_* 必须为 false）与"图可重绘"。
   工具链：PyTorch 2.7.1 (CPU, loongarch64)、pytest、ruff、matplotlib(Agg)。

## ③ 计算结果（全部由脚本从已入库 JSON 注入，零手抄）

<!-- BEGIN GENERATED:error-table -->
| 算例 / 估计器 | 解析 ω₂ (rad/s) | 实测 ω (rad/s) | 相对误差 | < 3 %? | < 15 %? | 出处 |
|---|---|---|---|---|---|---|
| CI 廉价算例 · 二阶矩阻尼拟合<br>`R=10 mm, dx=1 mm, 1.8 T` | 207.85 | 141.79 | **-31.78%** | 否 | 否 | `ci_pin_evidence.json`<br>拟合 r² = 0.787 |
| CI 廉价算例 · 过零估计器<br>`R=10 mm, dx=1 mm, 1.8 T` | 207.85 | 131.83 | **-36.57%** | 否 | 否 | `ci_pin_evidence.json`<br>与拟合差 4.79 pp |
| CI 廉价算例 · 动能半衰估计器<br>`R=10 mm, dx=1 mm, 1.8 T` | 207.85 | 117.37 | **-43.53%** | 否 | 否 | `ci_pin_evidence.json`<br>第三估计器，无阻尼假设 |
| 基准算例 · 二阶矩阻尼拟合<br>`R=20 mm, dx=2 mm, 1.8 T` | 73.48 | 41.50 | **-43.53%** | 否 | 否 | `validation_report.json`<br>文档/判定口径算例（入库） |
| 一周期守卫算例 · 二阶矩阻尼拟合<br>`R=20 mm, dx=2.5 mm, 1.0 T` | 73.48 | 28.15 | **-61.69%** | 否 | 否 | `validation_report_lamb.json`<br>particle left the box at step 778 (0.44 T) |

五个（算例, 估计器）组合的误差绝对值区间 31.78 % – 61.69 %：**没有任何一个 < 3 %，也没有任何一个 < 15 %**。同一算例三个估计器之间的分歧最大 11.75 pp，这本身就是「该量不可作为频率结论」的证据。
<!-- END GENERATED:error-table -->

**平台钉子与本机复核**（同一量在不同 CPU 上不是同一个数）：

<!-- BEGIN GENERATED:pins -->
| 平台 | 记录的模态 2 伪频率误差 | 越 15 % 门? | 钉法 |
|---|---|---|---|
| `x86_64`（GitHub 托管 runner） | **+8.64%** | 否 | `RECORDED_MODE2_ERR_PCT['x86_64']`，容差 2.00 pp |
| `loongarch64`（本机，CI runner） | **-31.78%** | 是 | `RECORDED_MODE2_ERR_PCT['loongarch64']`，容差 2.00 pp |

同一份代码、同一份输入的跨平台散布 = **40.42 pp**（-31.78% … +8.64%）：这是「伪频率按平台记录、判定按平台无关物证」的直接依据，也是**不得取平均**的理由（取平均会造出一个两个平台都不成立的 −11.57 %）。
<!-- END GENERATED:pins -->

**为什么这条轨迹不能当频率用**（稳定性与守恒性证据）：

<!-- BEGIN GENERATED:stability -->
| 量 | CI 廉价算例 | 基准算例（R=20 mm） | 判据 |
|---|---|---|---|
| 粒子数 N | 357 | 357 | — |
| 步数 / 模拟时长 | 2791 / 1.80 T | 3947 / 1.80 T | — |
| dt (s) | 1.950e-05 | 3.900e-05 | — |
| 面积等效半径 r_eff 变化 | 发散（见下行的 max\|v\|） | 20.0 mm → 99.9 m（**×4987**） | 守恒应为 ×1 |
| max\|v\| / c₀ | 1.632 | — | ≪1 才是弱可压 |
| 质量相对漂移 | 0.0e+00 | — | <1e-12（机器精度） |
| 线程确定性（1/4/8/16 线程, 200 步） | max\|Δx\| = 0.0e+00 m | — | 位级一致（无 RNG、无原子归约） |
| 守卫算例稳定性 | — | particle left the box at step 778 (0.44 T) | 不得离开钳位盒 |

说明：**质量守恒与线程确定性都成立**，所以失败点不在守恒律或并行不确定性上——它是格式缺陷（自由面欠密度 + 钳位 Tait 律导致的压强塌陷），在 1.8 个解析周期内把液滴的等效半径从 20.0 mm 推到 99.9 m（×4987，面积应守恒），任何从这条轨迹上拟合出来的频率都只能是伪频率。
<!-- END GENERATED:stability -->

## ④ 结果图

![m4802 error vs gate](docs/figures/m4802_error_vs_gate.png)

图 1：(a) 五个（算例, 估计器）组合的模态 2 频率误差，红虚线 = 15 % 门、黑点线 = 3 % 严格门；
可见**五者全部越 15 % 门**（逐项数字见 §③ 表 1），同一算例三个估计器之间的分歧本身即是"伪频率"的直接证据。
(b) 同一份代码、同一份输入在两个平台上的钉值（见 §③ 表 2）：x86_64 的 +8.64 % 落进 15 % 门内
**纯属巧合**——产生它的 run 与 loong64 一样在发散（`max|v|/c₀` 见 §③ 表 3），这不是物理改进。
图由 `docs/reports/m4802_make_report.py::render_figure` 从同一批 JSON 渲染（matplotlib Agg），
与 §③ 的表同源，可复算。

![m4802 release trace](docs/figures/m4802_release_trace.png)

图 2：基准算例（R = 20 mm、dx = 2 mm、1.8 T）的**实测轨迹 vs 解析解**。(a) 模态 2 信号 $s(t)=(a-b)/(a+b)$
与 Rayleigh–Lamb 阻尼余弦 $s_0e^{-\lambda t}\cos\omega_2 t$（$\omega_2$ 与 $\lambda$ 取自
`validation_report.json::reference`）：解析解包络衰减，实测包络**反而增长**（$s: 0.097 \to 0.936$）——
这时从轨迹上拟合出的"频率"当然是伪频率；(b) 同一 run 的动能（对数轴）由 $7.6\times10^{-5}$ J 涨到
$5.8\times10^{7}$ J：表面张力在向液滴**注入**能量（自由面欠密度 + 钳位 Tait 律的压强塌陷），
守恒律本身没坏（质量漂移 0.0e+00），坏的是格式。图由
`docs/reports/m4802_make_report.py::render_trace_figure` 从已入库 `trace_data.json` + `validation_report.json`
渲染，可复算。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

- **坑 1（语义）**：`xfail(strict=True)` 不是"更严格的钉子"，而是"要求永不匹配"。平台相关的量必须写成
  "按平台记录 + 平台无关判据"两件套，否则 CI 一换 runner 就红。
- **坑 2（口径）**：门槛不该依赖 CPU。x86_64 的 +8.64 % 恰好进 15 % 门，但它是同一缺陷的另一种噪声实现；
  拿它翻案等于用"哪一种浮点归约顺序"去判物理。
- **坑 3（产物）**：二进制物证（benchmarks 下的 png）不入库，任何"断言某 png 存在"的测试在 CI 上恒假；
  正确写法是"可由入库数据确定性重绘"。
- **坑 4（环境）**：`tensorsph` 是 editable 安装、可能指向别的 worktree，跨 worktree 验证必须显式
  `PYTHONPATH=$PWD/src`，否则测的不是本 worktree 的代码。
- **局限**：本单**没有**修格式缺陷（自由面欠密度 + 钳位 Tait 律导致的压强塌陷）——那才是 −43.53 % 与
  Laplace −37.4 % 的真因，属实现欠账，另行处理。钉子的紧度只在本机录得的平台上成立（2 pp / 1 pp）；
  换 torch 版本、线程数或编译选项后若漂移，测试会先变红，届时**重测**而不是改容差。
- **适用范围**：本报告的数值只对上述三个算例（R = 10/20 mm、dx = 1/2/2.5 mm、单相弱可压 SPH）成立，
  不外推到多相或三维。

## ⑥ 误差判定（逐项，显式回答「是否 < 3%」）

<!-- BEGIN GENERATED:gate-table -->
| 关键量 | 基准 | 实测 | 相对误差 | < 3 %? | < 15 %? |
|---|---|---|---|---|---|
| CI 廉价算例 · 二阶矩阻尼拟合 | 207.85 rad/s | 141.79 rad/s | **-31.78%** | 否 | 否 |
| CI 廉价算例 · 过零估计器 | 207.85 rad/s | 131.83 rad/s | **-36.57%** | 否 | 否 |
| CI 廉价算例 · 动能半衰估计器 | 207.85 rad/s | 117.37 rad/s | **-43.53%** | 否 | 否 |
| 基准算例 · 二阶矩阻尼拟合 | 73.48 rad/s | 41.50 rad/s | **-43.53%** | 否 | 否 |
| 一周期守卫算例 · 二阶矩阻尼拟合 | 73.48 rad/s | 28.15 rad/s | **-61.69%** | 否 | 否 |
| 质量守恒（现场 run） | 0 | 0.0e+00 | 0 % | 是 | 是 |
| 线程确定性（1/4/8/16 线程 × 200 步） | 位级一致 | max\|Δx\| = 0.0e+00 m | 0 % | 是 | 是 |

**逐项回答**：频率类关键量 **没有一项 < 3 %，也没有一项 < 15 %**——honest FAIL；守恒与确定性两项为 0 %（< 3 %），说明失效不在守恒律或并行不确定性。跨平台散布见 §③ 的平台钉子表，已如实记录、**不取平均掩盖**。
<!-- END GENERATED:gate-table -->

与 `docs/verification_oscillating_droplet.md`、`docs/verification_oscillating_droplet_lamb.md`
的判定一字不差；跨平台散布（§③ 表 2）是"平台相关"的定量表述，**不取平均掩盖**——
两个钉值的算术平均在两个平台上都不成立。

**复现命令**：
```
PYTHONPATH=src python docs/reports/m4802_make_report.py --check          # 本报告 vs 已入库工件（不跑求解器）
PYTHONPATH=src python -m pytest -q tests/test_report_m4802.py            # 守护：块新鲜度 + honest FAIL 口径 + 图可重绘
PYTHONPATH=src python benchmarks/oscillating_droplet/ci_pin_evidence.py --check   # 账本 vs 本机实跑（1 个算例，~170 s）
PYTHONPATH=src python -m pytest -q tests/test_oscillating_droplet.py tests/test_oscillating_droplet_lamb.py
```