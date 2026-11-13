# 03 报告：FSI2 闸门载荷基准（第 2 轮，m17584）

## ① 问题设定与基准出处

弹性闸门溃坝 FSI（水柱 0.146×0.292 m，闸门 0.36×0.004 m 竖直、底部固支、顶部自由）中
**流体对结构的力**如何被传出去、以及结构侧的响应是否可信。三处独立参考（全部非本仓自算）：

| 量 | 参考 | 出处 |
|---|---|---|
| 静水壁面合力 | `F = ½ρ g h_s²`（h_s=0.292 m ⇒ 418.22 N/m） | 静水力学：Landau & Lifshitz, *Fluid Mechanics* §1–§2 |
| 大变形梁顶点挠度 | Kirchhoff elastica，`EI dθ/ds = P(x_tip−x(s))`，弧长打靶自行求解 | Bisshopp & Drucker, *Q. Appl. Math.* 3 (1945) 272（不抄表） |
| 小变形梁 | Euler-Bernoulli Green 函数 | 教科书梁理论（本轮**不**用作判据，见 ⑤） |

Antoci, Gallati & Sibilla (*Comput. Struct.* 85:879-890, 2007) 的实验闸门顶点时程：**WITHHELD**
（Unpaywall `is_oa=false`、OpenAlex `any_repository_has_fulltext=false`，m16214/m16352/m17519 三次
尝试 5 条取件通道全失败；连台架参数本身都带 `PARAMETER_PROVENANCE=not-read-from-the-article`）。
本单因此不再把它写成可通过判据，判据全部建在可独立复算的参考上。

## ② 方法与实现（本单做了什么）

1. **结构子循环**（`ElasticGateCase.rod_substeps` / `rod_substeps_effective()`，`step()` 内）：
   流体按声学 CFL 步进，杆在**一个流体步内用 n_sub 个子步**推进，n_sub 由杆的轴向频率自动定
   （`ω_ax·dt_rod ≤ 0.25`）。这解掉了 m16352 的 `fail_stage=large_deflection_validity`：
   当时"让梁理论成立所需的刚度"会让显式杆在流体声学步长下失稳，端到端校验被迫 `not_run`。
2. **静态平衡求解的静默失效加护栏**（`rod_static_equilibrium`）：m16406 审计发现 L-BFGS-B 对
   刚性杆（E≥1e10 Pa）返回**初始直线构型**，调用方把它读成 −100 % 模型误差。本单
   ①加"迭代数≤1 且载荷非零"即抛错的护栏（收敛解不会触发），②把每次求解的迭代数/目标值记入
   `_ROD_SOLVE_LOG` 供审计引用。**该缺陷仍未根治**（见 ⑥）。
3. **基准 workload** `benchmarks/fsi2_gate_load/run_fsi2_benchmark.py`：按 TensorBench
   `json_stdout` 契约输出单份 JSON，判据数字由平台从原始 stdout 按点指针复算。
4. 未改动：面元载荷传递（`load_transfer="panel"`）、固支弯曲能、`load_scale=1.0`（无标定）。

## ③ 参数表（无量纲数完整）

| 量 | 值 |
|---|---|
| dx | 12.5 mm / 10 mm（两分辨率） |
| h | 1.3 dx；CFL 0.3；dt = 1.95×10⁻⁴ s |
| ρ_f, c₀, γ, ν | 1000 kg/m³，20 m/s，7，1×10⁻⁶ m²/s |
| g | 9.81 m/s² |
| 水柱 / 槽 | 0.146×0.292 m / 0.584×0.45 m |
| 闸门 | H=0.36 m，t=0.004 m，ρ_s=2500 kg/m³ ⇒ ρ_A=10 kg/m² |
| 杆节点 | n=25（l₀=15 mm），子步数（物理闸门 E=1e6 Pa）= 1 |
| 无量纲载荷 | 静水：ρg h_s²/(2) 归一；A2：PL²/EI = 10.000 |
| 网格雷诺数 | 无（WCSPH，弱可压，ν 仅用于人工粘性尺度） |

## ④ 对照表（ref vs got，逐项判定）

| 判据 | ref | got | 误差 | <3 %? | <15 %? |
|---|---|---|---|---|---|
| 壁面合力 dx=12.5 mm | 418.220 N/m | 419.223 N/m | **+0.240 %** | ✅ | ✅ |
| 壁面合力 dx=10 mm | 418.220 N/m | 418.592 N/m | **+0.089 %** | ✅ | ✅ |
| 杆大变形顶点（同帧横向） | 291.819 mm | 299.255 mm | **+2.548 %** | ✅ | ✅ |
| 闸门顶点位移时程（Antoci 2007） | **WITHHELD** | — | 不可计算 | ✗ 不可判 | ✗ 不可判 |

是否 <3 %：**三项判据全部 <3 %**（+0.240 / +0.089 / +2.548 %）。唯一不可判的是实验时程，原因见 ①。

## ⑤ 收敛性 / 参数敏感性

- 静水载荷两分辨率：+0.240 %（dx=12.5 mm）→ +0.089 %（dx=10 mm），随分辨率单调改善，差值 0.15 个百分点；
  即面元传递**无系统性亏缺**（对照 m16214 成对和基线：−28.4 % / −20.4 % / −8.3 %）。
- 杆：A2 在 PL²/EI=10 大变形区同帧误差 +2.548 %；小变形区（E 抬到 1e9 Pa，δ≈168 mm）本单测到 −2.34 %，
  但该点 δ/H≈0.47 已出 Euler-Bernoulli 适用范围，故**不列为判据**（列为敏感点，避免用越界参考充判据）。
- 未完成：耦合时程的两分辨率收敛（dx=12.5 mm 单点 ~250 s 量级，未在预算内跑第二分辨率）。

## ⑥ 失败模式与工件

- **未根治**：`rod_static_equilibrium` 在 E≥1e10 Pa 时仍会停滞（现已被护栏拦下而不是静默出数），
  因此"刚门 + 梁理论"的端到端判据本轮**不可用**；端到端（流→固整链）静态判据留作下一轮，
  需把静态求解换成带预条件的 Newton/弧长延拓。
- **ref 不可得**：Antoci 2007 实验曲线仍 WITHHELD（三次任务一致结论，非本轮新发现）。
- **耦合时程仍不可信**：闸门物理参数（E=1e6 Pa）下静水载荷已使梁理论失效（δ→∞），
  故时程峰值不能与文献对照；本轮只作诊断，不参与判分。
- repro_cmd：`python benchmarks/fsi2_gate_load/run_fsi2_benchmark.py --out artifacts/fsi2_gate_load/bench.json`
- 工件：`artifacts/fsi2_gate_load/{bench_local.json,stdout_local.log,stderr_local.log,fig1_benchmark.png}`
- 文档：`docs/fsi2_gate_load/{01_survey_oss.md,02_method_selection.md,03_report.md}`

## ⑦ 下一阶段（迁移路线，已登记）

1. 静态求解换预条件 Newton + 弧长延拓 ⇒ 端到端（SPH 载荷 → W^T → 杆）静态判据可加回。
2. 若 owner 认可用 SPHinXsys `test_2d_fsi2`（圆柱后弹性梁流致振动，Zhang/Rezavand/Hu *JCP* 404:109135，2020）
   作为"FSI2"的真身，则需引入不可压/低频长时积分面与 TLSPH 固体，属独立战役节点。
