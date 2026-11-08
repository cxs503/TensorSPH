# 接触角仪器修复（E3 欠账）：探针先校准，样品如实记账（m6503）

**仓库**: TensorSPH ｜ **工件**: `benchmarks/contact_angle/{report.json,contact_angle.png}` ｜
**测试**: `tests/test_contact_angle_instrument_m6503.py`（8 个）、
`tests/test_verification_ledger.py::test_contact_angle_harness_runs_and_its_instrument_is_repaired`
｜ **判据由 JSON 注入**: `docs/verification_contact_angle.md` 由 `run.py::make_doc` 生成

## 1. 欠账是什么

`docs/verification_contact_angle_m6453.md` 记的 E3：合并的接触角 BC
（`apply_contact_angle_boundary`）有可执行的 harness，但**仪器层 10 行解析行里 8 行是 NaN**，
唯一"读出"的 90°/120°/150° 全部落在 89.0°–89.8°。于是该能力**无法登记 claim**：
不是"测得不准"，而是**没有读数**。

## 2. 修了什么（三处真缺陷，均为 benchmark 代码，未改 `src/`）

| # | 缺陷（旧版） | 后果 | 修法 |
|---|---|---|---|
| D1 | `relax()` 传 `math.radians(t)` 并把弧度存进 `theta_set_deg`，却用 `error_deg = 读数(度) - 设定(弧度)` | 物理层表格结构性不可判：`60°` 一行存成 `1.047`，误差恒 >10° | 两侧统一为**度**，`error_deg == measured - set`（`test_physics_rows_are_in_degrees`） |
| D2 | `synthetic_interface` 直线支给出 `b=+tanθ`（增湿/非增湿都错），圆弧支圆心/半径公式使弧**根本不过壁面** | 直线全 NaN；圆弧恒 ~89°（与"角度无关"）| 精确参数化：直线 `P(s)=(x_c,0)+s(-cosθ,sinθ)`；圆弧 `C=(x_c,0)-R_c(sinθ,cosθ)`，`a0=π/2-θ`，对 θ=30°–150° 全域成立 |
| D3 | 估计器只取"x 向最低包络 + 二次拟合 + `θ=π/2-arctan b`"，接触点取包络 argmin | 非增湿（θ>90°）界面在接触线处**悬垂**，x 向包络是壁面而非自由面 → 一律读成 ~89°；接触点被一粒子量化 | 双向包络（x 最低 + y 最右）+ **圆拟合**（Kasa，对圆界面精确）/按共线性 `λ0/λ1≤1e-9` 回退直线 + 接触点取"模型与壁面的交点" + `θ=atan2(t_y,-t_x)` |

另外：`make_doc` 把"仪器层 10/10 合成界面全部在 2° 内"写死为字面量，而同一份
`report.json` 的 `all_within_2deg == false` —— 文档与工件互相矛盾。现改为从工件注入最差误差，
并由 `test_doc_numbers_come_from_the_artifact` 钉住那句假话不得回潮。

## 3. 结果（三层分开记账，数字全部由 `report.json` 注入）

* **仪器层（精确界面）**：直线 × 圆弧 × {30,60,90,120,150}° 共 10 行，最大偏差 **0.0003°**
  （门槛 2°）。估计器无偏、模型选择不误判。
* **分辨率层（粒子晶格，已知解析角）**：R/dx = 6.7 时最差 **3.7°**；R/dx = 10 → 1.3°、
  13.3 → 1.3°、20 → 0.85°。θ=30° 在 R/dx ≤ 13.3 时**包络点不足 → NaN**（不编数字）。
  这是任何接触角读数在生产分辨率下的误差地板。
* **物理层（松弛液滴，出荷配置 R/dx = 6.7）**：设定 60/90/120° 读得 152.9/163.6/168.4°，
  其中 60° 算例**流体穿壁 0.98 dx**、90° 伪流 0.109×`sqrt(σ/ρR)`（门 0.1）→ 两个算例**不可采信**；
  120° 算例可采信但偏差 +48°。判定 **WITHHELD**：不可采信的读数既不能证明 BC 准，
  也不能证明 BC 不准。

## 4. 为什么不再往前一步（局限）

1. 物理层的两个门是**先于读数**的：穿壁把接触线搬到了壁面以下，"接触角"没有定义；
   伪流超标说明液滴未静止，读数取的是呼吸中的表面。
2. 120° 可采信却偏 +48° 的读数是**已知模型误差**：整滴圆拟合只在液滴仍是圆帽时成立，
   而穿壁/铺展后的形状偏离圆帽；分辨率层量到的 3.7° 地板不覆盖"形状去圆"这项误差。
3. 关账路径（下一单，已在 §5 写明判据）：把 `R/dx` 提到 ≥ 13 并让穿壁 < 0.5 dx，
   同时复核伪流 < 0.1，重跑三档角度；届时 `test_shipped_physics_run_is_recorded_as_withheld`
   会转红，强制把本文与 `docs/verification_contact_angle.md` 一起更新。

## 5. 证据指针

* `benchmarks/contact_angle/report.json`（`instrument` / `resolution` / `physics` 三节）
* `benchmarks/contact_angle/contact_angle.png`（(a) 精确界面 (b) 晶格 (c) 松弛轨迹）
* `docs/verification_contact_angle.md`（`make_doc` 生成，六要素齐全）
* 账本：`benchmarks/verification_ledger/waivers.json` 中 `benchmarks/contact_angle/` 的豁免理由已更新为
  "仪器已修好、物理层仍不可采信"；顺带补 `benchmarks/ice_floe_added_mass/` 的目录豁免
  （HEAD 上该 harness 无 claim 无豁免，`uncovered_benchmark` 已在账本里现形）。
