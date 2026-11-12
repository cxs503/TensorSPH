# 独立复核：Plane Couette 壁面剪切 FAIL 判定（m16438）不成立于壁面模型

**复核对象**：`docs/dev_report_couette_verification.md`（commit `9c512a6`，TensorSPH main）
及其工件 `benchmarks/couette/out/results.json`。
**复核方式**：只读工件 + 同几何同估计器、仅改终端时间的重测（剧本 `benchmarks/couette/shear_reading_audit.py`）。
**独立性限制**：复核者与原作者同平台不同席位；重测脚本复用 `benchmarks/couette/run.py` 的
`run_one`（同一求解器、同一 lattice、同一指标代码），因此能反驳"读数/时程"类断言，
**不能**替代对壁面边界条件实现本身的代码级审查（见 §6 未做项）。

## 1. 被复核的断言

原文标题与 §4 结论：

> Verdict: FAIL at predeclared gates (wall shear under-predicted ~33%, steady drift 5.5%)
> "Wall shear under-prediction (33% deficit) … consistent with the known SPH wall
> treatment limitation: the no-slip condition is imposed through the kernel weight
> … so the effective wall position is offset from the nominal boundary."

即：**壁面剪切少 33% 被判为壁面处理缺陷**。

## 2. 复核对表（机器生成，零手抄）

数据源：`benchmarks/couette/out_full/shear_reading.json`（本次重测）与
`benchmarks/couette/out/results.json`（已合并工件）。生成器：`shear_reading_audit.py`，
自写 `out_full/summary.md` 表；下表逐字来自该 JSON。

| run | H/dx | t_end [s] | steps | drift | interior L2 | flux | tau 割线 | tau 内部拟合 | 下壁 slip/dx | 上壁 slip/dx |
|---|---|---|---|---|---|---|---|---|---|---|
| ladder-r8（已发布, `--quick`） | 8 | 0.150 | 615 | 0.0545 | 0.0565 | 0.9284 | **0.6735** | 0.8449 | 0.313 | 1.155 |
| ladder-r8-steady（本次） | 8 | 0.600 | 2462 | 0.0354 | 0.0299 | 1.0391 | **0.9566** | 0.9337 | 0.501 | 0.067 |
| ladder-r16-steady（本次） | 16 | 0.600 | 4923 | 0.0155 | 0.0293 | 1.0442 | **0.9952** | 0.9589 | 0.794 | -0.108 |

关键事实：`DESIGN.md`（先于任何运行写定）规定 `t_end = VISCOUS_TIMES·H²/ν = 1.2 × 0.5 s = 0.60 s`；
`--quick` 分支把 `t_end` 硬改成 **0.15 s**（0.3 个黏性时间），`run.py:715-718`。
同一格子、同一估计器 `τ = ρν(u₁-u₀)/dx`，**只把 t_end 放回设计值**：

* H/dx = 8：0.6735 → **0.9566**（−32.6 % → −4.3 %），同时 interior L2 5.65 % → 2.99 %；
* H/dx = 16：**0.9952**（−0.5 %）。

因此"壁面剪切亏 33 %"是**启动瞬态**（0.3 个黏性时间时流量仍在爬升，drift 5.45 %
对 1 % 门）而不是壁面传递的动量通量缺陷。若它是壁面模型缺陷，把 t_end 从 0.3 延长到
1.2 个黏性时间不应改变比值；实测改变了 28 个百分点。

## 3. 分级发现

**F1（阻断，证据-断言不符）**：标题级结论"wall shear under-predicted ~33 %"用的是
未到设计终端时间的状态，且原文 §2 已声明 `t_end = 1.2 H²/ν`。同程序在声明时间下同一
指标为 −4.3 %（H/dx=8）、−0.5 %（H/dx=16，`out_full/shear_reading.json`）。该结论必须
改写为"0.3 个黏性时间处的瞬态读数"，不得再表述为壁面模型缺陷。

**F2（阻断，机制归因错误）**：原文把亏缺归因于核权重壁面偏移，但工件里同一运行的两个
壁面读数自相矛盾——H/dx=8 短时程下 `tau_bottom = 0.6735`、`tau_top = 0.9920`。定常
Couette 中剪应力沿 y 为常数，两侧 1-dx 割线相差 32 个百分点只能是壁层局部形状/瞬态
噪声，不能是"壁面传递少一截"。设计时程下该不对称消失（0.9566 / 0.9952 与拟合 0.9337 /
0.9589 同阶）。

**F3（非阻断，证据缺失）**：原文 §7 列了 4 张图（`couette_profiles.png` 等）位于
`benchmarks/couette/out/`，但该目录在 main 上只有 `results.json`（`ls` 复核）。图未入库，
图注不可核。本次复核图 `benchmarks/couette/out_full/shear_reading.png` 随工件提交。

**F4（非阻断，诚实门仍未过）**：放回设计时程后 **steady drift 仍不过门**
（1.55 % vs 1 %，H/dx=16），flux +4.4 %（过 5 % 门）、interior L2 2.93 %（过门）、
壁面剪切 0.5 %（过门）。也就是说 1.2 个黏性时间仍略欠定常，**任何剪切数字自带 O(2–4 %)
的松弛不确定度**；要过 drift 门需 t_end ≈ 2–3 H²/ν。这属于真实欠账，不得抹去。

**F5（非阻断，估计器口径未声明）**：工件里 `wall_shear_ratio` 只用 1-dx 割线，而
`line_fit_slope_ratio` 同时存在且更稳。两者在 H/dx=8 设计时程下相差 2.3 pp
（0.9566 vs 0.9337，后者 −6.6 % **不过** 5 % 门）。报告应同时给出两个口径并声明门用哪个，
否则同一个运行可以出 PASS 也可以出 FAIL。

## 4. 误差判定（逐项回答"是否 < 3 %"）

| 量 | 基准 | 实测（H/dx=16, t_end=1.2 H²/ν） | 相对误差 | < 3 %? |
|---|---|---|---|---|
| 壁面剪切（1-dx 割线） | ρνU/H = 40.0 Pa | 39.81 Pa（ratio 0.9952） | −0.48 % | **是** |
| 壁面剪切（内部最小二乘斜率） | 40.0 Pa | 38.36 Pa（ratio 0.9589） | −4.11 % | **否** |
| 流量 Q | UH/2 = 0.01 m²/s | 0.010442 | +4.42 % | 否（<5 % 门内） |
| 内部型线 L2 | 0 | 2.93 % | 2.93 % | **是** |
| 定常性 drift | 0 | 1.55 % | 1.55 % | 否（>1 % 门） |

结论：**壁面剪切相对误差 < 3 % 只在割线口径下成立（−0.5 %）；内部拟合口径为 −4.1 %，
不成立。** 两个口径都远优于已发布的 −32.6 %，因此发布值不能再被引用为精度结论。

## 5. 处置建议

* **request changes**：`docs/dev_report_couette_verification.md` 标题与 §4 机制归因改写；
  保留诚实 FAIL 记录（`t_end=0.15 s` 的确不过门），但注明那是瞬态。
* 正式台账建议：CASE-COUETTE-2D 的验收量改为"`t_end ≥ 2 H²/ν`、内部拟合与割线两口径
  同时给、`drift < 1 %`"；本轮 H/dx=16 满足剪切/型线/流量三项，定常性仍欠账。

## 6. 未做项（不得当作已通过）

* 未做代码级壁面 BC 审查（`rhs` 里 `wall_friction` 混合与壁粒子位置的关系仍按
  `verification_poiseuille_wall_shear_estimator.md` 的既有结论引用）。
* 未跑 H/dx = 24/32（单核 ~10 min/档，本机 load15 ≈ 120，容量纪律下只跑到 16）。
* 未跑 `U`/`ν`/`friction` 扫描（工件里 `speed_sweep`、`viscosity_sweep` 为空）。
* 未做时间步/核宽敏感性；只改了 `t_end` 与 `dx` 两个轴。

## 7. 复现

```bash
PYTHONPATH=src python benchmarks/couette/shear_reading_audit.py       # ~70 s, 两档
PYTHONPATH=src python -m pytest tests/test_couette_shear_reading.py -q
```

工件：`benchmarks/couette/out_full/{shear_reading.json,shear_reading.png,summary.md}`。