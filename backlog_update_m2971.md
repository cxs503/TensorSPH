# m2971：Laplace 压差失败归因（验证欠账结清）＋ 新 P0/P1/P2

**日期**: 2026-11-03 ｜ **范围**: TensorSPH ｜ **改动类型**: 只增文件（benchmark/docs/tests），未改 `src/`

## 已结清：为什么连续三轮都测不到 Young–Laplace 压差

工件：`benchmarks/laplace_pressure_floor/`（`run.py` / `report.json` / `laplace_pressure_floor.png`）、
`docs/verification_laplace_pressure_floor.md`（数字由 `make_doc.py` 注入）、
`tests/test_laplace_pressure_floor.py`（8 passed + 1 strict-xfail）。

同一物理量四种口径 + c0 阶梯 + 分辨率阶梯，把 6 个算例的误差拆开：

| 误差源 | 实测 | 结论 |
|---|---|---|
| C1 测量口径（`p_core − p(0.80–0.95R)`） | −44…−55 个百分点 | 前三轮「−76%…−107%」的一半以上是口径造成的（上带落在自由面密度亏缺区） |
| C2 初始化核和偏置 `rho/rho0−1=+0.00588` | 稳态仅 ±9 pt，瞬态 2395 Pa（c0=20，为信号 6.7×） | 真缺陷但**不是**稳态主因，已降级为 P1 |
| C3 自由面压强闭合（Tait 钳 p≥0 + 亏缺 −6%…−34%） | 压缩比 **0.57±0.06**，对 c0/dx 不敏感；残差 −34%…−38% | **唯一门槛项** |
| C4 EOS 分辨率地板 | c0 20→5 时预测噪声 28%→2%，实测残差 38%→36%，**不跟随** | 被证伪（不是噪声地板问题） |

口径统一到「绝对压强 + 时间平均」后，最优读数 237.6 Pa vs 解析 360 Pa（**−34.0%**）：
Laplace 压差仍 **WITHHELD**，但欠账从「不知道错在哪」变成「只差一个自由面压强边界」。

## 新 P0：自由面压强闭合（唯一门槛项）

`rhs` 的压强装配处加 shell-pressure 外推（Adami/Ihmsen/Adams 2012 §3.3）：
自由面粒子（`rho/rho0<0.99` 或 `c<0.99`）用 `p_i = max(0, Σ(p_j + ρ_j g·r_ji)W_ij/ΣW_ij)`。
**验收判据（机器可裁）**：`benchmarks/laplace_pressure_floor/run.py` 的
`absolute_time_average_tail` 在 c0=20/5 × dx=0.002/0.0015 四点上全部 <15%，且
`tests/test_laplace_pressure_floor.py::test_laplace_jump_within_15pct` 由 xfail 转 xpass。

## P1：初始化质量校准（C2）

`make_particles`/密度初始化按核心区（r<0.5R）实测核和做一次 `m → m/(1+b)`。
影响面**不限表面张力**：任何 wcSPH 算例在 c0=20 时都带 2.4 kPa 初始虚假压强
（可与液滴撞池的 ρU² 相比）。验收：初始 `p_max/(ρ0 c0²) < 1e-4`。

## P2：验证口径统一 + 陈旧「钉死失败」测试

1. 后续所有表面张力/Laplace 验证一律「绝对压强 + 时间平均」，禁用 `p_core − p_shell`；
   已提交的 `docs/verification_surface_tension_{laplace,consistent,curvature}.md` 中
   −75.6%/−87.6% 等数字应标注为**历史口径值**，不得再作为当前精度引用。
2. `tests/test_laplace_droplet.py` 有 4 个**陈旧钉死失败断言**在 HEAD 上就红
   （`test_current_model_fails_laplace_criterion` 期望 within_15pct=False 但现为 True；
   `test_curvature_has_the_wrong_sign_on_the_interface` 期望 kappa<0 但现为 +15.57；
   `test_curvature_error_grows_under_refinement` 的比例判据不再成立）。
   这些断言编码的是 CSF 重构**之前**的破损事实，应由后续单改写为「记录当前状态 + 新判据」，
   不要直接删除（它们是历史的诚实留痕）。

## 环境踩坑

`/tmp/probe.py` 类临时脚本会被同机其它 agent 覆盖（本单实际踩到）；临时件放工作树或加 mission id 前缀。