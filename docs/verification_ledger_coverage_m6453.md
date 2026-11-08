# 证据覆盖的盲区：账本看不见的验证（m6453）

**范围**: TensorSPH ｜ **性质**: 验证欠账（证据覆盖修复 + 两个死 API）｜ **改动**: `benchmarks/verification_ledger/build_ledger.py`（覆盖规则）、`waivers.json`（欠账登记）、`benchmarks/contact_angle/run.py`（死 API）。全部数字由 `ledger.json` 注入（无手抄），`build_ledger.py --check` 校验漂移。

## 1. 盲区是什么

账本的任务是「仓库里每个机器可读验证工件都被登记、被定级」。它的覆盖规则一直按**文件名**匹配 `(validation_report|report|results|ladder|audit)\.json$`，并整段跳过 `out/`。两条后果都是静默的：

1. **换了名字就消失**：`benchmarks/density_diffusion_decay/decay_rate.json`（平面密度波衰减率 vs 解析 γ = D k²）是真实解析对照，只因文件名不叫 `*report.json` 就从未进账本；同类还有 `laplace_pressure_closure/verdict_matrix.json`。
2. **没有工件就更消失**：一个 benchmark 目录若一个 JSON 都不提交（`wave_tank`、`wave_flume`、`ice_splash`、`dam_break_3d`、`hydrostatic`、`capillary_wave`、`stability`、`convergence`、`scaling`、`contact_angle` 等），规则连一条 finding 都不会出——数字只活在报告里，**仓库内不可复算**。

## 2. 怎么修的（文件名 → 目录）

* `uncovered_benchmark`（新增，medium）：凡含 `run*.py` 的 `benchmarks/<dir>/` 必须有已登记 claim 落在该目录下，或一条**目录级豁免**（waiver 以 `/` 结尾）。HEAD 上一次性暴露 15 个「无账本、无工件」harness。
* `unregistered_artifact`（放宽）：直接位于 `benchmarks/<dir>/` 的任何 JSON（**不看名字**）以及 `out/` 下主报告（`results|report|summary|validation_report.json`）都必须登记或豁免；`cases/`、`raw/` 与 `out/` 其余文件按**片段**处理，继承父报告溯源。
* 豁免必须写理由（`test_every_waiver_has_a_reason`），目录级同规则，理由须点名欠账编号。
* 规则自身可测：`test_coverage_rule_is_directory_based` 用临时目录构造 alpha（怪名字工件）/beta（零工件）/gamma（claim + ladder 片段）/delta（目录豁免 + `out/` 片段），断言 finding 恰好落在前两者。

## 3. 登记进来的 5 个读数（此前不可见）

| claim | 基准 | 实测 | 相对误差 | 等级 | 工件 |
|---|---|---|---|---|---|
| density_diffusion.decay_rate.worst_of_ladder | 0.03 fraction | 0.010208 fraction | — | **PASS** | benchmarks/density_diffusion_decay/decay_rate.json |
| density_diffusion.finite_time.amplitude | 0.035 fraction | 0.00482 fraction | — | **PASS** | benchmarks/density_diffusion_decay/decay_rate.json |
| poiseuille.centre_velocity.finest | 1.22625 m/s | 1.24476 m/s | +1.51 % | **PASS** | benchmarks/poiseuille/out/results.json |
| poiseuille.wall_shear_ratio.finest | 1 - | 0.502508 - | -49.75 % | **WITHHELD** | benchmarks/poiseuille/out/results.json |
| effective_sigma.sigma_eff_over_sigma.dx5mm | 1 - | 0.710625 - | -28.94 % | **FAIL** | benchmarks/effective_sigma/out/report.json |

* **密度扩散**（Antuono, Colagrossi & Marrone 2012）：D = δ c0 h 不是拟合量（δ/c0/h 全是求解器输入），符号翻转、因子 2、dx/h 混淆都会远超 3% 门；登记前该算子只有守恒/平滑测试，这三类缺陷都能过。
* **Poiseuille 通道**：中心速度最细分辨率 +1.51%（PASS）。同一工件的壁面剪切只有解析的 0.503×，但**与自己的速度型内部矛盾**（速度拟合隐含 τ≈498 Pa、壁带读数 284 Pa），故按 WITHHELD 记账而非 FAIL：缺陷在壁带估计器，未见得在流动。
* **有效表面张力**：R = 40 mm、二次拟合曲率下 σ_eff/σ = 0.7106（-28.94%），与 R = 20 mm 的姊妹读数 0.7040（-29.60 %）差 1 pt 以内：~30% 的力亏损既不是半径效应也不是曲率算子效应。

账本规模：**36 claims / 18 物理量 / 8 findings**（本单前 31 / 14 / 8）；等级分布 `{"FAIL": 9, "MARGINAL": 5, "PASS": 7, "WITHDRAWN": 2, "WITHHELD": 13}`。

## 4. 顺带修好的：接触角 benchmark 的两个死 API

`benchmarks/contact_angle/run.py` 在 HEAD 上**无法执行**：

| 位置 | 缺陷 | 修法 |
|---|---|---|
| `synthetic_interface` | `math.asin(torch.linspace(...))` → `ValueError: only one element tensors can be converted to Python scalars` | `torch.asin(...)` |
| `contact_angle_from_envelope` | `torch.lstsq`（torch ≥ 1.9 已移除）→ `RuntimeError: removed` | `torch.linalg.lstsq(A, B).solution` |

修好后 harness 端到端跑通（`benchmarks/contact_angle/report.json` 已提交），但**测量层仍不可用**：仪器层 10 行读数全部 NaN（`all_within_2deg = false`），物理层 3 个设定角也全 NaN（内区 141 粒子，R/dx ≈ 6.7）。故接触角能力**仍不登记 claim**——诚实的 WITHHELD-by-absence，不把 NaN 抹成数字。`test_contact_angle_harness_runs_after_the_dead_api_fix` 钉住当前状态，仪器层修好时该断言转红、强制更新本文。

## 5. 注意点 / 局限

1. 本单**没有**修接触角仪器层（信封估计器在 141 粒子构型上凑不满拟合点），也没有让任何新能力「通过」：新增 5 个 claim 中 2 PASS、2 FAIL、1 WITHHELD。
2. 目录级豁免是**欠账登记**而非「已验收」：E1（harness 无工件）、E2（波浪能力报告只上传平台、被合并门禁滤除 → 仓库内不可复算）、E3（接触角测量层 NaN）。关账方式唯一：跑出工件 → 提交 → 换成 claim。
3. 覆盖规则仍不看 `tests/`：账本管工件，不管 pytest，这是有意的边界。
4. 机器 load15 ≈ 120，本单只做轻量工作；接触角 benchmark 是唯一重算例子（单次约 3 min、2 线程）。

## 6. 误差判定（逐项回答是否 < 3%）

| 关键量 | 基准 | 实测 | 误差 | 是否 < 3% |
|---|---|---|---|---|
| 密度扩散衰减率 γ（阶梯最差） | γ = D k², D = δ c0 h | 0.010208 | +1.02% | **是** |
| 密度波有限时间幅值 | ρ0 A e^(-γt) | 0.004820 | +0.48% | **是** |
| Poiseuille 中心速度（最细） | g H²/(8ν) = 1.22625 m/s | 1.24476 m/s | +1.51% | **是** |
| Poiseuille 壁面剪切比 | ρ g H/2 = 490.5 Pa | 0.5025 | -49.75% | 否（与自身速度型矛盾 → WITHHELD） |
| σ_eff/σ（R = 40 mm） | 1 | 0.7106 | -28.94% | 否（已知 CSF δ_s 归一化欠账） |
| 接触角测量 | 设定角 | NaN | 无数值 | 无法判定（E3 仪器层坏） |
