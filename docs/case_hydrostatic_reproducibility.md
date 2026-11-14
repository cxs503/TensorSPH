# CASE-HYDROSTATIC：证据复现闸（把「永远 skip 的复现检查」变成常开检查）

**日期**: 2026-11-14 ｜ **域**: 船舶·静水力（SPH）｜ **仓**: TensorSPH ｜ **锚点**: `CASE-HYDROSTATIC`（算例池·静水压剖面，ref: 解析，容差 <1%）

## 1. 案例与判据（不变，未重设）

矩形容器静水，解析基准 `p(y) = rho0 g (H − y)`（`rho0 = 1000 kg/m³`，`g = 9.81 m/s²`，
`H = 0.4 m`；与 Adami, Hu & Adams, JCP 231 (2012) 7057 的静水槽基准同式）。

* **门（gate）**：`window_8s` 配置（dx = 0.02 m、宽 1.0 m、8 s、1000 粒子）带时间平均的
  `max interior-band |p_SPH − p_ref| / p_ref`，限 **1.0 %**（自由面带按 2h 重叠规则排除）。
* 冻结证据（shipped）在 `tests/data/case_hydrostatic/`：`report.json` + 15 个 `runs/*.json`
  + `MANIFEST.sha256` + `channel.json`。
* 本单复核读数：gate = **0.7710424635550619 %** → `gate_pass: true`；负控 `control_g0` = 99.527 %
  （`negative_control_ok: true`）。

## 2. 发现的欠账（本单要修的东西）

`tests/test_case_hydrostatic.py` 的复现性检查
（`test_shipped_evidence_agrees_with_a_regenerated_bundle`）只有在本检出存在
`runs/case_hydrostatic/` 时才执行，而 `runs/` 不在交付白名单内（`docs/ src/ tests/ examples/ outputs/`），
**全新检出上必然 skip**：

```
SKIPPED [1] tests/test_case_hydrostatic.py:361: no regenerated runs/ bundle in this checkout
```

即：证据「可判读」（gate 值可复算、manifest 可校验）已经成立，但「可复现」在 CI 上
**从来不执行**——一份永远不跑的检查等于没有检查。全量重跑一遍声明网格要 ~11 min
（`run.py --config all --static`），不适合放进常规测试。

## 3. 做法（最小、可验证、无常量手抄）

关键观察：案例的**确定性层**（`run_initial_state`，t = 0 一次压力求值，无时间积分）
每配置 ~0.3 s，7 个配置合计 ~2 s，且**完全可复现**。于是

1. **新增工具** `benchmarks/case_hydrostatic/reproduce_static.py`：把每个已入库的
   `runs/static_*.json` 的旋钮（dx / 箱宽 / 深度 / 声速 / 带数，从记录自身读，不手抄）
   重新喂给 `run_initial_state`，逐带比对 `y_m / ref_Pa / got_Pa / rel_error_pct` 与
   gate 记录（`max_band_error_pct / mean_band_error_pct / worst_band / gated_bands /
   free_surface_bands`），写 `docs/case_hydrostatic_static_repro.json`，不一致则非零退出。
2. **新增常开测试**（`tests/test_case_hydrostatic.py`，全部走 `tests/data` 冻结证据，无需重跑 gate）：
   * `test_shipped_static_layer_reproduces_on_a_fresh_checkout` —— 7/7 逐带复现；
   * `test_static_truncation_ladder_is_monotone_and_wall_owned` —— 截断误差随 dx 细化
     **严格单调下降**、最差带恒为底部壁面带（worst band = 0）、且粗网格静态值必 > 1 %
     （否则「静态层是诊断量、不是验收量」这句话就会被悄悄写反）；
   * `test_repro_check_detects_a_tampered_static_record` —— 把某带 `got_Pa` 改 0.1 %
     必须被工具判为「未复现」（保证检查非空转）。
3. 原 11 min 全网格复现测试保留（有 `runs/` 时仍然跑），skip 文案改为指向新的常开层。

## 4. 结果（程序化生成，零手抄，取自 `docs/case_hydrostatic_static_repro.json`）

| 记录 | dx (m) | 粒子数 | shipped max 带误差 | 复算 max 带误差 | worst band | 复现 |
|---|---|---|---|---|---|---|
| `static_dx_0025.json` | 0.025 | 640 | 14.93777 % | 14.93777 % | 0 | ✅ |
| `static_dx_004.json` | 0.040 | 250 | 28.22267 % | 28.22267 % | 0 | ✅ |
| `static_sound_c30.json` | 0.020 | 1000 | 10.10480 % | 10.10480 % | 0 | ✅ |
| `static_width_w15.json` | 0.020 | 1500 | 9.91094 % | 9.91094 % | 0 | ✅ |
| `static_window_2s.json` | 0.020 | 1000 | 10.21605 % | 10.21605 % | 0 | ✅ |
| `static_window_4s.json` | 0.020 | 1000 | 10.21605 % | 10.21605 % | 0 | ✅ |
| `static_window_8s.json` | 0.020 | 1000 | 10.21605 % | 10.21605 % | 0 | ✅ |

**7/7 复现**（`all_reproduced: true`，`rel_tolerance = 1e-9`）。

分辨率阶梯（同一箱、c = 20 m/s，最差带恒为底部壁面带）：

| dx (m) | 粒子数 | t=0 max 带误差 | worst band | worst y (m) |
|---|---|---|---|---|
| 0.020 | 1000 | 10.2161 % | 0 | 0.0300 |
| 0.025 | 640 | 14.9378 % | 0 | 0.0250 |
| 0.040 | 250 | 28.2227 % | 0 | 0.0200 |

图：`docs/figures/case_hydrostatic_gate_profile.png`（gate 配置的压力剖面 vs 解析 + 逐带误差与 ±1 % 门）、
`docs/figures/case_hydrostatic_static_repro.png`（shipped vs 复算一一对应 + 截断阶梯）。

## 5. 检查真的能失败（非空转证据）

变异测试：把 `src/tensorsph/case_hydrostatic.py` 的 `H_FACTOR` 由 1.3 改为 1.32（一个真实的
核宽改动），重跑 `pytest -k static`：

```
FAILED tests/test_case_hydrostatic.py::test_shipped_static_layer_reproduces_on_a_fresh_checkout
FAILED tests/test_case_hydrostatic.py::test_repro_check_detects_a_tampered_static_record
```

随即还原（`git diff src/` 为空）。说明该层对求解器改动敏感，不是只读断言的自证。

## 6. 误差判定（逐项，显式回答是否 <3 %）

| 关键量 | 基准 | 本单读数 | 误差 | <3 % | <1 %（锚点容差） |
|---|---|---|---|---|---|
| gate 带误差 max（`window_8s`，动态） | 解析 `p = rho0 g (H−y)` | 0.7710 % | 0.7710 % | **是** | **是**（gate_pass） |
| t = 0 静态层（诊断量，dx = 0.02） | 同上 | 10.2161 % | 10.2161 % | 否 | 否（**诊断量，非验收量**） |

静态层误差唯一来源是壁面/自由面处求和密度截断：壁面/自由面带被门排除后即得 0.771 %。
静态层数值**不得**引用为案例精度；引用案例精度只认 gate 值。

## 7. 注意点 / 局限

* 本单**没有**重跑 11 min 的动态 gate 网格：shipped 的 gate 值来自既有 `window_8s` 运行，
  本单只验证「冻结证据自洽 + 确定性层可从源码复现」。动态层的重跑复现仍需
  `run.py --config all --static`（昂贵，不进常规 CI）。
* 浮点容差取 `rel = 1e-9`：shipped 与复算同代码同机器类型，唯一合法差异是密度求和顺序
  （架构/BLAS），远低于该量级；开发真正想改的东西（第 6 位数字的带值、带划分、阶梯）都会被抓住。
  跨架构（如 loong64）若出现超出该容差的漂移，说明求和顺序影响已达 1e-9 量级，需要重定容差并记录，
  不得直接放宽。
* `run_initial_state` 现在多返回 `geometry` / `reference.H_lattice` 两个键（旧记录没有），
  比对只比两边共有的键与每个带行，不因此报漂移。
* 同类「artifact 缺失即 skip」的测试在本仓还有十余处（`tests/test_wave_flume_case_artifacts.py`、
  `test_xval_*.py`、`test_kleefsman_c0_ladder.py` 等）——若其中有确定性层，可套本单模式。