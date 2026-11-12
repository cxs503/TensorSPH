# CASE-TAYLOR-GREEN-SPH — 合并门禁滤除证据工件的修复（m16303）

**日期**: 2026-11-12 · **分支**: agent/m16303 · **锚点**: CASE-TAYLOR-GREEN-SPH（feature_ledger: open）

## 1. 缺陷（在 main 上可复现）

`tests/test_taylor_green_benchmark.py::test_committed_run_artifacts_reproduce_the_reference`
要求仓内存在 `runs/taylor_green/m15860/` 的 8 个运行工件（results.json / evidence.json /
environment.json / run.log / fig1–fig4 png），但 main 上该目录不存在：

- m15860 分支提交 `524c3ef`（"…real run artifacts and machine-checked evidence"）
  确实包含全部 8 个工件（`git show 524c3ef --stat`，18 files）；
- 合并进 main 的 `82ce5b1`（Merge #517 via merge-master）只带入 10 个文件，
  `runs/` 下 8 个工件被门禁滤除（与 "N 个文件被门禁滤除" 的合并模式一致）；
- 后果：任何新鲜检出（含 CI）该测试 `FileNotFoundError` 失败。
  本单开工前实测：`1 failed, 6 deselected`（4.85s）。

## 2. 修复

从 m15860 工作树（分支 agent/m15860 @ 524c3ef，`git status` 对 runs/ 干净）
逐字节复制 8 个工件回本仓 `runs/taylor_green/m15860/`，sha256 两侧一致：

| 工件 | sha256（前 16 位） |
|---|---|
| environment.json | 1a818ce47060d6cb |
| evidence.json | 4d32f0ec9410ddd6 |
| fig1_tgv_velocity_field.png | 38b1e7d5f5c2a602 |
| fig2_tgv_energy_decay.png | ce81adb954687737 |
| fig3_tgv_convergence.png | 39a3a6941829cf6f |
| fig4_tgv_sensitivity.png | ba85a9f338243e9c |
| results.json | c7620d62e34d69ac |
| run.log | a4283703fc19ae23 |

工件自洽性由测试机器校验（非人工断言）：reference 块与当前代码闭式解
`rel=1e-12` 一致、NS 残差 <1e-10、每行 ref/got/rel_error/tolerance/pass 齐全、
evidence 指向的文件存在且非空、convergence ≥3 点、reynolds_window ≥3 点、
负对照 ν=0 已记录。

## 3. 验证（本单实测）

```
$ PYTHONPATH=src python -m pytest tests/test_taylor_green_benchmark.py -m "not slow"
4 passed, 3 deselected in 4.06s
```

- 修复前：`test_committed_run_artifacts_reproduce_the_reference` FAILED（FileNotFoundError）
- 修复后：4 passed（reference 层 3 项 + committed-artifacts 层 1 项）
- 3 个 `@pytest.mark.slow` 测试（真实 SPH 重跑）本单未跑（容量纪律：load15≈110≥100，
  不起新重算力）；其结论已固化在恢复的 results.json 中（见下）。

恢复工件中的基准数字（results.json，零手抄）：

| 量 | ref（闭式解） | got（SPH） | 相对误差 | 容差 | pass |
|---|---|---|---|---|---|
| energy_decay_rate (1/s) | 2.513274 | 2.483651 | 1.18% | 8% | ✓ |
| energy_ratio_at_t_end | 0.221360 | 0.214864 | 2.93% | 8% | ✓ |
| amplitude_ratio_at_t_end | 0.470489 | 0.465750 | 1.01% | 8% | ✓ |

收敛研究（3 分辨率）误差 3.00% → 1.18% → 0.75%（单调下降）；
负对照 ν=0 测得虚假耗散率 0.1305 1/s（该 SPH 离散化耗散底，ν 下限锚）。

## 4. 同族缺陷（本单未修，留线索）

同一"门禁滤除 runs/ 工件"模式还命中（新鲜检出即红/错）：

| 测试 | 缺失工件 | 状态 |
|---|---|---|
| test_case_hydrostatic.py | runs/case_hydrostatic/report.json | 旧工件（m16005）与现行测试 schema 不符（4 KeyError），需**重新生成**而非恢复 |
| test_sloshing_resonance.py | runs/sloshing_resonance/summary.json | 源工作树未找到，需重跑 |
| test_verification_bundle.py | runs/sph_free_surface_verification/ | 源在 m14534 工作树，需核对一致性 |
| test_surface_tension_dispersion.py | runs/surface_tension_dispersion | 源在 m14536/m14849/m15328，需核对一致性 |
| test_ice_thermal_integration.py | runs/IN13W2-N07 | 测试自带 skip（非阻塞） |

建议：平台侧在合并门禁中放行 `runs/**` 证据工件（或规定证据工件必须随代码同批入库），
否则此类"证据链断裂"会持续复发。
