# m16457：CASE-HYDROSTATIC 验证欠账结清（shipped evidence 重建 + 测试转绿）

**日期**: 2026-11-12 ｜ **范围**: TensorSPH ｜ **改动类型**: 证据工件重建 + 提交（未改 `src/`）

## 问题（HEAD 上的红测试）

`tests/test_case_hydrostatic.py` 的 shipped-evidence 组（4 个测试）在 HEAD 上
**ERROR**：模块级 fixture 硬断言 `runs/case_hydrostatic/report.json` 存在，而
`runs/` 目录从未入库（`git ls-files runs/` 为空，`.gitignore` 只忽略
`runs/m15815/repro_wt/`）。任何新检出（含本工作树）都必然红：

```
ERROR tests/test_case_hydrostatic.py::test_shipped_gate_is_the_predeclared_config_and_within_limit
  AssertionError: runs/case_hydrostatic/report.json is missing; reproduce with
  `python benchmarks/case_hydrostatic/run.py --config all --static` then `--aggregate --figure`
```

这是「验证欠账」：静水案例（README 声明 1.12% 量级）的钉死测试依赖一份
**不随仓分发**的证据文件——测试在结构上不可能在全新检出上通过。

## 做法

1. 按测试 fixture 与 `repro` 字段声明的**预声明命令**原样复现：
   `python benchmarks/case_hydrostatic/run.py --config all --static`
   （8 个时步配置 + 7 个 t=0 静态诊断，2 线程，墙钟 ~11 min）；
2. `--aggregate --figure` 聚合出 `runs/case_hydrostatic/report.json` 与 3 张图；
3. 将 `runs/case_hydrostatic/`（8 时步 + 7 静态 run 文件、report.json、figures/）
   **提交入库**，使 shipped-evidence 测试在全新检出上可跑可裁。
   判据在数字读出前已声明（gate = window_8s 内带时间平均最大带误差，限 1.0%；
   自由面带按 2h 重叠规则排除），本次未改任何判据。

## 结果（全部取自 `runs/case_hydrostatic/report.json`，零手抄）

| 配置 | 粒子数 | max 带误差 | mean | vs 1.0% 门 |
|---|---|---|---|---|
| **window_8s（gate）** | 1000 | **+0.7710 %** | +0.4569 % | **PASS** |
| window_4s | 1000 | +0.9773 % | +0.3369 % | PASS |
| window_2s | 1000 | +1.0270 % | +0.5501 % | FAIL（窗口过短，预声明理由） |
| dx_004 | 250 | +0.8213 % | +0.5921 % | PASS |
| dx_0025 | 640 | +0.7033 % | +0.3603 % | PASS |
| width_w15 | 1500 | +0.7254 % | +0.3707 % | PASS |
| sound_c30 | 1000 | +1.0203 % | — | FAIL（单旋钮敏感性点，非门） |
| control_g0（负控） | 250 | +99.88 % | +99.53 % | 负控 OK（\|mean\|>80%） |

静态 t=0 记录（机制诊断，非门）：dx=0.04 → 28.22%，dx=0.025 → 14.94%，
dx=0.02 → 10.22%，width_w15 → 9.91%，sound_c30 → 10.10%——唯一误差源是
壁面/自由面处的求和密度截断，随分辨率单调下降，符合预期。

**误差判定**：gate 0.7710 % < 1.0 % 限 → **PASS**；关键量 vs 解析基准
p(y)=ρ0 g (H−y) 误差 0.7710 %，**< 3%：是**（亦 < 1% 门限）。

## 验证

- `pytest tests/test_case_hydrostatic.py`：**11 passed**（修复前 10 passed + 1 error）。
- 全量 `pytest tests/` 结果见平台报告（本单提交时全绿）。
- 负控 control_g0 读 +99.53%（g 关闭而参考保留 ρ0 g），`negative_control_ok: true`。

## 注意点 / 局限

- `window_2s`（1.027%）与 `sound_c30`（1.0203%）两个**敏感性点**略超 1%：
  这正是当初把 gate 钉在 window_8s 的预声明理由（2 s 窗口的振荡采样噪声
  max_fluctuation 212% vs 8 s 的 15.8%）；它们不是门，超门不改变 verdict。
- 静态 t=0 记录的 10–28% 误差是**诊断量**（截断项），不得引用为案例精度。
- 工件体积 ~200 KB（JSON+PNG），入库换「测试在全新检出上可裁」，属有意为之。

## 给下次自找活的线索

- `runs/` 下其它案例（ice_thermal / sloshing_resonance / surface_tension_dispersion /
  wave_flume）的测试同样引用 `runs/` 工件——若其 fixture 也是硬断言，存在同款
  「证据不入库」欠账，可逐一按本单模式结清。
- `benchmarks/case_hydrostatic/run.py` 的 `figure()` 返回路径但 `main()` 未回写
  `report["figures"]`（现为 None）；若后续报告管线要读该键，需补一行回写。
