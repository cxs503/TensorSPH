# m22831 — XVAL-SPH-METHOD-SELECTION-2026-11-12: m20114 反漂移工件再生

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（`feature_ledger` status=in_progress）
**任务类型**: 验证欠账修复（反漂移工件随注册表增长而失效）
**前序**: m20114（transport_velocity_correction 证据闭合）→ 后续任务陆续注册了第 30 条证据记录

## ① 案例介绍

`benchmarks/method_selection/plot_method_selection_m20114.py` 产出的四件工件（两张 PNG 图、
一份 JSON 摘要、一份 Markdown 表格）是**派生文档**：它们从 `docs/reports/method_selection_registry.json`
读取记录数、方法覆盖等统计量，并绘制证据覆盖图。每当后续任务向注册表注入新记录，这些工件的
计数和图摘要就会过时——而 `--check` 门正是用来捕获这种漂移的。

本单发现 `test_method_selection_m20114_antidrift.py` 的 4 条测试失败，原因是注册表从 29 条
记录增长到 30 条，但工件未同步更新。

## ② 怎么做的

1. **定位漂移**：`PYTHONPATH=src python benchmarks/method_selection/plot_method_selection_m20114.py --check`
   输出精确 diff：
   - `method_selection_m20114_tables.md`：`records: 29` → `records: 30`
   - `method_selection_m20114.json`：`records: 29` → `records: 30`，
     `figure_registry_coverage.png` 摘要从 `ef3efb4b...` 变为 `d4f214b0...`
2. **再生工件**：去掉 `--check` 重新运行脚本，四件工件全部重写。
3. **验证**：124 条 method_selection 相关测试全部通过（含 5 条 m20114 反漂移测试）。

## ③ 计算结果

### 注册表现状（30 条记录，10 个方法全覆盖）

| 方法 | 记录数 | 最佳误差 | 最差误差 | 是否全在 15% 内 |
|---|---|---|---|---|
| heat_conduction_operator | 3 | 0.0002% | 0.150% | ✅（1 条 refuted） |
| compressible_riemann | 2 | 0.414% | 0.777% | ✅ |
| isph_projection | 2 | 0.046% | 0.867% | ✅ |
| density_diffusion | 2 | 0.482% | 1.021% | ✅ |
| wcsph_delta_sph | 1 | 2.968% | 2.968% | ✅ |
| wcsph_wall_no_slip | 11 | 0.478% | 32.649% | ❌（多条 >15%） |
| csf_surface_tension | 4 | 5.599% | 45.693% | ❌（2 条 >15%） |
| compression_regularization | 1 | 5.533% | 5.533% | ✅ |
| particle_shifting | 1 | 8.833% | 8.833% | ✅ |
| transport_velocity_correction | 2 | 10.810% | 16.963% | ❌（1 条 >15%） |

### 再生工件清单

| 文件 | 变更 |
|---|---|
| `docs/reports/method_selection_m20114_tables.md` | `records: 29` → `30` |
| `docs/reports/method_selection_m20114.json` | `records: 29` → `30`，图摘要更新 |
| `docs/figures/method_selection_m20114/figure_registry_coverage.png` | 重绘（新增第 30 条记录的 bar） |
| `docs/figures/method_selection_m20114/figure_tvc_front.png` | 未变（TVC 数据未动） |

## ④ 结果图

再生后的 `figure_registry_coverage.png` 现在包含 10 个方法的完整覆盖条，
横轴为对数刻度的测量误差，红色虚线为 15% 业主标准线，红色 × 标记每个方法的最差记录。
`figure_tvc_front.png` 未变（TVC 数据未动）。

## ⑤ 注意点

- 这是**派生文档的例行维护**，不是算法/精度修复。
- 注册表增长是好事（更多证据 = 更可靠的决策），但反漂移门必须保持绿色。
- 未来若有新记录注入，同样流程：`plot_method_selection_m20114.py`（无 `--check`）→ 验证测试。

## ⑥ 误差判定

本单不涉及求解器计算，无误差判定项。反漂移门验证的是**工件与注册表的一致性**，
124 条测试全部通过即证明一致性已恢复。

## ⑦ 测试

```
PYTHONPATH=src python -m pytest tests/test_method_selection*.py tests/test_xval_method_selection*.py -q
```

124 passed（含 5 条 m20114 反漂移测试、6 条覆盖审计测试、12 条严格边界测试、26 条核心路由测试等）。

## ⑧ 给下次自找活的建议

1. **注册表第 30 条记录**：来自 `transport_velocity_correction` 的 dx=0.04 m 档（16.963%），
   超过 15% 业主标准。建议研究该档为何比 dx=0.03 m 档（10.810%）差——是分辨率不足还是
   TVC 参数敏感？
2. **`wcsph_wall_no_slip` 多条 >15%**：Couette 稳态包络在 H/dx=16 的移动壁工况达 17.076%，
   壁面剪切误差 32.649%（refuted 但仍 banked）。建议总账新增
   `XVAL-SPH-WALL-SHEAR-ACCURACY` 专项审计。
3. **`csf_surface_tension` 的 45.693%**：准静态 Laplace 案误差极大，需确认是否物理上
   不适用（准静态假设与 CSF 动态曲率计算的内在冲突）。
