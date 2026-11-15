# m21797 开发记录：方法选择推荐的证据深度与逐条抽出脆弱性审计

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（自找活·验证欠账）｜**仓库**: TensorSPH
**平台报告**: id 19674 ｜ **审计产物**: `docs/xval_method_selection_fragility_m21797.md`,
`docs/reports/method_selection_fragility_m21797.json`,
`docs/figures/method_selection_fragility/evidence_depth.png`

## 结论（一句话）

选择器的精度上界此前只被问过「数字对不对」和「是否覆盖最差记录」；本轮问「站在几条独立记录上」：
10 个有证据格式中 **9 个的上界只来自单一基准族**，唯一跨族复现的是 `wcsph_wall_no_slip`。

## 交付

| 类型 | 路径 | 说明 |
|---|---|---|
| 能力增量 | `src/tensorsph/method_selection.py` | `case_family()`、`Decision.evidence_depth` / `is_single_source`、`Selection.single_sourced` / `weakest_link`、卡片新增 `## evidence depth` 段与单一来源 caveat |
| 缺陷修复 | `src/tensorsph/method_selection.py` | `EXCLUSIVE_RULES={shock, capillary}`：regime 触发而候选全不可用时不再向更宽泛规则降级 |
| 审计驱动 | `benchmarks/method_selection/audit_decision_fragility.py` | 独立复算证据深度 + 逐条抽出（130 次选择）+ regime 锁定探测 + 不变量；`--check` 防漂移 |
| 缺陷修复 | `benchmarks/method_selection/strict_bound_profiles.py` | m21537 报告把已是百分数的值又乘 100（23.80 % 印成 2379.93 %），表格与 PNG 一并修正 |
| 测试 | `tests/test_method_selection_fragility.py` | 16 项：深度语义、族归并、不变量、regime 锁定回归、单位回归、产物防漂移 |

## 验证

- `PYTHONPATH=src python -m pytest tests/test_method_selection_fragility.py` → 16 passed
- 方法选择相关全集：`134 passed, 1 skipped`
- `benchmarks/method_selection/audit_decision_fragility.py --check` → OK（产物为最新）
- `benchmarks/method_selection/strict_bound_profiles.py --check` / `build_registry.py --check` → up to date

## 留给下次

`compressible_riemann`、`heat_conduction_operator`、`compression_regularization`、`particle_shifting`
各只有 1 条记录，且在严格模式下**直接承重**（删掉即 `WITHHELD`）。次优项：给这四个格式在第二个基准族
上补一条独立记录（例：`heat_conduction_operator` 补 2-D 稳态导热 vs 闭式线性剖面；
`compressible_riemann` 补 Toro 测试 3/4 的压力平台/接触间断），可把单一来源从 9/10 降到 5/10。
