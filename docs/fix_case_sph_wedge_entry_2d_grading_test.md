# TensorSPH 缺陷修复：CASE-SPH-WEDGE-ENTRY-2D  grading 测试因 config 缺字段而失败

**Mission**: m16804 (自找活·tp-doc-3 data 席)
**锚点**: CASE-SPH-WEDGE-ENTRY-2D（算例池·二维楔形体入水砰击压力）
**日期**: 2026-11-12

## 问题描述

`tests/test_case_sph_wedge_entry_2d.py` 中 3 个 grading 测试失败：
- `test_grade_passes_on_exact_reference_field`
- `test_grade_fails_on_fifty_percent_bias`
- `test_grade_withholds_when_profile_missing`

失败原因：`_synthetic_case` 辅助函数构造的合成算例 config 字典缺少 `sound_speed_factor` 和 `gamma` 字段，而 `benchmarks/case_sph_wedge_entry_2d/run.py` 的 `grade()` 函数在第 362-367 行需要这两个字段来计算压缩性误差预算（compressibility error budget）：

```python
"c0_over_V": a["config"]["sound_speed_factor"],
"cp_density_budget": (2.0 * a["config"]["sound_speed_factor"] ** 2
                      / a["config"]["gamma"]
                      * (... ** a["config"]["gamma"] - 1.0)),
```

## 修复

在 `_synthetic_case` 的 config 字典中补全缺失字段：

```python
# 修复前
"config": {"beta_deg": beta_deg, "velocity": velocity, "dx": dx},

# 修复后
"config": {"beta_deg": beta_deg, "velocity": velocity, "dx": dx,
           "sound_speed_factor": 20.0, "gamma": 7.0},
```

字段值取自 `run.py` 的 `BASE` 字典（`sound_speed_factor=20.0`）和 `WedgeConfig` dataclass 默认值（`gamma=7.0`，Tait 状态方程标准指数）。

## 验证

```
$ PYTHONPATH=src python -m pytest tests/test_case_sph_wedge_entry_2d.py -v
======================== 13 passed, 1 warning in 5.03s =========================
```

修复前 3 个 grading 测试全部 FAIL，修复后 13/13 通过。

## 影响范围

- 仅影响测试夹具（synthetic case generator），不影响实际求解器或 benchmark 代码
- 修复后 grading 逻辑的正/负向控制测试均能正确执行
