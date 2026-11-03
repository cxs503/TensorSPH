# 恢复可导入性后暴露的既有失败清单（m2451 审计）

**日期**: 2026-11-03｜**范围**: TensorSPH｜**类型**: 只读审计 + 本单产物说明

## 1. 触发事实

`bec5848 (m2090)` 合入了接触角边界条件的**测试、文档、Config 字段**，但实现文件被合并门禁滤除
（commit message 自述「1 个文件被门禁滤除」），而 `src/tensorsph/__init__.py`（新增
`apply_contact_angle_boundary` 导入）与 `src/tensorsph/droplet_impact.py::rhs`
（调用 `compute_surface_tension_force(..., wall=wall)`）都保留了对它的依赖。结果：

- `import tensorsph` → `ImportError: cannot import name 'apply_contact_angle_boundary'`
- `pytest tests/` → **0 tests collected**（24 个测试文件全部 collection error）
- `rhs()` 走表面张力分支时 → `TypeError: unexpected keyword argument 'wall'`

即：从 2026-11-02 20:09 起，TensorSPH 在 main 上是**不可导入、不可测试**的状态。
历史上至少有 6 个分支（m2389/m2399/m2406/m2408/m2420 等）尝试过同一修复，均未合入 main
（`git log --all --oneline -S "def apply_contact_angle_boundary"`）。

本单按 `docs/contact_angle_boundary.md` 的规范补回算子（`src/tensorsph/surface_tension.py`），
包恢复可导入，套件恢复可收集。

## 2. 恢复后暴露的既有失败（12 项）

复现命令：`PYTHONPATH=src python -m pytest tests/ -q`

| # | 测试 | 根因分类 | 证据 |
|---|------|----------|------|
| 1 | `test_contact_angle.py::…::test_surface_tension_with_contact_angle` | 测试构造缺陷（**本单已修**） | 原 fixture 是 6 粒子簇，邻居共享同一坐标 → `div(n)` 恒为 0，任何接触角下 CSF 曲率与力都恒等于 0，断言不可满足 |
| 2 | `test_contact_angle.py::…::test_surface_tension_without_contact_angle` | 同上（**本单已修**） | 同上；改为「有壁 vs 无壁 bit-exact」的向后兼容断言 |
| 3-6 | `test_laplace_droplet.py::test_current_model_fails_laplace_criterion` / `…_curvature_error_grows_under_refinement` / `…_curvature_has_the_wrong_sign…` / `…_normal_field_is_not_radially_coherent` | 孤测试（pinned 值从未在 main 上执行过） | 这些测试由 m2229 写在分支上并 pin 了当时测得的具体数值（如 `dp_ratio≈0.43/1.39`）；实现文件被滤除/未合并，测试从未在 main 跑过，pin 与 main 现状不符 |
| 7 | `test_laplace_law.py::test_csf_force_is_net_inward` | 同上 | pin `0.3 < ratio < 1.8`，main 现状落在区间外 |
| 8 | `test_laplace_law.py::test_csf_force_direction_is_radial` | 同上（strict xfail 转 XPASS） | 该判据在 main 现状下**已通过**，`xfail(strict=True)` 因此报错——按设计这是在强制刷新验证记录 |
| 9 | `test_laplace_pressure.py::test_effective_surface_tension_at_the_documented_resolution` | 同上 | pin `ratio≈0.704`，实测 `0.2147` |
| 10-11 | `test_lattice_wall_clearance.py::test_lattice_never_overfills_the_tank` / `…_initial_density_error_at_fine_spacing` | 实现/测试口径不一致（真 bug 候选） | `make_particles` 在默认 `dx=0.025` 下给出 `y_max=0.565`，而 `Config.pool_depth=0.35`；粒子集合包含液滴本体而非仅水池格架 |
| 12 | `test_ritter.py::test_solver_reproduces_the_ritter_front_in_the_shallow_limit` | 真回归候选 | 波前速度 `1.229 m/s` vs 解析 `2·c0 = 1.981 m/s ±25%` |

**分类判读**：#3–#9 属于「孤测试」——测试与其实现在不同分支、实现未合入 main，pin 值不可能在
main 上成立。修法不是改数，而是**重跑对应 benchmark 得到 main 现状的测量值并连同结论一起更新验证记录**
（`docs/verification_surface_tension_laplace.md` 等），否则等于把回归掩盖成通过。
#10–#12 需要独立判定是几何/初始化 bug 还是测试口径问题，不属本单范围。

## 3. 本单未做的事（诚实边界）

- 未改动 CSF 物理模型（曲率仍为 `div(n)` 近似，`laplace_law` 的曲率/切割面力/伪流判据仍未通过）；
- 未修 #3–#12；
- 未做液滴铺展的时程验证（需时程积分，本机 load15 ≥ 100 按容量纪律未启动重算力）。

## 4. 给下一次自找活的线索（按价值排序）

1. **孤测试清算（P0）**：12 项中 #3–#9 可用「重跑 benchmark → 刷新 pin + 更新 docs」一次做完，
   需要 `benchmarks/laplace_law/run.py`（instant 模式秒级）与 `benchmarks/laplace_droplet/run.py`。
2. **CSF 曲率估计器替换（P0，物理）**：`div(n)` 在圆上误差 +322%…+1148% 且随加密反向；
   候选方案是色函数覆盖率场 + 局部二次 MLS 拟合（`κ = tr(H)/|g| − gᵀHg/|g|³`）。
   注意这会让 #3–#9 的 pin 全部失效，必须与上一条同单处理。
3. **测试门禁加「包可导入」冒烟**：`python -c "import tensorsph"` 应作为合并门禁的第一步，
   本次这类「导入即死」的状态本可被一条 0.2 秒的断言拦下。