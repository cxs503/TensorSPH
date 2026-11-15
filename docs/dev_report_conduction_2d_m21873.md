# m21873 开发记录：传导算子的第二个基准族（2-D Fourier 模态衰减）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（自找活·验证欠账）｜**仓库**: TensorSPH
**平台报告**: id 19744 ｜ **产物**: `src/tensorsph/heat_conduction_2d.py`, `benchmarks/case_conduction_2d/{run.py,result.json,mode_decay.png}`,
`docs/case-conduction-2d.md`, `docs/gen_case_conduction_2d.py`, `tests/test_conduction_2d.py`；
登记表 + 脆弱性/严格上界/覆盖审计三份产物随之重算。

## 结论（一句话）

`heat_conduction_operator` 此前只有 1-D erfc 一个基准族（脆弱性审计列为单一来源）；
本单补上 2-D 周期方域 Fourier 模态衰减族，并暴露出一个 1-D 案例看不到的**量纲缺陷**：
原始 2-D 配对算子的有效扩散率是 `α_eff/α ≈ 1.926·dx`（O(dx)），把出厂的 1-D 标定因子
(1.0218) 搬到 2-D 会把运行错标 ~`1/dx`（≈835%）；在 2-D 里重做 1-D 已做的那一步标定后，
模态衰减误差 0.150%（n=128）、二阶收敛（2.01），族数 1→2。

## 交付

| 类型 | 路径 | 说明 |
|---|---|---|
| 能力增量 | `src/tensorsph/heat_conduction_2d.py` | 2-D 配对算子（向量距离、`torch.roll` 快路径、稠密对照）、2-D 格点符号与 `k→0` 标定、模态衰减求解器（RK4/Euler）、独立 Crank–Nicolson 参考 |
| 验证欠账 | `benchmarks/case_conduction_2d/` | CASE-SPH-CONDUCTION-2D：主跑 n=128 / RK4 / 2-D 标定 = 0.150%；`none`/`lattice_1d` 两个陷阱臂、`mode_2d` 本征关系臂、等距性表、误差预算、等价性检查 |
| 证据入库 | `benchmarks/method_selection/build_registry.py` | 新增 `x_conduction_2d`（banked，gate 3%）与 `x_conduction_2d_one_d_calibration`（**refuted**，前置条件写明量纲误用） |
| 缺陷记录 | 同上 + `docs/case-conduction-2d.md` | 「1-D 标定不可迁移」+「原始算子 2-D 维度不一致」两条，含量化证据 |
| 防漂移 | `docs/gen_case_conduction_2d.py` | 文档表格/关键数字由 artefact 重算比对，`--check` 失败即漂移 |
| 测试 | `tests/test_conduction_2d.py` | 9 项：快/稠密等价(<1e-12)、退化 1-D 与出厂 `conduction_rhs` **恰好相等**、本征关系、`mode_2d` 精确性、1-D 标定失败(>100%)、二阶收敛+gate、artefact gate、登记表族数≥2 且数值与 artefact 一致、FD 参考收敛 |

## 验证

- `PYTHONPATH=src python3 -m pytest tests/test_conduction_2d.py -q` → **9 passed**
- `PYTHONPATH=src python3 -m pytest tests/test_method_selection.py tests/test_method_selection_fragility.py -q` → **42 passed**
- `PYTHONPATH=src python3 benchmarks/method_selection/build_registry.py`（重算：`heat_conduction_operator` 记录 3 条 = 1-D banked + 2-D banked + 2-D 误用 refuted）
- `PYTHONPATH=src python3 benchmarks/method_selection/audit_decision_fragility.py` → `depth mismatches: 0, invariants_ok: True`
- `PYTHONPATH=src python3 benchmarks/method_selection/strict_bound_profiles.py`、`audit_quantity_coverage.py` → 重算（`1-D transient conduction` 上界由单一 1-D 记录变为 0.1504% 双族上界）
- `PYTHONPATH=src python3 docs/gen_case_conduction_2d.py --check` → docs 与 artefact 一致

## 误差判定（vs 基准）

| 量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 2-D 模态场（n=128，2-D 标定，RK4） | 闭式 `exp(-α(kx²+ky²)t)cos cos` | 0.150% | 是 |
| 衰减率 | 同式 | 0.150% | 是 |
| 2-D 原始算子扩散率 | 连续 `α` | ≈1.93dx（O(dx)，不一致） | 否（诚实 FAIL，作为缺陷入账） |
| 1-D 标定搬 2-D | 同式 | ≈835% | 否（refuted，误用） |
| 独立 FD(Crank–Nicolson) vs 闭式 | 闭式 | 0.046%（n=128） | 是 |

## 注意点 / 局限

- 本 artefact 的阶梯只到 n=32/64（主跑 n=128）；更深的阶梯已递交集群（cj3762）但排队未回，为守 10min 本机纪律与提交时点而取消，缺口在表里标 `—` 而非猜值。
- 只做均匀导热率（`k_i=k_j`）、周期域：多材料调和平均与 2-D 壁面/鬼点未验，属未覆盖项。
- 用 Euler 时**时间离散误差主导**且与空间误差反号（n=32 时 1.8% 对 0.64%），故入库读数用 RK4；两臂都留在 artefact 里。

## 留给下次

单一来源从 9/10 降到 8/10（当前单一来源：compressible_riemann, compression_regularization, csf_surface_tension, density_diffusion, isph_projection, particle_shifting, transport_velocity_correction, wcsph_delta_sph）。次优项仍是 `compressible_riemann`（Toro 3/4：压力平台/接触间断）、
`compression_regularization`、`particle_shifting`（静水柱/湖静）各补第二族；
另：2-D 算子若要在不规则分布上使用，需先补一致性修正（本单只量化了格点符号）。
