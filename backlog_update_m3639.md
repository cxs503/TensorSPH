# m3639：初始密度偏置结清（m2971 P1/C2）＋ 新 P1

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ **改动类型**: 新增 `src/tensorsph/initial_conditions.py`
+ `Config.lattice_mass_calibration`（默认 `False`）+ benchmark/docs/tests

## 已结清：m2971 P1（C2 初始化质量偏置）

m2971 把「初始化核和偏置 `rho/rho0−1 = +0.00588`」列为 P1，但没给出它到底是
分辨率误差还是常数。m3639 用纯 `t=0` 测量（不积分时间）把它钉死：

| 结论 | 实测 |
|---|---|
| 偏置是**格点常数**，非离散误差 | 六角格 h=1.3dx：`b = +0.5884%`，dx=0.004/0.002/0.001 **完全相同** |
| 归一化判据与 c0 无关 | c0=5/10/20/40 → `max\|p\|/(ρ0c0²) = 5.988e-3` **恒定**；绝压 ∝ c0²（149.7→9581.4 Pa） |
| 在验证几何（R=0.02, σ=7.2）上的量级 | 2395 Pa = **6.65×** 的 σ/R=360 Pa 信号（backlog 里写的 6.7× 得到确认） |
| 方格 h=1.3dx 是唯一已平衡的格型 | `b = −0.0053%` → 生产算例（`make_particles` 用方格）从来没暴露此缺陷 |
| 校准后 | 内部 `\|ρ/ρ0−1\| < 6e-14`，`max\|p\|/(ρ0c0²)` 全 sweep < 1e-4（判据通过） |
| 代价 | 总质量 −0.585%（保密度不保质量，已在文档写明） |

**口径纠正**：校准必须在「邻居数 = 配置最大值」的**严格内核粒子**上拟合；
容差放宽到 max−2 会把不同支撑级别的粒子混进均值，使严格内核偏 `+7e-4`（实测）。
`supported_mask` 默认 `tolerance=0` 就是为此。

**归因边界（重要，防误伤 P0）**：把校准代入 `benchmarks/laplace_droplet` 后，
力积分口径的 `dp_ratio` 只动 `−0.33` 个百分点（dx=0.01：0.7942 → 0.7911），
`N_delta` 完全不变。→ **C2 只污染压强读数，不是 CSF 力积分 16–21% 缺口的原因**；
该缺口仍归 P0 自由面压强闭合。这条边界避免后续把 C2 当成 P0 的替罪羊。

工件：`src/tensorsph/initial_conditions.py`（`supported_mask` / `measure_density_bias` /
`calibrate_lattice_mass` / `spurious_pressure`，一次标定即精确——和密度对质量线性）、
`benchmarks/init_conditioning/{run.py,report.json,make_doc.py,init_conditioning.png}`、
`docs/verification_initial_pressure_offset.md`（数字由 make_doc.py 从 JSON 注入）、
`tests/test_init_conditioning.py`（8 passed）。

## 新 P1：`Config.lattice_mass_calibration` 默认值的决策（需 owner/复核）

flag 默认 `False`：打开会让所有已发表数字按 `−b/(1+b)` 平移，且生产几何（方格）
偏置只有 −0.005%，收益接近零。真正的收益方是**六角格验证流程**
（`benchmarks/laplace_droplet`、`laplace_law`、`laplace_pressure_floor` 的六角液滴）。
建议：把这些验证 harness 的初始化改成 `calibrate_lattice_mass`，**并同时重测所有
钉死数字**——这是一次 re-baseline，不是 bug fix，需单独一单，验收=
「同一算例 calibrated / uncalibrated 两套数字都在文档里，且注明哪套是当前口径」。

## 仍未动（排期不变）

- P0 自由面压强闭合（唯一门槛项，m2971 已定位）：`laplace_pressure_floor` 四点 <15%。
- 多相 SPH、3D 扩展、dam-break 收敛复现、dx=0.006 密度压缩 0.17 稳定性。

## 测试门禁

`tests/` 全量：**342 passed, 8 xfailed, 2 failed**。两个 failed 均为
`XPASS(strict)`（`test_oscillating_droplet.py` 的 mode-2 架构脆弱钉，pin 自带说明
「x86_64 CI +8.64% vs loong64 −31.78%，同一份代码」）——本机落在通过侧而翻转的
**既存**架构相关欠账，与本单改动无共享代码路径（flag 默认关闭）。留作下一单线索。