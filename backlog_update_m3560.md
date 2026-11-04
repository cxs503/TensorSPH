# backlog 更新（m3560：Laplace P0 结清 + 新 P1/P2）

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ **改动**: 新增 `src/tensorsph/fs_pressure.py`，
`src/tensorsph/droplet_impact.py`（`Config.free_surface_pressure`/`rhs` 装配，**默认仍 clamp**），
`benchmarks/fs_pressure/`，`tests/test_fs_pressure.py`，`docs/verification_fs_pressure.md`

## 已结清：m2971 的 P0「自由面压强闭合」

`rhs` 现在可对密度亏缺（`ρ/ρ0 < 0.99`）的**流体**粒子做核加权压强外推
（Adami, Hu & Adams 2012 §3.3），墙粒子此前已有等价处理。归因实验
`benchmarks/fs_pressure/`（10 算例，2 线程 ≈15 min）：

| 臂 | dx=2.0/c0=20 | dx=1.5/c0=20 | dx=2.0/c0=5 | dx=1.5/c0=5 |
|---|---|---|---|---|
| clamp | −37.9 % | −38.1 % | −34.0 % | −37.4 % |
| shell | **−7.2 %** | **−3.4 %** | **+4.9 %** | **−0.9 %** |

σ 阶梯（3.6/7.2/14.4 N/m）：clamp −29.1/−37.9/−47.0 %（随 σ 恶化），
shell −3.2/−7.2/+3.0 %（与 σ 解耦）→ **证实 m2971 的 C3 归因**（压强闭合缺陷，非力幅度缺陷）。
证据等级：Young–Laplace 压差 WITHHELD → **REAL**（六点全 <15 %，两点 <3 %）。

## 新 P1：伪流回升

shell 臂 `max|v|/√(σ/ρR)` = 0.09–0.21，多数越过 Adami(2012) 的 0.1 线（clamp 臂 0.03）。
静态压强修好而界面更"吵"。验收：毛细波/振荡液滴在 shell 臂上重测伪流与频率，
若伪流不降，考虑 δ-SPH 密度耗散或界面压强松弛（Adami §3.4）。

## 新 P1：切换默认值需要整套回归

`free_surface_pressure` 默认仍是 `clamp`；液滴撞池、静水、溃坝的已发表数字全部基于钳位路径。
切默认前须重跑 `benchmarks/{hydrostatic,dam_break,free_surface}` 与 `tests/test_wall_pressure.py`、
`test_dam_break*.py`，并逐条更新文档口径。

## 新 P2：分辨率/σ 交叉项

只有两点 dx 与单点 dx 的 σ 阶梯，`σ × dx` 交叉项未测；R/h ≈ 7.7 仍偏粗，
建议补 dx=1.0 mm + σ=14.4 的四点，判定收敛阶。

## 环境备注

本机 load15 ≈ 150（nproc=128）时，单个 1283 步/362 粒子算例约 90 s（2 线程）；
`benchmarks/fs_pressure/run.py --quick` 可 2 例快速回归。
