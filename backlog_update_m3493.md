# m3493：初始态质量校准（m2971 P1）结清 ＋ 口径纠正 ＋ 下一篇欠账

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ **改动**: `src/tensorsph/droplet_impact.py`（新增校准 API，默认启用）＋ 审计工件/文档/测试

## 已结清：P1 初始态核和偏置

m2971 的 P1 要求在 `make_particles`/密度初始化里按核心区实测核和做一次 `m → m/(1+b)`。
本单把它做成**闭式可算**而不是"量出来"：

```
1 + b = A * sum_sites W(|r_ij|, h)        # 只依赖 格点类型 + h/dx，与 dx 无关
m_cal = rho0 * A / (1 + b)
```

新增 `lattice_offsets` / `lattice_density_ratio` / `calibrated_particle_mass`，
由 `Config.mass_calibration`（默认 True）接到 `make_particles` 与 `wall_mass`；
`False` 时逐位复现校准前基线。

| 判据（机裁，18 项全 PASS） | 结果 |
|---|---|
| 解析 1+b vs 实测核心区 rho/rho0（5 个初始态） | 最大相对偏差 **5.6e-8** |
| 校准后核心区 rho/rho0 偏离 1 | ≤ **1.2e-7**（float32 求和噪声） |
| 初始 p_max/(rho0 c0²) < 1e-4（P1 验收） | hex 5.99e-3 → **6.7e-16**；square 0 → **7.2e-7** |
| 解析偏置的 dx 无关性 | 差 2.2e-16 |

工件：`benchmarks/init_mass_calibration/{run.py,make_doc.py,report.json,init_mass_calibration.png}`、
`docs/verification_init_mass_calibration.md`（数字由 `make_doc.py` 从 JSON 注入）、
`tests/test_mass_calibration.py`（11 passed，5.3 s）。

## 口径纠正（重要，别再追错方向）

m2971 写「任何 wcSPH 算例在 c0=20 时都带 2.4 kPa 初始虚假压强」。本单逐格点核过：

* **hex 格点**（`laplace_law`/`static_droplet` 等验证工装用的）：`1+b = 1.005884`（+0.588%）→ **2395 Pa** ✔ 属实；
* **square 格点**（主求解器 `pool_lattice`/`make_particles` 用的，h/dx=1.3）：`1+b = 0.999947`（−0.0053%），
  初始态是**轻微欠密**，Tait 钳到 0，**本来就没有正压**。

所以主求解路径的收益是"核心密度精确等于 rho0"（消除 5e-5 系统偏置），不是"压掉 2.4 kPa"。
符号依赖格点，也依赖 `h/dx`：square h/dx=1.5 时 `1+b = 1.00344` → 1390 Pa。

## 新 P1（接续）：hex 验证工装复算

`benchmarks/laplace_law`、`benchmarks/static_droplet` 仍用 `m = rho0 dx dy`（+0.588%），
即它们的 Laplace 比对在 2.4 kPa 的初始过压上叠加 σ/R = 360 Pa 的信号。
把校准接进去必须连同全部复算与再基线（`tests/test_laplace_law.py` 的 4 个 strict-xfail、
`docs/verification_surface_tension_*.md` 的历史数字）一起做单，本单**有意不动**，只提供 API + 判据。
验收判据不变：`laplace_pressure_floor` 的 `absolute_time_average_tail` 四点 <15%。

## 回归状态

`pytest tests/`：**333 passed, 8 xfailed, 3 failed**（243 s）。
3 红中 2 红（`test_oscillating_droplet` 的两个 strict-xfail 被 XPASS）是**既有**的 arch-fragile 钉死项
（该工装 `benchmarks/oscillating_droplet/run.py:102` 自建 `rho0*dx*dy`，不经过 `make_particles`，
本单改动无法波及；本机实测落在 loong64 分支 −31.78%，而 x86 CI 记 +8.64%）。
第 3 红 `test_wall_pressure[3.0]` 是本单引起的：闭合式里硬编码了 `rho0*dx²` 壁质量，
已改为读 `wall_mass(cfg, 1)`（与求解器同源），复跑 14 passed。

## 环境踩坑

`run_shell` 的后台进程会被整组收割（`(cmd &)` 无效），必须 `setsid nohup ... &` 才能真正脱离；
本单第一次全套回归就在 62% 处被静默杀掉。