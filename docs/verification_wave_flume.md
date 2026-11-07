# 波槽（`tensorsph.wave_flume`）验证状态：已经证明了什么、还欠什么

**范围**: TensorSPH ｜ 本文是 `src/tensorsph/wave_flume.py` 与 `tests/test_wave_flume.py`
docstring 所指向的验证入口。数字全部来自
`benchmarks/startup_quiescence/report.json`（由 `make_docs.py` 注入）。

## 1. 这个模块要做什么

矩形 2-D 波槽：活塞造波（一阶造波理论 / Goring 孤波条件）、海绵层吸收、波高计，
把 WCSPH 核心（`droplet_impact.rhs` + 运动壁面输入）当作求解器。它自身不含新算子，
新增的是**算例**与**解析锚**：Airy 色散 `omega^2 = g k tanh(kh)`（Dean & Dalrymple
1991, eq. 3.23）、活塞传递函数 `H/S = 2(cosh 2kh - 1)/(sinh 2kh + 2kh)`（同书 eq. 6.38）、
一阶孤波 `eta = H sech^2(alpha(x - ct))`。

## 2. 已验证（有可复算证据）

| 项 | 证据 | 状态 |
|---|---|---|
| 色散关系、波数反演、相速极限、Airy 波形、活塞传递函数、Goring 冲程/速度/加速度解析式 | `tests/test_wave_flume.py`（解析锚自洽，含深/浅水极限与级数分支连续） | 通过（解析对解析） |
| 水柱格点恰好填满指定水深、质量单位与 kcal 标定一致 | `tests/test_wave_flume.py` | 通过 |
| **静止水柱（t=0，活塞冻结）是离散静水平衡** | `benchmarks/startup_quiescence`（本页 §3） | 通过：内区压强偏差 `8.733e-04`·`rho0 g H` |

## 3. 未验证 / 未决

| 项 | 现状 |
|---|---|
| **波浪工况本身**（造波→传播→吸收，波高/周期/反射系数 vs Airy） | **未跑、无已提交证据**。`benchmarks/wave_flume/run.py` 只提供单算例记录器，`artifacts/` 被 gitignore，仓库内没有 `report.json`。本单**未**补，因为一次可用工况需要上万粒子 × 数万步（本机 load15 ≈ 120，容量纪律禁止起新的重算力进程）。 |
| **静止水柱的动态底噪**（F1，open） | 默认配置（`viscosity = 1e-6`、`density_diffusion = 0.1`）下噪声不衰减：内区 `v_late_rms = 0.0798` m/s，是 `u_orbital = 0.133` m/s（H=0.03 m）的 `0.60` 倍。消融见下表。 |
| 反射系数、海绵层吸收效率、孤立波传播保真度 | 模块里有 `reflection_coefficient` / `separate_incident_reflected` 工具，**均无定量验证证据**。 |

消融（同一静止水柱，只改一个数值设置，0.06 s）：

| 配置 | v_peak (m/s) | 内区 v_late_rms (m/s) |
|---|---|---|
| shipped | 0.1431 | 0.0798 |
| shipped, overrides viscosity=0.01 | 0.0608 | 0.0079 |
| shipped, overrides density_diffusion=0.0 | 0.0880 | 0.0472 |
| shipped, overrides density_diffusion=0.0,viscosity=0.01 | 0.0513 | 0.0072 |

结论：底噪由**数值粘性缺位**主导，加上 `viscosity = 0.01`（`droplet_impact` 的默认数值
粘度）后底噪降 `10.2` 倍。
这是工程决策（波槽是否该保持"物理水粘度"），需要另开一单决定后重跑分级工况。

## 4. 结果图

![flume quiescence](benchmarks/startup_quiescence/figures/startup_v_series.png)

## 5. 误差判定

| 关键量 | 基准 | 实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 静止水柱内区初始压强 | `rho0 g (H - y)` | `8.733e-04`·`rho0 g H` | `0.087 %` | **是** |
| 静止水柱动态底噪 | 无验收基准 | `0.0798` m/s | 相对 `u_orbital` `+60 %` | **否/无基准**：判据应是"底噪 ≪ 目标波高对应的轨道速度"，本配置不满足 |
| 波浪工况（波高/周期/反射） | Airy 解析 | **未测** | **未测** | **无法回答**——本页不声明任何波浪精度 |

## 6. 下一步（给接单的人）

1. 先定波槽的数值粘度/密度扩散设置（见 §3 消融），再跑分级工况；否则噪声与
   H=3 cm 波信号同量级，误差表没有意义。
2. 分级工况建议：同一 `H/T` 下 2–3 个分辨率、`dx` 由 `depth_cells = 20/30/40` 生成；
   读数用 `probe_elevation` + `dominant_omega`，与 `wavelength()` 对比；反射用
   `separate_incident_reflected` 两探头法。
3. 跑完把 `report.json` 提交进 `benchmarks/wave_flume/`（不要只留 `artifacts/`），
   并把本页 §3 的"未验证"逐条改成实测行。
