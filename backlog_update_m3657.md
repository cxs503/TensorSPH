# backlog 更新（m3657：振荡液滴 mode-2 频率不可测归因，验证欠账结清）

**日期**: 2026-11-05 ｜ **范围**: TensorSPH ｜ **改动类型**: 只增文件 + 测试注释（未改 `src/`）

## 已结清：「mode-2 fit 架构脆弱」（x86_64 +8.6 % vs loong64 −31.8 %）

工件：`benchmarks/oscillating_droplet_instability/`（`run.py` / `probe_release_case.py` /
`release_case_trace.json` / `report.json` / `oscillating_droplet_instability.png`）、
`docs/verification_oscillating_droplet_instability.md`（数字由 `make_doc.py` 注入）、
`tests/test_oscillating_droplet_instability.py`（4 passed）。

| 发现 | 实测 | 含义 |
|---|---|---|
| 能量硬上界 `E_avail = σ(P_ellipse − P_circle)` | R=2 cm：0.00617 J；R=1 cm：0.00308 J | 能量守恒格式的 KE 上界 |
| KE 峰值 / E_avail | **9.4e9×**（R=2 cm）、**474×**（R=1 cm） | 两个算例都**凭空造能**→ 算例无效 |
| 越界时刻 | 0.84 T / 1.10 T | 有效振荡信号 < 1.5 个周期 |
| 同一轨迹四窗口拟合误差极差 | **41.4 / 73.9 个百分点** | 「−43.5 %」与「−31.8 %」= 窗口选择差异，非架构 |
| 拟合器扰动条件数（ε ≤ 1e-4） | 散布 0.0 个百分点 | **不是**拟合器的问题 |

→ mode-2 频率证据等级由「FAIL（−43.5 %）」改判为 **WITHHELD（测量无效）**。
`tests/test_oscillating_droplet.py` 的两个 strict-xfail 保留（历史留痕），理由改写为发散；
待算例稳定后它们会自动 XPASS 失败，强制重基线。

## 新 P0：自由液滴算例自增能（现在唯一门槛项）

两个算例（R = 2 cm / 1 cm，界面周长方向均约 63 个粒子）在无重力、无外驱动下动能增长
2.7–10 个数量级。`stable_dt` 只约束声学 CFL，毛细 CFL 在此并不紧
（`dt_capillary = sqrt(ρ h³/(2πσ))` ≫ 所用 dt），故发散另有来源。候选（需逐项负控）：
1. CSF 表面张力在界面壳层（约 1 粒子厚）的离散不平衡；
2. 自由面压强钳位 `p ≥ 0` + Tait 亏缺与 CSF 力的组合（即 `backlog_update_m2971.md` 的 C3）；
3. 初始晶格核和偏置（同 C2）在 σ = 7.2 N/m 下被放大 100 倍。

**验收判据（机器可裁）**：`benchmarks/oscillating_droplet_instability/run.py` 的
`kinetic_energy_over_available` < 1（并建议 < 0.5），且有效窗口 ≥ 1.5 T；
届时 `tests/test_oscillating_droplet_instability.py` 的 `diverged is True` 断言会变红，
强制更新结论。

## P2：口径统一

后续引用振荡液滴频率必须同时给出（窗口长度, R², KE/E_avail）。
`docs/verification_oscillating_droplet.md` 的 −43.5 % 标注为**历史口径值**。

## 环境踩坑

`nohup ... &` 会被工具按进程组整组收割（120 s 超时）；远长于 2 min 的实算必须
`setsid ... < /dev/null &` 脱离进程组，并把结果落盘到工作树。