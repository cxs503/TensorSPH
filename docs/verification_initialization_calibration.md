# C2 初始化质量校准：验证记录

**Mission**: m3372 ｜ **范围**: TensorSPH ｜ **日期**: 2026-11-04
**改动**: 新增 `src/tensorsph/initialization.py`；`droplet_impact.make_particles` 集成（`calibrate_init` 默认开）；`benchmarks/oscillating_droplet` 集成；`tests/test_initialization_calibration.py`（5 门）。

## 1. 缺陷（error source C2）

规则格点上离散 SPH 密度和不为 1：格点常数质量 `m = rho0·dA` 下，内部
`rho_i = rho0·(1+b)`。`b` 是格点+核常数：

| 格点 | b（实测） | 后果（c0=20, gamma=7） |
|---|---|---|
| 六方（Laplace/振荡液滴线） | **+5.88e-3** | 虚假初始压强 `B[(1+b)^7-1]` ≈ **2.4 kPa** = 6.7× Laplace 信号 |
| 方形（impact 案例） | **−5.3e-5** | 密度亏缺，被 Tait 钳位掩盖（p=0） |

## 2. 修复

`calibrate_masses`：在核（完整核支撑区，2h 边距）测 `b`，一次 `m → m/(1+b)`。
均匀缩放，保相对质量分布（静水剖面仍静水），总质量恰变 `1/(1+b)`。
`|b|>0.05` 拒改（core 含自由面/格点不一致）。

## 3. 验收（backlog P1 判据：初始 `p_max/(rho0 c0^2) < 1e-4`）

| 量 | 校准前 | 校准后 | 判据 | 判定 |
|---|---|---|---|---|
| 六方液滴 p_max/(rho0 c0²) | 5.9e-3 | **<1e-6** | <1e-4 | **PASS** |
| 六方液滴 core rho/rho0 | 1.00588 | **1±1e-7** | =1 | **PASS** |
| 方形池 core rho/rho0 | 0.999947 | **1±1e-7** | =1 | **PASS** |
| impact p_max/(rho0 c0²) | 0（钳位） | 8.3e-7 | <1e-4 | **PASS** |

## 4. 对 mode-2 频率的连带影响（诚实记录）

C2 偏置此前**掩盖**了部分频率缺陷。振荡液滴基准集成校准后，R=1cm 释放算例
mode-2 误差由 −31.78%（未校准, loong64）变为 **−43.53%**，与文档记录的
R=2cm 参考值 −43.5% 一致。钉值 `RECORDED_MODE2_ERR_PCT` 由 31.78 重基线为
**43.53**。15% 门槛仍 FAIL（|−43.53|>15），证据等级不变。

## 5. 局限

- 仅 loong64 实测；x86_64 的校准后值未测（架构脆弱性仍在，strict-xfail 保留）。
- 两个 mode-2 红测试为**既有** strict-xfail（架构脆弱钉扎），本单未改变其红态，
  属独立的 fit-robustness 欠账，非本单回归。
- 未跑全套件（预算）；新模块 5 门 + 关键回归子集已绿。
