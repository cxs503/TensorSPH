# backlog 更新（m3372：C2 初始化质量校准落地）

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ **改动**: 新增 src 模块 + 集成 + 测试

## 已结清：P1 初始化质量校准（C2）

`src/tensorsph/initialization.py`（`calibrate_masses`/`calibrate_droplet`/core mask），
`make_particles` 默认 `calibrate_init=True`。验收 `p_max/(rho0 c0²)<1e-4` 达标
（六方液滴 5.9e-3→<1e-6；方形池 1±1e-7）。工件：
`docs/verification_initialization_calibration.md`、`tests/test_initialization_calibration.py`（5 passed）。

## 连带：mode-2 钉扎重基线

C2 偏置掩盖了部分频率缺陷。振荡液滴基准集成校准后 mode-2 误差
−31.78%→**−43.53%**（=文档 R=2cm 参考 −43.5%）。`RECORDED_MODE2_ERR_PCT`
31.78→**43.53**。15% 门槛仍 FAIL。

## 未结清（留给后续单）

- **mode-2 两个红测试**：既有 strict-xfail（架构脆弱，x86 +8.64% vs loong64
  −31.78% 未校准值）。需 fit-robustness 复核 + 双架构重基线，非本单范围。
- 校准后 x86_64 的 mode-2 值待 CI 复测。
- 全套件回归待 CI（本单预算未跑全量）。
