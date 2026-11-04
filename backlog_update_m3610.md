# m3610：自由面压强闭合（m2971 backlog P0）结清 + 后续 P1/P2

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ **改动**: `src/tensorsph/droplet_impact.py`（新增默认关闭的 `Config.free_surface_pressure`）、benchmark/doc/tests

## 已结清：P0 自由面压强闭合

`rhs` 压强装配段实现 Adami/Hu/Adams (2012) 式 (11) 壳层外推：自由面粒子（`rho/rho0 < 0.99`）
的压强由完整核邻居外推，替换 `tait_pressure` 的 `p >= 0` 钳位。默认关闭，显式开启。

验收（`benchmarks/laplace_pressure_floor/run_closure.py` → `report_shell.json`，与 `report.json`
未闭合基线同 dt/同步数配对）：

| dx [m] | c0 [m/s] | 钳位 | 闭合 | 改善 [pt] | <15 % | <3 % |
|---|---|---|---|---|---|---|
| 0.002 | 20 | −37.9 % | −7.2 % | 30.7 | ✅ | ❌ |
| 0.002 | 5 | −34.0 % | +4.9 % | 29.1 | ✅ | ❌ |
| 0.0015 | 20 | −38.1 % | −3.4 % | 34.7 | ✅ | ❌ |
| 0.0015 | 5 | −37.4 % | −0.9 % | 36.5 | ✅ | ✅ |

判定：`sigma/R` 压差 **15 % 判据四点全过**；**3 % 未全过**（粗网格高 c0 点 −7.2 %）。
记录：`docs/verification_free_surface_pressure_closure.md`（数字由 `make_closure_doc.py` 注入）、
`tests/test_free_surface_pressure_closure.py`（14 passed：8 单元 + 6 记录）。

## 新发现（闭合的代价）

`v_max/v_ref` 从 0.031 升到 0.142，跨过 Adami 0.1 判据——壳层压强会做功，虚假流增大。
已用测试钉住，属已知代价，不是回退。

## 后续 P1/P2

1. **P1（闭合精化）**：式 (11) 的 `a_i` 项当前取 0；用上一步加速度可望把 −7.2 % 压到 3 % 内。
   验收仍用 `run_closure.py`，判据加一条「四点全 <3 %」。
2. **P1（阈值敏感度）**：`free_surface_density_ratio = 0.99` 未做扫描；建议 0.95/0.97/0.99/1.01 四点，
   看误差与虚假流的折中。
3. **P1（初始化质量校准，m2971 的 C2）**：仍未做。影响面不限表面张力；验收
   `p_max/(rho0 c0^2) < 1e-4`。
4. **P2（既有红）**：`tests/test_oscillating_droplet.py` 两个 mode-2 测试在 loong64 上
   `XPASS(strict)`（marker 为无条件 strict xfail，注释记 x86_64 +8.64 % / loong64 −31.78 %）。
   与本单无关（闭合默认关闭，不触及振荡频率路径），需按架构分别 pin 或改为按 arch 跳过。
5. **P2（闭合的适用范围）**：三维 / 多相 / 带壁面组合只做了单步有限性检查，无端到端收敛记录；
   液滴撞池类强加速自由面未验证。