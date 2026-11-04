# backlog 更新（m3552：P0 自由面压强闭合结清 + 新 P1/P2）

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ 改动类型：改 `src/`（新增可选闭合，默认不变）+ 新增测试/基准/文档

## 已结清：P0 自由面压强闭合（m2971 唯一门槛项）

`Config.free_surface_pressure = "extrapolate"`（默认仍是 `"clamp"`，默认路径逐位不变）
实现 Adami/Ihmsen/Adams 2012 §3.3 shell 压强外推：
`p_i = max(0, Σ_j (p_j + ρ_j g·r_ji) W_ij / Σ_j W_ij)`，仅流体邻居参与（壁面邻居排除，
壁压随后由流体反向外推），`a_i` 丢弃、保留重力项；Jacobi 扫描 `free_surface_pressure_sweeps = 3`。

| dx [m] | c0 | clamp 尾均值误差 | extrapolate | 伪流 max\|v\|/v_ref |
|---|---|---|---|---|
| 0.002 | 20/10/5 | −37.9 / −37.2 / −34.0 % | **−1.8 / −2.0 / −0.6 %** | 0.029–0.031 |
| 0.0015 | 20/10/5 | −38.1 / −36.2 / −37.4 % | **−6.8 / −6.8 / −6.4 %** | 0.032–0.043 |

- 判据（m2971 定的机器可裁项）：4 点 `absolute_time_average_tail` < 15 % → **达成（6/6 < 15 %）**；
- 严格 3 % 只有 3/6（dx=0.002 列），dx=0.0015 列 −6.4…−6.8 %，且细化**不**改善 → 未收敛，如实记；
- 伪流全部 < Adami 0.1 gate（最坏 0.043）；扫描阶梯 1/2/3/6 → −20.1/−6.5/−1.8/+12.1 %、伪流 0.02/0.033/0.031/**0.267**，故默认 3。
- 工件：`benchmarks/laplace_pressure_floor/{report_extrapolate.json,sweep_ladder.json,closure_comparison.png}`、
  `docs/verification_free_surface_pressure.md`（数字由 `compare_closures.py` 注入）、
  `tests/test_free_surface_pressure.py`（10 passed；4 个 benchmark 门在有 report 时也通过）。

## 新 P1：闭合的 3 % 残差

dx=0.0015 列固定 −6.8 %，不随 c0 变，细化反而略差 → 疑与「固定 3 次扫描 + 单粒子厚壳」有关。
建议：扫描数随 dx/h 归一（例如按 `h/dx` 或壳厚度自适应），或改用隐式求解壳值。

## 新 P1：曲率/CSF 与闭合耦合后的再验证

本轮只测压差，未重测 `test_laplace_law.py` 的 4 个 xfail（曲率 κ、切割面力 F_cut）。
闭合改变的是压强场而非 CVD 算符，故预期独立；但应跑一次 A/B 记录。

## 新 P2：陈旧「钉死失败」测试（m2971 遗留）

`tests/test_laplace_droplet.py` 的 4 个断言仍编码 CSF 重构前的事实，未在本单范围内改动。

## 环境

本机（x86_64）上 `tests/test_oscillating_droplet.py` 的 2 个 strict-xfail（mode-2 频率，
架构脆弱：x86 +8.64 % vs loong64 −31.78 %）会 xpass→FAILED，属已登记欠账，与本单改动无关
（默认 clamp 路径逐位复现旧报告 −37.8760475657692）。
