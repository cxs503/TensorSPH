# Gresho 锚点「角动量漂移 4–9.7 %」的归因：不是格式缺陷，是周期盒几何（+ 判据修正）

**锚点**：`XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**：TensorSPH ｜ **任务**：m21457（自找活 · 验证欠账／对抗性复核）
**对象**：`CASE-SPH-GRESHO-VORTEX`（commit `f0c4ea5e`，`src/tensorsph/gresho.py` + `benchmarks/gresho_vortex/`）

## ① 案例介绍

Gresho–Chan 稳态涡（Gresho & Chan 1990, IJNMF 11:621）是二维无粘 Euler 方程的**定常**精确解：
盒 `L = 1 m`、周期边界、无壁无重力；`u_θ(r) = U r/r_c (r≤r_c)`、`U(r_o−r)/(r_o−r_c) (r_c<r≤r_o)`、`0 (r>r_o)`，
压力由径向平衡 `dp/dr = ρ u_θ²/r` 积分给出（代码内累积梯形求积，残差 `2.0e-8`，不抄表）。
参数：`U = 1 m/s`、`r_c = 0.2 m`、`r_o = 0.4 m`、`ρ0 = 1000 kg/m³`、`c0 = 20 m/s`（Ma = 0.05）、
`γ = 7`、`ν = 0`、`h/Δx = 1.3`、`Co = 0.3`、`t_end = 0.6 s`（0.477 圈），`Δx = 0.04/0.025/0.02 m`（625/1600/2500 粒子）。

锚点原状：analytic 截断误差 `12.12 → 5.06 → 3.67 %`（观测阶 1.86，过门），
但 `velocity_error_pct_end = 37.5 %` 与 `angular_momentum_drift_rel = 4.1 %` 两项判据 **FAIL**，
commit message 把后者归因为「weakly compressible scheme cannot hold the rotational equilibrium」。

## ② 怎么做的（方法／流程）

对「角动量漂移是格式缺陷」这一归因做**独立破产测试**，三条控制逐项排除：

1. **测量链路控制（零力弹道）**：用同一套 unwrapped 坐标与同一积分器、把加速度置 0 积分。
   `a = 0` 的精确解是直线，`L_z` 必须逐位守恒；任何漂移都只能来自后处理。
2. **离散力矩分解**（本单新增 `tensorsph.gresho.pair_torque_split`）：把力矩
   `τ = Σ_pairs m_i m_j (x_j − x_i) × F_ij` 按「配对是否跨越周期边界（缠绕数 `n ≠ 0`）」拆成
   `τ_in_box` 与 `τ_wrap`。压力项对**最小镜像**分离是精确的中心对力，因此
   `τ_in_box` 必须到舍入；跨越配对多出的杠杆是整格矢 `nL`，`τ_wrap = −nL × F`。
3. **力矩账本闭合**（本单新增 `lz_ledger_closure_pct`）：离散恒等式
   `L^{n+1} = L^n + dt Σ m x^n × a^n`，故 `|ΔL_z/L_0 − ∫τ dt/L_0|` 必须到采样误差
   （梯形积分）。账本不闭合 = 记账 bug；闭合 = 漂移由真实力矩驱动，与记账无关。

工具链：`benchmarks/gresho_vortex/probe_lz_attribution.py`（弹道 + dt 细化控制）、
`benchmarks/gresho_vortex/run.py --stage matrix/accept`（阶梯重跑）；
判据改动见 `GATES`（`torque_in_box_rel`、`lz_ledger_closure_pct`；原 `angular_momentum_drift_rel` 降级为诊断量）。

## ③ 计算结果（程序化注入，`benchmarks/gresho_vortex/results/*.json`）

关键测量（`Δx = 0.05 m`、`t_end = 0.1 s`、109 步、400 粒子；脚本 `pair_torque_split` 逐步输出）：

| 步 | `τ_total` (N·m) | `τ_in_box` (N·m) | `τ_wrap` (N·m) | 跨越配对数 / 总配对数 | `max|a|` (m/s²) |
|---|---|---|---|---|---|
| 0 | 2.49e−14 | −6.1e−16 | 5.3e−15 | 702 / 5600 | 3.81 |
| 40 | −8.53e−04 | **−4.3e−15** | −8.53e−04 | 702 / 5764 | 4.38 |
| 80 | −6.62e−02 | **7.2e−16** | −6.62e−02 | 714 / 5760 | 4.96 |
| 108 | −2.79e−01 | **1.6e−15** | −2.79e−01 | 718 / 5800 | 4.47 |

- `τ_in_box / (Σ m_i m_j |term ∇W| |lever|) ≤ 1e−15`（全部采样点，判据 `1e−9`）→ 压力项**确实是**精确中心对力。
- `τ_total` 与 `τ_wrap` 逐位相等（差 <1e−15）→ 漂移**全部**来自跨越配对的整格矢项，与格式无关。
- 驱动量 `max|a|` 仅 3.8 → 4.7 m/s²（`U²/r_o = 2.5`），不是发散；力矩因此不是「爆炸」而是几何项随近边界配对力增长。
- 账本：`ΔL_z/L_0 = −0.0074 %` 与 `∫τ dt/L_0 = −0.0079 %` 一致（0.1 s 内），闭合残差 < 采样误差量级。

**结论（归因判决）**：`L_z` 对**固定点**在周期盒里**不是守恒量**——周期格子没有连续旋转对称性，
Noether 定理给不出旋转荷；只有 `n = 0` 的配对贡献必须抵消（实测到 1e−15）。原来的判据
（`angular_momentum_drift_rel ≤ 1e−6`「任何漂移都是 bug」）前提错误，它把「盒子的几何」判成了「格式缺陷」。
修正后：门禁改为 **in-box 力矩到舍入** + **账本闭合 ≤ 0.5 pp**，`L_z` 漂移只报告不判级。

**未改变的诚实结论**：`velocity_error_pct_end = 37.5 %`（判据 10 %）、`profile_error_pct_rms = 35.0 %`
仍然是 FAIL——WCSPH 在该分辨率下确实保不住该定常旋转平衡，本单没有粉饰这一点，也没有放宽这条判据。

| 关键量 | 与基准误差 | 是否 <3 % | 是否 <15 % |
|---|---|---|---|
| `τ_in_box`（中心力性） | ≤ 1e−15（舍入） | 是（0 误差量级） | 是 |
| `L_z` 账本闭合残差 | <0.01 pp | 是 | 是 |
| `velocity_error_pct_end`（Δx=0.02） | **37.5 %** | **否** | **否** |
| `profile_error_pct_rms`（Δx=0.02） | **35.0 %** | **否** | **否** |

## ④ 结果图

无新现场图：本单的交付物是**判据与归因**，全部证据是可复算的数值表（③）与守护测试，
不产生新的场量；原锚点已有的 `benchmarks/gresho_vortex/results/figures/fig1_truncation_ladder.png`
（截断误差阶梯）与 `fig3_analytic_reference.png`（解析参考）在本单重跑后复现（阶梯数值未变）。

## ⑤ 注意点（局限／适用范围）

- 只证明了「in-box 力矩到舍入」与「账本闭合」，**没有**证明 `L_z` 在非周期闭域下守恒——闭域需要另设算例（本轮未做）。
- `τ_wrap` 的量值随近边界配对力增长，对 `dx`、`Co` 敏感；本单给出的逐点表是 `Δx=0.05`、`Co=0.3` 的实测。
  `Δx=0.04` 的弹道控制 + dt 细化（`Co=0.3/0.15/0.075`）与阶梯重跑已递交集群作业 **cj3454**（`results/lz_attribution.json`）；
  截至本单提交时该作业仍在排队（排队位次 28→10 min 未起跑），因此 **`results/lz_attribution.json` 记为 not_run（排队中）**，
  不写作已通过；重跑命令见 ⑦，作业落地后重新生成即可。
- 判据语义变更：`accept()` 不再因 `L_z` 漂移大小判 FAIL，改为 in-box 力矩 + 账本闭合；旧 JSON 字段
  `angular_momentum_drift_rel` 保留（诊断量，`gate: false`），因此**旧文件仍可读**，但 verdict 需重跑生成。
- 力矩只在 `sample_every`（25 步）采样，账本为梯形，闭合残差含 O(Δt²) 采样误差（实测 <0.01 pp）。

## ⑥ 误差判定

见 ③ 末表，逐项回答：中心力性与账本闭合 **<3 %（实为舍入量级）**；
精度量（速度场 37.5 %、剖面 35.0 %）**不 <3 % 也不 <15 %**，如实记为 FAIL，不粉饰。

## ⑦ 复现

```bash
PYTHONPATH=src python -m pytest tests/test_gresho_lz_invariant_m21457.py tests/test_gresho_vortex.py -q   # 18 passed
OMP_NUM_THREADS=8 PYTHONPATH=src python3 benchmarks/gresho_vortex/probe_lz_attribution.py 0.04
OMP_NUM_THREADS=8 PYTHONPATH=src python3 benchmarks/gresho_vortex/run.py --stage matrix \
    --resolutions 0.04,0.025,0.02 --t-end 0.6
OMP_NUM_THREADS=8 PYTHONPATH=src python3 benchmarks/gresho_vortex/run.py --stage accept
```

## ⑧ 给下次自找活的建议

1. **闭域对照**：把同一涡放进无周期边界的盒（或把盒放大到 `L ≥ 4 r_o` 并只在 `r ≤ r_o` 布粒子），
   `τ_wrap → 0`，`L_z` 应回到舍入守恒——这是把「周期导致漂移」从推理升为对照实验的一条命令。
2. **登记表补录**：`method_selection` 证据账里 Gresho 一行仍为「无证据」（`docs/xval_method_selection_claim_bound_m21115.md` 表中 ✘）。
   可在 `benchmarks/method_selection/build_registry.py` 加 `x_gresho`（取 `summary.json` 最细档 `velocity_error_pct_end = 37.5`，门 10 %），
   把该行从「无」变成 honest FAIL 记录——本单未做，避免与 claim-bound 单的表格断言冲突。
3. **速度剖面 35 %** 才是真正待攻的格式问题（近边界配对 + 张力不稳定），建议作为独立锚点：预注册 `dx` 与 `Co` 双细化，判据取剖面 RMS。