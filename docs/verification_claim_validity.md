# 验证读数有效性闸门 + 跨仓耦合算术（claim validity）

> 生成器 `benchmarks/claim_validity/validity.py`；工件 `benchmarks/claim_validity/audit.json`、图 `benchmarks/claim_validity/figures/claim_validity.png`。本文件与 JSON 全部由生成器输出，**不要手改**；数字无手抄。

## ① 为什么需要这一层

`docs/verification_ledger.md` 用 3 %/15 % 阶梯回答「离参考值多远」。它回答不了必须先问的那个问题：**采样时还剩下一个可测的流动吗？**本闸门给出机器可裁的答案，并顺便核对账本里声称的跨读数耦合算术是否成立。

## ② 判据（机器可裁）

| 闸门 | 判据 | 物理含义 | 类型 |
|---|---|---|---|
| 约束 `containment` | rms 半径/R ≤ 1.3 | ε=0.1 的毛细振荡必须留在 ~1.15R 内；跑飞=粒子喷射 | hard |
| `mach_number` | max\|u\|/c0 ≤ 1.0 | 越过声速后 Tait 状态方程无意义 | hard |
| `estimator_spread` | (max−min)/ω_ref ≤ 0.25 | 三个独立估计子必须互相自洽，否则「频率」不是测量量 | soft |
| `fit_quality` | r² ≥ 0.70 | 拟合必须解释信号本身 | soft |
| `null_control` | σ=0 负控 ω/ω_ref ≤ 0.25 | 无表面张力时同一估计子不得凭空造出半条 Lamb 频率 | soft |

hard 失败 ⇒ `INVALID`（读数不可用）；仅 soft 失败 ⇒ `SUSPECT`；全过 ⇒ `VALID`。

## ③ 逐条读数：有效性

### `oscillating_droplet.frequency.lamb` — **INVALID**

物理量：fundamental frequency of an inviscid 2-D droplet (Lamb omega_2)。

| 闸门 | 实测 | 限制 | 通过 | 说明 |
|---|---|---|---|---|
| `containment` (hard) | 3531.38 | <= 1.3 | ❌ | rms radius at the end of the recorded run over the droplet radius (r_eff/R = 4994.1); a capillary oscillation at eps = 0.1 must stay inside ~1.15 R |
| `mach_number` (hard) | 9092.27 | <= 1 | ❌ | peak particle speed 1.818e+05 m/s over c0 = 20.0 m/s: beyond Mach 1 the Tait EOS has no meaning and the cloud is a spray, not a liquid |
| `estimator_spread` (soft) | 4.35407 | <= 0.25 | ❌ | spread of the three independent frequency estimators (second moment / kinetic energy / zero crossing) divided by the reference: 41.50 … 361.46 rad/s vs 73.48 rad/s |
| `fit_quality` (soft) | 0.371244 | >= 0.7 | ❌ | coefficient of determination of the damped-cosine fit that produced the quoted frequency |
| `null_control` (soft) | 0.521367 | <= 0.25 | ❌ | sigma = 0 control of the sibling ladder reports 38.31 rad/s, i.e. 52.1 % of the reference: with no surface tension the same estimator sees lattice noise at more than half the Lamb frequency |

结论：The recorded trajectory is not a droplet: containment and Mach gates fail by three to four orders of magnitude, two of the three frequency estimators disagree with each other by more than the reference itself, and the fit explains 37 % of the variance. This reading may not be graded FAIL (which asserts a measured physics defect); it is WITHHELD.

工件：`benchmarks/oscillating_droplet/validation_report.json`、`benchmarks/oscillating_droplet/gauge_audit.json`、`benchmarks/oscillating_droplet/validation_report_lamb.json`

## ④ 耦合算术：动态 ω 是否复现静态 σ_eff

定律：`omega^2 = n(n^2-1) sigma_eff/(rho R^3)  =>  omega/omega_Lamb = sqrt(sigma_eff/sigma)`

静态力路径读数 σ_eff/σ = **0.7040**（`benchmarks/laplace_pressure/validation_report.json -> cases[0].sigma_eff_over_sigma`）⇒ 预测 ω 亏损 **-16.10 %**，而账本引用的读数是 **-43.53 %**。

| 读数 | ω (rad/s) | ω/ω_Lamb | 反解 σ_eff/σ | 相对静态 σ_eff | 相对预测 ω | 运行有效 |
|---|---|---|---|---|---|---|
| `quoted.ledger_claim.dx2.0mm` | 41.498 | 0.5647 | 0.3189 | -54.70 % | -32.70 % | INVALID |
| `lamb.mode2.n2.dx1.5mm` | 28.154 | 0.3831 | 0.1468 | -79.15 % | -54.34 % | SUSPECT |
| `lamb.mode3.n3.dx1.5mm` | 56.308 | 0.3831 | 0.1468 | -79.15 % | -54.34 % | SUSPECT |
| `lamb.coarse.n2.dx2.0mm` | 62.111 | 0.8452 | 0.7144 | +1.48 % | +0.74 % | SUSPECT |

账本旧注声称的耦合：残差 **27.43 pt** vs 判据 15 pt ⇒ **unsupported**（the ledger note claimed that a -43 % frequency is consistent with sigma_eff/sigma ~ 0.7）。

### ④b 阶梯独立性（同一比值的两行不是两次证据）

`benchmarks/oscillating_droplet/validation_report_lamb.json` 的 3 行里，`2` 个是**互相独立**的频率拟合：重复组 [['mode2', 'mode3']]。

| 算例 | n | dx (mm) | ω_Lamb | ω 实测 | ω/ω_Lamb | 自报误差 | 运行有效 |
|---|---|---|---|---|---|---|---|
| `mode2` | 2 | 1.5 | 73.485 | 28.154 | 0.383125000 | -61.688 % | SUSPECT |
| `mode3` | 3 | 1.5 | 146.969 | 56.308 | 0.383125000 | -61.688 % | SUSPECT |
| `coarse` | 2 | 2.0 | 73.485 | 62.111 | 0.845218750 | -15.478 % | SUSPECT |

结论：rows mode2+mode3 share one omega_measured/omega_analytic ratio to machine precision: the reported errors differ only by the analytic reference, so the ladder contains 2 independent frequency measurement(s), not 3.

## ⑤ 注意点 / 局限

1. 闸门只能用在**带遥测**的工件上（终态几何、能量、多估计子）。当前账本 21 条读数中只有 1 条可探，其余如实标 `UNPROBED`——不是「合格」，而是「无法从工件断言」，不得据此升级证据等级。
2. 约束闸门用 rms 半径而非单粒子半径：单体逃逸不足以判死（`escaped_but_bounded` 自检为 VALID），整体膨胀才算。
3. 耦合算术把 σ_eff 当作**唯一**误差源。粘性阻尼、离散化频散、模态耦合都会让 ω 偏离该预测；因此「落在 3 % 内」只支持「力路径缺陷是该动态读数的主导误差源」，不能反过来说明模型已通过动态验证。
4. 反解 σ_eff = (ω/ω_Lamb)² 只在谐振子近似下成立（ε=0.05…0.1 的小幅线性模态）。大幅振荡或跑飞轨迹上该式没有意义——这正是先过闸门再谈耦合的原因。

## ⑥ 自检（负控）

| 合成算例 | 闸门输出 | 期望 |
|---|---|---|
| `healthy_run` | VALID | VALID |
| `blown_up_run` | INVALID | INVALID |
| `escaped_but_bounded` | VALID | VALID |
| `noisy_estimator` | SUSPECT | SUSPECT |

闸门必须能**失败**：`blown_up_run` 判死、`noisy_estimator` 判 SUSPECT 而不判 INVALID，两条负控由 `tests/test_claim_validity.py` 守护。

## ⑦ 维护

```bash
# 新增遥测工件后：先重生成有效性别表，再重生成账本（账本读取本闸门的裁定）
python benchmarks/claim_validity/validity.py --figure
python benchmarks/verification_ledger/build_ledger.py --figure
python benchmarks/claim_validity/validity.py --check   # 漂移检查
```

