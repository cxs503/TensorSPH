# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 分辨率阶梯的收敛阶、Richardson 外推与不确定性带

TensorSPH · 分支 agent/m17129 · 本文档由 `benchmarks/kleefsman_obstacle/make_doc_ladder.py` 从 `results/resolution_study.json` + `results/convergence_ladder.json` + `dx*_time_series.csv` 程序化注入生成（零手抄）。图：`docs/figures/kleefsman_convergence_ladder.png`。

## ① 案例介绍
Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *J. Comput. Phys.* **206**:363–393, DOI 10.1016/j.jcp.2004.12.007：3.22 m × 1.00 m 水槽，左端 1.228 m × 0.550 m 水柱瞬时溃决，沿干底推进撞击 x = 2.476 m 处 0.161 m × 0.161 m 刚性方箱。几何自洽 1.228 + 1.248 + 0.744 = 3.220 m。
诊断量：箱前探针（x = 2.476 − 0.05 = 2.426 m）水位时程的**峰值**与**到达时刻**（越过 0.02 m）。

| 协议量 | 值 |
|---|---|
| t_end | 1.0000 |
| sound_speed | n/a |
| probe_offset | n/a |
| probe_halfwidth | n/a |
| surface_particles | n/a |
| surface_gap | n/a |
| arrival_threshold | n/a |
| h_factor | n/a |

**实验值状态：仍然 WITHHELD。** 原文 PDF 在本节点不可达（research.rug.nl 副本 Cloudflare 403、ScienceDirect 403、archive.org 不可达），H1–H4 水位**无公开数字化表**。因此本文件不做与实验的百分比对照；它回答的是一个前置问题——**该诊断量在当前数值设置下是否已经收敛到可以用 10 % 容差去比**。未收敛的量没有误差棒，任何「<10 %」的结论都不可falsify。

## ② 怎么做的
1. **协议一致性守卫（缺陷修复）**：旧 `resolution_study.json` 把 `t_end = 1.0 s` 的 dx=0.01 档与 `t_end = 1.2 s` 的 dx=0.02/0.04 档并列成一张「分辨率研究」表——三档物理时长不同，残差无意义。`convergence.py` 现在给每档写协议指纹（`t_end/sound_speed/probe_offset/probe_halfwidth/surface_particles/surface_gap/arrival_threshold/h_factor`），合并时以**新算档为基准**，指纹不符的旧档进 `runs_stale_protocol` 隔离区而不是混进阶梯。
2. **重跑一致性阶梯**：四档 dx = 0.04 / 0.02 / 0.01 全部以 `t_end = 1.00 s`、同一探针与同一自由面估计器重算
   （集群作业 cj1423；旧的三档结果保留在隔离区，不删除）。
3. **收敛阶 / 外推 / GCI**：`benchmarks/kleefsman_obstacle/convergence_analysis.py` 对每三个网格（均匀加密比 r）算观测阶 p = ln|f_med−f_coarse|/ln r ÷ …, Richardson 外推 f_exact ≈ f_fine + (f_fine−f_med)/(r^p−1)，Roache GCI_fine = Fs·|(f_med−f_fine)/f_fine|/(r^p−1)，Fs = 1.25。纯函数、有合成阶梯单测（已知 p 与精确值回读误差 <1e-9）。
4. **把「水位」与「瞬时最高粒子」分开**：瞬时峰值被抛射水花控制，不是收敛量（历史记录 0.426 / 0.518 / ≥0.377 m 在 dx = 0.04/0.02/0.01 上非单调）。实验电容式波高仪本身是低通的，故另算**仪表等效水位**：时程先做 ±15 ms 对称滑窗平均再取峰（窗口 20/30/50 ms 敏感性一并给出）。

## ③ 计算结果
| dx [m] | 流体粒子 | 步数 | 原始峰值 [m] @ t [s] | 仪表水位峰值(30 ms) [m] | 到达时刻 [s] | 密度压缩 | 越界 |
|---|---|---|---|---|---|---|---|
| 0.04 | 390 | 6001 | 0.4255 @ 0.920 | 0.4093 | 0.4841 | 14.15 % | 0 |
| 0.02 | 1647 | 6001 | 0.5175 @ 0.716 | 0.5019 | 0.4541 | 12.03 % | 0 |
| 0.01 | 6710 | 2056 | 0.3773 @ 0.966 | 0.3678 | 0.4443 | 13.05 % | 0 |

收敛诊断（最细三档 = 0.04, 0.02, 0.01）：

| 指标 | 最细三档值 | 观测阶 p | Richardson 外推 | GCI_fine | 判定 |
|---|---|---|---|---|---|
| arrival_time_s | 0.4841 / 0.4541 / 0.4443 | 1.620 | 0.4396 | 0.0133 | CONVERGED |
| peak_height_m_raw | 0.4255 / 0.5175 / 0.3773 | n/a | n/a | n/a | NOT_CONVERGED |
| gauge_level_peak_w020ms | 0.4149 / 0.5080 / 0.3705 | n/a | n/a | n/a | NOT_CONVERGED |
| gauge_level_peak_w030ms | 0.4093 / 0.5019 / 0.3673 | n/a | n/a | n/a | NOT_CONVERGED |
| gauge_level_peak_w050ms | 0.3911 / 0.4845 / 0.3660 | n/a | n/a | n/a | NOT_CONVERGED |

**阶梯总判定：PARTIALLY_CONVERGED**。原始三档的 `t_end` 不一致（[{'dx': 0.01, 'differs': {'t_end': [1.2, 1.0]}}]）已在 `results/resolution_study.json:protocol_mismatch` 显式登记；评分前用 `--window-t 1.00 s` 把各档统一截到公共时窗（三个峰值时刻都落在窗内：0.920 / 0.716 / 0.966 s），故求解器协议指纹一致、协议不匹配项 = 0。

删失检查（峰值/仪表峰距窗尾 < 5%% 窗口长即判删失）：`censored = True`。

## ④ 结果图
`docs/figures/kleefsman_convergence_ladder.png`：左=四档探针水位时程（细线=原始、粗线=30 ms 仪表等效）——原始曲线的尖峰在细网格上被孤立飞溅粒子抬起，仪表曲线族相互贴近；右=原始峰值/仪表峰值/到达时刻随 dx 的变化及 Richardson 外推水平线。

## ⑤ 注意点与局限
1. 实验值仍 WITHHELD（见①），本节全部判据都是**内部一致性**判据，不构成「验收通过」。
2. 仪表等效窗口 30 ms 是**建模选择**（模拟低通波高仪），不是 Kleefsman 实验的仪器参数；已给 20/30/50 ms 三档敏感性，结论须对窗口稳健才可用。
3. c0 = 20.0 m/s 下冲击区密度压缩仍高于 5 % 可控线（见③），Splash 细节不可信；c0 阶梯（`docs/verification_kleefsman_dt_sound_speed_family.md`）另有一档把压缩压到 3.4 %，但代价是 dt 减半，与本阶梯不同协议，故未混算。
4. 峰值若出现在 `t_end` 附近即为「窗口内最大」，属删失数据；③ 表已给峰值时刻，可据此判读。

## ⑥ 误差判定
| 事项 | 基准 | 误差/判定 |
|---|---|---|
| 阶梯协议一致性（修复前混用 t_end） | 协议指纹全等 | 修复后不一致项 = 0 |
| 仪表水位峰值收敛 | 最细三档自比较 | p = n/a，GCI = n/a，判定 NOT_CONVERGED |
| 原始峰值收敛 | 同上 | p = n/a，判定 NOT_CONVERGED |
| 到达时刻收敛 | 同上 | p = 1.620，GCI = 0.0133，判定 CONVERGED |
| 与 Kleefsman 2005 实验的 <10 % 判据 | 实验峰值/到达时刻 | **未评估**（实验值 WITHHELD，无基准不得声称通过） |
| 是否 <3 % | —— | 无外部基准，**不做该断言**（收敛阶与 GCI 只给离散误差带） |
