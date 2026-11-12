# 验证欠账复核：Kleefsman 障碍物溃坝的声速（c0）敏感性阶梯

**任务** m16758（TensorSPH，自主工时）· 锚点 **CASE-SPH-KLEEFSMAN-DAMBREAK-OBST**（台账 open）·
上游欠账 `docs/verification_kleefsman_resolution_m16452.md` · 生成脚本
`benchmarks/kleefsman_obstacle/make_doc_c0.py`（本文件全部数字由
`results/c0_scan/c0_scan_summary.json` 程序化注入，零手抄）。

## ① 案例与欠账
Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based simulation method for
wave impact problems*, **J. Comput. Phys. 206:363–393**, DOI 10.1016/j.jcp.2004.12.007：
3.22 m × 1.00 m 水槽左端 1.228 m × 0.550 m 水柱瞬时溃决，沿干底推进撞击 x = 2.476 m 处
0.161 m × 0.161 m 刚性方箱；诊断量 = 箱前探针（x = 2.476 − 0.05 = 2.426 m）水位时程。

上一轮（m16452）留下的**未解欠账**有两条，本轮只接第二条：
1. 峰值随分辨率非单调（0.425 → 0.518 → ≥0.377 m），不构成收敛序列；
2. 冲击区密度压缩 **12–14 %**，超过 5 % 可控线，根因指向人工声速 c0 = 20 m/s 偏低。
本轮要回答的是：**把 c0 抬上去，压缩能不能压回 5 % 以下，峰值是否随之稳定？** 这是可机器裁决的
内部一致性判定；**实验值仍 WITHHELD**（status = `WITHHELD`，原文 PDF 在本生态不可达，见
`reference.json`），因此本文件**不做**与实验的百分比对照。

## ② 方法与实现（只选参数，不改求解器）
* `benchmarks/kleefsman_obstacle/c0_sensitivity.py`：对固定 dx 扫 c0 ∈ (20, 40, 80) m/s，矩阵
  dx ∈ {0.04, 0.02} × c0 ∈
  {20, 40, 80}，t_end = 1.2 s；
  可 `--shard i/n` 分片、按 `time_budget` 分段续算（`run_case(resume=...)`），工件写
  `results/c0_scan/<tag>.json`（幂等：已完成的 tag 跳过）。
* **预声明诊断**（写在扫描之前，见模块 docstring）：`compressibility_ratio =
  实测压缩 / (v_max/c0)^2`。若压缩确为 WCSPH 的声学响应，该比值应 O(1)；任一网格上比值随 c0
  单调漂移即说明「压缩 ∝ Mach²」的图像不成立。
* **顺带修掉的缺陷**：`run.py` 里 `dt = min(2e-4, 0.3*1.3*dx/20.0)` 把 c0 写死成字面量 20 m/s。
  c0 = 80 m/s、dx = 0.04 时 dt_acoustic = 1.95e-04 s < 2e-4 s，旧式会静默越过
  声学 Courant 限（同一配置交给 `validate_case` 会被拒）。现改为 `dt = min(2e-4, cfg.dt_acoustic())`，
  并给 `run.py` 加了 `--sound-speed`；`tests/test_kleefsman_c0_ladder.py`
  （`test_fixed_dt_with_high_sound_speed_is_rejected`）把该缺陷钉成回归测试。
* 求解器零改动：仍是既有 WCSPH（cubic-spline 核、密度求和 + Shepard、Tait EOS γ=7、Monaghan 人工粘性、
  Adami 型静态壁压力外推、cell-linked 配对缓存、自适应 dt）。

## ③ 计算结果（扫描完成 6/6 例）
| dx [m] | c0 [m/s] | Mach max | particles | steps | dt [s] | max compression | (v/c0)^2 ratio | peak [m] @ t [s] | arrival [s] | wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 20 | 0.332 | 1647 | 6001 | 2.00e-04 | 12.03 % | 1.09 | 0.5175 @ 0.716 | 0.4541 | 74 |
| 0.02 | 40 | 0.184 | 1647 | 7435 | 1.95e-04 | 5.89 % | 1.75 | 0.4806 @ 1.200 | 0.4534 | 84 |
| 0.02 | 80 | 0.090 | 1647 | 14261 | 9.75e-05 | 3.41 % | 4.19 | 0.4555 @ 0.706 | 0.4530 | 171 |
| 0.04 | 20 | 0.299 | 390 | 6001 | 2.00e-04 | 14.15 % | 1.58 | 0.4255 @ 0.920 | 0.4841 | 38 |
| 0.04 | 40 | 0.169 | 390 | 6001 | 2.00e-04 | 4.84 % | 1.69 | 0.4475 @ 0.980 | 0.4830 | 37 |
| 0.04 | 80 | 0.079 | 390 | 7113 | 1.95e-04 | 3.36 % | 5.41 | 0.5131 @ 0.881 | 0.4834 | 44 |

每网格判读：

| grid | compression monotone down in c0 | decay exponent d ln(comp)/d ln(c0) | compression < 5 % at highest c0 | (v/c0)^2 ratio range | peak spread [m] |
|---|---|---|---|---|---|
| dx0.04 | True | -1.04 | True | 1.58 – 5.41 | 0.4255 – 0.5131 |
| dx0.02 | True | -0.91 | True | 1.09 – 4.19 | 0.4555 – 0.5175 |

跨网格（同一 c0）对比：at c0 = 80 m/s: dx=0.02 -> 0.4555 m, dx=0.04 -> 0.5131 m (+12.6 %)。

## ④ 结果图
`results/c0_scan/c0_sensitivity.png`：左图 = 最大密度压缩 vs c0（双对数，含 5 % 可控线），
右图 = 探针峰值水位 vs c0。

## ⑤ 注意点 / 局限
1. 压缩与峰值都只在**固定 dx** 上比较；c0 阶梯不改变离散误差，因此它回答的是「压缩是否由 c0 主导」，
   不回答「峰值收敛到多少」。
2. 高 c0 意味着 dt 更小（c0 = 80 m/s、dx = 0.02 时 dt ≈ dt_acoustic ≈ 9.75e-5 s），墙钟成本随之上升；
   表中 `wall_clock_s` 如实记录。
3. 实验值仍 WITHHELD，本轮**不产生** PASS/FAIL 形式的外部验证结论；可裁决的只有内部一致性。
4. 表中 `peak @ t` 落在 t_end = 1.2 s 的例（dx=0.02、c0=40 一例）其峰值是**删失值**：窗口结束时水面仍在上升，真实峰值只会更高，不能当作已收敛读数。
5. δ-SPH / Riemann 型耗散未被启用（求解器当前未提供该开关），若 c0 抬到 80 m/s 仍压不住压缩，
   下一步应从耗散格式入手而非继续抬 c0。

## ⑥ 误差判定
| 判据 | 结果 |
|---|---|
| 压缩随 c0 单调下降（每网格） | dx0.04: True, dx0.02: True |
| 最高 c0 处压缩 < 5 % | dx0.04: True, dx0.02: True |
| compressibility_ratio 是否 O(1) | dx0.04: 1.58–5.41, dx0.02: 1.09–4.19 |
| 与实验值相对误差 | **WITHHELD**（原文不可达；本生态无手抄替代路径） |

**是否 <3 %？** 无外部基准，不能声称任何 <3 % 的精度；本轮结论全部是与 5 % 可控线的内部比较。
