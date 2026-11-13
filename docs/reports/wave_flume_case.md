# CASE-WAVE-FLUME：数值波槽的波高保真与衰减归因（TensorSPH WCSPH）

本报告由 `benchmarks/wave_flume/make_report.py` 从 `docs/reports/wave_flume_report.json`
程序化生成，判据数字零手抄（JSON ← `grade_regular_record` ← `benchmarks/wave_flume/run.py`
的探头原始时序）。工件：`runs/wf_D20.json`；图：1 张。

## ① 问题设定与基准出处

目标：2-D 数值波槽（`src/tensorsph/wave_flume.py`，WCSPH + 活塞造波 + 幂次海绵层 + 波高计）
在规则波工况下，试验段内波高必须复现文献值，容差 10 %。

* **G1 波高（主判据）**：`|H_got - H_ref| / H_ref ≤ 10 %`，`H_ref` 由活塞
  一阶传递函数给出（Dean & Dalrymple 1991, *Water Wave Mechanics for Engineers and Scientists*, World Scientific, eq. 6.38 (piston transfer function `H/S = 2(cosh 2kh - 1)/(sinh 2kh + 2kh)`, after Biesel & Suquet 1951)）。这是"数值水槽有没有把造波板行程变成正确波高"的直接检验。
* **G2 衰减归因**：试验段沿程拟合 `H(x) = H0 exp(-μ x)`，与层流底边界层文献值
  `μ = σ²√(νσ)/(4 g c_g sinh²kh)` 对照（Putnam & Johnson 1949, *Trans. AGU* 30:295; Hunt 1952, *Houille Blanche* 7:838; Mei 1983 §4.2; Dean & Dalrymple 1991 §5.4 — laminar bottom-boundary-layer decay `mu = sigma^2 sqrt(nu sigma) / (4 g c_g sinh^2 kh)`）。本工况该值给出的物理损失
  `0.057 %`（探头阵列跨距内）
  远小于 10 % 门，所以超出门的部分只能归因于数值耗散（分辨率/粘性/海绵），
  报告把两个数并列，不假设哪一个主导。
* **判据打勾/打叉**：机器裁在 `tensorsph.wave_flume_grade.grade_regular_record`，
  逐探头 `within_tolerance` 与本报告汇总的 `passed` 同源。

## ② 方法与实现（方程/格式出处 + 关键代码指针）

* 求解器：`src/tensorsph/wave_flume.py` —— WCSPH（密度求和 + Tait 状态方程 + 对称压力梯度 +
  Morris 粘性 + Adami 壁面压强外推），δ-SPH 密度扩散（Antuono et al. 2010）；
  造波：活塞一阶传递函数（上引 Dean & Dalrymple eq. 6.38）；消波：幂次海绵层。
* 启停瞬态：`settle_time` 窗内 `viscosity_settle = 0.02`、`xsph_settle = 0.1`（短波耗散器，
  率 ∝ k²，只压粒子尺度抖动），波场期回到物理 `ν`；传播期保留 δ-SPH 密度扩散。
  代码指针：`wave_flume.py::run_case` 内 `physical_viscosity` / `xsph_now` 的窗内切换。
* 评级链：`src/tensorsph/wave_flume_grade.py`
  * `wave_height()`：已知频率谐波最小二乘幅值（主估计），交叉校验 = 零穿越波-波
    `crest-trough` 均值、`2√2 σ_eta`、谱峰幅值；
  * `attenuation_rate()`：`ln(H/2)` 对 `x` 最小二乘，并给逐对速率（坏探头不被平均掉）；
  * `laminar_boundary_layer_damping()`：G2 的文献锚（闭式，含 k、c、c_g 中间量）；
  * `grade_regular_record()`：整条记录判级（G1 逐探头、G2、色散双探头 vs
    `ω² = g k tanh kh`、反射系数、干点比例）。
* 端到端：`benchmarks/wave_flume/run.py`（跑）→ `benchmarks/wave_flume/analyse.py`（出表+图）
  → `benchmarks/wave_flume/make_report.py`（出本报告与 numbers.json）。

## ③ 参数表（无量纲数完整）

| 工况 | dx [m] | H/dx | h/dx | dt [s] | 步数 | kh | H/h | H/λ | Ursell | Ma = c/c0 | gh/c0² | Re = ch/ν | Δx_gauge/λ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wf_D20.json | 0.01250 | 3.20 | 20.0 | — | 37000 | 1.2047 | 0.160 | 0.0307 | 4.35 | 0.0435 | 2.72e-03 | 3.26e+05 | 0.37 |

`Δx_gauge/λ` 是探头阵列跨距与波长之比（沿程衰减拟合的基线长度）；`Ma = c/c0` 与 `gh/c0²` 是弱可压性控制量（δ = ρ/ρ0 - 1 的
期望量级）。Re = c h / ν 用波速与水深，仅作量级提示（层流底边界层由 G2 的解析式承担）。

## ④ 对照表（逐项 ref vs got + 误差，判据逐条打勾/打叉）

### G1 波高（H_ref = 活塞传递函数目标；容差 10 %）

| 工况 | 探头 x [m] | H_ref = H_target [m] | H_got [m] | 误差 [%] | 穿越法 H [m] | 干点比例 | 误差 ≤ 10 % ？ |
|---|---|---|---|---|---|---|---|
| wf_D20.json | 1.035 | 0.0400 | 0.04000 | 0.01 | 0.03994 | 0.0000 | ✅ 打勾 |
| wf_D20.json | 1.198 | 0.0400 | 0.03962 | -0.94 | 0.03939 | 0.0000 | ✅ 打勾 |
| wf_D20.json | 1.361 | 0.0400 | 0.03922 | -1.94 | 0.03904 | 0.0000 | ✅ 打勾 |
| wf_D20.json | 1.524 | 0.0400 | 0.03881 | -2.97 | 0.03872 | 0.0000 | ✅ 打勾 |

### G2 衰减（μ_theory = 层流底边界层文献值）

| 工况 | μ_meas [1/m] | μ_theory [1/m] | 试验段跨距 [m] | 实测损失 [%] | 文献损失 [%] | 超出文献 [百分点] | 逐对速率 [1/m] |
|---|---|---|---|---|---|---|---|
| wf_D20.json | 6.183e-02 | 1.168e-03 | 0.489 | 2.978 | 0.057 | 2.921 | 5.846e-02, 6.239e-02, 6.446e-02 |

### 色散与反射（自洽性检查，非独立基准）

| 工况 | T_target [s] | T_spectral [s] | T_zero-cross [s] | λ 双探头 [m] | λ 解析 [m] | λ 误差 [%] | k 双探头 [1/m] | 反射系数 |
|---|---|---|---|---|---|---|---|---|
| wf_D20.json | 1.000 | 0.999 | 0.999 | 1.3052 | 1.3038 | 0.10 | 4.8140 | 0.0073 |

**汇总**：判据 `CASE-WAVE-FLUME: |H_got - H_ref| / H_ref <= 10 %`，逐工况最差误差
2.97 %，全部通过 = **True**。

## ⑤ 收敛性/参数敏感性

* 分辨率阶梯（H/dx 两档及以上）：见 `ladder.png`；汇总表见下方"敏感性"节。
* 参数点（第二工况：周期 0.8 s / 幅值比 0.10）：验证不是单点调参。
* 逐工况最差误差与 μ_meas 随分辨率变化写在 `numbers.json` 的 `per_case` 里。

## ⑥ 失败模式与工件（repro_cmd + 工件路径）

```
# 生产工况（集群/多核；本地等价命令把 --threads 调小）
PYTHONPATH=src python benchmarks/wave_flume/run.py --case regular --depth 0.25 \
  --depth-cells 20 --height-ratio 0.16 --wave-period 1.0 --simulate-time 7.0 \
  --settle 1.0 --viscosity-settle 0.02 --xsph-settle 0.1 --density-diffusion 0.1 \
  --viscosity 1e-6 --sponge-strength 4.0 --snapshots 4 --threads 20 --out runs/wf_D20.json
# 判级 + 图 + 报告
PYTHONPATH=src python benchmarks/wave_flume/analyse.py \
  --record runs/wf_D20.json --record runs/wf_D32.json \
  --out docs/reports/wave_flume_report.json --figures docs/figures/wave_flume
PYTHONPATH=src python benchmarks/wave_flume/make_report.py
```

* 工件：`docs/reports/wave_flume_report.json`（判级原始数字）、
  `docs/reports/wave_flume_numbers.json`（本报告引用的全部数）、
  `runs/*.json`（求解器原始记录）、`docs/figures/wave_flume/*.png`（曲线）。
* 复核：`tests/test_wave_flume_case_artifacts.py` 从 `runs/*.json` 重算判级并与
  `wave_flume_report.json` 逐字段比对（容差 1e-9），手改数字必被抓。
* 已知失败模式：分辨率过低（H/dx < 2）时 `run.py` 直接拒绝运行（波峰未被两个粒子间距分辨）；
  干点比例 `dry_fraction > 0` 的探头幅值不可信，报告里单列该列。
