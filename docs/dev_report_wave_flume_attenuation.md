# CASE-WAVE-FLUME 第 2 轮：波高衰减评级链已建成，分级工况尚未取回（honest 部分分）

**范围**: TensorSPH ｜ 分支 `agent/m15493` ｜ 任务锚点 `CASE-WAVE-FLUME`
（判据：算例池 WAVE-FLUME，ref = 文献波高衰减，容差 <10%）

**结论先行（不粉饰）**: 本单**没有**拿到分级工况的实测波高，因此**不能**声明任何"波高
误差 <10%"的达标结论。已交付的是**把它判死/判活的那条链**：文献锚公式、波高与衰减估计器、
合成信号单测（13 项通过）、端到端评级/出图脚本（在合成记录上跑通），以及必须先把算例送上
集群的运行配置。失败阶段与原因见 §6。

---

## ① 问题设定与基准出处

目标：2-D 数值波槽（`src/tensorsph/wave_flume.py`，WCSPH 核心 + 活塞造波 + 幂次海绵层 +
波高计）在规则波工况下，波高必须复现文献值，容差 10%。判据拆两条（机器可裁）：

* **G1 造波保真**：`|H_meas - H_target|/H_target ≤ 10%`，`H_target` 由活塞一阶传递函数
  `H/S = 2(cosh 2kh - 1)/(sinh 2kh + 2kh)` 给出 —— Dean & Dalrymple 1991, *Water Wave
  Mechanics for Engineers and Scientists*, World Scientific, eq. 6.38（原始 Biesel & Suquet
  1951）。实验侧离散度量级 ≲5%（Ursell, Dean & Yu 1960, *J. Fluid Mech.* 7:33–52）。
* **G2 衰减归因**：沿波高计阵列拟合 `H = H0 exp(-μx)`，与层流底边界层参考
  `μ = σ²√(νσ)/(4 g c_g sinh²kh)`（由 `D=(1/4)ρ u_b²√(νσ)` 与能流 `(1/2)ρg a²c_g` 推出）
  对照 —— Putnam & Johnson 1949, *Trans. AGU* 30:295；Hunt 1952, *Houille Blanche* 7:838；
  Mei 1983 *The Applied Dynamics of Ocean Surface Waves* §4.2；Dean & Dalrymple 1991 §5.4。

**本单实测的物理参考值**（`tensorsph.wave_flume_grade.laminar_boundary_layer_damping`
计算，非手抄）：工况 `T = 1.0 s, h = 0.25 m, ν = 1e-6 m²/s` 给
`k = 4.81897 1/m`、`c = 1.30384 m/s`、`c_g = 0.93653 m/s`、`μ = 1.16847e-03 1/m`，
即跨 2 m 试验段的**物理**衰减只有 **0.233%**。所以 10% 门里几乎全部预算属于**数值**误差；
任何显著衰减都必须归因到数值粘性/分辨率，不能冒充物理阻尼。

## ② 方法与实现

* **求解器不变**：仍是 `src/tensorsph/wave_flume.py` 的 WCSPH 核心（密度求和 + Tait 状态方程
  + 对称压力梯度 + Morris 粘性 + Adami 壁面压强外推），本单不改算子。
* **启停瞬态处理**（前轮卡点 §3 直接对症）：`settle_time` 窗内用
  `viscosity_settle = 0.02`、`xsph_settle = 0.1` 强阻尼（两个都是短波耗散器，率 ∝ k²，
  对 k≈π/dx 的粒子尺度抖动是每秒数千，对 k≈4.8 的波是每秒几十分之一），波场期回物理
  `ν = 1e-6`；传播期保留 δ-SPH 密度扩散 `density_diffusion = 0.1`（Antuono et al. 2010）。
  代码指针：`wave_flume.py:run_case` 里 `physical_viscosity` / `xsph_now` 的窗内切换。
* **评级链（本单新增）**：`src/tensorsph/wave_flume_grade.py`
  * `wave_height()`：主估计 = 已知频率谐波最小二乘幅值（`harmonic_amplitude`，噪声按 √(2/N) 平均），
    交叉校验 = 零穿越波-波 `crest-trough` 均值、`2√2 σ_eta`、谱峰幅值；
  * `attenuation_rate()`：`ln(H/2)` 对 `x` 最小二乘，附逐对速率（单个坏探头不会被平均掉）；
  * `laminar_boundary_layer_damping()`：G2 的文献锚（闭式，含 `k`、`c`、`c_g` 中间量）；
  * `grade_regular_record()`：把 `run.py` 的 JSON 记录整体判级（G1 逐探头、G2、色散双探头
    波长 vs `ω²=gk tanh kh`、反射系数、干点比例），返回 `passed`。
* **端到端**：`benchmarks/wave_flume/run.py`（既有）→ `benchmarks/wave_flume/analyse.py`（新，
  逐算例出表 + 分辨率阶梯图 + 自由面剖面图，写 `docs/reports/*.json`）。

## ③ 参数表（分级工况，已递交集群，**尚未取回**）

| 量 | 值 | 无量纲 |
|---|---|---|
| 水深 h | 0.25 m | — |
| 波高 H | 0.04 m | H/h = 0.16；H/λ = 0.0307 |
| 周期 T | 1.0 s | kh = 1.2046；kh 属过渡水深 |
| 波长 λ | 1.30383 m（`ω²=gk tanh kh`） | — |
| Ursell 数 | H λ²/h³ = 4.35 | 弱非线性 |
| 声速 c0 | 30 m/s | M = c/c0 = 0.043；压缩性偏差 O(gh/c0²) = 0.27% |
| 分辨率档 | dx = 0.0125 m（D20，H/dx = 3.2）；dx = 0.0078125 m（D32，H/dx = 5.12） | 2 档 |
| 时间步 | dt = min(4e-4, 0.25h/dx/c0) | 1.04e-04 s（D20）/ 6.51e-05 s（D32） |
| 粘度/密度扩散 | ν = 1e-6 m²/s（波场期）；δ-SPH = 0.1 | Re = c h/ν ≈ 3.3e5 |
| 海绵层 | L_s = 0.98 m，强度 4.0，指数 3.0 | L_s/λ = 0.75 |
| 模拟时长 | 8.0 s（settle 1.0 s 另计） | 8 个周期 |

## ④ 对照表（**核心缺口：未取回实测，逐项打"未测"**）

| 判据项 | 文献基准 | 实测 | 误差 | 打勾/打叉 |
|---|---|---|---|---|
| G1 波高（D20，H/dx=3.2） | H_target = 0.0400 m | **未测**（作业 cj1147 在队列） | — | **未判** |
| G1 波高（D32，H/dx=5.12） | H_target = 0.0400 m | **未测**（作业 cj1148 在队列） | — | **未判** |
| G2 衰减 μ | μ_theory = 1.16847e-03 1/m（跨 2 m 0.233%） | **未测** | — | **未判** |
| 色散（双探头 λ vs 解析） | λ = 1.30383 m | **未测** | — | **未判** |
| 反射系数 | 无验收门（海绵层设计目标 ≲10%） | **未测** | — | 未判 |

**已实测、且可以复算的替代证据（不是本锚点判据，但证明"尺子"是准的）**：

| 项 | 方法 | 结果 | 结论 |
|---|---|---|---|
| 谐波幅值估计器 | `tests/test_wave_flume_grade.py::test_wave_height_recovers_the_amplitude_of_a_sinusoid` | H = 0.050 → 恢复误差 < 0.1% | 通过 |
| 幅度估计抗噪 | 同文件 `test_smoothing_removes_gauge_jitter...`（噪声 = 50% 半波高） | 误差 < 3% | 通过 |
| 衰减系数估计器 | `test_attenuation_recovers_a_known_decay_coefficient`（μ = 0.05 1/m） | 相对误差 < 1e-9 | 通过 |
| 整链判级 | 合成记录（decay=0.03）跑 `analyse.py` | 逐探头误差 −2.7%/−3.2%/−3.8%/−4.4%，`passed=True`；衰减拟合 3.00e-02 1/m（真值同） | 通过 |
| 整链判负 | `test_grade_of_a_synthetic_decaying_record_fails_the_height_gate` | 强衰减记录被判 `passed=False` | 通过 |
| 物理参考 μ 的量级 | `laminar_boundary_layer_damping(1.0, 0.25)` | 1.168e-03 1/m，且随深度 0.1→1.0 m 下降 >20 倍 | 通过 |

## ⑤ 收敛性/参数敏感性

* 分辨率阶梯：两档（H/dx = 3.2 与 5.12）已配置进作业，`analyse.py` 会自动出
  `docs/figures/wave_flume/ladder.png`（逐探头误差 vs H/dx，10% 门画成红线）。
* 参数点：`H/h = 0.16` 单点，`T = 1.0 s` 单点 —— **未做**第二参数点，属本单未完成部分。
* 敏感性分析未做（无实测数据可分析），不允许以合成信号冒充。

## ⑥ 失败模式与工件（fail_stage / failure_reason）

* **fail_stage**: `case-run-data-not-collected`（评级链已建成并通过合成信号验证；分级工况的
  实测波高未取回）。
* **failure_reason**: ①集群两个生产作业 `cj1147`（D20，8 核）与 `cj1148`（D32，16 核）
  递交后排队 340+ 项，本次会话结束前未派发；②本机（99）load average 380–420（128 核）下
  本地保底运行 `runs/wf_local_D20.json`（D20 等效）与 `runs/wf_quick.json`（低分辨率冒烟）
  在会话预算内未写出 JSON；③`oss_inventory` / `oss_solutions` / `negative_knowledge_ledger`
  本角色 `permission denied` 且无对应 API 写入口，方法选型正文改落
  `docs/wave_flume_method_research.md`（§4 给出 feature_slug/tag 字段映射，待有权限的席位代录）。
* **repro_cmd**（取回数据即可判级）：
  ```
  # 生产工况（集群，已递交 cj1147 / cj1148；本地等价命令去掉 --threads 16）
  PYTHONPATH=src python benchmarks/wave_flume/run.py --case regular --depth 0.25 \
    --depth-cells 20 --height-ratio 0.16 --wave-period 1.0 --simulate-time 8.0 \
    --settle 1.0 --viscosity-settle 0.02 --xsph-settle 0.1 --density-diffusion 0.1 \
    --viscosity 1e-6 --sponge-strength 4.0 --snapshots 4 --threads 8 --out runs/wf_D20.json
  # 评级（G1/G2/色散/反射 + 图）
  PYTHONPATH=src python benchmarks/wave_flume/analyse.py --record runs/wf_D20.json \
    --record runs/wf_D32.json --start 4.0 --out docs/reports/wave_flume_report.json \
    --figures docs/figures/wave_flume
  ```
* **工件**：`src/tensorsph/wave_flume_grade.py`、`tests/test_wave_flume_grade.py`（13 passed）、
  `benchmarks/wave_flume/analyse.py`、`docs/wave_flume_method_research.md`、
  `/tmp/figs/series_syn.png`（合成记录评级图，估计器验证用）。

## 误差判定（关键量 vs 基准；显式回答"是否 <3%"）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 波槽波高（G1，本锚点判据 10%） | H_target = 0.04 m | **未测** | **无法回答**（数据未取回，不声明） |
| 波高估计器（合成信号） | 解析 H = 0.05 m | 误差 < 0.1% | **是** |
| 衰减系数估计器 | 解析 μ = 0.05 1/m | 相对误差 < 1e-9 | **是** |
| 物理衰减参考量级 | 闭式解 | 1.168e-03 1/m | 是（闭式自洽，含 sqrt(ν) 与深度定标检验） |
