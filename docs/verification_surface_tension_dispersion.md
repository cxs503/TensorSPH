# 表面张力色散验证：CSF 算子的毛细波与 Lamb 模态（TensorSPH, m14536）

**结论先行（honest FAIL）**：TensorSPH 的粒子 CSF 表面张力算子**没有**达到任务判据
C1（毛细波 < 5 %）、C2（Lamb < 3 %）、C3（收敛阶 ≥ 2）；C4（σ 线性）与 C5（pytest）通过。
本轮把失败**量化成三个可复算的缺陷**，并证明瓶颈在**表面法向估计器**而不在散度算子：

1. **表面 delta 函数欠积分**：闭合液滴上 `sum_i delta_s,i V_i / A = 0.7135 ± 0.0006`
   （要求 1.0，稳定复现，`local_range` 可修到 0.948）；自由表面上该积分**没有稳定值**
   （0.93 / 1.23 随分辨率与点阵相位变，`legacy_std = 0.24`）——自由表面的 delta 归一化
   在现分辨率下不是一个良定义的常数。
2. **法向偏置**：`-grad c/|grad c|`（含 2 次 Shepard 平滑）在近平自由表面上偏 **5.6°**
   （p95 7.8°）；`kappa = div n` 把角误差放大 `1/h`，于是曲率噪声底
   （0.28–0.50 1/m）**大于**波自身曲率（0.126 1/m）2–4 倍 → 模态刚度被相位噪声支配
   （`std/|mean| = 0.4–1.0`，个别相位甚至变号）。
3. **对照实验证明散度算子无罪**：把**解析法向**喂给同一个 MLS 散度，波谷曲率
   `-0.105`（精确 `-0.126`，误差 17 %，**符号正确**）；用 CSF 法向则 `+0.014`（符号丢失）。

> 前一轮（m13895）的 C1/C2 数字不可用：几何（4 个自由表面）、`k` 扫描（尺度不变=三份拷贝）、
> 面力符号约定、相位平均各有结构性缺陷（§5 D1–D4），本轮还修正了一个**自引入**的
> 周期镜像排序缺陷（D5：主拷贝须排第一位，否则场量与位置错配，直接翻转测得的波刚度符号）。
> 旧几何保留为可复现的 A/B 开关（`--defective-control`）。

---

## 1. 物理背景、几何与参数

### 1.1 两条解析参考

| 模态 | 解析色散/频率 | 出处 |
|---|---|---|
| 深水毛细波（2-D，无粘） | `omega^2 = sigma k^3 / rho` | Lamb 1932, *Hydrodynamics* 6th ed., Art. 267 |
| 2-D 液滴第 `n` 阶振荡 | `omega^2 = n (n^2-1) sigma / (rho R^3)` | Rayleigh 1879；Lamb Art. 275 |

### 1.2 测的到底是什么（线性响应 / 离散色散）

线性区恢复力只有界面法向应力：波 `Delta p = sigma k^2 a cos(kx)`，液滴
`Delta p = sigma (n^2-1) eps cos(n theta)/R`；惯性取精确势流值（波 `rho/k` per area，
液滴 `rho R/n`）。因此

```
omega_measured^2 / omega_theory^2 = Delta p_measured / Delta p_theory
```

即**频率比 = 离散 CSF 算子的模态法向应力比**。这是对*算子*的测量（离散色散），
不是时间推进：`docs/verification_capillary_wave_open_surface.md`（m13812）已记录同一算子的
时间推进不可用（伪加速度 ≈ 波自身 `omega^2 A` 的 23 倍），本轮的审计给出了该现象的成因
（曲率在高频噪声下失真 + delta 欠积分）。

**符号约定**：`t = f·n` 为沿外法向的面力密度，CSF 给出 `t = -sigma kappa delta_s`；
稳定（恢复性）模态要求 `t` 与位移反号 ⇒ 下表「刚度比」应为**正**。

### 1.3 几何与参数表

| 参数 | 值 | 说明 |
|---|---|---|
| `rho0` | 1000 kg/m³ | 水 20 °C |
| `sigma` | 0.072 N/m | 水/空气 20 °C |
| `h/dx` | 1.3 | 核支持 2h |
| 波幅 `a/lambda` | 0.02 | 线性区 |
| 液滴 `eps`, `R` | 0.10, 0.02 m | `r(theta)=R(1+eps cos n theta)` |
| 波域长 `L` | `2 pi` 固定 | `dx = L/ppw` 与 k 无关 → `k=1,2,3` 是**独立**题 |
| 波域深 | `0.8 lambda` | 深水（`kH ≈ 5`） |
| 底面 | 底部约 3.5 排冻结为固壁 | 流体下方满核支持（c=1，实测 min=1.0），消掉「第二个自由表面」 |
| x 方向 | 周期（3 镜像，主拷贝排第一） | 消掉左右两个自由侧壁 |
| 亚格子相位 | `shifts x shifts` 集合平均 + 标准差 | 把点阵相位噪声分离出来（本轮主角） |

---

## 2. 方法、流程与工具链

1. **实现（库）** `src/tensorsph/surface_tension_dispersion.py`
   * 几何：`hex_lattice`（六方点阵，最近邻恰为 `dx`）、`wave_geometry`、`drop_geometry`；
   * 周期化：`periodic_images`（x 三镜像，**主拷贝第一位**）+ `csf_state`（把 `CSFState` 切回主拷贝）；
   * 模态投影：`wave_modal_stiffness` / `drop_modal_stiffness`（`t = -sigma kappa delta_s`，
     `V = m/rho0`，壁面按 mask 剔除）；
   * 收敛阶 `convergence_order`；σ 线性 `sigma_sensitivity`；
   * 审计：`curvature_audit`（波谷/波峰/平界面噪声底）、`normal_audit`（法向角误差 +
     解析法向对照）、`delta_measure_audit`（`sum delta_s V / A`）、`curvature_noise_scan`（噪声底的 h 依赖）；
   * 稳定性：`capillary_dt_limit(h, sigma, rho0, cfl, k_max)`（`dt = cfl/sqrt(sigma k^3/rho0)`），
     被 `threed/solver.py::acoustic_dt`（取栅格尺度 `k_max = pi/h`）与
     `surface_tension.py::capillary_dt_limit`（保持历史约定 `k_max = 1/h`，位对位兼容）调用
     ——真正的单点真理 + 两个真实调用方（见 §8）。
2. **入口** `scripts/run_surface_tension_dispersion.py`（一次跑完 C1–C4 + 4 项审计 + 图 + 出处），
   另支持 `python -m tensorsph.surface_tension_dispersion`。
3. **旧入口** `benchmarks/surface_tension_dispersion/dispersion.py` 改为薄转发（防止旧缺陷代码被复用）。
4. **工具链** Python 3.11.6 / PyTorch 2.7.1 / loongarch64 / 6 线程；每点 2×2 相位集合；全程约 40–60 s。

---

## 3. 计算结果（零手抄：全部来自 `runs/surface_tension_dispersion/report.json`）

### 3.1 判据总表

| 判据 | 要求 | 实测 | 判定 |
|---|---|---|---|
| C1 毛细波（k=1,2,3 最差档） | < 5 % | **198.4 %**（选项筛选后最优仍 **93.2 %**） | **FAIL** |
| C2 Lamb（l=2,3,4 最差档） | < 3 % | **19.32 %**（最优选项 **4.85 %**） | **FAIL** |
| C3 收敛阶（6 条阶梯最小值） | ≥ 2 | **−4.14**（波 −0.14/−0.21/−0.72；Lamb −4.14/−0.04/−0.10） | **FAIL** |
| C4 σ 敏感性 | < 5 % | **0.000 %**（`dp ∝ sigma^1.000`） | **PASS** |
| C5 pytest | ≥ 10 用例全绿 | 29 用例（19 本能力 + 10 算子级），另 42 个相邻回归用例全绿 | **PASS** |

### 3.2 毛细波（L=2π 固定；默认 legacy delta + fit_quadratic 曲率）

| k | ppw | 每波长格数 | ref ω [rad/s] | 刚度比 ± 相位标准差 | ω 误差 | S/A |
|---|---|---|---|---|---|---|
| 1 | 24 | 24 | 8.485e-3 | −1.319 ± 1.544 | −168.0 % | 0.910 |
| 1 | 32 | 32 | 8.485e-3 | −5.904 ± 2.140 | −338.8 % | 1.315 |
| 1 | 48 | 48 | 8.485e-3 | −2.637 ± 2.783 | −198.4 % | 0.931 |
| 2 | 24 | 12 | 2.400e-2 | −0.089 ± 0.381 | −114.9 % | 0.785 |
| 2 | 32 | 16 | 2.400e-2 | −1.353 ± 0.491 | −214.3 % | 0.958 |
| 2 | 48 | 24 | 2.400e-2 | −0.557 ± 0.794 | −141.2 % | 0.803 |
| 3 | 24 | 8 | 4.409e-2 | +0.144 ± 0.203 | −79.0 % | 0.749 |
| 3 | 32 | 10.7 | 4.409e-2 | −0.420 ± 0.164 | −163.5 % | 0.836 |
| 3 | 48 | 16 | 4.409e-2 | −0.282 ± 0.282 | −137.5 % | 0.753 |

**读法**：`std` 与 `|mean|` 同量级（0.2–1.0 倍），即单次实现（一个点阵相位）给出的刚度
不可复现，符号也不稳定；这不是"算子把波放大了"，而是**该模态的曲率信号被点阵相位噪声淹没**
（见 §3.5）。S/A 落在 0.75–1.32，围绕 1 抖动，与 §3.6 的 0.70 之系统性偏差叠加。

### 3.3 Lamb 模态（闭合强曲率界面，R = 0.02 m）

| l | R/dx | ref ω [rad/s] | measured ω | 误差 | 刚度比 ± 相位标准差 | S/A |
|---|---|---|---|---|---|---|
| 2 | 16 | 7.348 | 7.325 | −0.31 % | 0.9942 ± 0.0424 | 0.7022 |
| 2 | 24 | 7.348 | 6.962 | −5.26 % | 0.8980 ± 0.0430 | 0.7100 |
| 2 | 32 | 7.348 | 6.992 | −4.85 % | 0.9060 ± 0.0467 | 0.7135 |
| 3 | 16 | 14.697 | 13.336 | −9.26 % | 0.8234 ± 0.0167 | 0.7166 |
| 3 | 24 | 14.697 | 13.272 | −9.70 % | 0.8155 ± 0.0107 | 0.7242 |
| 3 | 32 | 14.697 | 13.304 | −9.48 % | 0.8194 ± 0.0039 | 0.7285 |
| 4 | 16 | 23.238 | 19.058 | −17.99 % | 0.6727 ± 0.0148 | 0.7214 |
| 4 | 24 | 23.238 | 19.020 | −18.15 % | 0.6699 ± 0.0066 | 0.7310 |
| 4 | 32 | 23.238 | 18.749 | −19.32 % | 0.6510 ± 0.0030 | 0.7342 |

相位标准差随 R/dx 下降（0.047 → 0.003）⇒ 残差是**系统性**的。误差随 `l` 单调变大
（−0.3 % → −19.3 %），而 S/A 同时从 0.702 升到 0.734：`l=2` 处曲率偏高（`kappa_ratio ≈ 1.42`）
恰好抵消了 delta 的欠积分，`l=4` 处曲率偏低（`kappa_ratio ≈ 0.93`）只剩欠积分的损失。
**这是"错误相互抵消"而不是算子正确**。

### 3.4 选项筛选（最细档：k=1/ppw=48，l=2/R/dx=32）

| delta 归一化 | 曲率格式 | 波误差 | 波 S/A | Lamb 误差 | Lamb S/A |
|---|---|---|---|---|---|
| legacy | mls | **+93.2 %** | 0.931 | +5.25 % | 0.714 |
| legacy | fit_quadratic | −198.4 % | 0.931 | **−4.85 %** | 0.714 |
| local_range | mls | +200.3 % | 1.651 | +19.44 % | 0.948 |
| local_range | fit_quadratic | −245.9 % | 1.651 | +9.48 % | 0.948 |

没有任何出厂组合达标。`local_range` 把液滴 S/A 从 0.714 修到 0.948（好），却把自由表面
S/A 推到 1.651（坏）；且同一归一化下换曲率格式，波误差从 +93 % 翻到 −198 %——
**波侧的主导误差不是 delta，而是曲率的相位噪声**。

### 3.5 曲率审计（核心证据：`curvature_audit` / `curvature_noise_scan` / `normal_audit`）

| ppw | dx | 平界面噪声底 RMS [1/m] | 信号 `a k²` [1/m] | 信噪比 | 波谷 κ（精确 −0.126） | 波峰 κ（精确 +0.126） |
|---|---|---|---|---|---|---|
| 24 | 0.262 | 0.277 | 0.126 | 0.45 | +0.050 | −0.017 |
| 32 | 0.196 | 0.502 | 0.126 | 0.25 | +0.324 | −0.024 |
| 48 | 0.131 | 0.405 | 0.126 | 0.31 | +0.014 | −0.044 |

* 平界面（精确 κ=0）上的曲率 RMS 在所有分辨率都 ≥ 2× 波信号，信噪比 0.25–0.45；
  加密不改善（0.277 → 0.502 → 0.405，与 `1/h` 同量级抖动）⇒ 这是**结构性**精度上限。
* `normal_audit`（ppw=48，2×2 相位）：
  * CSF 法向 vs 解析法向：**5.64° 平均 / 7.83° p95**；
  * 同一 MLS 散度，用 **CSF 法向**：κ_波谷 = **+0.0136**（精确 −0.126，符号丢失）；
  * 同一 MLS 散度，用**解析法向**：κ_波谷 = **−0.1046**（精确 −0.126，误差 17 %，**符号正确**）。
  ⇒ **散度算子无罪，法向估计器是瓶颈**（`normal_error_dominates = True`）。

### 3.6 表面 delta 函数的积分（`delta_measure_audit`，2×2 相位集合）

| 几何 | legacy `2|grad c|` | 相位 std | local_range | 相位 std | 要求 |
|---|---|---|---|---|---|
| 周期槽道毛细波（单自由表面，ppw=48） | 0.931 | 0.235 | 1.651 | 0.521 | 1.0 |
| 周期槽道毛细波（单自由表面，ppw=16） | 1.225 | 0.153 | 1.740 | 0.003 | 1.0 |
| 液滴 l=2（闭合凸界面，R/dx=32） | **0.7135** | **0.0006** | **0.9479** | 0.006 | 1.0 |

* **闭合液滴**：欠积分 29 %，且相位 std 仅 6e-4 ⇒ 这是一个**稳定、可校正**的系统性常数
  （`omega` 偏低 `sqrt(0.7135) ≈ 0.845`，即 −15.5 %）。`local_range` 把它修到 0.948。
* **自由表面**：同一条规则在 0.93–1.65 之间漂移（相位 std 0.15–0.52），**没有定义良好的
  离散积分**——这才是自由表面比液滴更难的定量表述。
* 既有台账条目 `colour_surface.delta_s_integral`（`benchmarks/verification_ledger/ledger.json`，
  grade `WITHHELD`，measured `0.7545` @ dx = 3 mm）与本轮液滴值 0.7135（@ dx = 0.625 mm）
  **独立互证**（不同代码路径、不同分辨率、同向且同量级）。

### 3.7 A/B：旧几何（闭环块，4 个自由表面）vs 修正几何（ppw = 48）

| k | 旧几何 刚度比 ± std | 旧几何 ω 误差 | 旧几何 S/A | 修正几何 刚度比 ± std | 修正几何 ω 误差 | 修正几何 S/A |
|---|---|---|---|---|---|---|
| 1 | −10.045 ± 0.238 | −416.9 % | 2.037 | −2.637 ± 2.783 | −198.4 % | 0.931 |
| 2 | −4.333 ± 0.351 | −308.0 % | 1.465 | −0.557 ± 0.794 | −141.2 % | 0.803 |
| 3 | −2.558 ± 0.149 | −259.9 % | 1.272 | −0.282 ± 0.282 | −137.5 % | 0.753 |

旧几何的 S/A 被底面与两侧壁的伪界面抬高到 1.27–2.04；修正后回到 0.75–0.93（与液滴 0.71 同量级）。
**几何修正是必要的**（否则测的是伪界面），但**不足以**让判据通过。

---

## 4. 结果图

`runs/surface_tension_dispersion/figures/`（4 张，随报告一起提交）

| 图 | 内容 |
|---|---|
| `capillary_wave_error.png` | k=1/2/3 的 ω 误差随分辨率变化（含 ±5 % 判据带） |
| `lamb_stiffness.png` | l=2/3/4 刚度比随 R/dx（误差棒=相位集合 std，虚线=精确值 1） |
| `curvature_audit.png` | 波谷/波峰/平界面噪声底：实测 vs 精确曲率 |
| `delta_measure.png` | `sum delta_s V/A` 在两种拓扑 × 两种归一化下（要求 1.0） |

---

## 5. 注意点：修正的缺陷、根因、局限

### 5.1 结构性缺陷（全部有 A/B 复现开关）

* **D1 闭环块几何**（旧）：上方波面 + 自由底面 + 两个自由侧壁，共 4 个自由表面，x 非周期。
  每个界面都产生自己的 `-sigma kappa delta_s n`，投影到 `cos(kx)` 上比波信号还大
  （刚度 −10.0、S/A 2.04）。修正：x 周期 + 底部冻结固壁。
* **D2 `k` 扫描退化**（旧）：`dx = lambda/ppw`（ppw 固定）⇒ `k=1,2,3` 是同一几何相似问题，
  三行数字到 15 位有效数字相同。修正：`L = 2 pi` 固定，`dx = L/ppw` 与 k 无关。
* **D3 面力符号约定错**（旧）：旧投影用 `t = +sigma kappa delta_s`（物理量是负号），
  与曲率误差的符号相互抵消，产出"看起来还行"的正刚度，掩盖了真实误差。修正：统一
  `t = -sigma kappa delta_s`。
* **D4 相位未平均**（旧）：单相位 + `shifts=2` 时点阵相位只取一个。修正：集合平均 + 报 std。
* **D5 周期镜像排序缺陷（本轮自引入并修正）**：`periodic_images` 首版把 `-L` 拷贝排在
  第一位，而 `csf_state` 用 `[:n]` 切回主拷贝 ⇒ **场量与位置错配一个周期相位**，直接把波
  刚度符号翻转（同一配置修正前 −2.06、修正后 +0.435，`curvature_audit` 波谷从 +1.24 变为
  −0.128）。已加回归测试 `test_periodic_images_keep_the_primary_copy_first`。

### 5.2 根因与修复路径

`kappa = div n`，`n = -grad c/|grad c|`。在**单侧自由表面**上 MLS 梯度有 5.6° 系统偏置，
散度把它放大 `1/h`，于是曲率噪声底 > 波信号（§3.5）。修复路径（本轮未完成）：

1. **几何法向**：从界面点集提取参数化/面元（marching-squares）或局部二次曲面拟合得到法向，
   而不是色梯度；本轮试过局部 PCA 平面拟合，在当前 5 % 阈值下的界面点集上角误差
   18–58°（**更差**，因为点集里混入本体粒子），需要先解决界面带提取。
2. **delta 归一化**：把 `delta_s` 归一化到 `sum delta_s V = A`。液滴侧差值稳定（0.7135），
   单这一项会把 `l=2` 从 −0.31 % 推到约 +16 %（方向相反），说明必须与曲率的
   `kappa_ratio ≈ 1.42` 一起修，否则只是把误差挪个位置。
3. 两项都修完后，`C1/C3` 才有望进入 5 % / 二阶；仅修一项会打破当前的"误差抵消"。

### 5.3 局限

* 只测线性响应/离散色散（无粘、无有限幅非线性）；未做时间推进验证（m13812 已判不可用）。
* `k=3 / ppw=24` 时每波长仅 8 个粒子，已接近可分辨极限，该行只作趋势参考。
* 液滴 `l=3/4` 的相位 std < 1e-3，说明误差系统化；但未把误差进一步分解到
  `delta_s` 与 `kappa` 的逐项贡献（只给了两者的乘积与各自诊断值）。
* 相位集合只取 2×2（4 个相位）；`std` 本身有 ~30 % 的抽样不确定度。

### 5.4 适用边界

* `kappa h ≳ 0.1` 的强曲率界面（液滴 l≥2、`kappa R = O(1)`）是本算子可用的区间（误差 5–19 %）。
* `kappa h ≲ 0.05` 的弱曲率自由表面（毛细波 k=1..3）**超出当前算子能力**。

---

## 6. 误差判定（逐项，不粉饰）

| 关键量 | 基准 | 实测 | 相对误差 | 是否 < 3 % |
|---|---|---|---|---|
| Lamb l=2 频率（R/dx=32） | 7.348 rad/s | 6.992 rad/s | 4.85 % | **否** |
| Lamb l=3 频率（R/dx=32） | 14.697 rad/s | 13.304 rad/s | 9.48 % | **否** |
| Lamb l=4 频率（R/dx=32） | 23.238 rad/s | 18.749 rad/s | 19.32 % | **否** |
| Lamb 最优（legacy+fit_quadratic, l=2, R/dx=32） | 7.348 rad/s | 6.992 rad/s | 4.85 % | **否（差 1.85 pt）** |
| Lamb 单点最好（l=2, R/dx=16） | 7.348 rad/s | 7.325 rad/s | 0.31 % | 是（**但 l=3/4 与更细网格不成立**，见 §3.3） |
| 毛细波 k=1（ppw=48） | 8.485e-3 rad/s | −1.753e-2 rad/s（刚度比 −2.64） | 198.4 % | **否** |
| 毛细波最优选项（legacy+mls, k=1, ppw=48） | 8.485e-3 rad/s | — | 93.2 % | **否** |
| 收敛阶（C3） | ≥ 2 | −0.14 … −4.14 | — | **否** |
| 液滴 delta 积分（l=2, R/dx=32） | 1.0 | 0.7135 ± 0.0006 | −28.7 % | **否** |
| σ 线性（C4） | < 5 % | 0.000 %（`dp ∝ sigma^1.000`） | — | **是** |

---

## 7. 复现

```bash
# 全套（约 40–60 s，6 线程）：报告 + 基准表 + 图 + 出处
PYTHONPATH=src python scripts/run_surface_tension_dispersion.py \
    --out-dir runs/surface_tension_dispersion --threads 6

# 库内入口（等价）
PYTHONPATH=src python -m tensorsph.surface_tension_dispersion \
    --ppw 24,32,48 --ks 1,2,3 --modes 2,3,4 --rdx 16,24,32 --shifts 2 \
    --out runs/surface_tension_dispersion/report.json

# 快速自检（约 20 s）
PYTHONPATH=src python -m tensorsph.surface_tension_dispersion --quick

# 测试（19 用例）
PYTHONPATH=src python -m pytest tests/test_surface_tension_dispersion.py -q

# 旧几何 A/B 对照（缺陷复现）
PYTHONPATH=src python -m tensorsph.surface_tension_dispersion --defective-control --quick
```

---

## 8. 审计链

| 项 | 值 |
|---|---|
| 报告 | `runs/surface_tension_dispersion/report.json`（sha256 见 `meta.json`） |
| 基准表（ref vs got） | `runs/surface_tension_dispersion/benchmark.json` |
| 出处 | `runs/surface_tension_dispersion/meta.json`（git rev、命令、Python/torch、平台、线程、墙钟、两个 sha256） |
| 摘要 | `runs/surface_tension_dispersion/summary.md` |
| 图 | `runs/surface_tension_dispersion/figures/*.png`（4 张） |
| 实现 | `src/tensorsph/surface_tension_dispersion.py` |
| 入口 | `scripts/run_surface_tension_dispersion.py`、`python -m tensorsph.surface_tension_dispersion` |
| 集成调用方 | `src/tensorsph/threed/solver.py::acoustic_dt`、`src/tensorsph/surface_tension.py::capillary_dt_limit` |
| 测试 | `tests/test_surface_tension_dispersion.py`（19 用例）+ `tests/test_csf_dispersion.py`（10 用例），共 29 用例全绿；相邻回归 `test_capillary_wave / test_csf_consistent / test_threed_operators` 42 用例全绿 |
| 交叉印证 | `benchmarks/verification_ledger/ledger.json` → `colour_surface.delta_s_integral`（0.7545 @ dx=3 mm） |

> 复算说明：`report.json` 中每个数字都由 `scripts/run_surface_tension_dispersion.py` 写入，
> 无手抄；`meta.json` 记录该报告的 sha256，改一个字节即可被发现。