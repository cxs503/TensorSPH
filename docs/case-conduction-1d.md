# CASE-SPH-CONDUCTION-1D — 一维瞬态热传导 SPH 导热算子 vs erfc 解析解

**算例锚点**：`CASE-SPH-CONDUCTION-1D` ｜ 仓库 TensorSPH ｜ 产物：
`benchmarks/case_conduction_1d/`（`run.py` / `result.json` / `profile.png` / `convergence.png`）、
`src/tensorsph/heat_conduction.py`、`tests/test_conduction_contact_1d.py`。
本文所有数字由 `run.py` 程序化写入 `result.json` 后注入，零手抄。

## ① 问题与 ref 出处

两块半无限平板（`x<0` 温度 `T_hot`，`x>0` 温度 `T_cold`）在 `t=0` 理想热接触，
温度场自相似：

```
T(x,t) = T_cold + (T_hot - T_cold)/2 · erfc(η),      η = x / (2√(αt))
```

- **主 ref**：Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., Oxford
  University Press, 1959, Ch. II（无限/半无限固体中的线性热传导），§2.6「两块初始
  均匀温度不同的半无限介质接触」——即本算例的 erf/erfc 相似解。
- **次级 ref（SPH 算子来源）**：Cleary & Monaghan (1999) *Conduction Modelling Using
  Smoothed Particle Hydrodynamics*, J. Comput. Phys. **148**:227–264,
  doi:10.1006/jcph.1998.6118。其书目字段（刊名/卷/页/年/作者/DOI）本单已用
  Crossref API **程序化核对**（见 `result.json.environment.crossref_verified`）。
- **零手抄声明（诚实边界）**：ref 数值不是从书中表格抄来的，而是由
  `tensorsph.heat_conduction.contact_temperature` 按闭式解**计算**得出；教科书文献
  本身在沙箱内不可访问，因此**页码/公式编号未经机器核对**——这一点如实登记为局限，
  并用三条独立证据支撑公式本身：(a) Crank–Nicolson 有限差分数值解（完全不共享代码
  路径）复核，见 ④；(b) 极限值 `x→±∞` 与 `x=0` 处 `(T_hot+T_cold)/2`，见测试；
  (c) 自相似性不变式（同一 η 不同 t 温度一致），见测试。

## ② 方法与实现（复用现有能力，不另起炉灶）

- 离散格式 = 仓库既有 **Cleary–Monaghan SPH 导热算子**
  （`src/tensorsph/ice_thermal/sph_thermal.py: conduction_rhs`），本单**未改物理**，
  仅给它加了一个可选参数 `fv`（几何冻结时复用 `V_j F_ij`，避免每步重建 O(N²)）。
- 在均匀格点上，M4 核 `h=1.3dx` 的支撑 `2h=2.6dx` 使得该算子**精确**退化为对称
  5 点模板（偏移 `m=1,2`），`O(N)` 每步；`stencil_vs_dense = 6.82e-12`
  （对随机场与稠密 pair-sum 之差，机器精度），并有同配置的稠密路径对照
  `dense_vs_fast = 0.0e+00`、两者误差
  `7.509e-05` vs `7.509e-05`。
- 两端远场：仓库既有镜像 ghost Dirichlet（`T_ghost = 2T_wall - T_mirror`，
  `run_stefan_sph` 同法）；墙位于 4.243 个相似长度处，域加倍检验差值
  `delta = 0.0e+00`（bitwise 相同）→ 边界不是误差项。
- 扩散率标定：仓库 `lattice_diffusion_factor`（3 个最长周期模的平均）与新增的
  `lattice_symbol_ratio`（同一符号的 `k→0` 极限）。两种标定都跑，误差都记账。
- 时间积分：显式 Euler，`dt = 0.2 dx²/α`（另做 Courant 敏感性，见 ⑤）。
- 独立参照：`fd_contact_reference` 用 Crank–Nicolson + `scipy.linalg.solve_banded`
  解同一 PDE（不共享 SPH 任何代码）。

## ③ 参数表（无量纲数完整）

| 量 | 值 | 说明 |
|---|---|---|
| α = k/(ρ c_p) | 1.0 m²/s | ρ=c_p=1 ⇒ k=1.0 W/(m·K) |
| T_hot / T_cold | 1.0 / 0.0 K | ΔT = 1.0 K（误差分母） |
| 参考长度 l | 1.0 m | 无量纲化用 |
| 相似/Fourier 时间 t* = αt/l² | 0.50 | 算例规定值 |
| 物理终点时间 t | 0.50 s | t = t* l²/α |
| 热渗透深度 2√(αt) | 1.4142 m | 相似长度尺度 |
| 计算域 | x ∈ [-6.0, 6.0] m | 两端镜像 Dirichlet |
| 墙处相似变量 η_wall | 4.243 | erfc(4.24)~1e-9，远场无扰动 |
| 误差评估区 | |x| ≤ 4 m | η ≤ 2.83 |
| 核 / 光滑长度 | M4 三次样条，2h 支撑，h = 1.3dx，η_reg = 0.01h | 模板偏移 m = [1, 2] |
| 主算例分辨率 | dx = 0.01 m，N = 1200，dt = 2e-05 s，25000 步 | 域 12 m |
| 显式 Courant 数 | 0.2（dt·α/dx²） | 稳定性上限经扫描确认 |

## ④ 对照表（ref vs got，逐项打勾/叉）

**T(x,t*) 剖面对照**（ref = 闭式 erfc；got = SPH，dx=0.01 m，k0 标定）：

| η | x [m] | T_ref [K] | T_SPH [K] | 相对误差 | 判定 |
|---|---|---|---|---|---|
| 0.00 | -0.0050 | 0.501995 | 0.50199472 | 0.000000 | 2.16e-08 | ✓ |
| 0.25 | +0.3550 | 0.361295 | 0.36129339 | 0.000001 | 1.40e-06 | ✓ |
| 0.50 | +0.7050 | 0.240405 | 0.24040298 | 0.000002 | 2.13e-06 | ✓ |
| 1.00 | +1.4150 | 0.078534 | 0.07853293 | 0.000001 | 1.31e-06 | ✓ |
| 1.50 | +2.1250 | 0.016793 | 0.01679325 | 0.000000 | 6.07e-08 | ✓ |
| 2.00 | +2.8250 | 0.002364 | 0.00236418 | 0.000000 | 1.48e-07 | ✓ |

全域（|x| ≤ 4 m，1200 个粒子上）最大值 **2.181e-06**，
RMS **1.057e-06**；界面温度 T(0) = 0.500000000000000
（解析 (T_hot+T_cold)/2 = 0.5，误差 2.2e-16）。

**是否 <3%？是。** 主指标 2.181e-06 = 目标值的
0.0073%（即比 3% 门槛小约 4 个数量级）。
对照三种标定：无标定 2.658e-03（0.266%）、
仓库 3 模标定 4.721e-05（0.0047%）、
k0 标定 **2.181e-06**。

**分辨率阶梯（两种标定）**

| 标定 | dx [m] | N | 步数 | α_eff/α 标定因子 | max 归一化误差 | RMS |
|---|---|---|---|---|---|---|
| lattice | 0.080 | 150 | 391 | 1.021816 | 9.388e-05 | 4.396e-05 |
| lattice | 0.040 | 300 | 1563 | 1.021816 | 2.153e-05 | 1.249e-05 |
| lattice | 0.020 | 600 | 6250 | 1.021816 | 4.104e-05 | 2.310e-05 |
| lattice | 0.010 | 1200 | 25000 | 1.021816 | 4.721e-05 | 2.606e-05 |
| k0 | 0.080 | 150 | 391 | 1.022232 | 1.410e-04 | 6.840e-05 |
| k0 | 0.040 | 300 | 1563 | 1.022232 | 3.518e-05 | 1.706e-05 |
| k0 | 0.020 | 600 | 6250 | 1.022232 | 8.773e-06 | 4.255e-06 |
| k0 | 0.010 | 1200 | 25000 | 1.022232 | 2.181e-06 | 1.057e-06 |
| k0 | 0.005 | 2400 | 100000 | 1.022232 | 5.324e-07 | 2.575e-07 |

**实测收敛阶**

| 标定 | dx 对 [m] | 观测阶 |
|---|---|---|
| lattice | 0.080 → 0.040 | 2.12 |
| lattice | 0.040 → 0.020 | -0.93 |
| lattice | 0.020 → 0.010 | -0.20 |
| k0 | 0.080 → 0.040 | 2.00 |
| k0 | 0.040 → 0.020 | 2.00 |
| k0 | 0.020 → 0.010 | 2.01 |
| k0 | 0.010 → 0.005 | 2.03 |

k0 标定下四段阶梯均为 **2.00–2.03**（二阶，与格点算子预期一致）；仓库 3 模标定在
细网格上停在 ~4.7e-5 的**平台**上——原因见 ⑤，是标定常数与 `k→0` 符号的 4.1e-4 差距
造成的残余扩散率，不是截断误差。

**独立参照（Crank–Nicolson vs 闭式 erfc）**

| N | dx [m] | 最大绝对温度误差 [K] |
|---|---|---|
| 601 | 0.0200 | 5.425e-06 |
| 1201 | 0.0100 | 1.357e-06 |
| 2401 | 0.0050 | 3.394e-07 |

这三组奇数网格在接触面都有一个节点。该节点用两侧初温的均值初始化后，网格间距每次减半，最大误差约缩小四倍。见[奇数网格接触导热验证记录](../benchmarks/verified/node_centered_contact.json)。

## ⑤ 收敛性 / 参数敏感性（≥2 项）

**(1) 核支撑 h/dx（两种标定 + 无标定，dx = 0.02 m）**

| h/dx | 无标定 max | 3 模标定 max | k0 标定 max | 3 模标定因子 |
|---|---|---|---|---|
| 1.1 | 1.289e-03 | 2.921e-05 | 3.723e-06 | 1.010467 |
| 1.3 | 2.652e-03 | 4.104e-05 | 8.773e-06 | 1.021816 |
| 1.5 | 1.534e-03 | 5.061e-05 | 1.296e-05 | 0.986990 |
| 1.7 | 2.262e-03 | 6.532e-05 | 1.938e-05 | 0.980955 |

要点：无标定误差 1.3e-3~2.7e-3（0.13%~0.27%，仍在 3% 内）；标定后再降 30~350 倍；
k0 标定在所有 h/dx 上最好（3.7e-6~1.9e-5）。标定因子随 h/dx 变号（1.1/1.3 时 >1，
1.5/1.7 时 <1），说明**不能把 1.0218 当常数硬编码**。

**(2) 时间步 Courant 数（dx = 0.02 m，k0 标定）**

| Courant | 步数 | max 归一化误差 |
|---|---|---|
| 0.05 | 25000 | 1.698e-05 |
| 0.10 | 12500 | 1.424e-05 |
| 0.20 | 6250 | 8.773e-06 |
| 0.30 | 4167 | 3.549e-06 |

Courant ≤ 0.3 全域稳定；误差随 dt 增大反而下降（显式 Euler 的时间误差与格点弥散部分
相消），0.2 落在平台区，选它作主算例。**注意**：这也说明本算例的误差不随时间步收敛
主导，因此「把 dt 调小」不是提高精度的有效手段。

**(3) 残余扩散率诊断（本单最实质的发现）**
把误差剖面回归到解析二阶导（线性热方程有 `∂T/∂α = t ∂²T/∂x²` 的精确关系）：
- 仓库 3 模标定残余 `Δα/α = 3.924e-04`，
  去除该项后余差仅 `6.13e-07`
  （= 原误差的 1.3%），
  即误差 99% 就是「多扩散了一点点」；
- 理论预示的差距：3 模平均 1.0218159 vs `k→0` 极限
  1.0222322，相对差 `4.073e-04` —— 与实测
  3.924e-04 相差约
  3.6%。
  两者吻合 ⇒ 平台被解释，不是拟合出来的。
- 换 k0 标定后残余降到 `-1.494e-05`，误差随 dx² 干净收敛。

格点符号 α_eff(k)/α 抽样：

| k·dx | α_eff/α |
|---|---|
| 0.005 | 1.0222287 |
| 0.010 | 1.0222175 |
| 0.025 | 1.0221397 |
| 0.050 | 1.0218619 |
| 0.100 | 1.0207516 |
| 0.200 | 1.0163254 |
| 0.400 | 0.9988575 |
| 0.800 | 0.9326317 |
| 1.200 | 0.8341277 |
| 1.600 | 0.7176346 |
| 2.000 | 0.5974300 |
| 2.400 | 0.4845700 |

**(4) 自相似性**（dx = 0.02 m，k0；同一相似变量下误差随时间的变化）

| t | max 归一化误差 |
|---|---|
| 0.08 | 5.496e-05 |
| 0.16 | 2.747e-05 |
| 0.24 | 1.831e-05 |
| 0.32 | 1.374e-05 |
| 0.40 | 1.099e-05 |
| 0.48 | 9.144e-06 |

初期（t=0.08，间断仍未解析）误差较大，随 t 单调下降并趋于 ~1e-5；与「初值间断
在格点上的瞬时正则化」解释一致。

## ⑥ 失败模式与工件

**失败模式（如实记录）**
1. **第一版解析解符号错**（写成 `(T_hot+T_cold)/2 + (T_hot-T_cold)/2·erf(η)`），
   全场误差 ~1.0（100%），被 FD 独立参照 `max|T_FD−T_erfc| = 1.0` 立即抓出 →
   证明「独立参照」不是装饰。修正为 erfc 形式后 4.7e-5。
2. **稠密路径 ghost 层长度不匹配**（内部 `_mirror_extend` 给 2 个 ghost，而 pair 矩阵按
   3N 建），torch 广播报错 → 改为整列镜像（与 `run_stefan_sph` 一致）。
3. **朴素的反射端 5 点模板与稠密算子不等价**（数值 67.4 的差值）——反射索引写错，
   改为「周期 + 中段切片」与稠密比较后降到 6.8e-12。
4. **误差平台**：仓库 3 模标定在细网格上不再下降（4.7e-5 平台），一度疑似算子不一致；
   用符号分析与残余扩散率回归定位为标定常数的 4.1e-4 偏差（见 ⑤(3)），不是 bug，
   但**意味着 `lattice_diffusion_factor` 的标定值不适用于要求 `k→0` 精度的传播型算例**。
5. 性能：N² 稠密算子 × 25000 步在单机不可行（>120 s 超时）；改 O(N) 模板后全量
   梯度扫描 157.7 s 跑完，稠密路径只用 125 步的小算例做等价性对照。

**工件**
- `benchmarks/case_conduction_1d/result.json`：全部数字 + 门（7 项全 ok，`passed=True`）。
- `benchmarks/case_conduction_1d/profile.png`：剖面 / 相似变量塌缩 / 逐点误差 vs 3% 门。
- `benchmarks/case_conduction_1d/convergence.png`：两种标定的收敛阶梯 / h·dx 敏感性 /
  格点符号与平台来源。
- `src/tensorsph/heat_conduction.py`：`contact_temperature`（erfc 解析）、
  `lattice_stencil` / `stencil_rhs` / `stencil_vs_dense`（O(N) 等价算子）、
  `lattice_symbol_ratio`（离散弥散）、`run_contact_sph`（算例驱动）、
  `fd_contact_reference`（独立 CN-FD 参照）。
- `tests/test_conduction_contact_1d.py`：12 项测试（极限/自相似/算子等价/二阶趋势/
  FD 参照/平台解释/边界无关），耗时约 14 s。

**一键复现**
```
PYTHONPATH=src python3 benchmarks/case_conduction_1d/run.py --figure     # 约 160 s，单核
python3 -m pytest tests/test_conduction_contact_1d.py -q                 # 约 14 s
```

**生态摩擦（如实记）**
- 开工三查拿不到 `knowledge_base` 表：`psql -d tensorplatform -c "select ... from knowledge_base"`
  以 root/kb 角色均 `permission denied`（或口令交互挂起）；`negative_knowledge_ledger`、
  `oss_solutions` 用 `-U foundation` 可读（已查：`NK-M15396-CASE-LAPLACE-SPH` 与多条
  `inflated_delivery` 门记录；oss_solutions 无热传导同域条目）。
- 平台 API 文档路径与实现不符：`/tensor/knowledge_base`、`/tensor/negative_knowledge_ledger`、
  `/tensor/oss_solutions` 均 404；登录实际在 `POST /api/v1/auth/login`（`/tensor/auth/login` 401）。
- 书本文献不可达（archive.org 连接被拒），只能靠 Crossref 核对次级文献书目；
  教科书页码/公式号未核验，已在 ① 明示。

## 结论（一句话）

在同域既有 SPH 导热算子上，一维接触算例 `t*=0.5` 温度剖面对 erfc 解析解的最大归一化
误差为 **2.18e-06（0.0002%）**，远优于 3% 判据；k0 标定下四段
网格均呈二阶收敛（2.00–2.03）；仓库既有 3 模标定留下 4.7e-5 平台，其成因已定量定位为
标定常数与 `k→0` 符号的 4.1e-4 差距并给出可用的 k0 标定。

## 固定端温有限板参考解

当扩散到达两端时，可用 `ContactConfig(reference="finite_slab")` 计算相同壁温下的解析误差。
板长取 `n_particles * dx`，接触面居中，端壁分别保持初始左右温度。默认参考仍为半无限域解。

令 `z=x/L`、`Fo=αt/L²`，按初始温差归一化的解析温度等于稳态直线
`1/2-z` 减去各阶 `exp(-4π²m²Fo) sin(2πmz)/(mπ)` 之和。板长 1 m、
热扩散率 0.25 m²/s、Fo=0.03 时，32、64 粒子的全板最大归一化误差分别为
3.7444e-4 和 9.3046e-5。129 节点 Crank–Nicolson 独立参照的误差为 6.8559e-6。
此时半无限域参考与有限板参考的最大差异已达初始温差的 2.0613%。

复现命令为 `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_conduction_contact_1d.py -k finite_slab`。

### 固定端壁热通量

`run_contact_sph` 在结果的 `wall_flux_left` 和 `wall_flux_right` 中给出终态的单位面积热流，
两端都以流入板内为正。计算采用实际参与跨壁导热的镜像粒子对、粒子体积和求解时的导热标定系数。
板长 1 m、壁温 310 K 和 302 K、热扩散率 0.25 m²/s、Fo=0.03 时，对有限板解析温度求导得到
左端热流 0.81117449 W/m²，右端为其相反数（体积热容取 1 J/(m³ K)）。
32、64 粒子的两端最大相对误差分别为 0.1482% 和 0.0370%。

### 固定端壁累计传热量

设置 `ContactConfig(track_wall_heat=True)` 后，`wall_heat_left` 和
`wall_heat_right` 返回从初始时刻到终态、各端流入板内的累计热量（J/m²）。
求解器在每个显式 Euler 步开始时取端壁热通量，乘以实际步长后累加。
该选项独立于温度剖面的采样频率，默认关闭，关闭时两个结果均为 `None`。

板长 1.5 m、左端 302 K、右端 310 K、热扩散率 0.25 m²/s、
体积热容 1 J/(m³ K) 时，对端壁热通量的 Fourier 级数从零积分至
Fo=0.06，得到左、右端累计热量分别为 −0.27689144 和 +0.27689144 J/m²。
32、64 粒子下，相对解析值的最大误差分别为 0.20134% 和 0.05059%。
复现命令为 `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_conduction_contact_1d.py -k wall_heat`。

### 偏心接触面的净传热量

`ContactConfig(contact_position=...)` 接受相对板中心的初始接触面坐标（m），
取值须在两端壁之间，默认值为 0。跨接触面的粒子按左右材料所占长度初始化温度。
`interface_temperature()` 在该初始接触面插值。偏心有限板参考使用初始阶跃的正弦级数，
在短时间使用包含左右端镜像的表达式。

在 1 m 板上取左侧初始热区长度 0.317 m、初始温差 4 K、体积热容
1 J/(m³ K)、Fo=0.04。积分解析温度剖面得到板内储热增加 0.14311165 J/m²。
32 和 64 粒子下，两端累计净流入量相对该参考的误差分别为 0.11036% 和 0.01118%。
复现命令为 `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_conduction_contact_1d.py -k 'offset or near_fixed'`。

独立的 Crank–Nicolson 有限差分参照也接受 `fd_contact_reference(..., contact_position=x0)`。
坐标以板中心为原点，单位为米。内节点的初值取节点两侧半个网格宽度内的温度平均值，
两个端点始终取规定壁温。`fourier=0` 可直接读取此初始剖面。

板长 1.5 m、热区占板长三分之一或三分之二、Fo=0.04 时，按初始温差归一化的全板最大温度误差
在 49、98、196 个网格区间上依次约为 1.122e-4、2.749e-5 和 7.013e-6。
最细有限差分网格的奇数号节点还与 98 个 SPH 粒子中心逐点对应，可直接交叉验证。
复现命令为 `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_conduction_contact_1d.py -k fd_offset`。

### 偏心初始温度阶跃的两端瞬时热通量

对于材料参数均匀、两端固定温度的板，`finite_slab_contact_wall_fluxes` 返回左右端
朝板内的热通量（W/m²）。传入物理时间（秒）、热扩散率、板长、导热系数及左右端温度，
并用 `contact_position` 指定初始阶跃相对板中心的位置（米）。
早期解使用周期高斯镜像求和，较晚时刻使用温度正弦级数在端点的导数。
复现两端 SPH 通量及独立有限差分对照可运行
`PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_conduction_contact_1d.py -k offset_contact_wall_flux`。

`finite_slab_contact_wall_heat(t, alpha, length, conductivity, t_hot, t_cold, contact_position=x0)`
给出从初始时刻到物理时间 `t` 两端各自的累计流入热量，单位 J/m²，返回顺序为左、右。
早期解对各高斯镜像逐项积分，较晚时刻使用积分后的傅里叶级数。在初始温度阶跃偏心时，
这一参考值可分别检验粒子计算的 `wall_heat_left` 和 `wall_heat_right`。
复现命令为 `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_conduction_contact_1d.py -k offset_wall_heat`。

`ContactConfig(wall_heat_window=(start, duration))` 可只记录指定观测窗口的两端传热量。
起始时刻和持续时间以秒计，两者指定的窗口须位于本次计算内。结果
`window_heat_left`、`window_heat_right` 以流入为正，单位为 J/m²。
每个欧拉时间步按实际落入窗口的时长累加步首热通量，允许窗口边界落在步内。
即使持续时间短于一个时间步，通量的时间分辨率仍由该步长决定。
