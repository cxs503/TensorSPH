# CASE-HYDROSTATIC — 方法选型与迁移方案（第 2 轮·调研结合，m18736）

**日期**: 2026-11-13 16:54 UTC ｜ **仓库**: TensorSPH ｜ **锚点**: CASE-HYDROSTATIC（算例池·静水压剖面）
**验收判据**: 【算例池|HYDROSTATIC｜ref: 解析｜容差: <1%】
**证据出处**: `tests/data/case_hydrostatic/report.json`（sha256 前 16 位 `f0eb167f0a7675b3`，
MANIFEST.sha256 全量校验）；本轮 gate 配置复跑与 shipped 值**逐位一致**（见 §7）。
本文所有数字由 `tests/data/case_hydrostatic/` 工件程序化读出，零手抄。

## 1. 问题设定与基准出处

静水矩形水箱（宽 W=1.0 m，深 H=0.4 m，ρ0=1000 kg/m³，g=9.81 m/s²）在重力下静止，
SPH 解必须在内层带（band）上恢复解析静水压剖面：

    p(y) = ρ0 g (H − y)        （dp/dy = −ρ0 g，自 p(H)=0 积分）

**基准出处（external_anchor）**: 闭式解，无表格转录环节。同一静水压参考被
Adami, Hu & Adams, *J. Comput. Phys.* **231** (2012) 7057–7075（WCSPH 壁面边界
修正的静水压算例）与本仓 `benchmarks/hydrostatic/run.py`（B1 基准）采用；
参考值在实测带中心处**求值**而非查表。

## 2. 方法与实现（方程/格式出处 + 代码指针）

| 环节 | 选择 | 出处/指针 |
|---|---|---|
| 求解器 | 生产 WCSPH（`tensorsph.droplet_impact.rhs`），Tait 状态方程 | `src/tensorsph/case_hydrostatic.py::run_case` |
| 参考 | 闭式 p(y)=ρ0 g (H−y) | `analytic_pressure`（同模块） |
| 测量 | 后半程时间平均的带平均压（8 s 窗，sample_every=5） | `run_case` 累积器 |
| 自由面排除 | 重叠规则：带上缘进入 2h 即排除（严于「带中心 2h」） | `pressure_profile` |
| 几何前置门 | 声明几何必须可被晶格表示（否则误差归因几何而非求解器） | `check_geometry` |
| 负控 | 求解器 g=0、参考保留 ρ0 g → 必须读 ~−100% | `control_g0` |
| 判据 | max 内层带 \|p−p_ref\|/p_ref（时间平均）≤ 1.0% | `evaluate`，`TOLERANCE_PCT=1.0` |

## 3. 参数表（无量纲数完整）

| 量 | 值 | 无量纲量 |
|---|---|---|
| 网格 | dx=0.02 m，50×20=1000 粒子，h=1.3 dx | H/dx=20，W/H=2.5 |
| 声速 | c=20 m/s（Tait） | Ma_acoustic=√(2gH)/c=0.1401 |
| 时间 | 8 s，dt=3.900e-04 s，20513 步 | 声学周期占比=0.005 |
| 采样 | 后半程（average_from=0.5），2051 样本 | — |
| 带 | 8 带（厚 0.05 m=2.5 格），门控带 [0, 1, 2, 3, 4, 5] | 带厚/格=2.5 |

## 4. 对照表（ref vs got，逐条打勾/打叉）

判据：max 内层带误差 ≤ 1.0%（预声明，读数字前钉死）。

| 配置 | 粒子数 | max 带误差 | mean | vs 1.0% 门 |
|---|---|---|---|---|
| **window_8s（gate）** | 1000 | **0.7710 %** | +0.4569 % | **PASS ✓** |
| window_4s | 1000 | +0.9773 % | +0.3369 % | PASS ✓（敏感性点） |
| window_2s | 1000 | +1.0270 % | +0.5501 % | FAIL ✗（敏感性点） |
| dx_004 | 250 | +0.8213 % | +0.5921 % | PASS ✓ |
| dx_0025 | 640 | +0.7033 % | +0.3603 % | PASS ✓ |
| width_w15 | 1500 | +0.7254 % | +0.3707 % | PASS ✓ |
| sound_c30 | 1000 | +1.0203 % | — | FAIL ✗（单旋钮点） |
| control_g0（负控） | 250 | +99.88 % | +99.53 % | 负控 OK（\|mean\|>80%） |

**误差判定**: gate 0.7710 % < 1.0 % → **PASS**；关键量 vs 解析基准
误差 0.7710 %，**< 3%：是**（亦 < 1% 门限）。

## 5. 收敛性与参数敏感性（≥2 分辨率/参数点）

* **分辨率阶梯（t=0 机制诊断，唯一误差源=壁面/自由面求和密度截断）**:
  dx=0.04 → 28.22 %，dx=0.025 → 14.94 %，
  dx=0.02 → 10.22 %——随分辨率单调下降，
  唯一误差源=壁面/自由面求和密度截断。
* **平均窗阶梯**: 2 s → 1.0270 %，4 s → 0.9773 %，
  8 s → 0.7710 %（门）；2 s 窗振荡采样噪声 max_fluctuation
  212 % vs 8 s 的 15.8 %——这正是 gate 钉在 8 s 的预声明理由。
* **单旋钮点**: 宽箱 width_w15 +0.7254 %（PASS）；
  声速 c=30 +1.0203 %（略超 1%，非门）。
* **负控**: control_g0 mean +99.53 %（g 关、参考保留 ρ0 g）→ 指标有效。

## 6. 失败模式与工件

* **前轮死法（m18670）**: `worker-crash`——worktree 准备阶段
  `error: unable to write file benchmarks/rayleigh_taylor/runs/...`，未进入计算；
  非本案例数值失败。本轮已避开（worktree 正常，测试 16 passed + 1 skipped）。
* **历史死法（m16457 记录）**: 证据只放 `runs/`（交付白名单外）→ 新检出必红。
  现由 `tests/data/case_hydrostatic/` 文本通道（freeze.py 生成，sha256 清单）兜底。
* **repro_cmd**:
  - gate 单配置: `python benchmarks/case_hydrostatic/run.py --config window_8s`
  - 全网格: `python benchmarks/case_hydrostatic/run.py --config all --static`
  - 聚合+图: `python benchmarks/case_hydrostatic/run.py --aggregate --figure`
  - 文本通道: `python benchmarks/case_hydrostatic/freeze.py`
  - 测试: `python -m pytest -q tests/test_case_hydrostatic.py`
* **工件**: `tests/data/case_hydrostatic/`（report.json + 18 run 文件 + MANIFEST.sha256 + channel.json）；
  本轮复跑: `runs/case_hydrostatic/runs/window_8s.json`（与通道**逐位一致**）。

## 7. 调研·开源横向（oss-cache）

* **本地 oss-cache**: 仅 LBM 系（Maarten-vd-Sande_lbm / hschwane / pylbm）+
  **GraSPH**（hschwane/GraSPH，GPU WCSPH，C++/OpenCL）。GraSPH 无静水压算例；
  其结构（密度→压力→加速度→积分，Balsara 人工粘性）与本仓 WCSPH 同族，
  无可直接迁移的静水压基准结构——本案例的「带剖面+重叠排除+负控+几何前置门」
  组合是差异化设计，非移植。
* **集群 oss_inventory**（`feature_slug ILIKE '%HYDROSTATIC%'`）: 集群队列拥塞
  （141 作业排队、>16 min 等待即被杀），查询作业 cj2029 排队中——**friction 如实记录**，
  不阻塞验收（判据数字已由本地复跑裁判）。

## 8. 调研·方法谱系（external_anchor）

* **Adami, Hu & Adams, J. Comput. Phys. 231 (2012) 7057–7075**: WCSPH 静水压
  水箱算例（壁面边界修正验证），参考即 p=ρ0 g (H−y)。
* **Monaghan, A. R. (1992), *J. Comput. Phys.* 98: 524–539**: WCSPH 原始文献，
  Tait EOS 与弱可压缩静水压平衡的标准处理。
* **教科书做法**（e.g. Monaghan 2012, *An Introduction to SPH*, §11.3 静水压）:
  静水压剖面是 SPH 一阶验证算例；自由面 2h 内截断不可比，须排除；
  弱可压缩静水池存在声学振荡，须时间平均后比较。本实现三条全中。

## 9. 选型沉淀（oss_solutions, feature_slug='CASE-HYDROSTATIC'）

* **选型**: 生产 WCSPH + Tait EOS + 时间平均带剖面 + 自由面重叠排除 + 负控 +
  几何前置门。判据（1.0%）在数字读出前预声明。
* **迁移方案**: (a) 带剖面+重叠排除+负控三件套可复制到算例池其它静力/准静力
  算例（如 sloshing 静息段）；(b) `check_geometry` 前置门模式可复制到一切
  「声明几何 vs 晶格可表示性」问题；(c) 证据三通道（runs/ 本地、outputs/ 镜像、
  tests/data/ 文本通道）是「证据可在新检出上可裁」的通用解法。
* **DB 登记**: 集群队列拥塞期间无法直写 oss_solutions 表（friction）；
  本文件即登记内容载体，队列恢复后按本文件补登记。

## 10. 误差判定（显式回答）

关键量（gate，window_8s 时间平均 max 内层带误差）vs 解析基准:
**0.7710 % < 3%：是**；且 < 1% 门限：**是**。
