# XVAL-SPH-PROJECTION-TAYLOR-GREEN round 2 — 23.6 % 的误差预算与 dx=0.04 爆破解剖 (m21075)

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（residual debt：m20500 留口 `XVAL-SPH-PROJECTION-TAYLOR-GREEN`）
**仓库/分支**: TensorSPH · `agent/m21075` ｜ 上游: `docs/xval_isph_projection_taylor_green.md`（m20500，honest FAIL 23.613 %）

## 1 案例与问题

m20500 把涡类-投影记录"测出来了"：周期 ISPH 投影在 Taylor-Green 涡上 **23.613 % > 15 %**，
并留下两个**未解释**的事实——① 23.6 % 由哪一项携带？② 为什么 `dx=0.04` 会爆破（2.1e5 %）？
两者不解，"分辨率阶梯不可信"，选型表也就只能停在 WITHHELD。本单只做这件事：把误差**拆开量**，
再据拆解结果试一次修复，成功与否都如实入库。

## 2 方法（程序化，零手抄）

| 环节 | 做法 | 指针 |
|---|---|---|
| 误差预算 | 最小二乘拟合 `u ≈ c·u_e`，用正交分解 `‖u-u_e‖² = ‖u-c u_e‖² + (c-1)²‖u_e‖²` 把总误差拆成**幅度项**(c-1)与**形状项** | `benchmarks/isph_taylor_green/diagnose_error_budget.py::decompose` |
| 时间/空间诊断 | 逐步记录 `‖u-u_e‖`、`E/E_exact`、离散散度 rms、`max|u|`、CG 迭代数、`div_after/div_before`、**最小粒子间距/dx** | 同上 `run_config(..., log_every=10)` |
| 算子体检 | 粘性算子的有效本征值 vs 解析 `2k²`（TG 速度场 `∇²u = -2k²u`）；投影对"纯无散场"的作用（应为恒等） | `benchmarks/isph_taylor_green/diagnose_operators.py` |
| 修复候选 | Lind et al. (2012) Fickian 粒子 shifting 的**周期版**（浓度 `C_i=ΣV_jW_ij` 含自项、`∇C` 直接求和、按 `C_d h|u|dt` 步长、逐粒子截断 0.2dx、环面回卷） | `src/tensorsph/isph_periodic.py::concentration / concentration_gradient / shift_positions` |
| 复现 | `PYTHONPATH=src python benchmarks/isph_taylor_green/diagnose_error_budget.py --out outputs/isph_taylor_green --configs "0.05:0.1,0.04:0.1"`；修复组 `run.py --shifting <C_d>` | 见 §6 |

## 3 结果

**3.1 误差预算（`outputs/isph_taylor_green/error_budget.json`）**

| 配置 | 总误差 | 幅度项 (c-1) | 形状项 | c | div_ratio 均值 | 结局 |
|---|---|---|---|---|---|---|
| dx=0.05, Co=0.1（门） | **23.588 %** | 14.98 % | 18.22 % | 0.850 | 0.792 | 稳定 |
| dx=0.04, Co=0.1 | 2.786e5 % | 2.089e4 % | 2.778e5 % | 209.9 | 3.129 | **58 步爆破** |

（拆分自洽：`sqrt(14.98²+18.22²) = 23.60`，与总误差一致到 0.01 %。）

**3.2 单步局部误差只有 0.0185 %** — 一步 `Heun+投影` 后的解 vs 解析解：0.0185 %（纯粘性 0.0185 %，
投影对无散场是恒等：`KE kept = 1.000000`）。77 步线性累积应 ≈ 1.4 %，实测 23.6 %：
**误差不是局部截断，而是被放大的**。

**3.3 放大来自"云退化"，不是算子** — 逐步记录显示最小粒子间距单调塌陷：

| dx=0.04 | 5 步 | 20 步 | 40 步 | 55 步 | 57 步 | 58 步 |
|---|---|---|---|---|---|---|
| 误差 | 1.22 % | 4.64 % | 28.99 % | 105.2 % | 1429 % | 2.17e5 % |
| `div_after/div_before` | 0.126 | 0.644 | 0.989 | 1.638 | 8.99 | 130.2 |
| `r_min/dx` | 0.880 | 0.571 | 0.458 | 0.358 | 0.129 | 0.067 |

因果链是清楚的：**间距塌陷 → `b_i`/图 Laplacian 的一阶一致性退化 → 投影不再消除散度（比值从 0.09 涨到 ≈1）
→ 比值 > 1（投影反而放大散度）→ 压力与速度一起爆掉**。`dx=0.05` 是同一机制只走到"比值 0.79、误差饱和 24 %"。

**3.4 修复尝试 = 实测失败（honest negative）** — 周期 Fickian shifting 两种标定都让运行更早发散：

| 修复组 | 标定 | 误差 | 结论 |
|---|---|---|---|
| `run.py --shifting 0.05`（几何式 `A=0.05h²`） | — | 1.96e40 % | 发散 |
| `run.py --shifting 0.5`（`A=0.5h|u|dt`） | Lind 推荐量级 | 3.33e4 % | 发散 |
| `run.py --shifting 2.0` | 更大 | 5.32e4 % | 发散 |

shifting 本身是**正常工作的**（`tests/test_isph_periodic_shifting.py`：格点上 `∇C`≈4e-14、浓度≈1.000、
抖动云 12 次 shifting 后最小间距**上升**、截断与回卷都守住）。所以失败点不在 shifting，而在于
**每步位移注入的粒子级噪声恰好落在投影"放大散度"的频段**——这反过来印证了 §3.3 的因果链：
当前投影对无序云是**反耗散**的，任何噪声注入都会自激。

## 4 注意点 / 局限

* `dx=0.04` 的爆破**已被解释**（间距塌陷 + `div_ratio>1`），但**未被修好**；分辨率阶梯仍不可用，
  `isph_projection` 的涡类记录仍是 FAIL，选型表对涡类剖面**继续 WITHHELD**（未改判定）。
* 修复方向因此被收窄（交给下一单）：先让投影对无序云不放大散度——① 交错/MAC 周期投影；
  ② 用 MLS 一阶一致算子对（`csf_consistent.mls_gradient/mls_divergence` 已存在）替换 pair-sum `D/G`；
  ③ shifting 与投影**耦合**（先 shifting 再投影，且投影算子在 shifting 后的云上重建）。
  单纯 shifting（本单做法）已被证伪，不要再重复。
* 只跑到 `dx=0.04`；更细网格未跑。本机 load15 66、单次运行 7–20 s，属轻量。

## 5 误差判定

| 关键量 | 基准 | 测量 | 是否 < 3 % | 是否 < 15 % |
|---|---|---|---|---|
| 速度场相对 L2 误差（门配置 dx=0.05） | Taylor & Green (1937) 解析解 | **23.588 %** | 否 | **否** |
| 同上（dx=0.04） | 同上 | 2.786e5 %（爆破） | 否 | 否 |
| 粘性算子有效本征值 | `2k²` | −2.66 % | 是 | 是 |
| 投影对纯无散场的保能 | 1.000000 | 1.000000 | 是 | 是 |

**结论：本单是 honest FAIL 的*深化*，不是修复。** 新增的是可复用的诊断口径（误差预算 + 云退化指标）
与一条被证伪的修复假设。

## 6 复现

```bash
PYTHONPATH=src python benchmarks/isph_taylor_green/diagnose_operators.py --out outputs/isph_taylor_green
PYTHONPATH=src python benchmarks/isph_taylor_green/diagnose_error_budget.py \
    --out outputs/isph_taylor_green --configs "0.05:0.1,0.04:0.1"
PYTHONPATH=src python benchmarks/isph_taylor_green/run.py --out /tmp/tg --configs "0.05:0.1" --shifting 0.5
PYTHONPATH=src python -m pytest tests/test_isph_periodic_shifting.py tests/test_isph_periodic.py -q
```

图：`docs/reports/isph_taylor_green_r2.png`（左=误差增长 log 曲线 + 15 % 线；右=最小间距/dx 先于爆破塌陷）。
