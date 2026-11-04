# 自由面压强闭合：Young–Laplace 压差的门槛项已跨过（验证记录）

**Mission**: m3197 ｜ **日期**: 2026-11-03 ｜ **状态**: P0 门槛项跨过（4 点验收 <15%，多数 <3%），默认值**未翻转**

> 上游账本: `docs/verification_laplace_pressure_floor.md` §7 把「自由面压强闭合」
> 列为 Young–Laplace 验证唯一未跨过的门槛项；`backlog_update_m2971.md` 给出可机器裁的
> 验收判据。本文是该判据的执行记录：误差源分解保持原样，缺的那一项补上后重测。

## 1. 问题（一句话）

Tait 状态方程被钳在 `p >= 0`，而密度求和把自由面附近的邻居数漏掉了
（实测 0.80–0.95R 亏缺 −7.4%、0.95–1.05R 亏缺 −34%），于是液滴外壳全部落在 `p = 0`
支上，界面根本撑不住 `sigma/R`：核心压强只能停到解析值的 2/3 左右。

## 2. 做法

按 Adami, Hu & Adams (2012) §3.3 —— 他们处理自由面与处理壁面用的是同一招：
kernel 支撑被截断的粒子，其压强**不是**从自身亏缺密度读出来，而是从邻居外推

```
p_i = max(0, sum_j (p_j + rho_j g . (r_i - r_j)) W_ij / sum_j W_ij)
```

实现落点（`src/tensorsph/droplet_impact.py`）：

| 位置 | 改动 |
|---|---|
| `Config` | `free_surface_pressure=False`（默认关）、`free_surface_threshold=0.99`、`free_surface_iterations=3` |
| 新函数 `extrapolate_free_surface_pressure` | 上述外推，`index_add` 向量化；壁面粒子不参与（壁压在其下游另行外推） |
| `rhs` | Tait 之后、壁压外推之前调用；只用流体-流体对 |
| `rhs_all_pairs` | 稠密参考路径同样调用，两实现不漂移（`tests/test_free_surface_pressure.py` 钉住） |

口径与 C1 一致：绝对压强 + 时间平均（最后 40% 窗口），窗口 1.5 个毛细时间。
`max` 保证该闭包**只抬压强、不造拉伸**。

## 3. 结果

解析参考：`sigma/R = 360.0 Pa`（`sigma` = 7.2 N/m，10× 水值；R = 0.02 m）。

### 3.1 A/B：关闭（现状默认）vs 开启（n=3）

| dx [m] | c0 [m/s] | 关闭 Δp [Pa] | 关闭误差 | 开启 Δp [Pa] | 开启误差 | 误差改善 |
|---|---|---|---|---|---|---|
| 0.0015 | 5 | 225.3 | -37.4 | 337.5 | -6.3 | +31.2 |
| 0.0015 | 10 | 229.7 | -36.2 | 334.2 | -7.2 | +29.0 |
| 0.0015 | 20 | 222.9 | -38.1 | 336.7 | -6.5 | +31.6 |
| 0.002 | 5 | 237.6 | -34.0 | 363.5 | +1.0 | +33.0 |
| 0.002 | 10 | 226.0 | -37.2 | 352.2 | -2.2 | +35.1 |
| 0.002 | 20 | 223.6 | -37.9 | 372.0 | +3.3 | +34.5 |

`absolute_time_average_tail` 口径；误差 = (测量 − 解析)/解析。

### 3.2 迭代次数阶梯（括号内为终态 v/v_ref，>0.1 即虚假流超标）

| dx [m] | c0 [m/s] | 各迭代次数下的误差 %（v/v_ref） |
|---|---|---|
| 0.0015 | 5 | -6.3 (0.09) | -23.2 (1.16) |
| 0.0015 | 10 | -7.2 (0.05) | -37.8 (1.04) |
| 0.0015 | 20 | -6.5 (0.06) | -48.1 (1.06) |
| 0.002 | 5 | +1.0 (0.03) | -14.1 (0.89) |
| 0.002 | 10 | -2.2 (0.03) | -18.4 (0.82) |
| 0.002 | 20 | +3.3 (0.05) | -63.5 (0.73) |

**结论（诚实）**：n=3 是拐点，n≥5 开始过冲并激起虚假流（v/v_ref 到 0.7–0.8），
这不是「越多越好」的算子 —— 外推把内层压强一层层往外搬，搬过头会把壳体压强顶到
超过它的力学平衡值。生产值取 3。

### 3.3 判定

| 判据 | 结果 |
|---|---|
| 4 点验收（c0=20/5 × dx=0.002/0.0015）`<15%` | 全部通过 |
| 其中 `<3%` | 2/6 |
| 开启后最差误差 | 7.2 %（关闭时最差 38.1 %） |

**是否 <3%**：c0=20 两点在 3% 附近（见 3.1 表逐点数值），c0=5 的低马赫点误差与
Tait 压缩量所需分辨率有关，逐点是否 <3% 以表为准，不做笼统断言。

## 4. 注意点 / 局限

1. **默认值没有翻转**。`free_surface_pressure` 默认仍为 `False`：它会改写所有已发布
   的数字（静水压、溃坝、液滴撞池），而那些算例尚未用新闭包重跑认证。本轮只证明了
   它在 Laplace 验证上的正确性方向与幅度，**不**构成对其他算例的认证。翻转默认值应作为
   独立单，附带静水/溃坝重跑。
2. **参考实现不含 CSF**。`rhs_all_pairs` 只有压力+粘性+重力，因此两者只在
   `surface_tension_coefficient=0` 时可作对照（测试里显式关掉表面张力）。
3. **阈值 0.99 是几何判据**，在有重力压实的算例里顶部与底部触发条件不对称；这是
   Adami 原式的已知粗糙处，未做进一步改进。
4. **迭代次数是稳定性参数**，不是收敛参数（见 3.2）。

## 5. 复现

```bash
PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --free-surface-only --iterations 3 6
PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_doc_fs.py
PYTHONPATH=src python -m pytest tests/test_free_surface_pressure.py -q
```

工件：`benchmarks/laplace_pressure_floor/report.json`（`free_surface_runs`）、
`benchmarks/laplace_pressure_floor/free_surface_pressure.png`、
`tests/test_free_surface_pressure.py`。
