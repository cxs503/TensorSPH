# 判定门缺陷修复：精度的前置条件是可解性

**锚点** `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID` · 仓 TensorSPH · 域 SPH / FSI

## 问题

`benchmarks/xval_fsi2/verdict_body_force_transient.py::build_verdict` 的判定逻辑是

```
regime_ok = (periods_in_window >= 1) and amplitude_stationary
verdict   = WITHHELD if not regime_ok else (PASS if all rel_error <= 15% else FAIL)
```

`amplitude_stationary`（包络后半段/前半段 ∈ [0.8, 1.25]）是为修掉「启动瞬态被判 FAIL」
加进来的。它判别的是**包络是否稳定**，而一个**发散的场**同样会给出稳定包络——它饱和了。

两个跑完的长时作业把这个洞打实了：

* `artifacts/xval_fsi2/viv_dx005_aitken.json`（dx=0.05，10000 步，26.0 s）
  `regime_ok = True` → 旧门判 `FAIL`，`lift_amplitude` 相对误差 `+109547154 %`。
  实际场是：`max|v| = 12.04 m/s`（声明 `c0 = 5 m/s`，Mach 0.06 配置）→ **2.41 × c0**，
  梁尖位移 1.238 m（梁长 0.35 m）→ **3.54 × 梁长**。
* `artifacts/xval_fsi2/viv_dx002_aitken.json`（dx=0.02，6000 步，6.24 s）
  `Cd` 均值 0.052 vs 基准 3.2（−98 %），包络衰减。

「发散的场离基准 +109547154 %」不是一个关于力算子精度的主张，把它写成 `FAIL`
是超范围结论。这是本轮修掉的东西。

## 修法

新增 `physical_admissibility()`，作为精度判定的**前置条件**；不满足则降级 `WITHHELD`
并在 `withheld_reason` 逐条写明哪一项、量值多少。四项检查，阈值全部来自工件自带的
声明量或基准几何，无自由参数：

| 检查 | 判据 | 阈值来源 |
|---|---|---|
| `finite` | 窗口内无 NaN/Inf | — |
| `blowup` | `max|v| / c0 <= 1` | `case.sound_speed`（工件声明） |
| `body_containment` | `max|tip| / beam_length <= 1` | 基准 L = 3.5 D；工件缺该字段时按 3.5 D 回填并记录 |
| `period_band` | 周期 ∈ `D/(St·U)`，St ∈ [0.13, 0.30] | 既有 `VIV_STROUHAL_BAND` |

设计约束：

* **不静默通过**。配置缺失的检查进 `skipped`，`checked` 只列真正跑过的项。
* **不留白丢失**。相对误差仍写进工件，`gates[*].comparable = False` 标为不可作结论。
* **可回放**。`build_verdict(..., enforce_admissibility=False)` 保留旧口径以复现历史判定，
  且仍记录 `admissibility`（`enforced: False` 显式可见）。
* `period_band` 依赖分析窗口（dx=0.05 在 warmup 0.5 出带、0.3 落带），因此不单独作为
  否决依据；`blowup` / `body_containment` 与窗口无关。

## 顺带修掉的第二个缺陷：骨架离散被静默夹住

`run_body_force_transient.py` 里 `n_beam_nodes = max(15, int(beam_length/dx))`：

| dx [m] | `beam_length/dx` | 实际节点 | 节点间距 [m] | 间距/dx |
|---|---|---|---|---|
| 0.10 | 3 | 15（夹） | 0.0233 | 0.23 |
| 0.05 | 7 | 15（夹） | 0.0233 | 0.47 |
| 0.02 | 17 | 17 | 0.0206 | 1.03 |

dx 0.05 → 0.02 流体细 2.5×，梁只细 1.1×。两个工件的差异里混着「结构没细」，
不能读成流体收敛性。夹紧保留（保护杆件求解器），但写进工件：`beam_nodes_clamped`、
`case.beam_length`、`case.beam_node_spacing`、`case.beam_spacing_over_dx`。

## 验证

```
PYTHONPATH=src python -m pytest tests/test_fsi2_verdict_admissibility.py \
                                 tests/test_fsi2_viv_verdict_pipeline.py -q
# 14 passed
```

新增 9 项守护（`tests/test_fsi2_verdict_admissibility.py`）：干净周期解仍 `PASS`（防过度拦截）、
发场降级 `WITHHELD` 且 `comparable=False`、梁尖越界降级、缺字段时按 3.5 D 回填、
出带周期降级、缺失配置进 `skipped` 不算通过、NaN 降级、关闭筛检可回放旧口径、
驱动写出夹紧披露。

## 未完成（如实）

锚点验收（<15 %）**未达成**。两个分辨率一炸一死，方向相反，`dx` 不是这里的控制参数。
下一步应先把 dx=0.02 的 `Cd`（0.05 vs 3.2，差 60 倍）拉回物理量级——那是载荷通道/标度问题，
不是离散误差——再谈周期窗口。详见 `VIV_PERIODIC_VERDICT.md` ⑤。