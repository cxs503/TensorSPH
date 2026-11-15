## ⑤ 注意点 / 本轮真正修掉的缺陷

### 5.1 判定门的第二个假阳性：收敛包络 ≠ 物理解

上一轮（报告 19734）修掉了「启动瞬态被判 FAIL」的假阳性：门只在「窗口 ≥1 个周期」时给结论，
漏掉了「幅值还在长」这一层。修法是**包络平稳性**（窗口后半段峰峰值 / 前半段峰峰值 ∈ [0.8, 1.25]）。

本轮两个长时作业跑完（`cj3755` dx=0.05 / 10000 步、`cj3756` dx=0.02 / 6000 步），
把这个修法的**反向漏洞**暴露出来：一个**发散的场**同样有平稳包络——它饱和了。
于是旧门对 dx=0.05 打出 `FAIL`，`lift_amplitude` 相对误差 `+109547154 %`。
这不是「力算子精度不够」，这是「场已经不是解了」。把它标成 FAIL 是一个没有证据支撑的精度主张。

本轮的修法：在统计判据之外增加**可判定性（admissibility）筛检**，作为任何精度判定的前置条件。
三项，全部读自工件自带的量或基准几何，无自由参数：

| 检查 | 判据 | 阈值来源 |
|---|---|---|
| `finite` | 窗口内无 NaN/Inf | 无 |
| `blowup` | `max|v| / c0 ≤ 1` | `c0` 由工件 `case.sound_speed` 声明（FSI2 配置 Mach 0.06） |
| `body_containment` | `max|tip| / beam_length ≤ 1` | 基准几何 L = 3.5 D（Turek–Hron） |
| `period_band` | 识别周期 ∈ `D/(St·U)`，St ∈ [0.13, 0.30] | `VIV_STROUHAL_BAND`（既有常量） |

不满足 → 判定一律降级 `WITHHELD`，并在 `withheld_reason` 写明哪一项、量值多少；
相对误差仍保留在工件里（`comparable: false`），但不作为精度结论。
筛检**未核**的项进 `skipped` 列表，绝不静默算通过。

### 5.2 两个长时工件本身：都不是 Turek–Hron 解

| 工件 | 现象 | 量值 |
|---|---|---|
| `viv_dx005_aitken`（dx=0.05，26.0 s） | 流体发散 | `max|v| = 12.04 m/s` = **2.41 × c0**；梁尖位移 1.238 m = **3.54 × 梁长** |
| `viv_dx002_aitken`（dx=0.02，6.24 s） | 过阻尼 | `Cd` 均值 0.052 vs 3.2（**−98 %**）；`tip_y` −31 %；包络衰减 → `WITHHELD` |

注意 `period_band` 一项**依赖分析窗口**：dx=0.05 在 `warmup=0.5`（`verdict_body_force_transient`
默认）识别到 0.320 s、出带；在 `warmup=0.3`（`make_viv_report` 默认）识别到 2.137 s、落带内。
即这条带对该工件时灵时不灵，**不能**单独作为否决依据；把它挡下来的是与该窗口无关的
`blowup`（2.41 × c0）和 `body_containment`（3.54 × 梁长）。这也是为什么三条检查各自独立记录、
任意一条不过即降级，而不是合成一个分数。

两者方向相反（一个炸、一个死），说明强耦合路径在两个分辨率上都没有收敛到物理解；
`dx` 不是这里的控制参数。

### 5.3 骨架离散被 `max(15, …)` 静默夹住 —— 这条梯子不是收敛研究

`run_body_force_transient.py` 用 `n_beam_nodes = max(15, int(beam_length/dx))`：

| dx [m] | `beam_length/dx` | 实际节点数 | 节点间距 [m] | 间距/dx |
|---|---|---|---|---|
| 0.05 | 7 | **15（被夹住）** | 0.0233 | 0.47 |
| 0.02 | 17 | 17 | 0.0206 | 1.03 |
| 0.10 | 3 | **15（被夹住）** | 0.0233 | 0.23 |

dx 从 0.05 收到 0.02（流体细 2.5×），梁只从 15 收到 17 个节点（细 1.1×）；
dx=0.10 时梁比流体**细 4.3 倍**。所以这两个工件的差异里混着「结构没细」这一项，
不能读成流体的收敛性。夹紧保留（保护杆件求解器），但**写进工件**：
`beam_nodes_clamped`、`case.beam_length`、`case.beam_node_spacing`、
`case.beam_spacing_over_dx`，让一条 dx 梯子无法被当成收敛证据。

### 5.4 本单没有拿到精度判定（如实）

漂移/发散两条腿都不成立 → 锚点验收（<15 %）**本轮仍未达成**，`WITHHELD` 不是 PASS。
下一步该做的不是再加分辨率，而是先让 dx=0.02 的场有物理量级的 `Cd`（0.05 → 3.2 差了 60 倍，
指向载荷通道或体积力标度，不是离散误差），再谈周期窗口。

## ⑥ 复现

```bash
# 从集群取回两个长时工件后，重跑判定（轻量，秒级）
PYTHONPATH=src python benchmarks/xval_fsi2/verdict_body_force_transient.py \
    --artifact artifacts/xval_fsi2/viv_dx005_aitken.json \
    --out artifacts/xval_fsi2/viv_dx005_aitken_admitted.json
PYTHONPATH=src python benchmarks/xval_fsi2/verdict_body_force_transient.py \
    --artifact artifacts/xval_fsi2/viv_dx002_aitken.json \
    --out artifacts/xval_fsi2/viv_dx002_aitken_admitted.json

# 报告（含图）
PYTHONPATH=src python benchmarks/xval_fsi2/make_viv_report.py \
    --artifact artifacts/xval_fsi2/viv_dx005_aitken.json \
    --artifact artifacts/xval_fsi2/viv_dx002_aitken.json \
    --out-md docs/xval_fsi2/VIV_PERIODIC_VERDICT.md --figures

# 守护
python -m pytest tests/test_fsi2_verdict_admissibility.py \
                 tests/test_fsi2_viv_verdict_pipeline.py -q
```
