# 一个「配对通过」不等于收敛：超 15% 记录的分辨率阶梯定级（m22957）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` ｜ **仓库**: TensorSPH ｜ **任务**: m22957（自找活·验证欠账）
**类型**: 验证欠账清偿（adversarial-review 口径：挑战已合并结论） ｜ **前序**: m22787（超限记录逐条裁定）、m22831（派生工件反漂移）
**产物**: 新裁定模块 + 生产者改造 + 15 条守卫测试 + 本报告 + 图

## ① 案例介绍（要挑战的结论是什么）

锚点的结论载体是 `docs/reports/method_selection_registry.json`（30 条证据记录）。m22787 把其中 7 条
超业主 15% 标准的记录逐条裁定，并给出类别 `resolution_attributable`，其定义是：

> 同一方法的同一量，在**更细一档**上落在标准内 ⇒ 超限值是「缺失分辨率列」掩盖的离散化症状。

本单要问的是：**一档配对能承载「离散化症状」这个结论吗？**

被挑战的两条记录：

| 记录 | 方法 | 登记 [%] | 配对 | m22787 类别 |
|---|---|---|---|---|
| `x_transport_velocity_coarse` | transport_velocity_correction | 16.963 | dx 0.04 → 0.03（10.647） | resolution_attributable |
| `x_couette_wall_shear` | wcsph_wall_no_slip | 32.649 | H/dx 8 → 16（0.478） | resolution_attributable |

## ② 怎么做的

1. **先看仓库自带的完整阶梯**（而不是只看注册表里那两档）：
   `docs/xval_sphinx_tvc/data/ladder_dense.json` 是四位分辨率阶梯
   （`dx = 0.04 / 0.03 / 0.02 / 0.01 m`，240 点稠密采样，同一个 `alpha=0.10` 臂）。
2. **把「标签」换成「测量」**：新模块 `benchmarks/method_selection/resolution_ladder_guard.py`
   从**已提交产物**重算阶梯（不手抄任何数字），按三条判据定级：
   - G1 阶梯至少 3 档（少于 3 档无法谈趋势）；
   - G2 误差随加密**不增**（任何一档回升都记 `reversals`，含 pp 增量）；
   - G3 最细档在 15% 标准内。
   三种输出：`resolution_attributable`（G1∧G2∧G3）/ `resolution_window`（通过区间是有界窗口，
   窗口外某一档失败或误差随加密回升）/ `resolution_pair_only`（档数不足，单调性未测）。
3. **接进生产者**：`adjudicate_over_tolerance.py` 不再硬编码 `resolution_attributable`，
   而是向该模块取类别，并把 `ladder_guard` 明细与 `downgrades` 通道写进 JSON/Markdown/图
   （`downgrades` 与 `problems` 分开：配对本身没矛盾，被降级的是**标签**）。
4. **守卫**：`tests/test_method_selection_resolution_ladder_guard.py` 15 条，含负控制
   （单调阶梯 → attributable；回升阶梯 → window；两档 → pair_only；只认已提交产物）。

## ③ 计算结果（程序化注入，来自 `docs/reports/method_selection_open_flags_adjudication_m22787.json` 与裁定模块）

### 3.1 transport_velocity_correction 的四位阶梯（同一 `alpha=0.10` 臂）

| `dx` [m] | 前沿位置平均相对误差 | 相对 15% 标准 |
|---|---|---|
| 0.04 | 17.06 % | 超限 |
| 0.03 | 10.65 % | 通过 |
| 0.02 | 12.48 % | 通过 |
| 0.01 | **16.39 %** | **重新超限** |

- 回升量：`0.03→0.02` **+1.83 pp**，`0.02→0.01` **+3.91 pp**；
- 通过区间（窗口）= `dx ∈ [0.02, 0.03] m`；
- 结论：**`resolution_window`**。登记值 16.963% 不是「细化就消失的离散化症状」——
  在同一个标定系数下细化到 0.01 m 它**回来了**（与 `docs/xval_sphinx_tvc/RESOLUTION_LADDER.md`
  §2–§3 已有的机制说明一致：`alpha` 不是分辨率无关的，每步位移 `|dr| = alpha h²|∇·|` 与分辨率无关，
  而同一物理时长内的施加次数 ∝ 1/dx）。

### 3.2 wcsph_wall_no_slip 的阶梯

已提交的多档产物只有设计时运行（`H/dx = 8 → 4.34 %`，`H/dx = 16 → 0.48 %`），**两档**；
超限记录 32.649% 来自另一个产物（`benchmarks/couette/out/results.json`，单档 `H/dx = 8`），
两者终止时刻不同，其共享档**不合并**。结论：**`resolution_pair_only`** ——
「细化能救」在这里同样只是与数据相容，没有被仓库自己的数据证明。

### 3.3 定级变化（m22787 → m22957）

| 类别 | m22787 | m22957 |
|---|---|---|
| `resolution_attributable` | 2 条 | **0 条** |
| `resolution_window` | — | 1 条（`x_transport_velocity_coarse`） |
| `resolution_pair_only` | — | 1 条（`x_couette_wall_shear`） |
| `reading_mobility` / `genuine_accuracy_failure` / `documented_trap_arm` | 1 / 3 / 1 | 1 / 3 / 1（不变） |
| 决策泄漏（严格模式下仍在标准内被推荐） | 0 | 0（不变） |

**对决策的影响**：没有变化，因为严格模式本来就按最差记录把 `transport_velocity_correction` 排除
（`docs/reports/method_selection_strict_bound_m21537.json`）。变化的是**证据标签的诚实度**：
账本里不再有任何一条记录被称作「离散化症状」。

## ④ 结果图

`docs/figures/method_selection_resolution_ladder_m22957.png`
（左：TVC 四位阶梯 vs 15% 标准线，橙色阴影=通过窗口，最细档越线；
右：Couette 设计时阶梯两档 + 另一产物的单档超限记录）。
`docs/figures/method_selection_m22787/open_flags.png` 已按新类别重绘（新增
`resolution_window` / `resolution_pair_only` 配色）。

## ⑤ 注意点 / 局限

1. **没有跑新算例**。本单只重读已提交产物；第五档 `dx = 0.005`（把 `0.02→0.01` 放进渐近组）
   已递交集群（`cj4494`，排队中，队列 90+ 深），未在本报告窗口内返回——裁定模块已把它声明为
   可选数据源 `docs/xval_sphinx_tvc/data/ladder_dx0005.json`，一旦提交即自动进阶梯，无需改代码。
2. **不覆盖「换系数后的窗口」**：`alpha` 扫掠（`sweep_dx001_alpha.json`，0.15 → 14.3% 通过）
   说明窗口可以平移，但不改变「固定系数下不是收敛」的结论。
3. **Couette 一侧结论弱**：只有两档，`resolution_pair_only` 是诚实降级，不是否证。
   要把它变成趋势需要 `H/dx = 32` 一档。
4. 类别的**判据是声明式的**（G1/G2/G3 写在模块 docstring 与 `classify()` 里），
   不是从数据反推的阈值调参。

## ⑥ 误差判定

| 关键量 | 实测 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 阶梯重算 vs 注册表登记（`x_transport_velocity_coarse`） | 16.963% | 16.963%（注册表） | 0 pp | 是（逐位一致，测试 `test_guard_reads_the_committed_tvc_ladder` 断言 `rel_tol=1e-9`） |
| 阶梯重算 vs 注册表登记（`x_couette_wall_shear`） | 32.649% | 32.649% | 0 pp | 是 |
| TVC 最细档前沿误差 `dx=0.01` | 16.39% | Martin & Moyce (1952) 数字化实验 | 16.39% | **否**（>15% 业主标准，正是本单结论；不粉饰） |

本单不含新求解器计算，因此「精度 vs 解析解」只有上表第三行一个量，其**超限**即为结论本身。

## ⑦ 验证

```
PYTHONPATH=src python -m pytest tests/test_method_selection_open_flags_adjudication.py \
    tests/test_method_selection_resolution_ladder_guard.py tests/test_method_selection*.py \
    tests/test_xval_method_selection*.py -q
# 137 passed
PYTHONPATH=src python benchmarks/method_selection/adjudicate_over_tolerance.py --check
# OK: adjudication documents are up to date
```

## ⑧ 给下次自找活的建议

1. `cj4494`（`dx=0.005` 第五档）落地后把 `ladder_dx0005.json` 提交，重跑裁定：
   若 `0.01→0.005` 仍回升，窗口结论升级为「系数-分辨率耦合的定量规律」；若回落进 15%，则窗口下边界要改写。
2. `wcsph_wall_no_slip` 缺 `H/dx = 32` 一档 → 建议总账新条目
   `XVAL-SPH-COUETTE-HDX32-RUNG`（验收：`H/dx=32` 一档提交，壁面剪切误差 <15%，且阶梯单调性可判）。
3. `csf_surface_tension` 的两条 `genuine_accuracy_failure`（23.8% / 45.7%）仍无阶梯可用，
   建议按同一 guard 口径补一条曲率阶梯（`R/dx` 三档）。