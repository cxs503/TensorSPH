# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 3-D 记录完整性闸门（height 算子恒零：可判定性边界）

TensorSPH · 分支 `agent/m22724` · mission 22724 · 锚点 `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`

**结论（本轮增量，不含新判决）：m22606 入账的「运行中 `height_operator_m` 落盘恒 0」缺陷，在提交代码里
不可复现 —— 两个 gauge 读的是同一张站位掩膜、同一份位置张量，因此「slab 湿 / height 恒零」是记录缺陷
（或非提交代码检出产的记录），不是物理读数。本轮把它从自由文本注记升级为机器判据 `record_integrity`：
这类记录的第二算子被判定为不可用（`admissible=false`），判决本身不变（仍为 honest FAIL，判据只用 slab 列）。**

## ① 案例与判据

Kleefsman et al. (2005) *J. Comput. Phys.* **206:363–393**，DOI 10.1016/j.jcp.2004.12.007：3.22 × 1.00 × 1.00 m
水槽，左端 1.228 × 0.550 m 水柱溃决，撞击 x = 2.476 m 处 0.161 × 0.161 × 0.403 m 方箱（仅挡 40 % 槽宽）。
实验锚点（Fig. 19 右 panel，m20884 数字化）：**峰值 0.370 m @ 1.7027 s；到达 0.3591 s（0.02 m 阈值）**；
账本容差 **< 10 %**。站位 = 箱面 −0.05 m = 2.426 m，定孔径 0.08 m。

同一站位上落盘**两个**读数算子（`tensorsph.kleefsman_obstacle_3d`）：

| 算子 | 定义 | 用途 |
|---|---|---|
| `slab_height` | `n·dx³ / (aperture_x · blocked_width)`，体积当量水位 | 冻结协议主指标（判决用） |
| `peak_height` | 站位掩膜内粒子的 `y` 最大值（最高湿点） | 2-D `height` 算子的 3-D 对应，交叉校验 |

## ② 怎么做的（本轮四步）

1. **静态复算**：`_station_mask()` 被两个算子共用，`slab_height` 与 `peak_height` 在同一 `run_case`
   采样点被连续调用、中间无任何写操作 —— 同一掩膜非空时，`slab > 0` 必然与 `max y` 同号成立。
   逐版核对 `src/tensorsph/kleefsman_obstacle_3d.py` 的 `peak_height`：`d120345a`（#826 / m21728）、
   `146e5660`（#860 / m22082）、`b5e0e492`（#909 / m22552）**三个版本代码逐字相同**，均为
   `float(pos[mask, 1].max())`。
2. **合成反例穷举**（`tests/test_kleefsman_3d_record_integrity.py`）：随机点集、空点集、站位窗外点集、
   贴地点集、30 个贴地 + 1 个抬高粒子。结果是：唯一能产出「slab 湿 / height 零」的构造是
   **掩膜内所有粒子 y ≤ 0**（贴地退化），而记录里每个湿样本有 ~31 个粒子（`slab` 0.12 m 对应 31 个 `dx=0.05` 粒子），
   一个粒子抬高到 y = 0.025 m 就足以让 height 列非零。
3. **落盘链路排查**：`benchmarks/kleefsman_obstacle_3d/run.py`（流式写盘）、`diag_abort.py`（崩溃诊断写盘）
   两条写出路径都把 `row["height_operator_m"]`（即 `run.height[-1]`）原样写出；`save_checkpoint`/`load_checkpoint`
   的 `height` 键与 `slab` 键对称，`CHECKPOINT_FINGERPRINT_FIELDS` 不含算子定义。三条恢复路径都没有把该列清零的代码。
4. **判据化**：新增 `benchmarks/kleefsman_obstacle_3d/record_integrity.py`，`integrity(series)` 报告
   记录是否自洽，并把结果接进 `recovered_verdict.score_run()` 的 `record_integrity` 字段 ——
   缺陷从脚注变成判决载荷里可断言的字段。

## ③ 结果（程序化注入，`record_integrity.integrity` 直接读数）

| 记录 | 样本数 | 湿样本 | 湿样本中 height=0 | `slab` 峰 [m] | `height` 峰 [m] | 首次见水 [s] | 闸门 |
|---|---|---|---|---|---|---|---|
| cj3887 dx=0.05 | 468 | 354 | 354 | 0.120192 | 0.000000 | 0.500175 | `admissible=false`，height 列退化 |
| cj3888 dx=0.04 | 327 | 194 | 194 | 0.093300 | 0.000000 | 0.466830 | `admissible=false`，height 列退化 |

判决不变（m22606，honest FAIL）：dx=0.05 峰误差 **67.5 %**、到达误差 **40.0 %**；dx=0.04 到达 **30.3 %**，
峰 WITHHELD（记录止于 1.144 s < 1.7027 s）。本轮**没有**产生新的判决，也不得据此改动任何判决。

![height 列退化诊断](../benchmarks/kleefsman_obstacle_3d/results/recovered/fig_record_integrity_m22724.png)

## ④ 误差判定（逐项，显式回答）

* 本轮输运的是**记录判据**，不对物理量新增误差声明：`slab` 峰误差 67.5 % / 到达 40.0 %（dx=0.05）延续 m22606 的 honest FAIL。
* 闸门自身的判对/判错：两条已入账记录均被正确标记为退化（2/2），合成自洽记录不被误标（1/1），
  格式噪声（1e-6 量级）不被误判为退化（1/1），空记录/干记录不误伤（2/2）。
* **是否 < 10 % / < 3 %？否** —— 案例验收（峰值与到达 < 10 %）本轮仍未达成，原因与 m22606 相同
  （3-D 臂峰值 0.12 m vs 0.370 m，非网格收敛的量级差）。

## ⑤ 注意点 / 局限（如实）

1. **不可复现性本身是结论**：三个版本 + 三条写盘路径 + 合成反例全部排除后，提交代码无法产出该记录。
   最可能的来源是「作业运行时的工作树与提交不一致」（staged tar 与已提交代码不同步），
   但**没有直接证据**（作业回执不含 run.py 的哈希），因此不写成定论，只记为「提交代码不可复现」。
2. **闸门不修数据**：`record_integrity` 不改写任何 CSV，也不把退化列补零成测量值；它只拒绝把该列当读数用。
3. **m22606 的 xfail 护栏保留**（`tests/test_kleefsman_3d_recovered_verdict.py`）：它钉的是「已入账记录的
   事实」，与代码修复无关，因此**永远不会**因为修代码而翻绿；本轮的闸门测试是它的机器化补强，而非替代。
4. **新记录尚未落地**：本轮递交三个满窗口 3-D rung 以取「覆盖实验峰时刻的完整记录」——
   `cj4323`（dx=0.05）、`cj4324`（dx=0.04）、`cj4326`（dx=0.03，为凑满三格阶梯）；
   截至本单收尾三者均为 `queued`（队列位次 33–35，前置 34 个作业）。**没有任何新数字被据此写出**，
   收敛阶梯与外推 `f_inf` 仍未回答。
5. **阈值选择**：`WET_EPS = 2e-4 m`（到达阈值 0.02 m 的 1 %），远高于六位小数格式噪声、远低于任何真实读数量级；
   改这个阈值会改变「湿样本」计数，但不改变两条记录退化与否的结论（height 列全零，slab 峰 0.09–0.12 m）。

## ⑥ 产物

* `benchmarks/kleefsman_obstacle_3d/record_integrity.py`：`integrity()`（纯函数，判退化/是否可用）
  与 `station_reads_share_mask()`（该判据成立所依赖的算子不变量）
* `benchmarks/kleefsman_obstacle_3d/recovered_verdict.py`：`score_run()` 输出新增 `record_integrity` 字段
  （判决逻辑未改，`verdict.json` 同步再生成）
* `tests/test_kleefsman_3d_record_integrity.py`：算子不变量 3 例 + 闸门逻辑 3 例 + 记录事实 3 例
* 复现：`python -m pytest tests/test_kleefsman_3d_record_integrity.py tests/test_kleefsman_3d_recovered_verdict.py -q`
  → 15 passed, 1 xfailed