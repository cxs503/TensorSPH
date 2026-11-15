# 入流时钟的第二级缺陷：**全局**通量时钟喂不动壁面附近的流线（行内堆叠 → 伪高速）

锚点 `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`（Xiangyu-Hu/SPHinXsys `test_2d_fsi2` +
`FSI::ForceFromFluid`）· 仓 TensorSPH · 分支 `agent/m22323` · 域 SPH / FSI
上游增量：`docs/xval_fsi2/INFLOW_CLOCK_STALL.md`（m22030，把几何时钟换成**全局**质量通量时钟）

## ① 案例介绍

Turek–Hron FSI2：2-D 通道 `L = 2.5 m × H = 0.41 m`，`D = 0.1 m` 圆柱（心 `(0.2, 0.2)`）+
`L_b = 0.35 m × t_b = 0.02 m` 弹性梁，`Re = 100`（ν = 2.0e-4 m²/s），
ρ = 1000 kg/m³，`U_mean = 0.2 m/s`，入口抛物 `u(y) = U_max 4 y (H−y)/H²`（`U_max = 0.3 m/s`），
`c0 = 5 m/s`（Mach 0.06），`dx = 0.02 m`，`h = 1.3 dx`，`dt = max_courant h / c0 = 1.04e-3 s`。

上一轮修掉的是「入口注入被**最慢粒子**钉死」（`injected = 0`、体平均速度 0.19 `U_mean`、
下游通量 0.18 入口通量）。修法把几何时钟换成**全局**质量通量时钟：
每个 `dx/U_mean` 注入**一整列**（每行一个粒子）。

修好之后仍有不物理的残差（同一工件 `artifacts/xval_fsi2/rigid_cfd_dx002_fixed.json`）：
`injected 240 / removed 234`、质量 +0.23 %，但 **`bulk_speed = 2.06 U_mean`、`max|u| = 2.201 m/s`
（= 7.3 `U_max`）、`flux_ratio = 2.159`**，瞬时 `Cd = +19.77`（基准 3.2）。

本轮的假设（开工前写死，未看结果后修改）：**全局时钟按 `U_mean` 均匀喂每一条流线，
而入口剖面在近壁行的输运速度只有 `u(dx/2) = 0.14 U_mean`**（`y = dx/2 = 0.01 m`：
`4·0.01·0.40/0.41² = 0.0952 → ×U_max = 0.0286 m/s`）。于是近壁行被**过度注入
`U_mean/u(y_j) ≈ 7×`**，多余的粒子无处可去，堆在入口第一格；堆叠即局部密度尖峰→
伪压力→伪高速。**判据**：近壁行每格粒子数应为 1（流线自洽），而不是 6。

## ② 怎么做的

1. **最小改动**（`src/tensorsph/fsi2_turek_hron.py::_apply_open_boundaries`）：把单一标量
   `inflow_deficit` 换成**逐行累加器** `inflow_deficit_rows`（`FSI2State` 新字段）。
   行 `j` 位于 `y_j = (j+1/2) dx`，其累加器只吃**本行局部**入流速度
   `u(y_j) dt × ramp(t)`；攒满 `dx` 才在该行落一个粒子。总注入体积仍严格等于入口交付体积：
   `Σ_j u(y_j) ≈ U_mean H/dx`（中点求积），故 `Σ_j` 注入通量 = `ρ U_mean H`（逐行自洽）。
   标量 `inflow_deficit` 保留为逐行均值的**日志量**（后向兼容）。
2. **控制臂**：`benchmarks/xval_fsi2/rigid_cfd_precursor.py` 新增 `FSI2_INFLOW_CLOCK={row,global}`
   （`global` = 上一轮实现的行为等价复刻，只作对照，不进生产），并在 `rigid_step` 里加
   「入口第一格（`x < dx`）逐行堆叠普查」诊断 `inlet_strip_max_per_row / inlet_strip_n`。
3. **受控对比**：同初值、同 `dx`、同 `dt`、同线程数，**只改时钟**，各 800 步（t = 0.832 s）。
   刚体前驱（结构全冻结）把界面子迭代与杆件都排除，量到的就是流体那一半。
4. **守护**：新增 `tests/test_fsi2_row_clock.py`（5 项）；既有
   `tests/test_fsi2_inflow_clock.py`（4 项）在关闭启动 ramp 后照旧通过。

## ③ 计算结果（程序化写入，`artifacts/xval_fsi2/clock_ab_{global,row}.json`）

| 时钟 | 入口第一格粒子数 | 单行格内最大堆叠 | `max\|u\|` [m/s] | 体平均 [m/s] | 通量比 | 流体粒子数 | 总质量 [kg] |
|---|---|---|---|---|---|---|---|
| `global`（上一轮） | **49** | **6** | **2.298** | 0.291 | 1.23 | 2656 | 1062.5 |
| `row`（本轮） | 12 | **1** | **0.956** | 0.080 | 0.48 | 2602 | 1040.9 |
| 初值 | — | — | — | — | — | 2559 | 1023.65 |

* 单行格内最大堆叠 **6 → 1**，入口第一格粒子数 **49 → 12**（稳态应 ≤ 一列 ≈ 20）；
* 伪高速 **2.298 → 0.956 m/s**（−58 %）；`global` 臂复现了上一轮记录的 2.201 m/s（差 4 %，
  采样步不同），说明控制臂是同一条代码路径；
* 质量增量 **+3.79 % → +1.68 %**（启动压缩减半）；
* `row` 臂此时仍在启动段（ramp 在 t = 1.0 s 结束）：体平均 0.080 m/s = 0.4 `U_mean`、
  通量比 0.48、`Cd = +37`（压缩主导）⇒ **本单不给 Cd 精度判定**。

## ④ 结果图

`docs/xval_fsi2/assets/row_clock_ab.png`：(a) 入口第一格逐行粒子数（两条时钟，堆叠签名）；
(b) `max|u|` 时间序列（两条时钟，`U_max = 0.3 m/s` 参考线）。

## ⑤ 注意点（坑与适用范围）

1. **这是同一缺陷的第二级**：上一轮修「注入被最慢粒子钉死」，本轮修「注入被均匀化喂给慢流线」。
   两次都在 `_apply_open_boundaries` 里，两次都靠**逐行/逐流线**而不是全局限量解决。
2. **启动压缩仍在**：`row` 臂 0.832 s 内质量 +1.68 %。WCSPH 里 1 % 密度误差 =
   `c0² Δρ ≈ 250 Pa`，而动压 `½ρU² = 20 Pa`，所以启动段的 `Cd` 必然被压缩主导；
   出口按几何阈值删粒子（`x > L − dx/4`）而无出口通量条件，是剩余欠账（未修）。
   因此**不能**用启动段判 Cd。
3. **长时证据已递交未回**：`cj4041`（刚体前驱 9000 步，dx = 0.02）、
   `cj4040`（强耦合 12000 步，时窗 ≈ 12.5 s ≈ 5 个通道输运时间），两者在本单收尾时仍在排队，
   回件后应重跑 `benchmarks/xval_fsi2/verdict_body_force_transient.py` 判定；
   **本单不因它们未回而改判**。
4. **适用范围**：只对「入口有一阶壁面（`u(y=0)=0`）+ 拉格朗日注入」的开边界有效；含周期入口、
   含潮汐/波浪入口的算例不适用。`inflow_deficit_rows` 与 `case.dx` 绑定，
   变分辨率时必须重建（已用 `numel() != n_rows` 守卫）。
5. **行速度用解析剖面而非求解器速度**：注入量必须等于**边界条件交付**的体积，
   用求解器速度会引入反馈（慢粒子→少注入→更慢）。

## ⑥ 误差判定（显式回答「是否 <3 %」，本单无新解析基准量，逐项列判定性质）

| 判定项 | 基准 | 实测 | <3 %？ | 性质 |
|---|---|---|---|---|
| 单行格内最大堆叠 | 1（流线自洽） | 1（`row`）/ 6（`global`） | ✔ / ✘ | 结构判据 |
| 逐行注入计数 vs `u(y_j) t/dx` | 解析剖面 | 每行偏差 < 1 次注入 | ✔（整数舍入内） | 解析 |
| 总注入通量 vs `ρ U_mean H` | 解析 | 6 列时间偏差 < 1.5 列 | ✔ | 解析 |
| `max|u|` vs `U_max` | ≤ 1.5 `U_max` = 0.45 m/s | 0.956 m/s（`row`，启动段） | ✘ | 启动瞬态，见⑤.2 |
| **`Cd_mean` vs Turek–Hron 3.2（<15 %）** | 3.2 | **未达统计窗口** | **不适用** | **锚点验收仍未达成** |

**honest**：本单把 `Cd` 偏差链上的第二个缺陷定位并修掉（有受控数字与守护测试），
`Cd` 的量级链**尚未**进入可判定区；锚点的 `<15 %` 验收仍是 `WITHHELD`，不是 PASS。

## ⑦ 复现

```bash
PYTHONPATH=src FSI2_INFLOW_CLOCK=global python benchmarks/xval_fsi2/rigid_cfd_precursor.py \
    --dx 0.02 --steps 800 --sample-every 100 --threads 7 --out artifacts/xval_fsi2/clock_ab_global.json
PYTHONPATH=src FSI2_INFLOW_CLOCK=row    python benchmarks/xval_fsi2/rigid_cfd_precursor.py \
    --dx 0.02 --steps 800 --sample-every 100 --threads 7 --out artifacts/xval_fsi2/clock_ab_row.json
PYTHONPATH=src python -m pytest tests/test_fsi2_row_clock.py tests/test_fsi2_inflow_clock.py -q  # 9 passed
```

改动文件：`src/tensorsph/fsi2_turek_hron.py`、`benckmarks/xval_fsi2/rigid_cfd_precursor.py`、
`tests/test_fsi2_row_clock.py`、`tests/test_fsi2_inflow_clock.py`（仅关闭 ramp 的测试脚手架）。