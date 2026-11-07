# IN53-N02 2-D WCSPH 波浪水槽：造波—色散—消波三锚（本轮执行报告）

> 状态：**部分完成（honest partial）**。水槽基础设施与判据矩阵已就位并机器可算，但
> 本轮集群执行被**系统性进程崩溃**打断，只有 (a) 静水压、(h) pytest 两项判据完成裁决；
> (b)(c)(d)(e)(g) 因算例未能跑完而**未裁决**（不是通过，也不是失败的物理结论）；
> (f) 跨求解器互裁为 **WITHHELD**（本工作树内没有 TensorLBM 二维重力波水槽驱动）。

## 1. 物理背景与算例设计

2-D 弱可压 SPH（WCSPH）数值波浪水槽，水深 `h = 0.5 m`，左端松弛区（waves2Foam 式
速度松弛）造波，右端海绵层速度阻尼消波，柱体积分探针测水位，单频最小二乘 + 两点相位差
提取实测色散，包络法 / 去趋势法估计反射系数。求解器复用仓内
`tensorsph.droplet_impact.rhs`（密度求和 + Tait 状态方程 + 对称化压力梯度 + Morris 粘性 +
Adami 壁面压力外推），本节点新增的是**算例**：造波器、海绵层、探针、后处理与解析锚。

算例表（`benchmarks/wave_tank/run.py::CASES`，16 个算例，判据 (a)–(h) 逐项绑定）：

| 算例 | 用途 | kh | dp/h | c0/√(gh) | 步数 | 粒子数 |
|---|---|---|---|---|---|---|
| hydrostatic | (a) 静水压 vs ρgh | 3.0 | 0.020 | 10 | 4974 | 15700 |
| dispersion_kh0.5/1/2/3 | (b) 线性色散 | 0.5–3.0 | 0.020 | 10 | ~9700 | 12550 |
| decay10 | (c) 10λ 后波高保持 | 3.0 | 0.025 | 6 | 14771 | 38840 |
| solitary | (d) 孤立波 c / 波形 / 质量 | — | 0.020 | 10 | ~10800 | 37206 |
| reflection(+control) | (e) 海绵层反射系数 | 3.0 | 0.020 | 10 | ~9700 | 12550 |
| ladder_dp0.020…0.0075 | (g) 分辨率收敛 | 3.0 | 0.020–0.0075 | 10 | 5828–15542 | 9400–66766 |
| csweep_c5/c20 | 人工声速扫描（诊断） | 3.0 | 0.020 | 5/20 | 3420–18100 | 10450 |

## 2. 方法要点

* **造波**：松弛区把入口区速度按 `w(x)=((x_r−x)/x_r)^2` 混向 Airy（或孤立波）解析速度场，
  线性造波传递函数只用于**定冲程**，实测波高独立报告，不做“假设即结论”。
* **色散**：四个探针的单频锁相振幅/相位 → 空间波数拟合；两点相位差作交叉核验
  （`celerity_error_spatial` 为判据值，两点法为对照）。
* **消波**：`v ← (1 − a(x) dt) v`，`a(x)=A ω ((x−x_s)/(L−x_s))^2`，只阻尼速度（密度由求和
  得到，不能向目标松弛，否则破坏质量守恒）。
* **反射**：探针包络法 `R=(Hmax−Hmin)/(Hmax+Hmin)`，并以“海绵层关闭”的对照算例验收
  探针阵列是否真能看见反射体（要求控制组 R ≥ 0.8）。
* **分辨率阶梯**：dp/h ∈ {0.02, 0.015, 0.01, 0.0075}，同一探针口径、同一物理时长，误差必须
  随 dp 减小而单调下降（判据 g）。

## 3. 本轮实测结果

### 3.1 判据 (a) 静水压 —— **PASS**

潭内 6 个水深层带的平均绝对误差 **0.495 %**（阈值 1 %）。这说明压力闭合与静水格点初始
化在含自由面水槽里没有引入可察觉偏置——它是后续所有波要素测量的地基。

### 3.2 判据 (h) pytest —— **PASS**

`74` 项：`tests/test_wave_tank.py` 33 项（锚/估计器/几何，7.2 s）+ `tests/test_wave_flume.py`
41 项（45.2 s，含一次短积分集成用例）。本节点修复了 `test_wave_flume.py` 中一处真实红测试
（`WaveFlumeRun.mass_drift` 属性缺失 → 现以 `mass_final` + 守恒断言实现）。

### 3.3 人工声速扫描（诊断，非判据）—— 已完成 1/3 点

| c0/√(gh) | 相位速度误差 | 反射系数 R | H/H0 |
|---|---|---|---|
| 5 | **9.56 %** | 0.481 | 0.351 |

低人工声速下相位速度偏慢 9.6 %、波高沿程衰减严重，与弱可压性（`O((c/c0)²)` 相速偏置）
及短波在 50 粒子/水深的数值耗散方向一致；这是 c0 扫描应有的物理趋势，待 c0=10/20 两点
补齐后才能给出定标曲线。

## 4. 未决事项与原因（honest）

* **(b)(c)(d)(e)(g) 未裁决**：集群执行中，长算例**几乎全部**以
  `free(): corrupted unsorted chunks`（SIGABRT, rc=-6）在 30–512 s 处崩溃，同一算例重试
  偶有通过（`csweep_c5` 第 2 次尝试成功）。已排除：内存不足（节点余量 ~950 GB）、
  OOM-kill（信号为 SIGABRT 而非 SIGKILL）、数值发散（代码会抛 `FloatingPointError` 而不是
  abort）。**本节点不把未跑完的算例写成“未通过物理判据”**，只记为未裁决，并把它登记为
  下一轮的第一优先级缺陷（疑点：loong64 torch 轮子 + 长时间配对循环的原生堆损坏；
  `PairListCache` 重建路径是首要怀疑对象）。
* **(f) WITHHELD**：`cross_solver.py --write-spec` 已生成与代码无关的工况定义
  `runs/wave_tank/cross_case.json`；本工作树内 TensorLBM 只有 2-D 自由面单步算子与 3-D Airy
  入口边界，缺二维重力波水槽驱动，无法产出同工况对照。按纪律记为 WITHHELD，不编造。

## 5. 证据与复现

```
python benchmarks/wave_tank/run.py --case hydrostatic --threads 32 \
    --json-out runs/wave_tank/hydrostatic.json
python benchmarks/wave_tank/run_group.py --group 3          # 集群一组
python benchmarks/wave_tank/run_cases.py --cases reflection --threads 8 --attempts 3
python benchmarks/wave_tank/summarise.py                    # -> runs/wave_tank/verdict.json
python benchmarks/wave_tank/cross_solver.py --compare none  # -> cross_solver.json (WITHHELD)
python benchmarks/wave_tank/make_figures.py                 # -> docs/figures/wave_tank_*.png
```

机器可读证据：`runs/wave_tank/<case>.json`（配置 + 探针统计 + 判据 + 墙钟）、
`runs/wave_tank/verdict.json`（判据矩阵，含 missing 列表）、`runs/wave_tank/viz_*.npz`
（粒子场快照，用于图 4）。图 4 由 `viz_case.py` 的**短算例**（3730 / 3029 步，分别 82 s /
102 s，均正常结束）生成，代表自由面形状与粒子场，不代表长时统计。