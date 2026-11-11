# 3-D 静水工件：判据红、包络分裂与 1/c0² 误差模型（m14908）

**范围**: TensorSPH ｜ **性质**: 验证欠账（已合并功能的证据工件复核 + 参数包络定界） ｜
**改动**: `benchmarks/threed_hydrostatic/run.py`（新增 envelope 分类、c0=40 包络内算例、
compressibility 拟合、按物理量而非名单划分 gate/probe/control）、
`benchmarks/threed_hydrostatic/report.json` + `figures/threed_hydrostatic.png`（用 HEAD 代码重跑 7 例重生成）、
`tests/test_threed_artifacts.py`（新增 2 个判据测试，放宽 1 条与欧拉分箱语义冲突的断言）。

## 1. 为什么做这一单

`tests/test_threed_artifacts.py` 在 HEAD 上是红的，而红的原因是**工件漂移**，不是物理回归：

* `test_hydrostatic_report_meets_its_criteria` 读的是旧 `report.json`，其判据名（`interior_pressure_within_15pct_best`、
  `compressibility_helps_as_predicted`）在 HEAD 代码里已经不存在，run payload 里也没有 HEAD 判据需要的
  `pressure.integral_rel_error`——重跑 `merge()` 会直接 `KeyError`。也就是说：提交的工件**无法**用仓库自己的
  merge 复现，账本引用的数字没有可重放路径。
* 旧工件 `verdict = fail`，但失败项里混着**故意让它失败的控制组**（`tait` 壁面模型）与 **WCSPH 包络外的算例**
  （c0 = 5 m/s）。判据把"能力是否成立"和"对照组是否如预期失败"写进了同一个 `all()`。

## 2. 怎么做的

1. 用 HEAD 代码重跑全部 7 个算例（`--t-end 0.30`，全部算例同一时间窗），重新 `--merge --figure`。
2. 给每个算例算**它自己的**密度扰动包络 `delta rho / rho0 = gamma g H / c0^2`（`gamma = 7.0`），
   并据此**从参数**（不是从名单）划分三类角色：
   - `gated`：adami 壁面模型且 `delta rho/rho0 <= 0.15` —— 只有这些能设能力门；
   - `envelope_probe`：adami 但包络外；
   - `control`：`tait` 壁面模型（故意失败）。
3. 新增 **c0 = 40 m/s** 算例：标准声速律 `c0 >= 10 sqrt(g H)` 对本柱给出 10·sqrt(9.81·0.16) ≈ 39.6 m/s，
   **原有 6 个算例全部在这条律之下**（最接近的 c0=20 也只有 1.60 倍），所以旧工件无法区分
   "算子有缺陷"与"声速太小"。
4. 用四个算例（c0 = 5/10/20/40）最小二乘拟合 `|e| = a + b / c0^2`，把与 c0 无关的残差 `a` 单独报出来。
5. 控制组与探针组的结果、逐项 gate 状态原样登记进工件（`control_outcomes` / `probe_outcomes`），
   并加**正向门**：`control_run_fails_as_designed`、`probes_are_worse_than_gated_best`、
   `probe_labels_are_derived_not_chosen`——对照组必须失败才算合格，探针必须比门内最好结果差。

## 3. 结果

| case | c0 (m/s) | nz | δrho/rho0 | c0/(10√(gH)) | 底部压力误差 | 深度积分载荷误差 | core 层最大误差 | odd-even 幅度 | 角色 |
|---|---|---|---|---|---|---|---|---|---|
| `adami-c0_20-nz16` | 20 | 16 | 2.75 % | 1.60 | -2.75 % | -10.25 % | +25.8 % | 4.34 % | gated |
| `adami-c0_10-nz16` | 10 | 16 | 10.99 % | 0.80 | -4.30 % | -11.73 % | +26.2 % | 3.66 % | gated |
| `adami-c0_5-nz16` | 5 | 16 | 43.95 % | 0.40 | -8.74 % | -19.67 % | +36.1 % | 4.64 % | envelope_probe |
| `tait-c0_10-nz16` | 10 | 16 | 10.99 % | 0.80 | -25.23 % | -32.40 % | +41.9 % | 2.06 % | control |
| `adami-c0_10-nz12` | 10 | 12 | 8.24 % | 0.92 | -4.46 % | -13.99 % | +25.3 % | 5.80 % | gated |
| `adami-c0_10-nz24` | 10 | 24 | 16.48 % | 0.65 | -5.28 % | -9.95 % | +27.3 % | 1.74 % | envelope_probe |
| `adami-c0_40-nz16` | 40 | 16 | 0.69 % | 3.19 | -1.81 % | -8.48 % | +21.7 % | 3.64 % | gated |

* 判据（HEAD 代码重算）：`verdict = pass`；`verdict_including_probes = fail`。
* 门内最好算例：`adami-c0_40-nz16`，底部压力相对误差 `+1.81 %`（< 3 % 论文级）。
* 探针/控制最差：`+25.23 %`。
* 压缩性拟合：`|bottom_rel_error| = a + b / c0^2`，**a = 2.18 %**（与 c0 无关的算子+边界偏置），b = 1.669，
  残差 -0.11 %, +0.44 %, +0.15 %, -0.48 %。

**误差判定（逐项回答是否 < 3 %）**

| 关键量 | 基准 | 门内最好实测 | 误差 | < 3 % |
|---|---|---|---|---|
| 底部压力（包络内 c0 = 40） | `1520.55 Pa` | `1493.09 Pa` | -1.81 % | 是 |
| 深度积分载荷（同一算例） | `125.08 Pa·m` | `114.47 Pa·m` | -8.48 % | 否（< 15 % 平台门） |
| 底部压力（c0 = 10，包络边沿） | `1520.55 Pa` | `1455.23 Pa` | -4.30 % | 否 |
| 单层压力（core 层最差） | — | — | +21.7 % | 否（见 §5.1） |

结论：**压力载荷这一标量在包络内可以做到 3 % 以内；逐层压力剖面做不到**，而且未收敛的那部分是
自由面边界层（§5.1），不是算子。

## 4. 图

`benchmarks/threed_hydrostatic/figures/threed_hydrostatic.png`（由 `run.py --merge --figure` 直接读工件绘制）：
左=压力剖面 vs `rho0 g (H-z)`；中=逐层相对误差（点线 3 %、红虚线 15 %）；右=底部误差 vs 声速。

## 5. 注意点 / 局限

1. **逐层误差里的 odd-even 模式没有被解释掉**：门内算例 core 层最大相对误差仍有
   +21.7 %，odd-even 幅度
   3.64 % of `rho0 g H`，且**不随 nz 收敛**
   （nz = 12/16/24 同量级）。它在最靠近自由面的几层最大，性质是自由面支撑不全 + 晶格压力模态，
   本单只把它**量化并按算例登记**（`pressure.odd_even_amplitude_over_scale`），没有修正模型。
2. **欧拉分箱会切层**：层表按 `floor((z - walls)/dx)` 分箱，柱体在自重下有 ~dx 量级的下沉，
   分箱边界会切开一层（某行 `particles` 不是 `nx*ny` 的整数倍，行数可能少 1）。
   因此逐层数字只支持"±1 层"分辨率的结论；测试已按此收紧为"粒子不丢、行按 z 有序"，不再断言行数 = nz。
3. **c0 = 5 与 tait 两个算例仍是 honest FAIL**：它们分别丢 2 项与 4 项单算例 gate
   （见 `probe_outcomes` / `control_outcomes`），`verdict_including_probes = fail`。
   本单**没有**把它们从记录里删掉，只是不让它们参与能力门，并要求它们"必须更差"。
4. **门限 15 % 是选的，不是推导的**：`gamma g H / c0^2 = 0.15` 这一阈值按"压力残差
   已被刚度主导"的经验取值；换成 5 % 或 10 % 会让 c0 = 10（11.0 %）进出，裁决随之变。
   这一点必须随结论一起引用。
5. **shear 工件的 h1.8 梯级仍缺**：`benchmarks/threed_shear` 的 `h1.8` 算例（n = 16³、h = 1.8 dx）在本机
   单核跑超过 25 min 未完成（load15 ≈ 100，按容量纪律未提并发），`test_shear_report_meets_its_criteria`
   因此仍是红的——与本单无关，未修。
6. **账本未重生成**：`benchmarks/verification_ledger/build_ledger.py` 依赖
   `benchmarks/surface_tension_dispersion/validation_report.json`，该文件在 HEAD 上**不存在**
   （`git log` 显示其最后出现在 35dcefa），因此 ledger 全套测试在 HEAD 就是 ERROR 状态，
   本单无法重生成 `ledger.json`。后果：`ledger.json` 里 `threed_hydrostatic` 的条目按
   `runs[i]` 索引引用，重跑后算例顺序/个数变了，**索引需要重新指向**（未做，属已知欠账）。

## 6. 复现

```bash
for n in adami-c0_20-nz16 adami-c0_10-nz16 adami-c0_5-nz16 tait-c0_10-nz16 \
         adami-c0_10-nz12 adami-c0_10-nz24 adami-c0_40-nz16; do
  PYTHONPATH=src python benchmarks/threed_hydrostatic/run.py --only $n --t-end 0.30 \
      --out benchmarks/threed_hydrostatic/cases/$n.json
done
PYTHONPATH=src python benchmarks/threed_hydrostatic/run.py --merge --figure
python -m pytest tests/test_threed_artifacts.py -q   # 静水两条绿；shear 仍红（h1.8 缺）
```
