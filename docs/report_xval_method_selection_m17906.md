# XVAL-SPH-METHOD-SELECTION-2026-11-12 · 返修与机理诊断（mission 17906）

锚点：`XVAL-SPH-METHOD-SELECTION-2026-11-12`（TensorSPH，CASE-WAVE-FLUME 的方法选型）。
上一轮 mission 17501 被审计**打回（reopened）**，判级 L2，理由见 `verdict_m17501.md`：
①发布报告 JSON 与数据不符（把存在的记录写成 `present:false`）；②`grade_matrix.py`
只看记录级 `failed` 标记，不看稳定性判据，数值发散的算例被当成正常行；
③全部交付物从未入库；④`test_report_artifact_matches_a_fresh_regrade` 失败；
⑤7 个预注册算例只落地 1 个且数值失稳。

本单=返修 + 机理诊断。**不重跑原 7 格栅格**（在负载 ≈370 的共享机上重跑同一发散
方法没有信息量），而是把「发散为什么发生」变成可复核的定量证据，并修掉让发散被
误读成数据的判级缺陷。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

2-D 数值波槽（`src/tensorsph/wave_flume.py`）：WCSPH（密度求和 + Tait 状态方程 +
δ-SPH 密度扩散 δ=0.1）+ 活塞一阶造波 + 下游幂次海绵层 + 启停窗（settle 0.8 s，
窗内 `ν=0.02`、`XSPH=0.1`）。目标波高由**活塞一阶传递函数**
`H/S = 2(cosh 2kh − 1)/(sinh 2kh + 2kh)`（Dean & Dalrymple 1991 eq. 6.38）给出，
无自由参数，因而是可以逐探头对标的解析基准。

| 参数 | 值 | 出处 |
|---|---|---|
| 静水深 h | 0.25 m（返修算例）/ 0.20 m（存档记录） | protocol §4 |
| 槽长 L | 3.2 m（按 λ 取整到 dx 的整数倍） | protocol §4 |
| 周期 T / 波高 H | 1.0 s / 0.16 h | protocol §4 |
| 声速 c0 / 运动粘度 ν | 30 m/s / 1e-6 m²/s | protocol §4 |
| δ-SPH / 海绵 | 0.1 / 强度 4.0、指数 3.0、长 0.9λ | protocol §4 |
| 分析窗 | start = 4.0 s、5.0 s（双窗） | protocol §7 |

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **判级缺陷修复**（`benchmarks/xval_method_selection/grade_matrix.py`）：
   新增 `stability()`，把 protocol §3 **预先写定**的两条稳定性判据
   （`max|v|/c0 < 1`、密度压缩 < 10 %）作为硬门，`verdict()` 合成
   `pass / fail_tolerance / fail_stability / fail_stability+tolerance`；
   新增 `gradeable` 状态：记录存在但分析窗无有效样本时记为
   `ungradeable`（证据缺口），**不再**伪造成 `present:false`，也不再被任何计数
   当成 0。所有图（`ladder/series/attenuation/sensitivity`）改用统一谓词
   `measurable()`，发散记录不进误差图。
2. **重生成发布工件**（`docs/reports/xval_method_selection_m17501.json`）：
   脚本 `grade_matrix.py` 从真实存在的记录重算，未执行的预注册格点在
   `predeclared_unexecuted` 里**逐条声明为缺口**（不静默丢弃）。
3. **机理诊断**（`benchmarks/xval_method_selection/diagnose_divergence.py`）：
   只读存档记录（`docs/reports/artifacts_wave_flume/wf_D20_unstable.json`），
   ①每 0.5 周期滑动的谐波最小二乘，得到逐探头基频振幅包络与 e 倍时间；
   ②双探头入射/反射分解 `A(x)=I e^{ikx}+R e^{-ikx}`，把「海绵不吸波」与
   「造波过强」分开；③给出 Mach 数过 1 的时刻与压缩历史。
4. **返修算例**（`benchmarks/xval_method_selection/run_cell.py`，3 个决策定向格点：
   D20 基线 / D13 阶梯 / D20+XSPH 机理探针）在共享机上排队运行；本报告如实标注
   它们的落地状态。

## 3. 计算结果（程序化注入，零手抄）

下表与图由 `diagnose_divergence.py` 直接写入，数字来源见
`docs/reports/xval_method_selection_diagnostics.json`。

### archive_h020（`docs/reports/artifacts_wave_flume/wf_D20_unstable.json`）

| 量 | 值 |
|---|---|
| 目标波高 H（活塞一阶传递） | 30.00 mm |
| 模拟时长 / 粒子数 | 3.70 s / 6400 |
| max\|v\| / c0 | 3.552（门 1.0）|
| max 密度压缩 | 29.9 %（门 10 %）|
| 逃逸粒子 | 144 |
| Mach 1 时刻 | 3.25 s |

| 窗口 [s] | A0 [mm] | A1 [mm] | A2 [mm] | A3 [mm] | R01 | R23 |
|---|---|---|---|---|---|---|
| 0.00–1.00 | 0.03 | 0.03 | 0.03 | 0.03 | 1.01 | 1.00 |
| 0.50–1.50 | 0.48 | 0.39 | 0.13 | 0.41 | 1.13 | 0.74 |
| 1.00–2.00 | 3.74 | 3.18 | 3.69 | 4.13 | 0.54 | 1.33 |
| 1.50–2.50 | 10.89 | 15.15 | 14.97 | 10.09 | 1.83 | 0.72 |
| 2.00–3.00 | 24.19 | 30.81 | 32.38 | 32.93 | 0.68 | 0.68 |
| 2.50–3.50 | 35.82 | 26.10 | 16.56 | 25.47 | 0.57 | 3.24 |

振幅包络的 e 倍时间（最小二乘）：0.35–0.37 s。

![growth.png](docs/figures/xval_method_selection_m17501/growth.png)

![divergence.png](docs/figures/xval_method_selection_m17501/divergence.png)

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

* **「有记录」≠「有证据」**：存档记录 175 kB、4 个探头、971 个采样点，看上去
  是一份完整数据；但它只模拟到 3.70 s，而预注册分析窗从 4.0 s 起——任何对它的
  「波高误差」都是无源之数。上一轮把它写成 `present:false` 属于另一头错误。
* **发散是缓变的**：Mach 数从 1 往上冲之前，振幅包络已经连续若干周期单调增长，
  这正是为什么「只看波高误差、不看诊断量」的判级会让一次爆炸的算例通过。
* **双探头反射比在瞬态下不可全信**：两对探头的 R 值不一致（见 §3 表），因为
  分解假设稳态；这里只用它定性区分「入射分量本身在增长」（→ 造波/数值增益，
  不是单纯反射叠加），不用它给出反射系数的定量值。
* 结论只覆盖：顺直渠、规则波、2-D、H/h ≤ 0.16、kh ≈ 1.2（过渡水深）、无破碎。
  不外推到破碎波、三维、主动吸收。
* 本机 load15 长期 >100（本次诊断期间 ≈190–370），因此**没有**在本机起新的
  长算例；机理结论全部来自已提交记录，重算部分排在集群/后台队列并如实标注状态。

## 5. 误差判定（对基准逐项，显式回答「是否 <3%」）

| 关键量 | 基准 | 本单结果 | 是否 <3% |
|---|---|---|---|
| 波高 G1（逐探头 `|H_meas−H_target|/H_target`） | 活塞一阶传递（解析） | **无可用数字**：存档记录在 4.0/5.0 s 窗内无样本，判级器拒收（`ungradeable`）；返修算例见 §3 状态 | **无法回答（否，不构成达标）**——原因的定量证据是 §3 的 Mach/压缩表：记录在 3.25 s 越过 `max|v|/c0 = 1`，压缩远超 10 %，属数值发散，其后的波形不是物理波 |
| 稳定性 `max|v|/c0` | 门 1.0（protocol §3） | 见 §3 表 | 否（>1，判 `fail_stability`） |
| 密度压缩 | 门 10 %（protocol §3） | 见 §3 表 | 否 |
| 目标波高 H（解析自洽） | 活塞传递函数，`spec` 自洽 | `target_height` 由 `wave_height_target(spec)` 复算，与记录一致（CI 断言 rel=1e-12） | 是（0 %，属恒等式校验，非物理精度） |

结论：本锚点在**精度门（<15 %，更不用说 <3 %）上仍是 honest-FAIL**——上一轮
报告的「97 % 误差」本身也是无源之数。本单把失败从「看起来像通过/像有数」变成
「有诊断量支撑的、可复现的失败」，并给出唯一站得住的下一步：先让稳定性过门
（`max|v|/c0 < 1`），再谈波高精度。

## 6. 复现

```bash
# 判级（重生成发布工件 + 四张图）
PYTHONPATH=src python benchmarks/xval_method_selection/grade_matrix.py \
    --out docs/reports/xval_method_selection_m17501.json \
    --figures docs/figures/xval_method_selection_m17501
# 机理诊断（本报告 §3 的表与图）
PYTHONPATH=src python benchmarks/xval_method_selection/diagnose_divergence.py
# 单测（含「发散记录永不通过」「太短的记录=缺口不是 0」两条新判据）
PYTHONPATH=src python -m pytest tests/test_xval_method_selection_matrix.py -q
# 返修算例（每格一个进程，可 shard）
PYTHONPATH=src python benchmarks/xval_method_selection/run_cell.py 0 --out runs/xval/cell_0.json
```
