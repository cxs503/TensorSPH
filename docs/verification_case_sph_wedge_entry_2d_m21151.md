# CASE-SPH-WEDGE-ENTRY-2D — m21151：验收测试完整性缺陷修复 + 已发布阶梯的可复现性裁定

**锚点**: `CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力）
**参考**: Zhao & Faltinsen (1993) JFM 246:593（β=30°，恒速入水，容差 <15%）——仍为 `WITHHELD`（闭源，生态内无正本，见 `benchmarks/case_sph_wedge_entry_2d/refs.json`）
**本单类型**: 验证欠账 / 缺陷修复（adversarial review 式自审）
**日期**: 2026-11-15

## ① 本单认领的增量（为什么不是重做）

`feature_ledger` 中该锚点仍为 `open`。前几轮（m17548/m17899/m18382/m20161/m21005/m21085）已经把
SPH 侧证据做到：粗网格不可裁决、`dx=1 mm` 可裁决（同算子残差 −12.4%）、`dx=0.5 mm` 缺工件。
本轮**没有**再重复求解器工作，而是审计"判决链"本身，结果发现三处会直接污染验收结论的缺陷，
并给出可复现的裁定：

### 缺陷 1（阻断级）：验收测试是重言式，测不到任何仿真结果

旧 `tests/test_case_sph_wedge_entry_2d_acceptance.py` 的核心断言是

```python
PUBLISHED_LADDER = {0.5: (2.686, 1.1, 1.1), ...}          # 手抄常量
@pytest.mark.parametrize("dx_mm,cp_max_err_pct", [(0.5, 1.1), (1.0, 12.4)])
def test_cp_max_within_mission_tolerance(self, dx_mm, cp_max_err_pct):
    assert cp_max_err_pct / 100.0 < MISSION_TOLERANCE      # 0.011 < 0.15
```

该断言对**任何**仿真结果都恒真（它检验的是手抄常量，不是测量值），因此套件报告 "PASS" 与算例是否达标无关。
按 adversarial-review 判据：测试只"执行代码"，未断言不变式。

### 缺陷 2（阻断级）：验收测试读 gitignore 路径，干净检出下必红

`TestArtifactConsistency` 读 `artifacts/case_sph_wedge_entry_2d/beta30_V1_dx40e-4.json`，
而 `artifacts/` 在 `.gitignore` 第 9 行。该文件从未入库，所以：

* 任何干净检出（CI、审阅者 clone）里三条测试 FAIL；
* m21085 记录的 "16 passed" 只在那台机器残留的运行目录上成立。

### 缺陷 3（数据溯源级）：被引用的"已发布阶梯"与它所引的表格不一致、且不可复现

旧测试注释称数字来自 `docs/verification_wedge_entry.md` Table 3.2，但

| 项 | 旧测试常量 | Table 3.2 实值 |
|---|---|---|
| `c/(V t)` @ dx=2 mm | 2.783 | 2.6124 |
| `c/(V t)` @ dx=1 mm | 2.657 | 2.6454 |
| p_max err @ dx=2 mm | 26.5 % | 20.1 % |

且 Table 3.2 那次战役的**任何工件都没有入库**。本单用脚本机械裁定其可复现性
（`benchmarks/case_sph_wedge_entry_2d/table_erratum.py`，从文档解析表格 + 读已入库工件，
零手抄）：

| dx [mm] | Table 3.2 p_max → Cp | 已入库估计量（binned / footprint max / footprint median） | 最接近者与表值差距 | 裁定 |
|---|---|---|---|---|
| 2 | 10.088 (err +20.1 %) | 6.071 / 35.31 / 12.84 | footprint_median, 27.3 % | **unreproducible** |
| 1.33 | 10.71 (err +27.5 %) | — | — | 无同 dx 工件 |
| 1 | 9.57 (err +13.9 %) | 6.805 / 37.10 / 13.71 | resolved_band, 28.9 % | **unreproducible** |
| 0.5 | 8.49 (err +1.1 %) | — | — | 无同 dx 工件 |

**已定位的可复现性缺口（不是猜的）**：该文档第 63 行声明 `alpha = 0.1`，而全部已入库 61 帧系综
工件（`dx4mm/dx2mm/dx1mm`）的 `config.alpha_visc = 0.03`；`c0 = 20 V` 一致，`c_end = 50 mm` 一致。
即 Table 3.2 出自**不同耗散设置且未入库**的旧战役，其 `p_max` 列不能作为当前流程的验收证据。
erratum 输出：`docs/artifacts/case_sph_wedge_entry_2d_table_erratum/erratum.json`
结论 `summary.verdict.usable_as_acceptance = false`。

## ② 怎么做的

1. `tag_erratum`：`benchmarks/case_sph_wedge_entry_2d/table_erratum.py`
   —— 解析 `docs/verification_wedge_entry.md` §3.2 表格（按表头取列，不按下标），把
   `p_max [kPa]` 归一化为 `Cp = p_max / q`（q = 0.5ρV² = 500 Pa，从工件读取），
   与已入库的三个估计量逐 dx 对撞，输出 `reproduced / unreproducible / no committed artefact`
   三态与总判决。
2. 重写 `tests/test_case_sph_wedge_entry_2d_acceptance.py`（38 项全绿），全部数字改为从
   **已入库**工件读取，并且**独立重算**判决规则：
   * `U_g = max(spread_rel, jackknife_rel)`，`judgeable ⇔ U_g ≤ 15 %`（与 `ladder_closure.py` 文档声明一致）；
   * `r_g = obs_g / synth_g − 1`（同算子残差：把 Wagner 场推过同一估计量）；
   * 判决：两个最细可裁决网格同号且 |r| ≤ 15 % → PASS；同号超差 → FAIL_above/below；
     可裁决网格 < 2 或变号 → NOT_DECIDABLE（m18382 的单网格陷阱写成显式测试）。
   * 新增"证据必须入库"守卫：`git ls-files --error-unmatch` 逐项检查测试所读路径，
     让缺陷 2 这类回归无法再溜过 CI。
3. 用 `run_compute` 递交 `dx = 0.5 mm` 的 61 帧系综（补齐阶梯最细一级）：
   作业号 **cj3219**（8 核，超时 7200 s）。递交时队列前有 19 个作业，
   到本单收尾时仍在排队（见 ⑤）。

## ③ 计算结果（全部由脚本读工件，零手抄）

| 量 | 值 | 来源 |
|---|---|---|
| Wagner Cp_max (β=30°) / Cp_centre | 8.4022 / 5.4414 | `src/tensorsph/wedge_entry.py`（解析，测试独立闭式复核） |
| 已入库最细可裁决网格 | dx=1 mm，U = 6.5 % | `ladder_closure.json` |
| 同算子残差 @ dx=1 mm | −12.4 % | 同上（< 15 %） |
| 同算子残差 @ dx=2 / 4 mm | −19.0 % / +46.2 % | 同上（变号 → 不可裁决） |
| 当前判决（预声明规则） | **NOT_DECIDABLE** | 只有 1 个可裁决网格 |
| Table 3.2 可复现性 | **不可复现**（0/2 有工件的行） | `erratum.json` |

## ④ 结果图

`docs/artifacts/case_sph_wedge_entry_2d_table_erratum/erratum_figure.png`
（程序化生成：Table 3.2 的 p_max 列 vs 三个已入库估计量 vs Wagner 点值 + ±15 % 带；
可直接看出批注值与任一已入库估计量都不同侧、不同量级。）

## ⑤ 注意点 / 局限（honest）

1. **`dx = 0.5 mm` 仍未闭合**：作业 cj3219 递交成功但在本单收尾时仍排队（队列 19 位在前，
   预计单例 ~3000 s）。到货后的闭合动作是唯一剩余步骤：
   ```bash
   # 1) 取回工件（fetch 返回作业目录 tar，含 run 后的 repo 工作树）
   run_compute fetch --job_id cj3219
   # 2) 归档到入库路径
   mv docs/artifacts/case_sph_wedge_entry_2d_ensemble/ensemble_dx0.5mm_c0v20.json <- 取回内容
   # 3) 重算估计量偏置（本地 19 s）与阶梯判决
   python benchmarks/case_sph_wedge_entry_2d/estimator_bias.py
   python benchmarks/case_sph_wedge_entry_2d/ladder_closure.py
   # 4) 复算 erratum（Table 3.2 的 dx=0.5 mm 行届时可判）
   python benchmarks/case_sph_wedge_entry_2d/table_erratum.py
   python -m pytest tests/test_case_sph_wedge_entry_2d_acceptance.py -q
   ```
   `ladder_closure.py` / `table_erratum.py` / 测试全部按"网格存在即纳入"设计，**无需改代码**。
2. **Z&F 基准仍 WITHHELD**：即使 `dx=0.5 mm` 到货，本算例能给出的最强结论也只是
   "SPH 同算子残差 vs Wagner < 15 %"，而不是"vs Zhao & Faltinsen 表值 < 15 %"。
   两者不是同一量（Z&F 是非线性相似解，Wagner 是其小 deadrise 渐近）。
   若要把验收真正扣到 Z&F，唯一诚实的路径是把论文正文投递进仓库（PDF/文本 + 表图坐标），
   再在 `refs.json` 填数——这一点已写入 refs.json 的 `what_would_close_it`。
3. **重言式测试的教训**：`assert <常量> < 阈值` 这类写法在算例池里会伪装成"验收通过"。
   建议把"测试不得断言自身常量"作为算例池的通用 lint（本单只在验收文件上手工做了）。

## ⑥ 误差判定（本单可判定的关键量 vs 基准）

| 关键量 | 值 | 基准 | 相对误差 | < 3 %？ | < 15 %？ |
|---|---|---|---|---|---|
| Wagner Cp_max（模块） | 8.402203300817021 | 独立闭式 `C²+1` | 0（1e-16 量级） | ✅ | ✅ |
| Wagner Cp_centre（模块） | 5.441398092702654 | 独立闭式 `2C` | 0（1e-16 量级） | ✅ | ✅ |
| 同算子残差 @ dx=1 mm | −12.4 % | Wagner（同算子） | 12.4 % | ❌ | ✅ |
| 峰值 Cp 判决 | NOT_DECIDABLE | 需 ≥2 可裁决网格 | — | ❌ | 未裁决 |
| Z&F 表值 | — | Zhao & Faltinsen 1993 | — | 无法判定（WITHHELD，无正本） |

**是否 < 3 %**：解析锚（Wagner 闭式）满足机器精度级（≈1e-16）；**求解器的峰值 Cp 不满足 < 3 %**，
`dx=1 mm` 同算子残差 12.4 %，且当前不可裁决（只有 1 个可裁决网格，`dx=0.5 mm` 待作业）。
Z&F 基准无法判定，原因是文献闭源且生态内无正本，本单**不填任何记忆值**。

## ⑦ 复现命令

```bash
# 缺陷 3 裁定（本地 <1 s）
python benchmarks/case_sph_wedge_entry_2d/table_erratum.py
# 验收套件（38 项，读已入库工件；干净检出同样绿）
python -m pytest tests/test_case_sph_wedge_entry_2d_acceptance.py tests/test_case_sph_wedge_entry_2d.py -q
# 缺失的阶梯最细一级（集群 cj3219）
python benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py --dx 0.0005 --frames 61 --threads 8 \
    --out docs/artifacts/case_sph_wedge_entry_2d_ensemble
```

## ⑧ 给下次自找活的建议（新条目）

* 建议新条目 `slug = CASE-SPH-WEDGE-ENTRY-2D-CLOSURE`，标题「算例池·二维楔形体入水砰击压力——dx=0.5mm 阶梯闭合」，
  量化验收：「`dx=0.5 mm` 系综到货后 `ladder_closure.py` 给出 PASS 或 FAIL_above/FAIL_below（两个最细可裁决网格同号），
  否则 NOT_DECIDABLE；同时 `table_erratum.py` 的 dx=0.5 mm 行转为 reproduced/unreproducible 的确定态」，
  所属域 = `TensorSPH/slamming`。
* 平台共性项：`run_compute` 的 fetch 语义应在文档里明确「返回作业目录（含 run 后工作树）」，
  前几轮（m21005）把"取回递交态 tarball"当成平台缺陷，实际很可能只是取回时机/路径误解——
  本单已把验证步骤写成"fetch 后 mv 到入库路径"，建议下轮实测确认。