# 验证账本：TensorSPH 表面张力 / Young–Laplace 证据分级（机器生成）

**本文件由 `benchmarks/verification_ledger/ledger.py` 从 `claims.json` + 已提交工件 JSON 生成，
所有数字程序化注入，无手抄**。`tests/test_verification_ledger.py` 会在文档与工件不一致时变红。

## ⓪ 为什么需要这本账

仓库里有五代表面张力验证文档，各自都是诚实的，但同一个物理量（Young–Laplace 压差）
报出的 headline 误差依次是 −93.6% → −75.6% → −87.6% → −34.0%。差异**不是模型在变差又变好**，
而是**测量口径**与**报的量**在变。读者无法判断「今天哪个数是当前的」，平台的证据分级也无法落账。
本账把每一行锁死到「量 + 配置 + 工件 + JSON 路径 + 判据」，并把被取代的行显式标 `SUPERSEDED`。

## ① 结论（先给判定）

1. **表面张力「力」这条线是 REAL**：切割力恒等式、delta_s 加权曲率、密均两相力积分口径
   三项都落在 15% 门内（其中 delta_s 加权曲率 +0.86% 已在 3% 内）。
2. **端到端 Young–Laplace 压差在单相自由面构型下仍是 WITHHELD**：
   阶梯最好读数 237.6 Pa vs 解析 360.0 Pa = **-34.0%**，最差 -38.1%；
   全阶梯散布只有 4.1 个百分点，
   即**不是**声速/分辨率噪声地板，调参救不回来（这与 m2971 的 C4 证伪一致）。
3. **历史 headline 误差里有一大块是口径**：在同一个对照算例上只换测量口径，
   「带相减」-97.9% → 「绝对压强 + 时间平均」-37.8%，
   |误差| 直接掉 **60.1 个百分点**。
   账本里那条 −78.6%（`laplace_band_differencing_mouth`）就是旧口径的产物，已标 SUPERSEDED。
4. **两相（等密度）构型下 Laplace 力积分口径是 REAL**：
   +4.37%（dx = 5 mm）。
   该构型没有密度跳变与真空，**不经过**自由面压强闭合，所以它验证的是算子，不是闭合。

## ② 账本总表（证据分级）

| claim id | 证据等级 | 判定 | 实测值 | 相对误差 | 参考值 | 工件 |
|---|---|---|---|---|---|---|
| `laplace_single_phase_best` | **WITHHELD** | FAIL | 237.64 Pa | -33.99% | 360.00 Pa | `benchmarks/laplace_pressure_floor/report.json` |
| `laplace_single_phase_worst` | **WITHHELD** | FAIL | 222.85 Pa | -38.10% | 360.00 Pa | `benchmarks/laplace_pressure_floor/report.json` |
| `laplace_band_differencing_mouth` | **SUPERSEDED** | FAIL | 77.19 Pa | -78.56% | 360.00 Pa | `benchmarks/csf_consistent/validation_report.json` |
| `laplace_mouth_correction` | **REAL** | MEASURED | +60.1 pt | — | — | `benchmarks/laplace_pressure_floor/report.json` |
| `cut_force_identity` | **REAL** | PASS | -12.865 N/m | +10.66% | -14.400 N/m | `benchmarks/csf_curvature_fit/validation_report.json` |
| `curvature_delta_s_weighted` | **REAL** | PASS | 50.43 1/m | +0.86% | 50.00 1/m | `benchmarks/csf_curvature_fit/validation_report.json` |
| `curvature_band_unweighted` | **WITHHELD** | FAIL | 149.96 1/m | +199.92% | 50.00 1/m | `benchmarks/csf_curvature_fit/validation_report.json` |
| `two_phase_force_integral_laplace` | **REAL** | PASS | 0.69 Pa | +4.37% | 0.72 Pa | `benchmarks/surface_tension_laplace/validation_report.json` |
| `surface_delta_integral` | **WITHHELD** | FAIL | 0.739 | -26.09% | 1 | `benchmarks/csf_curvature_fit/validation_report.json` |
| `spurious_current_adami` | **REAL** | PASS | 0.0331 | — | — | `benchmarks/csf_consistent/validation_report.json` |

统计：REAL 5 行，WITHHELD 4 行，SUPERSEDED 1 行。

「实测值/参考值」的单位见 ③ 每行注明；相对误差一律是 (实测 − 参考)/参考。

## ③ 逐组明细（量 / 配置 / 数值 / 为什么这样定级）

### Young-Laplace pressure jump (single-phase free surface)

**`laplace_single_phase_best`** — Delta p = p_core, absolute pressure, time-averaged tail

- 配置：single-phase disc, vacuum outside, R=20 mm, sigma=7.2 N/m, p>=0 Tait
- 工件：`benchmarks/laplace_pressure_floor/report.json`
- 实测：237.64 Pa；参考：360.00 Pa；相对误差 **-33.99%**
- 判据：|误差| ≤ 15% → FAIL（3% 内：否）
- 定级理由：best reading over the c0 x dx ladder; still outside the 15 % gate, so the end-to-end jump is not evidence of a correct surface-tension model

**`laplace_single_phase_worst`** — same quantity, worst reading of the ladder (shows the spread)

- 配置：single-phase disc, vacuum outside, R=20 mm, sigma=7.2 N/m
- 工件：`benchmarks/laplace_pressure_floor/report.json`
- 实测：222.85 Pa；参考：360.00 Pa；相对误差 **-38.10%**
- 判据：|误差| ≤ 15% → FAIL（3% 内：否）
- 定级理由：spread over c0 = 5..20 and dx = 1.5..2 mm is 4.1 pts, i.e. the residual is not an EOS noise floor that tuning can remove

### Measurement-mouth artefacts (superseded)

**`laplace_band_differencing_mouth`** — Delta p = p_center - p_interface_band (legacy band differencing)

- 配置：single-phase disc, R=20 mm, sigma=7.2 N/m, dx=2 mm, c0=20, 600 relax steps
- 工件：`benchmarks/csf_consistent/validation_report.json`
- 实测：77.19 Pa；参考：360.00 Pa；相对误差 **-78.56%**
- 判据：|误差| ≤ 15% → FAIL（3% 内：否）
- 定级理由：the interface band sits inside the free-surface density deficit (-6..-34 %), so this mouth reports a pressure difference that the fluid does not carry; see laplace_mouth_correction
- 被取代于：`laplace_band_differencing_mouth` → `laplace_single_phase_best`（当前值 -34.0%）

**`laplace_mouth_correction`** — percentage points of |error| removed by switching from band differencing to absolute + time-averaged pressure

- 配置：uncalibrated control run, dx=2 mm, c0=20 (same disc, same scheme as the row above)
- 工件：`benchmarks/laplace_pressure_floor/report.json`
- 数值：|-97.9 pt| − |-37.8 pt| = **+60.1 pt**（误差幅值的下降量，非误差）
- 定级理由：same control run (dx=2 mm, c0=20, 1283 steps): the only change is the measurement mouth, so the drop is attributable to the mouth alone

### CSF force operator

**`cut_force_identity`** — F_cut / (2 sigma) vs 1 (cutting-plane force identity)

- 配置：single-phase disc, R=20 mm, sigma=7.2 N/m, dx=1.5 mm, consistent CSF + fit_quadratic curvature
- 工件：`benchmarks/csf_curvature_fit/validation_report.json`
- 实测：-12.865 N/m；参考：-14.400 N/m；相对误差 **+10.66%**
- 判据：|误差| ≤ 15% → PASS（3% 内：否）
- 定级理由：the volume-force integral is the part that the CSF refactor fixed; it is inside the gate at every resolution in the artifact

**`curvature_delta_s_weighted`** — delta_s-weighted mean curvature vs 1/R

- 配置：single-phase disc, R=20 mm, dx=1.5 mm, fit_quadratic curvature
- 工件：`benchmarks/csf_curvature_fit/validation_report.json`
- 实测：50.43 1/m；参考：50.00 1/m；相对误差 **+0.86%**
- 判据：|误差| ≤ 3% → PASS（3% 内：是）
- 定级理由：the physically weighted curvature (weight = interface measure) is the quantity that enters the CSF force; it is inside the 3 % band used by the plateau/convergence claims

**`curvature_band_unweighted`** — unweighted band mean curvature vs 1/R

- 配置：single-phase disc, R=20 mm, dx=1.5 mm, fit_quadratic curvature
- 工件：`benchmarks/csf_curvature_fit/validation_report.json`
- 实测：149.96 1/m；参考：50.00 1/m；相对误差 **+199.92%**
- 判据：|误差| ≤ 15% → FAIL（3% 内：否）
- 定级理由：per-particle curvature scatter on the band is large and biased; the claim only holds after delta_s weighting, which is exactly why the force integral (REAL) is the right instrument and the raw band mean is not

**`two_phase_force_integral_laplace`** — radial line integral of the CSF volume force vs sigma/R (density-matched two-phase)

- 配置：two-phase, equal density, sharp colour step + Shepard smoothing, R=100 mm, sigma=0.072 N/m, dx=5 mm
- 工件：`benchmarks/surface_tension_laplace/validation_report.json`
- 实测：0.69 Pa；参考：0.72 Pa；相对误差 **+4.37%**
- 判据：|误差| ≤ 15% → PASS（3% 内：否）
- 定级理由：with no density jump and no vacuum the pressure closure is not exercised, so this row validates the force operator, not the free-surface pressure closure

**`surface_delta_integral`** — integral |grad c| dn (CSF colour-field normalisation)

- 配置：single-phase disc, R=20 mm, dx=1.5 mm, consistent CSF
- 工件：`benchmarks/csf_curvature_fit/validation_report.json`
- 实测：0.739；参考：1；相对误差 **-26.09%**
- 判据：|误差| ≤ 15% → FAIL（3% 内：否）
- 定级理由：the colour field carries only ~0.74 of the unit surface delta; it is compensated inside the force normalisation but it is the reason absolute force magnitudes must not be read off N_delta directly

### Stability / spurious currents

**`spurious_current_adami`** — max |v| / sqrt(sigma / (rho R)) against the Adami 2012 bound 0.1

- 配置：single-phase disc, R=20 mm, dx=2 mm, c0=20, relaxed 600 steps
- 工件：`benchmarks/csf_consistent/validation_report.json`
- 实测：0.0331；判据上限 0.1；判定 PASS
- 定级理由：parasitic current is the second necessary condition for a usable surface-tension model; it passes with margin at every resolution in the artifact

## ④ 图：每一行 vs 它自己的参考值

![verification ledger](benchmarks/verification_ledger/ledger.png)

蓝色 = REAL，红色 = WITHHELD；虚线为 15% 验收门，点线为 3% 门。横轴每根柱是一个 claim，
柱高是它对**自己**参考值的相对误差——跨组不可横向比较绝对值，只能读「是否在自己的门内」。

## ⑤ 误差判定：是否 < 3%？

| claim id | 相对误差 | < 3%? |
|---|---|---|
| `laplace_single_phase_best` | -33.99% | 否 |
| `laplace_single_phase_worst` | -38.10% | 否 |
| `laplace_band_differencing_mouth` | -78.56% | 否 |
| `cut_force_identity` | +10.66% | 否 |
| `curvature_delta_s_weighted` | +0.86% | 是 |
| `curvature_band_unweighted` | +199.92% | 否 |
| `two_phase_force_integral_laplace` | +4.37% | 否 |
| `surface_delta_integral` | -26.09% | 否 |

**直答**：只有 delta_s 加权曲率一项在 3% 内；判据意义上的关键量——**端到端 Young–Laplace 压差**——
最优也只有 -34.0%，**不满足 3%，也不满足 15%**，故该能力在单相自由面构型下
的证据等级维持 **WITHHELD**，不得被引用为「已验证」。

## ⑥ 注意点 / 局限

1. **口径必须写进 claim**。本账只有把「带相减 / 绝对压强 / 力积分 / 时间平均」显式记下才有意义；
   同一份工件（`benchmarks/csf_consistent/validation_report.json`）两种口径可以差 40 个百分点以上。
2. **两相构型不能替代单相**。等密度两相没有密度比、没有真空、没有自由面压强闭合，
   所以它的 4.4% 不能用来证明液滴撞池/自由面算例的正确性。
3. **拉格朗日弛豫步数不同会影响读数**。`laplace_mouth_correction` 的两行是同 dx/c0/算子的对照，
   但步数 600 vs 1283，故该行只作「口径量级」证据，不作精度结论。
4. **本账不重跑任何仿真**，只读已提交工件；因此它继承工件自身的时效与配置，
   但把「工件 ↔ 声称」的映射变成机器可判。
5. **过期的验证文档**（`docs/verification_surface_tension_*.md`）已在头部加 `SUPERSEDED` 标注，
   指向本账；它们保留原数字作为诚实留痕，不再作为当前精度引用。

## ⑦ 复现

```bash
PYTHONPATH=src python benchmarks/verification_ledger/ledger.py
python -m pytest tests/test_verification_ledger.py -q
```
