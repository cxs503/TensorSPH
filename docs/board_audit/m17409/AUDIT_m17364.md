# 交差审计（board-audit）— m17364 XVAL-SPH-COMPRESSION-BASED-DENSITY-REGULARIZATION

**审计单**：m17409（本单）｜**被审单**：m17364（succeeded）｜**仓**：TensorSPH｜**被审分支/提交**：
`agent/m17364` @ `fef547d`（"m17364: XVAL of SPHinXsys compression-based density regularization"）｜
**被审 worktree**：`/root/tp-agent/work/m17364`｜**审计员**：board-audit

**裁定**：**L3（系统集成）**，附 1 条阻断级缺陷（L4 门槛未过）。理由见 §4；逐项取证见 §2/§3。

---

## ① 审计范围与方法（independent re-execution，不采信报告）

| 步骤 | 动作 | 结果 |
|---|---|---|
| S1 | 读提交内容（`git show --stat fef547d`，11 文件 +3134/−1） | 新模块 + 求解器接线 + 16 测试 + 基准驱动 + 报告 + 工件，与报告声明一致 |
| S2 | 复跑单元测试 | `pytest tests/test_compression_regularization.py -q` → **16 passed**（7.2 s） |
| S3 | 复跑文档给出的 repro_cmd，逐字 | **崩溃**：`TypeError: unsupported operand type(s) for /: 'str' and 'str'`（run.py:222）→ **F1 阻断** |
| S4 | 在 scratch 副本（`docs/artifacts/repro_m17364/tree`）内只修那一行 argparse 默认值，重跑 | 4 臂全部复现；`records_repro.json` 与交付 `records_dx004_quick.json` **逐字段零差**（max\|Δ\|=0.00e+00，profile/ks/samples/steps 全同） |
| S5 | 重跑出图链（`figures.py`） | 4 张 PNG **sha256 全同**（`66902145… / f3aa1ccc… / 3893b1b3… / 563cb0bc…`）→ 整链确定性可复现 |
| S6 | 独立取上游源码核对移植（本单新做，被审单未归档原文） | API `1612330e` 取 `density_regularization.{h,hpp,cpp}`，逐段比对公式：**忠实**（§3） |
| S7 | 查基准裁判表 | `tensor_benchmark_cases` 全表 52 行，**anchor_slug 命中 0 行** → 平台容差裁判对本锚点不存在（F2） |

复现脚本（本单交付，可一键跑）：`bash docs/board_audit/m17409/repro_m17364.sh`（含 F1 一行修复 + 对比 sha256）。

---

## ② 三件套取证（run_artifacts / ref vs got / repro_cmd）

**run_artifacts（客观存在、非空）**：交付 4 图 52 822 / 71 607 / 86 577 / 167 969 B；
`records_dx004_quick.json`（1833 行）含 4 臂 ×（gate / profile / series / nondimensional）。
审计侧重跑产物：`docs/artifacts/repro_m17364/records_repro.json`、`…/figures/*.png`。

**ref vs got（基准判据）**：平台侧无该锚点基准行（F2），故只能对解析静水参考
`p(y)=ρ0 g(H−y)` 核数。审计重跑值 = 交付值（零差）：

| 臂 | 内部带最大 p 误差（<-15% 判据 / <3% 线） | ρ 误差 | v_rms | K_ref(末) |
|---|---|---|---|---|
| summation（本仓基线，对照臂） | **1.401829 %** ✅/<3% ✅ | 0.0130 % | 0.0500 m/s | 1.000000 |
| compression_none（OSS 版式 A） | **1.426631 %** ✅/<3% ✅ | 0.0132 % | 0.0502 m/s | 0.994799 |
| compression_refmean（OSS master） | **9.099796 %** ✅/<3% ❌ | 0.0761 % | 0.0875 m/s | 0.993827 |
| compression_refmean_internal | **9.156891 %** ✅/<3% ❌ | 0.0782 % | 0.0875 m/s | 0.993962 |

报告里的 1.40/1.43/9.10/9.16 与上表**逐位相同**，无拔高：`compression_refmean` 的 9.10 % 被如实写成
“>3 % 不可接受线”、且四臂 `gate.pass` 全为 `false`（本仓 1 % 门限连基线都不过）——**honest，未粉饰**。

**repro_cmd 可执行性**：交付命令不可执行（F1）；修一行后 16 s 出全量结果且字节级复现 → 数值本身真实，
只是“出厂包装”坏了。

---

## ③ 外部锚点核验（本单独立取证）

上游 `Xiangyu-Hu/SPHinXsys` @ tree `1612330e…`，文件已归档本单
`artifacts/audit_m17364/upstream/`（blob+sha256 见 `upstream/SHA256SUMS.txt`）。逐条比对：

| 上游代码 | 移植实现 | 判决 |
|---|---|---|
| `compression_sum_[i] = W0*Vol_ref[i]; … += (*inv_ref)*W_ij*Vol_ref[j]`（`Inner`），`Contact` 追加同一项 | `compression_summation`：`S = w_self·Vref_i + K⁻¹ Σ W_ij Vref_j`，**自项不乘 K⁻¹** | 一致（含自项非对称这一细节） |
| `reduce`: `S>1 ? Sample(S,1) : Sample(0,0)`；`FinishDynamics`: `inv = 1/(ΣS/Σ1)` | `average_inverse_reference_compression`：`mask=S>1`；`counted/ΣS` | 一致（空集 0/0 → 本仓取恒等 1，已在 docstring 声明为偏差） |
| `Regularization<Internal>` 原值；`<FreeSurface>` `SMAX(S,1)`；`<FreeStream>` `indicator!=0 ? 1 : S` | `regularize()` 三规则同名同义 | 一致 |
| `rho = K·rho0`（`eos_.getReferenceDensity`） | `regularized_density` / `rho=(compression*rho0)` | 一致 |
| 单变量 `InverseReferenceCompression` 默认 1、跨步滞后 | `CompressionRegularization.inverse_reference_compression` + `rhs(state=…)` | 一致（顺序见 `test_two_stage_order_*`） |

→ **external_anchor = REAL**（本单从上游原文独立复核，不是读移植者自己的字面转写）。

---

## ④ 判级（L0–L5）

- **L1 通过**：16 测试，含「向量化 vs 循环体字面转写」「Σ 与求和密度等价」「两阶段顺序」等不变量断言（非只跑不判）。
- **L2 通过**：`src/tensorsph/compression_regularization.py`（251 行，真实现，无 placeholder）+ 入口驱动 + 真实运行工件（含本单重跑）。
- **L3 通过**：接入主求解器——`src/tensorsph/droplet_impact.py:227-231`（`Config.density_regularization` /
  `compression_regularization_rule` / `compression_normalization`）与 `:882-930`（`rhs()` 覆盖分支、
  互斥校验、跨步 `state`），默认 `"none"` 时分支不进入（老算例逐位不变是结构性成立，非靠调用方自觉）。
- **L4 未通过（阻断）**：完整算例 + 误差分析 + 解析参考都在，但**交付的 repro_cmd 不可执行**（F1）；
  L4 判据里的“可复现”按出厂状态不成立。修一行即复现（本单已验证字节级），故不是科学问题而是交付缺陷。
  另：多分辨率收敛（报告 ⑤-#4 自认未完成，集群 `cj1504` 审计时仍 `queued`，age 26.5 min）→ 收敛性证据缺。
- **L5 未通过**：无外部实验/文献绝对量级对标，且缺收敛阶。

**声明级别 = 3**；L4 待一行修复 +（建议）补 2 点分辨率后由下一轮审计认领。

---

## ⑤ 发现清单（severity / 证据 / 复现）

| ID | 级别 | 发现 | 证据与复现 |
|---|---|---|---|
| **F1** | **阻断（L4）** | 交付 `benchmarks/xval_sphinxsys_compression/run.py:222` 的 argparse 默认值是 `str("xval_output" / "xval_sphinxsys_compression" / "records.json")`，`str/str` 在 `str()` 之前求值 → **任何调用**都在 parse 阶段 `TypeError`。故交付工件不可能由交付代码产生，`repro_cmd` 不可执行 | `python3 benchmarks/xval_sphinxsys_compression/run.py --quick`（`cwd=/root/tp-agent/work/m17364`）→ 见 §①-S3；修复补丁 `docs/board_audit/m17409/fix_repro_cmd.patch`，修后 `repro_m17364.sh` 通过 |
| **F2** | 高（平台侧） | `tensor_benchmark_cases` 对 `anchor_slug='XVAL-SPH-COMPRESSION-BASED-DENSITY-REGULARIZATION'` **0 行**（全表 52 行）→ “基准判据是唯一裁判”在本锚点上无裁判可依，容差只能自定（<15 %//<3 %）；也意味着该能力未进平台基准账 | `SELECT count(*) …` 见 §①-S7；建议条目见 `docs/board_audit/m17409/proposed_benchmark_rows.sql`（**未执行**，避免审计员自设裁判） |
| **F3** | 中（文档） | 报告 ③ 参数表写“32 壁面粒子”，实际运行工件为 **105**（`records_repro.json` 四臂均为 `wall_particles=105`），几何/规模描述与工件不符 | 报告第 62 行 vs `records_dx004_quick.json.records[*].wall_particles` |
| **F4** | 低（文档） | 报告写“`compression_regularization.py`（新，236 行）”，实际 **251** 行；公式级等价写“1.4e-16”，本单独立复算为 4.44e-16 / 3.36e-16 / 6.51e-16（Kref 三值），量级对但具体数字不可复现 | `wc -l`；`docs/artifacts/repro_m17364/` 复算输出 |
| **F5** | 低（证据完整性） | 被审单未归档上游 C++ 原文（只在报告里引 sha），`external_anchor` 原为“不可离线复核”；本单已补齐归档 `artifacts/audit_m17364/upstream/` | 目录 + `SHA256SUMS.txt` |
| **F6** | 信息 | 集群 `cj1504`（分辨率扫描）审计时仍 `queued`（age 26.5 min），报告“未完成”声明**属实**，不是缩水借口 | `run_cluster ps job_id=cj1504` |

**未做过/不能判**（不得当作 PASS）：多分辨率收敛阶（未取得）；上游不同 `h/dx`、Wendland 核下的对标
（本仓用 cubic spline，报告已声明为受控对照前提）；任何船舶工况下的适用性（本单是数值方法交叉对标，无船型量）。

---

## ⑥ 与平台机器的交互预告

`maturity_gate` 会按 `JUDGE_SCHEME` 复判：本单 evidence 中 `impl_files/entry` 指 `/root/tp-agent/work/m17364/…`
（绝对路径，因为 `agent/m17364` **未并入** `/root/tensor-ecosystem/TensorSPH`，ECO 里无该文件），
`integration_callers` 的 grep 在 ECO 内必然落空（未合并），故 gate 预计给 **L2 confirmed**；
本审计声明的 L3 是“实质集成成立、但合并前机器不可见”的诚实结果，差异原因即此，非虚标。