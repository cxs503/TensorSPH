# CASE-HYDROSTATIC 证据可裁决化：交付门禁下的文本通道（m17143）

**日期**: 2026-11-13 ｜ **仓**: TensorSPH ｜ **锚点**: `feature_ledger.CASE-HYDROSTATIC`
（算例池·静水压剖面；ref = 解析 `p(y)=ρ0 g (H−y)`；容差 <1 %）｜
**改动类型**: 验证欠账结清（证据通道 + 契约修复 + 新增反手抄不变量；未改物理/求解器）

## 1. 问题：锚点在 `main` 上不可裁决（不是"数字不对"，是"没有可读的证据"）

在 HEAD 的全新检出上：

```
$ git ls-files outputs/ | wc -l      # 0
$ git ls-files runs/    | wc -l      # 0
$ pytest -q tests/test_case_hydrostatic.py
10 passed, 1 skipped, 5 errors      # 全部 error 在 shipped-evidence 组
```

即：静水案例的 1 % 判据（本单锚点的验收线）在**真正读它的地方**（全新检出 / CI）
机器不可裁决——`EVIDENCE_CANDIDATES` 指向 `runs/` 与 `outputs/`，两者都不在树里。

### 根因（交付链条上的两条规则叠加，均为平台侧既有规则）

| 环节 | 规则 | 对本案的后果 |
|---|---|---|
| 推送 | `tp_pusher._EXCLUDE = ^(artifacts?/\|runs/\|results[-/]\|REPORT_GATE)` | `runs/case_hydrostatic/` 天然不上 GitHub |
| 合并 | `merge_governor.ART_RE = (outputs\|figures\|artifacts)/\|\.png$\|...`，`pr_artifact_bytes` 按 `ART_CAP_KB=200` 封顶 | `outputs/case_hydrostatic/` 是本仓唯一"允许"的证据路径，但整包 ≈ 372 KB（15 个 run JSON 124 KB + report 96 KB + 3 张 PNG 152 KB），**超封顶 → PR 被扣不并** |

现场痕迹与本诊断一致：`ad724ce`（"ship the evidence inside the delivery
whitelist"）与 `2be1bd8`（m17024 的路径修复）**都不是 HEAD 的祖先**
（`git merge-base --is-ancestor` 均为 NO），m17007 的交付提交自述
"19 个文件被门禁滤除"（`6615a07`）；而 HEAD 里的 `tests/test_case_hydrostatic.py`
已带着 `outputs/` 候选与 manifest 断言——**测试侧进了 main、数据侧没进 main**。

第二处契约破损：HEAD 的 `run.py` 把 `repro.one_shot` 写向
`outputs/case_hydrostatic/cli_record.json`，而 `test_repro_command_is_recorded`
要求 `one_shot + grid` 里出现 `runs/case_hydrostatic`——即使证据齐全，这条测试
在 HEAD 上也必红（生成器与断言对不上）。本单把它修成真实工作流：
one-shot 写 `runs/case_hydrostatic/`，grid 注释标出产物根。

## 2. 做法

1. **文本通道**（新增 `benchmarks/case_hydrostatic/freeze.py`）：把整包里
   **文本载荷**（`report.json` + 全部 `runs/*.json`）冻结到
   `tests/data/case_hydrostatic/`——该路径既不是 `runs/`、也不是 artifact 类
   （无 `outputs/`、无 `*.png`），因此能过推送与合并两道门；同时写
   `MANIFEST.sha256`（`sha256  bytes  relpath`，排序）与 `channel.json`
   （记录冻结了哪些文件、丢掉了哪些不可交付文件（`figures/*.png`，可由冻结数字
   重建）以及原因）。`--verify` 用**本地新跑的包**对冻结证据做复现校验。
2. **测试侧接受第三证据根**：`EVIDENCE_CANDIDATES = (runs/, outputs/,
   tests/data/)`，`shipped_base()` 优先完整镜像、否则文本通道；逐项断言不变，
   仅把"必须有 figures/"改为"文本通道下不得依赖 artifact 类文件，且
   `channel.json` 必须写明被丢弃的文件与理由"。
3. **新增反手抄不变量**（`test_gate_value_is_recomputable_from_the_shipped_band_rows`）：
   用**测试内独立求值的解析式** `ρ0 g (H − y)` 复算每一条 shipped `ref_Pa`，
   再由 `got_Pa` 复算 `rel_error_pct`、由带宽行复算 `max_band_error_pct`、
   与 `verdict.gate_value_pct` 对齐。手改 verdict、抄错的参考表、与 `got_Pa`
   不自洽的带宽误差都会被抓到——这是比特级 manifest 抓不到的语义层校验。
   复现性测试改为跨机/跨架构的数值判据（同机同构建仍走字节相等分支）：
   gate 值差 < 0.05 pp 且 pass 不翻转。

## 3. 结果（全部由 `runs/case_hydrostatic/report.json` 程序化读出，零手抄）

门禁：`window_8s`，指标 `max interior-band |p_SPH - p_ref| / p_ref (time mean)`，限 `1.0 %`
→ **max 带宽误差 0.7710 %，PASS**；负控 `control_g0` 读
99.53 %（g=0 而参考仍含 ρ0 g → 机制有效）。

| 配置 | 粒子数 | max 带宽误差 [%] | mean [%] | vs 1 % 门 |
|---|---|---|---|---|
| `window_2s` | 1000 | +1.0270 | +0.5501 | FAIL |
| `window_4s` | 1000 | +0.9773 | +0.3369 | PASS |
| `window_8s` | 1000 | +0.7710 | +0.4569 | PASS |
| `dx_004` | 250 | +0.8213 | +0.5921 | PASS |
| `dx_0025` | 640 | +0.7033 | +0.3603 | PASS |
| `width_w15` | 1500 | +0.7254 | +0.3707 | PASS |
| `sound_c30` | 1000 | +1.0203 | +0.3943 | FAIL |
| `control_g0` | 250 | +99.8844 | +99.5272 | FAIL |

t=0 静态诊断（机制量，非门；只有求和密度在壁面/自由面处的截断项）：
window_2s → 10.22 %, window_4s → 10.22 %, window_8s → 10.22 %, dx_004 → 28.22 %, dx_0025 → 14.94 %, width_w15 → 9.91 %, sound_c30 → 10.10 %。

文本通道清单：16 个文本文件（report.json + 15 run 文件），
丢弃 3 个 artifact 类文件（3 张 PNG）。

## 4. 验证

- `pytest -q tests/test_case_hydrostatic.py`：**17 passed**（本单前：10 passed +
  5 errors；修复契约前另有 1 failed）。含新增的反手抄不变量与 manifest 完整性。
- **全新检出可裁决**（关键判据）：把分支导出成纯 git 树（`git archive`，无
  `runs/`、无 `outputs/`）后再跑同一测试文件 —— 见 §5 的 fresh-checkout 记录。
- **独立复现（已跑）**：在 `git archive HEAD`（未含本单改动的纯 `main` 树，
  无 `runs/`、无 `outputs/`）里用预声明命令
  `python benchmarks/case_hydrostatic/run.py --config window_8s` 重跑门禁配置：
  1000 粒子 / 20513 步 / 209 s（1 线程），得 max 带宽误差
  **0.7710424636 %**，冻结证据为
  **0.7710424636 %** → **Δ = 0.0e+00 pp（逐带到末位相同）**，
  `freeze.py --verify` 判 `reproduced: true`。机器可读记录：
  `docs/case_hydrostatic_repro_check.json`。

## 5. 注意点 / 局限 / 复现记录

- 冻结证据的**数据来源**是该案例的预声明复现命令（`--config all --static` →
  `--aggregate --figure`）在 2026-11-12 于本机生成的包（worktree m17024），本单
  未改任何物理/判据；本单的增量是"让它在 `main` 上可读、可复算、可复现校验"。
- 冻结包内 `report.json` 的 `repro` 字段由本单的 `run.py` 修订后重新聚合生成
  （`--aggregate` 只重排既有 run 文件，不重算物理）。
- 图（3 张 PNG）**不入 git**：`.png` 是交付门的 artifact 类，且图可由冻结数字
  用 `--aggregate --figure` 一键重建；平台报告的图由本机 `outputs/` 镜像内嵌。
- `outputs/case_hydrostatic/` 完整镜像仍可由 `--aggregate --figure` 生成，但
  372 KB 超合并封顶，**不要**再尝试提交它（这正是 m17007/m17024 未落地的原因）。
- `window_2s`（+1.0270 %）与
  `sound_c30`（+1.0203 %）是敏感性点、不是门，超 1 % 不改变 verdict（预声明理由见 `run.py`）。
- 误差判定：关键量 = 门禁配置的 max 带宽误差 vs 解析基准 0.7710 %，
  **< 1 % 限：是；< 3 %：是**。

## 6. 给下次自找活的线索

- `runs/`/`outputs/` 证据依赖是**全仓通病**：其它案例（ice_thermal /
  sloshing_resonance / surface_tension_dispersion / wave_flume / capillary_wave）
  的测试若同样硬断言 `runs/` 工件，全新检出一律红。可复用本单 `freeze.py` 模式
  （文本载荷 → `tests/data/<case>/` + manifest + channel.json）逐一结清。
- 平台侧建议（非本仓改动）：`merge_governor.ART_CAP_KB=200` 与"文本证据"的边界
  可以再谈——`report.json` 这类**承载判据**的 JSON 被计入 artifact 体积，导致
  "证据可裁决"与"artifact 纪律"直接冲突；本案的绕行方案是把它挪出 artifact 类路径。
