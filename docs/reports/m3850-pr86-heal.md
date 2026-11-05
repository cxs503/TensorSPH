# PR #86 (agent/m2926) 治愈记录：三方合并语义并集 + 证据

任务 m3850。仓库 TensorSPH，PR #86 分支 `agent/m2926` 与 `main` 变脏（mergeable=false）。
本文件记录冲突解剖、逐文件解法、内容保全审计与测试证据；所有数字由
`benchmarks/merge_audit/run.py` 与 `benchmarks/delta_s_normalization/validation_report.json`
程序化注入（`benchmarks/merge_audit/make_doc.py` 生成本文件）。

## 1. 案例介绍

**物理对象**（被合并内容所属的 SPH 验证案例，2-D 静态液滴，`benchmarks/delta_s_normalization`）：
半径 R = 20 mm、σ = 7.2 N/m、
ρ0 = 1000 kg/m³、c0 = 20 m/s、γ = 7、
h/dx = 1.3、无重力；六边形格点分辨率 dx ∈
{3.0, 2.0, 1.5} mm；解析参考
κ = 1/R = 50.0 1/m、F_cut = 2σ = 14.4 N/m、
Δp = σ/R = 360 Pa。

**合并对象**：`agent/m2926` 提交 34cf622（表面 δ 函数的精确梯度格式 +
CSF 力密度开关）与 `main` ba07a10（40 个提交：曲率拟合默认、`curvature_color_smoothing`、
新 benchmark 体系、ruff lint 门）在共同祖先 5800532 之后各自演进。

## 2. 怎么做的

1. `git worktree` 内 `git fetch origin main agent/m2926`，以 `origin/agent/m2926` 为 ours、
   `origin/main` 为 theirs 做 `git merge --no-ff --no-commit`（三方合并，非 overlay）。
2. 冲突 3 处、涉 2 个文件；逐文件按「语义并集」解冲：
   - `src/tensorsph/csf_consistent.py`
     - 签名：ours 的 `delta_s_scheme` 与 theirs 的 `curvature_color_smoothing` 都是追加的关键字参数 → 两个都保留（trunk 参数在前）。
     - docstring：两侧新增的段落按 trunk 先、ours 后拼接，`"""` 终止符与 ours 的参数校验一并保留。
     - 该 hunk 里 ours 保留了 `n = pos.shape[0]` 死变量，theirs 已在 lint 清理中删除同类死变量 →
       采用 theirs 的删除（lint 语义），并在下方「内容保全」中核实：这是唯一一处 ours 行未进合并树的
       「非新增行」（新增行 0 丢失，见 §3 表）。
   - `src/tensorsph/droplet_impact.py`
     - `csf_curvature` 采用 theirs 的注释与默认值 `fit_quadratic`（main 的默认切换是主干决定）；
     - ours 新增的 `csf_delta_s`、`csf_force_density` 两个字段连同注释整段保留（默认 `two_sided`/`local` = 历史行为，向后兼容）。
   - `src/tensorsph/surface_tension.py` 由 Git 自动合并成功，人工复核确认：theirs 的 `contact_angle`
     传参与 ours 的 `delta_s_scheme`/`force_density` 传参同时存在。
3. 内容保全审计（机械判据，防 overlay 静默丢内容）：对两侧共同修改的每个文件，取
   「base→ours」「base→theirs」的全部新增行，逐行在合并树中查找，统计缺失数。
4. lint 门：m2926 的 3 个新文件有 11 处 ruff 违规（7×E501 长行、1×F841 死变量、3×W292 缺行尾换行），
   按仓库 CI（`.github/workflows/tests.yml` 第 `python -m ruff check .` 步）修好；E501 仅做字符串折行，
   用 `python benchmarks/delta_s_normalization/make_doc.py` 重新生成
   `docs/verification_delta_s_normalization.md` 后 `git diff` 为空 → 文本零漂移。
5. 测试：合并树与一份干净 `origin/main` worktree 各跑一次全量 pytest，同一机器、同一命令
   （`OMP_NUM_THREADS=1 pytest -q -n 8`）以消除配置差异。

## 3. 计算结果

### 3.1 内容保全（关键判据：缺失新增行总数 = 0）

| 文件 | m2926 新增行 | 合并后缺失 | main 新增行 | 合并后缺失 |
|---|---|---|---|---|
| `src/tensorsph/csf_consistent.py` | 90 | 0 | 35 | 0 |
| `src/tensorsph/droplet_impact.py` | 19 | 0 | 54 | 0 |
| `src/tensorsph/surface_tension.py` | 3 | 0 | 6 | 0 |

两侧文件集合比对：ours 独有 5 个文件、theirs 独有 105 个文件，合并树中两者**全部存在**（缺失 0）。

### 3.2 测试证据（不得低于 main 基线）

| 树 | passed | failed | xfailed | 墙钟 | 失败用例 |
|---|---|---|---|---|---|
| 合并树 (m2926+main) | 369 | 2 | 8 | 154.00s (0:02:34) | `test_oscillation_frequency_gate_is_not_met`, `test_recorded_mode2_frequency_deficiency_is_reproduced` |
| 基线 origin/main | 363 | 2 | 8 | 168.35s (0:02:48) | `test_oscillation_frequency_gate_is_not_met`, `test_recorded_mode2_frequency_deficiency_is_reproduced` |

合并树比基线多 6
个通过用例（即 m2926 的 `tests/test_delta_s_normalization.py` 6 个新用例），失败集合与基线**逐条相同**：
两条失败都是 `strict xfail` 的 XPASS（mode-2 振荡频率拟合在本机 x86 落在界内，而该用例按跨架构脆弱性被钉成严格 xfail），
在 `origin/main` 上以完全相同方式失败 → 环境既有状态，与本合并无关。

lint：`All checks passed!`。δ_s 定向回归：`pytest tests/test_delta_s_normalization.py tests/test_csf_consistent.py` → 26 passed。

### 3.3 被合并内容的关键量（4 组合 × 3 分辨率，`benchmarks/delta_s_normalization`）

| δ_s 方案 | 力密度 | dx [mm] | N | ∫δ_s dn | 偏差 | Σδ_s n_x dV/2R | 偏差 | F_cut/2σ | 偏差 | κ 偏差 | ρ_i/ρ0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kernel_sum | local | 3.0 | 161 | 1.0677 | +6.77% | 0.9600 | -4.00% | 1.6507 | +65.07% | +23.56% | 0.727 |
| kernel_sum | local | 2.0 | 362 | 1.0487 | +4.87% | 0.9722 | -2.78% | 1.4455 | +44.55% | +9.67% | 0.804 |
| kernel_sum | local | 1.5 | 642 | 1.0415 | +4.15% | 0.9822 | -1.78% | 1.4356 | +43.56% | +4.23% | 0.852 |
| kernel_sum | reference | 3.0 | 161 | 1.0677 | +6.77% | 0.9600 | -4.00% | 1.2018 | +20.18% | +23.56% | 0.727 |
| kernel_sum | reference | 2.0 | 362 | 1.0487 | +4.87% | 0.9722 | -2.78% | 1.0691 | +6.91% | +9.67% | 0.804 |
| kernel_sum | reference | 1.5 | 642 | 1.0415 | +4.15% | 0.9822 | -1.78% | 1.0622 | +6.22% | +4.23% | 0.852 |
| two_sided | local | 3.0 | 161 | 0.7545 | -24.55% | 0.6592 | -34.08% | 1.0123 | +1.23% | +17.78% | 0.727 |
| two_sided | local | 2.0 | 362 | 0.7510 | -24.90% | 0.6752 | -32.48% | 0.8992 | +10.08% | +5.31% | 0.804 |
| two_sided | local | 1.5 | 642 | 0.7391 | -26.09% | 0.6829 | -31.71% | 0.8934 | +10.66% | +0.86% | 0.852 |
| two_sided | reference | 3.0 | 161 | 0.7545 | -24.55% | 0.6592 | -34.08% | 0.7864 | +21.36% | +17.78% | 0.727 |
| two_sided | reference | 2.0 | 362 | 0.7510 | -24.90% | 0.6752 | -32.48% | 0.7132 | +28.68% | +5.31% | 0.804 |
| two_sided | reference | 1.5 | 642 | 0.7391 | -26.09% | 0.6829 | -31.71% | 0.7102 | +28.98% | +0.86% | 0.852 |

推荐组合（`kernel_sum` + `reference`）随网格收敛：dx = 3.0 / 2.0 / 1.5 mm 下
F_cut/2σ = 1.202 / 1.069 / 1.062，
κ 偏差 +23.6% / +9.7% / +4.2%；
历史组合（`two_sided`+`local`，dx = 1.5 mm）的 F_cut/2σ = 0.893
是 δ_s 少 25% 与 ρ_i/ρ0 = 0.85 放大 35% 相消的「假通过」。

### 3.4 端到端 Young–Laplace（dx = 4 mm，t ≈ 0.89 t_σ，ρ 松弛）

| δ_s 方案 | 力密度 | Δp 实测 [Pa] | Δp 理论 σ/R [Pa] | 偏差 | t/t_σ | Adami 伪流 |
|---|---|---|---|---|---|---|
| two_sided | local | 389.2 | 360.0 | +8.1% | 0.89 | 超阈 |
| two_sided | reference | 360.5 | 360.0 | +0.1% | 0.89 | 超阈 |
| kernel_sum | local | 591.6 | 360.0 | +64.3% | 0.89 | 合格 |
| kernel_sum | reference | 788.4 | 360.0 | +119.0% | 0.89 | 超阈 |

Δp 读数在四个解析行为几乎相同的组合间散布 +0.1% … +119%，量本身在当前分辨率不可用 → 维持 WITHHELD。

## 4. 结果图

- `docs/reports/m3850-pr86-merge-audit.png`：左 Σδ_s n_x dV/2R、右 F_cut/2σ 对 dx（4 组合），
  画出 ±3% / ±15% 带与解析目标 1。
- `docs/reports/m3850-pr86-merge-audit-preservation.png`：每个共同修改文件两侧新增行数，
  标题内嵌「合并丢失行数 = 0」。

## 5. 注意点

- **不能 overlay**：m2926 与 main 都改过 `csf_consistent.csf_fields`，机械取一侧会丢另一侧的关键字参数
  （ours 的 δ_s 开关 / theirs 的曲率平滑开关），二者都在默认关闭/历史值下才保持回归数不变。
- **`n` 死变量**：ours 分支的 `n = pos.shape[0]` 在 main 的 lint 清理中已被删除，合并采用删除；
  新增行保全审计仍为 0 丢失（该行不是任何一侧的新增行）。
- **默认值语义**：`csf_curvature` 的默认是 `fit_quadratic`（main），`csf_delta_s`/`csf_force_density`
  的默认是历史值（m2926），所以合并后既有 benchmark 的数值不变；建议后续把
  `kernel_sum`+`reference` 提为默认需另立任务重跑全部回归。
- **环境**：`tensorsph` 的可编辑安装指向别的工作树，跑测试必须显式 `PYTHONPATH=<worktree>/src`，
  否则会静默导入别的分支的代码（本轮踩过：collection error 显示导入路径是 `work/m3482`）。
- 本次未重跑 `benchmarks/delta_s_normalization/run.py`（4 组合 × 3 分辨率 + 4 次松弛，成本高），
  3.3/3.4 的数字取自 m2926 入库的 `validation_report.json`，与合并树代码路径一致（该 benchmark 只依赖被合并的两个开关）。

## 6. 误差判定（关键量 vs 基准，逐项回答「是否 < 3%」）

| 关键量 | 基准 | dx = 2 mm 误差 | dx = 1.5 mm 误差 | < 3%？ |
|---|---|---|---|---|
| ∫δ_s dn | 1（解析归一） | +4.87% | +4.15% | 否（< 15% 门内，趋势向 1） |
| Σδ_s n_x dV/2R | 1（矢量恒等） | -2.78% | -1.78% | **是**（两档均 < 3%） |
| F_cut/2σ | 1（Laplace 力平衡） | +6.91% | +6.22% | 否（< 15% 门内） |
| κ | 1/R = 50.0 1/m | +9.67% | +4.23% | 否（dx=1.5 mm 4.2%） |
| 力方向 cos | 1 | 0.99972 | 0.99974 | 是（中位 cos 误差 < 0.03%） |
| Δp（端到端，dx = 4 mm） | σ/R = 360 Pa | — | — | **无可用基准**：四组合散布 0.1%…119%，判据本身失效 → WITHHELD |

**合并任务本身的验收判据（无需物理基准）**：冲突 3 处全部语义并集解冲、
两侧新增行缺失 0、
合并树 passed 369 ≥ 基线 363、
失败集合与基线相同、`ruff check .` 通过。

## 7. 局限

- 合并树与基线两次 pytest 都在一台高负载共享机（load average ≈ 200）上跑，墙钟数字仅供参考，判定只用 pass/fail 集合。
- `.scratch/` 下的历史调试脚本随分支带入，未清理（属 m2926 既有内容，删除会丢内容）。
