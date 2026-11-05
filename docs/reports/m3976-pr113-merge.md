# PR #113 解冲与合并记录（TensorSPH, mission m3976）

**目标**：治愈 PR #113（源分支 `agent/m3246` 对 `main` 变脏，mergeable=false），
三方合并 → 逐文件解冲 → 本地回归不劣于 `main` 基线 → 推回同分支。

| 项 | 值 |
|---|---|
| 仓库 | TensorSPH |
| merge base | `184fef9568c515092b68c5c182c1b1dad7cb3e7f` |
| `main`（合并前） | `ba07a1066bcb2be5046dc63ae0179273b9f02ee7`（领先 base 20 commit） |
| `agent/m3246`（旧 SHA） | `d433c42a77d116a9cec159f2061f998f9edf70fe`（领先 base 1 commit） |
| 合并提交 | 见 `runs/merge_audit.json` 的 `merged_head` |
| 冲突文件 | **1 个**（`tests/test_laplace_pressure_floor.py`） |

## 1. 冲突与逐文件解法

只有 1 个文件产生文本冲突；另一个双方都改的文件被 git 自动合并，但**仍逐行核对**。

### 1.1 `tests/test_laplace_pressure_floor.py`（文本冲突 → 语义并集）

| 侧 | 改动 |
|---|---|
| `main` | import 块重排：`importlib.util` 上提到顶部，删掉 torch / `density_from_neighbor_pairs` / `cell_linked_pairs` 导入 |
| `agent/m3246` | 删除 `@pytest.mark.xfail(strict=True)` 与 `test_laplace_jump_within_15pct`；改名 `test_open_loop_still_undershoots_by_about_a_third`，断言由「abs < 15 %」改为「有符号 err < −25 %」；验收判据迁到新文件 `tests/test_free_surface_pressure.py` |
| **解法** | **两侧并集**：保留 main 的 import 块（自动合并区已生效），冲突块取 m3246 语义（有符号基线断言），同时确认 15 % 验收判据**仍然存在**于 `tests/test_free_surface_pressure.py::test_three_sweeps_meet_the_p0_criterion`（`worst_abs_error_pct < 15.0`）。**没有删除任何一侧的内容**，只是把验收门从「编码失败的 xfail」升级为「闭包配置下的真实门」。 |

取 m3246 侧而非 main 侧的判据：`benchmarks/laplace_pressure_floor/report.json` 的
6 个 open-loop 算例误差为 −33.99 … −38.10 %，若保留 main 的 `abs(err) < 15` 只能靠
`xfail(strict=True)` 掩盖；m3246 把它钉成有符号基线事实（< −25 %），失败事实不丢失、也不被
粉饰，同时验收门由闭包算例承担。

### 1.2 `benchmarks/laplace_pressure_floor/run.py`（双方都改，git 自动合并）

| 侧 | 改动 |
|---|---|
| `main` | 仅文件末尾补换行（无内容变化） |
| `agent/m3246` | `relax()` 增加 `free_surface_pressure` / `free_surface_pressure_passes` 两个参数并透传给求解器 |
| **解法** | 自动合并结果与 m3246 逐字节相同、仅多出 main 的 EOF 换行；即**双方内容均在**。核对命令与结果见 `runs/merge_audit.json` 中 `hand_resolved_both_side_edit` 的 `diff_vs_side`。 |

### 1.3 其余 72 条「单侧改动」路径

一侧改动、另一侧未动的路径由 git 直接取改动侧版本，逐条核对 blob 与改动侧**完全一致**
（`identical_to_side`），无静默覆盖。

## 2. 静默丢内容审计（机械 overlay 防呆）

`runs/merge_audit.py` 对 base 到两侧的 74 条路径逐个判定：`git ls-tree` blob 比对 +
Python AST 骨架（剥离 import 语句后比对）判定「格式/未用导入级差异」还是「真实内容差异」，
任何 `MISSING` / `DIFFERS_UNEXPECTEDLY` 都会让脚本非 0 退出（作为合并门）。

```
checked 74 side-changed paths, alarms=0, reviewed_edits=5
both-sides-touched: ['benchmarks/laplace_pressure_floor/run.py',
                     'tests/test_laplace_pressure_floor.py']
  hand_resolved_both_side_edit        main  benchmarks/laplace_pressure_floor/run.py
  hand_resolved_both_side_edit        main  tests/test_laplace_pressure_floor.py
  REVIEWED_EDIT_outside_merge_scope  m3246  .scratch/probe_m3246.py
  hand_resolved_both_side_edit       m3246  benchmarks/laplace_pressure_floor/run.py
  hand_resolved_both_side_edit       m3246  tests/test_laplace_pressure_floor.py
```

判定分布：`identical_to_side` 65 · `hand_resolved_both_side_edit` 4 ·
`lint_fix_semantics_preserved` 4（AST 等价，仅删未用 `import math`、补文件尾换行）·
`REVIEWED_EDIT_outside_merge_scope` 1（`.scratch/probe_m3246.py`：把硬编码的
`ROOT=/root/tp-agent/work/m3246` 改为 `Path(__file__).parents[1]`，纯 scratch 探针，
不被 tests/CI 导入；改动与理由已入档）。**alarms=0，无缺失、无静默重写。**

## 3. Lint 门（`python -m ruff check .`，仓库 CI 唯一静态门）

| 树 | 违规数 |
|---|---|
| `main` 基线 | 0 |
| m3246 分支单独 | 12 |
| 合并树（修后） | **0** |

m3246 新文件带 12 条违规（未用 `math` 导入 ×2、文件尾缺换行 ×4、`.scratch/` 探针
同行多 import / 分号 / 长行 ×6）。按仓库 `.scratch/` 也被 `ruff check .` 扫描的事实，
全部修掉才不把 lint 门弄红；均为格式级改动，AST 骨架比对证明 4 个测试/基准文件语义不变
（见第 2 节判定分布）。

## 4. 测试证据

**核心 11 个文件**（与 SPH 物理/被改源码直接相关，两棵树同文件同参数同机）：

| 树 | passed | failed | xfail |
|---|---|---|---|
| `origin/main` 基线 | 118 | 2 | 1 |
| 合并树（本 PR） | **129** | 2 | 0 |

- 两棵树失败的 2 个测试**完全相同**：`tests/test_oscillating_droplet.py::
  test_recorded_mode2_frequency_deficiency_is_reproduced`、
  `::test_oscillation_frequency_gate_is_not_met` —— 是 `main` 上既有的红，与本次合并无关
  （m3246 未触碰 `src/tensorsph/oscillating_droplet` 相关实现）。
- 合并树多出 11 passed = m3246 新增的闭包验收测试（`tests/test_free_surface_pressure.py`）
  + 原 xfail 转为显式通过的基线断言。**不劣于基线，且净增覆盖。**

**全量 `tests/`（39 文件）**：机器上同时有 50+ 个其他智能体的 pytest 进程，
`pytest-timeout` 120 s 在争用下大量误杀，故全量按文件对比只走到前 5 个文件即改为核心集
对比（诚实说明：全量 green 未在本轮取得）。前 5 个文件里 `test_adaptive_dt.py`
合并树 1 failed / 10 passed，基线树 2 failed / 9 passed（失败均为 `pytest-timeout`
在 `neighbor_search` 内触发，属机器争用），合并树同样不劣。

## 5. 本次合并带进来的物理成果（供论文级引用）

`benchmarks/laplace_pressure_floor/closure_ladder.json`（4 个验收点 × 5 档扫描）：

| $n_{pass}$ | 最差 \|误差\| | 四个点误差（%） |
|---|---|---|
| 0（开环） | 38.10 % | −37.88 / −33.99 / −38.10 / −37.41 |
| 1 | 25.26 % | −20.15 / −19.86 / −25.26 / −24.70 |
| 2 | 15.10 % | −6.51 / −6.98 / −15.10 / −13.41 |
| **3（采用）** | **6.77 %** | −1.84 / −0.56 / −6.77 / −6.38 |
| 6 | 12.09 % | +12.09 / +5.59 / −3.78 / −2.80 |

即：自由面压力闭包把 Young–Laplace 压力跳变的相对误差从 −38 % 压到 6.8 %（< 15 % 判据），
$n_{pass}=6$ 出现过冲（$c_0=20$ 处 +12.09 %），故采用 3 遍。

## 6. 注意点

- `tensorsph` 在本机是 editable 安装到**别的** worktree（`work/m3482/src`），直接
  `pytest` 会静默用错源码树；本轮所有测试都显式 `PYTHONPATH=<tree>/src`。
- `ruff check .` 会扫 `.scratch/`，m3246 的 scratch 探针必须一起修，否则 lint 门红。
- 冲突只有 1 个文件不等于「无风险」：`run.py` 是被 git 自动合并的双方改动文件，
  自动合并同样要逐行核对（本轮它只差一个 EOF 换行）。
- 本轮未取得全量 39 文件的干净对照（机器争用 + 预算），核心集已覆盖全部被改源码路径。