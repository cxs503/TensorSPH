# CI lint gate: clear the residual debt in the benchmark doc generators (m4685)

## 1. 背景与定位

`main` 的 CI 第一步是 `python -m ruff check .`，它**先于** pytest 运行，所以
lint 一红，整条流水线（含全部单测）都不会给出结论。这条门禁已因**与功能无关的
benchmark/文档脚本**红过多次（m3804、m4638），每次靠人工扫。

本单是 PR #307（原 m4609）的重开。定位时先核对了当前 `main`：

- PR #308（m4635）已经把原 m4609 清单里 **9 个文件中的 7 个**落地了（删多余
  `import math`、补文件尾换行、两个测试显式写 `density_calibration=False`）；
- 但 **lint 门禁在 `main` 上仍然是红的**：33 个错误（32 × E501 + 1 × F841），
  全部落在 benchmark 目录，其中 `F841` 是**真 bug 级**的死变量。

因此本单交付的是**剩余 delta**：把 `ruff check .` 在 `main` 上打到 0，并把
“谁来豁免、豁免多宽”写成可机器检查的约束。

## 2. 改了什么（3 处，最小改动）

| 位置 | 改动 | 理由 |
|---|---|---|
| `pyproject.toml` | 新增 `[tool.ruff.lint.per-file-ignores]`：`"benchmarks/**/make_doc.py" = ["E501"]` | 这两个脚本把长中文 markdown/LaTeX 行写进 `docs/`，**100 列上限对 CJK 文本无意义**；它们不是运行时路径。只豁免 E501 |
| `benchmarks/contact_angle_geometry/run.py` | 删除 `far = d[:n_fluid] > 2.0 * h`（F841，死变量） | 文档承诺的控制项 C2（`2h` 之外逐位不变）在下方用 `far_all = d > 2.0 * h` **已经正确实现**（`locality_max_abs_delta` / `locality_result`），这行只是残留副本 |
| `tests/test_lint_gate.py`（新增） | ①跑 CI 原命令并断言 `returncode == 0`；②解析 `pyproject.toml` 断言豁免**只**覆盖 `benchmarks/**/make_doc.py` 且**只**含 `E501`、`exclude` 里不含 `benchmarks`、`select` 仍是 `{E,F,W}` | 让门禁从“人工记得扫”变成“回归测试会红” |

### 为什么不采用 m4609 草稿的 `exclude = [".scratch", "benchmarks"]`

那一版把**整棵 benchmark 树**从 lint 里摘出去，等于连带放弃 `run.py` 的检查 ——
而本单清掉的 F841（死变量）恰恰就在 `run.py` 里，会被那版豁免永久掩盖。
`per-file-ignores` 精确到 `make_doc.py` 且只免 E501，pyflakes（未用导入/未定义名/
死变量）在所有 benchmark runner 上**继续生效**。`test_ruff_waiver_is_narrow`
就是防止后人把这条豁免扩宽的门。

## 3. 结果

| 状态 | `ruff check .` |
|---|---|
| `origin/main` @ 3605ebe（改前） | **33 errors**（29×E501 `contact_angle_geometry/make_doc.py`、3×E501 `laplace_pressure_jump/make_doc.py`、1×F841 `contact_angle_geometry/run.py`） |
| 本单（改后） | **0 errors** |

复现：

```bash
python -m ruff check .                      # All checks passed!
python -m pytest -q tests/test_lint_gate.py # 2 passed
```

## 4. 注意点 / 局限

1. **豁免是判据的一部分，不是绕过**：`E501` 只对 `make_doc.py` 免；若有人把长行
   写进 `run.py`，门禁照红。要新增豁免必须先改 `test_ruff_waiver_is_narrow`，
   即豁免范围的扩张必须在测试里显式表达。
2. `test_lint_gate.py` 依赖 `ruff`（在 `.[test]` extra 里）；未安装时**跳过**而不是
   失败，避免本地裸环境假红。CI 装 `.[test]`，因此该测试在 CI 里真实执行。
3. 本单**不含**运行时刻（wall-clock）意义上的性能/数值判据 —— 它是门禁修复，
   正确性判据就是“`ruff check .` 返回 0 且既有测试全绿”，见第 3 节与 mission 报告
   里的 pytest 汇总。
4. 未做什么：没有改 benchmark 的数值逻辑，没有重跑 `make_doc.py` 生成文档
   （改动全在注释/豁免/死代码层面，生成的 markdown 逐字节不变）。

## 5. 误差判定

本任务无物理量基准，故不适用「vs 实验/解析解 <3%」判据。可判定的量是**离散的**
lint 计数与测试通过数，二者都是精确值、无误差：

| 关键量 | 基准（期望） | 实测 | 误差 |
|---|---|---|---|
| `ruff check .` 错误数 | 0 | 0 | 0（精确） |
| `tests/test_lint_gate.py` | 2 passed | 2 passed | 0（精确） |
| 全套 pytest | 0 failed | 见 mission 报告汇总 | 0 failed |

## 6. 全套测试结果与既有失败的归因（honest）

`PYTHONPATH=$PWD/src python -m pytest -q`（本分支，约 9 min）：

```
4 failed, 494 passed, 9 xfailed
FAILED tests/test_laplace_pressure_jump.py::test_figure_exists
FAILED tests/test_oscillating_droplet.py::test_recorded_mode2_frequency_deficiency_is_reproduced
FAILED tests/test_oscillating_droplet.py::test_oscillation_frequency_gate_is_not_met
FAILED tests/test_oscillating_droplet_lamb.py::TestSolverRegression::test_mode_oscillates_near_the_capillary_frequency
```

基线对照（`git worktree add /tmp/base4685 origin/main`，**未打本单改动**，只跑受影响的测试）：

```
# /tmp/base4685, PYTHONPATH=/tmp/base4685/src
3 failed, 32 passed in 323.82s
FAILED tests/test_oscillating_droplet.py::test_recorded_mode2_frequency_deficiency_is_reproduced
FAILED tests/test_oscillating_droplet.py::test_oscillation_frequency_gate_is_not_met
FAILED tests/test_oscillating_droplet_lamb.py::TestSolverRegression::test_mode_oscillates_near_the_capillary_frequency
```

即：**这 3 项失败先于本单存在**，与本次改动无因果关系（本单未触碰 `src/` 与这三个测试文件）。

`test_figure_exists` 同理是既有欠账：它断言
`benchmarks/laplace_pressure_jump/laplace_pressure_jump.png` 存在，而 `git ls-files` 显示该路径
**从未入库**（该目录只跟踪 `make_doc.py` / `run.py` / `validation_report.json`）。补这张图属于
另一个功能点（二进制产物），不并入本 lint 修复单，以免混单。

环境备注：本机 `tensorsph` 是 editable 安装且指向**另一个 worktree**（`work/m4635/src`）。首轮全套
测试实际导入的不是本 worktree 的 `src/`；已用 `diff -rq` 证明两者逐字节相同（仅差
`__pycache__`/`egg-info`），复跑时显式设 `PYTHONPATH=$PWD/src`。跨 worktree 验证必须显式固定导入路径。
