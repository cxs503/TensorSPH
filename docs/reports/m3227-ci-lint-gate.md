# CI 红→绿：ruff 门禁 54 条违规清零（main@184fef9）

**任务**：m3227　**仓库**：TensorSPH　**分支**：`agent/m3227`
**基线**：`184fef9`（Merge PR #90）
**门禁**：`.github/workflows/ci.yml` → `python -m ruff check .`（`pyproject.toml`：`select = ["E", "F", "W"]`，`line-length = 100`，`target-version = "py310"`）

---

## ① 案例介绍

TensorSPH 的 CI 是「先 lint 后测试」的单 job 串行流水线：`Lint` → `pytest -q` →
`examples/droplet_impact_2d/run.py --steps 3` 冒烟 → run_manifest 断言，矩阵 `3.10 / 3.11 / 3.12`。
`Lint` 一红，后面的测试/冒烟/清单断言全部不执行 —— 于是 pytest 的真实状态在 CI 上完全不可见。
#85–#101 波新合入的单据把 #62 清过的 223 条 lint 债重新引入了一部分。

| 项 | 值 |
|---|---|
| 本单基线提交 | `184fef9` |
| 基线 `ruff check .` 违规数 | **54** |
| 违规规则分布 | `E501`×26, `W292`×15, `F401`×6, `E401`×2, `E702`×2, `F541`×2, `E402`×1 |
| 涉及文件数 | 15 |
| CI 死于哪个 step | `Lint`（3.10 / 3.11 / 3.12 三腿同死，同一份清单） |
| 验收判据 | ① 违规 = 0；② pytest 失败集与基线**同集**；③ 断言零改动 |

本单不引入任何数值算法改动，因此**不存在「与解析解误差 <3%」这类物理判据**；
判据换成两条机器等价性：违规计数 `54 → 0`，以及 pytest 失败集集合相等。

## ② 怎么做的

1. **先量后改**：在 `184fef9` 的干净导出（`git archive`）里跑
   `ruff check . --output-format=json`，把 54 条 `(file, row, col, code)`
   落盘成 `benchmarks/ci_lint_gate/ruff_before.json`，不在脏树上直接 `--fix`。
2. **按语义风险分层**：25 条机械类（行尾换行 / 未用 import / 空 f-string /
   多 import 同行）交给 `ruff check --fix`；29 条需人判（长行 26、分号 2、
   `sys.path` 前置导入 1）。
3. **长行的关键约束**：26 条里 14 条落在 `make_doc.py` 的三引号**文档模板**内部，
   而模板直接生成 `docs/verification_*.md`。改行长不许改内容，做法是
   「在字符串字面量内部用相邻字面量拼接断开」：`"...前段"` + `"后段..."`。
   f-string 两段都保留 `f` 前缀、断点绝不落在 `{...}` 字段里；多行模板则
   `..."""` 收尾再 `"""...` 续开。跨行拼接需要一层括号，故模板赋值与模块
   docstring 各加一对括号（`TEMPLATE = ("""...""")`），语义不变。
4. **两处真改结构的例外**（`benchmarks/oscillating_droplet/make_doc.py`）：
   把重复的长表达式提成局部变量 `amp_rel / amp_err_pct`、`omega_zc / period_zc`，
   并**保留原式符号口径**（`pct(amp_rel*100)` 吃带符号值、`yn(abs(...)*100 < 3)`
   吃绝对值）。渲染结果逐字节不变，见 ④ 的 diff 证据。
5. **`E402` 不做结构改造**：`sys.path.insert(0, ...)` 必须先于 `import tensorsph`，
   按仓库**既有约定**加 `# noqa: E402`（先例：`benchmarks/csf_curvature_fit/field_figure.py:14-17`）。
   除此之外全仓**没有新增任何 noqa**，也不改 `pyproject.toml` 的
   `select` / `ignore` / `extend-exclude` —— 门禁强度不降。
6. **不改 `tests/` 断言**：`tests/` 内只有 6 处改动 —— 5 个文件补行尾换行、
   1 个文件删 3 个未用 import 并把 `import importlib.util` 归到标准库组；
   `assert` 行零改动。

## ③ 结果：违规计数前后表

### 3.1 按规则

| 规则 | 含义 | 修复前 | 修复后 | 修法 |
|---|---|---|---|---|
| `E401` | multiple imports on one line | 2 | 0 | 拆成逐行 import（自动修复） |
| `E402` | module level import not at top | 1 | 0 | `sys.path` 前置导入脚本，按仓库既有约定加 `# noqa: E402` |
| `E501` | line too long | 26 | 0 | 在字符串字面量内部按相邻字面量拼接断开（输出零改动） |
| `E702` | multiple statements on one line | 2 | 0 | 按分号拆行 |
| `F401` | unused import | 6 | 0 | 删除（自动修复） |
| `F541` | f-string without placeholders | 2 | 0 | 去掉 `f` 前缀（自动修复） |
| `W292` | no newline at end of file | 15 | 0 | 补行尾换行（自动修复） |
| **合计** | — | **54** | **0** | — |

### 3.2 按文件

| 文件 | 修复前 | 修复后 | 规则码 |
|---|---|---|---|
| `benchmarks/laplace_pressure_floor/make_doc.py` | 22 | 0 | E501, F401, W292 |
| `.scratch/probe_front.py` | 6 | 0 | E401, E402, E702, F401, W292 |
| `benchmarks/oscillating_droplet/make_doc.py` | 5 | 0 | E501, F541, W292 |
| `tests/test_laplace_pressure_floor.py` | 4 | 0 | F401, W292 |
| `benchmarks/laplace_plateau/make_doc.py` | 3 | 0 | E501, F541, W292 |
| `.scratch/rep_list.py` | 2 | 0 | E401, W292 |
| `benchmarks/dam_break/make_audit_doc.py` | 2 | 0 | E501, W292 |
| `benchmarks/dam_break/reference_audit.py` | 2 | 0 | E501, W292 |
| `benchmarks/oscillating_droplet/run.py` | 2 | 0 | F401, W292 |
| `benchmarks/laplace_plateau/run.py` | 1 | 0 | W292 |
| `benchmarks/laplace_pressure_floor/run.py` | 1 | 0 | W292 |
| `tests/test_dam_break_reference_audit.py` | 1 | 0 | W292 |
| `tests/test_laplace_plateau.py` | 1 | 0 | W292 |
| `tests/test_oscillating_droplet.py` | 1 | 0 | W292 |
| `tests/test_ritter_reference.py` | 1 | 0 | W292 |

### 3.3 pytest：与基线同败集（存量红未增未洗）

| 状态 | 通过 | 失败 | xfail |
|---|---|---|---|
| main@184fef9 | 327 | 4 | 12 |
| agent/m3227 | 327 | 4 | 12 |

| 测试节点 | main@184fef9 | 修复后 | 结论 |
|---|---|---|---|
| `tests/test_laplace_law.py::test_committed_verdicts_are_self_consistent` | FAIL | FAIL | 存量红（未动） |
| `tests/test_laplace_pressure.py::test_effective_surface_tension_at_the_documented_resolution` | FAIL | FAIL | 存量红（未动） |
| `tests/test_oscillating_droplet.py::test_oscillation_frequency_matches_the_analytic_law` | FAIL | FAIL | 存量红（未动） |
| `tests/test_paper_laplace_uncertainty.py::test_figures_exist` | FAIL | FAIL | 存量红（未动） |

四条存量红都是**本单之前**就存在的账，本单只保证它们「不增、不洗」：

- `test_laplace_pressure.py::test_effective_surface_tension_at_the_documented_resolution`、
- `test_laplace_law.py::test_committed_verdicts_are_self_consistent`、
- `test_oscillating_droplet.py::test_oscillation_frequency_matches_the_analytic_law`

  这三条是仓库自己记的 WITHHELD / honest-FAIL 账（自由面压强闭合未做、
  振荡算例数值发散），修它们要改 `src/` 物理，不在 lint 单范围。
- `test_paper_laplace_uncertainty.py::test_figures_exist`：依赖未提交的生成图件
  （本仓库按惯例不跟踪 png），本地跑必红，CI 侧同样非本单引入。

lint 判定与 Python 小版本无关（ruff 是独立二进制，`target-version = py310`
固定语法目标）。**honest 说明**：本机只装了 CPython 3.11.6，3.10 / 3.12 两腿
未本地实测，只能给推论，不给实测结论；已实测的是 `ruff check .` = 0、`pytest -q`
= 327 passed / 4 存量红、examples 冒烟与 run_manifest 断言全过，即 CI 三个 step 的本地等价。

## ④ 误差判定（本单的「基准」是等价性，不是物理量）

| 判据项 | 基准 | 实测 | 判定 |
|---|---|---|---|
| `ruff check .` 违规数 | 0 | 0 | **PASS**（计数判等，3% 不适用） |
| 模板/文档字符串常量 | `184fef9` 版本 | 逐一比对一致 | **PASS**（差异 0 条） |
| `make_doc.py` 输出 | 已提交 `docs/*.md` | `git diff` 空（3/4 文件） | **PASS**（第 4 个见 ⑤） |
| pytest 失败集 | 基线 4 条 | 4 条、集合相同 | **PASS**（新增 0 条） |
| `tests/` 断言改动 | 0 行 | 0 行 | **PASS** |

**显式回答「是否 < 3%」**：本单无可与实验/解析解对比的物理量，3% 精度门不适用；
对应的「零差异」判据（违规 0 条、字符串常量差异 0 条、失败集差异 0 条）全部满足。

## ⑤ 注意点（踩过的坑）

1. **`E501` 量的是显示宽度，不是字符数**：ruff 按 Unicode East-Asian Width 计宽，
   一行 60 个汉字（=120 列）即超长；按 `len()` 排断点会漏一半。
2. **f-string 的 `{...}` 字段不能跨字面量拆**：断在字段中间会让两段都缺 `}`，
   直接 `SyntaxError`（本单先踩了一次，被 `py_compile` 拦住）。
3. **`.scratch/` 是被 git 跟踪的**：`ruff check .` 会扫它，6 条违规就出自
   `.scratch/probe_front.py`（它的 docstring 还写着 "not committed"）。
4. **不要用 `noqa` / 改 `select` 洗单**：唯一保留的 `# noqa: E402` 属结构性例外
   且仓库已有同款先例，其余 54 − 1 条全部真改。
5. **生成型文档的等价性要机验**：`make_doc.py` 重跑 + `git diff` 最硬；
   但与已提交版本**本来就不同**的文件（`docs/verification_oscillating_droplet.md`
   末尾 26 行是历史 hand-append 的「⑦ 诚实记录」，`make_doc.py` 不产出它）要用
   「改前生成 vs 改后生成」互比，否则会把历史差异误判成自己引入的。
6. **符号口径**：长表达式提局部变量时 `pct()` 吃带符号相对误差、`yn()` 吃绝对值；
   第一版误把带符号值换成 `abs()`，文档里 `-3.0%` 变 `+3.0%`，靠重生成 diff 抓到。
7. `benchmarks/ci_lint_gate/*.json` 是本单证据工件（54 + 0 条），
   重跑 `make_doc.py` 即可重建本文，无需手抄数字。

## ⑥ 复现（全部命令在本仓库根目录执行）

```bash
# 1. 基线导出（不改工作树、不动 .git）
mkdir -p /tmp/base && git archive 184fef9 -o /tmp/base2.tar
tar -xf /tmp/base2.tar -C /tmp/base

# 2. 前后两次 ruff 快照 + 两次 pytest 结果 -> benchmarks/ci_lint_gate/*.json
python benchmarks/ci_lint_gate/collect_evidence.py /tmp/base

# 3. 证明长行改写没有动过任何字符串常量（输出 OK = 逐一相同）
python benchmarks/ci_lint_gate/const_check.py \
    /tmp/base/benchmarks/laplace_pressure_floor/make_doc.py \
    benchmarks/laplace_pressure_floor/make_doc.py

# 4. 重渲染本文（含 png 图件到 artifacts/）
python benchmarks/ci_lint_gate/make_doc.py
```

（图件 `artifacts/ci-lint-gate/ruff_before_after.png` 不提交仓库：本仓库不跟踪 png。）
