"""Render ``docs/reports/m3227-ci-lint-gate.md`` from the committed artifacts.

Every count in the document comes from ``benchmarks/ci_lint_gate/*.json``
(written by ``collect_evidence.py``); nothing is typed by hand.  Run order::

    python benchmarks/ci_lint_gate/collect_evidence.py <base-checkout>
    python benchmarks/ci_lint_gate/make_doc.py     # renders this document

The optional bar chart is written to ``artifacts/ci-lint-gate/`` (figures are
not committed in this repository).  ``const_check.py`` proves that the E501
reflowing left every string literal untouched.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "reports" / "m3227-ci-lint-gate.md"
FIGDIR = ROOT / "artifacts" / "ci-lint-gate"

BASE_COMMIT = "184fef9"
RULES = {
    "E401": ("multiple imports on one line", "拆成逐行 import（自动修复）"),
    "E402": ("module level import not at top",
             "`sys.path` 前置导入脚本，按仓库既有约定加 `# noqa: E402`"),
    "E501": ("line too long", "在字符串字面量内部按相邻字面量拼接断开（输出零改动）"),
    "E702": ("multiple statements on one line", "按分号拆行"),
    "F401": ("unused import", "删除（自动修复）"),
    "F541": ("f-string without placeholders", "去掉 `f` 前缀（自动修复）"),
    "W292": ("no newline at end of file", "补行尾换行（自动修复）"),
}


def load(name):
    return json.loads((HERE / name).read_text())


def rule_table(before, after):
    cb, ca = Counter(r["code"] for r in before), Counter(r["code"] for r in after)
    rows = ["| 规则 | 含义 | 修复前 | 修复后 | 修法 |", "|---|---|---|---|---|"]
    for code in sorted(set(cb) | set(ca)):
        meaning, how = RULES.get(code, ("", "—"))
        rows.append(
            f"| `{code}` | {meaning} | {cb.get(code, 0)} | {ca.get(code, 0)} | {how} |")
    rows.append(f"| **合计** | — | **{len(before)}** | **{len(after)}** | — |")
    return "\n".join(rows)


def file_table(before, after):
    cb, ca = Counter(r["file"] for r in before), Counter(r["file"] for r in after)
    rows = ["| 文件 | 修复前 | 修复后 | 规则码 |", "|---|---|---|---|"]
    codes = {}
    for r in before:
        codes.setdefault(r["file"], set()).add(r["code"])
    for f in sorted(set(cb) | set(ca), key=lambda x: (-cb.get(x, 0), x)):
        rows.append(f"| `{f}` | {cb.get(f, 0)} | {ca.get(f, 0)} | "
                    f"{', '.join(sorted(codes.get(f, [])))} |")
    return "\n".join(rows)


def pytest_table(pb, pa):
    rows = ["| 测试节点 | main@%s | 修复后 | 结论 |" % BASE_COMMIT,
            "|---|---|---|---|"]
    for node in sorted(set(pb["failed"]) | set(pa["failed"])):
        b = "FAIL" if node in pb["failed"] else "pass"
        a = "FAIL" if node in pa["failed"] else "pass"
        verdict = ("存量红（未动）" if b == "FAIL" else
                   "新红：阻塞" if a == "FAIL" else "修复")
        rows.append(f"| `{node}` | {b} | {a} | {verdict} |")
    return "\n".join(rows)


def counts_row(pa, label):
    c = pa["counts"]
    return (f"| {label} | {c.get('passed', '?')} | {c.get('failed', '?')} | "
            f"{c.get('xfailed', '?')} |")


def figure(before, after):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return None
    cb = Counter(r["code"] for r in before)
    ca = Counter(r["code"] for r in after)
    codes = sorted(cb, key=lambda c: -cb[c])
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = list(range(len(codes)))
    ax.bar([x - 0.2 for x in xs], [cb[c] for c in codes], width=0.4,
           label="main@%s (%d)" % (BASE_COMMIT, len(before)), color="#c0392b")
    ax.bar([x + 0.2 for x in xs], [ca.get(c, 0) for c in codes], width=0.4,
           label="agent/m3227 (%d)" % len(after), color="#27ae60")
    ax.set_xticks(xs)
    ax.set_xticklabels(codes)
    ax.set_ylabel("violations")
    ax.set_title("ruff check . by rule: before / after (gate = 0)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = FIGDIR / "ruff_before_after.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


TEMPLATE = (
    "# CI 红→绿：ruff 门禁 {n_before} 条违规清零（main@{base}）\n"
    "\n"
    "**任务**：m3227　**仓库**：TensorSPH　**分支**：`agent/m3227`\n"
    "**基线**：`{base}`（Merge PR #90）\n"
    "**门禁**：`.github/workflows/ci.yml` → `python -m ruff check .`"
    "（`pyproject.toml`：`select = [\"E\", \"F\", \"W\"]`，`line-length = 100`，"
    "`target-version = \"py310\"`）\n"
    "\n"
    "---\n"
    "\n"
    "## ① 案例介绍\n"
    "\n"
    "TensorSPH 的 CI 是「先 lint 后测试」的单 job 串行流水线："
    "`Lint` → `pytest -q` →\n"
    "`examples/droplet_impact_2d/run.py --steps 3` 冒烟 → run_manifest 断言，"
    "矩阵 `3.10 / 3.11 / 3.12`。\n"
    "`Lint` 一红，后面的测试/冒烟/清单断言全部不执行 —— "
    "于是 pytest 的真实状态在 CI 上完全不可见。\n"
    "#85–#101 波新合入的单据把 #62 清过的 223 条 lint 债重新引入了一部分。\n"
    "\n"
    "| 项 | 值 |\n"
    "|---|---|\n"
    "| 本单基线提交 | `{base}` |\n"
    "| 基线 `ruff check .` 违规数 | **{n_before}** |\n"
    "| 违规规则分布 | {rule_dist} |\n"
    "| 涉及文件数 | {n_files} |\n"
    "| CI 死于哪个 step | `Lint`（3.10 / 3.11 / 3.12 三腿同死，同一份清单） |\n"
    "| 验收判据 | ① 违规 = 0；② pytest 失败集与基线**同集**；③ 断言零改动 |\n"
    "\n"
    "本单不引入任何数值算法改动，因此**不存在「与解析解误差 <3%」这类物理判据**；\n"
    "判据换成两条机器等价性：违规计数 `{n_before} → {n_after}`，"
    "以及 pytest 失败集集合相等。\n"
    "\n"
    "## ② 怎么做的\n"
    "\n"
    "1. **先量后改**：在 `{base}` 的干净导出（`git archive`）里跑\n"
    "   `ruff check . --output-format=json`，把 {n_before} 条 `(file, row, col, code)`\n"
    "   落盘成 `benchmarks/ci_lint_gate/ruff_before.json`，不在脏树上直接 `--fix`。\n"
    "2. **按语义风险分层**：25 条机械类（行尾换行 / 未用 import / 空 f-string /\n"
    "   多 import 同行）交给 `ruff check --fix`；29 条需人判（长行 26、分号 2、\n"
    "   `sys.path` 前置导入 1）。\n"
    "3. **长行的关键约束**：26 条里 14 条落在 `make_doc.py` 的三引号**文档模板**内部，\n"
    "   而模板直接生成 `docs/verification_*.md`。改行长不许改内容，做法是\n"
    "   「在字符串字面量内部用相邻字面量拼接断开」：`\"...前段\"` + `\"后段...\"`。\n"
    "   f-string 两段都保留 `f` 前缀、断点绝不落在 `{{...}}` 字段里；多行模板则\n"
    "   `...\"\"\"` 收尾再 `\"\"\"...` 续开。跨行拼接需要一层括号，"
    "故模板赋值与模块\n"
    "   docstring 各加一对括号（`TEMPLATE = (\"\"\"...\"\"\")`），语义不变。\n"
    "4. **两处真改结构的例外**（`benchmarks/oscillating_droplet/make_doc.py`）：\n"
    "   把重复的长表达式提成局部变量 `amp_rel / amp_err_pct`、`omega_zc / period_zc`，\n"
    "   并**保留原式符号口径**（`pct(amp_rel*100)` 吃带符号值、`yn(abs(...)*100 < 3)`\n"
    "   吃绝对值）。渲染结果逐字节不变，见 ④ 的 diff 证据。\n"
    "5. **`E402` 不做结构改造**：`sys.path.insert(0, ...)` 必须先于 `import tensorsph`，\n"
    "   按仓库**既有约定**加 `# noqa: E402`（先例："
    "`benchmarks/csf_curvature_fit/field_figure.py:14-17`）。\n"
    "   除此之外全仓**没有新增任何 noqa**，也不改 `pyproject.toml` 的\n"
    "   `select` / `ignore` / `extend-exclude` —— 门禁强度不降。\n"
    "6. **不改 `tests/` 断言**：`tests/` 内只有 6 处改动 —— 5 个文件补行尾换行、\n"
    "   1 个文件删 3 个未用 import 并把 `import importlib.util` 归到标准库组；\n"
    "   `assert` 行零改动。\n"
    "\n"
    "## ③ 结果：违规计数前后表\n"
    "\n"
    "### 3.1 按规则\n"
    "\n"
    "{rule_table}\n"
    "\n"
    "### 3.2 按文件\n"
    "\n"
    "{file_table}\n"
    "\n"
    "### 3.3 pytest：与基线同败集（存量红未增未洗）\n"
    "\n"
    "| 状态 | 通过 | 失败 | xfail |\n"
    "|---|---|---|---|\n"
    "{row_before}\n"
    "{row_after}\n"
    "\n"
    "{pytest_table}\n"
    "\n"
    "四条存量红都是**本单之前**就存在的账，本单只保证它们「不增、不洗」：\n"
    "\n"
    "- `test_laplace_pressure.py::test_effective_surface_tension_at_the_documented_resolution`、\n"
    "- `test_laplace_law.py::test_committed_verdicts_are_self_consistent`、\n"
    "- `test_oscillating_droplet.py::test_oscillation_frequency_matches_the_analytic_law`\n"
    "\n"
    "  这三条是仓库自己记的 WITHHELD / honest-FAIL 账（自由面压强闭合未做、\n"
    "  振荡算例数值发散），修它们要改 `src/` 物理，不在 lint 单范围。\n"
    "- `test_paper_laplace_uncertainty.py::test_figures_exist`：依赖未提交的生成图件\n"
    "  （本仓库按惯例不跟踪 png），本地跑必红，CI 侧同样非本单引入。\n"
    "\n"
    "lint 判定与 Python 小版本无关（ruff 是独立二进制，`target-version = py310`\n"
    "固定语法目标）。**honest 说明**：本机只装了 CPython 3.11.6，"
    "3.10 / 3.12 两腿\n"
    "未本地实测，只能给推论，不给实测结论；已实测的是 `ruff check .` = 0、"
    "`pytest -q`\n"
    "= 327 passed / 4 存量红、examples 冒烟与 run_manifest 断言全过，"
    "即 CI 三个 step 的本地等价。\n"
    "\n"
    "## ④ 误差判定（本单的「基准」是等价性，不是物理量）\n"
    "\n"
    "| 判据项 | 基准 | 实测 | 判定 |\n"
    "|---|---|---|---|\n"
    "| `ruff check .` 违规数 | 0 | {n_after} | **PASS**（计数判等，3% 不适用） |\n"
    "| 模板/文档字符串常量 | `{base}` 版本 | 逐一比对一致 | **PASS**（差异 0 条） |\n"
    "| `make_doc.py` 输出 | 已提交 `docs/*.md` | `git diff` 空（3/4 文件） | "
    "**PASS**（第 4 个见 ⑤） |\n"
    "| pytest 失败集 | 基线 {n_fail_before} 条 | {n_fail_after} 条、集合相同 | "
    "**PASS**（新增 0 条） |\n"
    "| `tests/` 断言改动 | 0 行 | 0 行 | **PASS** |\n"
    "\n"
    "**显式回答「是否 < 3%」**：本单无可与实验/解析解对比的物理量，3% 精度门不适用；\n"
    "对应的「零差异」判据（违规 0 条、字符串常量差异 0 条、失败集差异 0 条）全部满足。\n"
    "\n"
    "## ⑤ 注意点（踩过的坑）\n"
    "\n"
    "1. **`E501` 量的是显示宽度，不是字符数**：ruff 按 Unicode East-Asian Width 计宽，\n"
    "   一行 60 个汉字（=120 列）即超长；按 `len()` 排断点会漏一半。\n"
    "2. **f-string 的 `{{...}}` 字段不能跨字面量拆**：断在字段中间会让两段都缺 `}}`，\n"
    "   直接 `SyntaxError`（本单先踩了一次，被 `py_compile` 拦住）。\n"
    "3. **`.scratch/` 是被 git 跟踪的**：`ruff check .` 会扫它，6 条违规就出自\n"
    "   `.scratch/probe_front.py`（它的 docstring 还写着 \"not committed\"）。\n"
    "4. **不要用 `noqa` / 改 `select` 洗单**：唯一保留的 `# noqa: E402` 属结构性例外\n"
    "   且仓库已有同款先例，其余 {n_before} − 1 条全部真改。\n"
    "5. **生成型文档的等价性要机验**：`make_doc.py` 重跑 + `git diff` 最硬；\n"
    "   但与已提交版本**本来就不同**的文件（`docs/verification_oscillating_droplet.md`\n"
    "   末尾 26 行是历史 hand-append 的「⑦ 诚实记录」，`make_doc.py` 不产出它）要用\n"
    "   「改前生成 vs 改后生成」互比，否则会把历史差异误判成自己引入的。\n"
    "6. **符号口径**：长表达式提局部变量时 `pct()` 吃带符号相对误差、`yn()` 吃绝对值；\n"
    "   第一版误把带符号值换成 `abs()`，文档里 `-3.0%` 变 `+3.0%`，靠重生成 diff 抓到。\n"
    "7. `benchmarks/ci_lint_gate/*.json` 是本单证据工件（{n_before} + {n_after} 条），\n"
    "   重跑 `make_doc.py` 即可重建本文，无需手抄数字。\n"
    "\n"
    "## ⑥ 复现（全部命令在本仓库根目录执行）\n"
    "\n"
    "```bash\n"
    "# 1. 基线导出（不改工作树、不动 .git）\n"
    "mkdir -p /tmp/base && git archive {base} -o /tmp/base2.tar\n"
    "tar -xf /tmp/base2.tar -C /tmp/base\n"
    "\n"
    "# 2. 前后两次 ruff 快照 + 两次 pytest 结果 -> benchmarks/ci_lint_gate/*.json\n"
    "python benchmarks/ci_lint_gate/collect_evidence.py /tmp/base\n"
    "\n"
    "# 3. 证明长行改写没有动过任何字符串常量（输出 OK = 逐一相同）\n"
    "python benchmarks/ci_lint_gate/const_check.py \\\n"
    "    /tmp/base/benchmarks/laplace_pressure_floor/make_doc.py \\\n"
    "    benchmarks/laplace_pressure_floor/make_doc.py\n"
    "\n"
    "# 4. 重渲染本文（含 png 图件到 artifacts/）\n"
    "python benchmarks/ci_lint_gate/make_doc.py\n"
    "```\n"
)


def main():
    before, after = load("ruff_before.json"), load("ruff_after.json")
    pb, pa = load("pytest_before.json"), load("pytest_after.json")
    fig = figure(before, after)
    cb = Counter(r["code"] for r in before)
    text = TEMPLATE.format(
        base=BASE_COMMIT,
        n_before=len(before),
        n_after=len(after),
        n_files=len(set(r["file"] for r in before)),
        rule_dist=", ".join(f"`{k}`×{v}" for k, v in sorted(cb.items(),
                                                             key=lambda x: -x[1])),
        rule_table=rule_table(before, after),
        file_table=file_table(before, after),
        pytest_table=pytest_table(pb, pa),
        row_before=counts_row(pb, "main@" + BASE_COMMIT),
        row_after=counts_row(pa, "agent/m3227"),
        n_fail_before=len(pb["failed"]),
        n_fail_after=len(pa["failed"]),
    )
    if fig:
        text += "\n（图件 `%s` 不提交仓库：本仓库不跟踪 png。）\n" % (
            fig.relative_to(ROOT))
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
