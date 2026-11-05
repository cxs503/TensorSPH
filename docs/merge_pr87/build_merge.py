# ruff: noqa: E501  (verbatim source blocks are embedded as literals)
"""Three-way semantic merge helper for PR #87 (agent/m2919) -> main.

Builds the 4 conflicted files as a semantic union (both sides kept, nothing
silently dropped).  Inputs are the two clean sides extracted with ``git show``.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent          # docs/merge_pr87
REPO = HERE.parent.parent                        # repository root
SIDES = HERE / "sides"                           # git-show dumps of the two sides

MAIN_T = SIDES / "t_main.py.txt"
BR_T = SIDES / "t_br.py.txt"
MAIN_RUN = SIDES / "run_main.py.txt"
BR_RUN = SIDES / "run_br.py.txt"
MAIN_DOC = SIDES / "doc_main.md.txt"
BR_DOC = SIDES / "doc_br.md.txt"
MAIN_REP = SIDES / "rep_main.json.txt"
BR_REP = (REPO / "benchmarks" / "oscillating_droplet" /
          "validation_report_m2919.json")  # the archived branch report

BENCH = REPO / "benchmarks" / "oscillating_droplet"
DOCS = REPO / "docs"
TESTS = REPO / "tests"

log = []


# --------------------------------------------------------------------------- #
# 1. run.py -> main verbatim; branch implementation preserved as run_m2919.py
# --------------------------------------------------------------------------- #
run_main = MAIN_RUN.read_text()
(BENCH / "run.py").write_text(run_main)
log.append("benchmarks/oscillating_droplet/run.py <- main verbatim")

br = BR_RUN.read_text()

# provenance header, inserted right after the module docstring (so the
# docstring stays a docstring)
tree = ast.parse(br)
insert_line = 0
if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(
        tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
    insert_line = tree.body[0].end_lineno

HEADER = '''
# --------------------------------------------------------------------------- #
# Preserved PR #87 (branch agent/m2919) implementation, merged into main next
# to the canonical run.py instead of replacing it: the two sides are independent
# implementations of the same benchmark (lobed_droplet geometry + inertia-tensor
# mode observable + two-level (omega, gamma) grid fit, mode-2/mode-3 ladder),
# not versions of one script, so a textual overlay would silently delete one of
# them.  Run this one explicitly; outputs carry the *_m2919 suffix so a run here
# can never overwrite main's validation_report.json or figures.
#
#   PYTHONPATH=src python benchmarks/oscillating_droplet/run_m2919.py
# --------------------------------------------------------------------------- #
'''

lines = br.splitlines(keepends=True)
br_hdr = "".join(lines[:insert_line]) + HEADER + "".join(lines[insert_line:])

REPL = [
    ('(out / "validation_report.json").write_text',
     '(out / "validation_report_m2919.json").write_text'),
    ("""print(f"report -> {out / 'validation_report.json'}")""",
     """print(f"report -> {out / 'validation_report_m2919.json'}")"""),
    ('path = out_dir / "oscillating_droplet.png"',
     'path = out_dir / "oscillating_droplet_m2919.png"'),
    ('path2 = out_dir / "oscillating_droplet_decay.png"',
     'path2 = out_dir / "oscillating_droplet_m2919_decay.png"'),
]
# Semantic merge of a *configuration* default, not just text: main changed
# ``Config.csf_curvature`` from "mls" to "fit_quadratic" after this branch was
# cut, so the branch's validated numbers (and its <30 % solver guard) would
# silently become a different physics experiment.  Pin the scheme the branch's
# report was produced with and expose it as a parameter.
PIN = [
    ("""             sound_speed: float = 10.0, samples_per_period: int = 120,
             device: str = "cpu", store_field: bool = False) -> dict:""",
     """             sound_speed: float = 10.0, samples_per_period: int = 120,
             csf_curvature: str = "mls",
             device: str = "cpu", store_field: bool = False) -> dict:
    # ``csf_curvature`` is pinned to "mls" on purpose: main's Config default
    # drifted mls -> fit_quadratic after this benchmark was baselined, and the
    # numbers/tests below are only meaningful for the scheme they were tuned on."""),
    ("""                 adaptive_dt=False, density_diffusion=0.0, dt=dt)""",
     """                 adaptive_dt=False, density_diffusion=0.0, dt=dt,
                 csf_curvature=csf_curvature)"""),
]
for old, new in PIN:
    assert br_hdr.count(old) == 1, old
    br_hdr = br_hdr.replace(old, new)

for old, new in REPL:
    assert br_hdr.count(old) == 1, (old, br_hdr.count(old))
    br_hdr = br_hdr.replace(old, new)
# lint fixups so the preserved copy passes main's ruff gate (the branch never
# reached main, so it was never linted): E501 on the per-case print, W292 EOF.
LONG_PRINT = """        print(f"{name:9s} n={row['mode_n']} dx={row['dx_m']*1e3:.2f} mm "
              f"N={row['n_particles']:5d} omega={row['omega_measured']:8.2f}/"
              f"{row['omega_analytic']:8.2f} rad/s "
              f"({row['omega_error_pct'] if row['omega_error_pct'] is None else round(row['omega_error_pct'],1)}%) "
              f"gamma={row['gamma_measured']:.2f} 1/s resid={row['fit_residual_rel']:.3f} "
              f"t={row['wall_clock_s']:.0f}s")
"""
NEW_PRINT = """        err = ("-" if row["omega_error_pct"] is None
               else f"{row['omega_error_pct']:.1f}")
        print(f"{name:9s} n={row['mode_n']} dx={row['dx_m']*1e3:.2f} mm "
              f"N={row['n_particles']:5d} omega={row['omega_measured']:8.2f}/"
              f"{row['omega_analytic']:8.2f} rad/s ({err}%) "
              f"gamma={row['gamma_measured']:.2f} 1/s "
              f"resid={row['fit_residual_rel']:.3f} "
              f"t={row['wall_clock_s']:.0f}s")
"""
assert br_hdr.count(LONG_PRINT) == 1
br_hdr = br_hdr.replace(LONG_PRINT, NEW_PRINT)
br_hdr = br_hdr.rstrip("\n") + "\n"
ast.parse(br_hdr)
(BENCH / "run_m2919.py").write_text(br_hdr)
log.append("benchmarks/oscillating_droplet/run_m2919.py <- branch verbatim "
           "+ provenance header + *_m2919 output names")

# --------------------------------------------------------------------------- #
# 2. validation_report.json -> main; branch report kept under its own name
# --------------------------------------------------------------------------- #
main_rep = MAIN_REP.read_text()
br_rep = BR_REP.read_text()
json.loads(main_rep)  # validate
json.loads(br_rep)
(BENCH / "validation_report.json").write_text(main_rep)
(BENCH / "validation_report_m2919.json").write_text(br_rep)
log.append("benchmarks/oscillating_droplet/validation_report.json <- main "
           "(135 lines); branch's 8934-line report -> validation_report_m2919.json")

# --------------------------------------------------------------------------- #
# 3. tests -> union: main's suite verbatim + branch's suite bound to run_m2919
# --------------------------------------------------------------------------- #
t_main = MAIN_T.read_text()
t_br = BR_T.read_text()

br_tree = ast.parse(t_br)
br_doc = ast.get_docstring(br_tree)
# drop the branch module docstring, turn it into a comment block (no info loss)
doc_lines = t_br.splitlines()
br_body_lines = t_br.splitlines(keepends=True)
br_no_doc = "".join(br_body_lines[br_tree.body[0].end_lineno:])  # skip docstring

IMPORT_RE = re.compile(
    r"from benchmarks\.oscillating_droplet\.run import \([^)]*\)\n", re.S)
m = IMPORT_RE.search(br_no_doc)
assert m, "branch import block not found"

LOADER = '''# The branch's tests are bound to the preserved PR #87 implementation, not to
# the canonical run.py (the two expose different APIs: lobed_droplet /
# drop_frequency_2d / two-level fit vs elliptical_drop / analytic_omega).
def _bench_m2919():
    spec = importlib.util.spec_from_file_location(
        "oscillating_run_m2919", BENCH / "run_m2919.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_M2919 = _bench_m2919()
drop_frequency_2d = _M2919.drop_frequency_2d
drop_frequency_3d = _M2919.drop_frequency_3d
fit_damped_oscillation = _M2919.fit_damped_oscillation
lobed_droplet = _M2919.lobed_droplet
mode_amplitude = _M2919.mode_amplitude
run_case = _M2919.run_case
zero_crossing_period = _M2919.zero_crossing_period
'''
br_tests = IMPORT_RE.sub(lambda _m: LOADER, br_no_doc, count=1)

# main's module docstring stays first; a banner separates the two suites and
# carries the branch's intent verbatim.
banner = (
    "\n\n# ========================================================================== #\n"
    "# Suite 2 -- preserved from PR #87 (branch agent/m2919).\n"
    "# Merged as a union with Suite 1 above: each suite guards a different\n"
    "# implementation (run.py vs run_m2919.py) and neither replaces the other.\n"
    "# Original module docstring, verbatim:\n"
    + "".join(("#   " + ln).rstrip() + "\n" for ln in (br_doc or "").splitlines())
    + "# ========================================================================== #\n"
)

# branch module docstring/imports already exist in main's header (future, math,
# pytest, torch); keep only the numpy import the branch suite needs and drop the
# duplicated ones so the merged module has a single import block.
DUP = ("from __future__ import annotations\n", "import math\n",
       "import pytest\n", "import torch\n")
for d in DUP:
    br_tests = br_tests.replace(d, "", 1)
assert "import numpy as np" in br_tests
br_tests = br_tests.replace("import numpy as np\n", "", 1)
assert "import numpy as np" not in t_main
t_main = t_main.replace("import math\n", "import math\n\nimport numpy as np\n", 1)


# main marks simulation-heavy tests ``slow`` (conftest.py); the branch's two
# solver regressions run the real SPH solver for one capillary period, so they
# get the same mark instead of silently lengthening the default suite.
for fn in ("def test_mode_oscillates_near_the_capillary_frequency(self):",
           "def test_without_surface_tension_the_mode_does_not_oscillate(self):"):
    assert br_tests.count(fn) == 1, fn
    br_tests = br_tests.replace("    " + fn, "    @pytest.mark.slow\n    " + fn)

merged = t_main.rstrip("\n") + "\n" + banner + br_tests.strip("\n") + "\n"
ast.parse(merged)
(TESTS / "test_oscillating_droplet.py").write_text(merged)
log.append("tests/test_oscillating_droplet.py <- main suite (17 tests) "
           "+ branch suite (19 tests bound to run_m2919)")

# --------------------------------------------------------------------------- #
# 4. docs -> union: main's doc as canonical, branch's doc as an appendix with
#    its path references retargeted at the preserved files.
# --------------------------------------------------------------------------- #
doc_main = MAIN_DOC.read_text()
doc_br = BR_DOC.read_text()

fixed = doc_br
for old, new in [
    ("`benchmarks/oscillating_droplet/{run.py,validation_report.json,oscillating_droplet*.png}`",
     "`benchmarks/oscillating_droplet/{run_m2919.py,validation_report_m2919.json,"
     "oscillating_droplet_m2919*.png}`"),
    ("`benchmarks/oscillating_droplet/run.py`", "`benchmarks/oscillating_droplet/run_m2919.py`"),
    ("oscillating_droplet/oscillating_droplet.png",
     "oscillating_droplet/oscillating_droplet_m2919.png"),
    ("oscillating_droplet/oscillating_droplet_decay.png",
     "oscillating_droplet/oscillating_droplet_m2919_decay.png"),
]:
    fixed = fixed.replace(old, new)

# branch headings are demoted one level so the appendix nests under main's doc
fixed = re.sub(r"^## ", "### ", fixed, flags=re.M)
fixed = re.sub(r"^# ", "## ", fixed, count=1, flags=re.M)

appendix = (
    "\n\n---\n\n"
    "## 附 A：PR #87（agent/m2919）独立实现与结论（保留原文，仅改工件路径）\n\n"
    "> 本节来自分支 `agent/m2919` 的同名文档，属**独立实现**（`run_m2919.py`，lobed 几何 +\n"
    "> 惯性张量模态 + 两级网格拟合，mode-2/mode-3 阶梯），与正文的 `run.py` 并行保留：两套实现\n"
    "> 的数字不可互相替代，合并时既未覆盖也未删除任何一侧。工件路径已改为 `*_m2919` 以免与正文\n"
    "> 的 `validation_report.json` / 图重名。\n\n"
    + fixed.strip("\n") + "\n"
)
(DOCS / "verification_oscillating_droplet.md").write_text(doc_main.rstrip("\n") + appendix)
log.append("docs/verification_oscillating_droplet.md <- main body + 附 A (branch doc, "
           "paths retargeted)")

print("\n".join(log))
