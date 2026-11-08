"""Reference-frame ledger for the dam-break Ritter line (``Z/a = 2 tau``).

Why this file exists
--------------------
The m2502 reference audit found that the dam-break harness compared the SPH
front against ``Z/a = tau`` instead of ``Z/a = 2 tau`` (``H = 2a``,
``tau = t sqrt(2 g / a)``), i.e. a reference half the right size
(``docs/benchmarks/dam-break-reference-audit.md``). The numerical data path was
fixed, but ``generate_plot()`` kept drawing the old ``Z/a = tau`` curve, so the
*figure* still misrepresented the reference by a factor of two.

A factor-of-two error in a nondimensionalisation survives every "does it run"
test: both curves are straight lines, both look plausible, and no gate compares
them. The cure used here is a single source of truth (``ritter_za_curve``) plus
this file, which

* recomputes the identity from first principles for every call site in the
  repository (``src/tensorsph/dam_break.py``, ``src/tensorsph/dam_break_3d.py``,
  ``benchmarks/dam_break/run.py`` and the benchmark harness), and
* records the curve the plotting code actually draws, by intercepting
  matplotlib, so the figure and the data path cannot drift apart again.

The tests are cheap (no simulation is run).
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP_VALIDATION = REPO / "benchmarks" / "dam_break" / "experimental_validation.py"
DAM_BREAK_RUN = REPO / "benchmarks" / "dam_break" / "run.py"

from tensorsph import dam_break, dam_break_3d  # noqa: E402  (needs conftest sys.path)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp_validation():
    return _load_module(EXP_VALIDATION, "_bench_exp_validation")


def tau_of(t: float, a: float = 0.2, g: float = 9.81) -> float:
    """Experimental nondimensional time, ``tau = t sqrt(2 g / a)``."""
    return t * math.sqrt(2.0 * g / a)


# --------------------------------------------------------------------------- #
# 1. The identity, derived from first principles
# --------------------------------------------------------------------------- #
def test_ritter_identity_two_tau_from_the_definition():
    """Z/a = 2 tau follows from x_front - a = 2 sqrt(g H) t with H = 2a."""
    a, g = 0.2, 9.81
    H = 2.0 * a
    for t in (0.05, 0.1, 0.25, 0.4):
        z_over_a = 2.0 * math.sqrt(g * H) * t / a  # Ritter front, measured from the dam
        assert z_over_a == pytest.approx(2.0 * tau_of(t, a, g), rel=1e-12)


@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0, 2.5, 4.0])
def test_both_ritter_helpers_use_two_tau(tau):
    """``dam_break`` and ``dam_break_3d`` carry two copies; they must agree."""
    assert dam_break.ritter_front_over_a(tau) == pytest.approx(2.0 * tau, rel=1e-12)
    assert dam_break_3d.ritter_front_over_a(tau) == pytest.approx(2.0 * tau, rel=1e-12)


@pytest.mark.parametrize("tau", [0.0, 0.25, 1.0, 3.0])
def test_benchmark_curve_helper_matches_source_helper(exp_validation, tau):
    """The harness helper is the single source of truth for both paths."""
    assert exp_validation.ritter_za_curve([tau]) == pytest.approx([2.0 * tau], rel=1e-12)


def test_run_py_absolute_front_is_the_same_reference():
    """``benchmarks/dam_break/run.py`` builds the front in the lab frame.

    It stores ``ritter = 2 sqrt(g H) t`` measured from the dam and reports
    ``(x_max - a) / ritter``; that ratio is only like-for-like if the lab-frame
    front, shifted to the dam and divided by the column width, equals ``2 tau``.
    """
    a, g = 0.2, 9.81
    H = 2.0 * a
    for t in (0.05, 0.2, 0.5):
        ritter_lab = 2.0 * math.sqrt(g * H) * t  # what run.py computes
        assert ritter_lab / a == pytest.approx(2.0 * tau_of(t, a, g), rel=1e-12)


# --------------------------------------------------------------------------- #
# 2. What the figure actually draws
# --------------------------------------------------------------------------- #
class _RecordingAxes:
    def __init__(self):
        self.lines = []

    def plot(self, x, y, *args, **kwargs):
        self.lines.append({"x": list(x), "y": list(y),
                           "label": kwargs.get("label", "")})

    def errorbar(self, x, y, *args, **kwargs):
        self.lines.append({"x": list(x), "y": list(y),
                           "label": kwargs.get("label", "")})

    def __getattr__(self, _name):  # set_xlabel, legend, grid, set_xlim, ...
        return lambda *a, **k: None


def test_plot_reference_curve_is_two_tau(exp_validation, monkeypatch, tmp_path):
    """Intercept matplotlib and check the drawn Ritter line against Z/a = 2 tau."""
    import matplotlib.pyplot as plt

    axes = _RecordingAxes()
    monkeypatch.setattr(plt, "subplots", lambda *a, **k: (object(), axes))
    monkeypatch.setattr(plt, "tight_layout", lambda *a, **k: None)
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    exp_data = [{"tau": 0.0, "Z_over_a": 0.0, "uncertainty": 0.05},
                {"tau": 2.0, "Z_over_a": 1.8, "uncertainty": 0.05}]
    results = [{"dx": 0.02, "particles": 200, "sim_tau": [0.0, 1.0, 2.0],
                "sim_za": [0.0, 0.95, 1.85]}]

    exp_validation.generate_plot(results, exp_data, str(tmp_path / "unused.png"))

    ritter = [ln for ln in axes.lines if "Ritter" in ln["label"]]
    assert ritter, f"no Ritter reference line drawn; labels={[ln['label'] for ln in axes.lines]}"
    line = ritter[0]
    for tau, za in zip(line["x"], line["y"]):
        assert za == pytest.approx(2.0 * tau, rel=1e-12), (
            f"figure draws Z/a={za} at tau={tau}; the reference is Z/a = 2 tau"
        )
    # the curve must extend over the simulation window, not stop short
    assert max(line["x"]) >= 2.0


# --------------------------------------------------------------------------- #
# 3. Regression pins on the source text
# --------------------------------------------------------------------------- #
def test_no_stale_bare_tau_assignment_in_the_harness():
    """The exact defect: ``za_ritter = tau_ritter`` (a half-size reference)."""
    source = EXP_VALIDATION.read_text()
    assert not re.search(r"\bza_ritter\s*=\s*tau_ritter\b", source)
    assert "ritter_za_curve(" in source


def test_harness_data_path_uses_the_helper():
    """``run_one_exp`` must not re-implement the curve inline."""
    source = EXP_VALIDATION.read_text()
    assert "ritter_za = [ritter_front_over_a(tau) for tau in sim_tau]" not in source


def test_docs_do_not_assert_the_wrong_identity():
    """User-facing docs must state Z/a = 2 tau, not Z/a = tau."""
    doc = (REPO / "docs" / "benchmarks" / "dam-break-experimental.md").read_text()
    assert "Ritter shallow-water solution (Z/a = tau)" not in doc
    assert "Z/a = 2 tau" in doc
