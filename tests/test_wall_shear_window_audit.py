"""Audit tests for the channel wall-shear gate (mission m16654, anchor KPI-QC-RECHECK).

These tests exist because the merged gate (m16581, PR #573) quotes a *single*
drift window and, next to it, a precision (``-0.05 %`` on the ISPH ladder) that
is twenty times smaller than the same estimator's window sensitivity.  They
pin three things that were previously only prose:

1. the window convention of :mod:`tensorsph.wall_shear` (``fraction`` = window
   **length**) and its collision with the legacy
   ``benchmarks/poiseuille/wall_shear_estimators.py`` (``fraction`` = window
   **start**);
2. that the gate verdict uses the whole post-transient **band**, so a run whose
   residual transient happens to look steady over one window cannot pass;
3. that the published m16581 numbers reproduce from the committed artefacts,
   and that a whole-run window does *not* (which is why the window has to be
   post-transient at all).
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from tensorsph import wall_shear
from tensorsph.isph_channel import ChannelConfig, gates

REPO = Path(__file__).resolve().parents[1]
SENSITIVITY = REPO / "benchmarks" / "wall_shear_gate_audit" / "sensitivity.json"
LEGACY = REPO / "benchmarks" / "poiseuille" / "wall_shear_estimators.py"

H, G, RHO = 0.1, 9.81, 1000.0
TAU_REF = 0.5 * RHO * G * H


def _trace(u_of_t, n=201, t_end=1.0):
    return [{"t": t_end * i / (n - 1), "u_mean": float(u_of_t(t_end * i / (n - 1)))}
            for i in range(n)]


def _load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_wall_shear_estimators", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 1. window conventions
# --------------------------------------------------------------------------- #

def test_module_fraction_is_window_length():
    """``fraction`` = window length: window = [(1-f) t_end, t_end]."""
    trace = _trace(lambda t: t)
    d = wall_shear.drift_rate(trace, t_end=1.0, fraction=0.4)
    assert d.t_lo == pytest.approx(0.6, abs=0.01)
    assert d.t_hi == pytest.approx(1.0, abs=1e-9)


def test_legacy_script_fraction_is_window_start():
    """The legacy script's 0.4 is the *start* of the window, i.e. the last 60 %.

    This is the collision the module docstring warns about: the same number
    selects a different window in the two files.  It also pins the legacy
    comment/code disagreement (the comment says "last 40 %" while the code
    takes ``t >= fraction * t_end``) so a future edit cannot silently change
    the numbers stored under ``benchmarks/poiseuille/out``.
    """
    if not LEGACY.exists():  # pragma: no cover - repository layout guard
        pytest.skip("legacy estimator script not present")
    legacy = _load_legacy()
    trace = _trace(lambda t: 0.4 * t + 0.05 * t ** 2)
    out = legacy.drift_rate(trace, 1.0, fraction=0.4)
    # legacy: window length = (1 - fraction) * t_end, i.e. the last 60 %
    assert out["window_s"] == pytest.approx(0.6, abs=1e-9)
    module_same_window = wall_shear.drift_rate(trace, t_end=1.0, fraction=0.6)
    assert out["rate"] == pytest.approx(module_same_window.slope, rel=1e-12)
    module_same_number = wall_shear.drift_rate(trace, t_end=1.0, fraction=0.4)
    assert abs(out["rate"] - module_same_number.slope) > 1e-3, (
        "the two '0.4's must select different windows; if they agree the "
        "conventions have been unified and the artefacts need a re-read")


# --------------------------------------------------------------------------- #
# 2. the band is part of the verdict
# --------------------------------------------------------------------------- #

def _gates_for(trace, t_end=1.0):
    cfg = ChannelConfig()
    result = {
        "trace": trace,
        "t_reached": t_end,
        "u_max_measured": cfg.u_max_analytic,          # centre gate is not the subject
        "u_max_analytic": cfg.u_max_analytic,
        "density_error_max": 0.0,
        "divergence_reduction": {"pass": True},
    }
    fit = {"tau_wall_per_unit_rho": math.nan}
    return gates(result, fit, cfg)


def test_settled_trace_passes_both_windows():
    out = _gates_for(_trace(lambda t: 1.0 + 1e-4 * math.sin(20 * t)))
    gate = out["wall_shear_budget"]
    assert gate["available"] is True
    assert abs(gate["rel_error_pct"]) < 3.0
    assert gate["pass"] is True
    assert gate["rel_error_pct_band"][0] <= gate["rel_error_pct"] <= gate["rel_error_pct_band"][1]


def test_one_good_window_cannot_buy_a_pass():
    """A run still accelerating at ``t_end`` passes the declared window but not the band.

    ``u = a t**6`` is convex enough that the last 40 % of the trace is nearly
    flat while the last 20 % still has a slope: exactly the failure mode a
    single-window gate cannot see.
    """
    a = 0.0091 * G
    out = _gates_for(_trace(lambda t: a * t ** 6))
    gate = out["wall_shear_budget"]
    assert -3.0 < gate["rel_error_pct"] < 0.0, gate["rel_error_pct"]
    assert gate["window_sensitivity"]["worst_abs_rel_error_pct"] > 3.0
    assert gate["pass"] is False


def test_no_trace_means_no_pass():
    out = _gates_for([])
    gate = out["wall_shear_budget"]
    assert gate["available"] is False
    assert gate["pass"] is False


# --------------------------------------------------------------------------- #
# 3. reproducibility of the audited numbers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def report():
    if not SENSITIVITY.exists():  # pragma: no cover
        pytest.skip("run benchmarks/wall_shear_gate_audit/run.py first")
    return json.loads(SENSITIVITY.read_text())


def test_published_numbers_reproduce_from_committed_artefacts(report):
    rows = report["kpi_recheck"]["wcsph"] + report["kpi_recheck"]["isph"]
    assert rows
    for row in rows:
        assert row["published_matches_declared_window"], row["ladder"]


def test_every_post_transient_window_of_the_ladder_is_within_gate(report):
    assert report["kpi_recheck"]["all_bands_within_gate"] is True
    for group in ("wcsph", "isph"):
        for case in report[group]["cases"]:
            assert case["worst_abs_rel_error_pct"] <= 3.0, case["resolution_H_over_dx"]


def test_window_sensitivity_is_bigger_than_the_quoted_precision(report):
    """The audited claim is 0.05 pt; the estimator's own spread is comparable or larger.

    For the ISPH ladder this holds with a modest margin (0.077-0.086 pt), for
    the committed WCSPH artefacts the spread reaches 2.16 pt -- i.e. the
    WCSPH "confirmation" keeps only a ~2x margin to its own 3 % gate.
    """
    isph_spreads = [c["spread_pct_points"] for c in report["isph"]["cases"]]
    wcsph_spreads = [c["spread_pct_points"] for c in report["wcsph"]["cases"]]
    assert min(isph_spreads) > 0.05
    assert max(wcsph_spreads) > 1.0


def test_whole_run_window_fails_the_gate(report):
    """Including the spin-up breaks the 3 % gate on the same artefacts.

    Documents why the window must be post-transient and why "the closure is
    resolution independent" is a property of the window, not of the physics.
    """
    for group in ("wcsph", "isph"):
        for case in report[group]["cases"]:
            assert abs(case["full_run_rel_error_pct"]) > 3.0


def test_gate_reports_the_window_convention(report):
    assert report["definitions"]["fraction"].startswith("window length")
    conv = report["wcsph"]["convention"]
    assert conv["legacy_0.4_equals_module_0.6"] is True
    assert conv["same_number_different_window"] is True