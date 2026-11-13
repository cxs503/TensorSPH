"""Board audit m17046 — drift/correctness guards for REPRO-SPH-MONAGHAN1994.

Two jobs:

1. **Rederivation guard** — the graded numbers in
   ``benchmarks/repro_monaghan_1994/results/summary.json`` must be reproducible
   from the committed trajectories and reference tables by an *independent*
   implementation (this file's ``_metrics`` uses bisection lookup and does not
   import ``benchmarks/repro_monaghan_1994/run.py``).  If a trajectory or the
   summary is regenerated and they disagree, this goes red instead of the
   discrepancy entering a report.
2. **Claim guards for the m17020 error-budget increment** — that add-on lives on
   ``agent/m17020`` (commit c670396) and was *not* in the mainline at audit time;
   the guards below skip while the files are absent and fail loudly if the
   artefacts appear with a non-closing "error budget" or a Ritter comparison that
   runs past Ritter's validity limit (tau < sqrt 2), which is what the audit
   found (docs/board_audit/m17046/AUDIT_m17020.md, findings F1/F2).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "benchmarks" / "repro_monaghan_1994"
RES = CASE / "results"
DAM = ROOT / "benchmarks" / "dam_break"

WINDOW = (0.5, 3.0)          # pre-registered in run.py / summary.json:protocol
TOL_REL = 0.15               # platform tolerance: 15 % of the reference amplitude
TOL_ABS = 1e-9               # stored metrics must be rederivable to 9 decimals
RITTER_LIMIT = math.sqrt(2.0)


def _lerp(xs, ys, xq):
    """Clamped linear interpolation by bisection (independent of run.py:interp)."""
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    w = 0.0 if span == 0 else (xq - xs[lo]) / span
    return ys[lo] + w * (ys[hi] - ys[lo])


def _metrics(sim_tau, sim_za, ref_tau, ref_za, n=161):
    lo = max(WINDOW[0], sim_tau[0], ref_tau[0])
    hi = min(WINDOW[1], sim_tau[-1], ref_tau[-1])
    grid = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    sim = [_lerp(sim_tau, sim_za, t) for t in grid]
    ref = [_lerp(ref_tau, ref_za, t) for t in grid]
    d = [s - e for s, e in zip(sim, ref)]
    amp = max(ref)
    mae = sum(abs(x) for x in d) / len(d)
    return {"amplitude": amp, "mae": mae, "mae_over_amplitude": mae / amp,
            "bias": sum(d) / len(d), "max_abs": max(abs(x) for x in d),
            "n_samples": len(grid)}


def _tables():
    cross = json.loads((DAM / "martin_moyce_1952_crosscheck.json").read_text())
    bundled = json.loads((DAM / "martin_moyce_1952.json").read_text())
    cross_tau = [r["tau"] for r in cross["data"]]
    cross_za = [r["X_over_a"] - 1.0 for r in cross["data"]]   # X/a from wall: Z/a = X/a - 1
    rows = bundled.get("data") or bundled.get("points") or bundled
    b_tau = [r["tau"] for r in rows]
    b_za = [r.get("Z_over_a", r.get("X_over_a")) for r in rows]
    return (cross_tau, cross_za), (b_tau, b_za)


@pytest.fixture(scope="module")
def summary():
    path = RES / "summary.json"
    if not path.exists():
        pytest.skip("summary.json absent")
    return json.loads(path.read_text())


@pytest.mark.parametrize("case", ["dx0.04_c025", "dx0.02_c025", "dx0.01_c025"])
def test_summary_metrics_rederive_from_committed_trajectories(summary, case):
    """Every stored MAE/bias/max-abs must follow from the committed trajectory."""
    traj = json.loads((RES / f"trajectory_{case}.json").read_text())
    (c_tau, c_za), (b_tau, b_za) = _tables()
    for name, (r_tau, r_za) in (("crosscheck", (c_tau, c_za)), ("bundled", (b_tau, b_za))):
        got = _metrics(traj["tau"], traj["za"], r_tau, r_za)
        stored = summary["comparisons"][case][name]
        assert got["mae"] == pytest.approx(stored["mae"], abs=TOL_ABS), (
            f"{case}/{name}: stored mae={stored['mae']!r} but an independent "
            f"implementation gives {got['mae']!r} -- trajectory and summary drifted")
        assert got["mae_over_amplitude"] == pytest.approx(
            stored["mae_over_amplitude"], abs=1e-12)
        assert got["bias"] == pytest.approx(stored["bias"], abs=TOL_ABS)
        assert got["max_abs"] == pytest.approx(stored["max_abs"], abs=TOL_ABS)
        assert got["amplitude"] == pytest.approx(stored["amplitude"], rel=1e-12)


def test_graded_error_is_inside_the_platform_tolerance(summary):
    """The only graded reference must pass the pre-registered 15 % amplitude bar."""
    comp = summary["comparisons"]["dx0.01_c025"]["crosscheck"]
    assert comp["mae_over_amplitude"] <= TOL_REL, (
        f"graded MAE/amplitude={comp['mae_over_amplitude']:.4%} exceeds the "
        f"pre-registered {TOL_REL:.0%}")
    # ...and the number must be reported as-is, not rounded into the 3 % band:
    assert comp["mae_over_amplitude"] > 0.03, (
        "the finest-resolution error was recorded below 3 %; verify before "
        "changing this guard (it pins the honest 'not < 3 %' answer)")


def test_reference_pair_is_declared_inconsistent(summary):
    """A reference pair that disagrees by >2x its claimed uncertainty may not grade."""
    bracket = summary["bracket"]
    assert bracket["reference_spread_over_claimed"] > 2.0
    assert bracket["reference_self_consistent"] is False
    assert summary["gates"]["reference_self_consistency"]["result"] == "fail"


def test_bundled_transcription_is_never_a_pass_source(summary):
    assert summary["gates"]["trajectory_vs_bundled"]["result"] == "withheld"
    assert summary["verdict"]["withheld_reference_used_as_pass_source"] is False


# --- guards for the m17020 increment (absent from mainline at audit time) -----

def _error_budget():
    path = RES / "error_budget.json"
    if not path.exists():
        pytest.skip("results/error_budget.json absent (m17020 c670396 not landed)")
    return json.loads(path.read_text())


def test_error_budget_components_must_close_if_present():
    """If a file calls itself an error budget, its components must add up to the total.

    m17020's version does not: 31.86 % (reference spread) + 84.52 % (Ritter) +
    1.94 % (compressibility) = 118.3 % of amplitude against a total of 4.38 %
    (= 27x the total), and its printed residual of -27.48 % is attributed to
    discretization.  A scale mismatch of this size means the pieces are not
    terms of one budget; they must be relabelled as separate diagnostics.
    """
    budget = _error_budget()["error_budget"]
    total = budget["total_mae_over_amplitude"]
    comp = budget["components"]
    pieces = [comp["reference_uncertainty"]["spread_over_amplitude"],
              comp["ritter_deviation"]["mae_over_amplitude"],
              comp["compressibility"]["max_rho_over_rho0_minus_1"]]
    assert sum(pieces) == pytest.approx(total, rel=0.25), (
        f"components sum to {sum(pieces):.4f} ({sum(pieces)/total:.1f}x the total "
        f"{total:.4f}) -- this is not a decomposition; relabel as diagnostics")


def test_ritter_comparison_stays_inside_ritter_validity_if_present():
    """Ritter Z/a = 2*tau is a shallow-water result; beyond tau=sqrt2 it has no claim."""
    eb = _error_budget()
    window = eb["ritter_comparison"]["window"]
    assert window[1] <= RITTER_LIMIT + 1e-9, (
        f"Ritter comparison window reaches tau={window[1]:.3f} > sqrt(2)="
        f"{RITTER_LIMIT:.3f}: the deviation there measures shallow-water "
        f"breakdown, not SPH error")


def test_document_does_not_attribute_a_negative_residual_if_present():
    if not (RES / "error_budget.json").exists():
        pytest.skip("results/error_budget.json absent (m17020 c670396 not landed)")
    doc = ROOT / "docs" / "repro_monaghan_1994.md"
    if not doc.exists():
        pytest.skip("document absent")
    text = doc.read_text()
    if "误差分解" not in text:
        pytest.skip("error-budget section absent from the document")
    assert "residual (-" not in text and "residual(-" not in text, (
        "the document carries a negative 'residual' attributed to discretization "
        "physics; see docs/board_audit/m17046/AUDIT_m17020.md F1")