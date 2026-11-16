"""Invariants for the force-based (Newton) Couette wall-shear reading.

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12``.  The artifact is produced by
``benchmarks/couette/wall_force_reading.py``; these tests assert the two
bookkeeping identities the force reading needs, the sign convention, the
faithful reproduction of the banked secant records, and the predeclared gates
being applied -- they do not re-run the solver.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "benchmarks" / "couette" / "out_force" / \
    "wall_force_reading.json"
TAU_EXACT = 1000.0 * 0.02 * 0.2 / 0.1          # rho nu U / H = 40 Pa
#: One-spacing secant at the sliding wall, from the committed sweep
#: ``benchmarks/couette/out_design/viscous_time_convergence.json``
#: (r8-vt1.2 ratio_top = 0.72476, r16-vt1.2 ratio_top = 0.79133).
SWEEP_TOP_PCT = {(8, 1.2): 27.523791790008534, (16, 1.2): 20.867085456848045}
#: The registry records the same estimator at the longest admissible run.
REGISTRY_TOP_PCT = {8: 14.203286170959451, 16: 17.076468467712292}


@pytest.fixture(scope="module")
def art():
    if not ART.exists():
        pytest.skip(f"artifact missing: {ART}")
    return json.loads(ART.read_text())


def _arms(art, layout):
    return [a for a in art["arms"] if a["layout"] == layout]


def test_artifact_metadata(art):
    assert art["case"] == "plane_couette_channel"
    assert art["anchor"] == "XVAL-SPH-METHOD-SELECTION-2026-11-12"
    assert art["generated_by"].endswith("wall_force_reading.py")
    assert "viscous_times_by_arm" in art["setup"]
    assert {a["viscous_times"] for a in art["arms"]} <= {1.2, 3.6}
    for arm in art["arms"]:
        assert arm["tau_exact_pa"] == pytest.approx(TAU_EXACT, rel=1e-9)


def test_errors_are_rederived_from_the_force_reading(art):
    """Every stored percentage must follow from tau_force and tau_exact."""
    for arm in art["arms"]:
        tau = arm["tau_exact_pa"]
        assert arm["err_top_force_pct"] == pytest.approx(
            100.0 * (arm["tau_top_force_pa"] / tau - 1.0), rel=1e-9, abs=1e-9)
        assert arm["err_bottom_force_pct"] == pytest.approx(
            100.0 * (arm["tau_bottom_force_pa"] / tau - 1.0), rel=1e-9, abs=1e-9)


def test_sign_convention_is_physical(art):
    """The fluid drags the sliding wall backwards, so the reported stress
    magnitude must be positive on both walls."""
    for arm in art["arms"]:
        assert arm["tau_top_force_pa"] > 0.0, arm["layout"]
        assert arm["tau_bottom_force_pa"] > 0.0, arm["layout"]


def test_newtons_third_law_closes(art):
    """sum_all F_x + sum_fluid m a_x = 0  (the identity the legacy layout missed)."""
    gate = art["verdict"]["gates"]["third_law_residual_pct"]
    for arm in _arms(art, "periodic3"):
        assert abs(arm["third_law_residual_pct"]) <= gate, arm["resolution_H_over_dx"]


def test_periodic_copy_bookkeeping(art):
    """The x in [0,L) copy must carry one third of the load summed over all three.

    The a-priori gate is 1 %; the committed arms sit slightly above it
    (1.31-1.58 %), which the artifact must record honestly rather than round
    away, so this test bounds the diagnostic at 2 % and requires the flag to be
    the truth of the number it is derived from.
    """
    gate = art["verdict"]["gates"]["copy_invariance_pct"]
    for arm, verdict in zip(sorted(_arms(art, "periodic3"),
                                   key=lambda a: (a["resolution_H_over_dx"],
                                                  a["viscous_times"])),
                            art["verdict"]["periodic3"]):
        assert arm["copy_invariance_pct"] <= 2.0, arm["resolution_H_over_dx"]
        assert verdict["copy_invariance_ok"] == (
            abs(arm["copy_invariance_pct"]) <= gate)


def test_legacy_layout_reproduces_the_committed_sweep(art):
    """The legacy layout is the code path of the banked records: at the same time
    level its one-spacing secant must come back to the last digit, otherwise this
    audit reads a different quantity than the registry.  (The registry's own
    numbers -- 14.203 / 17.076 % -- are the same estimator at t_end = 3.6
    H^2/nu; that force arm is a submitted cluster job, not in this artifact.)"""
    for arm in _arms(art, "legacy"):
        key = (arm["resolution_H_over_dx"], arm["viscous_times"])
        assert key in SWEEP_TOP_PCT
        assert abs(arm["err_top_secant_pct"]) == pytest.approx(
            SWEEP_TOP_PCT[key], rel=1e-6, abs=1e-6)


def test_force_and_secant_read_the_same_run(art):
    """Both estimators must come from the same time average, otherwise the
    comparison is meaningless: identical steps/dt/profile length."""
    for arm in art["arms"]:
        assert arm["steps"] > 0 and arm["samples"] > 0
        assert len(arm["profile"]) >= 6
        assert arm["dt"] == pytest.approx(0.3 * 1.3 * (0.1 / arm["resolution_H_over_dx"])
                                          / 20.0, rel=1e-9)


def test_verdict_applies_the_predeclared_gates(art):
    v = art["verdict"]
    for arm in v["periodic3"]:
        res = arm["resolution_H_over_dx"]
        assert arm["owner_standard"] == (
            abs(arm["top_pct"]) <= 15.0 and abs(arm["bottom_pct"]) <= 15.0), res
        assert arm["bookkeeping_ok"] == (
            abs(arm["third_law_pct"]) <= v["gates"]["third_law_residual_pct"]
            and abs(arm["copy_invariance_pct"]) <= v["gates"]["copy_invariance_pct"])
    assert v["all_bookkeeping_ok"] == all(p["bookkeeping_ok"] for p in v["periodic3"])
    assert v["all_third_law_ok"] == all(p["third_law_ok"] for p in v["periodic3"])
    assert v["all_under_owner_standard"] == \
        all(p["owner_standard"] for p in v["periodic3"])


def test_the_decisive_rung_is_present(art):
    """The 15 % question needs the finer rung: the artifact must contain the
    H/dx = 16 force arm (its run is the one whose profile the banked 17 %
    secant record is taken on)."""
    main = _arms(art, "periodic3")
    assert 16 in {a["resolution_H_over_dx"] for a in main}
    fine = [a for a in main if a["resolution_H_over_dx"] == 16][0]
    assert fine["n_fluid"] > 0 and fine["samples"] > 0
    # the secant on that very run is what the registry charges against the scheme
    assert abs(fine["err_top_secant_pct"]) > 15.0
