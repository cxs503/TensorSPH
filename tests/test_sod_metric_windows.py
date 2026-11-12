"""Metric-integrity tests for CASE-SPH-SOD-SHOCK-COMPRESSIBLE (TensorSPH).

These tests guard the *measurement*, not the scheme.  They exist because the
case was accepted (m16184) with a plateau metric whose windows were anchored on
a fixed fraction of the plateau width: at coarse resolution that fraction is
thinner than the SPH smearing width, so the "error" being gated was partly the
error of the measuring window.  Findings and their quantification:
``docs/audit/audit_verdict_compressible_sod.md`` and
``docs/audit/compressible_sod_metric_audit.md``.

Each test asserts an invariant that must survive future tuning:

* the plateau trim is at least ``WINDOW_TRIM_H * h`` (resolution-aware windows);
* the production point (N=800) is *unchanged* by the fix (the accepted numbers
  must not drift);
* the coarse-lattice point is no longer dominated by window contamination;
* the analytic profile used as the yardstick agrees with an independent
  finite-volume HLLC solution.
"""
import numpy as np
import pytest

from tensorsph.compressible import sod as S
from tensorsph.compressible import fv_reference as FV


# ---------------------------------------------------------------------------
# window invariants (geometry only: no simulation needed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [200, 400, 800, 1600])
def test_plateau_trim_is_at_least_three_h(n):
    """Every wave-adjacent window edge must sit >= 3h away from the wave."""
    cfg = S.SodSPHConfig(n_particles=n)
    pw = S.plateau_windows(cfg, 0.2)
    trim_h = S.WINDOW_TRIM_H * cfg.h
    for edge in ("left", "star_left", "star_right", "right"):
        assert pw["trim"][edge] >= trim_h - 1e-15, (n, edge, pw["trim"])
    # windows must be non-empty and correctly ordered
    w = pw["windows"]
    assert w["left_plateau"][1] < w["star_left"][0] < w["star_left"][1]
    assert w["star_left"][1] < w["star_right"][0] < w["star_right"][1]
    assert w["star_right"][1] < w["right_plateau"][0]


def test_window_trim_uses_fraction_when_it_is_larger():
    """At N=800 the 10%-of-plateau rule binds, so the production windows (and
    therefore the accepted numbers) must be bit-identical to the old ones."""
    cfg = S.SodSPHConfig(n_particles=800)
    pw = S.plateau_windows(cfg, 0.2)
    waves = pw["waves"]
    f = S.WAVE_WINDOW_FRACTION
    assert pw["trim"]["star_left"] == pytest.approx(
        f * (waves["x_contact"] - waves["x_rarefaction_tail"]))
    assert pw["trim"]["star_right"] == pytest.approx(
        f * (waves["x_shock"] - waves["x_contact"]))
    assert pw["trim"]["left"] == pytest.approx(
        f * (waves["x_rarefaction_head"] - cfg.x_left))
    assert pw["trim"]["right"] == pytest.approx(0.02)


def test_empty_window_is_an_error_not_a_nan():
    """A degenerate configuration must raise, not silently return NaN metrics."""
    cfg = S.SodSPHConfig(n_particles=8)   # dx so large that windows collapse
    with pytest.raises(ValueError):
        S.plateau_windows(cfg, 0.2)


# ---------------------------------------------------------------------------
# the fix must not move the accepted production numbers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prod_metrics():
    cfg = S.SodSPHConfig(n_particles=800)
    res = S.run_sod_sph(cfg, output_times=[0.2])
    return cfg, res, S.evaluate_metrics(res["snapshots"][-1], cfg)


def test_production_numbers_unchanged_by_window_fix(prod_metrics):
    """Locks the values published in REPORT.md §④ (N=800, h=2.5dx)."""
    cfg, res, met = prod_metrics
    assert met["max_abs_rel_err"] == pytest.approx(0.0078, abs=2e-3)
    assert met["l1"]["rho"] == pytest.approx(0.0114, abs=2e-3)
    assert met["rel_err"]["x_shock"] == pytest.approx(-0.0002, abs=2e-3)
    assert met["rel_err"]["x_contact"] == pytest.approx(-0.0010, abs=2e-3)


def test_coarse_lattice_error_not_window_dominated():
    """N=200: the shipped fixed-fraction window reported 4.03% (see audit doc);
    with h-scaled windows the gated error must be well below 1%."""
    cfg = S.SodSPHConfig(n_particles=200)
    res = S.run_sod_sph(cfg, output_times=[0.2])
    met = S.evaluate_metrics(res["snapshots"][-1], cfg)
    assert met["max_abs_rel_err"] < 0.01, met["rel_err"]
    # ... and the old rule demonstrably inflated it (reproduce the finding)
    old = S.WINDOW_TRIM_H
    S.WINDOW_TRIM_H = 0.0
    try:
        legacy = S.evaluate_metrics(res["snapshots"][-1], cfg)
    finally:
        S.WINDOW_TRIM_H = old
    assert legacy["max_abs_rel_err"] > 5.0 * met["max_abs_rel_err"]


def test_residual_plateau_offset_does_not_vanish_with_refinement():
    """Documented limitation, asserted so it cannot silently 'converge'.

    The star-left pressure error is non-monotone and stays ~0.3-0.8% from
    N=200 to N=1600 (measured), i.e. it is a systematic model offset of the
    isentropically smeared diaphragm, not discretisation error.  If a future
    change makes it converge to zero, this test fails on purpose and the claim
    in REPORT.md ~5 must be re-derived.
    """
    errs = {}
    for n in (400, 800):
        cfg = S.SodSPHConfig(n_particles=n)
        res = S.run_sod_sph(cfg, output_times=[0.2])
        met = S.evaluate_metrics(res["snapshots"][-1], cfg)
        errs[n] = abs(met["rel_err"]["p_star_L"])
    assert all(1e-3 < e < 2e-2 for e in errs.values()), errs


# ---------------------------------------------------------------------------
# the yardstick itself: analytic profile vs independent FV
# ---------------------------------------------------------------------------
def test_exact_profile_matches_independent_finite_volume():
    """exact_riemann_profile() is never compared with anything but itself in the
    case tests.  A first-order HLLC FV solve shares no code with it; at 1000
    cells the two must agree to ~1% in L1 (residual is the FV contact smearing).
    """
    fv = FV.run_hllc(n_cells=1000, t_end=0.2, cfl=0.4)
    err = FV.compare_with_exact(fv)
    assert err["rho"] < 0.02, err
    assert err["p"] < 0.02, err
    assert err["u"] < 0.03, err
    # sanity: the FV run really reached t=0.2 and conserved the star state
    assert fv["t"] == pytest.approx(0.2, abs=1e-9)
    x = np.asarray(fv["x"])
    star = (x > 0.60) & (x < 0.66)          # star region, away from both jumps
    assert np.mean(np.asarray(fv["p"])[star]) == pytest.approx(0.30313, abs=3e-3)
    assert np.mean(np.asarray(fv["rho"])[star]) == pytest.approx(0.42632, abs=5e-3)


def test_hllc_solution_is_entropy_admissible():
    """Cross-check of the FV reference: pressure and density must be positive and
    the rarefaction fan monotone (no expansion shock)."""
    fv = FV.run_hllc(n_cells=800, t_end=0.1, cfl=0.4)
    rho = np.asarray(fv["rho"])
    p = np.asarray(fv["p"])
    x = np.asarray(fv["x"])
    assert np.all(rho > 0) and np.all(p > 0)
    fan = (x > 0.32) & (x < 0.42)           # inside the left rarefaction at t=0.1
    order = np.argsort(x[fan])
    dp = np.diff(p[fan][order])
    assert np.all(dp <= 1e-9), "rarefaction fan must not increase pressure"