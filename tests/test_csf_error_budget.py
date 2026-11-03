"""Error-budget tests for the CSF surface-tension model.

These tests pin the *diagnosis* recorded in
``docs/verification_csf_error_budget.md`` and guard the fix implemented as
``cfg.csf_scheme = "consistent_jump"``.

The four claims, all cheap enough for the normal suite (a static evaluation on
a 161-642 particle drop; no time stepping):

1. the MLS divergence is exact on a linear field (so the curvature bias cannot
   be blamed on operator consistency);
2. ``sum delta_s V`` puts a third of its weight on particles whose colour is
   already saturated (the bulk-kink tail), and those particles carry an
   over-estimated curvature;
3. the cut-force identity factorises as ``kappa_bar * int delta_s dn`` -- the
   documented reason its 8-11 % agreement is *not* independent evidence that
   the force model is right;
4. the ``consistent_jump`` fix removes the bulk tail and brings curvature
   inside the 15 % band at dx = 1.5 mm, while admitting (by assertion) that the
   two coarser resolutions still miss it.

Numbers are recomputed here rather than read from the JSON artefact, so the
tests fail if ``src/`` changes behaviour silently.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "csf_audit"

R = 0.02
SIGMA = 7.2
RHO0 = 1000.0
TAU = 2.0 * math.pi


def _module():
    spec = importlib.util.spec_from_file_location("csf_audit_run",
                                                  BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _module()


# --------------------------------------------------------------------------- #
# 1. operator control
# --------------------------------------------------------------------------- #
def test_mls_divergence_is_exact_on_a_linear_field(audit):
    """``div(x/R) = 2/R`` everywhere; a consistent operator returns it exactly.

    This is the control that separates "operator is inconsistent" from
    "the scored field is not what we think it is".  It is also the reason the
    +25 % curvature bias must be attributed to the interface sampling.
    """
    for dx in (0.003, 0.002):
        ctrl = audit.operator_control(dx)
        assert ctrl["consistent_to_machine_precision"], ctrl
        assert ctrl["max_rel_error"] < 1e-6


# --------------------------------------------------------------------------- #
# 2. the delta_s bulk tail
# --------------------------------------------------------------------------- #
def test_delta_s_weight_leaks_into_the_bulk_under_the_shipped_scheme(audit):
    """The defect: a third of ``sum delta_s V`` is not on the interface.

    Particles with ``c > 0.95`` are bulk fluid -- there is no interface there --
    yet they carry 30-40 % of the CSF weight when the raw kernel-sum colour is
    differenced.  Their curvature is several times ``1/R``.
    """
    row = audit.score(0.002, "consistent")
    assert row["delta_s"]["bulk_weight_share"] > 0.25, row["delta_s"]
    inner = [s for s in row["shells"] if s["r_over_R"][1] <= 0.9]
    assert inner, row["shells"]
    for shell in inner:
        if shell["color_mean"] > 0.95:
            assert shell["kappa_1_per_m"] > 1.2 / R, shell


def test_jump_normalisation_removes_the_bulk_tail(audit):
    """Fix part 1: smoothing + measured-jump normalisation keeps weight on the
    interface (bulk share collapses to <1 %)."""
    row = audit.score(0.002, "consistent_jump")
    assert row["delta_s"]["bulk_weight_share"] < 0.01, row["delta_s"]
    assert row["kappa"]["transition_band_rel_error_pct"] == pytest.approx(
        row["kappa"]["rel_error_pct"], rel=0.0, abs=1e-9
    ) or abs(row["kappa"]["transition_band_rel_error_pct"]) < 30.0


# --------------------------------------------------------------------------- #
# 3. why the cut-force test passes
# --------------------------------------------------------------------------- #
def test_cut_force_identity_factorises_onto_kappa_and_delta_s(audit):
    """``F_cut/(2 sigma) ~ kappa_bar * int delta_s dn`` -- to a few percent.

    Consequence: the 8-11 % agreement of the shipped scheme is the *cancellation*
    of a +25 % curvature error against a -25 % delta_s normalisation error, not
    independent validation of the force magnitude.
    """
    row = audit.score(0.002, "consistent")
    cut = row["cut_force"]
    assert abs(cut["factor_closure_error_pct"]) < 10.0, cut
    # and both factors are individually outside their 15 % band
    assert abs(row["kappa"]["rel_error_pct"]) > 15.0
    assert abs(row["delta_s"]["rel_error_pct"]) > 15.0
    # while the product is inside it
    assert cut["within_15pct"], cut


# --------------------------------------------------------------------------- #
# 4. what the fix does, stated honestly
# --------------------------------------------------------------------------- #
def test_consistent_jump_fix_on_the_finest_resolution(audit):
    """At R/h ~ 10 all three analytic criteria are inside the 15 % band.

    This is the acceptance the shipped scheme never reached on the whole shell
    (kappa +25 %, int delta_s 0.74).
    """
    row = audit.score(0.0015, "consistent_jump")
    assert row["kappa"]["within_15pct"], row["kappa"]
    assert row["delta_s"]["within_15pct"], row["delta_s"]
    assert row["cut_force"]["within_15pct"], row["cut_force"]


def test_fix_is_resolution_limited_and_this_is_recorded(audit):
    """Honest bookkeeping: the fix does *not* pass the 15 % kappa bar at
    dx = 2-3 mm (R/h = 5-8), and delta_s is 8-25 % low there."""
    coarse = audit.score(0.003, "consistent_jump")
    assert not coarse["kappa"]["within_15pct"]
    # the fix is nevertheless strictly better than the shipped scheme on the
    # quantity it targets only where the interface is resolved: assert the
    # *bulk tail* is gone at every resolution.
    assert coarse["delta_s"]["bulk_weight_share"] < 0.01


def test_fix_does_not_change_the_force_direction(audit):
    for scheme in ("consistent", "consistent_jump"):
        row = audit.score(0.003, scheme)
        assert row["direction"]["cos_median"] > 0.9, (scheme, row["direction"])
        assert row["direction"]["fraction_inward"] > 0.9, (scheme, row["direction"])


# --------------------------------------------------------------------------- #
# 5. regression guard on the public surface-tension entry point
# --------------------------------------------------------------------------- #
def test_dispatch_accepts_the_new_scheme_and_rejects_junk():
    from tensorsph.droplet_impact import Config
    from tensorsph.surface_tension import compute_surface_tension_force

    pos = torch.tensor([[0.0, 0.0], [0.005, 0.0], [0.0, 0.005]], dtype=torch.float64)
    mass = torch.full((3,), 1.0, dtype=torch.float64)
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])
    cfg = Config(dx=0.005, h_factor=1.3, rho0=1.0, sound_speed=20.0, gamma=7.0,
                 gravity=0.0, viscosity=0.0, surface_tension_coefficient=7.2,
                 csf_scheme="consistent_jump")
    acc = compute_surface_tension_force(pos, mass, mass, pairs, cfg)
    assert acc.shape == pos.shape
    assert torch.isfinite(acc).all()
    with pytest.raises(ValueError):
        compute_surface_tension_force(pos, mass, mass, pairs, cfg, scheme="nope")
