"""Gates for the long-horizon well-balanced projection-SPH Young-Laplace drop.

The anchor ``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`` had one open branch after
``docs/xval_projection_laplace_horizon_instability.md``: the dynamic march
diverges with the adjoint projection pair at ``~18 t_sigma`` while the lattice is
healthy, i.e. the CSF traction is mis-projected.  ``run_balanced_horizon.py``
runs the same horizon with the traction's discrete-gradient part split off
(``relax_projection_balanced``), which used to be validated to 50 steps only.

These tests keep two things honest:

* the harness's own logic -- the pre-registered breakage rules and the gate
  arithmetic are pure functions, tested on synthetic samples so that a green
  summary cannot come from a classifier that never fires;
* the record -- every declared gate is asserted, and the *absence* of a record
  is a counted ``xfail`` rather than a silent skip (the repo rule: a skipped
  verification module reports green, and that is how a debt disappears).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_sph"
OUTDIR = BENCH / "results_balanced_horizon"
SUMMARY = OUTDIR / "summary.json"
sys.path.insert(0, str(BENCH))

import run_balanced_horizon as H  # noqa: E402


# --------------------------------------------------------------------------- #
# harness logic (no record needed)
# --------------------------------------------------------------------------- #
def _healthy(step=10):
    return {"step": step, "lattice": {
        "nn_min": 6.0, "nn_median": 18.0, "pair_min_over_dx": 0.99,
        "pair_p01_over_dx": 1.0, "color_min": 0.0, "color_max": 1.0,
        "kappa_R_max": 2.3, "gradc_dx_max": 0.48, "acc_over_aref_max": 14.9}}


def test_declared_jobs_are_unique_and_cover_the_control():
    names = H.job_names()
    assert len(names) == len(set(names))
    assert "dx8_h120_ctl_nu01" in names          # control arm must exist
    prim = next(j for j in H.JOBS if j["name"] == "dx8_h120_bal_nu01")
    ctl = next(j for j in H.JOBS if j["name"] == "dx8_h120_ctl_nu01")
    # the control differs from the primary in exactly one knob: the split
    assert prim["level"] == ctl["level"] and prim["nu_factor"] == ctl["nu_factor"]
    assert prim["balanced"] and not ctl["balanced"]


def test_breaks_rules_fire_on_synthetic_breakage():
    """Each pre-registered rule must fire, and a healthy sample must not."""
    assert H._breakages(_healthy()) == []
    for key, bad in (("pair_min_over_dx", 0.1), ("nn_min", 2.0),
                     ("kappa_R_max", 9.0), ("gradc_dx_max", 8.0),
                     ("color_min", -0.2), ("color_max", 1.3)):
        s = _healthy()
        s["lattice"][key] = bad
        assert H._breakages(s), key
    # a diverged sample whose entry state was already non-finite carries no
    # lattice block and must NOT be read as healthy
    assert H._breakages({"step": 3, "diverged": True}) == ["no_lattice_block"]


def test_gate_thresholds_match_the_pages_that_declare_them():
    assert H.ADAMI_GATE == 0.10                      # Adami et al. 2012
    assert H.LAPLACE_GATE_PCT == 15.0                # owner acceptance
    assert H.CLAIM_PCT == 3.0                        # claim level
    assert H.GATES_DECLARED["spurious_adami"]["value"] == H.ADAMI_GATE


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason="no results_balanced_horizon record yet")
def test_record_present():
    assert SUMMARY.exists(), (
        "run the four declared jobs of "
        "benchmarks/laplace_sph/run_balanced_horizon.py then --summarize")


@pytest.fixture(scope="module")
def summary():
    if not SUMMARY.exists():
        pytest.skip("no balanced-horizon record")
    return json.loads(SUMMARY.read_text())


def test_every_declared_job_is_in_the_record(summary):
    assert summary["jobs_declared"] == H.job_names()
    missing = [n for n in summary["jobs_declared"] if n not in summary["jobs_present"]]
    assert not missing, "declared jobs without a record: %s" % missing


def test_balanced_arm_does_not_diverge_over_the_horizon(summary):
    assert summary["verdicts"]["no_divergence_measured_horizon"], summary["arms"]
    assert summary["measured_horizon_steps"] == 120


def test_spurious_current_meets_the_adami_line(summary):
    assert summary["verdicts"]["spurious_adami"], summary["arms"]


def test_laplace_jump_within_owner_acceptance_at_every_sample(summary):
    assert summary["verdicts"]["laplace_all_samples_15pct"], summary["arms"]


def test_laplace_jump_within_claim_level(summary):
    assert summary["verdicts"]["laplace_all_samples_3pct"], summary["arms"]


def test_lattice_healthy_and_area_conserved(summary):
    assert summary["verdicts"]["lattice_healthy"], summary["arms"]
    assert summary["verdicts"]["area_drift_2pct"], summary["arms"]


def test_long_horizon_arms_are_declared_and_not_claimed(summary):
    """The 600-step arms are outstanding debt; the report must say so."""
    if not summary["verdicts"]["long_horizon_600_all_arms_present"]:
        assert summary["standing_debt"] == "long_horizon_600_all_arms_present"


def test_the_split_is_what_makes_the_difference(summary):
    """The control (no split, same knobs) must diverge while the primary does not."""
    assert summary["verdicts"]["split_is_the_cause"], summary["arms"]