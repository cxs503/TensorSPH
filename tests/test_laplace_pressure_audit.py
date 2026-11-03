"""Regression guards for the Young-Laplace pressure audit.

The audit (``benchmarks/laplace_pressure_audit/run.py``) exists because the
historical harness scored

    p(r < 0.5R) - p(0.80R < r < 0.95R)   against   sigma / R

a difference between two *interior* points, whose continuum value is 0.  The
tests below pin the pieces a future edit could silently break:

* the two criteria stay distinguishable and the acceptance flag hangs on the
  correct one (interior pressure vs ``sigma / R`` with ``p_gas = 0``);
* the radial profile keeps exposing the mechanism (free-surface density deficit
  where the CSF force acts);
* the CSF force keeps its sign convention (inward), so ``int f_r dn > 0``.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_pressure_audit"


def _bench():
    spec = importlib.util.spec_from_file_location("laplace_pressure_audit_run",
                                                  BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _bench()


@pytest.fixture(scope="module")
def case(bench):
    """Half a capillary time on a coarse drop -- a few seconds."""
    return bench.simulate(radius=0.02, dx=0.003, sigma=7.2, rho0=1000.0,
                          sound_speed=20.0, viscosity=0.01,
                          t_end_over_t_sigma=0.5)


# --------------------------------------------------------------------------- #
# definitions
# --------------------------------------------------------------------------- #
def test_reference_values(case):
    assert case["reference"]["delta_p_pa"] == pytest.approx(360.0)   # sigma/R
    assert case["reference"]["gate_pct"] == pytest.approx(15.0)
    # Tait compression needed for the jump: (sigma/R) / (rho0 c0^2 / gamma).
    assert case["reference"]["compression_required"] == pytest.approx(6.3e-3,
                                                                     rel=0.02)


def test_interior_criterion_is_the_one_scored(case):
    """The acceptance flags must sit on the physical criterion."""
    interior = case["interior"]
    assert interior["theory_pa"] == pytest.approx(case["reference"]["delta_p_pa"])
    rel = (interior["measured_pa"] - interior["theory_pa"]) / interior["theory_pa"]
    assert interior["rel_error_pct"] == pytest.approx(100.0 * rel)
    assert interior["within_15pct"] == (abs(100.0 * rel) < 15.0)
    assert interior["within_3pct"] == (abs(100.0 * rel) < 3.0)


def test_historical_measure_has_zero_theoretical_value(case):
    """``p_core - p_shelf`` is an interior difference: theory says 0 Pa.

    If someone ever flips this back into a pass/fail test against ``sigma/R``,
    this assertion is what explains why that is wrong.
    """
    legacy = case["legacy_measure"]
    assert legacy["own_theory_pa"] == 0.0
    assert "p(r<0.5R)" in legacy["criterion"]
    # It is still reported against sigma/R so the historical -70..-100 % numbers
    # stay reproducible, but that field is explicitly not an acceptance flag.
    assert "rel_error_pct_vs_sigma_over_R" in legacy
    assert "within_15pct" not in legacy


def test_run_length_is_in_capillary_times(case):
    setup = case["setup"]
    t_sigma = math.sqrt(setup["rho0"] * setup["radius_m"] ** 3
                        / setup["sigma_N_per_m"])
    assert setup["t_end_s"] == pytest.approx(setup["steps"] * setup["dt_s"])
    assert setup["t_end_s"] / t_sigma == pytest.approx(0.5, rel=0.05)


# --------------------------------------------------------------------------- #
# mechanism
# --------------------------------------------------------------------------- #
def test_pressure_is_non_negative_and_clamped_at_the_surface(case):
    """``p = max(Tait, 0)``: the outermost shell of a single-phase drop has
    ``rho < rho0`` and therefore exactly zero pressure."""
    prof = case["profile"]
    assert all(b["p_pa"] >= 0.0 for b in prof)
    outer = [b for b in prof if b["r_over_R"][0] >= 0.95]
    assert outer, "profile lost its outermost bin"
    assert min(b["rho_over_rho0"] for b in outer) < 0.97
    assert max(b["p_pa"] for b in outer) == pytest.approx(0.0, abs=1.0)


def test_shell_carries_the_force_where_the_density_is_deficient(case):
    """The CSF force peaks in the shell that the free-surface deficit pins to
    zero pressure -- the root cause the audit reports."""
    prof = sorted(case["profile"], key=lambda row: row["r_over_R"][0])
    interior = [b for b in prof if b["r_over_R"][1] <= 0.6]
    shell = [b for b in prof if b["r_over_R"][0] >= 0.8]
    assert max(b["abs_a_sf"] for b in shell) > 5.0 * max(
        b["abs_a_sf"] for b in interior)
    assert min(b["rho_over_rho0"] for b in shell) < 0.97


def test_force_flux_has_the_right_sign(case):
    """The CSF force points inward, so the pressure jump balancing it,
    ``-int f_r dn``, must come out positive."""
    assert case["shell_flux"]["measured_pa"] > 0.0
    assert case["resolved_jump"]["p_outside_pa"] == 0.0
    assert case["resolved_jump"]["jump_pa"] > 0.0


def test_noise_budget_is_reported(case):
    noise = case["noise"]
    for key in ("temporal_p_std_pa", "eos_noise_pa_from_spatial_std",
                "noise_over_theory"):
        assert key in noise
    # The EOS pressure noise implied by the core density spread is a sizeable
    # fraction of sigma/R -- that is the whole point of the error budget.
    assert 0.0 < noise["eos_noise_pa_from_spatial_std"]