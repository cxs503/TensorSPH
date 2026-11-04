"""What the CSF shell force actually integrates to, scored against Young-Laplace.

``tests/test_csf_consistent.py`` scores the three *ingredients* of the
consistent scheme separately (curvature, ``delta_s`` normalisation, force
direction).  None of those tests measures the quantity the pressure field sees:
the radial integral of the force across the interface, which by construction
equals the Laplace jump ``sigma*kappa = sigma/R``.  That integral is what this
file pins, together with an exact decomposition of its shortfall:

    A = f_kappa * f_delta * f_n

See ``benchmarks/csf_force_amplitude/run.py`` for the derivation and
``docs/verification_csf_force_amplitude.md`` for the numbers and the verdict.

The measured state (recorded, not assumed): ``A`` sits at 0.85-0.89 on
``R/dx = 5..16``, i.e. the shell force is 11-15 % short of the Laplace jump.
``f_delta`` (0.63-0.70) is the dominant deficit and ``f_kappa`` (+25..+42 %)
partly cancels it; the normals are coherent to 1e-3.  The deficit therefore
does *not* converge away quickly enough to reach the 15 % acceptance band, which
is why this file pins the decomposition rather than claiming the amplitude.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "csf_force_amplitude"
REPORT = BENCH / "report.json"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("csf_force_amplitude", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def coarse(bench):
    return bench.run_resolution(0.004)


@pytest.fixture(scope="module")
def medium(bench):
    return bench.run_resolution(0.002)


# --------------------------------------------------------------------------- #
# the decomposition must be an identity, not an approximation
# --------------------------------------------------------------------------- #
def test_product_identity_holds_to_round_off(coarse, medium):
    for row in (coarse, medium):
        assert abs(row["product_residual"]) < 1e-3, row["product_residual"]
        product = row["f_kappa"] * row["f_delta"] * row["f_normal"]
        # float32 fields: the residual is round-off (3e-5 relative at R/dx=10)
        assert product == pytest.approx(row["amplitude_A"], rel=5e-4)


def test_amplitude_is_normalised_to_the_analytic_jump(coarse, medium):
    """``A`` is dimensionless and normalised by ``sigma * kappa * L``.

    A wrong normalisation (a missing ``R`` or ``2*pi``) shows up immediately as
    an amplitude off by an order of magnitude rather than by tens of percent.
    """
    for row in (coarse, medium):
        assert 0.5 < row["amplitude_A"] < 1.5
        assert row["sigma_over_R_Pa"] == pytest.approx(7.2 / 0.02, rel=1e-12)
        assert row["interface_length_m"] == pytest.approx(2 * math.pi * 0.02, rel=1e-12)


# --------------------------------------------------------------------------- #
# the measured facts, pinned per resolution
# --------------------------------------------------------------------------- #
def test_force_amplitude_is_between_eleven_and_sixteen_percent_short(medium):
    """Measured 0.854 at ``R/dx = 10``; the band brackets the ladder 0.85-0.89."""
    assert 0.82 <= medium["amplitude_A"] <= 0.92, medium["amplitude_A"]
    assert 1.0 - medium["amplitude_A"] > 0.10, medium["amplitude_A"]


def test_delta_normalisation_is_the_dominant_deficit(medium):
    """``f_delta`` ~ 0.68 against a curvature bias of +25 %: the delta wins.

    This is the attribution the field reports did not have.  It also means the
    fix is in ``|grad c|`` (kernel truncation at the interface), not in the MLS
    curvature, and definitely not in the normal field.
    """
    assert medium["f_delta"] < 0.75, medium["f_delta"]
    assert medium["f_kappa"] > 1.15, medium["f_kappa"]
    assert abs(1.0 - medium["f_delta"]) > abs(1.0 - medium["f_kappa"])


def test_normals_are_radially_coherent(medium):
    """``f_n`` measures the radial alignment; truncation does not spoil it."""
    assert medium["f_normal"] > 0.99, medium["f_normal"]


def test_kernel_truncation_is_visible_in_the_colour_field(medium):
    """The mechanism: a shell particle's kernel sum is ~88 % of a bulk one."""
    assert 0.80 < medium["support_ratio"] < 0.95, medium["support_ratio"]
    assert medium["color_bulk_mean"] == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------- #
# two integral measures of the same field must agree
# --------------------------------------------------------------------------- #
def test_cut_force_and_radial_integral_agree(coarse, medium):
    """The x-projected line-tension identity and the radial integral agree.

    They are different projections of the same discrete force, so a disagreement
    would mean one of the two normalisations is wrong.  Measured |cut - A| =
    0.005 (coarse) / 0.004 (medium), i.e. both measures say the same thing --
    which also retires the stale "measured 1.08" note in
    ``tests/test_csf_consistent.py`` (see the verification doc).
    """
    for row in (coarse, medium):
        assert abs(row["cut_force_ratio"] - row["amplitude_A"]) < 0.05, row
        assert 0.80 < row["cut_force_ratio"] < 0.95, row["cut_force_ratio"]


# --------------------------------------------------------------------------- #
# the deficit is systematic, and anisotropy decays with resolution
# --------------------------------------------------------------------------- #
def test_amplitude_does_not_converge_to_one(coarse, medium):
    """11-15 % short at both resolutions: this is a systematic, not a transient.

    Pinned so a future fix cannot silently claim convergence, and so a
    regression that pushes the amplitude *up* through 1.0 is caught.
    """
    assert coarse["amplitude_A"] < 0.95
    assert medium["amplitude_A"] < 0.92


def test_angular_anisotropy_shrinks_under_refinement(coarse, medium):
    """Hex lattice anisotropy of the shell force: 1.06 -> 0.82 relative spread.

    The spread is what limits the *local* Laplace jump; it is the reason a
    single global amplitude number cannot be tighter than ~15 % at these
    resolutions.
    """
    assert coarse["A_theta_spread_rel"] > medium["A_theta_spread_rel"]
    assert medium["A_theta_spread_rel"] < 0.90
    # the mean over uniform angular bins returns the global amplitude
    assert medium["A_theta_mean"] == pytest.approx(medium["amplitude_A"], rel=0.10)


def test_committed_report_matches_a_fresh_run(medium):
    """The artifact in the repo is reproducible from the code."""
    if not REPORT.exists():
        pytest.skip("run benchmarks/csf_force_amplitude/run.py first")
    rows = json.loads(REPORT.read_text())["rows"]
    by_dx = {round(row["dx"], 6): row for row in rows}
    row = by_dx[0.002]
    assert row["amplitude_A"] == pytest.approx(medium["amplitude_A"], rel=1e-3)
    assert row["f_kappa"] == pytest.approx(medium["f_kappa"], rel=1e-3)
    assert row["f_delta"] == pytest.approx(medium["f_delta"], rel=1e-3)