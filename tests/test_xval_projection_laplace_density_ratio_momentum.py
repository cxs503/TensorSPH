"""Round 5 gates: the density ratio attributed to the phase it acts on.

Round 3-4 (`docs/xval_projection_laplace_density_ratio.md`) validated the
Young-Laplace jump at a real density ratio but reported the momentum-balance
residual as a median over **all** active particles.  That average is dominated by
the heavy phase, and the light phase is where a density ratio does its damage
(`a = f/rho`).  Round 5 ships `residual_phase` on `laplace_isph_case` and the
sweep in `scripts/xval_projection_laplace/density_ratio_momentum.py`.

This module holds the gates.  The one measurement that breaks its gate -- the
per-particle imbalance grows under refinement -- is a strict ``xfail``: the
defect is registered, and a fix makes this file fail loudly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.laplace_isph import laplace_isph_case  # noqa: E402

torch.set_num_threads(2)

DATA = (ROOT / "scripts" / "xval_projection_laplace" / "results"
        / "density_ratio_momentum.json")
GATES = json.loads(DATA.read_text())["summary"]["gates"] if DATA.exists() else {}
GROWTH_GATE_KEY = "G6_resolution_resid_growth_Rdx40_over_20"


@pytest.fixture(scope="module")
def payload():
    if not DATA.exists():
        pytest.skip("density_ratio_momentum.json missing: run "
                    "scripts/xval_projection_laplace/density_ratio_momentum.py")
    return json.loads(DATA.read_text())


def _row(payload, axis, ratio, arm, level=None):
    hits = [r for r in payload["rows"] if r["axis"] == axis
            and abs(r["ratio"] - ratio) < 1e-9 and r["arm"] == arm
            and (level is None or abs(r["R_over_dx"] - level) < 1e-9)]
    assert len(hits) == 1, f"no unique row for {axis}/{ratio}/{arm}/{level}"
    return hits[0]


@pytest.fixture(scope="module")
def small_case():
    """A cheap carrier re-solved live: the API contract of ``residual_phase``."""
    return laplace_isph_case(radius=0.02, dx=0.004, outer=0.012, rho0=1000.0,
                             rho_vapour=1.0, poisson_weighting="force_density")


# --------------------------------------------------------------------------- #
# the instrument itself
# --------------------------------------------------------------------------- #
def test_case_reports_residual_per_phase(small_case):
    """`residual_phase` is part of the case contract, both phases populated."""
    ph = small_case["residual_phase"]
    assert set(ph) == {"liquid", "vapour"}
    for phase in ("liquid", "vapour"):
        assert ph[phase]["n"] > 0
        assert ph[phase]["median_over_local_csf_acc"] > 0.0
        assert ph[phase]["mean_abs_m_per_s2"] >= 0.0
    # the phases partition the active set, and the liquid side cannot exceed
    # the total number of liquid particles in the block
    assert ph["liquid"]["n"] <= small_case["n_liquid"]
    assert ph["liquid"]["n"] + ph["vapour"]["n"] < small_case["n_particles"]


def test_global_median_is_phase_blind(payload):
    """The round-3/4 global median hides the heavy phase (why this round exists).

    For the variable-coefficient arm at ratio 1000 the global median sits between
    the two phase medians and is at least the declared margin below the heavy
    phase: averaging over particles is not a neutral operation here.
    """
    row = _row(payload, "density_ratio", 1000, "density")
    lo = min(row["resid_liquid_median"], row["resid_vapour_median"])
    hi = max(row["resid_liquid_median"], row["resid_vapour_median"])
    assert lo < row["resid_global_median"] < hi
    hiding = ((row["resid_liquid_median"] - row["resid_global_median"])
              / row["resid_global_median"] * 100.0)
    assert hiding >= GATES["phase_blind_hiding_min_pct"]


# --------------------------------------------------------------------------- #
# the physics claim: the ratio is a scale factor, not a defect
# --------------------------------------------------------------------------- #
def test_force_density_residual_is_ratio_blind():
    """`-grad^2 p = -div(f)` must reproduce itself from ratio 1 to 1000.

    Nothing in the operator or the source carries `rho`, so the *dimensionless*
    per-particle imbalance has to be invariant; the case-level check is done
    live, not read from the JSON.
    """
    out = {}
    for ratio in (1.0, 1000.0):
        out[ratio] = laplace_isph_case(radius=0.02, dx=0.002, outer=0.012,
                                       rho0=1000.0, rho_vapour=1000.0 / ratio,
                                       poisson_weighting="force_density")
    a, b = out[1.0], out[1000.0]
    assert b["dp_measured_Pa"] == pytest.approx(a["dp_measured_Pa"], rel=1e-12)
    for phase in ("liquid", "vapour"):
        assert (b["residual_phase"][phase]["median_over_local_csf_acc"]
                == pytest.approx(
                    a["residual_phase"][phase]["median_over_local_csf_acc"],
                    rel=1e-12))


def test_vapour_residual_scales_exactly_as_one_over_rho(payload):
    """The light-phase imbalance grows as `1/rho_v` — a scale, not an error.

    Both terms of the residual (`a = f/rho` and `grad p/rho`) carry the same
    factor `1/rho_v = ratio/rho_l`, and this arm's pressure is ratio blind, so
    `mean|resid|_vapour` must equal `value(ratio 1) * ratio` to round-off: a
    heavier ratio means a lighter vapour and a *linearly* larger imbalance.
    """
    rows = {r["ratio"]: r for r in payload["rows"]
            if r["axis"] == "density_ratio" and r["arm"] == "force_density"}
    base = rows[1.0]["resid_vapour_abs_m_per_s2"]
    for ratio, row in rows.items():
        assert row["resid_vapour_abs_m_per_s2"] == pytest.approx(
            base * ratio, rel=1e-9), f"ratio {ratio} breaks the 1/rho_v law"
    assert (payload["summary"]["verdicts"]["G3_vapour_residual_one_over_rho_scaling"]["pass"])


def test_force_density_phase_residual_ratio_spread(payload):
    """Declared gate G2: the invariant is a *spread*, so a knob-dependent
    improvement cannot pass it."""
    v = payload["summary"]["verdicts"]["G2_force_density_phase_resid_ratio_spread"]
    assert v["measured"] <= GATES["force_density_phase_resid_ratio_spread_max"]
    assert v["pass"]


def test_jump_regression_round4(payload):
    """Round-4 numbers are untouched by this round's instrument (regression)."""
    row = _row(payload, "density_ratio", 1000, "force_density")
    assert row["dp_err_pct"] == pytest.approx(-0.2133, abs=0.01)
    assert abs(_row(payload, "density_ratio", 1, "force_density")["dp_err_pct"]
               ) == pytest.approx(0.2133, abs=0.01)


# --------------------------------------------------------------------------- #
# the ablation: the naive operator damages the *light* phase
# --------------------------------------------------------------------------- #
def test_naive_operator_blows_up_in_the_light_phase(payload):
    """`-grad^2 p = -rho_l div(a)` over-predicts the jump and wrecks the vapour."""
    naive = _row(payload, "density_ratio", 1000, "uniform")
    ok = _row(payload, "density_ratio", 1000, "force_density")
    assert abs(naive["dp_err_pct"]) >= GATES["uniform_dp_err_min_pct_at_ratio_1000"]
    blowup = naive["resid_vapour_median"] / max(ok["resid_vapour_median"], 1e-300)
    assert blowup >= GATES["uniform_vapour_resid_min_ratio_to_validated"]
    # ... and it is the light phase that is hit hardest, at any ratio
    assert naive["resid_vapour_median"] > naive["resid_liquid_median"]


def test_equal_density_arms_collapse(payload):
    """At ratio 1 all three arms are the same operator: spread exactly 0."""
    ctrl = payload["controls"]["equal_density_arm_spread"]
    assert ctrl["dp_pa"] == 0.0
    assert ctrl["resid_liquid_median"] <= 1e-12


def test_sigma_zero_control(payload):
    """No traction => no pressure => no residual: the instrument has no offset."""
    z = payload["controls"]["sigma_zero"]
    assert abs(z["dp_measured_Pa"]) <= GATES["sigma_zero_max_pa"]
    assert max(z["resid_liquid_abs_m_per_s2"], z["resid_vapour_abs_m_per_s2"]) \
        <= GATES["sigma_zero_resid_max"]


# --------------------------------------------------------------------------- #
# the registered debt
# --------------------------------------------------------------------------- #
def test_only_the_registered_gate_fails(payload):
    """Every gate except the declared refinement debt has to pass.

    If the refinement debt is ever fixed, G6 starts passing and this test fails
    on purpose -- the xfail below and the doc have to be updated together.
    """
    failing = set(payload["summary"]["headline"]["all_pass_except"])
    assert failing == {GROWTH_GATE_KEY}, failing
    assert payload["summary"]["verdicts"][GROWTH_GATE_KEY]["pass"] is False


@pytest.mark.xfail(strict=True, reason=(
    "round-5 debt: the per-particle momentum imbalance grows under refinement "
    "(0.405 -> 0.657 at R/dx = 20 -> 40, ratio 1000, rho-blind arm) even though "
    "the integral jump converges; see docs/xval_projection_laplace_density_ratio_momentum.md"))
def test_per_particle_balance_does_not_degrade_under_refinement(payload):
    v = payload["summary"]["verdicts"][GROWTH_GATE_KEY]
    assert v["measured"] <= GATES["resolution_resid_growth_max"]