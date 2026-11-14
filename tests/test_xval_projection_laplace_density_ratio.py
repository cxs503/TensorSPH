"""Gates for round 3 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE (density ratio).

Round 1 measured ``dp = sigma/R`` at one set-up, round 2 showed the number is not
an artefact of the set-up -- both on a **1:1 density carrier**, i.e. with the
``1/rho`` coefficient of the pressure Poisson switched off by construction.
Round 3 turns the density ratio on (``rho_liquid/rho_vapour`` up to 1000) and
asks two questions:

1. does the discrete Young-Laplace jump survive the density ratio (the physics
   says it must -- ``sigma/R`` contains neither ``rho`` nor the ratio);
2. which Poisson operator carries it.

The gates were declared in ``scripts/xval_projection_laplace/density_ratio.py``
(``GATES``) *before* the sweep ran and are re-stated here.  The one gate the
measurement broke -- the 5 % band for ``R/dx >= 20`` at high contrast -- is
recorded as a strict ``xfail``, not relaxed: the defect is registered, and the
day it is fixed this test starts failing loudly.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.laplace_isph import hex_block, laplace_isph_case  # noqa: E402

torch.set_num_threads(2)

RESULTS = ROOT / "scripts" / "xval_projection_laplace" / "results"
DATA = RESULTS / "density_ratio.json"
GATES = json.loads(DATA.read_text())["summary"]["gates"] if DATA.exists() else {}


@pytest.fixture(scope="module")
def payload():
    if not DATA.exists():
        pytest.skip("density_ratio.json missing: run "
                    "scripts/xval_projection_laplace/density_ratio.py")
    return json.loads(DATA.read_text())


# --------------------------------------------------------------------------- #
# the control: nothing moves when the ratio is 1
# --------------------------------------------------------------------------- #
def test_equal_density_path_bit_identical():
    """``rho_vapour=None`` and ``rho_vapour=rho0`` must give the same answer.

    This is the regression guard for the whole round: the density-ratio plumbing
    must be inert on the historical carrier, otherwise every published round-1/2
    number would have been silently re-measured.
    """
    base = laplace_isph_case(radius=0.02, dx=0.001, outer=0.012)
    same = laplace_isph_case(radius=0.02, dx=0.001, outer=0.012,
                             rho_vapour=1000.0)
    assert base["dp_measured_Pa"] == pytest.approx(same["dp_measured_Pa"],
                                                   rel=1e-12, abs=0.0)
    # round-1 baseline: -0.21 % at R/dx = 20 (docs/xval_projection_laplace_oss.md)
    assert (base["dp_over_theory"] - 1.0) * 100.0 == pytest.approx(-0.2133, abs=0.01)


def test_volume_per_phase_identical(payload):
    """The lattice, the cell volume and hence the colour function are unchanged.

    Without this the sweep would confuse "density ratio" with "interface moved":
    the traction shell and the readout bands have to sit on the same geometry in
    every row.
    """
    ctrl = payload["controls"]
    assert ctrl["vol_phase_spread"] <= GATES["vol_phase_spread_max"]
    assert ctrl["mass_per_phase"]["liquid"] / ctrl["mass_per_phase"]["vapour"] == \
        pytest.approx(1000.0, rel=1e-9)


def test_sigma_zero_control(payload):
    """sigma = 0 at ratio 1000 must return dp = 0 (no jump without tension)."""
    ctrl = payload["controls"]
    assert abs(ctrl["sigma_zero"]["dp_measured_Pa"]) < GATES["sigma_zero_max_pa"]
    assert abs(ctrl["sigma_zero"]["dp_flux_Pa"]) < GATES["sigma_zero_max_pa"]


# --------------------------------------------------------------------------- #
# the claim: the jump is density-ratio blind
# --------------------------------------------------------------------------- #
def test_density_arm_inside_three_percent(payload):
    """Every ratio up to 1000, R/dx = 20: the validated arm stays inside 3 %."""
    by_ratio = payload["summary"]["density_arm_err_pct_by_ratio"]
    assert len(by_ratio) == 8
    for ratio, err in by_ratio.items():
        assert abs(err) < GATES["dp_err_max_pct_density_arm_R_over_dx_le_25"], \
            f"ratio {ratio}: {err:+.3f} %"
    assert abs(payload["summary"]["err_pct_ratio_1000_R_over_dx_20"]) < \
        GATES["dp_err_max_pct_ratio_1000_R_over_dx_20"]


def test_ratio_spread(payload):
    """The law is ratio-blind, so the spread over the ladder must collapse too."""
    assert payload["summary"]["ratio_spread_pct"] < GATES["ratio_spread_pct"]


def test_traction_flux_instrument_is_ratio_blind(payload):
    """The independent instrument must not see the density at all.

    ``-\\int rho a_r dA / 2 pi R`` integrates the CSF force density: with
    ``a = f/rho`` and ``m = rho V`` the density cancels *exactly*, so the spread
    over the ratio ladder is the test that the implementation is consistent
    (a leak would show up here before it showed up in the pressure).
    """
    assert payload["summary"]["flux_spread_pct"] < GATES["flux_spread_pct"]
    assert payload["summary"]["flux_err_pct_max_abs"] < GATES["flux_spread_pct"]


def test_translation_invariance_at_ratio_1000(payload):
    rows = payload["translation"]
    assert len(rows) == 3
    spread = max(r["err_pct"] for r in rows) - min(r["err_pct"] for r in rows)
    assert spread < GATES["translation_spread_pct"]


# --------------------------------------------------------------------------- #
# the ablation: the naive operator must fail, loudly
# --------------------------------------------------------------------------- #
def test_naive_constant_coefficient_operator_fails(payload):
    """A Poisson without ``1/rho`` is wrong by orders of magnitude at ratio > 1.

    ``-grad^2 p = -rho_l div(a)`` reacts to the *vapour-side* divergence (which
    carries ``1/rho_v``) with a liquid-side stiffness: at ratio 1000 the error
    is ~4.7e4 %.  This is the reason the library gained the density-weighted
    operator instead of a "source correction".
    """
    naive = {float(k): v
             for k, v in payload["summary"]["naive_err_pct_by_ratio"].items()}
    assert abs(naive[1000.0]) > GATES["naive_err_min_pct_at_ratio_1000"]
    # monotone in the ratio: the defect is driven by the contrast, not by luck
    ladder = [abs(naive[r]) for r in (2, 5, 10, 30, 100, 300, 1000)]
    assert ladder == sorted(ladder)


def test_diffuse_density_hypothesis_is_falsified(payload):
    """Carrying ``rho`` on the smoothed colour field makes it *worse*.

    Registered as a negative result: the bias of the sharp-density carrier is not
    caused by the density step sitting inside the traction shell, because
    removing that step (diffuse interface) triples it.
    """
    diff = payload["diffuse_density"]
    sharp = {r["R_over_dx"]: r["err_pct"] for r in payload["resolution"]
             if r["ratio"] == 1000}
    for row in diff:
        ref = sharp[row["R_over_dx"]]
        assert row["err_pct"] < ref - 10.0, (
            f"R/dx={row['R_over_dx']}: diffuse {row['err_pct']:+.1f} % vs "
            f"sharp {ref:+.1f} %")


# --------------------------------------------------------------------------- #
# the honest failure: high contrast drifts with refinement
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "measured live defect (m20557): with rho_l/rho_v=1000 the density-weighted "
    "pair-Laplacian loses jump accuracy as the mesh is refined "
    "(-6.36 % at R/dx=30, -12.73 % at R/dx=40; the ratio-1 arm stays <1 % and "
    "the flux instrument stays -0.6 %, so the loss is in the Poisson solve). "
    "Gate: |err| < 5 % for every R/dx >= 20."))
def test_resolution_band_at_high_contrast(payload):
    rows = [r for r in payload["resolution"]
            if r["ratio"] == 1000 and r["R_over_dx"] >= 20]
    assert rows
    for r in rows:
        assert abs(r["err_pct"]) < GATES["resolution_err_max_pct_R_over_dx_ge_20"], \
            f"R/dx={r['R_over_dx']}: {r['err_pct']:+.2f} %"


def test_owner_gate_holds_everywhere(payload):
    """The 15 % acceptance holds in *every* measured configuration (honest floor)."""
    for key in ("density_ratio", "resolution", "translation"):
        for row in payload[key]:
            if row["arm"] in ("density", "diffuse"):
                assert abs(row["err_pct"]) < 15.0, (key, row)


# --------------------------------------------------------------------------- #
# reproducibility: the JSON is the measurement, not a transcription
# --------------------------------------------------------------------------- #
def test_measurement_reproduced_live():
    r = laplace_isph_case(radius=0.02, dx=0.001, outer=0.012, rho0=1000.0,
                          rho_vapour=10.0)
    assert r["density_ratio"] == pytest.approx(100.0, rel=1e-9)
    expected = json.loads(DATA.read_text())
    row = next(x for x in expected["density_ratio"]
               if x["ratio"] == 100 and x["arm"] == "density")
    assert (r["dp_over_theory"] - 1.0) * 100.0 == pytest.approx(row["err_pct"],
                                                               abs=1e-6)
    assert r["dp_flux_over_theory"] == pytest.approx(1.0 + row["flux_err_pct"] / 100.0,
                                                     abs=1e-9)