"""Gates for the 3-D density-ratio arm of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE.

The 2-D arm established (rounds 3-4, ``docs/xval_projection_laplace_density_ratio.md``)
that the Young-Laplace jump must be measured with the *force-density* balance
``grad p = f`` -- i.e. a pressure Poisson operator that carries no ``1/rho`` --
because the traction ``f = -sigma kappa delta_s n`` is rho-independent by
construction.  The 3-D module had no density ratio at all, so this file pins the
ported behaviour on the coarse 3-D level (``R/dx = 5``, 4096 particles, ~1 s per
run on one thread):

* the jump is ratio blind to round-off for the force-density arm,
* the equal-density path is bit-identical to the historical numbers,
* the naive constant-coefficient arm must blow up at high contrast,
* the interface geometry (colour function, cell volume) does not move with the
  ratio, so the ratio is the only knob,
* ``sigma = 0`` returns a zero jump.

The full ladder (ratio sweep + refinement + figure) is
``scripts/xval_projection_laplace_3d/density_ratio.py``; its JSON is the source
of every number in ``docs/xval_projection_laplace_3d_density_ratio.md``.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch.set_num_threads(int(os.environ.get("TENSORSPH_THREADS", "1")))

from tensorsph.laplace_isph3d import (  # noqa: E402
    csf_state3_two_phase,
    laplace_isph3d_case,
    pressure_poisson3,
    sphere_block,
)

RADIUS = 0.02
SIGMA = 0.072
RHO0 = 1000.0
DX = 0.004                      # R/dx = 5 (the historical coarse level)
BASE = dict(radius=RADIUS, sigma=SIGMA, rho0=RHO0, outer=0.012, h_factor=1.3,
            dx=DX, curvature_method="mls", curvature_stencil=2.0)
DP_THEORY = 2.0 * SIGMA / RADIUS          # 7.2 Pa
# The reservoir of historical 3-D numbers this file must not move
HISTORICAL_COARSE_ERR_PCT = 3.914244       # benchmarks/xval_projection_laplace_3d


def _run(**kw):
    args = dict(BASE)
    args.update(kw)
    return laplace_isph3d_case(**args)


def test_equal_density_path_is_bit_identical():
    """G2: the force-density arm at 1:1 reproduces the historical jump exactly."""
    hist = _run()
    fd = _run(poisson_weighting="force_density")
    assert fd["poisson_weighting"] == "uniform"           # 1:1 shortcut is taken
    assert fd["dp_measured_Pa"] == pytest.approx(hist["dp_measured_Pa"], rel=1e-12)
    assert abs(hist["dp_error_pct"] - HISTORICAL_COARSE_ERR_PCT) < 1e-6


def test_force_density_is_ratio_blind_3d():
    """G3: dp does not move when the vapour density drops by 1000x."""
    one = _run(rho_vapour=RHO0, poisson_weighting="force_density")
    thousand = _run(rho_vapour=1.0, poisson_weighting="force_density")
    assert thousand["density_ratio"] == pytest.approx(1000.0, rel=1e-12)
    assert one["density_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert thousand["dp_measured_Pa"] == pytest.approx(one["dp_measured_Pa"],
                                                       rel=1e-9)
    assert one["p_core_Pa"] / one["dp_theory_Pa"] == pytest.approx(
        thousand["p_core_Pa"] / thousand["dp_theory_Pa"], rel=1e-9)


def test_naive_uniform_arm_must_fail_at_high_contrast():
    """G5: the constant-coefficient arm with a ratio-aware source over-predicts."""
    naive = _run(rho_vapour=1.0, poisson_weighting="uniform")
    assert abs(naive["dp_error_pct"]) > 15.0, naive["dp_error_pct"]
    # and the variable-coefficient (density) arm degrades but stays far below it
    dens = _run(rho_vapour=1.0, poisson_weighting="density")
    assert abs(dens["dp_error_pct"]) < abs(naive["dp_error_pct"]) / 10.0


def test_owner_gate_forcedensity_all_ratios():
    """G1: the validated arm stays inside the 15 % owner gate at every ratio."""
    for ratio in (1.0, 10.0, 100.0, 1000.0):
        rep = _run(rho_vapour=RHO0 / ratio, poisson_weighting="force_density")
        assert abs(rep["dp_error_pct"]) < 15.0, (ratio, rep["dp_error_pct"])


def test_traction_flux_is_ratio_blind():
    """The independent traction readout never touches the Poisson operator."""
    ref = _run(rho_vapour=RHO0, poisson_weighting="force_density")["dp_flux_Pa"]
    for ratio in (10.0, 1000.0):
        rep = _run(rho_vapour=RHO0 / ratio, poisson_weighting="force_density")
        assert rep["dp_flux_Pa"] == pytest.approx(ref, rel=1e-9)
        assert rep["dp_flux_over_theory"] == pytest.approx(1.0, abs=0.10)


def test_sigma_zero_gives_zero_jump_at_high_contrast():
    rep = _run(rho_vapour=1.0, poisson_weighting="force_density", sigma=0.0)
    assert rep["dp_theory_Pa"] == 0.0
    assert abs(rep["dp_measured_Pa"]) < 1e-9
    assert abs(rep["dp_flux_Pa"]) < 1e-9


def test_carrier_geometry_is_ratio_invariant():
    """The ratio changes mass only: cell volume, colour field and interface stay."""
    pos1, m1, liq1 = sphere_block(RADIUS, DX, 0.012, RHO0, rho_vapour=None)
    pos2, m2, liq2 = sphere_block(RADIUS, DX, 0.012, RHO0, rho_vapour=1.0)
    assert torch.equal(pos1, pos2) and torch.equal(liq1, liq2)
    assert float(m2[liq2].mean() / m2[~liq2].mean()) == pytest.approx(1e3, rel=1e-12)
    # vol = m / rho_phase is the same number in both phases
    vol1 = m1 / RHO0
    vol2 = m2 / torch.where(liq2, torch.full_like(m2, RHO0), torch.full_like(m2, 1.0))
    assert float(vol2.max() - vol2.min()) == pytest.approx(0.0, abs=1e-18)
    assert torch.allclose(vol1, vol2)
    st1 = csf_state3_two_phase(pos1, m1, liq1, 1.3 * DX, RHO0, SIGMA,
                              curvature_method="mls", curvature_stencil=2.0)
    rho_phase = torch.where(liq2, torch.full_like(m2, RHO0),
                            torch.full_like(m2, 1.0))
    st2 = csf_state3_two_phase(pos2, m2, liq2, 1.3 * DX, RHO0, SIGMA,
                               curvature_method="mls", curvature_stencil=2.0,
                               rho_phase=rho_phase)
    # vacuous-neighbour particles can be NaN (no support); compare with that mask
    finite = torch.isfinite(st1["color"]) & torch.isfinite(st2["color"])
    assert bool(finite.any())
    assert torch.allclose(st1["color"][finite], st2["color"][finite], atol=1e-12)
    assert torch.allclose(st1["kappa"][finite], st2["kappa"][finite], atol=1e-12)
    assert torch.allclose(st1["delta_s"][finite], st2["delta_s"][finite], atol=1e-12)


def test_density_weighted_poisson_scales_with_uniform_rho():
    """Unit check: at uniform ``rho`` the operator is ``(1/rho)`` times the plain one.

    ``-div((1/rho) grad p) = s`` with constant ``rho`` is exactly
    ``-grad^2 p = rho s``, so the solution must be ``rho`` times the plain solve
    to round-off.  (In the case itself the source carries the matching ``1/rho``,
    which is why the resulting pressure -- and the jump -- is unchanged.)
    """
    pos, mass, liquid = sphere_block(0.02, 0.004, 0.012)
    vol = mass / RHO0
    from tensorsph.laplace_isph3d import box_pairs
    pairs = box_pairs(pos, 2.6 * 0.004)
    source = -torch.ones(pos.shape[0], dtype=torch.float64)
    half = 0.02 + 0.012
    edge = 1.6 * 0.004
    fixed = (pos[:, 0].abs() > half - edge)
    p_ref, _ = pressure_poisson3(pos, source, pairs, vol, 1.3 * 0.004, fixed)
    rho = torch.full_like(mass, RHO0)
    p_den, _ = pressure_poisson3(pos, source, pairs, vol, 1.3 * 0.004, fixed,
                                 rho=rho, weighting="density")
    expect = RHO0 * p_ref
    scale = float(expect.abs().max())
    assert scale > 0.0
    assert float((p_den - expect).abs().max()) < 1e-9 * scale


def test_artifact_ledger_matches_library_when_present():
    """If the sweep JSON exists, its validated arm must respect the same gates."""
    js = ROOT / "scripts/xval_projection_laplace_3d/results/density_ratio.json"
    if not js.exists():
        pytest.skip("sweep artifact not generated in this checkout")
    import json
    data = json.loads(js.read_text())
    fd = [r for r in data["ratio_ladder"]
          if r["requested_weighting"] == "force_density"]
    assert fd, "ratio ladder missing the force-density arm"
    spread = max(r["dp_measured_Pa"] for r in fd) - min(r["dp_measured_Pa"] for r in fd)
    assert 100.0 * spread / fd[0]["dp_theory_Pa"] < data["gates"]["ratio_spread_pct"]
    assert data["verdict"]["all_pass"] is True