"""Regression and analytic tests for the CSF surface delta function.

``delta_s`` is the one part of the CSF force that the earlier verification
rounds did not check directly: the harness scored the curvature, the force
direction and the *acceleration-space* cut force, but never the defining
identity of a surface delta function,

    int delta_s dn = 1     (a delta function has unit mass per unit area),

and it measured the cut force as ``sum_i m_i a_i``, which divides the CSF force
density by the *summation* density.  At a free surface that density is
under-counted by ~20 %, so the sum is inflated by 1/(rho/rho0) ~ 1.25 -- which
is almost exactly the factor by which the old delta function was too weak.

These tests pin the corrected accounting:

* ``int delta_s dn`` for both normalisation schemes, against the analytic 1;
* the geometric cut force ``sum_i (m_i/rho0) rho_i a_i`` -- the force the
  continuum identity ``F = 2 sigma`` refers to -- for both schemes;
* the relationship between the two bookkeepings, which is what makes the old
  "10.7 %" a compensating error rather than an accuracy statement;
* the identity ``int_fluid |grad c| dn = 1 - c_surf`` the "jump" scheme rests on;
* defaults and argument validation.

Measured numbers come from ``benchmarks/csf_delta_s/run.py``; see
``docs/verification_delta_s_normalization.md``.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "csf_delta_s"


def _bench_module():
    spec = importlib.util.spec_from_file_location("csf_delta_s_run", BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _bench_module()


@pytest.fixture(scope="module")
def rows(bench):
    """One static evaluation per scheme at dx = 2 mm (R/h ~ 7.7)."""
    return {norm: bench.evaluate(dx=0.002, normalization=norm)
            for norm in ("fixed", "jump")}


def test_measured_jump_normalisation_reaches_unity(rows):
    """The defining identity of the CSF delta function: int delta_s dn = 1.

    The "jump" scheme reaches it to 0.5 % at dx = 2 mm; the historical "fixed"
    scheme misses it by 25 % because it assumes a colour jump of 1/2 where the
    discrete field delivers 1 - 0.63.
    """
    jump = rows["jump"]["delta_s"]["integral_dn"]
    fixed = rows["fixed"]["delta_s"]["integral_dn"]
    assert jump == pytest.approx(1.0, abs=0.05)
    assert 0.70 < fixed < 0.80          # documented systematic deficit
    assert jump / fixed > 1.25


def test_the_colour_field_span_is_the_only_missing_factor(rows):
    """``int_fluid |grad c| dn`` equals the measured colour span to <2 %.

    That identity is *why* the normalisation can be closed by dividing by
    ``1 - c_surf``: no ghost layer is needed, the discrete colour field is
    already consistent, it simply does not run from 1 to 0 inside the fluid.
    """
    colour = rows["jump"]["color"]
    assert colour["grad_integral_rel_error_pct"] == pytest.approx(0.0, abs=2.0)
    # the outermost layer carries ~0.63, not the assumed 0.5
    assert 0.55 < colour["c_surf_q10"] < 0.70
    assert colour["c_bulk_interior"] == pytest.approx(1.0, abs=1e-6)


def test_geometric_cut_force_is_the_criterion(rows):
    """``sum (m/rho0) rho a`` reproduces ``2 sigma`` only with the closed delta_s.

    This is the accounting the continuum identity refers to: the CSF force
    density is a force per unit *geometric* volume, so a particle's volume is
    the lattice volume ``m/rho0``, not ``m/rho``.
    """
    jump = rows["jump"]["cut_force"]["ratio_geometric"]
    fixed = rows["fixed"]["cut_force"]["ratio_geometric"]
    assert abs(jump - 1.0) < 0.15        # inside the platform gate
    assert fixed < 0.80                  # default scheme: -21...-29 %
    assert jump / fixed > 1.25


def test_acceleration_space_cut_force_is_inflated_by_the_density_deficit(rows):
    """The old metric differs by exactly ``rho0 / rho_w`` -- the compensating error.

    ``sum m a`` divides by the summation density, which at the interface is
    ``rho_w ~ 0.80 rho0``.  The inflation is therefore ~1.25, and it cancelled
    the 25 % weakness of the old delta function.  Pinning the ratio keeps that
    cancellation from being reported as accuracy again.
    """
    row = rows["jump"]
    rho_w = row["interface_density"]["delta_s_weighted_rho_over_rho0"]
    ratio = (row["cut_force"]["ratio_acceleration_space"]
             / row["cut_force"]["ratio_geometric"])
    assert ratio == pytest.approx(1.0 / rho_w, rel=0.03)
    assert 0.75 < rho_w < 0.85


def test_curvature_and_direction_are_normalisation_independent(rows):
    """Changing delta_s must not move the curvature or the force direction."""
    assert rows["jump"]["curvature_kappa"]["rel_error_pct"] == pytest.approx(
        rows["fixed"]["curvature_kappa"]["rel_error_pct"], abs=1e-9)
    assert rows["jump"]["direction"]["cos_band_median"] > 0.95
    assert rows["fixed"]["direction"]["cos_band_median"] > 0.95


def test_default_scheme_is_unchanged():
    """The default stays "fixed": the dynamic trade-off is documented, not hidden."""
    from tensorsph.csf_consistent import DELTA_S_SCHEMES
    from tensorsph.droplet_impact import Config

    assert DELTA_S_SCHEMES == ("fixed", "jump")
    assert Config().csf_delta_s == "fixed"


def test_invalid_scheme_is_rejected():
    from tensorsph.csf_consistent import csf_fields

    pos = torch.rand(8, 2, dtype=torch.float64)
    mass = torch.ones(8, dtype=torch.float64)
    pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="delta_s_normalization"):
        csf_fields(pos, mass, pairs, 0.1, 1000.0, delta_s_normalization="bogus")
    from tensorsph.droplet_impact import Config, validate_config
    with pytest.raises(ValueError, match="csf_delta_s"):
        validate_config(Config(csf_delta_s="bogus"))


def test_config_plumbs_the_scheme_into_the_force():
    """``cfg.csf_delta_s`` must reach the force, otherwise the switch is decorative."""
    from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
    from tensorsph.neighbor_search import cell_linked_pairs
    from tensorsph.surface_tension import compute_surface_tension_force

    spec = importlib.util.spec_from_file_location(
        "laplace_law_run", ROOT / "benchmarks" / "laplace_law" / "run.py")
    law = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(law)
    pos, mass = law.hexagonal_droplet(0.02, 0.004, 1000.0)
    h = 1.3 * 0.004
    pairs = cell_linked_pairs(pos, 4.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, cell_linked_pairs(pos, 2.0 * h))
    accs = {}
    for norm in ("fixed", "jump"):
        cfg = Config(dx=0.004, h_factor=1.3, surface_tension_coefficient=7.2,
                     csf_scheme="consistent", csf_curvature="fit_quadratic",
                     csf_delta_s=norm)
        accs[norm] = float(torch.linalg.vector_norm(
            compute_surface_tension_force(pos, rho, mass, pairs, cfg), dim=1).sum())
    # delta_s ratio = (1/span) / 2, span = 1 - c_surf ~ 0.37...0.38
    assert 1.25 < accs["jump"] / accs["fixed"] < 1.45