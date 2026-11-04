"""Structural invariants of the CSF surface tension model, and the harness
that measures them (Young-Laplace benchmark B5).

The quantitative Young-Laplace balance is *not* asserted to within a tolerance
here: it is a documented, measured failure of the current model (see
``docs/benchmarks/surface-tension-verification.md``). What is pinned is

* the exact invariances that no correct implementation can violate -- mirror
  symmetry, zero net force on a closed surface, zero traction in the bulk --
  because those are the properties the 0.8.0 formulation did violate, and
* the measurement machinery itself, so the numbers in the benchmark report are
  produced by verified code.
"""
import math

import pytest
import torch

from tensorsph import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

import importlib.util
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure" / "run.py"
_spec = importlib.util.spec_from_file_location("laplace_benchmark", _BENCH)
laplace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(laplace)


SIGMA = 0.072
DX = 0.005
RADIUS = 0.04


def _droplet_force(radius=RADIUS, dx=DX, sigma=SIGMA, **config_kwargs):
    """Acceleration on a static drop; ``config_kwargs`` selects the scheme."""
    cfg = Config(dx=dx, rho0=1000.0, sound_speed=20.0, gravity=0.0,
                 surface_tension_coefficient=sigma, **config_kwargs)
    pos, mass, density = laplace.circular_droplet(radius, dx)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * cfg.dx)
    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)
    return pos, mass, density, acc


def test_lattice_sampling_is_symmetric():
    """The droplet sampling must not itself break the symmetry being tested."""
    pos, mass, density = laplace.circular_droplet(RADIUS, DX)
    key = pos.round(decimals=12)
    as_set = {tuple(row.tolist()) for row in key}
    for row in key:
        assert (-row[0].item(), row[1].item()) in as_set
        assert (row[0].item(), -row[1].item()) in as_set


def test_measurement_recovers_a_known_force():
    """The force integration must reproduce an analytic traction exactly.

    A radial spring ``a = -k r`` has no net force on a closed particle set and
    an exactly known force ``-k * sum(m x)`` on the right half, so the harness
    can be checked against numbers that do not come from SPH at all.
    """
    pos, mass, _, _ = _droplet_force()
    k = 3.0
    acc = -k * pos
    measured = laplace.measure(pos, mass, torch.full_like(mass, 1000.0), acc, RADIUS)
    expected = -k * float((mass[pos[:, 0] > 0] * pos[pos[:, 0] > 0, 0]).sum())
    got = measured["force_right_half"][0]
    assert abs(got - expected) <= 1e-9 * abs(expected), (got, expected)
    assert abs(measured["force_right_half"][1]) < 1e-9
    assert math.hypot(*measured["force_total"]) < 1e-9
    assert measured["mirror_asymmetry"] < 1e-9


def test_mirror_symmetry_of_the_traction():
    """A symmetric droplet must get a purely radial force on either half.

    The 0.8.0 formulation measured 0.40-0.46 here (40 % of the radial force
    appearing out of plane); the kernel-sum colour gradient is symmetric to
    round-off.
    """
    pos, mass, density, acc = _droplet_force()
    measured = laplace.measure(pos, mass, density, acc, RADIUS)
    assert measured["mirror_asymmetry"] < 1e-9, measured
    assert measured["rotation_asymmetry"] < 1e-9, measured


def test_closed_surface_carries_no_net_force():
    """A closed free surface cannot create momentum."""
    pos, mass, density, acc = _droplet_force()
    total = (mass[:, None] * acc).sum(dim=0)
    assert float(torch.linalg.vector_norm(total)) < 1e-9 * (2.0 * SIGMA)


def test_bulk_traction_is_exactly_zero():
    """grad c of a complete stencil is zero, so the interior must not be pushed."""
    pos, mass, density, acc = _droplet_force()
    bulk = torch.linalg.vector_norm(pos, dim=1) < 0.5 * RADIUS
    assert bulk.any()
    assert float(torch.linalg.vector_norm(acc[bulk], dim=1).max()) == 0.0


def test_traction_points_inward_on_a_convex_droplet():
    """Surface tension must pull a convex droplet towards its centre."""
    pos, mass, density, acc = _droplet_force()
    measured = laplace.measure(pos, mass, density, acc, RADIUS)
    assert measured["force_right_half"][0] < 0.0
    assert measured["force_top_half"][1] < 0.0
    assert measured["inward_fraction"] > 0.9


def test_force_is_linear_in_sigma():
    """The traction integrates to a force proportional to sigma."""
    pos, mass, density, acc1 = _droplet_force(sigma=SIGMA)
    _, _, _, acc2 = _droplet_force(sigma=2.0 * SIGMA)
    f1 = float((mass[pos[:, 0] > 0, None] * acc1[pos[:, 0] > 0]).sum(dim=0)[0])
    f2 = float((mass[pos[:, 0] > 0, None] * acc2[pos[:, 0] > 0]).sum(dim=0)[0])
    assert abs(f2 / f1 - 2.0) < 1e-10


def _half_force_ratio(**config_kwargs):
    """|F_cut| / (2 sigma) on the right half of the documented drop."""
    pos, mass, density, acc = _droplet_force(**config_kwargs)
    half = (mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)
    return float(half.norm() / (2.0 * SIGMA))


def test_effective_surface_tension_at_the_documented_resolution():
    """Pin the measured amplitude so a silent change of the model is caught.

    Measurement history on this lattice (square, ``R/h = 6.15``, ``dx = 5 mm``,
    sigma = 0.072 N/m, 2-D), all reproduced by the parametrisations below:

    ===========================================  ===========================
    formulation                                  ``|F_cut| / (2 sigma)``
    ===========================================  ===========================
    legacy neighbour-count colour                0.215
    consistent colour + MLS curvature            0.791
    consistent colour + quadratic interface fit  0.711  (current default)
    ===========================================  ===========================

    * the legacy value is the recorded deficiency of m2399;
    * 0.791 is what m2447 pinned as ``0.79 +- 0.06`` -- it is the MLS curvature
      variant, still reachable through ``csf_curvature="mls"``;
    * m2979 made ``csf_curvature="fit_quadratic"`` the default
      (``docs/verification_csf_curvature_default.md``).  The quadratic fit reads
      the curvature *correctly* (error +5.3 % instead of +24.9 % at dx = 2 mm),
      which removes the accidental cancellation between the +25 % curvature bias
      and the -25 % ``delta_s`` deficit, so the cut force drops to 0.711.
      That is a documented trade, not a regression, but m2979 did not update
      this pin; it is updated here from the measurement above.

    The three-way comparison is asserted, not just the default, so that a silent
    swap of *either* operator (colour field or curvature) still turns this red.
    """
    h = Config(dx=DX).h_factor * DX
    assert RADIUS / h == pytest.approx(6.1538, abs=1e-3)  # documented resolution

    default = _half_force_ratio()
    assert abs(default - 0.711) < 0.01, (default, RADIUS / h)

    mls = _half_force_ratio(csf_curvature="mls")
    assert abs(mls - 0.791) < 0.01, mls

    legacy = _half_force_ratio(csf_scheme="legacy")
    assert legacy < 0.4, legacy
