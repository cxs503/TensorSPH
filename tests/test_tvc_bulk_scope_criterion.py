"""The ``bulk`` TVC scope: first-moment criterion, and the analytic law behind
the density-gradient term.

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

``tests/test_tvc_hydrostatic.py`` records that ``tvc_scope="bulk"`` blows a
resting column up (2.9 m/s against 0.38 m/s for ``all``) and attributes it to
"correcting the interior while leaving the surface uncorrected".  That is a
symptom, not a mechanism.  This file pins the mechanism, which is a defect in
the *scope criterion* rather than in the transport-velocity idea:

* the criterion ``C_i >= tvc_bulk_c * c_ref`` is a **zeroth-moment** test
  (``sum_j V_j W_ij``).  ``W`` saturates near full support while ``grad W`` --
  the quantity the correction is built from -- does not: a particle one shell
  inside the last missing neighbour shell has ``C_i ~ 1`` and a badly truncated
  ``grad-1`` integral.  So the criterion admits a tail of particles with
  ``|g| h`` up to 0.20, ~3000x the true interior value (6.7e-5), and those
  particles receive a spurious displacement of ``alpha * 0.20 * h`` per step.
* tightening the concentration threshold does not help (``C`` is flat near 1),
* replacing it by the **first-moment** criterion ``|g_i| h <= tvc_bulk_g``
  does: the resting column returns to the no-TVC noise floor.

The analytic anchor used by the operator tests: on a uniform lattice carrying a
volume field ``V(x) = V0 exp(kappa y)`` the ported integral is exactly

    g_i = -2 grad ln V_i = +2 grad ln rho_i

independent of ``h`` and ``dx`` (measured, <3% at the hydrostatic ``kappa =
g / c0^2``; see ``benchmarks/xval_sphinx_tvc/bulk_scope_criterion.py``).
This is the term that keeps the operator from being an exact fixed point on a
compressed state, and it is *not* what makes the resting column unstable (its
volume field is uniform, ``V = dx^2``, so the term vanishes there).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.dam_break import (  # noqa: E402
    DamBreakConfig, make_column, run_case,
)
from tensorsph.droplet_impact import hydrostatic_density, wall_lattice  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.particle_shifting import (  # noqa: E402
    concentration, estimate_reference_concentration,
)
from tensorsph.transport_velocity_correction import (  # noqa: E402
    kernel_gradient_integral, transport_velocity_correction,
)


def _lattice(nx: int, ny: int, dx: float) -> torch.Tensor:
    xs = torch.arange(nx, dtype=torch.float64) * dx
    ys = torch.arange(ny, dtype=torch.float64) * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def _interior(pos: torch.Tensor, margin: float) -> torch.Tensor:
    return ((pos[:, 0] > margin) & (pos[:, 0] < pos[:, 0].max() - margin)
            & (pos[:, 1] > margin) & (pos[:, 1] < pos[:, 1].max() - margin))


# --------------------------------------------------------------------------
# operator: the scope criterion
# --------------------------------------------------------------------------
def test_scope_tol_masks_by_the_gradient_integral_not_by_concentration():
    """``scope_tol`` keeps exactly the particles with ``|g| h <= tol``."""
    dx = 0.02
    h = 1.3 * dx
    pos = _lattice(30, 20, dx)
    volume = torch.full((pos.shape[0],), dx * dx, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    g = kernel_gradient_integral(pos, volume, h, pairs=pairs)
    gh = torch.linalg.vector_norm(g, dim=1) * h
    tol = 1.0e-3
    out = transport_velocity_correction(pos, volume, h, alpha=0.05,
                                        pairs=pairs, scope_tol=tol)
    kept = torch.linalg.vector_norm(out.shift, dim=1) > 0.0
    # the boundary particles are the only ones with a truncated support here
    assert int(kept.sum()) > 0, "no particle should survive a 1e-3 tolerance"
    assert int(kept.sum()) < pos.shape[0], "the truncated shell must be masked"
    assert bool((gh[kept] <= tol * (1.0 + 1e-12)).all())
    assert bool((gh[~kept] > tol).all())
    assert out.extras["scope_tol_kept_fraction"] == pytest.approx(
        float(kept.to(g.dtype).mean()))
    # the masked particles are on the boundary of the cloud, the kept ones
    # inside it: the criterion is geometric, not a random subsample
    assert float(pos[~kept, 0].max()) >= float(pos[kept, 0].max())


def test_scope_tol_validation():
    dx = 0.02
    pos = _lattice(6, 6, dx)
    volume = torch.full((pos.shape[0],), dx * dx, dtype=torch.float64)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            transport_velocity_correction(pos, volume, 1.3 * dx, scope_tol=bad)
    # scope_tol = None is the upstream behaviour (no criterion)
    out = transport_velocity_correction(pos, volume, 1.3 * dx)
    assert out.extras["scope_tol"] is None
    assert out.extras["scope_tol_kept_fraction"] is None


# --------------------------------------------------------------------------
# operator: the analytic density-gradient law
# --------------------------------------------------------------------------
@pytest.mark.parametrize("h_ratio", [1.1, 1.3, 1.5])
def test_density_gradient_gives_minus_two_grad_log_volume(h_ratio):
    """``g = -2 grad ln V`` on a uniform lattice, within 3%, h-independent."""
    dx = 0.02
    h = h_ratio * dx
    kappa = 9.81 / 20.0 ** 2            # hydrostatic dln(rho)/dz with c0 = 20 m/s
    pos = _lattice(60, 30, dx)
    volume = dx * dx * torch.exp(kappa * pos[:, 1])
    pairs = cell_linked_pairs(pos, 2.0 * h)
    internal = _interior(pos, 2.0 * h)
    g = kernel_gradient_integral(pos, volume, h, pairs=pairs)
    predicted = -2.0 * kappa            # -2 d(ln V)/dy, the only non-zero entry
    measured = float(g[internal, 1].mean())
    assert abs(measured - predicted) / abs(predicted) < 0.03, (
        f"h/dx={h_ratio}: g_y={measured:.5e} vs analytic {predicted:.5e}")
    # no spurious component across the field
    assert abs(float(g[internal, 0].mean())) < 1.0e-4 * abs(predicted)


def test_uniform_volume_is_the_fixed_point():
    """With ``V = dx^2`` everywhere the same lattice gives a vanishing integral."""
    dx = 0.02
    h = 1.3 * dx
    pos = _lattice(40, 20, dx)
    volume = torch.full((pos.shape[0],), dx * dx, dtype=torch.float64)
    g = kernel_gradient_integral(pos, volume, h, pairs=cell_linked_pairs(pos, 2.0 * h))
    gh = torch.linalg.vector_norm(g, dim=1)[_interior(pos, 2.0 * h)] * h
    assert float(gh.max()) < 1.0e-9


# --------------------------------------------------------------------------
# the actual initial state of the resting column
# --------------------------------------------------------------------------
def _column_state(dx: float = 0.04, column_width: float = 0.2):
    cfg = DamBreakConfig(dx=dx, dt=1.0e-4, steps=2, output_every=1, device="cpu",
                         column_width=column_width, column_height=0.4,
                         pool_width=2.0, tank_height=0.75, hydrostatic_init=True)
    device = torch.device("cpu")
    pos, mass = make_column(cfg, device)
    wall = wall_lattice(cfg, device, top=cfg.tank_top())
    rho = hydrostatic_density(pos[:, 1], cfg, surface=cfg.column_height)
    return cfg, pos, mass, wall, rho


def test_concentration_criterion_admits_truncated_support_particles():
    """The zeroth-moment criterion lets a 10^2 tail of large ``|g| h`` through."""
    cfg, pos, mass, wall, rho = _column_state()
    h = cfg.h_factor * cfg.dx
    volume = mass / rho
    all_positions = torch.cat((pos, wall), dim=0)
    g = kernel_gradient_integral(
        all_positions,
        torch.cat((volume, torch.full((wall.shape[0],), cfg.dx ** 2,
                                      dtype=volume.dtype))),
        h, pairs=cell_linked_pairs(all_positions, 2.0 * h),
        wall_count=wall.shape[0])
    gh = torch.linalg.vector_norm(g, dim=1) * h
    c = concentration(pos, volume, h)
    c_ref = estimate_reference_concentration(c)
    selected = c >= 0.95 * c_ref
    full = ((pos[:, 1] < cfg.column_height - 2.0 * h)
            & (pos[:, 0] < cfg.column_width - 2.0 * h))

    assert int(selected.sum()) > int(full.sum()), (
        "the concentration criterion is supposed to over-select")
    # the tail: selected particles whose gradient integral is 10^3 times the
    # full-support value -- these are what the correction mis-places
    assert float(gh[selected].max()) > 100.0 * float(gh[full].max())
    assert float(gh[full].max()) < 1.0e-3
    assert int(selected[gh > 0.01].sum()) > 0, "no truncated-support particle selected"
    # ... and the volume field is uniform at t = 0, so the density-gradient law
    # above contributes nothing to the resting column
    assert float(volume.max() - volume.min()) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# solver: the criterion decides the stability of the resting column
# --------------------------------------------------------------------------
def _resting_cfg(**kw) -> DamBreakConfig:
    base = dict(dx=0.04, dt=1.0e-4, steps=200, output_every=50, device="cpu",
                column_width=0.2, column_height=0.4, pool_width=2.0,
                tank_height=0.75, hydrostatic_init=True)
    return DamBreakConfig(**base, **kw)


def test_first_moment_scope_keeps_the_resting_column_at_rest():
    """``tvc_scope="bulk_g"`` restores the no-TVC noise floor."""
    baseline = run_case(_resting_cfg(tvc_every=0))
    run = run_case(_resting_cfg(tvc_every=1, tvc_alpha=0.05,
                                tvc_scope="bulk_g", tvc_bulk_g=0.02))
    assert run.mass_drift == 0.0
    assert run.max_speed < 2.0 * baseline.max_speed + 0.05, (
        f"bulk_g max_speed={run.max_speed:.3f} vs baseline {baseline.max_speed:.3f}")
    assert run.max_density_compression <= 2.0 * baseline.max_density_compression + 0.01
    h = 1.3 * 0.04
    assert run.max_tvc_shift < 0.02 * h, "the kept set must not be shifted far"


def test_tightening_the_concentration_threshold_does_not_cure_the_bulk_scope():
    """``C_i`` saturates: even ``c = 0.999`` leaves the column unstable.

    This is the regression test for the *diagnosis*: the defect is the choice of
    zeroth moment, not the value of its threshold.  The legacy criterion is kept
    (and kept unstable) on purpose -- see ``docs/xval_sphinx_tvc/
    BULK_SCOPE_CRITERION.md``.
    """
    run = run_case(_resting_cfg(tvc_every=1, tvc_alpha=0.05,
                                tvc_scope="bulk", tvc_bulk_c=0.999))
    assert run.max_speed > 1.0, (
        f"expected the documented concentration-criterion instability, got "
        f"{run.max_speed:.3f} m/s")
    assert run.max_density_compression > 0.05