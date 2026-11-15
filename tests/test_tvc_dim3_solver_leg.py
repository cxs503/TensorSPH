"""3-D solver leg of ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

Pins (a) the TVC hook of the 3-D dam break, and (b) the analytic anchors the
3-D correction is graded against in ``docs/xval_sphinx_tvc/SOLVER_3D_LEG.md``.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

import dimension_leg as DL  # noqa: E402

from tensorsph.dam_break_3d import (  # noqa: E402
    DamBreak3DConfig,
    DamBreak3DRun,
    ritter_front_over_a,
    run_case,
)


def short_case(**kw) -> DamBreak3DConfig:
    cfg = DamBreak3DConfig(dx=0.04, sound_speed=25.0, steps=30, output_every=15)
    for key, value in kw.items():
        setattr(cfg, key, value)
    return cfg


def test_tvc_defaults_off():
    cfg = short_case()
    assert cfg.tvc_every == 0 and cfg.tvc_alpha is None
    run = run_case(cfg)
    assert run.tvc_log == []


def test_tvc_negative_cadence_rejected():
    cfg = short_case(tvc_every=-1)
    with pytest.raises(ValueError, match="tvc_every"):
        run_case(cfg)


def test_tvc_hook_applies_and_conserves_mass():
    """The correction moves particles, keeps mass, and stays bounded."""
    base = run_case(short_case())
    tvc = run_case(short_case(tvc_every=1, tvc_alpha=0.05))
    assert len(tvc.tvc_log) > 0
    assert tvc.final_mass == base.final_mass
    assert tvc.mass_drift == 0.0
    assert tvc.escaped == 0
    row = tvc.tvc_log[-1]
    assert set(row) >= {"step", "time", "max_shift_over_h",
                        "mean_shift_over_h", "clipped_fraction"}
    numeric = [v for v in row.values() if isinstance(v, (int, float))]
    assert len(numeric) == len(row) - 1          # only kept_fraction may be None
    assert all(torch.isfinite(torch.tensor(numeric)))
    # a single acoustic step may not teleport a particle past its own kernel
    assert row["max_shift_over_h"] < 2.0
    # the correction must actually have moved the cloud
    assert not torch.allclose(tvc.final_positions, base.final_positions)


def test_tvc_max_shift_clips():
    run = run_case(short_case(tvc_every=1, tvc_alpha=0.2, tvc_max_shift=1e-3))
    assert max(r["clipped_fraction"] for r in run.tvc_log) > 0.0
    h = 1.3 * 0.04
    assert max(r["max_shift_over_h"] for r in run.tvc_log) <= 1e-3 / h + 1e-9


def test_front_frame_starts_at_zero_and_matches_ritter_convention():
    cfg = short_case()
    run = run_case(cfg)
    tau, za = run.dimensionless(cfg)
    assert za[0] == 0.0
    assert ritter_front_over_a(0.0) == 0.0
    # the two curves are then directly comparable
    assert za[-1] >= 0.0 and len(za) == len(run.times)
    offset = run.initial_front / cfg.column_width
    # the pre-fix convention reported x_max/a: at t = 0 that is the fraction of
    # the column the lattice actually fills (0.9 when a/dx is an integer)
    assert 0.5 < offset < 1.0
    assert run.initial_front <= cfg.column_width


def test_front_frame_offset_is_the_documented_one():
    """A synthetic run reproduces the +offset of the pre-fix convention."""
    cfg = DamBreak3DConfig(dx=0.03)
    run = DamBreak3DRun(
        times=[0.0, 0.01], front=[0.135, 0.20], kinetic_energy=[0, 0],
        potential_energy=[0, 0], mass=1.0, escaped=0, rebuilds=0, max_speed=0.0,
        max_density_compression=0.0, compression_before_wall_impact=0.0,
        wall_impact_time=None, initial_mass=1.0, final_mass=1.0,
        final_positions=torch.zeros(1, 3), initial_front=0.135)
    _, za = run.dimensionless(cfg)
    assert za[0] == 0.0
    assert za[1] == pytest.approx((0.20 - 0.135) / cfg.column_width)
    # pre-fix convention: absolute x / a, i.e. +0.9 at t = 0 on this lattice
    assert 0.135 / cfg.column_width == pytest.approx(0.9)


@pytest.mark.parametrize("dx", [0.03, 0.02])
def test_anchor_bulk_fixed_point_is_exact_in_3d(dx):
    assert DL.bulk_fixed_point(dx, 3) < 1e-10


@pytest.mark.parametrize("dx", [0.03, 0.02])
def test_anchor_wall_branch_3d(dx):
    assert DL.wall_residual(dx, 3, 2.0) < 1e-10
    assert DL.wall_residual(dx, 3, 1.0) > 0.4


def test_anchor_surface_3d_reproduces_dimension_leg():
    """The 3-D free-surface anchor is the dimension leg's number, re-measured."""
    row = DL.surface_shift(0.02, 3)
    assert row["max_shift_over_h_per_alpha"] == pytest.approx(1.1747849080245327, rel=1e-9)
    cont = 2.0 * DL.plane_integral(0.13, 3) * 0.13
    # honest: the cubic lattice misses the owner 15 % gate by 1.1 pp
    bias = abs(row["max_shift_over_h_per_alpha"] - cont) / cont
    assert 0.15 < bias < 0.17


def test_pre_fix_under_correction_factor():
    factor = 7.0 / (10.0 * DL.HBAR * 0.02)
    assert factor == pytest.approx(26.923076923076923)
    assert factor > 20.0