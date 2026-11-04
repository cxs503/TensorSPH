"""Verification of the free-surface pressure closure (Adami et al. 2012, 3.3).

The closure exists to remove one specific, measured defect: with density
summation and a Tait equation of state clamped at zero, the outermost fluid
shell carries ``p = 0`` because the kernel is truncated there. The pressure
gradient is then absorbed across one ligand spacing instead of being balanced by
the surface-tension force, which is the residual error source C3 of
``docs/verification_laplace_pressure_floor.md`` and the reason the Young-Laplace
jump there is still WITHHELD.

The tests below verify the *operator*, not the physics it feeds: a linear
hydrostatic pressure field is the one case where the correct answer is known in
closed form, and the body-force term is exactly what makes the kernel-weighted
average reproduce it. That gives a machine-judgeable acceptance value (error at
round-off) and a negative control (drop the term, error of order rho g dx).
"""
import math

import pytest
import torch

from tensorsph import Config, make_particles
from tensorsph.droplet_impact import (
    free_surface_pressure_closure,
    pair_kernels,
    rhs,
    rhs_all_pairs,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

FLOAT_TOL_PA = 0.01  # float32 round-off on a ~5 kPa field, ~20 summed terms


def lattice(nx: int, ny: int, dx: float) -> torch.Tensor:
    """Cell-centre lattice, the arrangement the solver's own initialisers use."""
    xs = (torch.arange(nx) + 0.5) * dx
    ys = (torch.arange(ny) + 0.5) * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def disc(radius: float, dx: float, centre=(0.0, 0.0)) -> torch.Tensor:
    """Particles of a lattice that fall inside a circle, as in the Laplace case."""
    side = int(2.0 * radius / dx) + 2
    pos = lattice(side, side, dx)
    shift = torch.tensor(centre) - torch.tensor(
        [0.5 * side * dx, 0.5 * side * dx])
    pos = pos + shift
    return pos[torch.linalg.vector_norm(pos - torch.tensor(centre), dim=-1) <= radius]


def pair_data(pos: torch.Tensor, cfg: Config):
    """Pair list in the form ``rhs`` builds: ``(i, j, delta_y, W_ij)``."""
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs[:, 0], pairs[:, 1]
    delta = pos[i] - pos[j]
    weight, _ = pair_kernels(delta, h)
    return i, j, delta[:, 1], weight


def truncated_column(cfg: Config, nx: int = 24, ny: int = 16):
    """Lattice whose top row has truncated support, with the rest interior."""
    pos = lattice(nx, ny, cfg.dx)
    rho = torch.full((pos.shape[0],), cfg.rho0)
    top = pos[:, 1] > pos[:, 1].max() - 1.5 * cfg.dx
    rho[top] = 0.97 * cfg.rho0
    return pos, rho, top


# --------------------------------------------------------------------------- #
# Exactness on the field whose answer is known in closed form
# --------------------------------------------------------------------------- #

def test_hydrostatic_field_is_reproduced_at_round_off():
    """p(y) = p_ref - rho0 g y is invariant under the extrapolation.

    Every summand ``p_j + rho_j g.(r_i - r_j)`` is algebraically ``p_i`` for a
    linear profile, and a weighted average of a constant is that constant
    whatever the kernel normalisation. So the closure cannot be blamed for any
    part of a hydrostatic error: any residual is elsewhere.
    """
    cfg = Config(dx=0.02, h_factor=1.3, rho0=1000.0, gravity=9.81)
    pos, rho, top = truncated_column(cfg)
    i, j, dy, w = pair_data(pos, cfg)
    p_ref = 5000.0
    p = p_ref - cfg.rho0 * cfg.gravity * pos[:, 1]

    out, replaced = free_surface_pressure_closure(
        p, rho, dy, w, (i, j), pos.shape[0], cfg)

    assert int(replaced.sum()) == int(top.sum()), "closure missed a surface particle"
    error = (out[replaced] - p[replaced]).abs().max().item()
    assert error < FLOAT_TOL_PA, f"hydrostatic error {error:g} Pa"
    # Interior particles carry their own Tait value by construction.
    assert torch.equal(out[~replaced], p[~replaced])


def test_body_force_term_is_what_makes_it_exact():
    """Negative control: without gravity the same field is biased by rho g <dy>.

    This is the measured quantity the term is there for, not a plausibility
    check: the bias is the kernel-weighted mean height offset of the interior
    neighbours, of order half a spacing, times rho g.
    """
    cfg = Config(dx=0.02, h_factor=1.3, rho0=1000.0, gravity=9.81)
    pos, rho, top = truncated_column(cfg)
    i, j, dy, w = pair_data(pos, cfg)
    p = 5000.0 - cfg.rho0 * cfg.gravity * pos[:, 1]

    cfg_nog = Config(dx=cfg.dx, h_factor=cfg.h_factor, rho0=cfg.rho0, gravity=0.0)
    with_term, _ = free_surface_pressure_closure(
        p, rho, dy, w, (i, j), pos.shape[0], cfg)
    without_term, _ = free_surface_pressure_closure(
        p, rho, dy, w, (i, j), pos.shape[0], cfg_nog)

    err_with = (with_term[top] - p[top]).abs().max().item()
    err_without = (without_term[top] - p[top]).abs().max().item()
    assert err_with < FLOAT_TOL_PA
    assert err_without > 10.0, (
        f"without the body term the bias should be tens of Pa, measured {err_without:g}")
    assert err_without / max(err_with, 1e-12) > 1.0e3


def test_bias_without_the_term_matches_rho_g_times_mean_height_offset():
    """The control above is the right order of magnitude, not just 'large'."""
    cfg = Config(dx=0.02, h_factor=1.3, rho0=1000.0, gravity=9.81)
    pos, rho, top = truncated_column(cfg)
    i, j, dy, w = pair_data(pos, cfg)
    p = 5000.0 - cfg.rho0 * cfg.gravity * pos[:, 1]
    cfg_nog = Config(dx=cfg.dx, h_factor=cfg.h_factor, rho0=cfg.rho0, gravity=0.0)
    without_term, _ = free_surface_pressure_closure(
        p, rho, dy, w, (i, j), pos.shape[0], cfg_nog)

    # Kernel-weighted mean of (y_i - y_j) over the interior sources of the top row.
    interior = rho >= cfg.free_surface_rho_threshold * cfg.rho0
    num = torch.zeros(pos.shape[0])
    den = torch.zeros(pos.shape[0])
    keep = interior[j]
    num = num.index_add(0, i[keep], (dy[keep] * w[keep]))
    den = den.index_add(0, i[keep], w[keep])
    predicted = cfg.rho0 * cfg.gravity * (num / den.clamp_min(1e-30))

    measured = without_term - p
    sel = top & (den > 0)
    ratio = (measured[sel] / predicted[sel].clamp_min(1e-12))
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=0.02), ratio


# --------------------------------------------------------------------------- #
# The source set matters: truncated neighbours must not feed the average
# --------------------------------------------------------------------------- #

def test_truncated_neighbours_do_not_contaminate_the_average():
    """Only interior neighbours are summed; a p = 0 surface neighbour is ignored."""
    cfg = Config(dx=0.01, h_factor=1.3, rho0=1000.0, gravity=0.0)
    n = 3
    rho = torch.tensor([0.95, 1.0, 0.95]) * cfg.rho0   # 0 and 2 truncated
    p = torch.tensor([0.0, 100.0, 0.0])
    dy = torch.tensor([0.0, 0.0])
    w = torch.tensor([0.4, 0.6])
    pairs = (torch.tensor([0, 0]), torch.tensor([1, 2]))

    out, replaced = free_surface_pressure_closure(p, rho, dy, w, pairs, n, cfg)
    # 2 has support but no interior neighbour, so it is left alone; 0 is lifted
    # to the interior value and cannot see its p = 0 neighbour at all.
    assert replaced.tolist() == [True, False, False]
    assert out[1].item() == pytest.approx(100.0)
    assert out[0].item() == pytest.approx(100.0), "surface source leaked into the mean"
    assert out[2].item() == pytest.approx(0.0)

    # Control: the filter is on density, not on pressure. A *genuinely interior*
    # neighbour that happens to be at a lower pressure does contribute, so the
    # assertion above is about truncated sources specifically.
    rho_full = torch.tensor([0.95, 1.0, 1.0]) * cfg.rho0
    p_low = torch.tensor([0.0, 100.0, 20.0])
    out_ctrl, _ = free_surface_pressure_closure(
        p_low, rho_full, dy, w, pairs, n, cfg)
    assert out_ctrl[0].item() == pytest.approx((100.0 * 0.4 + 20.0 * 0.6) / 1.0)


def test_particle_without_interior_support_keeps_its_tait_value():
    """A fully isolated surface particle is not invented a pressure."""
    cfg = Config(dx=0.01, rho0=1000.0, gravity=9.81, free_surface_pressure=True)
    rho = torch.tensor([0.9, 0.92]) * cfg.rho0
    p = tait_pressure(rho, cfg)          # both clamped to 0
    pairs = (torch.tensor([0]), torch.tensor([1]))
    out, replaced = free_surface_pressure_closure(
        p, rho, torch.tensor([0.01]), torch.tensor([0.3]), pairs, 2, cfg)
    assert not replaced.any()
    assert torch.equal(out, p)


def test_closure_is_idempotent_on_a_fully_interior_cloud():
    """No truncated particle means no change at all."""
    cfg = Config(dx=0.02, rho0=1000.0, gravity=9.81)
    pos = lattice(10, 10, cfg.dx)
    i, j, dy, w = pair_data(pos, cfg)
    rho = torch.full((pos.shape[0],), cfg.rho0)
    p = torch.full((pos.shape[0],), 1234.5)
    out, replaced = free_surface_pressure_closure(
        p, rho, dy, w, (i, j), pos.shape[0], cfg)
    assert not replaced.any()
    assert torch.equal(out, p)


# --------------------------------------------------------------------------- #
# Integration with the production right-hand side
# --------------------------------------------------------------------------- #

def test_flag_off_reproduces_the_tait_clamp_bit_for_bit():
    """Default behaviour is unchanged: the closure is opt-in."""
    cfg = Config(dx=0.004, h_factor=1.3, rho0=1000.0, gravity=0.0,
                 viscosity=0.0, sound_speed=20.0)
    assert cfg.free_surface_pressure is False
    pos = disc(0.02, cfg.dx)
    # 2 % over-dense by mass: an uncompressed cloud has p ~ 0 in the interior, so
    # lifting the shell from 0 to the interior value would be invisible.
    mass = torch.full((pos.shape[0],), 1.02 * cfg.rho0 * cfg.dx * cfg.dx)
    vel = torch.zeros_like(pos)
    rho, pressure, _ = rhs(pos, vel, mass, cfg)
    assert torch.equal(pressure, tait_pressure(rho, cfg))


def test_flag_on_lifts_the_zero_pressure_shell_and_stays_finite():
    """The closed surface carries the interior pressure instead of p = 0."""
    cfg = Config(dx=0.004, h_factor=1.3, rho0=1000.0, gravity=0.0,
                 viscosity=0.0, sound_speed=20.0)
    pos = disc(0.02, cfg.dx)
    # 2 % over-dense by mass: an uncompressed cloud has p ~ 0 in the interior, so
    # lifting the shell from 0 to the interior value would be invisible.
    mass = torch.full((pos.shape[0],), 1.02 * cfg.rho0 * cfg.dx * cfg.dx)
    vel = torch.zeros_like(pos)
    rho, p_clamped, _ = rhs(pos, vel, mass, cfg)

    cfg_on = Config(dx=cfg.dx, h_factor=cfg.h_factor, rho0=cfg.rho0,
                    gravity=0.0, viscosity=0.0, sound_speed=cfg.sound_speed,
                    free_surface_pressure=True)
    rho_on, p_closed, acc = rhs(pos, vel, mass, cfg_on)

    assert torch.equal(rho, rho_on), "the closure must not touch the density"
    assert torch.isfinite(p_closed).all() and torch.isfinite(acc).all()
    zero = p_clamped <= 0.0
    assert int(zero.sum()) > 0, "test needs a truncated shell to begin with"
    assert (p_closed[zero] > 0.0).float().mean() > 0.9
    assert p_closed.max().item() >= p_clamped.max().item()


def test_rhs_all_pairs_applies_the_same_rule():
    """The dense reference path implements the closure too, not silently ignores it."""
    cfg = Config(dx=0.004, h_factor=1.3, rho0=1000.0, gravity=0.0,
                 viscosity=0.0, sound_speed=20.0)
    pos = disc(0.02, cfg.dx)
    # 2 % over-dense by mass: an uncompressed cloud has p ~ 0 in the interior, so
    # lifting the shell from 0 to the interior value would be invisible.
    mass = torch.full((pos.shape[0],), 1.02 * cfg.rho0 * cfg.dx * cfg.dx)
    vel = torch.zeros_like(pos)
    rho, p_off, _ = rhs_all_pairs(pos, vel, mass, cfg)
    cfg_on = Config(dx=cfg.dx, h_factor=cfg.h_factor, rho0=cfg.rho0,
                    gravity=0.0, viscosity=0.0, sound_speed=cfg.sound_speed,
                    free_surface_pressure=True)
    rho_on, p_on, _ = rhs_all_pairs(pos, vel, mass, cfg_on)
    assert torch.equal(rho, rho_on)
    zero = p_off <= 0.0
    assert int(zero.sum()) > 0
    assert (p_on[zero] > 0.0).float().mean() > 0.9
    assert torch.isfinite(p_on).all()


def test_dense_and_pair_paths_agree_where_both_have_full_support():
    """On an interior particle the two paths must not disagree."""
    cfg = Config(dx=0.004, h_factor=1.3, rho0=1000.0, gravity=0.0,
                 viscosity=0.0, sound_speed=20.0, free_surface_pressure=True)
    pos = disc(0.03, cfg.dx)
    # 2 % over-dense by mass: an uncompressed cloud has p ~ 0 in the interior, so
    # lifting the shell from 0 to the interior value would be invisible.
    mass = torch.full((pos.shape[0],), 1.02 * cfg.rho0 * cfg.dx * cfg.dx)
    vel = torch.zeros_like(pos)
    rho_pair, p_pair, _ = rhs(pos, vel, mass, cfg)
    rho_dense, p_dense, _ = rhs_all_pairs(pos, vel, mass, cfg)
    core = torch.linalg.vector_norm(pos, dim=-1) < 0.6 * 0.03
    assert int(core.sum()) > 10
    assert torch.allclose(p_pair[core], p_dense[core], rtol=2e-3, atol=1.0)


def test_pool_case_remains_stable_with_the_closure_on():
    """One short run of the shipped drop-impact case: finite state, no gate breach."""
    cfg = Config(dx=0.01, steps=60, output_every=30, sound_speed=20.0,
                 free_surface_pressure=True, device="cpu")
    pos, mass = make_particles(cfg, torch.device("cpu"))
    vel = torch.zeros_like(pos)
    for _ in range(3):
        rho, pressure, acc = rhs(pos, vel, mass, cfg)
        assert torch.isfinite(rho).all() and torch.isfinite(acc).all()
        assert (pressure >= 0.0).all()
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
    assert rho.max().item() < 1.2 * cfg.rho0

# --------------------------------------------------------------------------- #
# Momentum bookkeeping and the P0 acceptance criterion
# --------------------------------------------------------------------------- #

def test_pairwise_antisymmetry_of_the_pressure_force_is_preserved():
    """The closure changes p_i, not the symmetry of the pair force.

    The momentum equation applies ``-(p_i/rho_i^2 + p_j/rho_j^2) grad W`` to i and
    the same scalar with the opposite sign to j, so replacing a particle's
    pressure with the extrapolated value keeps the pair anti-symmetric. The net
    force on an isolated droplet therefore stays at round-off instead of picking
    up a spurious drift -- which matters because a closed droplet has no external
    force to hide it behind.
    """
    cfg = Config(dx=0.004, h_factor=1.3, rho0=1000.0, gravity=0.0,
                 viscosity=0.0, sound_speed=20.0)
    pos = disc(0.02, cfg.dx)
    mass = torch.full((pos.shape[0],), 1.02 * cfg.rho0 * cfg.dx * cfg.dx)
    vel = torch.zeros_like(pos)
    defect = {}
    for flag in (False, True):
        cfg_f = Config(dx=cfg.dx, h_factor=cfg.h_factor, rho0=cfg.rho0,
                       gravity=0.0, viscosity=0.0, sound_speed=cfg.sound_speed,
                       free_surface_pressure=flag)
        _, _, acc = rhs(pos, vel, mass, cfg_f)
        net = (mass[:, None] * acc).sum(dim=0).norm()
        scale = (mass[:, None] * acc).abs().sum()
        defect[flag] = (net / scale).item()
    assert defect[True] < 1.0e-6, f"spurious net force {defect[True]:.2e}"
    assert defect[True] < 10.0 * max(defect[False], 1.0e-9)


def _closure_report():
    import json
    from pathlib import Path
    here = Path(__file__).resolve().parent.parent
    path = here / "benchmarks" / "laplace_pressure_floor" / "report_closure.json"
    if not path.exists():
        pytest.skip(f"{path.name} not generated yet")
    return json.loads(path.read_text())


def test_p0_acceptance_all_four_points_within_15pct():
    """The P0 acceptance criterion from backlog_update_m2971.md.

    ``absolute_time_average_tail`` must be inside 15 % at c0 = 20/5 m/s crossed
    with dx = 0.002/0.0015 m. Nothing here is recomputed: the numbers are the
    committed report, so a regression in the solver that is not re-benchmarked
    cannot silently pass.
    """
    report = _closure_report()
    assert report["variant"] == "free_surface_pressure"
    runs = {(r["dx_m"], r["sound_speed"]): r for r in report["runs"]}
    wanted = {(0.002, 20.0), (0.002, 5.0), (0.0015, 20.0), (0.0015, 5.0)}
    missing = wanted - set(runs)
    assert not missing, f"ladder incomplete: {sorted(missing)}"
    errors = {}
    for key in wanted:
        run = runs[key]
        assert run["finite"], f"{key} not finite"
        errors[key] = run["absolute_time_average_tail"]["rel_error_pct"]
        assert abs(errors[key]) < 15.0, (
            f"dx={key[0]:g} c0={key[1]:g}: tail-averaged Laplace error "
            f"{errors[key]:+.1f} % exceeds 15 %")
    # Three of the four are also inside 8 %; recorded so a regression that
    # leaves them all at 14.9 % is visible.
    assert sum(abs(e) < 8.0 for e in errors.values()) >= 3, errors


def test_p0_acceptance_is_an_improvement_over_the_tait_clamp():
    """The clamped report is the baseline: the closure must beat it everywhere."""
    import json
    from pathlib import Path
    here = Path(__file__).resolve().parent.parent
    base = json.loads((here / "benchmarks" / "laplace_pressure_floor"
                       / "report.json").read_text())
    closed = _closure_report()
    base_runs = {(r["dx_m"], r["sound_speed"]): r for r in base["runs"]}
    for run in closed["runs"]:
        key = (run["dx_m"], run["sound_speed"])
        if key not in base_runs:
            continue
        err_closed = abs(run["absolute_time_average_tail"]["rel_error_pct"])
        err_clamped = abs(base_runs[key]["absolute_time_average_tail"]["rel_error_pct"])
        assert err_closed < err_clamped, (
            f"{key}: closure {err_closed:.1f} % did not improve on "
            f"clamp {err_clamped:.1f} %")
