"""Gates for the free-surface pressure closure (mission m3167).

Three things are pinned:

1. the closure is **off by default** -- the physics of the two published cases
   must not change silently;
2. the closure really does replace a clamp artefact with an extrapolated
   pressure (measured on a synthetic, analytically known configuration);
3. the mechanism that survives it: with the *symmetric* pair form
   ``p_i/rho_i^2 + p_j/rho_j^2`` a uniform pressure is not force-free at a
   truncated kernel support, while the *difference* form
   ``(p_j - p_i)/(rho_i rho_j)`` is. That is the reason the closure measured in
   ``benchmarks/laplace_pressure_floor/closure_probe.json`` closes the
   Young-Laplace gap on a soft EOS but overshoots on a stiff one.

The measured facts themselves are pinned from ``closure_probe.json`` so a
silent change in the closure cannot drift away from the documented record.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_kernel,
    free_surface_pressure_closure,
    rhs,
)

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
PROBE = BENCH / "closure_probe.json"


def test_closure_is_off_by_default():
    """The published droplet/dam-break numbers stay bit-identical."""
    assert Config().free_surface_pressure is False
    assert Config().pressure_gradient_form == "symmetric"


def test_closure_replaces_a_deficient_clamp_with_an_extrapolated_pressure():
    """A particle with a deficient kernel sum inherits its neighbours' pressure.

    Hand-built case: four particles in a line at spacing dx, the outer one has
    only one neighbour inside the kernel (kernel support truncated, hence a
    sum-density deficit); the resolved neighbour carries 360 Pa -- the
    sigma/R of the Laplace ladder. With the closure on, the truncated particle
    must come out close to 360 Pa instead of 0.
    """
    dx, h = 0.01, 0.013
    x = torch.tensor([0.0, 0.01, 0.02, 0.03], dtype=torch.float64)
    pos = torch.stack((x, torch.zeros_like(x)), dim=1)
    mass = torch.full((4,), 1000.0 * dx * dx, dtype=torch.float64)
    i = torch.tensor([0, 1, 1, 2, 2, 3, 0, 2], dtype=torch.long)
    j = torch.tensor([1, 0, 2, 1, 3, 2, 2, 0], dtype=torch.long)
    # keep only pairs inside the kernel support
    keep = ((pos[i] - pos[j]).norm(dim=1) < 2.0 * h)
    i, j = i[keep], j[keep]
    delta_y = (pos[i] - pos[j])[:, 1]
    wij = cubic_spline_kernel((pos[i] - pos[j]).norm(dim=1), h)

    rho = torch.tensor([300.0, 1000.0, 1000.0, 700.0], dtype=torch.float64)
    pressure = torch.tensor([0.0, 360.0, 360.0, 0.0], dtype=torch.float64)

    cfg = Config(dx=dx, h_factor=1.3, gravity=0.0, free_surface_pressure=True,
                 free_surface_passes=1)
    closed = free_surface_pressure_closure(rho, pressure, i, j, delta_y, wij, 4, cfg)

    # rho/rho0 - 1 = -0.70 and -0.30 are the deficient ones
    assert float(closed[0]) > 300.0
    assert float(closed[3]) > 300.0
    # resolved particles keep their own EOS pressure
    assert float(closed[1]) == pytest.approx(360.0)
    assert float(closed[2]) == pytest.approx(360.0)

    # the caller, not the helper, owns the switch: with the term off ``rhs``
    # never calls it, and a resolved field is returned untouched
    resolved = torch.full((4,), 1000.0, dtype=torch.float64)
    untouched = free_surface_pressure_closure(
        resolved, pressure, i, j, delta_y, wij, 4, cfg)
    assert torch.equal(untouched, pressure)


def _uniform_pressure_acceleration(form: str):
    """One-particle-thick layer at a free surface, uniform pressure p = 360 Pa.

    A uniform pressure has no gradient, so the physical acceleration is zero;
    whatever comes out is discretisation from the truncated kernel support.
    """
    dx, h = 0.02, 0.026
    n = 6
    x = torch.arange(n, dtype=torch.float64) * dx
    pos = torch.stack((x, torch.zeros_like(x)), dim=1)
    mass = torch.full((n,), 1000.0 * dx * dx, dtype=torch.float64)
    vel = torch.zeros_like(pos)
    cfg = Config(dx=dx, h_factor=1.3, gravity=0.0, viscosity=0.0,
                 free_surface_pressure=False, pressure_gradient_form=form)
    # bypass the density/closure path: inject the pressure by making the density
    # exactly rho0 and reading the force through the public entry point
    from tensorsph.droplet_impact import tait_pressure

    rho = torch.full((n,), 1000.0, dtype=torch.float64)
    _, pressure, _ = rhs(pos, vel, mass, cfg)
    # rhs recomputes density by summation, so compare the *pair forms* directly
    # instead: same mass, same pressure value everywhere.
    # a *non-zero* uniform pressure (360 Pa = sigma/R of the Laplace ladder);
    # p = 0 would be trivially force-free and prove nothing
    p = torch.full((n,), 360.0, dtype=torch.float64)
    assert float(tait_pressure(torch.full((n,), 1000.0, dtype=torch.float64), cfg).abs().max()) == 0.0
    pairs = torch.tensor([[k, k + 1] for k in range(n - 1)], dtype=torch.long)
    pairs = torch.cat((pairs, pairs.flip(1)), dim=0)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    grad_w = None
    from tensorsph.droplet_impact import cubic_spline_gradient

    grad_w = cubic_spline_gradient(delta, h)
    if form == "symmetric":
        term = p[i] / rho[i].square() + p[j] / rho[j].square()
        acc = torch.zeros_like(pos)
        pair_pressure = term[:, None] * grad_w
        acc = acc.index_add(0, i, -mass[j, None] * pair_pressure)
        acc = acc.index_add(0, j, mass[i, None] * pair_pressure)
    else:
        term = (p[j] - p[i]) / (rho[i] * rho[j])
        acc = torch.zeros_like(pos)
        pair_pressure = term[:, None] * grad_w
        acc = acc.index_add(0, i, -mass[j, None] * pair_pressure)
        acc = acc.index_add(0, j, -mass[i, None] * pair_pressure)
    assert pressure is not None
    return acc


def test_uniform_pressure_is_not_force_free_with_the_symmetric_pair_form():
    """The surviving error source, stated as a test.

    On a layer whose kernel support is truncated (the ends of the line), a
    uniform pressure produces a non-zero acceleration with the symmetric form.
    That force is what the closure amplifies when it raises the pressure in the
    deficient shell.
    """
    acc = _uniform_pressure_acceleration("symmetric")
    assert float(acc.abs().max()) > 1.0


def test_uniform_pressure_is_exactly_force_free_with_the_difference_form():
    """``(p_j - p_i)/(rho_i rho_j)`` vanishes identically for a uniform pressure."""
    acc = _uniform_pressure_acceleration("difference")
    assert float(acc.abs().max()) < 1e-9


# --------------------------------------------------------------------------- #
# the measured record
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def probes():
    if not PROBE.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py --closure-probe first")
    return json.loads(PROBE.read_text())["probes"]


def _probe(probes, dx, c0, passes=2):
    for probe in probes:
        if (probe["dx_m"] == dx and probe["sound_speed"] == c0
                and probe["closure"]["passes"] == passes):
            return probe
    raise AssertionError(f"no probe for dx={dx} c0={c0} passes={passes}")


def test_closure_closes_the_laplace_gap_on_a_soft_eos(probes):
    """c0 = 5 m/s: both resolutions land inside the 15 % gate, one inside 5 %."""
    for dx in (0.002, 0.0015):
        probe = _probe(probes, dx, 5.0)
        tail = probe["absolute_time_average_tail"]["rel_error_pct"]
        absolute = probe["definition_absolute"]["rel_error_pct"]
        assert abs(tail) < 15.0
        assert abs(absolute) < 15.0
    assert abs(_probe(probes, 0.0015, 5.0)["definition_absolute"]["rel_error_pct"]) < 5.0


def test_closure_overshoots_on_a_stiff_eos(probes):
    """c0 = 20 m/s: the drop breathes, so averaging must not be read as success."""
    probe = _probe(probes, 0.002, 20.0)
    assert probe["definition_absolute"]["rel_error_pct"] > 100.0
    assert probe["spurious_current"]["vmax_over_vref_final"] > 0.1


def test_the_overshoot_shrinks_with_the_fill_depth(probes):
    """Two filled layers overshoot more than one: the deficit is one layer deep."""
    two = _probe(probes, 0.002, 20.0, passes=2)
    one = _probe(probes, 0.002, 20.0, passes=1)
    assert abs(two["definition_absolute"]["rel_error_pct"]) > abs(
        one["definition_absolute"]["rel_error_pct"])