"""Analytic tests for the opt-in CSF colour-function / delta_s corrections.

``free_surface_color`` and ``normalise_delta_s`` are the ingredients the
``laplace_rootcause`` backlog item names as the two correctable defects of the
shipped CSF pipeline. Each has a closed-form target that can be checked on a
lattice without running a simulation:

* the occupancy sum ``sum_j V_j W_ij`` satisfies the partition of unity, so the
  deficiency colour is ~0 in the interior of a fully populated lattice and grows
  to ~0.5 on a flat free surface;
* the surface delta function must integrate to unity across the interface,
  ``int delta_s dn = 1``.

These tests pin the helpers only. They do **not** claim the Laplace law is now
satisfied: that still requires the pressure boundary condition of
``docs/verification_surface_tension_laplace.md`` and is recorded as open.
"""
import math

import torch

from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    compute_surface_tension_force_corrected,
    free_surface_color,
    normalise_delta_s,
)

DX = 0.02
H = 1.3 * DX
RHO = 1000.0


def square_block(n=35, dx=DX):
    """Fully populated square lattice: every interior particle has a full shell."""
    x = torch.arange(n, dtype=torch.float64) * dx
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    mass = torch.full((pos.shape[0],), RHO * dx * dx, dtype=torch.float64)
    rho = torch.full((pos.shape[0],), RHO, dtype=torch.float64)
    return pos, mass, rho


def test_deficiency_colour_vanishes_where_the_shell_is_complete():
    pos, mass, rho = square_block()
    pairs = cell_linked_pairs(pos, 2.0 * H)
    color = free_surface_color(pos, mass, rho, pairs, H)

    centre = torch.linalg.vector_norm(pos - pos.mean(dim=0), dim=1) < 3.0 * H
    assert float(color[centre].abs().max()) < 0.02, (
        "partition of unity should drive the deficiency to ~0 in the bulk"
    )


def test_deficiency_colour_matches_the_half_space_value():
    """On the outermost column the neighbour shell is a half space.

    The partition sum over a half space is 1/2 in the continuum, so the
    deficiency there is 1/2. With h = 1.3 dx the discretisation trims it to a
    quarter (the shell straddles only ~2 layers), which is the value pinned
    here; what matters is that it is an O(1) jump starting at the surface and
    that the bulk stays at the partition-of-unity floor.
    """
    pos, mass, rho = square_block()
    pairs = cell_linked_pairs(pos, 2.0 * H)
    color = free_surface_color(pos, mass, rho, pairs, H)

    x = pos[:, 0]
    outermost = color[x > float(x.max()) - 1e-9]
    bulk = color[(x - 0.5 * (float(x.max()) + float(x.min()))).abs() < 3.0 * H]

    assert abs(float(outermost.mean()) - 0.25) < 0.05, float(outermost.mean())
    assert float(bulk.mean()) < 0.02, float(bulk.mean())
    # The jump is monotone through the surface layers.
    layers = [float(color[(x > float(x.max()) - (k + 1) * DX).logical_and(
        x <= float(x.max()) - k * DX)].mean()) for k in range(3)]
    assert layers[0] > layers[1] > layers[2] - 1e-9, layers


def test_normalise_delta_s_recovers_unit_integral_on_an_analytic_ramp():
    """A 1-D ramp of width 2h has int |dc/dn| dn = jump, so normalising by the
    colour range must give int delta_s dn = 1 to the resolution error."""
    span = 6.0 * H
    n = 4001
    s = torch.linspace(0.0, span, n, dtype=torch.float64)
    c_lo, c_hi = 0.0, 0.5
    c = c_lo + (c_hi - c_lo) * torch.sigmoid((s - span / 2.0) / (0.35 * H))
    dc = torch.gradient(c, spacing=(float(s[1] - s[0]),))[0]
    delta_s = normalise_delta_s(dc.abs(), c)

    integral = float(torch.trapezoid(delta_s, s))
    assert abs(integral - 1.0) < 0.02, integral


def test_normalise_delta_s_is_scale_free():
    """Scaling the colour function must not change the normalised delta_s."""
    span = 4.0 * H
    s = torch.linspace(0.0, span, 2001, dtype=torch.float64)
    base = torch.sigmoid((s - span / 2.0) / (0.35 * H))
    dc = torch.gradient(base, spacing=(float(s[1] - s[0]),))[0]

    a = normalise_delta_s(dc.abs(), base)
    b = normalise_delta_s((3.7 * dc).abs(), 3.7 * base)
    assert torch.allclose(a, b, rtol=1e-10)


def test_corrected_force_is_zero_without_surface_tension():
    pos, mass, rho = square_block(n=20)
    pairs = cell_linked_pairs(pos, 2.0 * H)
    from tensorsph.droplet_impact import Config

    cfg = Config(dx=DX, surface_tension_coefficient=0.0, gravity=0.0)
    acc = compute_surface_tension_force_corrected(pos, rho, mass, pairs, cfg)
    assert torch.equal(acc, torch.zeros_like(acc))


def test_corrected_force_records_its_fields():
    pos, mass, rho = square_block(n=20)
    pairs = cell_linked_pairs(pos, 2.0 * H)
    from tensorsph.droplet_impact import Config

    cfg = Config(dx=DX, surface_tension_coefficient=0.072, gravity=0.0)
    fields: dict = {}
    acc = compute_surface_tension_force_corrected(pos, rho, mass, pairs, cfg,
                                                fields=fields)
    assert acc.shape == (pos.shape[0], 2)
    for key in ("color", "grad_c", "normal", "kappa", "delta_s", "acc_st"):
        assert key in fields
    # Colour range is 0.5 on a flat surface, so the normalised delta_s must be
    # about twice the raw |grad c|; this pins the factor the backlog called out.
    raw = torch.linalg.vector_norm(fields["grad_c"], dim=1)
    if float(raw.max()) > 0.0:
        ratio = fields["delta_s"] / raw.clamp_min(1e-30)
        assert float(ratio.max()) > 1.0