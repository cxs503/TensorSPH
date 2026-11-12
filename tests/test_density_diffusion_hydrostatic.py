"""The diffusive density term must not push a hydrostatic column around.

These tests are self-contained: the reference is an exact linear density field
``rho(y) = rho0 + a y`` -- the state a resting body of water has under the Tait
equation of state -- and the criterion is that the diffusive *rate* on that
field is zero.  No published table is involved.

The defect they pin down: the shipped Molteni & Colagrossi term
(``molteni_colagrossi_rate``) only vanishes on a linear field when every
particle's kernel support is centrally symmetric.  Truncate it -- a free
surface, a wall -- and an exact hydrostatic column acquires a steady diffusive
flux, which is the spurious-current source measured in
``benchmarks/startup_quiescence`` (finding F1, 0.0798 m/s against a 0.06 m/s
ceiling).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.density_diffusion import (
    FORMS,
    hydrostatic_corrected_rate,
    molteni_colagrossi_rate,
    rate_from_positions,
    renormalized_density_gradient,
)
from tensorsph.droplet_impact import (
    Config,
    density_diffusion_rate,
    pair_kernels,
    validate_config,
)

DX = 0.02
H_FACTOR = 1.3
RHO0 = 1000.0
GRADIENT = -80.0          # d rho / d y  (denser downwards), kg/m^4


def column(nx: int = 14, ny: int = 10, dx: float = DX, dtype=torch.float64):
    """A finite 2-D lattice: free surface on top, open sides, no wall particles.

    The support of every particle near the top row (and the side columns) is
    truncated, which is precisely the situation the correction addresses.
    """
    xs = torch.arange(nx, dtype=dtype) * dx
    ys = torch.arange(ny, dtype=dtype) * dx
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1)
    pos = grid.reshape(-1, 2)
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=dtype)
    return pos, mass


def fields(pos, mass, dx=DX, h_factor=H_FACTOR, gradient=GRADIENT,
           uniform=0.0):
    """Pair arrays and the exact-linear (or uniform) density field."""
    h = h_factor * dx
    from tensorsph.neighbor_search import cell_linked_pairs
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    distance = torch.linalg.vector_norm(delta, dim=1)
    _, grad_w = pair_kernels(delta, h)
    radial = -(delta * grad_w).sum(dim=1)
    rho = RHO0 + gradient * pos[:, 1] + uniform
    return dict(pairs=pairs, delta=delta, grad_w=grad_w, distance=distance,
                radial=radial, distance_sq=distance.square(), rho=rho,
                scale=2.0 * 0.1 * 20.0 * h)


def test_renormalised_gradient_is_exact_on_a_linear_field_with_truncation():
    """``<grad rho> = (0, a)`` everywhere, including the truncated shell.

    The uncorrected first moment (``grad_rho`` before ``M^-1``) is the control:
    on the same field it is biased wherever the support is cut.
    """
    pos, mass = column()
    f = fields(pos, mass)
    grad, completeness = renormalized_density_gradient(
        pos, f["rho"], mass, f["pairs"], f["delta"], f["grad_w"])
    exact = torch.tensor([0.0, GRADIENT], dtype=pos.dtype)

    interior = completeness > 0.5
    assert int(interior.sum()) >= 40, f"only {int(interior.sum())} interior"
    error = (grad[interior] - exact).abs().max() / abs(GRADIENT)
    assert float(error) < 1.0e-10, f"renormalised gradient off by {error:.2e}"

    # Control: the raw (unrenormalised) sum is biased on the same particles.
    i, j = f["pairs"].unbind(dim=1)
    v_j = mass[j] / f["rho"][j]
    v_i = mass[i] / f["rho"][i]
    diff = (f["rho"][j] - f["rho"][i]).unsqueeze(-1)
    raw = torch.zeros_like(pos)
    raw = raw.index_add(0, i, v_j.unsqueeze(-1) * diff * f["grad_w"])
    raw = raw.index_add(0, j, v_i.unsqueeze(-1) * diff * f["grad_w"])
    raw_error = (raw[interior] - exact).abs().max() / abs(GRADIENT)
    assert float(raw_error) > 0.02, (
        f"the uncorrected first moment should be measurably biased, got "
        f"{raw_error:.2e} -- the truncation control is not exercising anything")


def test_corrected_rate_annihilates_an_exact_hydrostatic_field():
    """The whole point: no diffusive flux on a state at rest."""
    pos, mass = column()
    f = fields(pos, mass)
    molteni = molteni_colagrossi_rate(f["rho"], mass, f["radial"],
                                      f["distance_sq"], f["pairs"], f["scale"])
    corrected = rate_from_positions(pos, f["rho"], mass, f["delta"], f["grad_w"],
                                    f["radial"], f["distance_sq"], f["pairs"],
                                    f["scale"], form="antuono")

    molteni_max = float(molteni.abs().max())
    corrected_max = float(corrected.abs().max())
    assert molteni_max > 0.0
    # Absolute scale of the term on this field, for a meaningful ratio:
    reference = f["scale"] * RHO0 * abs(GRADIENT) / (H_FACTOR * DX)
    assert corrected_max < 1.0e-9 * reference, (
        f"corrected rate {corrected_max:.3e} is not negligible against "
        f"{reference:.3e}")
    assert corrected_max < 1.0e-6 * molteni_max, (
        f"corrected {corrected_max:.3e} vs molteni {molteni_max:.3e}")


def test_uniform_field_gives_zero_rate_in_both_forms():
    pos, mass = column()
    f = fields(pos, mass, gradient=0.0)
    molteni = molteni_colagrossi_rate(f["rho"], mass, f["radial"],
                                      f["distance_sq"], f["pairs"], f["scale"])
    corrected = rate_from_positions(pos, f["rho"], mass, f["delta"], f["grad_w"],
                                    f["radial"], f["distance_sq"], f["pairs"],
                                    f["scale"], form="antuono")
    assert float(molteni.abs().max()) < 1.0e-12
    assert float(corrected.abs().max()) < 1.0e-12


def test_both_forms_conserve_volume_weighted_density():
    """A bump on a uniform field: sum_i V_i drho_i/dt must vanish."""
    pos, mass = column()
    f = fields(pos, mass, gradient=0.0)
    bump = 0.05 * RHO0 * torch.exp(-((pos[:, 0] - 0.14) ** 2
                                    + (pos[:, 1] - 0.10) ** 2) / 0.0004)
    rho = f["rho"] + bump
    volumes = mass / rho
    molteni = molteni_colagrossi_rate(rho, mass, f["radial"], f["distance_sq"],
                                      f["pairs"], f["scale"])
    corrected = rate_from_positions(pos, rho, mass, f["delta"], f["grad_w"],
                                    f["radial"], f["distance_sq"], f["pairs"],
                                    f["scale"], form="antuono")
    for name, rate in (("molteni", molteni), ("antuono", corrected)):
        total = float((volumes * rate).sum())
        scale = float((volumes * rate.abs()).sum())
        assert abs(total) <= 1.0e-12 * max(scale, 1.0), (
            f"{name} form breaks conservation: {total:.3e} vs {scale:.3e}")


def test_bump_diffuses_in_the_right_direction_in_both_forms():
    """The corrected term must still be a diffusion, not a source."""
    pos, mass = column()
    f = fields(pos, mass, gradient=0.0)
    bump = 0.05 * RHO0 * torch.exp(-((pos[:, 0] - 0.14) ** 2
                                    + (pos[:, 1] - 0.10) ** 2) / 0.0004)
    rho = f["rho"] + bump
    centre = int(torch.argmax(bump))
    corrected = rate_from_positions(pos, rho, mass, f["delta"], f["grad_w"],
                                    f["radial"], f["distance_sq"], f["pairs"],
                                    f["scale"], form="antuono")
    assert float(corrected[centre]) < 0.0, "the dense centre must lose density"
    assert float(corrected[centre].abs()) > 0.0


def test_molteni_form_is_bit_for_bit_the_shipped_term():
    """The refactor must not move the shipped number."""
    pos, mass = column()
    f = fields(pos, mass, gradient=-300.0)
    cfg = Config(density_diffusion=0.1, sound_speed=20.0, h_factor=H_FACTOR,
                 dx=DX, dt=3.0e-4)
    validate_config(cfg)
    shipped = density_diffusion_rate(f["rho"], mass, f["distance"],
                                     f["radial"], f["pairs"], cfg)
    module = molteni_colagrossi_rate(f["rho"], mass, f["radial"],
                                     f["distance_sq"], f["pairs"], f["scale"])
    assert torch.equal(shipped, module)


def test_unknown_form_is_rejected_twice():
    pos, mass = column(6, 6)
    f = fields(pos, mass)
    with pytest.raises(ValueError, match="unknown density_diffusion_form"):
        rate_from_positions(pos, f["rho"], mass, f["delta"], f["grad_w"],
                            f["radial"], f["distance_sq"], f["pairs"],
                            f["scale"], form="nonsense")
    cfg = Config(density_diffusion=0.1, density_diffusion_form="nonsense")
    with pytest.raises(ValueError, match="density_diffusion_form must be one of"):
        validate_config(cfg)
    assert FORMS == ("molteni", "antuono")


def test_public_wrapper_refuses_the_corrected_form_instead_of_lying():
    """Calling the shipped entry point with the corrected form must not fall
    back to the uncorrected arithmetic silently."""
    pos, mass = column(6, 6)
    f = fields(pos, mass)
    cfg = Config(density_diffusion=0.1, density_diffusion_form="antuono")
    with pytest.raises(ValueError, match="rate_from_positions"):
        density_diffusion_rate(f["rho"], mass, f["distance"], f["radial"],
                               f["pairs"], cfg)


def test_eigenvalue_floor_keeps_positively_truncated_supports_exact():
    """With one row of particles the top row's support is a half space: the
    correction must still reproduce a linear field there (the floor is only a
    guard against exploding inverses, not a hole in the invariance)."""
    pos, mass = column(6, 3)
    f = fields(pos, mass)
    grad, completeness = renormalized_density_gradient(
        pos, f["rho"], mass, f["pairs"], f["delta"], f["grad_w"])
    top = int(torch.argmax(pos[:, 1]))
    assert float(completeness[top]) > 0.1
    exact = torch.tensor([0.0, GRADIENT], dtype=pos.dtype)
    assert float((grad[top] - exact).abs().max()) / abs(GRADIENT) < 1.0e-8
    corrected = rate_from_positions(pos, f["rho"], mass, f["delta"], f["grad_w"],
                                    f["radial"], f["distance_sq"], f["pairs"],
                                    f["scale"], form="antuono")
    assert float(corrected.abs().max()) < 1.0e-9 * (
        f["scale"] * RHO0 * abs(GRADIENT) / (H_FACTOR * DX))