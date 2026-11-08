"""2-D added mass of a circular section from a potential-flow solve.

Why this module exists
----------------------
The bow ice-splash case carries a *reduced* floe model: the momentum flux onto
a virtual force plate is integrated into an impulse ``J`` and turned into a
launch speed by dividing by the floe's **dry** mass ``rho_ice pi r^2``
(:func:`tensorsph.ice_splash.floe_launch_estimate`). A body accelerated in a
liquid must accelerate the surrounding liquid as well, so the inertia resisting
the impulse is ``m_ice + C_m rho_w pi r^2``, with ``C_m`` the added-mass
coefficient. Dividing by the dry mass therefore returns an *upper bound*, and
the overestimate is exactly ``1 + C_m rho_w / rho_ice``.

For a circular section in an unbounded inviscid fluid the coefficient is known
in closed form: the added mass per unit depth equals the displaced mass,
``C_m = 1`` (Lamb, *Hydrodynamics*, 6th ed., §71; Landau & Lifshitz, *Fluid
Mechanics*, §11). The number is not quoted from memory: :func:`cylinder_added_mass`
recomputes it from the potential-flow problem and the unit tests check the
numeric value against ``1``.

Geometry and boundary conditions
--------------------------------
The disc translates along ``+x`` with unit speed, so its rigid-body potential is
``phi = -(x - x_c)`` on the surface, continuous with the decaying dipole
``-a^2 (x - x_c) / r^2`` at infinity. Imposing that value on the body nodes is
equivalent to the Neumann condition ``dphi/dn = U n_x`` on the fluid side.
Three surroundings are supported:

``"unbounded"``
    Free field; the closed-form anchor is ``C_m = 1``.
``"rigid_wall"``
    Rigid plane ``gap`` radii below the lowest point of the disc, mirror ``+``
    (``dphi/dy = 0``) -- the deck the floe rests on.
``"free_surface"``
    Plane ``gap`` radii above the highest point of the disc, mirror ``-``
    (``phi = 0``, the linearised free-surface condition) -- a floe in the water
    that the splash tongue pushes.

The kinetic energy of the flow gives the added mass,

.. math:: M = 2 K / U^2, \\qquad K = \\tfrac12 \\rho \\int_{fluid} |\\nabla\\phi|^2 dA,

which is exact for an impulsively started rigid body with the fluid at rest at
infinity (Landau & Lifshitz §11). The slowly decaying dipole tail outside the
computational box is added analytically, so the reported number is not
grid-starved by the far field.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "AddedMassResult",
    "DISC_ADDED_MASS_COEFFICIENT",
    "cylinder_added_mass",
    "floe_added_mass_coefficient",
    "speed_ratio_with_added_mass",
]

#: Added-mass coefficient of a circular section in an unbounded fluid
#: (Lamb 1932, §71): the added mass equals the displaced mass, ``C_m = 1``.
DISC_ADDED_MASS_COEFFICIENT = 1.0

_BOUNDARIES = ("unbounded", "rigid_wall", "free_surface")


@dataclass(frozen=True)
class AddedMassResult:
    """Outcome of one potential-flow added-mass solve."""

    coefficient: float          # M / (rho pi a^2)
    direction: str              # translation axis the solve was done for
    kinetic_energy: float       # K / rho, in units of a^2 (dimensionless)
    boundary: str
    gap_radii: float | None     # wall/surface distance in radii ("unbounded": None)
    cells_per_radius: int
    iterations: int
    residual: float
    grid: tuple[int, int]
    tail_fraction: float        # share of K that came from the analytic tail

    def as_dict(self) -> dict:
        return {
            "coefficient": self.coefficient,
            "direction": self.direction,
            "kinetic_energy_over_rho": self.kinetic_energy,
            "boundary": self.boundary,
            "gap_radii": self.gap_radii,
            "cells_per_radius": self.cells_per_radius,
            "iterations": self.iterations,
            "residual": self.residual,
            "grid": list(self.grid),
            "tail_fraction": self.tail_fraction,
        }


def _dipole(x: torch.Tensor, y: torch.Tensor, xc: float, yc: float,
            radius: float, weight: float, axis: str = "x") -> torch.Tensor:
    """Decaying dipole potential ``-weight * a^2 (e . (r - r_c)) / r^2``."""
    dx = x - xc
    dy = y - yc
    r2 = dx.square() + dy.square()
    r2 = torch.clamp(r2, min=(0.5 * radius) ** 2)
    offset = dx if axis == "x" else dy
    return -weight * radius * radius * offset / r2


def cylinder_added_mass(radius: float = 1.0, cells_per_radius: int = 32,
                        box_radii: float = 16.0, boundary: str = "unbounded",
                        direction: str = "x", gap: float = 0.5, omega: float = 1.85,
                        tolerance: float = 1e-5, max_iterations: int = 40_000,
                        initial: str = "dipole",
                        device: str | torch.device = "cpu") -> AddedMassResult:
    """Solve the potential flow around a translating circular section.

    Parameters
    ----------
    radius, cells_per_radius, box_radii
        Disc radius ``a``, grid resolution ``a/dx``, and box extent in radii.
    boundary
        ``"unbounded"``, ``"rigid_wall"`` or ``"free_surface"``.
    direction
        Translation axis, ``"x"`` (parallel to the plane) or ``"y"``
        (perpendicular). A linearised free surface is *not* isotropic: with
        ``phi = 0`` there, horizontal motion sees an almost rigid boundary
        (``u_x = dphi/dx = 0``) while vertical motion sees a pressure-release
        surface, and the two added masses differ.
    gap
        Distance from the disc surface to the wall / free surface, in radii.
    initial
        ``"dipole"`` starts from the free-space dipole (the exact solution of
        the unbounded problem, so it only *verifies* the quadrature), while
        ``"zero"`` starts from a motionless fluid and lets the relaxation find
        the field -- that is the variant which verifies solver, boundary
        conditions and energy integral together.
    """
    if boundary not in _BOUNDARIES:
        raise ValueError(f"boundary must be one of {_BOUNDARIES}, got {boundary!r}")
    if radius <= 0 or cells_per_radius < 4 or box_radii <= 1.0:
        raise ValueError("radius > 0, cells_per_radius >= 4 and box_radii > 1 required")
    if gap < 0 or not math.isfinite(gap):
        raise ValueError("gap must be finite and non-negative")
    if initial not in ("dipole", "zero"):
        raise ValueError("initial must be 'dipole' or 'zero'")
    if direction not in ("x", "y"):
        raise ValueError("direction must be 'x' or 'y'")

    a = float(radius)
    dx = a / float(cells_per_radius)
    half_width = box_radii * a
    nx = int(round(2.0 * half_width / dx))
    if nx < 16 or nx > 1024:
        raise ValueError(f"grid width {nx} out of range (16..1024); adjust the inputs")

    dev = torch.device(device)
    dtype = torch.float64

    # Vertical layout: the wall/plane sits on y = 0; the disc is lifted/sunk by
    # a + gap*a from it. The unbounded case centres the disc.
    if boundary == "unbounded":
        yc = 0.0
        y0, y1 = -half_width, half_width
    elif boundary == "rigid_wall":
        yc = a + gap * a
        y0, y1 = 0.0, yc + half_width
    else:  # free_surface -- the plane is *above* the disc
        yc = -(a + gap * a)
        y0, y1 = yc - half_width, 0.0
    ny = int(round((y1 - y0) / dx))

    xs = -half_width + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    ys = y0 + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dx
    xgrid = xs.unsqueeze(0).expand(ny, nx)
    ygrid = ys.unsqueeze(1).expand(ny, nx)

    # Sub-sampled area fraction of every cell that lies inside the disc. The
    # solver fixes cells whose centre is covered (``>= 0.5``), while the energy
    # integral weights each fluid cell by the part of it that is really wet:
    # without that weight the half-cell annulus of fluid hugging the boundary is
    # lost and the coefficient comes out low by ~1/cells_per_radius.
    sub = 4
    inside = torch.zeros((ny, nx), dtype=dtype, device=dev)
    for sy in range(sub):
        oy = (sy + 0.5) / sub - 0.5
        for sx in range(sub):
            ox = (sx + 0.5) / sub - 0.5
            xx = xgrid + ox * dx
            yy = ygrid + oy * dx
            inside = inside + ((xx.square() + (yy - yc).square()) <= a * a).to(dtype)
    wet_fraction = 1.0 - inside / (sub * sub)
    body = wet_fraction <= 0.5 + 1e-12
    phi_far = _dipole(xgrid, ygrid, 0.0, yc, a, 1.0, axis=direction)
    if boundary != "unbounded":
        # Mirror about the plane y = 0. An x-dipole is even under y -> -y, a
        # y-dipole is odd, so the "even" (rigid-wall) weight is +1 or -1; the
        # free surface takes the opposite sign (phi = 0 on the plane).
        even_weight = 1.0 if direction == "x" else -1.0
        mirror_weight = even_weight if boundary == "rigid_wall" else -even_weight
        phi_far = phi_far + _dipole(xgrid, ygrid, 0.0, -yc, a, mirror_weight,
                                    axis=direction)

    rows = torch.arange(ny, device=dev).unsqueeze(1).expand(ny, nx)
    cols = torch.arange(nx, device=dev).unsqueeze(0).expand(ny, nx)
    outer = (rows == 0) | (rows == ny - 1) | (cols == 0) | (cols == nx - 1)
    if boundary != "unbounded":
        # The wall/free-surface row must *not* carry the far-field value: on a
        # rigid wall it is solved together with its mirror ghost, and on a free
        # surface it is held at phi = 0.
        if boundary == "rigid_wall":
            outer[0, :] = False
        else:
            outer[-1, :] = False
    fixed = body | outer

    body_value = -xgrid if direction == "x" else -(ygrid - yc)
    if initial == "dipole":
        phi = torch.where(body, body_value, phi_far)
    else:
        phi = torch.zeros((ny, nx), dtype=dtype, device=dev)
    phi[body] = body_value[body]
    phi[outer] = phi_far[outer]
    if boundary == "free_surface":
        phi[-1, :] = 0.0

    flat = phi.reshape(-1)
    free = ~fixed.reshape(-1)
    parity = ((rows + cols) % 2).reshape(-1)

    plane_row = 0 if boundary == "rigid_wall" else ny - 1
    mirror_sign = 1.0 if boundary == "rigid_wall" else -1.0
    padded = torch.zeros((ny + 2, nx + 2), dtype=dtype, device=dev)

    def _apply_ghost(padded_view: torch.Tensor) -> None:
        padded_view[1:-1, 1:-1] = phi
        padded_view[:, 0] = padded_view[:, 1]
        padded_view[:, -1] = padded_view[:, -2]
        padded_view[0, :] = padded_view[1, :]
        padded_view[-1, :] = padded_view[-2, :]
        if boundary != "unbounded":
            ghost = mirror_sign * phi[plane_row]
            if plane_row == 0:
                padded_view[0, 1:-1] = ghost
                padded_view[0, 0] = ghost[0]
                padded_view[0, -1] = ghost[-1]
            else:
                padded_view[-1, 1:-1] = ghost
                padded_view[-1, 0] = ghost[0]
                padded_view[-1, -1] = ghost[-1]

    residual = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        for p in (0, 1):
            updatable = free & (parity == p)
            if not bool(updatable.any()):
                continue
            _apply_ghost(padded)
            avg = 0.25 * (padded[0:-2, 1:-1] + padded[2:, 1:-1]
                          + padded[1:-1, 0:-2] + padded[1:-1, 2:])
            current = flat[updatable]
            target = avg.reshape(-1)[updatable]
            delta = torch.clamp(omega * (target - current), min=-1.0, max=1.0)
            flat[updatable] = current + delta
        _apply_ghost(padded)
        lap = (padded[0:-2, 1:-1] + padded[2:, 1:-1]
               + padded[1:-1, 0:-2] + padded[1:-1, 2:] - 4.0 * phi)
        residual = float(lap.reshape(-1)[free].abs().max())
        if residual < tolerance:
            break

    # Kinetic energy of the flow on fluid nodes. Inside the disc the field is
    # the rigid-body potential (harmless for the mask), and the far ring uses
    # the analytic values, so no spurious boundary layer enters the integral.
    phi_eval = phi.clone()
    phi_eval[body] = body_value[body]
    grad_x = torch.zeros_like(phi_eval)
    grad_y = torch.zeros_like(phi_eval)
    grad_x[:, 1:-1] = (phi_eval[:, 2:] - phi_eval[:, :-2]) / (2.0 * dx)
    grad_x[:, 0] = (phi_eval[:, 1] - phi_eval[:, 0]) / dx
    grad_x[:, -1] = (phi_eval[:, -1] - phi_eval[:, -2]) / dx
    grad_y[1:-1, :] = (phi_eval[2:, :] - phi_eval[:-2, :]) / (2.0 * dx)
    grad_y[0, :] = (phi_eval[1, :] - phi_eval[0, :]) / dx
    grad_y[-1, :] = (phi_eval[-1, :] - phi_eval[-2, :]) / dx
    # A central difference that straddles the body would average the (opposite)
    # normal derivatives of the fluid and the rigid-body potential and return
    # ~0 exactly where the field is strongest. Nodes next to the body therefore
    # use the one-sided difference on the wet side.
    wet_pad = torch.nn.functional.pad(wet_fraction, (1, 1, 1, 1), value=1.0)
    dry_xp = wet_pad[1:-1, 2:] < 0.5
    dry_xm = wet_pad[1:-1, 0:-2] < 0.5
    dry_yp = wet_pad[2:, 1:-1] < 0.5
    dry_ym = wet_pad[0:-2, 1:-1] < 0.5
    left = torch.nn.functional.pad(phi_eval[:, :-1], (1, 0), value=0.0)
    right = torch.nn.functional.pad(phi_eval[:, 1:], (0, 1), value=0.0)
    down = torch.nn.functional.pad(phi_eval[:-1, :], (0, 0, 1, 0), value=0.0)
    up = torch.nn.functional.pad(phi_eval[1:, :], (0, 0, 0, 1), value=0.0)
    gx_right = (right - phi_eval) / dx
    gx_left = (phi_eval - left) / dx
    gy_up = (up - phi_eval) / dx
    gy_down = (phi_eval - down) / dx
    grad_x = torch.where(dry_xm, gx_right, torch.where(dry_xp, gx_left, grad_x))
    grad_y = torch.where(dry_ym, gy_up, torch.where(dry_yp, gy_down, grad_y))
    energy_density = 0.5 * (grad_x.square() + grad_y.square())
    kinetic_box = float((energy_density * wet_fraction).sum()) * dx * dx

    # Analytic dipole tail beyond the box: K_tail/(rho) = pi a^4 / (2 R^2) for a
    # single dipole. A rigid wall doubles the far-field dipole strength and
    # halves the volume, net 2x; the free-surface mirror is a quadrupole whose
    # tail decays as R^-6 and is negligible.
    tail = 0.5 * math.pi * a ** 4 / (half_width ** 2)
    if boundary == "rigid_wall":
        tail *= 2.0
    elif boundary == "free_surface":
        tail = 0.0
    kinetic = kinetic_box + tail

    displaced = math.pi * a * a
    coefficient = 2.0 * kinetic / displaced
    return AddedMassResult(
        coefficient=coefficient,
        direction=direction,
        kinetic_energy=kinetic,
        boundary=boundary,
        gap_radii=None if boundary == "unbounded" else float(gap),
        cells_per_radius=int(cells_per_radius),
        iterations=int(iterations),
        residual=float(residual),
        grid=(ny, nx),
        tail_fraction=float(tail / kinetic) if kinetic else 0.0,
    )


def floe_added_mass_coefficient(submergence: str = "submerged") -> float:
    """Added-mass coefficient recommended for the reduced floe model.

    A thin floe is a circular section per unit depth. ``"submerged"`` selects
    the free-field value, which is the conservative (largest-inertia) choice.
    ``"free_surface"`` returns ``nan``-free placeholder documentation only via
    :func:`cylinder_added_mass`; use it to get a configuration-specific value.
    """
    if submergence not in ("submerged", "free_surface"):
        raise ValueError("submergence must be 'submerged' or 'free_surface'")
    if submergence == "submerged":
        return DISC_ADDED_MASS_COEFFICIENT
    raise ValueError(
        "a free-surface coefficient depends on the draft: call "
        "cylinder_added_mass(boundary='free_surface', gap=...) instead")


def speed_ratio_with_added_mass(rho_ice: float, rho_water: float,
                                coefficient: float = DISC_ADDED_MASS_COEFFICIENT) -> float:
    """``v_wet / v_dry = m_ice / (m_ice + C_m m_displaced)`` for one impulse.

    Both masses scale with the same section area, so only the density ratio
    survives: ``ratio = rho_ice / (rho_ice + C_m rho_water)``.
    """
    if rho_ice <= 0 or rho_water <= 0:
        raise ValueError("densities must be positive")
    if coefficient < 0:
        raise ValueError("added-mass coefficient must be non-negative")
    return rho_ice / (rho_ice + coefficient * rho_water)
