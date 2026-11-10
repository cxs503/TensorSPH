"""Dilute bubbly-phase transport: rise velocity, void fraction, bubble clouds.

Three closures are needed before an aerated free surface can be followed:

1. **Rise velocity.**  ``stokes_rise_velocity`` is exact in the creeping-flow
   limit; ``drag_balance_rise_velocity`` solves the standard force balance
   ``(4/3) pi R^3 (rho_l - rho_g) g = (1/2) C_d rho_l pi R^2 V^2`` with the
   Schiller-Naumann drag of a *rigid* sphere, so it inherits the creeping-flow
   limit exactly (checked to <1% at ``R = 10 um``) but is known to fail by a
   *factor of three to five* once the bubble interface becomes mobile and the
   bubble flattens.  ``mendelson_rise_velocity`` is Mendelson's (1967)
   wave-theory result ``V = sqrt(2 sigma/(rho_l R) + 2 g R)``, which is the
   closure that applies in the ellipsoidal regime ``R = 0.5 .. 5 mm`` covered by
   this campaign.  ``rise_velocity`` switches between the two and blends across
   ``R = 0.2 .. 0.5 mm``; the blend is a declared interpolation, not a
   calibrated law (no bubble-rise dataset is bundled with this package).

2. **Void fraction field.**  ``solve_void_fraction_field`` integrates the
   advective-diffusion equation for the air concentration
   ``dC/dt = D d2C/dy2 + w dC/dy`` with ``y`` the depth below the free surface,
   ``C(0, t) = 1`` (100% air at the surface) and buoyant drift ``w`` directed
   back towards the surface.  The drift-free solution is exactly the
   complementary error function, which is the analytic anchor used by the
   hydraulic-jump benchmark: ``C = erfc(y / (2 sqrt(D t)))`` with the aeration
   layer thickness ``Y_90 = 2 sqrt(2) sqrt(D t)``.  ``analytic_void_fraction``
   is the closed-form solution *with* drift.

3. **Lagrangian bubbles and their SPH estimate.**  ``transport_bubbles``
   advances a bubble cloud with buoyant drift, slip and turbulent diffusion
   (random walk with the same ``D``, i.e. the Fokker-Planck consistent
   discretisation of the Eulerian equation above), reflecting at the free
   surface so the air volume is conserved to round-off.  ``void_fraction_from_bubbles``
   is the SPH summation estimate ``alpha_i = sum_j V_j W(r_ij, h)`` built on the
   package kernel, i.e. the map from Lagrangian bubbles to the Eulerian void
   fraction the momentum equation would see.
"""
from __future__ import annotations

import math

import torch

from ..droplet_impact import cubic_spline_kernel

WATER_20C = {"rho": 998.2, "mu": 1.002e-3, "sigma": 0.0728}
AIR_20C = {"rho": 1.204}

GRAVITY = 9.81

#: Blend window (m) between the rigid-sphere drag balance and Mendelson theory.
RIGID_SPHERE_MAX = 2.0e-4
MOBILE_INTERFACE_MIN = 5.0e-4

SQRT2 = math.sqrt(2.0)


def stokes_rise_velocity(
    radius,
    rho_l: float = WATER_20C["rho"],
    rho_g: float = AIR_20C["rho"],
    mu_l: float = WATER_20C["mu"],
    gravity: float = GRAVITY,
):
    """``V = 2 g R^2 (rho_l - rho_g) / (9 mu_l)`` -- exact for ``Re -> 0``."""
    r = torch.as_tensor(radius, dtype=torch.float64)
    v = 2.0 * gravity * r.square() * (rho_l - rho_g) / (9.0 * mu_l)
    return float(v) if v.ndim == 0 else v


def mendelson_rise_velocity(
    radius,
    rho_l: float = WATER_20C["rho"],
    sigma: float = WATER_20C["sigma"],
    gravity: float = GRAVITY,
):
    """Mendelson (1967) wave-theory rise velocity (ellipsoidal regime)."""
    r = torch.as_tensor(radius, dtype=torch.float64)
    v = torch.sqrt(2.0 * sigma / (rho_l * r) + 2.0 * gravity * r)
    return float(v) if v.ndim == 0 else v


def schiller_naumann_cd(reynolds):
    """Drag coefficient of a rigid sphere, ``Re < ~800``: ``24/Re (1+0.15 Re^0.687)``."""
    re = torch.as_tensor(reynolds, dtype=torch.float64).clamp_min(1.0e-12)
    cd = 24.0 / re * (1.0 + 0.15 * re.pow(0.687))
    return float(cd) if cd.ndim == 0 else cd


def drag_balance_rise_velocity(
    radius,
    rho_l: float = WATER_20C["rho"],
    rho_g: float = AIR_20C["rho"],
    mu_l: float = WATER_20C["mu"],
    gravity: float = GRAVITY,
    iterations: int = 500,
    tolerance: float = 1.0e-12,
):
    """Terminal velocity of a rigid sphere from the drag balance.

    Solved by fixed-point iteration on
    ``V = sqrt(8 g R (rho_l - rho_g) / (3 rho_l C_d(Re)))``.  This is the
    *rigid-sphere* branch: it is exact in the Stokes limit and wrong once the
    bubble interface is mobile, which is why ``rise_velocity`` hands over to
    Mendelson's closure above ``R = 0.5 mm``.
    """
    r = float(radius) if not torch.is_tensor(radius) else float(radius)
    v = float(stokes_rise_velocity(r, rho_l, rho_g, mu_l, gravity))
    for _ in range(iterations):
        re = 2.0 * r * v * rho_l / mu_l
        cd = float(schiller_naumann_cd(re))
        v_new = math.sqrt(8.0 * gravity * r * (rho_l - rho_g) / (3.0 * rho_l * cd))
        if abs(v_new - v) <= tolerance * max(1.0e-30, v_new):
            v = v_new
            break
        v = 0.5 * (v + v_new)
    return v


def rise_velocity(
    radius,
    rho_l: float = WATER_20C["rho"],
    rho_g: float = AIR_20C["rho"],
    mu_l: float = WATER_20C["mu"],
    sigma: float = WATER_20C["sigma"],
    gravity: float = GRAVITY,
):
    """Rise velocity across regimes, vectorised over ``radius``.

    ``R <= 0.2 mm`` rigid-sphere drag balance; ``R >= 0.5 mm`` Mendelson wave
    theory; linear blend in between (declared interpolation).
    """
    r = torch.as_tensor(radius, dtype=torch.float64)
    scalar = r.ndim == 0
    flat = r.reshape(-1)
    out = torch.empty_like(flat)
    for index, value in enumerate(flat.tolist()):
        v_sn = drag_balance_rise_velocity(value, rho_l, rho_g, mu_l, gravity)
        v_m = float(mendelson_rise_velocity(value, rho_l, sigma, gravity))
        if value <= RIGID_SPHERE_MAX:
            out[index] = v_sn
        elif value >= MOBILE_INTERFACE_MIN:
            out[index] = v_m
        else:
            weight = (value - RIGID_SPHERE_MAX) / (MOBILE_INTERFACE_MIN - RIGID_SPHERE_MAX)
            out[index] = (1.0 - weight) * v_sn + weight * v_m
    out = out.reshape(r.shape)
    return float(out) if scalar else out


# --------------------------------------------------------------------------- #
# Eulerian void fraction
# --------------------------------------------------------------------------- #

def analytic_void_fraction(y, transit_time: float, diffusivity: float, drift: float = 0.0):
    """Closed-form air concentration below a surface held at ``C = 1``.

    Solves ``dC/dt = D C_yy + w C_y`` on ``y > 0`` with ``C(0, t) = 1``,
    ``C(y, 0) = 0`` and ``w`` the buoyant drift directed back towards the
    surface.  With ``w = 0`` this collapses to ``C = erfc(y / (2 sqrt(D t)))``,
    which is the erf anchor of the campaign.
    """
    yy = torch.as_tensor(y, dtype=torch.float64)
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    scale = 2.0 * math.sqrt(diffusivity * transit_time)
    w = float(drift)
    first = torch.erfc((yy + w * transit_time) / scale)
    if w == 0.0:
        profile = first
    else:
        second = torch.exp(-w * yy / diffusivity) * torch.erfc(
            (yy - w * transit_time) / scale
        )
        profile = 0.5 * (first + second)
    return profile


def erf_anchor(y, y90):
    """Campaign anchor: ``C(y) = 1 - erf(y / (Y_90 / sqrt(2)))``."""
    yy = torch.as_tensor(y, dtype=torch.float64)
    return 1.0 - torch.erf(yy / (float(y90) / SQRT2))


def y90_for_diffusion(transit_time: float, diffusivity: float) -> float:
    """``Y_90`` that makes the erf anchor coincide with the diffusion solution."""
    return 2.0 * SQRT2 * math.sqrt(diffusivity * transit_time)


def solve_void_fraction_field(
    depth: float = 2.0,
    cells: int = 400,
    diffusivity: float = 0.02,
    drift: float = 0.0,
    transit_time: float = 4.0,
    dt: float | None = None,
    surface_value: float = 1.0,
):
    """Finite-volume integration of the air-concentration equation.

    Returns ``(y, C)`` with ``y`` the depth of each cell centre below the free
    surface.  Cell 0 is a Dirichlet node (``C = surface_value``); the bottom
    ghost carries ``C = 0``.
    """
    if cells < 4:
        raise ValueError("need at least four cells")
    dy = depth / cells
    if dt is None:
        dt_diff = 0.4 * dy * dy / diffusivity
        dt_drift = 0.4 * dy / drift if drift > 0.0 else math.inf
        dt = min(dt_diff, dt_drift)
    steps = max(1, int(round(transit_time / dt)))
    step = transit_time / steps

    y = (torch.arange(cells, dtype=torch.float64) + 0.5) * dy
    c = torch.zeros(cells, dtype=torch.float64)
    c[0] = surface_value

    for _ in range(steps):
        # Downward (+) face fluxes between interior cells, upwinded on the
        # bubble drift (which points to -y).
        flux = -diffusivity * (c[1:] - c[:-1]) / dy - drift * c[1:]
        bottom = diffusivity * c[-1] / dy
        delta = torch.empty_like(c)
        delta[0] = 0.0
        delta[1:-1] = -(flux[1:] - flux[:-1]) / dy
        delta[-1] = -(bottom - flux[-1]) / dy
        c = c + step * delta
        c[0] = surface_value
    return y, c


# --------------------------------------------------------------------------- #
# Lagrangian bubbles
# --------------------------------------------------------------------------- #

def transport_bubbles(
    depths,
    steps: int,
    dt: float,
    drift: float,
    diffusivity: float,
    generator: torch.Generator | None = None,
    surface: str = "reflect",
):
    """Advance bubble depths with buoyant drift + turbulent random walk.

    ``surface="reflect"`` folds bubbles that reach the free surface back into the
    water (zero net air flux through the surface, so the cloud's air volume is
    conserved to round-off); ``surface="escape"`` removes them and returns the
    escaped mask so the volume bookkeeping stays auditable.
    """
    y = torch.as_tensor(depths, dtype=torch.float64).clone()
    sigma = math.sqrt(2.0 * diffusivity * dt)
    escaped_any = torch.zeros_like(y, dtype=torch.bool)
    for _ in range(steps):
        noise = torch.randn(y.shape, dtype=torch.float64, generator=generator)
        y = y - drift * dt + sigma * noise
        if surface == "reflect":
            y = y.abs()
        else:
            escaped_any |= y < 0.0
            y = y.clamp_min(0.0)
    return y, escaped_any


def void_fraction_histogram(depths, edges):
    """Histogram of bubble depths, normalised so that ``sum(C) * dy = 1``."""
    d = torch.as_tensor(depths, dtype=torch.float64)
    e = torch.as_tensor(edges, dtype=torch.float64)
    index = torch.bucketize(d, e) - 1
    counts = torch.zeros(e.numel() - 1, dtype=torch.float64)
    valid = (index >= 0) & (index < counts.numel())
    counts.index_add_(0, index[valid], torch.ones(int(valid.sum()), dtype=torch.float64))
    if counts.sum() > 0:
        counts = counts / counts.sum()
    dy = float(e[1] - e[0])
    return counts / dy


def void_fraction_from_bubbles(bubble_pos, bubble_vol, fluid_pos, h: float):
    """SPH summation estimate of the void fraction at the fluid particles.

    ``alpha_i = sum_j V_j W(r_ij, h)``.  For a cloud filling the support of the
    kernel the sum tends to the bubble volume fraction, which is what the
    momentum equation consumes; the two-way mapping is checked by the
    conservation benchmark.
    """
    bubbles = torch.as_tensor(bubble_pos, dtype=torch.float64)
    volumes = torch.as_tensor(bubble_vol, dtype=torch.float64).reshape(-1)
    fluid = torch.as_tensor(fluid_pos, dtype=torch.float64)
    if bubbles.shape[0] != volumes.numel():
        raise ValueError("bubble_pos and bubble_vol disagree on the bubble count")
    delta = fluid[:, None, :] - bubbles[None, :, :]
    r = torch.linalg.norm(delta, dim=-1)
    weights = cubic_spline_kernel(r, h)
    return (weights * volumes[None, :]).sum(dim=1)


def l2_relative_error(profile, reference):
    """``||a - b||_2 / ||b||_2`` over the sampled points."""
    a = torch.as_tensor(profile, dtype=torch.float64)
    b = torch.as_tensor(reference, dtype=torch.float64)
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))
