"""Brine drainage: gravity drainage through the brine pore network.

Model
-----
Brine is a *mobile* phase only above the irreducible saturation
``v_irr`` (standard porous-media residual saturation, 0.02-0.05 for a
water-wet matrix).  The mobile fraction and its relative permeability are

    s_m = clamp((v_b - v_irr) / (1 - v_irr), 0, 1)
    k_r = s_m^3                              (Brooks-Corey, exponent 3)
    w   = k_r * k(v_b) * (rho_b - rho_i) g / mu_b        (downward, m/s)

with ``k(v_b)`` the Golden-Ackley-Lytle permeability of
:mod:`tensorsph.ice_thermal.properties`.  Salt is carried with the brine, so
the bulk salinity obeys the conservative balance

    d(rho_i S)/dt = -dF/dz ,        F = v_b rho_b S_b w     [g salt/(m^2 s)]

Everything in ``w`` is measured or textbook -- ``v_irr`` is the only
porous-media constant and it has a narrow, standard range; the model contains
**no calibrated rate coefficient**.  Two consequences follow and both are
checked in ``tests/test_ice_thermal_brine.py``:

* ;math:`k_r = s_m^3` is what makes drainage slow: at ``v_b = 0.06``,
  ``s_m = 0.031``, ``k_r = 3e-5``, so ``w ~ 1e-7 m/s`` -- a 0.5 m layer
  desalinates in ``O(1 month)``, matching the observed desalination timescale
  (and *not* the ``O(minute)`` a saturated Darcy flux would give);
* drainage stops at ``v_b = v_irr``, so the model has an analytic equilibrium

      S_eq(T) = v_irr rho_b(T) S_b(T) / rho_i

  which is a steady state of the discrete scheme for *any* resolution -- an
  exact, machine-checkable anchor.

Literature anchor (honest status)
---------------------------------
The campaign node asks for the brine-drainage experiment of Greeley & Wilson
(1987).  That primary data table is **not reachable from this sandbox**, so it
is registered as ``WITHHELD`` and no digit of it is reproduced here.  What is
used instead is the *band* of first-year sea-ice bulk salinity from the
standard review of Cox & Weeks (1983): 1-year ice sits at 3-8 psu against a
32-34 psu ocean, i.e. a desalination ratio of 0.09-0.25.  The band check is
what :func:`band_check` automates; it is weaker evidence than a point-by-point
comparison and is labelled as such in the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import properties as props

__all__ = [
    "V_IRR",
    "SEA_ICE_FIRST_YEAR_BAND",
    "LITERATURE_ANCHORS",
    "mobile_brine_fraction",
    "relative_permeability",
    "drainage_velocity",
    "residual_salinity",
    "salt_flux",
    "drainage_step",
    "drainage_step_implicit",
    "drainage_advance",
    "DesalinationResult",
    "run_desalination",
    "band_check",
]

#: irreducible brine saturation (fraction of pore space), textbook 0.02-0.05
V_IRR = 0.03

#: first-year sea-ice bulk salinity band (psu) and the ocean it grew from
SEA_ICE_FIRST_YEAR_BAND = (3.0, 8.0)
OCEAN_SALINITY = 32.0

LITERATURE_ANCHORS = {
    "first_year_ice_salinity_band_psu": SEA_ICE_FIRST_YEAR_BAND,
    "ocean_salinity_psu": OCEAN_SALINITY,
    "source": "Cox & Weeks (1983) review of sea-ice salinity; band only",
    "withheld": "Greeley & Wilson (1987) brine-drainage data table: "
                "not accessible offline, anchor registered WITHHELD",
}


def mobile_brine_fraction(v_b, v_irr=V_IRR):
    """Mobile brine saturation ``s_m`` -- zero below the irreducible saturation."""
    v = torch.as_tensor(v_b, dtype=torch.float64)
    return ((v - v_irr) / (1.0 - v_irr)).clamp(0.0, 1.0)


def relative_permeability(s_m, exponent=3.0):
    """Brooks-Corey relative permeability ``s_m^exponent``."""
    return torch.as_tensor(s_m, dtype=torch.float64).clamp(0.0, 1.0) ** exponent


def drainage_velocity(temperature, salinity, v_irr=V_IRR, exponent=3.0,
                      brine_salinity=None):
    """Downward Darcy drainage velocity (m/s, positive downward) of the brine.

    Returns ``(w, v_b, S_b)``: the velocity, the brine volume fraction and the
    brine salinity of every particle (all torch, float64).
    """
    t = torch.as_tensor(temperature, dtype=torch.float64)
    s = torch.as_tensor(salinity, dtype=torch.float64)
    t, s = torch.broadcast_tensors(t, s)
    tf = props.freezing_point_vec(s)
    # S_b depends on T only: the caller may pass it in, which removes the
    # dominant cost of the drainage sub-cycling (a 14-step liquidus inversion
    # per particle per sub-step)
    sb = props.brine_salinity_vec(t) if brine_salinity is None else brine_salinity
    rb = 1000.0 + 0.8 * sb
    v_b = torch.where(s <= 0.0, torch.zeros_like(s),
                      (props.RHO_ICE * s / (rb * sb.clamp_min(1e-9))).clamp(0.0, 1.0))
    v_b = torch.where(t >= tf, torch.ones_like(v_b), v_b)
    k_matrix = props.permeability_vec(v_b)
    k_r = relative_permeability(mobile_brine_fraction(v_b, v_irr), exponent)
    w = k_r * k_matrix * (rb - props.RHO_ICE) * props.GRAVITY / props.MU_BRINE
    return w, v_b, sb


def residual_salinity(temperature, v_irr=V_IRR):
    """Equilibrium bulk salinity (psu) at which drainage stops: ``v_b = v_irr``."""
    t = torch.as_tensor(temperature, dtype=torch.float64)
    sb = props.brine_salinity_vec(t)
    rb = 1000.0 + 0.8 * sb
    return v_irr * rb * sb / props.RHO_ICE


def salt_flux(temperature, salinity, v_irr=V_IRR, brine_salinity=None):
    """Downward brine salt flux ``F = v_b rho_b S_b w`` [g salt /(m^2 s)]."""
    w, v_b, sb = drainage_velocity(temperature, salinity, v_irr,
                                   brine_salinity=brine_salinity)
    rb = 1000.0 + 0.8 * sb
    return v_b * rb * sb * w


def drainage_step(salinity, temperature, dz, dt, v_irr=V_IRR, brine_salinity=None):
    """One explicit (upwind) conservative drainage step of the bulk salinity.

    ``salinity``/``temperature`` are ``(n,)`` with index 0 at the top and
    ``n-1`` at the bottom (growth interface).  Salt leaves the bottom face only;
    the returned ``rejected`` is the salt mass flux (g/(m^2 s)) leaving the ice,
    so ``(sum(rho_i S dz) + dt*rejected)`` is conserved to round-off.
    """
    s = torch.as_tensor(salinity, dtype=torch.float64).clone()
    t = torch.as_tensor(temperature, dtype=torch.float64)
    flux_faces = torch.zeros_like(s)          # flux on face below particle i
    f_particle = salt_flux(t, s, v_irr, brine_salinity=brine_salinity)
    # Availability limiter: only the salt in excess of the irreducible-brine
    # equilibrium can leave a cell, F <= rho_i dz (S - S_eq)/dt.  Without it a
    # coarse sub-step drains a cell *through* the equilibrium floor (the
    # relative permeability only ever approaches zero, it never reaches it in
    # one finite step) and the column silently over-desalinates.
    s_mobile = (s - residual_salinity(t, v_irr)).clamp_min(0.0)
    f_particle = torch.minimum(f_particle, props.RHO_ICE * dz * s_mobile / max(dt, 1e-9))
    flux_faces[:-1] = f_particle[:-1]
    flux_faces[-1] = f_particle[-1]           # last particle exits the ice bottom
    # dS/dt = -(1/rho_i) dF/dz ; F positive downward, z index increasing downward
    div = torch.zeros_like(s)
    div[0] = flux_faces[0] / dz              # top face: no influx (sea ice top is impermeable)
    div[1:] = (flux_faces[1:] - flux_faces[:-1]) / dz
    s_new = s - dt * div / props.RHO_ICE
    # a face cannot remove more salt than the particle holds (positivity limiter)
    s_new = torch.clamp(s_new, min=0.0)
    lost = float((props.RHO_ICE * dz * (s - s_new)).sum())  # g/m^2 actually removed
    return s_new, lost / dt


def drainage_step_implicit(salinity, temperature, dz, dt, v_irr=V_IRR, picard=3):
    """Backward-Euler (upwind) drainage step -- stable at any ``dt``.

    The explicit operator of :func:`drainage_step` is limited by
    ``dt < dz / w``; with ``w ~ 1e-4 m/s`` early in a run that is a few tens of
    seconds, which is far below the hours a 45 d integration wants.  Linearising
    the face flux as ``F_i = a_i S_i`` with ``a_i = F_i / S_i`` (Picard over
    ``a``) turns the balance into a bidiagonal implicit system that is solved
    directly.  It reduces to the explicit scheme as ``dt -> 0`` (checked in
    ``tests/test_ice_thermal_brine.py``) and conserves salt exactly: only the
    bottom face leaves the ice.
    """
    s = torch.as_tensor(salinity, dtype=torch.float64).clone()
    t = torch.as_tensor(temperature, dtype=torch.float64)
    s_old = s.clone()
    n = s.numel()
    for _ in range(picard):
        f = salt_flux(t, s, v_irr)
        a = torch.where(s > 1e-9, f / s.clamp_min(1e-9), torch.zeros_like(s))
        mat = torch.eye(n, dtype=torch.float64) * (1.0 + dt * a / (props.RHO_ICE * dz))
        if n > 1:
            idx = torch.arange(1, n)
            mat[idx, idx - 1] = -dt * a[:-1] / (props.RHO_ICE * dz)
        s = torch.linalg.solve(mat, s_old).clamp_min(0.0)
    flux_out = float(a[-1] * s[-1])
    return s, flux_out


def drainage_advance(salinity, temperature, dz, dt, v_irr=V_IRR, cfl=0.3, max_sub=48):
    """Advance the drainage by ``dt`` with CFL-limited explicit sub-steps.

    The Darcy velocity early in a run is ``O(1e-5 m/s)``, so ``dt = 900 s`` is
    ``w dt / dz ~ 1``: a single step would advect the whole column and (worse)
    drop the bulk salinity *below* the irreducible-brine equilibrium, which is
    physically impossible.  Sub-cycling at ``cfl = 0.3`` keeps every step inside
    the stability region and lets the ``k_r -> 0`` limit stop the drainage at
    ``v_b = v_irr`` by itself.  Returns ``(salinity, removed_g_per_m2, n_sub)``.
    """
    s = torch.as_tensor(salinity, dtype=torch.float64).clone()
    t = torch.as_tensor(temperature, dtype=torch.float64)
    sb = props.brine_salinity_vec(t)
    n_sub = 1
    w_max = 0.0
    if s.numel() > 1:
        # the face flux below particle i is carried by particle i, so the CFL
        # limit is the *particle* maximum (a face average under-estimates it by
        # up to 2x and that is enough to destabilise the upwind scheme)
        w_max = float(drainage_velocity(t, s, v_irr, brine_salinity=sb)[0].max())
    if w_max > 0.0:
        n_sub = int(min(max_sub, max(1, math.ceil(w_max * dt / (cfl * dz)))))
    removed = 0.0
    for _ in range(n_sub):
        s, flux = drainage_step(s, t, dz, dt / n_sub, v_irr, brine_salinity=sb)
        removed += flux * dt / n_sub
    return s, removed, n_sub


@dataclass
class DesalinationResult:
    """Time series of a cooled sea-ice column undergoing gravity drainage."""

    time: list = field(default_factory=list)
    bulk_salinity: list = field(default_factory=list)
    residual: list = field(default_factory=list)
    profile_t: list = field(default_factory=list)
    profile_s: list = field(default_factory=list)
    profile_z: list = field(default_factory=list)
    salt_removed: float = 0.0
    salt_error: float = 0.0
    dz: float = 0.0
    steps: int = 0
    dt: float = 0.0


def run_desalination(thickness=0.5, n=25, s0=12.0, t_air=263.15, t_initial=271.65,
                     days=60.0, v_irr=V_IRR, theta=1.0, profile_every=24):
    """Cool a sea-ice slab from the top and drain brine by gravity.

    The temperature field is a quasi-steady conductive profile between the
    prescribed surface temperature and the freezing point at the brine
    interface (the thermal relaxation time of a 0.5 m ice sheet,
    ``h^2/alpha ~ 2.2e5 s ~ 2.5 d``, is short against the 60 d drainage window,
    so the thermal field is not the slow variable -- the desalination is).
    ``theta`` scales the drainage velocity and is used only as a sensitivity
    parameter (its default 1 means "no calibration").
    """
    dz = thickness / n
    z = (torch.arange(n, dtype=torch.float64) + 0.5) * dz   # 0 = top
    s = torch.full((n,), float(s0), dtype=torch.float64)
    t = torch.linspace(float(t_air), float(t_initial), n, dtype=torch.float64)
    dt = min(3600.0, 0.4 * dz / 1e-6)
    steps = int(math.ceil(days * 86400.0 / dt))
    res = DesalinationResult(dz=dz, steps=steps, dt=dt)
    mass0 = float((props.RHO_ICE * s * dz).sum())
    removed = 0.0
    for k in range(steps + 1):
        if k % profile_every == 0:
            res.time.append(k * dt)
            res.bulk_salinity.append(float(s.mean()))
            res.residual.append(float(residual_salinity(t).mean()))
            res.profile_t.append([float(x) for x in t])
            res.profile_s.append([float(x) for x in s])
            res.profile_z.append([float(x) for x in z])
        if k == steps:
            break
        s, lost, _ = drainage_advance(s, t, dz, dt * theta, v_irr)
        removed += lost
    res.salt_removed = removed
    mass1 = float((props.RHO_ICE * s * dz).sum())
    res.salt_error = abs((mass0 - mass1) - removed) / max(mass0, 1e-30)
    return res


def band_check(bulk_salinity, band=SEA_ICE_FIRST_YEAR_BAND):
    """Is the modelled bulk salinity inside the observed first-year-ice band?"""
    lo, hi = band
    return {"value_psu": float(bulk_salinity), "band_psu": [lo, hi],
            "inside_band": bool(lo <= bulk_salinity <= hi),
            "ratio_to_ocean": float(bulk_salinity) / OCEAN_SALINITY}
