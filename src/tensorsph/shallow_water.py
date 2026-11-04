"""1-D shallow-water (Saint-Venant) dam break over a dry bed, as a reference.

Why this module exists
---------------------
The B2 benchmark compares the SPH leading edge with **Ritter's** solution, which
assumes a *semi-infinite* reservoir.  The case it actually simulates is a
*finite* column of width ``a`` and height ``H = 2a`` inside a tank, so Ritter's
``Z/a = 2 tau`` is not the right reference once the reservoir starts to drain --
and after ``t > a / c0`` the rarefaction has reached the back wall, so the
self-similar solution is void anyway (``tensorsph.ritter.validity_window``).
The repository had no second reference to tell "the solver is wrong" apart from
"Ritter does not apply", which is why the SPH ``front/ritter`` ratio has been
sitting at a systematic -27 % with no way to attribute it.

This module supplies that second reference for the *same idealisation*
(depth-averaged, inviscid, hydrostatic, dry bed, frictionless walls) but with
the finite extent of the reservoir retained.  No closed form exists for that
case, so the reference is a numerical solution -- and a numerical reference is
only worth something if its own code is verified first.  Hence:

* :func:`dam_break` is a finite-volume solver (HLL fluxes with Einfeldt wave
  speeds, MUSCL second order in space with a minmod limiter and Hancock
  half-step predictor, reflective walls, positivity clamping, dry-state guard);
* :func:`front_position` is a *threshold* leading-edge estimator, and the same
  estimator is applied to the exact profile when the error is measured, so the
  estimator bias cancels instead of hiding inside the error;
* ``tests/test_shallow_water_reference.py`` requires the code to reproduce the
  exact Ritter solution in the semi-infinite limit before any finite-reservoir
  number is quoted.

Scope, stated up front: shallow water is not valid for ``H/a`` of order one, and
B2's column has ``H/a = 2``.  This reference is therefore **not** physical
validation of B2 against the 1952 experiment; it is a same-idealisation
reference that separates "Ritter no longer applies" from "the solver is wrong".
See ``docs/benchmarks/dam-break-finite-reservoir-reference.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

GRAVITY = 9.81

#: Relative depth used by :func:`front_position`; the estimator runs at two
#: levels so that the leading edge is reported as a band instead of a single
#: number that hides its own resolution sensitivity.
FRONT_LEVELS = (1.0e-3, 1.0e-2)

#: Cells below this depth are treated as dry (no momentum, no wave speed).
DRY_DEPTH = 1.0e-12


@dataclass
class ShallowWaterRun:
    """Result of :func:`dam_break`."""

    times: list[float] = field(default_factory=list)
    fronts: list[dict] = field(default_factory=list)
    profiles: list[tuple] = field(default_factory=list)
    cells: int = 0
    dx: float = 0.0
    mass_error: float = 0.0
    steps: int = 0
    cfl: float = 0.0
    order: int = 2
    extra: dict = field(default_factory=dict)

    def front_curve(self, level: float = FRONT_LEVELS[0]):
        """``(times, fronts)`` for one threshold level, lab frame."""
        values = [row[level] for row in self.fronts]
        return list(self.times), values

    def dimensionless(self, column_width: float, column_height: float,
                      gravity: float = GRAVITY, level: float = FRONT_LEVELS[0]):
        """``tau`` and ``Z/a`` with ``Z`` measured from the initial column face.

        ``tau = t sqrt(2 g / a)`` and ``Z/a = (x_front - a) / a``: the same
        non-dimensionalisation ``benchmarks/dam_break/run.py`` reports, so the
        two curves overlay without a frame conversion.
        """
        scale = math.sqrt(2.0 * gravity / column_width)
        return ([t * scale for t in self.times],
                [(z - column_width) / column_width
                 for z in self.front_curve(level)[1]])


def _states(h: torch.Tensor, hu: torch.Tensor, g: float):
    """Primitive variables with the dry-state guard applied."""
    wet = h > DRY_DEPTH
    u = torch.where(wet, hu / torch.where(wet, h, torch.ones_like(h)), torch.zeros_like(h))
    c = torch.sqrt(g * h.clamp_min(0.0))
    return u, c, wet


def _physical_flux(h: torch.Tensor, u: torch.Tensor, g: float) -> torch.Tensor:
    """Conservative flux ``(h u, h u^2 + g h^2 / 2)`` for primitive fields."""
    hu = h * u
    return torch.stack((hu, hu * u + 0.5 * g * h * h))


def _hll(hL, uL, hR, uR, g) -> torch.Tensor:
    """HLL flux for interface states, with Einfeldt (HLLE) wave speeds."""
    cL = torch.sqrt(g * hL.clamp_min(0.0))
    cR = torch.sqrt(g * hR.clamp_min(0.0))
    root_sum = (torch.sqrt(hL) + torch.sqrt(hR)).clamp_min(DRY_DEPTH)
    u_hat = (torch.sqrt(hL) * uL + torch.sqrt(hR) * uR) / root_sum
    c_hat = torch.sqrt(g * (0.5 * (hL + hR)).clamp_min(0.0))
    sL = torch.minimum(uL - cL, u_hat - c_hat)
    sR = torch.maximum(uR + cR, u_hat + c_hat)

    flux_L = _physical_flux(hL, uL, g)
    flux_R = _physical_flux(hR, uR, g)
    u_left = torch.stack((hL, hL * uL))
    u_right = torch.stack((hR, hR * uR))

    denom = sR - sL
    safe = denom.abs() > DRY_DEPTH
    denom = torch.where(safe, denom, torch.ones_like(denom))
    hll = (sR * flux_L - sL * flux_R + sL * sR * (u_right - u_left)) / denom

    flux = torch.where((sL >= 0.0).unsqueeze(0), flux_L, hll)
    flux = torch.where((sR <= 0.0).unsqueeze(0), flux_R, flux)
    both_dry = (hL <= DRY_DEPTH) & (hR <= DRY_DEPTH)
    return torch.where(both_dry.unsqueeze(0), torch.zeros_like(flux), flux)


def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minmod slope of the two one-sided differences."""
    same_sign = a * b > 0.0
    magnitude = torch.where(a.abs() < b.abs(), a, b)
    return torch.where(same_sign, magnitude, torch.zeros_like(a))


def _limited_slopes(h, u, wet) -> tuple[torch.Tensor, torch.Tensor]:
    """Minmod slopes of ``(h, u)``; zero at the walls and next to a dry cell."""
    def slope(w):
        zero = torch.zeros_like(w)
        backward = torch.cat([zero[0:1], w[1:] - w[:-1]])
        forward = torch.cat([w[1:] - w[:-1], zero[-1:]])
        s = _minmod(backward, forward)
        return torch.cat([zero[0:1], s[1:-1], zero[-1:]])

    left_wet = torch.cat([wet[0:1], wet[:-1]])
    right_wet = torch.cat([wet[1:], wet[-1:]])
    interior = (wet & left_wet & right_wet).to(h.dtype)
    return slope(h) * interior, slope(u) * interior


def _interface_fluxes(h, hu, g, dt, dx, order) -> torch.Tensor:
    """Interface fluxes ``(2, cells + 1)`` including the two reflective walls."""
    n = h.shape[0]
    u, c, wet = _states(h, hu, g)

    if order == 1:
        hL, uL = h[:-1], u[:-1]
        hR, uR = h[1:], u[1:]
    elif order == 2:
        # MUSCL with a minmod limiter and the Hancock half-step predictor: the
        # first-order scheme diffuses the drying front over O(10) cells, which
        # on this case costs ~17 % of the front speed (measured, see the doc).
        sh, su = _limited_slopes(h, u, wet)

        h_left = h - 0.5 * sh
        u_left = u + 0.5 * su
        h_right = h + 0.5 * sh
        u_right = u - 0.5 * su
        for hcand, ucand in ((h_left, u_left), (h_right, u_right)):
            degenerate = hcand <= DRY_DEPTH
            if bool(degenerate.any()):
                h_left = torch.where(h_left <= DRY_DEPTH, h, h_left)
                h_right = torch.where(h_right <= DRY_DEPTH, h, h_right)
                u_left = torch.where(degenerate, u, u_left)
                u_right = torch.where(degenerate, u, u_right)

        flux_left = _physical_flux(h_left, u_left, g)
        flux_right = _physical_flux(h_right, u_right, g)
        h_star = h - 0.5 * dt / dx * (flux_right[0] - flux_left[0])
        hu_star = hu - 0.5 * dt / dx * (flux_right[1] - flux_left[1])
        h_star = h_star.clamp_min(0.0)
        dry_star = h_star <= DRY_DEPTH
        h_star = torch.where(dry_star, torch.zeros_like(h_star), h_star)
        hu_star = torch.where(dry_star, torch.zeros_like(hu_star), hu_star)
        u_star, _, _ = _states(h_star, hu_star, g)
        hL, uL = h_star[:-1], u_star[:-1]
        hR, uR = h_star[1:], u_star[1:]
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")

    interior = _hll(hL, uL, hR, uR, g)

    # Reflective walls: no mass crosses, and the momentum flux is the
    # hydrostatic wall pressure 0.5 g h^2 of the adjacent cell -- the exact flux
    # for u = 0. (A mirrored ghost state would additionally advect h u^2 through
    # a solid surface.)
    walls = torch.zeros((2, n + 1), dtype=h.dtype)
    walls[:, 1:n] = interior
    walls[1, 0] = 0.5 * g * h[0] * h[0]
    walls[1, n] = 0.5 * g * h[-1] * h[-1]
    return walls


def front_position(x: torch.Tensor, h: torch.Tensor, depth0: float,
                   level: float = FRONT_LEVELS[0],
                   dry: float = 0.0) -> float:
    """Rightmost ``x`` whose depth exceeds ``level * depth0``.

    ``dry`` is the coordinate where the tank ends; anything at or past it is not
    water.  Returns ``0.0`` for a fully dry field.
    """
    threshold = level * depth0
    mask = h > threshold
    if dry > 0.0:
        mask = mask & (x < dry)
    if not bool(mask.any()):
        return 0.0
    return float(x[mask].max())


def ritter_profile(x: torch.Tensor, time: float, depth0: float,
                   dam_x: float = 0.0, gravity: float = GRAVITY):
    """Exact Ritter ``(h, u)`` sampled on ``x`` (see ``tensorsph.ritter``)."""
    from . import ritter as _ritter
    return (_ritter.depth(x, time, depth0, dam_x=dam_x, gravity=gravity),
            _ritter.velocity(x, time, depth0, dam_x=dam_x, gravity=gravity))


def dam_break(cells: int = 2000, tank_width: float = 2.0, column_width: float = 0.2,
              column_height: float = 0.4, t_end: float = 0.5, cfl: float = 0.4,
              order: int = 2, sample_times: list[float] | None = None,
              gravity: float = GRAVITY) -> ShallowWaterRun:
    """Integrate the dam break and report the leading edge over time.

    Initial condition: ``h = column_height`` for ``x <= column_width`` and ``0``
    beyond, ``u = 0`` everywhere.  The column occupies ``[0, a]``, so the tank's
    back wall is at ``x = 0`` and the rarefaction reaches it at
    ``t = a / sqrt(g H)``, exactly as in the SPH case; the far wall sits at
    ``tank_width`` and must be far enough away that the front never reaches it
    inside the reported window (otherwise the run stops early and says so).

    Parameters
    ----------
    cells:
        Number of uniform cells across ``tank_width``.
    t_end:
        Physical end time (s).
    cfl:
        Courant number on ``(|u| + sqrt(g h)) dt / dx``; required ``< 0.5``.
    order:
        ``1`` (HLL only) or ``2`` (MUSCL + Hancock).  Order 1 is kept because it
        is what the verification in ``docs/`` measures the order of.
    sample_times:
        Times at which profiles are stored.  The leading edge is sampled every
        step regardless; these only bound the memory held by profiles.
    """
    if cells < 8:
        raise ValueError("cells must be at least 8")
    if not 0.0 < cfl < 0.5:
        raise ValueError("cfl must be in (0, 0.5) for an explicit scheme")
    if not column_width > 0.0 or not column_width < tank_width:
        raise ValueError("the column must be strictly inside the tank")
    if column_height <= 0.0:
        raise ValueError("column_height must be positive")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    dx = tank_width / cells
    x = (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    h = torch.where(x <= column_width, torch.full_like(x, float(column_height)),
                    torch.zeros_like(x))
    hu = torch.zeros_like(x)

    pending = sorted(float(t) for t in sample_times) if sample_times else []
    run = ShallowWaterRun(cells=cells, dx=dx, cfl=cfl, order=order)
    volume0 = float(h.sum()) * dx

    def record(time_value):
        run.times.append(time_value)
        run.fronts.append({level: front_position(x, h, column_height, level,
                                                 dry=tank_width)
                           for level in FRONT_LEVELS})

    record(0.0)
    time = 0.0
    steps = 0
    while time < t_end - 1e-15:
        u, c, _ = _states(h, hu, gravity)
        speed = float((u.abs() + c).max().item())
        if speed <= 0.0:
            break
        dt = min(cfl * dx / speed, t_end - time)
        if dt <= 0.0:
            break

        fluxes = _interface_fluxes(h, hu, gravity, dt, dx, order)
        h = h - dt / dx * (fluxes[0, 1:] - fluxes[0, :-1])
        hu = hu - dt / dx * (fluxes[1, 1:] - fluxes[1, :-1])

        # Positivity guard: the HLL average cannot make a wet cell negative,
        # but it can produce -1e-18, and one negative depth poisons sqrt(g h).
        h = h.clamp_min(0.0)
        dry = h <= DRY_DEPTH
        h = torch.where(dry, torch.zeros_like(h), h)
        hu = torch.where(dry, torch.zeros_like(hu), hu)

        time += dt
        steps += 1
        record(time)

        while pending and pending[0] <= time + 1e-12:
            pending.pop(0)
            u_now, _, _ = _states(h, hu, gravity)
            run.profiles.append((time, x.clone(), h.clone(), u_now.clone()))

        if run.fronts[-1][FRONT_LEVELS[0]] >= tank_width - 1.5 * dx:
            run.extra["front_reached_wall"] = True
            break

    run.extra.setdefault("front_reached_wall", False)
    u_final, _, _ = _states(h, hu, gravity)
    if not run.profiles or run.profiles[-1][0] < time - 1e-12:
        run.profiles.append((time, x.clone(), h.clone(), u_final.clone()))
    run.mass_error = abs(float(h.sum()) * dx - volume0) / max(volume0, 1e-30)
    run.steps = steps
    return run


def front_coefficient(tau, za):
    """Least-squares slope of ``Z/a`` against ``tau`` over a supplied window.

    The finite reservoir makes ``Z/a`` sub-linear in ``tau``, so a slope is only
    meaningful with its window attached; this helper keeps the two together.
    """
    if len(tau) != len(za) or len(tau) < 2:
        raise ValueError("need at least two matching samples")
    n = float(len(tau))
    mean_t = sum(tau) / n
    mean_z = sum(za) / n
    sxx = sum((t - mean_t) ** 2 for t in tau)
    if sxx <= 0.0:
        raise ValueError("degenerate tau window")
    sxy = sum((t - mean_t) * (z - mean_z) for t, z in zip(tau, za))
    return sxy / sxx