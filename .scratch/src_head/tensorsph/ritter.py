"""Ritter (1892) shallow-water dam-break solution, and depth-profile measures.

Ritter's solution is the exact solution of the **shallow-water** equations for a
semi-infinite reservoir of still water of depth ``h0`` that is released onto a
dry, frictionless bed at ``t = 0``. With ``c0 = sqrt(g h0)`` and the dam at
``x = x_dam``, the water occupies the interval

    x_tail(t) = x_dam - c0 t  <=  x  <=  x_front(t) = x_dam + 2 c0 t

and, writing ``xi = (x - x_dam) / t``,

    h(x, t) = h0                              for x <= x_tail
            = (2 c0 - xi)^2 / (9 g)           for x_tail < x < x_front
            = 0                               for x >= x_front
    u(x, t) = 0                               for x <= x_tail
            = (2/3) (c0 + xi)                 for x_tail < x < x_front
            = 2 c0                            at the front (= d x_front/dt)

Three properties are exact and are what the tests pin, so that this module is
verified before it is used as a reference:

* depth at the dam section is the classical ``4 h0 / 9`` for every ``h0``;
* the volume released up to time ``t`` is exactly ``h0 c0 t`` (the reservoir
  level falls nowhere yet, the bed is dry, so all the water between the tail and
  the front came out of the reservoir);
* the Riemann invariant ``u + 2 sqrt(g h)`` equals ``2 c0`` across the
  rarefaction fan.

Scope: this is a *shallow-water* solution. It is only a valid reference while

* ``H / a << 1`` (the column is shallow, so hydrostatic pressure holds), and
* ``t < a / c0`` (the rarefaction has not reached the back wall of a *finite*
  reservoir; after that the wall reflects it and the solution above is void).

:func:`validity_window` returns the second bound. The first one has no sharp
threshold; ``docs/benchmarks/ritter-shallow-limit.md`` measures how the error
grows with ``H / a`` and is the reason this module exists.
"""
from __future__ import annotations

import math

import torch

GRAVITY = 9.81


def wave_speed(depth0: float, gravity: float = GRAVITY) -> float:
    """``c0 = sqrt(g h0)``."""
    return math.sqrt(gravity * float(depth0))


def tail_x(time: float, depth0: float, dam_x: float = 0.0,
           gravity: float = GRAVITY) -> float:
    """Upstream edge of the rarefaction fan."""
    return dam_x - wave_speed(depth0, gravity) * time


def front_x(time: float, depth0: float, dam_x: float = 0.0,
            gravity: float = GRAVITY) -> float:
    """Leading edge, ``x_dam + 2 c0 t``. Its speed is ``2 c0``."""
    return dam_x + 2.0 * wave_speed(depth0, gravity) * time


def validity_window(depth0: float, reservoir_width: float, dam_x: float | None = None,
                    gravity: float = GRAVITY) -> float:
    """Time until the rarefaction reaches the back wall: ``a / c0``.

    ``reservoir_width`` is ``a`` for a column occupying ``[0, a]`` whose dam is
    at ``x = a``. The wall at ``x = 0`` is then still outside the disturbance.
    """
    if dam_x is None:
        dam_x = reservoir_width
    return float(dam_x) / wave_speed(depth0, gravity)


def depth(x, time: float, depth0: float, dam_x: float = 0.0,
          gravity: float = GRAVITY) -> torch.Tensor:
    """Ritter water depth at positions ``x`` (metres) and time ``time`` (s)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    if time <= 0.0:
        return torch.full_like(x, float(depth0))
    c0 = wave_speed(depth0, gravity)
    xi = (x - dam_x) / time
    fan = (2.0 * c0 - xi).clamp_min(0.0).square() / (9.0 * gravity)
    h = torch.where(x <= dam_x - c0 * time, torch.full_like(fan, float(depth0)), fan)
    return torch.where(x >= dam_x + 2.0 * c0 * time, torch.zeros_like(fan), h)


def velocity(x, time: float, depth0: float, dam_x: float = 0.0,
             gravity: float = GRAVITY) -> torch.Tensor:
    """Ritter depth-averaged velocity ``u`` (m/s), see the module docstring."""
    x = torch.as_tensor(x, dtype=torch.float64)
    if time <= 0.0:
        return torch.zeros_like(x)
    c0 = wave_speed(depth0, gravity)
    xi = (x - dam_x) / time
    fan = (2.0 / 3.0) * (c0 + xi)
    u = torch.where(x <= dam_x - c0 * time, torch.zeros_like(fan), fan)
    return torch.where(x >= dam_x + 2.0 * c0 * time, torch.zeros_like(fan),
                       u.clamp(min=0.0))


def released_volume(time: float, depth0: float, gravity: float = GRAVITY) -> float:
    """Exact volume per unit width that has left the reservoir by ``time``.

    ``h0 c0 t``: the reservoir has not dropped anywhere yet (the tail still moves
    upstream at ``c0``), so the water in the fan came out of the reservoir.
    """
    return float(depth0) * wave_speed(depth0, gravity) * float(time)


def profile_mass(time: float, depth0: float, points: int = 20001,
                 gravity: float = GRAVITY) -> float:
    """Numerical integral of :func:`depth` over ``[tail, front]`` (trapezoid).

    Independent of :func:`released_volume` (which is closed-form), so comparing
    the two checks the profile's shape rather than its normalisation.
    """
    if time <= 0.0:
        return 0.0
    c0 = wave_speed(depth0, gravity)
    lo, hi = -c0 * time, 2.0 * c0 * time
    x = torch.linspace(lo, hi, points, dtype=torch.float64)
    h = depth(x, time, depth0, dam_x=0.0, gravity=gravity)
    return float(torch.trapezoid(h, x))


# --------------------------------------------------------------------------- #
# Measures of the simulated profile
# --------------------------------------------------------------------------- #

def particle_depth(positions: torch.Tensor, dx: float, bin_width: float,
                   lo: float = 0.0, hi: float | None = None):
    """Equivalent water depth of a 2-D column of particles, per x-bin.

    ``h = (number of particles in the bin) * dx^2 / bin_width`` — a counting
    measure, so it is *exactly* the discrete mass distribution and cannot be
    biased by the solver's compressibility (a density-weighted measure would
    carry a ``rho / rho0`` factor of order one percent into every comparison).
    Valid while particle masses are uniform, ``rho0 dx^2``, which is the case
    for ``hydrostatic_init = False``.

    Returns ``(x_centres, depth)`` as float64 tensors on the CPU.
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    pos = positions.detach().to("cpu", torch.float64)
    if pos.numel() == 0:
        return torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
    x = pos[:, 0]
    if hi is None:
        hi = float(x.max()) + bin_width
    count = max(1, int(math.ceil((hi - lo) / bin_width)))
    edges = torch.linspace(lo, hi, count + 1, dtype=torch.float64)
    index = torch.bucketize(x, edges) - 1
    inside = (index >= 0) & (index < count)
    histogram = torch.bincount(index[inside], minlength=count).to(torch.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, histogram * (dx * dx) / bin_width


def front_speed(times, front, fraction: float = 0.4) -> float:
    """Leading-edge speed over the *developed* stage of a front trajectory.

    ``(z[-1] - z[k]) / (t[-1] - t[k])`` with ``k`` chosen so that the last
    ``fraction`` of the samples are used. The mean over the whole record is not
    a front speed: the initial vertical face is a delta function in the
    shallow-water initial data, so the first samples sit inside the release
    transient, where the leading edge has not yet been accelerated to its
    developed value. At ``dx = 0.02`` (``H/dx = 5``) the leading edge covers
    10 % of its analytic advance in the first 10 % of the window while the
    developed-stage speed is already constant at 73 % of ``2 c0``; averaging
    from ``t = 0`` therefore reports 62 % of ``2 c0`` for the same run. The
    developed value is the one that can be compared with the analytic wave
    speed. See ``docs/benchmarks/ritter-shallow-limit.md``.

    Both sequences must be ordered in time and share one sample per entry.
    """
    t = torch.as_tensor(times, dtype=torch.float64).reshape(-1)
    z = torch.as_tensor(front, dtype=torch.float64).reshape(-1)
    if t.numel() != z.numel():
        raise ValueError("times and front must have the same length")
    if t.numel() < 2:
        raise ValueError("front_speed needs at least two samples")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    k = int(math.floor((1.0 - float(fraction)) * (t.numel() - 1) + 0.5))
    if k >= t.numel() - 1:
        raise ValueError("fraction leaves fewer than two samples")
    span = float(t[-1] - t[k])
    if span <= 0.0:
        raise ValueError("the time samples must be increasing")
    return float((z[-1] - z[k]) / span)


def profile_l1_error(x: torch.Tensor, measured: torch.Tensor,
                     reference: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """``||measured - reference||_1 / ||reference||_1`` on a sampled grid."""
    if x.numel() == 0:
        raise ValueError("empty profile")
    if mask is not None:
        x, measured, reference = x[mask], measured[mask], reference[mask]
    if x.numel() < 2:
        raise ValueError("profile needs at least two samples")
    numerator = float(torch.trapezoid((measured - reference).abs(), x))
    denominator = float(torch.trapezoid(reference.abs(), x))
    if denominator <= 0.0:
        raise ValueError("reference profile integrates to zero")
    return numerator / denominator


__all__ = [
    "GRAVITY", "depth", "front_speed", "front_x", "particle_depth",
    "profile_l1_error", "profile_mass", "released_volume", "tail_x",
    "validity_window", "velocity", "wave_speed",
]
