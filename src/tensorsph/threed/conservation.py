"""Conservation diagnostics for the 3-D WCSPH runs.

Nothing here is a solver: these are the meters the benchmarks read.  Every one of
them is defined from the *billed* quantities (masses, velocities, densities), so a
value can be reproduced from a snapshot without re-running anything.

Which readers are meaningful depends on the case, and the benchmarks say so:

* **triply periodic box** -- no external force, no wall, every pair force is
  central and equal-and-opposite, so total linear momentum is conserved to
  round-off and that is a real gate (``benchmarks/threed_shear``);
* **wall-bounded hydrostatic column** -- gravity and the wall reaction are
  external, so momentum is *not* conserved and quoting a "momentum drift" there
  would be meaningless.  The meters used are the centre-of-mass drift and the
  density/temperature-free energy budget instead.

Angular momentum is reported for completeness but is **not** gated in a periodic
box: with the minimum-image convention the pair torque is only conserved modulo
``L x (lattice translation)``, so the number is evidence, not a criterion.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "mass_total",
    "momentum",
    "angular_momentum",
    "kinetic_energy",
    "internal_energy",
    "total_energy",
    "centre_of_mass",
    "diagnostics",
    "relative_drift",
]


def mass_total(mass: torch.Tensor) -> float:
    return float(mass.sum())


def momentum(mass: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    return (mass.unsqueeze(-1) * vel).sum(dim=0)


def angular_momentum(mass: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor,
                     origin=None) -> torch.Tensor:
    if origin is None:
        origin = torch.zeros(pos.shape[1], dtype=pos.dtype, device=pos.device)
    r = pos - origin
    return (mass.unsqueeze(-1) * torch.cross(r, vel, dim=-1)).sum(dim=0)


def kinetic_energy(mass: torch.Tensor, vel: torch.Tensor) -> float:
    return float(0.5 * (mass.unsqueeze(-1) * vel.square()).sum())


def internal_energy(rho: torch.Tensor, rho0: float, sound_speed: float,
                    gamma: float) -> torch.Tensor:
    """Specific internal energy of the Tait fluid, zero at ``rho = rho0``.

    ``de = -p d(1/rho)`` integrated in closed form for ``p = B[(rho/rho0)^gamma - 1]``::

        e = c0^2 / gamma * [ ((rho/rho0)^(gamma-1) - 1) / (gamma - 1) + rho0/rho - 1 ]

    It is quadratic in the density perturbation, so for a weakly compressible run
    it is a small number that a 10 % error in ``rho`` moves by 100 %: reported,
    never used as the primary accuracy gate.
    """
    ratio = rho / rho0
    return (sound_speed ** 2 / gamma) * (
        (ratio.pow(gamma - 1.0) - 1.0) / (gamma - 1.0) + 1.0 / ratio - 1.0)


def total_energy(mass: torch.Tensor, vel: torch.Tensor, rho: torch.Tensor,
                 rho0: float, sound_speed: float, gamma: float) -> float:
    thermal = float((mass * internal_energy(rho, rho0, sound_speed, gamma)).sum())
    return kinetic_energy(mass, vel) + thermal


def centre_of_mass(mass: torch.Tensor, pos: torch.Tensor,
                   periodic_box=None) -> torch.Tensor:
    """Mass centre; for a periodic box the unwrapped mean is taken on the circle.

    A plain mean of wrapped coordinates jumps by ``L`` whenever a particle
    crosses the boundary, which would show up as a bogus "centre of mass drift".
    The circular mean (``atan2`` of the sine/cosine moments) is used instead;
    for a lattice that only collapses slightly it is well defined.
    """
    if periodic_box is None:
        return (mass.unsqueeze(-1) * pos).sum(dim=0) / mass.sum()
    box = torch.as_tensor(periodic_box, dtype=pos.dtype,
                          device=pos.device).reshape(-1)
    if box.numel() == 1:
        box = box.expand(pos.shape[1])
    angle = 2.0 * math.pi * pos / box.reshape(1, -1)
    weight = mass / mass.sum()
    mean_angle = torch.atan2((weight.unsqueeze(-1) * torch.sin(angle)).sum(dim=0),
                             (weight.unsqueeze(-1) * torch.cos(angle)).sum(dim=0))
    return (mean_angle / (2.0 * math.pi)).remainder(1.0) * box


def diagnostics(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                rho: torch.Tensor, *, rho0: float, sound_speed: float,
                gamma: float = 7.0, periodic_box=None, origin=None) -> dict:
    """One row of the conservation ledger."""
    p = momentum(mass, vel)
    return {
        "mass": mass_total(mass),
        "momentum": p.tolist(),
        "momentum_magnitude": float(torch.linalg.vector_norm(p)),
        "angular_momentum": angular_momentum(mass, pos, vel, origin).tolist(),
        "kinetic_energy": kinetic_energy(mass, vel),
        "internal_energy": float((mass * internal_energy(
            rho, rho0, sound_speed, gamma)).sum()),
        "total_energy": total_energy(mass, vel, rho, rho0, sound_speed, gamma),
        "centre_of_mass": centre_of_mass(mass, pos, periodic_box).tolist(),
    }


def relative_drift(values, *, scale: float | None = None) -> dict:
    """Drift of a scalar series: absolute change, relative change and worst step.

    ``scale`` overrides the denominator for quantities whose initial value is
    zero or near zero (momentum in a periodic box, for instance): the honest
    denominator there is the scale of the momentum the flow *could* carry,
    ``sum m |v|``, not the initial value.
    """
    first = float(values[0])
    last = float(values[-1])
    peak = max(abs(float(v)) for v in values)
    denominator = abs(scale) if scale is not None else abs(first)
    delta = last - first
    return {
        "initial": first,
        "final": last,
        "delta": delta,
        "peak_abs": peak,
        "scale": denominator,
        "relative_drift": (delta / denominator) if denominator > 0.0 else None,
        "relative_to_peak": (delta / peak) if peak > 0.0 else None,
    }
