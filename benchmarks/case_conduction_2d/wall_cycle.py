"""A periodically heated finite slab in its analytic time-periodic state."""
import cmath
import json
import math

import torch

from .wall_decay import evolve_slab


def measure(n):
    dx = 1 / n
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    period = .08
    omega = 2 * math.pi / period
    wave = cmath.sqrt(1j * omega)
    amplitude = torch.sinh(wave * (1 - x.to(torch.complex128))) / cmath.sinh(wave)

    def exact(t):
        return (amplitude * cmath.exp(1j * omega * t)).real.expand(4, -1)

    def walls(t):
        return math.cos(omega * t), 0.

    # Use an odd fraction of the period so phase and amplitude both matter.
    end = 3 * period / 8
    field, _, _ = evolve_slab(exact(0.), dx, walls, end, .16)
    error = (field - exact(end)).abs()
    return {"n": n, "max_error": float(error.max()),
            "wall_adjacent_max_error": float(error[:, [0, -1]].max())}


if __name__ == "__main__":
    print(json.dumps([measure(n) for n in (16, 32, 64)], indent=2))
