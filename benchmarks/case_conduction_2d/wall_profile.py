"""Check a harmonic temperature field with spatially varying Dirichlet data."""
import json
import math

import torch

from .wall_decay import evolve_slab


def measure(n):
    dx = 1 / n
    coords = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    k = 2 * math.pi
    left = torch.cos(k * coords)
    right = 0.5 * torch.sin(k * coords)
    # Each term has zero Laplacian and matches its respective wall profile.
    exact = (left[:, None] * torch.sinh(k * (1 - coords))
             + right[:, None] * torch.sinh(k * coords)) / math.sinh(k)
    temperature, _, _ = evolve_slab(exact, dx, (left, right), 0.02, 0.16)
    error = (temperature - exact).abs()
    return {"n": n, "max_error": float(error.max()),
            "wall_adjacent_max_error": float(error[:, [0, -1]].max())}


if __name__ == "__main__":
    print(json.dumps([measure(n) for n in (16, 32, 64)], indent=2))
