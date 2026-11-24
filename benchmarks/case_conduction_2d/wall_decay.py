"""Transient Dirichlet slab on [0,1]^2 with periodic y and unit diffusivity."""
import json
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d,
    lattice_offsets,
    stencil_rhs_2d,
)


def measure(n, courant=0.16):
    dx = 1 / n
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    steady = 3 - 2 * x
    mode = torch.cos(2 * math.pi * x[:, None]) * torch.sin(math.pi * x)
    temperature = steady + mode
    offsets = lattice_offsets(dx, 1.3 * dx)
    factor = lattice_diffusion_factor_2d(1.3, dx)
    end = 0.02
    steps = math.ceil(end / (courant * dx**2))
    dt = end / steps

    def rhs(field):
        return stencil_rhs_2d(field, offsets, x_wall_temperatures=(3., 1.)) / factor

    for _ in range(steps):
        midpoint = temperature + 0.5 * dt * rhs(temperature)
        temperature = temperature + dt * rhs(midpoint)
    exact = steady + math.exp(-5 * math.pi**2 * end) * mode
    error = (temperature - exact).abs()
    return {
        "n": n, "steps": steps, "dt": dt, "max_error": float(error.max()),
        "wall_adjacent_max_error": float(error[:, [0, -1]].max()),
        "mean_temperature_drift": float((temperature.mean() - steady.mean()).abs()),
    }


if __name__ == "__main__":
    print(json.dumps([measure(n) for n in (16, 32, 64)] + [measure(64, 0.08)], indent=2))
