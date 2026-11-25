"""Integrate inward heat at both fixed-temperature walls of a cooling slab."""
import json
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, stencil_rhs_2d,
    stencil_rhs_with_wall_power_2d,
)


def measure(n):
    dx = 1 / n
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    # T(x,t) = 5 - 4x + sin(pi*x) exp(-pi²*t), uniform in y.
    initial = (5 - 4 * x + torch.sin(math.pi * x)).expand(n, -1)
    temperature = initial
    walls = (5., 1.)
    offsets = lattice_offsets(dx, 1.3 * dx)
    factor = lattice_diffusion_factor_2d(1.3, dx)
    end = .02
    steps = math.ceil(end / (.16 * dx**2))
    dt = end / steps
    heat = initial.new_zeros(2)
    for _ in range(steps):
        predictor = stencil_rhs_2d(temperature, offsets, x_wall_temperatures=walls)
        mid = temperature + .5 * dt / factor * predictor
        rate, power = stencil_rhs_with_wall_power_2d(mid, offsets, dx, walls)
        temperature = temperature + dt / factor * rate
        heat += dt / factor * power

    loss_per_wall = -math.expm1(-math.pi**2 * end) / math.pi
    exact = initial.new_tensor([4 * end - loss_per_wall, -4 * end - loss_per_wall])
    change = (temperature - initial).sum() * dx**2
    return {"n": n, "wall_heat": heat.tolist(), "analytic_wall_heat": exact.tolist(),
            "wall_max_abs_error": float((heat - exact).abs().max()),
            "heat_balance_residual": float((change - heat.sum()).abs())}


if __name__ == "__main__":
    print(json.dumps([measure(n) for n in (16, 32, 64)], indent=2))
