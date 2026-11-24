"""Transient mode of two equal rods with fixed end temperatures and a contact."""
import json
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, stencil_rhs_2d,
)


def measure(n):
    dx = 1 / n
    mid = n // 2
    if n < 4 or n % 2:
        raise ValueError("use an even mesh with at least two cells per layer")
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    a = 1.5 * math.pi
    # T_left=sin(ax), T_right=-sin(a(1-x)); equal flux and jump=R*q.
    profile = torch.where(x < .5, torch.sin(a*x), -torch.sin(a*(1-x)))
    faces = x.new_zeros(n - 1)
    faces[mid - 1] = 2 / a
    offsets = lattice_offsets(dx, 1.3 * dx)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    k = torch.ones_like(x)

    def rhs(field):
        return stencil_rhs_2d(field, offsets, x_wall_temperatures=(0., 0.),
                              conductivity=k, layered_x=True, dx=dx,
                              contact_resistance_x=faces) / factor

    field = profile[None, :]
    end = .015
    steps = math.ceil(end / (.16 * dx**2))
    dt = end / steps
    for _ in range(steps):
        field = field + dt * rhs(field + .5 * dt * rhs(field))
    temp = field[0]
    decay = math.exp(-a**2 * end)
    left = temp[mid-1] + .5 * (temp[mid-1] - temp[mid-2])
    right = temp[mid] - .5 * (temp[mid+1] - temp[mid])
    return {"n": n, "temperature_max_abs_error": float((temp - decay * profile).abs().max()),
            "jump_abs_error": abs(float(left - right) - 2 * math.sin(a/2) * decay)}


if __name__ == "__main__":
    print(json.dumps([measure(n) for n in (16, 32, 64)], indent=2))
