"""Transient mode of two equal rods with fixed end temperatures and a contact."""
import json
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, stencil_rhs_2d,
)


def measure(n, *, heterogeneous=False):
    dx = 1 / n
    mid = n // 2
    if n < 4 or n % 2:
        raise ValueError("use an even mesh with at least two cells per layer")
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    a = 1.5 * math.pi
    b = 4 * math.pi / 3 if heterogeneous else a
    k_right = 4. if heterogeneous else 1.
    # Match decay rates k*b²/c and interface fluxes on both rods.
    capacity_right = k_right * (b / a)**2
    amplitude = a * math.cos(a/2) / (k_right * b * math.cos(b/2))
    jump = math.sin(a/2) + amplitude * math.sin(b/2)
    profile = torch.where(x < .5, torch.sin(a*x), -amplitude * torch.sin(b*(1-x)))
    faces = x.new_zeros(n - 1)
    faces[mid - 1] = jump / (-a * math.cos(a/2))
    offsets = lattice_offsets(dx, 1.3 * dx)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    k = torch.ones_like(x)
    k[mid:] = k_right
    capacity = torch.ones_like(x)
    capacity[mid:] = capacity_right

    def rhs(field):
        return stencil_rhs_2d(field, offsets, x_wall_temperatures=(0., 0.),
                              conductivity=k, layered_x=True, dx=dx,
                              contact_resistance_x=faces, specific_heat=capacity) / factor

    field = profile[None, :]
    end = .015
    steps = math.ceil(end * max(1., k_right / capacity_right) / (.16 * dx**2))
    dt = end / steps
    for _ in range(steps):
        field = field + dt * rhs(field + .5 * dt * rhs(field))
    temp = field[0]
    decay = math.exp(-a**2 * end)
    left = temp[mid-1] + .5 * (temp[mid-1] - temp[mid-2])
    right = temp[mid] - .5 * (temp[mid+1] - temp[mid])
    # Integrate each half-sine exactly, weighting by its volumetric heat capacity.
    initial_heat = ((1 - math.cos(a/2)) / a
                    - capacity_right * amplitude * (1 - math.cos(b/2)) / b)
    heat_change = float((capacity * (temp - profile)).sum() * dx)
    return {"n": n, "right_specific_heat": capacity_right,
            "stored_heat_change_abs_error": abs(heat_change - (decay - 1) * initial_heat), "temperature_max_abs_error": float((temp - decay * profile).abs().max()),
            "jump_abs_error": abs(float(left - right) - jump * decay)}


if __name__ == "__main__":
    print(json.dumps([measure(n, heterogeneous=h) for h in (False, True) for n in (16, 32, 64)], indent=2))
