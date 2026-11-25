"""Transient mode of two equal rods with fixed end temperatures and a contact."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, stencil_rhs_2d, solve_transient_layered_slab,
    stencil_x_face_power_2d,
)


def measure(n, *, heterogeneous=False, implicit=False):
    dx = 1 / n
    mid = n // 2
    if n < 4 or n % 2:
        raise ValueError("use an even mesh with at least two cells per layer")
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    a = 1.5 * math.pi
    b = 4 * math.pi / 3 if heterogeneous else a
    k_right = 4. if heterogeneous else 1.
    # Match decay rates k*b²/c and interface fluxes on both rods.
    left_loss,right_loss = (2.,7.) if implicit else (0.,0.)
    rate = a*a+left_loss
    capacity_right = (k_right*b*b+right_loss)/rate
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
    if implicit:
        times = torch.linspace(0.,end,n//2+1,dtype=x.dtype)
        loss = torch.where(x<.5,left_loss,right_loss)
        states = solve_transient_layered_slab(profile,times,dx,(0.,0.),conductivity=k,
                    volumetric_heat_capacity=capacity,contact_resistance_x=faces,heat_loss_coefficient=loss)
        field = states[-1:]
        # Each row here is a time level; transverse invariance is required,
        # so evaluate the two-dimensional link flux one row at a time.
        current = torch.stack([stencil_x_face_power_2d(stage[None,:],offsets,dx,mid,(0.,0.),
                    conductivity=k,layered_x=True,contact_resistance_x=faces)/(dx*factor)
                    for stage in .5*(states[:-1]+states[1:])])
        dt = times.diff()
        exact_current = -a*math.cos(a/2)*torch.exp(-rate*times[:-1])*(-torch.expm1(-rate*dt))/(rate*dt)
    else:
        steps = math.ceil(end * max(1., k_right / capacity_right) / (.16 * dx**2))
        dt = end / steps
        for _ in range(steps):
            field = field + dt * rhs(field + .5 * dt * rhs(field))
    temp = field[0]
    decay = math.exp(-rate * end)
    left = temp[mid-1] + .5 * (temp[mid-1] - temp[mid-2])
    right = temp[mid] - .5 * (temp[mid+1] - temp[mid])
    # Integrate each half-sine exactly, weighting by its volumetric heat capacity.
    initial_heat = ((1 - math.cos(a/2)) / a
                    - capacity_right * amplitude * (1 - math.cos(b/2)) / b)
    heat_change = float((capacity * (temp - profile)).sum() * dx)
    result = {"n": n, "right_specific_heat": capacity_right,
            "stored_heat_change_abs_error": abs(heat_change - (decay - 1) * initial_heat), "temperature_max_abs_error": float((temp - decay * profile).abs().max()),
            "jump_abs_error": abs(float(left - right) - jump * decay)}
    if implicit:
        result["contact_flux_error"] = float((current-exact_current).abs().max())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implicit",action="store_true",help="verify two-material implicit cooling with bulk loss")
    if not parser.parse_args().implicit:
        print(json.dumps([measure(n, heterogeneous=h) for h in (False,True) for n in (16,32,64)],indent=2))
    else:
        rows = []
        for n in (16,32,64,128):
            row = measure(n,heterogeneous=True,implicit=True)
            metrics = ("stored_heat_change_abs_error","temperature_max_abs_error","jump_abs_error","contact_flux_error")
            row["orders"] = {k:math.log2(rows[-1][k]/row[k]) for k in metrics} if rows else {}
            row["passed"] = n<128 or min(row["orders"].values())>1.8
            rows.append(row)
        rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
        print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
        raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
