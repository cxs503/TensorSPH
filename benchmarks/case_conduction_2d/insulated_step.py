"""Relax a unit temperature step in a closed unit slab with unit diffusivity."""
import json
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, stencil_rhs_2d, stencil_x_face_power_2d,
)


def measure(n):
    if n % 2:
        raise ValueError("the initial jump must coincide with a cell face")
    dx, end = 1/n, .02
    x = (torch.arange(n, dtype=torch.float64) + .5)*dx
    field = (x < .5).to(x.dtype).expand(5, -1).clone()
    offsets = lattice_offsets(dx, 1.3*dx)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")

    def rhs(t):
        return stencil_rhs_2d(t, offsets, x_wall_temperatures=(None, None))/factor

    steps = math.ceil(end/(.16*dx**2))
    dt = end/steps
    for _ in range(steps):
        field += dt*rhs(field + .5*dt*rhs(field))

    # Neumann cosine series for the indicator of [0, 1/2).
    # Only odd modes contribute. At t=.02 the first omitted exponential is <1e-90.
    m = torch.arange(1, 33, 2, dtype=x.dtype)*math.pi
    decay = torch.exp(-m*m*end)
    exact = .5 + (2*torch.sin(m/2)*decay/m) @ torch.cos(m[:, None]*x)
    exact_flux = float(2*decay.sum())
    power = stencil_x_face_power_2d(field, offsets, dx, n//2, (None, None))
    flux = float(power)/(factor*field.shape[0]*dx)
    return dict(n=n, max_temperature_error=float((field-exact).abs().max()),
                center_flux=flux, center_flux_error=abs(flux-exact_flux),
                mean_drift=abs(float(field.mean())-.5))


if __name__ == "__main__":
    print(json.dumps([measure(n) for n in (16,32,64)], indent=2))
