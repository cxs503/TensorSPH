"""Cooling with one fixed temperature, one insulated wall and periodic y."""
import json
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, stencil_rhs_with_wall_power_2d,
)


def measure(n, courant=.16):
    dx, end = 1/n, .02
    offsets = lattice_offsets(dx, 1.3*dx)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    a, b = math.pi/2, 2*math.pi

    def amplitude(t):
        return math.exp(-a*a*t) + .5*torch.cos(b*x)*math.exp(-(a*a+b*b)*t)

    def rhs(field):
        rate, power = stencil_rhs_with_wall_power_2d(field, offsets, dx, (2., None),
                                                      power_by_row=True)
        return rate/factor, power/factor

    initial = 2 + amplitude(0)[:, None]*torch.sin(a*x)
    field = initial
    steps = math.ceil(end/(courant*dx**2))
    dt = end/steps
    heat = 0.
    for _ in range(steps):
        rate, _ = rhs(field)
        rate, power = rhs(field + .5*dt*rate)
        field = field + dt*rate
        heat += float(power.sum())*dt
    _, power = rhs(field)
    exact = 2 + amplitude(end)[:, None]*torch.sin(a*x)
    return dict(n=n, steps=steps, dt=dt, max_temperature_error=float((field-exact).abs().max()),
                fixed_flux_error=float((power[0]/dx + a*amplitude(end)).abs().max()),
                insulated_flux=float(power[1].abs().max()/dx),
                heat_balance=abs(float((field-initial).mean())-heat))


if __name__ == "__main__":
    print(json.dumps([measure(n) for n in (16,32,64)] + [measure(64, .08)], indent=2))
