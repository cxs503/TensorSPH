"""Forced thermal waves in a homogeneous unit slab with periodic transverse axis."""
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, solve_harmonic_slab_2d,
    stencil_rhs_with_wall_power_2d,
)


def measure(n, mixed):
    x = (torch.arange(n, dtype=torch.float64)+.5)/n
    dx, k, capacity, omega = 1/n, 2., 3., 15.
    a = math.pi/2 if mixed else math.pi
    b = 2*math.pi
    transverse = torch.exp(1j*b*x)[:, None]
    mode = torch.sin(a*x)*transverse
    # Unit forcing of one spatial eigenmode has a known complex response.
    gain = 1/(k*(a*a+b*b)+1j*omega*capacity)
    base = .03+.02j
    source = mode+1j*omega*capacity*base
    walls = (base, None if mixed else base)
    field = solve_harmonic_slab_2d(source, dx, walls, omega, conductivity=k,
                                   volumetric_heat_capacity=capacity)
    observed = ((field-base)*mode.conj()).sum()/mode.abs().square().sum()
    exact = base+gain*mode
    # Obtain powers independently from real-space particle exchanges.
    offsets = lattice_offsets(dx, 1.3*dx)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")

    def powers(component):
        wall_values = tuple(None if w is None else getattr(w, component) for w in walls)
        _, power = stencil_rhs_with_wall_power_2d(getattr(field, component), offsets, dx,
                        wall_values, conductivity=k, power_by_row=True)
        return power/factor

    power = torch.complex(powers("real"), powers("imag"))
    west = -k*a*gain*transverse[:, 0]
    east = torch.zeros_like(west) if mixed else west
    flux_error = (power/dx-torch.stack((west, east))).abs().max()
    balance = (1j*omega*capacity*field.sum()-source.sum())*dx**2-power.sum()
    return (float((field-exact).abs().max()), float(abs(abs(observed/gain)-1)),
            float(abs(torch.angle(observed/gain))), float(flux_error), float(abs(balance)))


if __name__ == "__main__":
    for mixed in (False, True):
        previous = None
        for n in (16, 32, 64, 128):
            values = measure(n, mixed)
            orders = [math.log2(a/b) for a, b in zip(previous, values[:4])] if previous else []
            print(f"mixed={mixed} n={n} errors(T,rel_amplitude,phase_rad,flux,balance)={values} orders={orders}")
            if not all(map(math.isfinite, values)) or values[-1] > 1e-10 or (orders and min(orders) < 1.8):
                raise SystemExit("harmonic validation failed")
            previous = values[:4]
