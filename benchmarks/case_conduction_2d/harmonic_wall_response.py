"""Compare prescribed traveling wall temperatures with the complex slab admittance."""
import cmath
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, solve_harmonic_slab_2d,
    stencil_rhs_with_wall_power_2d,
)


def measure(n, omega):
    k, capacity, dx = 2., 3., 1/n
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    wave = torch.exp(2j*math.pi*x)[:, None]
    west = (1+.6j)*wave[:, 0]+.5
    wall = (west, None)
    t = solve_harmonic_slab_2d(torch.zeros((n, n)), dx, wall, omega, conductivity=k,
                               volumetric_heat_capacity=capacity)
    exact = torch.zeros_like(t)
    exact_flux = torch.zeros_like(west)
    for b, trace in ((0., .5), (2*math.pi, (1+.6j)*wave)):
        lam = cmath.sqrt(b*b+1j*omega*capacity/k)
        exact += trace*torch.cosh(lam*(1-x))/cmath.cosh(lam)
        exact_flux += torch.as_tensor(trace, dtype=t.dtype).reshape(-1)*k*lam*cmath.tanh(lam)
    offsets = lattice_offsets(dx, 1.3*dx)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    quadratures = []
    for component in ("real", "imag"):
        _, exchange = stencil_rhs_with_wall_power_2d(
            getattr(t, component), offsets, dx, (getattr(west, component), None),
            conductivity=k, power_by_row=True)
        quadratures.append(exchange[0]/factor)
    power = torch.complex(*quadratures)
    balance = abs(1j*omega*capacity*dx**2*t.sum()-power.sum())
    flux = power/dx
    metrics = (float((t-exact).abs().max()), float((flux-exact_flux).norm()/exact_flux.norm()),
            float(abs(torch.angle(power.sum()/exact_flux.mean()))), float(balance))
    return metrics, flux, exact_flux


if __name__ == "__main__":
    for omega in (.5, 15.):
        previous = None
        previous_recovered = None
        for n in (16, 32, 64, 128, 256):
            errors, flux, reference = measure(n, omega)
            print(f"omega={omega} n={n} errors(T,rel_flux,net_flux_phase_rad,balance)={errors}")
            if not all(map(math.isfinite, errors)) or errors[-1] > 1e-10:
                raise SystemExit("wall power balance failed")
            if previous is not None:
                old_errors, coarse, coarse_reference = previous
                orders = [math.log2(a/b) for a, b in zip(old_errors[:3], errors[:3])]
                # Restrict fine face fluxes to coarse face centers, then cancel
                # the leading row-wise wall transfer error by Richardson extrapolation.
                recovered = 2*flux.reshape(-1, 2).mean(1)-coarse
                err = float((recovered-coarse_reference).norm()/coarse_reference.norm())
                print(f"  orders={orders} extrapolated_relative_flux_error={err:.8g}")
                if orders[0] < 1.8 or orders[2] < 1.8 or errors[1] >= old_errors[1]:
                    raise SystemExit("wall convergence failed")
                if previous_recovered is not None and err >= previous_recovered/3.5:
                    raise SystemExit("two-grid wall flux failed to converge")
                if n == 256 and (orders[1] < .9 or err >= .001):
                    raise SystemExit("finest wall flux failed")
                previous_recovered = err
            previous = errors, flux, reference
