"""Heated periodic strip with two fixed walls or one insulated right wall."""
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, solve_steady_slab_2d,
    stencil_rhs_with_wall_power_2d,
)


def measure(n, insulated):
    dx, k, b = 1/n, 2., 2*math.pi
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    wave = .4*torch.cos(b*x)[:, None]
    # T = 3 + p(x)*(1+.4 cos(by)); p(0)=0, p(1)=0 or p'(1)=0.
    slope = 2 if insulated else 1
    p = x*(slope-x)
    exact = 3 + p*(1+wave)
    source = k*(2*(1+wave)+b*b*p*wave)
    walls = (3., None if insulated else 3.)
    field = solve_steady_slab_2d(source, dx, walls, conductivity=k)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    rate, powers = stencil_rhs_with_wall_power_2d(
        field, lattice_offsets(dx, 1.3*dx), dx, walls, conductivity=k, power_by_row=True)
    flux = powers/(factor*dx)
    target = torch.stack((-k*slope*(1+wave[:, 0]), k*(slope-2)*(1+wave[:, 0])))
    error = (field-exact).abs()
    return (float(error.max()), float(error[:, [0, -1]].max()), float((flux-target).abs().max()),
            float((rate/factor+source).abs().max()), float(abs(dx*flux.sum()+dx**2*source.sum())))


if __name__ == "__main__":
    for insulated in (False, True):
        previous = None
        for n in (16, 32, 64, 128):
            errors = measure(n, insulated)
            orders = [math.log2(a/b) for a, b in zip(previous, errors[:3])] if previous else []
            print(f"insulated={insulated} n={n} max(T,near_wall_T,wall_q,residual,balance)={errors} orders={orders}")
            if not all(map(math.isfinite, errors)) or errors[3] > 1e-8 or errors[4] > 1e-10:
                raise SystemExit("discrete energy equation failed")
            if orders and min(orders) < 1.75:
                raise SystemExit("convergence gate failed")
            previous = errors[:3]
