"""Spatial Robin profiles on the periodic-y unit square, k=rho=cp=1."""
import json
import math

import torch

from tensorsph.heat_conduction_2d import lattice_offsets, prepare_slab_film_rhs


def measure(n):
    dx = 1/n
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    k = 2*math.pi
    left = torch.cos(k*x)
    right = .5*torch.sin(k*x)
    exact = (left[:,None]*torch.sinh(k*(1-x)) + right[:,None]*torch.sinh(k*x))/math.sinh(k)
    walls = torch.stack((left, right))
    flux = k*torch.stack((left*math.cosh(k)-right, right*math.cosh(k)-left))/math.sinh(k)
    films = torch.stack((6+torch.sin(k*x), 8+torch.cos(k*x)))
    ambient = walls + flux/films
    rhs = prepare_slab_film_rhs(exact, lattice_offsets(dx, 1.3*dx), dx, films, local_films=True)
    field = exact
    end = .02
    steps = math.ceil(end/(.16*dx**2))
    dt = end/steps
    heat = 0.
    for _ in range(steps):
        predictor, _, _ = rhs(field, ambient)
        rate, _, wall = rhs(field + .5*dt*predictor, ambient)
        heat += dt*dx*float((films*(ambient-wall)).sum())
        field = field + dt*rate
    _, power, wall = rhs(field, ambient)
    return {"cells_per_side": n, "field_max_error_K": float((field-exact).abs().max()),
            "wall_max_error_K": float((wall-walls).abs().max()),
            "flux_max_error_W_m2": float((power/dx-flux).abs().max()),
            "local_balance_W_m2": float((power/dx-films*(ambient-wall)).abs().max()),
            "storage_balance_J": abs(float((field-exact).mean()) - heat)}


if __name__ == "__main__":
    print(json.dumps([measure(n) for n in (16, 32, 64)], indent=2))
