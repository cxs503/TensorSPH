"""An inviscid circular liquid patch stretching to an ellipse (Monaghan, 1994)."""
import argparse
import json
import math

import numpy as np
import torch

from tensorsph.delta_plus import delta_plus_step
from tensorsph.neighbor_search import PairListCache


def exact_stretches(t):
    """Unit radius, unit initial strain. Incompressibility gives b=1/a.

    Energy conservation gives a'²(1+a^-4)=2. Invert its time integral
    independently of the particle discretization or time step.
    """
    nodes, weights = np.polynomial.legendre.leggauss(24)
    lo, hi = 1., 1.+math.sqrt(2)*t
    for _ in range(48):
        a = (lo+hi)/2
        z = 1+(1+nodes)*(a-1)/2
        elapsed = (a-1)/2 * np.dot(weights, np.sqrt((1+z**-4)/2))
        if elapsed < t:
            lo = a
        else:
            hi = a
    return [a, 1/a]


def run(cells, end=.4, courant=.20):
    if cells < 4 or not math.isfinite(end) or end <= 0 or not 0 < courant <= .3:
        raise ValueError("expected at least four cells per radius, end > 0 and CFL in (0, .3]")
    dx, c0 = 1/cells, 20.
    axis = (torch.arange(-cells, cells, dtype=torch.float64)+.5)*dx
    pos = torch.cartesian_prod(axis, axis)
    r2 = pos.square().sum(1)
    pos = pos[r2 < 1]
    # The initial Euler pressure (rho0/2)*(1-r²) and linear barotropic EOS.
    density = 1000*(1+.5*(1-r2[r2 < 1])/c0**2)
    mass = density*dx**2
    vel = pos*pos.new_tensor([1., -1.])
    moment0 = (mass[:, None]*pos.square()).sum(0)
    extent0 = pos.abs().amax(0)
    cache = PairListCache(4*dx, .4*dx)
    peak_density_error = peak_mach = 0.
    t, steps = 0., 0
    while t < end:
        dt = min(end-t, courant*2*dx / (c0 + float(vel.norm(dim=1).max())))
        pairs, _ = cache.get(pos)
        step = delta_plus_step(pos, vel, density, mass, dx=dx, dt=dt, sound_speed=c0,
                               reference_speed=1., gravity=0., viscosity=0., pairs=pairs)
        pos, vel, density = step.position, step.velocity, step.density
        peak_density_error = max(peak_density_error, float((density/1000-1).abs().max()))
        peak_mach = max(peak_mach, float(vel.norm(dim=1).max()) / c0)
        t += dt
        steps += 1
    reference = exact_stretches(end)
    stretch = ((mass[:, None]*pos.square()).sum(0)/moment0).sqrt()
    return dict(cells_per_radius=cells, end=end, courant=courant, particles=len(mass),
                steps=steps, analytic_stretches=reference, moment_stretches=stretch.tolist(),
                moment_relative_errors=(stretch/stretch.new_tensor(reference)-1).tolist(),
                envelope_stretches=(pos.abs().amax(0)/extent0).tolist(),
                peak_relative_density_deviation=peak_density_error, peak_mach=peak_mach,
                final_momentum_norm=float((mass[:, None]*vel).sum(0).norm()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--end", type=float, default=.4)
    parser.add_argument("--courant", type=float, default=.2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps([run(k, args.end, args.courant) for k in args.cells],
                     indent=2, allow_nan=False))
