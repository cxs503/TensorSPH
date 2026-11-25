"""Three internally heated layers with insulated ends and two massless thermal contacts."""
import cmath
import json
import math

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, lattice_offsets, solve_harmonic_layered_slab,
    stencil_x_face_power_2d,
)


def measure(n):
    dx, omega = 1/n, 5.
    counts = (n//4, n//2, n//4)
    conductivities = (1., 4., 2.)
    capacities = (3., 1., 2.)
    sources = (1.+.5j, -.3+.2j, .4-.7j)
    resistances = (.18, .42)

    def propagate(t, q, distance, j):
        k, c = conductivities[j], capacities[j]
        wave = cmath.sqrt(1j*omega*c/k)
        particular = sources[j]/(1j*omega*c)
        ch, sh = cmath.cosh(wave*distance), cmath.sinh(wave*distance)
        return (particular+ch*(t-particular)-sh*q/(k*wave),
                ch*q-k*wave*sh*(t-particular))

    def traces(t):
        q = 0j
        starts, fluxes = [], []
        for j, count in enumerate(counts):
            starts.append((t, q))
            t, q = propagate(t, q, count*dx, j)
            fluxes.append(q)
            if j < len(resistances):
                t -= resistances[j]*q
        return starts, fluxes

    # The terminal flux depends affinely on the unknown insulated left-wall temperature.
    q0, q1 = (traces(t)[1][-1] for t in (0., 1.))
    starts, reference_flux = traces(-q0/(q1-q0))
    exact = torch.tensor([propagate(*starts[j], (i+.5)*dx, j)[0]
                          for j, count in enumerate(counts) for i in range(count)],
                         dtype=torch.complex128)

    def profile(values, dtype):
        return torch.cat([torch.full((count,), value, dtype=dtype)
                          for count, value in zip(counts, values)])

    k, c = (profile(values, torch.float64) for values in (conductivities, capacities))
    source = profile(sources, torch.complex128)
    resistance = torch.zeros(n-1, dtype=torch.float64)
    cuts = [counts[0], n-counts[-1]]
    for face, value in zip(cuts, resistances):
        resistance[face-1] = value
    u = solve_harmonic_layered_slab(source, dx, (None,None), omega, conductivity=k,
                                    volumetric_heat_capacity=c, contact_resistance_x=resistance)
    offsets = lattice_offsets(dx, 1.3*dx)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    flux = [complex(stencil_x_face_power_2d(u[None,:], offsets, dx, face, (None,None),
                    conductivity=k, layered_x=True, contact_resistance_x=resistance))/(dx*factor)
            for face in cuts]
    faces, powers = [0, *cuts, n], [0., *flux, 0.]
    imbalance = []
    for j, (a,b) in enumerate(zip(faces, faces[1:])):
        storage = dx*(1j*omega*c[a:b]*u[a:b]-source[a:b]).sum()
        imbalance.append(float(abs(storage-powers[j]+powers[j+1])))
    return dict(n=n, temperature_error_K=float(abs(u-exact).max()),
                interface_flux_errors_W_m2=[abs(q-r) for q,r in zip(flux, reference_flux)],
                layer_residuals_W_m2=imbalance)


if __name__ == "__main__":
    rows, previous = [], None
    for n in (16,32,64,128):
        row = measure(n)
        errors = [row["temperature_error_K"], *row["interface_flux_errors_W_m2"]]
        orders = [] if previous is None else [math.log2(a/b) for a,b in zip(previous, errors)]
        row["orders"] = orders
        row["passed"] = (all(math.isfinite(v) for v in errors)
                         and max(row["layer_residuals_W_m2"]) < 1e-10
                         and all(order > 1.8 for order in orders))
        rows.append(row)
        previous = errors
    print(json.dumps(rows, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
