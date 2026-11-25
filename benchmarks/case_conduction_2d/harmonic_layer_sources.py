"""Three internally heated layers with insulated ends and two massless thermal contacts."""
import argparse
import cmath
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import (
    layered_slab_contact_temperatures, solve_harmonic_layered_slab,
)


def measure(n, losses=(0.,0.,0.)):
    dx, omega = 1/n, 5.
    counts = (n//4, n//2, n//4)
    conductivities = (1., 4., 2.)
    capacities = (3., 1., 2.)
    sources = (1.+.5j, -.3+.2j, .4-.7j)
    resistances = (.18, .42)

    def propagate(t, q, distance, j):
        k, c = conductivities[j], capacities[j]
        shift = losses[j]+1j*omega*c
        wave = cmath.sqrt(shift/k)
        particular = sources[j]/shift
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
    loss = profile(losses, torch.float64)
    resistance = torch.zeros(n-1, dtype=torch.float64)
    cuts = [counts[0], n-counts[-1]]
    for face, value in zip(cuts, resistances):
        resistance[face-1] = value
    u,sections = solve_harmonic_layered_slab(source,dx,(None,None),omega,conductivity=k,
        volumetric_heat_capacity=c,contact_resistance_x=resistance,heat_loss_coefficient=loss,return_x_flux=True)
    flux = [complex(sections[i]) for i in cuts]
    left,right = layered_slab_contact_temperatures(u,sections,dx,k)
    trace_indices = torch.tensor(cuts)-1
    observed = torch.stack((left[trace_indices],right[trace_indices]),dim=1)
    expected = torch.tensor([[propagate(*starts[j],counts[j]*dx,j)[0],starts[j+1][0]]
                             for j in range(2)],dtype=u.dtype)
    jump = observed[:,0]-observed[:,1]
    jump_reference = expected[:,0]-expected[:,1]
    faces, powers = [0, *cuts, n], [0., *flux, 0.]
    imbalance = []
    for j, (a,b) in enumerate(zip(faces, faces[1:])):
        storage = dx*((loss[a:b]+1j*omega*c[a:b])*u[a:b]-source[a:b]).sum()
        imbalance.append(float(abs(storage-powers[j]+powers[j+1])))
    return dict(n=n, temperature_error_K=float(abs(u-exact).max()),
                interface_flux_errors_W_m2=[abs(q-r) for q,r in zip(flux, reference_flux)],
                trace_error_K=float(abs(observed-expected).max()),
                trace_amplitude_error_K=float(abs(abs(observed)-abs(expected)).max()),
                trace_phase_error_rad=float(abs(torch.angle(observed*expected.conj())).max()),
                jump_error_K=float(abs(jump-jump_reference).max()),layer_residuals_W_m2=imbalance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-loss",action="store_true",help="apply distinct bulk sinks in the three layers")
    args = parser.parse_args()
    rows, previous = [], None
    for n in (16,32,64,128):
        row = measure(n,(2.,0.,7.)) if args.bulk_loss else measure(n)
        errors = [row["temperature_error_K"], *row["interface_flux_errors_W_m2"],
                  *(row[k] for k in ("trace_error_K","trace_amplitude_error_K","trace_phase_error_rad","jump_error_K"))]
        orders = [] if previous is None else [math.log2(a/b) for a,b in zip(previous, errors)]
        row["orders"] = orders
        row["passed"] = (all(math.isfinite(v) for v in errors)
                         and max(row["layer_residuals_W_m2"]) < 1e-10
                         and all(order > 1.8 for order in orders))
        rows.append(row)
        previous = errors
    report = rows
    if args.bulk_loss:
        rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
        report = dict(commit=rev,heat_loss_coefficients=[2.,0.,7.],runs=rows)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
