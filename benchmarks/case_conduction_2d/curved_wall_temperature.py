"""Spatial refinement of interval means and integrated heat for a cubic slab profile."""
import json
from pathlib import Path
import subprocess

import numpy as np

from tensorsph.heat_conduction_2d import solve_transient_layered_slab


def measure(n, boundary):
    times = np.array([0., .09, .20, .35])
    switch = .13
    dx = 1/n
    faces = np.linspace(0., 1., n+1)
    centers = (faces[1:]+faces[:-1])/2
    k, film, capacity = 1.7, 2.1, .35

    # T(x,t) = base(t) + P(x). The source is base'(t) - k P''(x).
    def profile(x):
        return .4*x+.3*x**2-.12*x**3

    def heat_flux(x):
        return -k*(.4+.6*x-.36*x**2)

    def base(t):
        return 1+2*t-3*np.maximum(t-switch, 0.)

    def rate(t):
        return 2. if t < switch else -1.

    primitive = times+times**2-1.5*np.maximum(times-switch, 0.)**2
    east_capacity = capacity if boundary == "storing" else 0.

    def east_air(t):
        return base(t)+profile(1.)+(-heat_flux(1.)+east_capacity*rate(t))/film

    west = base if boundary == "fixed" else None
    west_flux = 0. if west else heat_flux(0.)
    field, heat, wall, external, mean = solve_transient_layered_slab(
        base(0.)+profile(centers), times, dx, (west, None),
        conductivity=np.full(n, k), volumetric_heat_capacity=1.,
        heat_source=lambda t: rate(t)-k*(.6-.72*centers),
        x_wall_inward_flux=(west_flux, 0.), x_wall_films=(None, (film, east_air)),
        wall_areal_heat_capacity=(0., east_capacity),
        initial_wall_temperatures=(None, base(0.)+profile(1.) if east_capacity else None),
        load_jump_times=[switch], method="exponential", return_x_heat=True,
        return_wall_temperatures=True, return_external_heat=True, return_interval_wall_temperatures=True)
    dt = np.diff(times)
    field_error = field.numpy()-profile(centers)-base(times)[:, None]
    heat_error = heat.numpy()-dt[:, None]*heat_flux(faces)
    wall_error = mean.numpy()-np.diff(primitive)[:, None]/dt[:, None]-profile(faces[[0,-1]])
    source_heat = dx*(np.diff(base(times))[:, None]-k*dt[:, None]*(.6-.72*centers))
    residual = dx*np.diff(field.numpy(), axis=0)+np.diff(heat.numpy(), axis=1)-source_heat
    east_balance = east_capacity*np.diff(wall.numpy()[:,1])-external.numpy()[:,1]-heat.numpy()[:,-1]
    external_error = external.numpy()[:,1]-(east_capacity*np.diff(base(times))-dt*heat_flux(1.))
    return dict(east_external_heat_J_m2=float(abs(external_error).max()), field_K=float(abs(field_error).max()), wall_mean_K=float(abs(wall_error).max()),
                section_heat_J_m2=float(abs(heat_error).max())), float(abs(residual).max()), float(abs(east_balance).max())


if __name__ == "__main__":
    rows = []
    for boundary in ("fixed", "flux", "storing"):
        previous = None
        for n in (8, 16, 32, 64):
            errors, residual, east_balance = measure(n, boundary)
            orders = {k: float(np.log2(previous[k]/v)) for k,v in errors.items()} if previous else {}
            rows.append(dict(n=n, west_east={"fixed":"temperature/film", "flux":"flux/film",
                             "storing":"flux/storing film"}[boundary], errors=errors, orders=orders,
                             cell_balance_J_m2=residual, east_balance_J_m2=east_balance,
                             passed=max(residual,east_balance)<1e-11 and all(v>1.8 for v in orders.values())))
            previous = errors
    version = subprocess.check_output(["git","rev-parse","HEAD"], cwd=Path(__file__).parent, text=True).strip()
    print(json.dumps(dict(commit=version, runs=rows), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
