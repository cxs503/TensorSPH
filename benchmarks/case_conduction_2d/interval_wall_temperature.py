"""Interval-mean surface temperatures for a translating linear slab profile."""
import json
from pathlib import Path
import subprocess

import numpy as np

from tensorsph.heat_conduction_2d import solve_transient_layered_slab


def measure(n, method, storing):
    times = np.array([0., .2, .47, .7])
    switch = .13
    dx = 1/n
    conductivity, slope, film, capacity = 1.7, .4, 2.1, .35
    flux = conductivity*slope
    rate = lambda t: 2. if t < switch else -1.
    base = lambda t: 1+2*t-3*np.maximum(t-switch, 0.)
    primitive = lambda t: t+t*t-1.5*np.maximum(t-switch, 0.)**2
    # At the east surface the reservoir supplies the inward Fourier flux
    # and, for a storing surface, its areal enthalpy change.
    east_air = lambda t: base(t)+slope+(flux+capacity*rate(t)*storing)/film
    _, heat, means = solve_transient_layered_slab(
        1+slope*(np.arange(n)+.5)*dx, times, dx, (None, None),
        conductivity=np.full(n, conductivity), volumetric_heat_capacity=1.,
        heat_source=rate, x_wall_inward_flux=(-flux, 0.),
        x_wall_films=(None, (film, east_air)),
        wall_areal_heat_capacity=(0., capacity if storing else 0.),
        initial_wall_temperatures=(None, 1+slope if storing else None),
        method=method, load_jump_times=[switch], return_x_heat=True,
        return_interval_wall_temperatures=True)
    exact_mean = np.diff(primitive(times))/np.diff(times)
    wall_error = np.max(np.abs(means.numpy()-exact_mean[:, None]-[0., slope]))
    heat_error = np.max(np.abs(heat.numpy()+flux*np.diff(times)[:, None]))
    return dict(n=n, method=method, storing=storing, wall_error_K=float(wall_error),
                section_heat_error_J_m2=float(heat_error), passed=bool(max(wall_error, heat_error)<1e-10))


if __name__ == "__main__":
    rows = [measure(n, method, storing) for n in (6, 12, 24)
            for method in ("theta", "sdirk2", "exponential", "exponential_quadratic")
            for storing in (False, True)]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    print(json.dumps(dict(commit=commit, runs=rows), indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
