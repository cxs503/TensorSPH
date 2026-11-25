"""Compare a periodic slab driven by uniform wall flux phasors with its thermal wave."""
import cmath
import json
import math

import torch

from tensorsph.heat_conduction_2d import solve_harmonic_slab_2d


def measure(n, mixed):
    k, capacity, omega = 2., 3., 5.
    west, east = 1.+.4j, -.3+.5j
    wave = cmath.sqrt(1j*omega*capacity/k)
    x = (torch.arange(n, dtype=torch.float64)+.5)/n
    source = torch.zeros((4, n), dtype=torch.complex128)
    if mixed:
        walls = (None, 0.)
        east = 0.
        exact = west*torch.sinh(wave*(1-x))/(k*wave*cmath.cosh(wave))
    else:
        walls = (None, None)
        exact = (west*torch.cosh(wave*(1-x))+east*torch.cosh(wave*x))/(k*wave*cmath.sinh(wave))
    u = solve_harmonic_slab_2d(source, 1/n, walls, omega, conductivity=k,
                               volumetric_heat_capacity=capacity, x_wall_inward_flux=(west, east))
    # Use a central particle for amplitude/phase, away from the fixed-temperature node.
    transfer = u[0,n//2]/exact[n//2]
    row = dict(n=n, max_error_K=float((u-exact).abs().max()),
               relative_amplitude_error=float(abs(transfer.abs()-1)),
               phase_error_rad=float(torch.angle(transfer).abs()))
    if not mixed:
        row["storage_balance_W_m2"] = float(abs(1j*omega*capacity*u.mean()-west-east))
    return row


if __name__ == "__main__":
    runs = []
    for mixed in (False, True):
        previous = None
        for n in (16, 32, 64, 128):
            row = measure(n, mixed)
            order = None if previous is None else math.log2(previous/row["max_error_K"])
            passed = (all(math.isfinite(value) for value in row.values())
                      and (order is None or order > 1.85)
                      and row.get("storage_balance_W_m2", 0.) < 1e-11)
            if n == 128:
                passed &= row["relative_amplitude_error"] < 2e-4 and row["phase_error_rad"] < 2e-4
            runs.append(dict(mixed=mixed, order=order, passed=passed, **row))
            previous = row["max_error_K"]
    print(json.dumps(runs, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in runs) else 1)
