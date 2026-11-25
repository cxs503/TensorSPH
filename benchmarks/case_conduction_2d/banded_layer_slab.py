"""Thousands of columns spanning two resistive contacts under harmonic heat flow."""
import json

import torch

from tensorsph.heat_conduction_2d import solve_harmonic_layered_slab


def measure(n):
    dx = 1/n
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    k = torch.where(x<.25, 1., torch.where(x<.75, 4., 2.)).double()
    c = 1.+x
    resistance = torch.zeros(n-1, dtype=x.dtype)
    resistance[n//4-1], resistance[3*n//4-1] = .15, .4
    path = (dx/k).cumsum(0)-.5*dx/k
    path[n//4:] += .15
    path[3*n//4:] += .4
    q = .7-.4j
    walls = (1.+.3j, 1.+.3j-q*(.25+.5/4+.25/2+.15+.4))
    exact = walls[0]-q*path
    # Heating balances local storage. The remaining rightward Fourier flux
    # is uniform, even though conductivity, temperature and capacity vary.
    source = 5j*c*exact
    opts = dict(conductivity=k, volumetric_heat_capacity=c, contact_resistance_x=resistance)
    u = solve_harmonic_layered_slab(source, dx, walls, 5., backend="banded", **opts)
    row = dict(columns=n, max_error_K=float((u-exact).abs().max()))
    if n <= 256:
        dense = solve_harmonic_layered_slab(source, dx, walls, 5., **opts)
        row["dense_difference_K"] = float((u-dense).abs().max())
    row["passed"] = row["max_error_K"] < 1e-7 and row.get("dense_difference_K", 0.) < 1e-10
    return row


if __name__ == "__main__":
    rows = [measure(n) for n in (64,256,1024,4096,8192)]
    print(json.dumps(rows, indent=2, allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
