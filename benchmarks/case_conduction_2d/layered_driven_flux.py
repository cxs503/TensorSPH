"""Mixed and double flux walls in a driven two-material continuum with a contact jump."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import solve_transient_layered_slab


def measure(n, mixed, smoothing):
    dx, split, resistance = 1/n,.5,.18
    edges = torch.arange(n+1,dtype=torch.float64)*dx
    x = edges[:-1]+dx/2
    k = torch.where(x<split,1.,3.).to(x)
    c = 1.+2.*(x>=split).to(x)
    contact = torch.zeros_like(x[:-1])
    contact[n//2-1] = resistance
    # The rightward Fourier flux is sin(6t)*(1-1.4x).
    def primitive(z):
        return z-.7*z*z
    def profile(z):
        left = -primitive(z)
        right = -primitive(split)-resistance*(1-1.4*split)-(primitive(z)-primitive(split))/3
        return torch.where(z<split,left,right)
    f = profile(x)
    times = torch.linspace(0.,.3,n+1,dtype=x.dtype)
    def source(t):
        return c*(-2*math.sin(2*t)+6*math.cos(6*t)*f)-1.4*math.sin(6*t)
    west = (lambda t:math.cos(2*t)) if mixed else None
    west_flux = 0. if mixed else lambda t:math.sin(6*t)
    states,heat = solve_transient_layered_slab(torch.ones_like(x),times,dx,(west,None),
        conductivity=k,volumetric_heat_capacity=c,heat_source=source,contact_resistance_x=contact,
        x_wall_inward_flux=(west_flux,lambda t:.4*math.sin(6*t)),return_x_heat=True,h_factor=smoothing)
    exact = torch.cos(2*times[:,None])+torch.sin(6*times[:,None])*f
    dt = times.diff()[:,None]
    exact_heat = (-torch.cos(6*times).diff()[:,None]/6)*(1.-1.4*edges)
    source_values = torch.stack([source(float(t)) for t in times])
    balance = dx*c*states.diff(dim=0)+heat.diff(dim=1)-.5*dx*dt*(source_values[:-1]+source_values[1:])
    return dict(temperature=float((states-exact).abs().max()),
                mean_flux=float(((heat-exact_heat)/dt).abs().max()),
                balance=float(balance.abs().max()))


if __name__=="__main__":
    rows = []
    for mixed in (False,True):
        for smoothing in (.8,1.3):
            previous = {}
            for n in (16,32,64,128):
                errors = measure(n,mixed,smoothing)
                orders = {k:math.log2(previous[k]/errors[k]) for k in ("temperature","mean_flux") if previous}
                passed = errors["balance"]<1e-11 and (n<128 or min(orders.values())>1.8)
                rows.append(dict(columns=n,mixed=mixed,h_factor=smoothing,errors=errors,orders=orders,passed=passed))
                previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
