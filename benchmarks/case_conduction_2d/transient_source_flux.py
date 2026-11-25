"""Time-dependent volumetric heating with one temperature wall and one driven flux wall."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import solve_transient_slab_2d


def measure(n):
    dx = 1/n
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    times = torch.linspace(0.,.15,n//2+1,dtype=x.dtype)
    # The quadratic mean profile reverses the rightward flux at the midplane.
    profile = .5+x-x.square()
    mode = torch.cos(2*math.pi*x)[:,None]*torch.sin(.5*math.pi*x)
    rate = (4+.25)*math.pi**2
    def source(t):
        return profile+2*t+(2*t+rate*t*t)*mode
    fields,heat = solve_transient_slab_2d(torch.zeros_like(mode),times,dx,(lambda t:.5*t,None),
                     heat_source=source,x_wall_inward_flux=(0.,lambda t:-t),return_x_heat=True)
    t = times[:,None,None]
    exact = t*profile+t*t*mode
    dt = times.diff()[:,None]
    mean_source = profile+(times[:-1]+times[1:])[:,None]
    budget = dx*fields.diff(dim=0).mean(dim=1)+heat.diff(dim=1)-dx*dt*mean_source
    faces = torch.arange(n+1,dtype=x.dtype)*dx
    # Integrate Fourier's law over y and each output interval.
    exact_heat = .5*times.square().diff()[:,None]*(2*faces-1)
    return dict(cells=n,max_temperature_error_K=float((fields-exact).abs().max()),
                section_balance_J_m2=float(budget.abs().max()),
                section_flux_error_W_m2=float(((heat-exact_heat)/dt).abs().max()),
                east_heat_error_J_m2=float((heat[:,-1]-exact_heat[:,-1]).abs().max()))


if __name__=="__main__":
    rows = []
    for n in (16,32,64,128):
        row = measure(n)
        error = row["max_temperature_error_K"]
        order = math.log2(rows[-1]["max_temperature_error_K"]/error) if rows else None
        flux_order = math.log2(rows[-1]["section_flux_error_W_m2"]/row["section_flux_error_W_m2"]) if rows else None
        row.update(order=order,flux_order=flux_order,passed=(order is None or min(order,flux_order)>1.5)
                   and max(row["section_balance_J_m2"],row["east_heat_error_J_m2"])<1e-12)
        rows.append(row)
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
