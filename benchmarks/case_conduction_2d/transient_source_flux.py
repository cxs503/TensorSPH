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
    # T=t*(.5+x)+t²*cos(2πy)*sin(πx/2), with unit conductivity and heat capacity.
    mode = torch.cos(2*math.pi*x)[:,None]*torch.sin(.5*math.pi*x)
    rate = (4+.25)*math.pi**2
    def source(t):
        return .5+x+(2*t+rate*t*t)*mode
    fields,heat = solve_transient_slab_2d(torch.zeros_like(mode),times,dx,(lambda t:.5*t,None),
                     heat_source=source,x_wall_inward_flux=(0.,lambda t:t),return_x_heat=True)
    t = times[:,None,None]
    exact = t*(.5+x)+t*t*mode
    budget = dx*fields.diff(dim=0).mean(dim=1)+heat.diff(dim=1)-dx*times.diff()[:,None]*(.5+x)
    east = -.5*times.square().diff()
    return dict(cells=n,max_temperature_error_K=float((fields-exact).abs().max()),
                section_balance_J_m2=float(budget.abs().max()),
                east_heat_error_J_m2=float((heat[:,-1]-east).abs().max()))


if __name__=="__main__":
    rows = []
    for n in (16,32,64,128):
        row = measure(n)
        error = row["max_temperature_error_K"]
        order = math.log2(rows[-1]["max_temperature_error_K"]/error) if rows else None
        row.update(order=order,passed=(order is None or order>1.5)
                   and max(row["section_balance_J_m2"],row["east_heat_error_J_m2"])<1e-12)
        rows.append(row)
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
