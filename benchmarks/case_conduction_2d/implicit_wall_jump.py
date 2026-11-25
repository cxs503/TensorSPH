"""Sudden unit-temperature step at the west wall of a cold unit slab."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import step_transient_slab_2d


def measure(n, damp_start):
    dx, end = 1/n, .04
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    steps = n//4
    dt = end/steps
    field = torch.zeros((1,n),dtype=x.dtype)
    first_overshoot = 0.
    wall_heat = torch.zeros(2,dtype=x.dtype)
    for i in range(steps):
        substeps = (1.,1.) if damp_start and i==0 else (.5,)
        for theta in substeps:
            sub_dt = dt/len(substeps)
            field,flux = step_transient_slab_2d(field,sub_dt,dx,(1.,0.),theta=theta,return_x_flux=True)
            wall_heat += sub_dt*flux[[0,-1]]
        if i==0:
            first_overshoot = max(0.,float(field.max())-1.)
    j = torch.arange(1,129,dtype=x.dtype)*math.pi
    # Subtract the steady affine profile and evolve its sine coefficients.
    exact = 1.-x-(2*torch.exp(-end*j.square())/j)@torch.sin(j[:,None]*x)
    return dict(n=n,steps=steps,damp_start=damp_start,first_overshoot_K=first_overshoot,
                final_error_K=float((field-exact).abs().max()),
                energy_residual_J_m2=float(abs(dx*field.sum()-wall_heat[0]+wall_heat[1])))


if __name__ == "__main__":
    rows = []
    for damp_start in (False,True):
        previous = None
        for n in (32,64,128,256):
            row = measure(n,damp_start)
            error = row["final_error_K"]
            order = None if previous is None else math.log2(previous/error)
            row.update(order=order,passed=row["energy_residual_J_m2"]<1e-12 and
                       (not damp_start or (row["first_overshoot_K"]==0. and (order is None or order>1.8))))
            rows.append(row)
            previous = error
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
