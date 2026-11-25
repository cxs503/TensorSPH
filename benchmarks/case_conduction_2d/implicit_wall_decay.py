"""Refinement of implicit slab cooling with nonuniform, time-dependent wall temperatures."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import (
    lattice_diffusion_factor_2d, step_transient_slab_2d, symbol_rate,
)


def measure(n, steps, theta, temporal):
    dx, end = 1/n, .04
    dt = end/steps
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    ky, kx = 2*math.pi, math.pi
    transverse = torch.cos(ky*x)[:,None]
    # A phase offset gives nonzero row-dependent temperatures at both walls.
    phase = 0. if temporal else .4
    mode = transverse*torch.sin(kx*x+phase)
    rate = (symbol_rate(dx,1.3*dx,kx,ky)/lattice_diffusion_factor_2d(1.3,dx,mode="moment")
            if temporal else kx**2+ky**2)
    background = 2.-x
    field = background+mode
    balance = 0.
    for i in range(steps):
        avg = (1-theta)*math.exp(-rate*i*dt)+theta*math.exp(-rate*(i+1)*dt)
        left = avg*math.sin(phase)*transverse[:,0]
        updated,flux = step_transient_slab_2d(field,dt,dx,(2.+left,1.-left),theta=theta,return_x_flux=True)
        balance = max(balance,float((dx*(updated-field).mean(dim=0)+dt*flux.diff()).abs().max()))
        field = updated
    error = (field-background-math.exp(-rate*end)*mode).abs()
    return dict(n=n,steps=steps,theta=theta,reference="semidiscrete" if temporal else "continuum",
                max_error_K=float(error.max()),wall_adjacent_error_K=float(error[:,[0,-1]].max()),
                balance_J_m2=balance)


if __name__ == "__main__":
    rows = []
    for temporal,theta in ((False,.5),(True,.5),(True,1.)):
        previous = None
        for size in (8,16,32,64):
            row = measure(32 if temporal else size,size,theta,temporal)
            error = row["max_error_K"]
            rate = None if previous is None else math.log2(previous/error)
            row.update(order=rate,passed=(rate is None or rate>(.9 if theta==1 else 1.8))
                       and row["balance_J_m2"]<1e-12)
            rows.append(row)
            previous = error
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
