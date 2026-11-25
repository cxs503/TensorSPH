"""Bulk cooling in a slab with a driven temperature wall and an insulated end."""
import cmath
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import (
    solve_harmonic_slab_2d, solve_steady_slab_2d, solve_transient_slab_2d,
)


def measure(n):
    k,c,loss = .8,2.1,3.
    dx = 1/n
    faces = torch.arange(n+1,dtype=torch.float64)*dx
    x = faces[:-1]+dx/2
    source = torch.zeros((4,n),dtype=x.dtype)
    opts = dict(conductivity=k,heat_loss_coefficient=loss)
    errors = {}
    for name,omega in (("steady",0.),("harmonic",5.)):
        if omega:
            field,flux = solve_harmonic_slab_2d(source,dx,(1.,None),omega,volumetric_heat_capacity=c,
                                                return_x_flux=True,**opts)
        else:
            field,flux = solve_steady_slab_2d(source,dx,(1.,None),return_x_flux=True,**opts)
        wave = cmath.sqrt((loss+1j*omega*c)/k)
        ref = torch.cosh(wave*(1-x))/cmath.cosh(wave)
        current = k*wave*torch.sinh(wave*(1-faces))/cmath.cosh(wave)
        errors[name+"_temperature"] = float((field-ref).abs().max())
        errors[name+"_flux"] = float((flux-current).abs().max())

    # Homogeneous mixed walls select a sine eigenmode for the cooling transient.
    wave = math.pi/2
    rate = (loss+k*wave**2)/c
    initial = source+torch.sin(wave*x)
    times = torch.linspace(0.,.5,n//2+1,dtype=x.dtype)
    fields,heat = solve_transient_slab_2d(initial,times,dx,(0.,None),volumetric_heat_capacity=c,
                                         return_x_heat=True,**opts)
    amplitude = torch.exp(-rate*times)
    errors["transient_temperature"] = float((fields-initial*amplitude[:,None,None]).abs().max())
    dt = times.diff()[:,None]
    exact_heat = (-k*wave*torch.cos(wave*faces))*amplitude[:-1,None]*(-torch.expm1(-rate*dt))/rate
    errors["transient_flux"] = float(((heat-exact_heat)/dt).abs().max())
    mean = fields.mean(dim=1)
    budget = dx*(c*mean.diff(dim=0)+loss*dt*(mean[1:]+mean[:-1])/2)+heat.diff(dim=1)
    return errors,float(budget.abs().max())


if __name__=="__main__":
    rows = []
    previous = {}
    for n in (16,32,64,128):
        errors,residual = measure(n)
        orders = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
        passed = residual<1.e-12 and (n<128 or min(orders.values())>1.85)
        rows.append(dict(columns=n,errors=errors,orders=orders,max_cell_balance=residual,passed=passed))
        previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
