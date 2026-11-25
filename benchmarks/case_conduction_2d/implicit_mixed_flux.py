"""Transient mixed-wall slab driven by a travelling transverse heat-flux wave."""
import cmath
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import step_transient_slab_2d


def measure(n, flux_at_east):
    k, capacity, omega = 1.7, 2.3, 5.
    dx, dt, steps = 1/n, .3/n, n
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    beta = 2*math.pi
    row = torch.exp(1j*beta*x)[:,None]
    wave = cmath.sqrt(1j*omega*capacity/k)
    transverse = cmath.sqrt(beta**2+wave**2)
    # Superpose two exact solutions of iω C T = k (T_xx + T_yy).
    distance = 1-x if flux_at_east else x
    amplitude = torch.exp(-wave*distance)+.4*row*torch.exp(-transverse*distance)
    inward = k*(wave+.4*row[:,0]*transverse)
    fixed = cmath.exp(-wave)+.4*row[:,0]*cmath.exp(-transverse)
    field = amplitude.real
    balance = 0.
    input_heat = torch.zeros(2,dtype=x.dtype)
    for step in range(steps):
        stage = (cmath.exp(1j*omega*dt*step)+cmath.exp(1j*omega*dt*(step+1)))/2
        wall,flux = (stage*fixed).real,(stage*inward).real
        updated,sections = step_transient_slab_2d(field,dt,dx,(wall,None) if flux_at_east else (None,wall),
                    conductivity=k,volumetric_heat_capacity=capacity,
                    x_wall_inward_flux=(0.,flux) if flux_at_east else (flux,0.),return_x_flux=True)
        residual = capacity*dx*(updated-field).mean(dim=0)+dt*sections.diff()
        balance = max(balance,float(residual.abs().max()))
        input_heat += dt*sections[[0,-1]]*sections.new_tensor([1.,-1.])
        field = updated
    phase = cmath.exp(1j*omega*dt*steps)
    integral = (phase-1)/(1j*omega)
    exact_energy = (capacity*(phase-1)*(1-cmath.exp(-wave))/wave).real
    delta_energy = capacity*float((field-amplitude.real).mean())
    exact_flux_heat = (integral*k*wave).real
    exact_fixed_heat = -(integral*k*wave*cmath.exp(-wave)).real
    return dict(n=n,flux_at_east=flux_at_east,
                temperature_error_K=float((field-(phase*amplitude).real).abs().max()),
                imposed_heat_error_J_m2=abs(float(input_heat[int(flux_at_east)])-exact_flux_heat),
                fixed_wall_heat_error_J_m2=abs(float(input_heat[int(not flux_at_east)])-exact_fixed_heat),
                storage_error_J_m2=abs(delta_energy-exact_energy),column_balance_J_m2=balance)


if __name__ == "__main__":
    rows = []
    for east in (False,True):
        previous = {}
        for n in (8,16,32,64):
            row = measure(n,east)
            errors = {k:v for k,v in row.items() if "_error_" in k}
            rates = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            row.update(orders=rates,passed=all(v>1.8 for v in rates.values()) and row["column_balance_J_m2"]<1e-12)
            rows.append(row)
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
