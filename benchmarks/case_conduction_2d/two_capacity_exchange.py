"""Time refinement against two insulated lumped heat capacities coupled by a resistance."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import solve_transient_layered_slab


def measure(steps, theta):
    dx, resistance = .35,.16
    k = torch.tensor([.7,2.3],dtype=torch.float64)
    c = torch.tensor([1.2,4.1],dtype=k.dtype)
    initial = torch.tensor([4.,.5],dtype=k.dtype)
    mass = c*dx
    # The centre-to-centre resistance includes the two half cells and the contact.
    conductance = 1/(resistance+dx/(2*k[0])+dx/(2*k[1]))
    rate = conductance*(1/mass).sum()
    equilibrium = (mass @ initial)/mass.sum()
    time = torch.linspace(0.,float(3/rate),steps+1,dtype=k.dtype)
    exact = equilibrium+(initial-equilibrium)*torch.exp(-rate*time[:,None])
    field = solve_transient_layered_slab(initial,time,dx,(None,None),conductivity=k,
                volumetric_heat_capacity=c,contact_resistance_x=[resistance],theta=theta,h_factor=.8)
    total = field @ mass
    transported = mass[1]*(field[:,1]-initial[1])
    ref = conductance*(initial[0]-initial[1])*(-torch.expm1(-rate*time))/rate
    return dict(temperature=float((field-exact).abs().max()),
                transferred_heat=float((transported-ref).abs().max()),
                energy_drift=float((total-total[0]).abs().max()))


if __name__=="__main__":
    rows = []
    for theta in (.5,1.):
        previous = {}
        for steps in (8,16,32,64):
            errors = measure(steps,theta)
            orders = {k:math.log2(previous[k]/errors[k]) for k in ("temperature","transferred_heat") if previous}
            passed = errors["energy_drift"]<1e-12 and (steps<64 or min(orders.values())>(1.9 if theta==.5 else .95))
            rows.append(dict(steps=steps,theta=theta,errors=errors,orders=orders,passed=passed))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
