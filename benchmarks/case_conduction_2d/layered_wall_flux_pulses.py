"""Wall-flux pulses in two materials of equal effusivity, with a Fourier-series reference."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensorsph.heat_conduction_2d import solve_transient_layered_slab


def measure(n):
    dx = 1/n
    x = (np.arange(n)+.5)*dx
    edges = np.arange(n+1)*dx
    conductivity = np.where(x<.5,1.,4.)
    # In r=integral(dx/k), k*C=2 gives 2*T_t=T_rr across the perfect interface.
    capacity = 2./conductivity
    def r(z):
        return np.minimum(z,.5)+np.maximum(z-.5,0.)/4.
    length = r(1.)
    modes = np.pi*np.arange(1,81)
    def step(t,z):
        # Temperature and integral(C*T dx) from the driven wall to z.
        y = np.asarray(z)[...,None]/length
        decay = np.exp(-modes**2*t/(2*length**2))
        temp = t/(2*length)+length*((y*y/2-y+1/3)[...,0]
                    -2*(np.cos(y*modes)*decay/modes**2).sum(axis=-1))
        stored = t*y[...,0]+2*length**2*((y**3/6-y**2/2+y/3)[...,0]
                    -2*(np.sin(y*modes)*decay/modes**3).sum(axis=-1))
        return temp,stored
    pulses = ((.027,1.3,False),(.083,-1.3,False),(.151,-.8,True),(.203,.8,True))
    def inward(t,right):
        return sum(q for start,q,side in pulses if t>=start and side==right)
    times = np.linspace(0.,.36,3*n+1)
    states,heat = solve_transient_layered_slab(np.zeros(n),times,dx,(None,None),
            conductivity=conductivity,volumetric_heat_capacity=capacity,
            x_wall_inward_flux=(lambda t:inward(t,False),lambda t:inward(t,True)),
            load_jump_times=[start for start,_,_ in pulses],return_x_heat=True,rannacher=True)
    cumulative = heat.cumsum(dim=0).numpy()
    errors = dict(temperature=0.,section_heat=0.,net_heat=0.)
    for idx in (n,2*n,3*n):
        t = times[idx]
        temperature = np.zeros(n)
        transport = np.zeros(n+1)
        supplied = 0.
        for start,q,right in pulses:
            age = t-start
            if age<=0:
                continue
            z,zf = (length-r(x),length-r(edges)) if right else (r(x),r(edges))
            temperature += q*step(age,z)[0]
            transport += (-1 if right else 1)*q*(age-step(age,zf)[1])
            supplied += q*age
        errors["temperature"] = max(errors["temperature"],float(abs(states[idx].numpy()-temperature).max()))
        errors["section_heat"] = max(errors["section_heat"],float(abs(cumulative[idx-1]-transport).max()))
        errors["net_heat"] = max(errors["net_heat"],float(abs(dx*(capacity @ states[idx].numpy())-supplied)))
    return errors


if __name__=="__main__":
    rows = []
    for n in (16,32,64,128):
        errors = measure(n)
        orders = {k:math.log2(rows[-1]["errors"][k]/errors[k]) for k in ("temperature","section_heat")} if rows else {}
        passed = errors["net_heat"]<1e-11 and (n<128 or min(orders.values())>1.8)
        rows.append(dict(columns=n,errors=errors,orders=orders,passed=passed))
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
