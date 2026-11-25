"""Temperature and cumulative section heat for two decaying modes in a periodic slab."""
import json
import math
from pathlib import Path
import subprocess

import torch
from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, solve_transient_slab_2d, symbol_rate


def measure(n,steps,discrete):
    dx = 1/n
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    times = torch.linspace(0.,.04,steps+1,dtype=x.dtype)
    profile = torch.sin(math.pi*x)
    row = .4*torch.cos(2*math.pi*x)[:,None]
    states,heat = solve_transient_slab_2d(profile*(1+row),times,dx,(0.,0.),
                                         method="sdirk2",return_x_heat=True)
    if discrete:
        factor = lattice_diffusion_factor_2d(1.3,dx,mode="moment")
        rates = [symbol_rate(dx,1.3*dx,math.pi,w)/factor for w in (0.,2*math.pi)]
    else:
        rates = [math.pi**2,5*math.pi**2]
    amplitudes = [torch.exp(-r*times)[:,None,None] for r in rates]
    exact = profile*(amplitudes[0]+row*amplitudes[1])
    cut = n//4
    if discrete:
        # By symmetry there is no mean power at x=1/2. The integral
        # crossing x=1/4 equals the change in storage between the two cuts.
        expected_heat = dx*profile[cut:n//2].sum()*torch.expm1(-rates[0]*times[1:])
    else:
        expected_heat = math.cos(math.pi/4)/math.pi*torch.expm1(-rates[0]*times[1:])
    return dict(cells=n,steps=steps,temperature_error_K=float(abs(states-exact).max()),
                cut_heat_error_J_m2=float(abs(heat[:,cut].cumsum(0)-expected_heat).max()),
                column_residual_J_m2=float(abs(dx*states.diff(dim=0).mean(1)+heat.diff(dim=1)).max()))


if __name__=="__main__":
    rows = []
    for study in ("time","space"):
        previous = None
        for level in (1,2,4):
            row = measure(12,4*level,True) if study=="time" else measure(8*level,64,False)
            metrics = ("temperature_error_K","cut_heat_error_J_m2")
            orders = {k:math.log2(previous[k]/row[k]) for k in metrics} if previous else {}
            row.update(study=study,orders=orders,passed=row["column_residual_J_m2"]<1e-11 and
                       (level<4 or min(orders.values())>1.9))
            rows.append(row)
            previous = row
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
