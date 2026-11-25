"""Two-dimensional fixed-temperature pulse with independently decaying transverse modes."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import solve_transient_slab_2d


def measure(n,step_factor=1):
    start,stop,end = .013,.047,.1
    time = torch.linspace(0.,end,step_factor*n//2+1,dtype=torch.float64)
    x = (torch.arange(n,dtype=time.dtype)+.5)/n
    mode_y = torch.cos(2*math.pi*x+.25)
    def wall(t):
        return float(start<=t<stop)*(.7+.8*mode_y)
    field,heat = solve_transient_slab_2d(torch.zeros((n,n),dtype=x.dtype),time,1/n,(wall,0.),
                      wall_jump_times=(start,stop),rannacher=True,return_x_heat=True)
    j = math.pi*torch.arange(1,129,dtype=x.dtype)
    sine = torch.sin(x[:,None]*j)
    def response(beta):
        # sine coefficients of the steady wall lift are 2*j/(j²+beta²).
        rate = j.square()+beta**2
        return sine @ ((2*j/rate)*(torch.exp(-(end-stop)*rate)-torch.exp(-(end-start)*rate)))
    transverse = .8*response(2*math.pi)
    reference = .7*response(0.)+mode_y[:,None]*transverse
    recovered = 2*(field[-1]*mode_y[:,None]).mean(dim=0)
    return dict(n=n,steps=len(time)-1,max_error_K=float((field[-1]-reference).abs().max()),
                transverse_error_K=float((recovered-transverse).abs().max()),
                section_balance_J_m2=float((field.diff(dim=0).mean(dim=1)/n+heat.diff(dim=1)).abs().max()))


if __name__=="__main__":
    rows = []
    previous = {}
    for n in (16,32,64,128,256):
        row = measure(n)
        errors = {key:row[key] for key in ("max_error_K","transverse_error_K")}
        orders = {key:math.log2(previous[key]/val) for key,val in errors.items()} if previous else {}
        row.update(orders=orders,passed=row["section_balance_J_m2"]<1e-12 and
                   all(v>0 for v in orders.values()) and (n<256 or row["max_error_K"]<1e-4))
        rows.append(row)
        previous = errors
    time_check = measure(128,2)
    time_check["passed"] = time_check["max_error_K"]<1e-4 and time_check["section_balance_J_m2"]<1e-12
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows,time_check=time_check),indent=2,allow_nan=False))
    raise SystemExit(0 if time_check["passed"] and all(row["passed"] for row in rows) else 1)
