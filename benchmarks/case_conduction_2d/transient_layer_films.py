"""Two convective films on a perfectly bonded slab with matched thermal effusivities."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import solve_transient_layered_slab


def measure(n):
    dx = 1/n
    edges = torch.arange(n+1,dtype=torch.float64)*dx
    x = edges[:-1]+dx/2
    k = 1.+3.*(x>=.5)
    c = 2./k.to(x)
    def resistance(z):
        return z.clamp(max=.5)+(z-.5).clamp(min=0)/4
    r = resistance(x)
    length = float(resistance(edges)[-1])
    phase,wave = .37,1.2/length
    rate = wave**2/2
    h0,h1 = wave*math.tan(phase),wave*math.tan(wave*length-phase)
    # T = A(t)+D(t)*r+exp(-rate*t)*cos(wave*r-phase).
    # The last term satisfies both homogeneous films. The affine terms
    # determine the independent ambient histories and the volume source.
    def west(t):
        return math.sin(3*t)-.4*math.cos(2*t)/h0
    def east(t):
        return math.sin(3*t)+.4*math.cos(2*t)*(length+1/h1)
    def source(t):
        return c*(3*math.cos(3*t)-.8*math.sin(2*t)*r)
    mode = torch.cos(wave*r-phase)
    times = torch.linspace(0.,.45,n+1,dtype=x.dtype)
    states,heat = solve_transient_layered_slab(mode+.4*r,times,dx,(None,None),conductivity=k,
            volumetric_heat_capacity=c,heat_source=source,x_wall_films=((h0,west),(h1,east)),return_x_heat=True)
    t = times[:,None]
    exact = torch.sin(3*t)+.4*torch.cos(2*t)*r+torch.exp(-rate*t)*mode
    dt = times.diff()
    avg_d = .2*torch.sin(2*times).diff()/dt
    avg_b = -torch.exp(-rate*times).diff()/(rate*dt)
    re = resistance(edges)
    exact_flux = -avg_d[:,None]+wave*avg_b[:,None]*torch.sin(wave*re-phase)
    flux = heat/dt[:,None]
    avg_a = -torch.cos(3*times).diff()/(3*dt)
    wall_error = 0.
    for side,ambient,h,inward in ((0,west,h0,1),(-1,east,h1,-1)):
        data = times.new_tensor([ambient(float(ti)) for ti in times])
        tw = .5*(data[1:]+data[:-1])-inward*flux[:,side]/h
        reference = avg_a+avg_d*re[side]+avg_b*torch.cos(wave*re[side]-phase)
        wall_error = max(wall_error,float((tw-reference).abs().max()))
    source_values = torch.stack([source(float(ti)) for ti in times])
    residual = dx*c*states.diff(dim=0)+heat.diff(dim=1)-dx*dt[:,None]*(source_values[:-1]+source_values[1:])/2
    # Integrate C*T dx = 2*T dr analytically on each layer.
    cuts = re[[0,n//2,-1]]
    stored = 2*(torch.sin(3*t)*cuts.diff()+.2*torch.cos(2*t)*cuts.square().diff()
                +torch.exp(-rate*t)*torch.sin(wave*cuts-phase).diff()/wave)
    numerical = (dx*c*states).reshape(n+1,2,n//2).sum(dim=2)
    flux_error = flux-exact_flux
    return dict(temperature=float((states-exact).abs().max()),wall_temperature=wall_error,
                mean_flux=float(flux_error.abs().max()),wall_flux=float(flux_error[:,[0,-1]].abs().max()),
                cumulative_heat=float((flux_error*dt[:,None]).cumsum(0).abs().max()),
                layer_heat=float((numerical-stored).abs().max()),
                balance=float(residual.abs().max()))


if __name__=="__main__":
    rows = []
    for n in (16,32,64,128):
        errors = measure(n)
        orders = {k:math.log2(rows[-1]["errors"][k]/v) for k,v in errors.items() if k!="balance"} if rows else {}
        # The peak interval flux is at the material interface in the first
        # time step. Track it separately from wall flux and integrated heat.
        passed = errors["balance"]<1e-11 and (n<128 or all(
            order>(1.4 if k=="mean_flux" else 1.8) for k,order in orders.items()))
        rows.append(dict(columns=n,errors=errors,orders=orders,passed=passed))
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
