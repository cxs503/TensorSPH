"""Independent Fourier solutions for two flux walls, in steady and harmonic conduction."""
import cmath
import json
import math
from pathlib import Path
import subprocess

import torch

from tensorsph.heat_conduction_2d import solve_harmonic_slab_2d, solve_steady_slab_2d


def measure(n, omega):
    k,c = 1.7,2.3
    x = (torch.arange(n,dtype=torch.float64)+.5)/n
    faces = torch.arange(n+1,dtype=x.dtype)/n
    mode_x = cmath.sqrt(1j*omega*c/k) if omega else 0.
    if omega:
        uniform = torch.exp(-mode_x*x)
        section_flux = k*mode_x*torch.exp(-mode_x*faces)
    else:
        uniform = 2.-.7*x
        section_flux = torch.full_like(faces,.7*k)
    exact = uniform.expand(n,-1).to(torch.complex128).clone()
    west = section_flux[0].expand(n).to(exact.dtype).clone()
    east = -section_flux[-1].expand(n).to(exact.dtype).clone()
    # Each wall drives a different transverse wavelength and axial decay.
    for m,amplitude,from_east in ((1,.6+.2j,False),(-2,.2-.3j,True)):
        beta = 2*math.pi*m
        wave = cmath.sqrt(beta**2+mode_x**2)
        profile = amplitude*torch.exp(1j*beta*x)
        distance = 1-x if from_east else x
        exact += profile[:,None]*torch.exp(-wave*distance)
        at_wall = k*wave*profile
        far_wall = -at_wall*cmath.exp(-wave)
        west += far_wall if from_east else at_wall
        east += at_wall if from_east else far_wall
    source = torch.zeros((n,n),dtype=x.dtype)
    opts = dict(conductivity=k,return_x_flux=True)
    if omega:
        field,flux = solve_harmonic_slab_2d(source,1/n,(None,None),omega,volumetric_heat_capacity=c,
                                            x_wall_inward_flux=(west,east),**opts)
    else:
        exact = exact.real
        field,flux = solve_steady_slab_2d(source,1/n,(None,None),mean_temperature=float(uniform.mean()),
                                          x_wall_inward_flux=(west.real,east.real),**opts)
    balance = flux.diff()+1j*omega*c*field.mean(dim=0)/n
    return dict(n=n,omega=omega,temperature_error_K=float((exact-field).abs().max()),
                section_flux_error_W_m2=float((flux-section_flux).abs().max()),
                column_residual_W_m2=float(balance.abs().max()))


if __name__ == "__main__":
    rows = []
    for omega in (0.,5.):
        previous = {}
        for n in (16,32,64,128):
            row = measure(n,omega)
            errors = {k:v for k,v in row.items() if "_error_" in k and (omega or not k.startswith("section"))}
            rates = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            row.update(orders=rates,passed=all(v>1.8 for v in rates.values()) and row["column_residual_W_m2"]<1e-10
                       and (omega!=0. or row["section_flux_error_W_m2"]<1e-10))
            rows.append(row)
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
