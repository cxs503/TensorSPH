"""Two heat capacities with contact, sinusoidal west wall and switched east wall and heaters."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import expm
import torch

from tensorsph.heat_conduction_2d import solve_transient_layered_slab


def measure(steps,theta):
    dx,resistance,omega = .35,.17,8.
    k = np.array([.7,2.3])
    c = np.array([1.2,4.1])
    loss = np.array([1.,.4])
    mass = dx*c
    wall_g = 2*k/dx
    g = 1/(resistance+sum(dx/(2*k)))
    jumps = (.073,.191)
    def west(t):
        return .7+.4*math.sin(omega*t)
    def east(t):
        return -.3 if t<jumps[0] else 1.1
    def source(t):
        return np.array([2.,-.5]) if t<jumps[1] else np.array([-.8,3.])
    initial = np.array([1.4,-.2])
    times = np.linspace(0.,.35,steps+1)
    field = solve_transient_layered_slab(initial,times,dx,(west,east),conductivity=k,
              volumetric_heat_capacity=c,contact_resistance_x=[resistance],heat_source=source,
              heat_loss_coefficient=loss,theta=theta,h_factor=.8,load_jump_times=jumps).numpy()
    # Advance the exact lumped energy equations on each constant-load segment.
    # Augment both temperatures by 1, sin(omega*t), cos(omega*t) to use exp(A dt).
    matrix = np.zeros((5,5))
    matrix[:2,:2] = (np.array([[-g,g],[g,-g]])-np.diag(wall_g+dx*loss))/mass[:,None]
    matrix[0,3] = .4*wall_g[0]/mass[0]
    matrix[3,4],matrix[4,3] = omega,-omega
    state = np.r_[initial,1.,0.,1.]
    segments = np.union1d(times,jumps)
    exact = [initial]
    for start,stop in zip(segments,segments[1:]):
        matrix[:2,2] = (dx*source(start)+wall_g*np.array([.7,east(start)]))/mass
        state = expm(matrix*(stop-start)) @ state
        if stop==times[len(exact)]:
            exact.append(state[:2])
    error = field-np.asarray(exact)
    return dict(temperature=float(abs(error).max()),contact_flux=float(abs(g*(error[:,0]-error[:,1])).max()))


if __name__=="__main__":
    rows = []
    for theta in (.5,1.):
        previous = {}
        for steps in (16,32,64,128):
            errors = measure(steps,theta)
            orders = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            passed = steps<128 or min(orders.values())>(1.85 if theta==.5 else .9)
            rows.append(dict(steps=steps,theta=theta,errors=errors,orders=orders,passed=passed))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
