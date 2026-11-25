"""Temporal accuracy of symmetric storing films for a switched ambient drive."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import expm
from tensorsph.heat_conduction_2d import solve_transient_layered_slab


DX,K,C,CW,H = .5,1.4,2.,.45,3.
JUMP,STOP = .073,.19
INITIAL = np.array([.1,.8])  # particle and wall temperatures


def exact(time,harmonic=False):
    # By reflection symmetry midplane power vanishes. Each half contains
    # one particle-column and one wall; their mutual conductance is 2*k/dx.
    g = 2*K/DX
    matrix = np.array([[-g,g],[g,-g-H]])/np.array([C*DX,CW])[:,None]
    if harmonic:
        # For u'=A u + b cos(omega*t), the particular solution is
        # Re[(i*omega*I-A)^(-1) b exp(i*omega*t)].
        amplitude = np.linalg.solve(1j*OMEGA*np.eye(2)-matrix,[0.,H/CW])
        return (amplitude*np.exp(1j*OMEGA*time)).real+expm(matrix*time)@(INITIAL-amplitude.real)
    state = INITIAL
    for dt,air in ((min(time,JUMP),1.6),(max(time-JUMP,0.),-.5)):
        if dt:
            state = air+expm(matrix*dt)@(state-air)
    return state


OMEGA = 23.


def measure(steps,method,uniform,harmonic=False):
    # Locate the off-grid jump internally for all integrators.
    if uniform or harmonic:
        times = np.linspace(0.,STOP,steps+1)
    else:
        # Keep the first post-jump damping substep at the same fraction of
        # the local step on every refinement so its error coefficient is fixed.
        half = steps//2
        times = np.r_[np.arange(half)*JUMP/half,
                      JUMP+(np.arange(half)+.5)*(STOP-JUMP)/half,STOP]
    drive = (lambda t: np.cos(OMEGA*t)) if harmonic else ambient
    u,q,w,external = solve_transient_layered_slab(np.full(2,INITIAL[0]),times,DX,(None,None),
        conductivity=np.full(2,K),volumetric_heat_capacity=C,method="theta" if method=="rannacher" else method,
        rannacher=method=="rannacher",x_wall_films=((H,drive),(H,drive)),load_jump_times=[] if harmonic else [JUMP],
        wall_areal_heat_capacity=(CW,CW),initial_wall_temperatures=(INITIAL[1],INITIAL[1]),
        return_x_heat=True,return_wall_temperatures=True,return_external_heat=True)
    reference = np.stack([exact(t,harmonic) for t in times])
    qmat = C*DX*(reference[1:,0]-INITIAL[0])
    qext = qmat+CW*(reference[1:,1]-INITIAL[1])
    return dict(particle_error_K=float(abs(u.numpy()-reference[:,:1]).max()),
                wall_error_K=float(abs(w.numpy()-reference[:,1:]).max()),
                material_heat_error_J_m2=float(abs(q.numpy()[:,[0,-1]].cumsum(0)*[1.,-1.]-qmat[:,None]).max()),
                external_heat_error_J_m2=float(abs(external.numpy().cumsum(0)-qext[:,None]).max()),
                midplane_heat_J_m2=float(q[:,1].abs().max()))


def ambient(t):
    return 1.6 if t<JUMP else -.5


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-grid",action="store_true",help="let the jump move within its containing time step under refinement")
    parser.add_argument("--harmonic",action="store_true",help="use a smooth cosine film drive on uniform time steps")
    args = parser.parse_args()
    harmonic = args.harmonic
    uniform = args.uniform_grid or harmonic
    rows = []
    for method in ("theta","rannacher","sdirk2","exponential","exponential_quadratic"):
        previous = None
        for steps in (12,24,48,96):
            metrics = measure(steps,method,uniform,harmonic)
            errors = {k:v for k,v in metrics.items() if "_error_" in k}
            exact_load = method.startswith("exponential") and not harmonic
            orders = {} if previous is None or exact_load else {k:float(np.log2(previous[k]/v)) for k,v in errors.items()}
            passed = max(errors.values())<1e-12 if exact_load else all(v>(3.7 if method=="exponential_quadratic" else 1.7) for v in orders.values())
            rows.append(dict(method=method,steps=steps,**metrics,orders=orders,
                             passed=metrics["midplane_heat_J_m2"]<1e-12 and passed))
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,uniform_grid=uniform,harmonic=harmonic,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
