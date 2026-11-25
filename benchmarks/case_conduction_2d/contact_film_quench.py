"""Uniformly hot composite slab quenched by an east film, with an insulated west wall."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import brentq

from tensorsph.heat_conduction_2d import layered_slab_contact_temperatures, solve_transient_layered_slab


def reference(x, faces, time, h, resistance, count=32, *, interval_start=None):
    # Write the decay rate as p². Each layer satisfies k*f'' + p²*C*f = 0.
    k1,k2,c1,c2 = 1.,3.,2.,.8
    length = .5
    def coefficients(p):
        a,b = p*math.sqrt(c1/k1),p*math.sqrt(c2/k2)
        q = k1*a*math.sin(a*length)
        return a,b,math.cos(a*length)-resistance*q,-q/(k2*b)
    def boundary(p):
        a,b,A,B = coefficients(p)
        v = b*length
        return k2*b*(A*math.sin(v)-B*math.cos(v))-h*(A*math.cos(v)+B*math.sin(v))
    field = np.zeros_like(x)
    stored = np.zeros_like(faces)
    wall = 0.
    mean_traces = np.zeros(2)
    roots = []
    lo = 1e-9
    # Subdivide more finely than the oscillations in the two layer traces.
    for hi in np.arange(.025,10*count,.025):
        if boundary(lo)*boundary(hi)<0.:
            roots.append(brentq(boundary,lo,hi))
            if len(roots)==count:
                break
        lo = hi
    if len(roots)!=count:
        raise ValueError("insufficient roots for the quench series")
    for p in roots:
        a,b,A,B = coefficients(p)
        def integral(z):
            left = np.minimum(z,length)
            right = np.maximum(z-length,0.)
            return c1*np.sin(a*left)/a+c2*(A*np.sin(b*right)+B*(1-np.cos(b*right)))/b
        # Project the uniform initial temperature onto C-weighted eigenfunctions.
        norm = c1*(length/2+math.sin(2*a*length)/(4*a))
        norm += c2*((A*A+B*B)*length/2+(A*A-B*B)*math.sin(2*b*length)/(4*b)
                    +A*B*math.sin(b*length)**2/b)
        coefficient = integral(1.)/norm
        amplitude = coefficient*math.exp(-p*p*time)
        if interval_start is not None:
            duration = time-interval_start
            mean_signal = math.exp(-p*p*interval_start)*(-math.expm1(-p*p*duration))/(p*p*duration)
            mean_traces += coefficient*mean_signal*np.array([math.cos(a*length),A])
        def profile(z):
            y = z-length
            return np.where(z<length,np.cos(a*z),A*np.cos(b*y)+B*np.sin(b*y))
        field += amplitude*profile(x)
        stored += amplitude*integral(faces)
        wall += amplitude*profile(1.)
    initial = c1*np.minimum(faces,length)+c2*np.maximum(faces-length,0.)
    result = (field,wall,initial-stored)
    return result if interval_start is None else (*result,mean_traces)


def measure(n,h,resistance,switched=False):
    dx = 1/n
    faces = np.linspace(0.,1.,n+1)
    x = faces[:-1]+dx/2
    k,c = np.where(x<.5,1.,3.),np.where(x<.5,2.,.8)
    contacts = np.zeros(n-1)
    contacts[n//2-1] = resistance
    # Hold the time grid fixed for switched loads to isolate spatial errors
    # from startup transients after each change in the film reservoir.
    steps = 2048 if switched else 4*n
    times = np.linspace(0.,.4,steps+1)
    changes = ((.071,1.6),(.233,-1.4)) if switched else ()
    def ambient(t):
        return sum(delta for start,delta in changes if t>=start)
    states,heat,walls,mean = solve_transient_layered_slab(np.ones(n),times,dx,(None,None),conductivity=k,
            volumetric_heat_capacity=c,contact_resistance_x=contacts,x_wall_films=(None,(h,ambient)),
            load_jump_times=[start for start,_ in changes],
            rannacher=True,return_x_heat=True,return_wall_temperatures=True,return_interval_temperature=True)
    interval_flux = heat/heat.new_tensor(np.diff(times))[:,None]
    left,right = layered_slab_contact_temperatures(mean,interval_flux,dx,k)
    traces = np.column_stack((left[:,n//2-1],right[:,n//2-1]))
    cumulative = heat.cumsum(0).numpy()
    wall = walls[:,1].numpy()
    errors = dict(temperature=0.,wall=0.,section_heat=0.,left_trace_K=0.,right_trace_K=0.,contact_law_K=0.)
    for i in (steps//8,steps//4,steps//2,steps):
        exact,tw,q,target = reference(x,faces,times[i],h,resistance,interval_start=times[i-1])
        for start,delta in changes:
            if times[i]>start:
                elapsed = times[i]-start
                begin = max(0.,times[i-1]-start)
                f,w,removed,ts = reference(x,faces,elapsed,h,resistance,interval_start=begin)
                target += delta*(1-ts)*(elapsed-begin)/(times[i]-times[i-1])
                exact += delta*(1-f)
                tw += delta*(1-w)
                q -= delta*removed
        jump = traces[i-1,0]-traces[i-1,1]
        errors["contact_law_K"] = max(errors["contact_law_K"],abs(jump-resistance*float(interval_flux[i-1,n//2])))
        trace_error = abs(traces[i-1]-target)
        errors["left_trace_K"] = max(errors["left_trace_K"],float(trace_error[0]))
        errors["right_trace_K"] = max(errors["right_trace_K"],float(trace_error[1]))
        errors["temperature"] = max(errors["temperature"],float(abs(states[i].numpy()-exact).max()))
        errors["wall"] = max(errors["wall"],float(abs(wall[i]-tw)))
        errors["section_heat"] = max(errors["section_heat"],float(abs(cumulative[i-1]-q).max()))
    errors["balance"] = float(abs(dx*((states.numpy()[1:]-1)*c).sum(axis=1)+cumulative[:,-1]).max())
    return errors


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switched",action="store_true",help="apply two later ambient-temperature changes")
    args = parser.parse_args()
    rows = []
    for h,resistance in ((1.,0.),(8.,.24)):
        previous = {}
        for n in (16,32,64,128):
            errors = measure(n,h,resistance,args.switched)
            orders = {k:math.log2(previous[k]/v) for k,v in errors.items() if previous and k!="balance"}
            passed = errors["balance"]<1e-11 and (n<128 or min(orders.values())>1.8)
            rows.append(dict(columns=n,time_steps=2048 if args.switched else 4*n,film=h,contact=resistance,errors=errors,orders=orders,passed=passed))
            previous = errors
    version = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=version,switched=args.switched,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
