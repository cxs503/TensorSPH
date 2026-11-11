import sys, math, time
sys.path.insert(0,"/root/tp-agent/work/m15332/src"); sys.path.insert(0,"/root/tp-agent/work/m15332/benchmarks/capillary_wave")
import numpy as np, torch
from tensorsph.droplet_impact import wall_lattice, wall_mass, apply_boundary, rhs, drop_wall_wall_pairs
from tensorsph.neighbor_search import PairListCache
import validation as V
torch.set_grad_enabled(False)
lam=0.02; dx=0.001
def run(adx, periods=1.0, boundary="wall"):
    pos,mass,cfg=V.make_case(lam,lam,dx,adx*dx,1000.0,7.2,0.0,10.0)
    pos,mass=pos.float(),mass.float()
    cfg.boundary=boundary
    cfg.dt=cfg.max_courant*cfg.h_factor*cfg.dx/cfg.sound_speed
    dt=float(cfg.dt)
    h=cfg.h_factor*dx; n=pos.shape[0]
    top=lam+adx*dx+4*dx
    wall=None; nw=0
    if boundary=="wall":
        wall=wall_lattice(cfg,pos.device,top=top).float(); nw=wall.shape[0]
        allp=torch.cat((pos,wall),0)
    vel=torch.zeros_like(pos)
    k=2*math.pi/lam; omega=V.angular_frequency(7.2,k,1000.0,lam); amp=adx*dx
    u_ref=amp*omega
    steps=int(math.ceil(periods*2*math.pi/omega/dt))
    every=max(1,steps//60)
    n_bins=max(8,int(round((lam/2)/(2*dx))))
    cache=PairListCache(2.0*h, cfg.pair_skin_factor*h, fluid_count=n)
    times=[];amps=[];peak=0.0;esc=None;rmin=1.0
    t0=time.perf_counter()
    for step in range(steps+1):
        if wall is None:
            pairs,_=cache.get(pos)
            rho,_,acc=rhs(pos,vel,mass,cfg,pairs=pairs,wall=None,dt=dt)
        else:
            allp=torch.cat((pos,wall),0)
            pairs,_=cache.get(allp)
            rho,_,acc=rhs(pos,vel,mass,cfg,pairs=pairs,wall=wall,dt=dt)
        rmin=min(rmin,float((rho/cfg.rho0).min()))
        if step%every==0 or step==steps:
            times.append(step*dt)
            amps.append(V.column_elevation(pos,mass,lam/2,lam,1000.0,k,n_bins))
        vel=vel+dt*acc; pos=pos+dt*vel
        peak=max(peak,float(torch.linalg.vector_norm(vel,dim=1).max()))
        if apply_boundary(pos,vel,cfg,top=top):
            esc=step; break
    row=dict(adx=adx,boundary=boundary,status="aborted" if esc is not None else "completed",
             esc=esc,steps=steps,peak_v=peak,v_over_uref=peak/u_ref,rmin=rmin,n=n,nw=nw,
             t=round(time.perf_counter()-t0,1))
    if esc is None:
        fit=V.fit_damped_oscillation(np.asarray(times),np.asarray(amps),omega)
        row.update(omega_err=100*(fit["omega"]-omega)/omega,resid=fit["residual_rel"])
    print(row, flush=True)
    return row
run(0.5,1.0,"wall")
run(0.5,1.0,"clamp")
run(2.0,1.0,"wall")
