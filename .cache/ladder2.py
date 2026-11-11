import sys, math, time, json
sys.path.insert(0,"/root/tp-agent/work/m15332/src"); sys.path.insert(0,"/root/tp-agent/work/m15332/benchmarks/capillary_wave")
import numpy as np, torch
from tensorsph.droplet_impact import wall_lattice, apply_boundary, rhs
from tensorsph.neighbor_search import PairListCache
import validation as V
torch.set_grad_enabled(False)
lam=0.02
def run(ppw, adx, periods=2.0, over=None):
    dx=lam/ppw; over=dict(over or {})
    pos,mass,cfg=V.make_case(lam,lam,dx,adx*dx,1000.0,7.2,0.0,10.0)
    for kk,vv in over.items(): setattr(cfg,kk,vv)
    cfg.boundary="wall"
    cfg.dt=cfg.max_courant*cfg.h_factor*cfg.dx/cfg.sound_speed
    pos,mass=pos.float(),mass.float(); dt=float(cfg.dt)
    h=cfg.h_factor*dx; n=pos.shape[0]; top=lam+adx*dx+4*dx
    wall=wall_lattice(cfg,pos.device,top=top).float(); nw=wall.shape[0]
    vel=torch.zeros_like(pos)
    k=2*math.pi/lam; omega=V.angular_frequency(7.2,k,1000.0,lam); amp=adx*dx
    u_ref=amp*omega
    steps=int(math.ceil(periods*2*math.pi/omega/dt)); every=max(1,steps//120)
    n_bins=max(8,int(round((lam/2)/(2*dx))))
    cache=PairListCache(2.0*h, cfg.pair_skin_factor*h, fluid_count=n)
    times=[];amps=[];peak=0.0;esc=None;rmin=1.0
    t0=time.perf_counter()
    for step in range(steps+1):
        allp=torch.cat((pos,wall),0)
        pairs,_=cache.get(allp)
        rho,_,acc=rhs(pos,vel,mass,cfg,pairs=pairs,wall=wall,dt=dt)
        rmin=min(rmin,float((rho/cfg.rho0).min()))
        if step%every==0 or step==steps:
            times.append(step*dt); amps.append(V.column_elevation(pos,mass,lam/2,lam,1000.0,k,n_bins))
        vel=vel+dt*acc; pos=pos+dt*vel
        peak=max(peak,float(torch.linalg.vector_norm(vel,dim=1).max()))
        if apply_boundary(pos,vel,cfg,top=top): esc=step; break
    row=dict(ppw=ppw,adx=adx,over=over,status="aborted" if esc is not None else "completed",
             esc=esc,steps=steps,v_over_uref=round(peak/u_ref,2),rmin=round(rmin,3),n=n,nw=nw,
             steep=round(2*math.pi*adx/ppw,3),t=round(time.perf_counter()-t0,1))
    if esc is None:
        fit=V.fit_damped_oscillation(np.asarray(times),np.asarray(amps),omega)
        row.update(omega_err=round(100*(fit["omega"]-omega)/omega,2),resid=round(fit["residual_rel"],3),
                   amp0_mm=round(float(amps[0])*1e3,3), gamma=round(fit["gamma"],1), gamma_th=round(V.damping_rate(0.0,k),1))
    print(json.dumps(row), flush=True)
    return row
out=[]
out.append(run(20,2.0))
out.append(run(20,2.0,over={"density_renormalization":True}))
out.append(run(20,4.0))
out.append(run(20,4.0,over={"density_renormalization":True}))
out.append(run(40,2.0))
out.append(run(40,4.0))
json.dump(out, open(".cache/ladder2.json","w"), indent=1)
