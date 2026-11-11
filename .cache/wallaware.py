import sys, math
sys.path.insert(0,"/root/tp-agent/work/m15332/src"); sys.path.insert(0,"/root/tp-agent/work/m15332/benchmarks/capillary_wave")
import torch
from tensorsph.csf_consistent import csf_fields
from tensorsph.droplet_impact import wall_lattice, wall_mass, cubic_spline_kernel, cell_linked_pairs, drop_wall_wall_pairs, apply_boundary, rhs
import validation as V
torch.set_grad_enabled(False)
lam=0.02; dx=0.001
for adx in (0.5, 2.0):
    pos,mass,cfg=V.make_case(lam,lam,dx,adx*dx,1000.0,7.2,0.0,10.0)
    pos,mass=pos.float(),mass.float()
    h=cfg.h_factor*dx; rho0=1000.0
    wall=wall_lattice(cfg, pos.device, top=lam+adx*dx).float()
    mw=wall_mass(cfg, wall.shape[0], pos.device).float()
    n=pos.shape[0]
    # fluid-only colour (current behaviour)
    pf=drop_wall_wall_pairs(cell_linked_pairs(pos,2.0*h), n)
    st_f=csf_fields(pos,mass,pf,h,rho0)
    # wall-aware colour
    pc=torch.cat((pos,wall),0); mc=torch.cat((mass,mw),0)
    pc_pairs=drop_wall_wall_pairs(cell_linked_pairs(pc,2.0*h), n)
    st_c=csf_fields(pc,mc,pc_pairs,h,rho0)
    def stats(st, label):
        k=st.kappa[:n]; ds=st.delta_s[:n]; col=st.color[:n]
        acc=(-7.2*k*ds/rho0)[:,None]*st.normal[:n]
        a=torch.linalg.vector_norm(acc,dim=1)
        sig=(2*math.pi/lam)**2*adx*dx
        print(f"  {label:10s} c_med={float(col.median()):.2f} c_min={float(col.min()):.2f} ds_max={float(ds.max()):.0f} kappa_rms={float((k**2).mean().sqrt()):.1f} a_max={float(a.max()):8.1f} a_med={float(a.median()):7.1f} a_phys={sig*rho0/rho0*0:0.0f}")
        x=pos[:,0]; W=lam/2
        wallz=(x<2.5*h)|(x>W-2.5*h)
        print(f"      wall-adj a_max={float(a[wallz].max()):8.1f} | interior a_max={float(a[~wallz].max()):8.1f}")
        return a
    print(f"A/dx={adx}: fluid-only vs wall-aware colour")
    a_f=stats(st_f,"fluid-only")
    a_c=stats(st_c,"wall-aware")
    print(f"   ratio a_max wall-aware/fluid-only = {float(a_c.max()/a_f.max()):.3f}")
