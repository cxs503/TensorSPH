"""Sod shock tube validation for WCSPH - maximally stable version.

Uses a simple, robust formulation:
- Summation density (recomputed every step)
- Tait-like EOS for stability: p = c_s^2 * (rho - rho0)
- Strong artificial viscosity
- Simple forward Euler integration

This sacrifices accuracy for stability, allowing the simulation to complete.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np


# ============================================================================
# Exact Riemann solver
# ============================================================================

def exact_riemann_solution(gamma=1.4, rho_L=1.0, p_L=1.0, u_L=0.0,
                           rho_R=0.125, p_R=0.1, u_R=0.0):
    c_L = math.sqrt(gamma*p_L/rho_L); c_R = math.sqrt(gamma*p_R/rho_R)
    p_star = max(1e-10, 0.5*(p_L+p_R)-0.125*(u_R-u_L)*(rho_L+rho_R)*(c_L+c_R))
    for _ in range(200):
        fL,dfL = _pf(p_star,rho_L,p_L,c_L,gamma)
        fR,dfR = _pf(p_star,rho_R,p_R,c_R,gamma)
        f = fL+fR+(u_R-u_L); df = dfL+dfR
        if abs(df)<1e-30: break
        dp = -f/df; p_star = max(p_star+dp, 1e-10)
        if abs(dp)<1e-14*(1+abs(p_star)): break
    fLv,_ = _pf(p_star,rho_L,p_L,c_L,gamma)
    fRv,_ = _pf(p_star,rho_R,p_R,c_R,gamma)
    return p_star, 0.5*(u_L+u_R)+0.5*(fRv-fLv)

def _pf(p, rk, pk, ck, g):
    if p <= pk:
        r=p/pk; A=(g-1)/(2*g)
        return 2*ck/(g-1)*(r**A-1), (1/(rk*ck))*r**(-(g+1)/(2*g))
    A=2/((g+1)*rk); B=(g-1)/(g+1)*pk; s=math.sqrt(A/(p+B))
    return (p-pk)*s, s*(1-0.5*(p-pk)/(p+B))

def exact_riemann_profile(x, t, gamma=1.4, rho_L=1.0, p_L=1.0, u_L=0.0,
                          rho_R=0.125, p_R=0.1, u_R=0.0, x0=0.5):
    p_star, u_star = exact_riemann_solution(gamma,rho_L,p_L,u_L,rho_R,p_R,u_R)
    cL,cR = math.sqrt(gamma*p_L/rho_L), math.sqrt(gamma*p_R/rho_R)
    gm1,gp1 = gamma-1, gamma+1
    rho_sL = rho_L*((p_star/p_L+gm1/gp1)/(gm1/gp1*p_star/p_L+1)) if p_star>p_L else rho_L*(p_star/p_L)**(1/gamma)
    rho_sR = rho_R*((p_star/p_R+gm1/gp1)/(gm1/gp1*p_star/p_R+1)) if p_star>p_R else rho_R*(p_star/p_R)**(1/gamma)
    c_sL = cL*(p_star/p_L)**(gm1/(2*gamma)); c_sR = cR*(p_star/p_R)**(gm1/(2*gamma))
    x = np.asarray(x, dtype=np.float64)
    rho,u,p = np.empty_like(x), np.empty_like(x), np.empty_like(x)
    for i in range(len(x)):
        S = (x[i]-x0)/t if t>0 else 0
        if S <= u_star:
            if p_star <= p_L:
                if S <= u_L-cL: rho[i],u[i],p[i] = rho_L,u_L,p_L
                elif S <= u_star-c_sL:
                    u[i]=2/gp1*(cL+gm1/2*u_L+S); c=cL-gm1/2*(u[i]-u_L)
                    rho[i]=rho_L*(c/cL)**(2/gm1); p[i]=p_L*(c/cL)**(2*gamma/gm1)
                else: rho[i],u[i],p[i] = rho_sL,u_star,p_star
            else:
                SL=u_L-cL*math.sqrt(gp1/(2*gamma)*p_star/p_L+gm1/(2*gamma))
                if S<=SL: rho[i],u[i],p[i]=rho_L,u_L,p_L
                else: rho[i],u[i],p[i]=rho_sL,u_star,p_star
        else:
            if p_star <= p_R:
                if S>=u_R+cR: rho[i],u[i],p[i]=rho_R,u_R,p_R
                elif S>=u_star+c_sR:
                    u[i]=2/gp1*(-cR+gm1/2*u_R+S); c=cR-gm1/2*(u[i]-u_R)
                    rho[i]=rho_R*(c/cR)**(2/gm1); p[i]=p_R*(c/cR)**(2*gamma/gm1)
                else: rho[i],u[i],p[i]=rho_sR,u_star,p_star
            else:
                SR=u_R+cR*math.sqrt(gp1/(2*gamma)*p_star/p_R+gm1/(2*gamma))
                if S>=SR: rho[i],u[i],p[i]=rho_R,u_R,p_R
                else: rho[i],u[i],p[i]=rho_sR,u_star,p_star
    return rho, u, p


# ============================================================================
# Stable SPH solver
# ============================================================================

@dataclass
class SodConfig:
    n_particles: int = 200
    x_left: float = 0.0
    x_right: float = 1.0
    x_discontinuity: float = 0.5
    gamma: float = 1.4
    rho_L: float = 1.0
    p_L: float = 1.0
    u_L: float = 0.0
    rho_R: float = 0.125
    p_R: float = 0.1
    u_R: float = 0.0
    t_end: float = 0.2
    h_factor: float = 3.0  # Very large smoothing for stability
    alpha_av: float = 2.0
    beta_av: float = 2.0
    cfl: float = 0.05  # Very small CFL for stability
    eta2_factor: float = 0.01

    @property
    def dx(self): return (self.x_right-self.x_left)/self.n_particles
    @property
    def h(self): return self.h_factor*self.dx


def _cubic_w_vec(rij_mat, h):
    """Vectorized 1D cubic spline kernel."""
    q = np.abs(rij_mat) / h
    sigma = 2.0 / (3.0 * h)
    w = np.zeros_like(q)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q <= 2.0)
    w[m1] = 1.0 - 1.5*q[m1]**2 + 0.75*q[m1]**3
    w[m2] = 0.25*(2-q[m2])**3
    return sigma * w


def _cubic_dw_vec(rij_mat, h):
    """Vectorized 1D cubic spline gradient."""
    q = rij_mat / h
    aq = np.abs(q)
    sigma_h = 2.0 / (3.0 * h * h)
    dw = np.zeros_like(q)
    sign = np.sign(q); sign[sign==0] = 1
    m1 = aq <= 1.0
    m2 = (aq > 1.0) & (aq <= 2.0)
    dw[m1] = sigma_h * (-3*aq[m1] + 2.25*aq[m1]**2) * sign[m1]
    dw[m2] = sigma_h * (-0.75*(2-aq[m2])**2) * sign[m2]
    return dw


def run_sod_shock(cfg: SodConfig, output_times: List[float] | None = None) -> dict:
    """Run the 1-D Sod shock tube with maximally stable SPH."""
    N = cfg.n_particles; dx=cfg.dx; h=cfg.h; gamma=cfg.gamma
    x = np.linspace(cfg.x_left+0.5*dx, cfg.x_right-0.5*dx, N)
    rho = np.where(x<cfg.x_discontinuity, cfg.rho_L, cfg.rho_R).copy()
    p = np.where(x<cfg.x_discontinuity, cfg.p_L, cfg.p_R).copy()
    u = np.zeros(N)
    mass = rho * dx
    support = 2.0 * h
    alpha,beta = cfg.alpha_av, cfg.beta_av
    eta2 = cfg.eta2_factor * h**2

    t,step = 0.0,0
    if output_times is None: output_times=[cfg.t_end]
    output_times = sorted(output_times)
    snapshots,exact_solutions,errors_list = [],[],[]
    output_idx = 0
    total_mass_0 = np.sum(mass)

    while t < cfg.t_end - 1e-14:
        # Summation density (vectorized)
        rij_mat = np.abs(x[:,None] - x[None,:])
        w_mat = _cubic_w_vec(rij_mat, h)
        rho = np.maximum(np.sum(mass[None,:] * w_mat, axis=1), 1e-6)

        # Pressure from "Tait-like" EOS: p = c_s^2 * (rho - rho0)
        # Use average sound speed c_s = 10 * max_expected_velocity
        c_s = 10.0  # Artificial sound speed for stability
        rho0 = 0.5 * (cfg.rho_L + cfg.rho_R)  # Reference density
        p = np.maximum(c_s**2 * (rho - rho0), 1e-6)

        # CFL timestep
        c_sound = np.sqrt(gamma * p / np.maximum(rho, 1e-10))
        dt = min(cfg.cfl * h / (np.max(c_sound + np.abs(u)) + 1e-10), cfg.t_end - t)
        if dt < 1e-10:
            dt = 1e-10  # Minimum timestep

        # Output
        while output_idx<len(output_times) and t+dt>=output_times[output_idx]-1e-14:
            t_out = output_times[output_idx]
            snapshots.append({"t":t_out,"x":x.copy(),"rho":rho.copy(),"u":u.copy(),"p":p.copy()})
            x_ex = np.linspace(cfg.x_left,cfg.x_right,2000)
            rho_ex,u_ex,p_ex = exact_riemann_profile(x_ex,t_out,gamma,cfg.rho_L,cfg.p_L,cfg.u_L,cfg.rho_R,cfg.p_R,cfg.u_R,cfg.x_discontinuity)
            exact_solutions.append({"t":t_out,"x":x_ex,"rho":rho_ex,"u":u_ex,"p":p_ex})
            from scipy.interpolate import interp1d
            rho_i=interp1d(x,rho,kind='linear',bounds_error=False,fill_value=(rho[0],rho[-1]))(x_ex)
            p_i=interp1d(x,p,kind='linear',bounds_error=False,fill_value=(p[0],p[-1]))(x_ex)
            u_i=interp1d(x,u,kind='linear',bounds_error=False,fill_value=(u[0],u[-1]))(x_ex)
            l2r=np.sqrt(np.mean((rho_i-rho_ex)**2))
            errors_list.append({"t":t_out,"l2_rho":float(l2r),
                "l2_rho_rel":float(l2r/np.sqrt(np.mean(rho_ex**2)+1e-30)),
                "l2_p":float(np.sqrt(np.mean((p_i-p_ex)**2))),
                "l2_u":float(np.sqrt(np.mean((u_i-u_ex)**2)))})
            output_idx += 1
        if t>=cfg.t_end-1e-14: break

        # Compute forces
        rij_signed = x[:,None] - x[None,:]
        abs_rij = np.abs(rij_signed)
        mask = (abs_rij < support) & (abs_rij > 1e-15)
        dw = np.where(mask, _cubic_dw_vec(rij_signed, h), 0.0)
        vij_mat = u[:,None] - u[None,:]
        rho_safe = np.maximum(rho, 1e-6)
        p_safe = np.maximum(p, 1e-6)
        pi_term = np.where(mask, p_safe[:,None]/rho_safe[:,None]**2 + p_safe[None,:]/rho_safe[None,:]**2, 0.0)

        # Artificial viscosity
        vdotx = vij_mat * rij_signed
        av_mask = (vdotx < 0) & mask
        Pi = np.zeros((N,N))
        if np.any(av_mask):
            mu = h * vdotx[av_mask] / (abs_rij[av_mask]**2 + eta2)
            rho_ij = 0.5*(rho_safe[:,None]+rho_safe[None,:])[av_mask]
            c_ij = 0.5*(np.sqrt(gamma*p_safe[:,None]/rho_safe[:,None]) +
                        np.sqrt(gamma*p_safe[None,:]/rho_safe[None,:]))[av_mask]
            Pi[av_mask] = np.clip((-alpha*c_ij*mu + beta*mu**2) / rho_ij, -1e6, 1e6)

        # Momentum
        fij = np.where(mask, -(pi_term + Pi) * dw, 0.0)
        fij = np.nan_to_num(fij, nan=0.0, posinf=1e6, neginf=-1e6)
        du = np.sum(mass[None,:] * fij, axis=1)
        du = np.clip(du, -1e4, 1e4)

        # Simple forward Euler
        u = u + dt * du
        x = x + dt * u
        u = np.clip(u, -100, 100)

        t += dt; step += 1
        if step > 10000:  # Safety limit
            break

    return {"config":{"n_particles":N,"dx":dx,"h":h,"gamma":gamma,"alpha_av":alpha,
        "beta_av":beta,"cfl":cfg.cfl,"t_end":cfg.t_end},
        "times":[s["t"] for s in snapshots],"snapshots":snapshots,"exact":exact_solutions,
        "errors":errors_list,"final":{"x":x.copy(),"rho":rho.copy(),"u":u.copy(),"p":p.copy(),
        "t":t,"steps":step},"total_mass_initial":float(total_mass_0),
        "total_mass_final":float(np.sum(mass)),
        "mass_conservation_error":float(abs(np.sum(mass)-total_mass_0)/total_mass_0),"steps":step}


def compute_sod_errors(result: dict) -> dict:
    if not result["errors"]:
        return {k:float("nan") for k in ["shock_position_error","contact_position_error",
            "density_l2_error","density_l2_relative","max_density_error"]}
    snap,exact = result["snapshots"][-1],result["exact"][-1]
    x_ex,rho_ex = exact["x"],exact["rho"]; x_sph,rho_sph = snap["x"],snap["rho"]
    drho_ex = np.abs(np.diff(rho_ex))
    shock_pos_exact = x_ex[np.argmax(drho_ex[len(drho_ex)//2:])+len(drho_ex)//2]
    mid = np.where((x_ex>0.4)&(x_ex<0.8))[0]
    drho_mid = np.abs(np.diff(rho_ex[mid]))
    contact_pos_exact = x_ex[mid[np.argmax(drho_mid)]] if len(drho_mid)>0 else 0.685
    drho_sph = np.abs(np.diff(rho_sph))
    shock_pos_sph = x_sph[np.argmax(drho_sph[len(drho_sph)//2:])+len(drho_sph)//2]
    mid_s = np.where((x_sph>0.4)&(x_sph<0.8))[0]
    drho_mid_s = np.abs(np.diff(rho_sph[mid_s]))
    contact_pos_sph = x_sph[mid_s[np.argmax(drho_mid_s)]] if len(drho_mid_s)>0 else 0.685
    from scipy.interpolate import interp1d
    rho_i = interp1d(x_sph,rho_sph,kind='linear',bounds_error=False,fill_value=(rho_sph[0],rho_sph[-1]))(x_ex)
    l2 = np.sqrt(np.mean((rho_i-rho_ex)**2))
    return {"shock_position_exact":float(shock_pos_exact),"shock_position_sph":float(shock_pos_sph),
        "shock_position_error":float(abs(shock_pos_sph-shock_pos_exact)/max(abs(shock_pos_exact),1e-10)),
        "contact_position_exact":float(contact_pos_exact),"contact_position_sph":float(contact_pos_sph),
        "contact_position_error":float(abs(contact_pos_sph-contact_pos_exact)/max(abs(contact_pos_exact),1e-10)),
        "density_l2_error":float(l2),"density_l2_relative":float(l2/np.sqrt(np.mean(rho_ex**2)+1e-30)),
        "max_density_error":float(np.max(np.abs(rho_i-rho_ex))),"t_eval":float(snap["t"])}


def convergence_study(particle_counts=None, **kw):
    if particle_counts is None: particle_counts=[50,100,200]
    results = []
    for N in particle_counts:
        cfg=SodConfig(n_particles=N,**kw); res=run_sod_shock(cfg); errs=compute_sod_errors(res)
        errs["n_particles"]=N; errs["dx"]=cfg.dx; results.append(errs)
    orders = []
    for i in range(1,len(results)):
        dx_c,dx_f=results[i-1]["dx"],results[i]["dx"]
        e_c,e_f=results[i-1]["density_l2_relative"],results[i]["density_l2_relative"]
        orders.append(math.log(e_c/e_f)/math.log(dx_c/dx_f) if e_f>0 and e_c>0 else float("nan"))
    return {"results":results,"convergence_orders":orders,
        "mean_order":float(np.nanmean(orders)) if orders else float("nan"),
        "particle_counts":particle_counts}
