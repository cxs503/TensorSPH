"""Plane Poiseuille flow: analytic verification of the viscous operator and the
no-slip wall boundary condition.

Why this benchmark exists
-------------------------
Every other number TensorSPH publishes is pressure- or gravity-driven.  The
viscous term has never been checked against a flow with an exact solution: it
was *calibrated* (``VISCOUS_LAPLACIAN_FACTOR`` was fitted so that a sinusoidal
shear wave decays at the analytic rate on a smooth field, see
``tests/test_pair_based_rhs.py``) but never *verified* in a wall-bounded flow,
where the same operator has to (a) balance a body force, (b) resolve a
parabolic profile down to the wall, and (c) work together with the static
dummy-particle wall of Adami, Hu & Adams (2012).  This harness closes that gap
with the textbook solution.

Exact solution
--------------
A channel of width ``H`` between two no-slip walls at ``x = +-a`` (``a = H/2``)
filled with an incompressible fluid of kinematic viscosity ``nu`` and driven by
a uniform body acceleration ``g`` along ``y`` has the steady state

    u(x) = g / (2 nu) * (a^2 - x^2),      u_max = g a^2 / (2 nu),      (1)

with the mean velocity ``ubar = (2/3) u_max`` and the flow rate per unit depth
``Q = (2/3) u_max H``.  The pressure field is uniform: the body force is
balanced by the shear stress, not by a pressure gradient, so the SPH pressure
stays at its gauge value everywhere.  That is a useful property here -- the
measurement isolates the viscous term and the wall condition instead of mixing
them with the Tait-EOS/density noise that limits the free-surface benchmarks.

How the flow is driven
----------------------
``Config.gravity`` is a scalar applied along ``-y``, so the body force is
aligned with the flow by putting the *flow* along ``y`` and the *walls* at
``x = +-a``.  No source change is needed: the same gravity term that drives the
dam break drives this channel.

How the channel is made periodic
--------------------------------
``rhs`` builds its pair list from raw positions, so a solver-level periodic
image is not available.  This harness supplies the images itself: particles
within ``2h`` of the streamwise seam are duplicated at ``y +- L_y`` with the
*parent* velocity and mass, the pair list is built over fluid + images + wall,
and only the real fluid block is integrated.  A particle therefore sees the
kernel support it would have in an infinite channel, which is what makes the
density summation uniform in a periodic domain (tested in
``tests/test_poiseuille.py``).

References
    Morris, Fox & Zhu (1997), J. Comput. Phys. 136:214 (periodic SPH channel).
    Adami, Hu & Adams (2012), J. Comput. Phys. 231:7057 (static wall BC).

Usage::

    PYTHONPATH=src python benchmarks/poiseuille/run.py --dx 0.004,0.0025,0.002 \\
        --t-end 0.12 --out benchmarks/poiseuille
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    drop_wall_wall_pairs,
    rhs,
    stable_dt,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# analytic reference
# --------------------------------------------------------------------------- #
def analytic_profile(x, half_width: float, nu: float, body: float):
    """Exact steady profile (1); returns a list of floats for JSON safety."""
    a = half_width
    return [body / (2.0 * nu) * (a * a - float(xi) * float(xi)) for xi in x]


def analytic_scalars(half_width: float, nu: float, body: float) -> dict:
    a = half_width
    u_max = body * a * a / (2.0 * nu)
    return {
        "u_max_m_per_s": u_max,
        "u_mean_m_per_s": (2.0 / 3.0) * u_max,
        "flow_rate_m2_per_s": (2.0 / 3.0) * u_max * (2.0 * a),
        "wall_shear_Pa": None,  # filled by the caller (needs rho0)
        "reynolds_channel": u_max * (2.0 * a) / nu,
        "viscous_time_s": (2.0 * a) ** 2 / nu,
    }


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def build_channel(dx: float, width: float, length: float, rho0: float,
                  wall_layers: int = 3, h: float = 0.0, y_lo: float = 0.0):
    """Square-lattice fluid channel plus static vertical wall layers.

    The fluid occupies ``x in (-H/2, H/2)`` with cell centres half a spacing
    inside the nominal wall face, and ``y in [0, L_y)`` (periodic).  The wall
    particles sit half a spacing *outside* the face and then one spacing apart,
    which is the convention of ``tensorsph.droplet_impact.wall_lattice``, so
    the effective no-slip plane can be compared with the one used elsewhere in
    the repository.
    """
    a = 0.5 * width
    nx = int(round(width / dx))
    ny = int(round(length / dx))
    if abs(nx * dx - width) > 1e-12 or abs(ny * dx - length) > 1e-12:
        raise ValueError("width and length must be integer multiples of dx")
    x = -a + dx * (torch.arange(nx, dtype=torch.float64) + 0.5)
    y = y_lo + dx * (torch.arange(ny, dtype=torch.float64) + 0.5)
    gx, gy = torch.meshgrid(x, y, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), rho0 * dx * dx, dtype=torch.float64)

    # wall: `wall_layers` columns on each side, streamwise period extended by
    # the support radius so the seam sees the same wall as the bulk.
    margin = 2.0 * h if h > 0.0 else 2.0 * dx
    n_ext = int(math.ceil(margin / dx))
    ys = y_lo - margin + dx * (torch.arange(ny + 2 * n_ext, dtype=torch.float64) + 0.5)
    wall_parts = []
    for side in (-1.0, 1.0):
        for layer in range(1, wall_layers + 1):
            xw = side * (a + dx * (layer - 0.5))
            gxw, gyw = torch.meshgrid(
                torch.tensor([xw], dtype=torch.float64), ys, indexing="xy")
            wall_parts.append(torch.stack((gxw.reshape(-1), gyw.reshape(-1)), dim=-1))
    wall = torch.cat(wall_parts, dim=0) if wall_parts else None
    return pos, mass, wall


def periodic_images(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                    length: float, support: float, y_lo: float = 0.0):
    """Index map for the streamwise images used by the pair list.

    Returns ``(idx, shift)``: image ``k`` is particle ``idx[k]`` translated by
    ``shift[k] * length`` along ``y``.  Particles near the lower seam are
    copied *up* and particles near the upper seam are copied *down*.

    The copied strip is ``2 * support`` deep, not ``support`` deep.  A ghost
    that is close enough to interact with a real particle sits at most
    ``support`` from the seam, but that ghost still needs its *own* kernel
    support, which reaches ``2 * support`` into the strip.  A one-support-deep
    strip therefore loses the far half of the images' neighbourhoods: measured
    ghost densities drop to 0.3-0.7 rho0, and because the regularised viscous
    weight scales as ``m / rho`` those wrong states act as a 1.5x viscosity in
    the seam region.  ``length > 4 * support`` keeps the two strips disjoint.
    """
    depth = 2.0 * support
    if length <= 2.0 * depth:
        raise ValueError(
            "the periodic length must exceed the two image strips "
            f"(2*depth = {2.0*depth:.4g} m); got {length:.4g} m")
    y = pos[:, 1]
    lo = y < y_lo + depth
    hi = y > y_lo + length - depth
    idx = torch.cat((torch.nonzero(lo).reshape(-1), torch.nonzero(hi).reshape(-1)))
    shift = torch.cat((
        torch.full((int(lo.sum()),), 1.0, dtype=pos.dtype),
        torch.full((int(hi.sum()),), -1.0, dtype=pos.dtype),
    ))
    return idx, shift


def augment(pos, vel, mass, length, support, y_lo=0.0):
    """Fluid block followed by its periodic images (velocities copied)."""
    idx, shift = periodic_images(pos, vel, mass, length, support, y_lo)
    if idx.numel() == 0:
        return pos, vel, mass, pos.shape[0]
    off = torch.zeros((idx.numel(), pos.shape[1]), dtype=pos.dtype)
    off[:, 1] = shift * length
    g_pos = pos[idx] + off
    return (torch.cat((pos, g_pos), dim=0),
            torch.cat((vel, vel[idx]), dim=0),
            torch.cat((mass, mass[idx]), dim=0),
            pos.shape[0])


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def make_config(dx, nu, body, rho0, sound_speed, dt=None) -> Config:
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=body, viscosity=nu, wall_layers=3,
                 wall_friction=1.0, surface_tension_coefficient=0.0,
                 pair_skin_factor=0.0)
    cfg.dt = float(stable_dt(cfg)) if dt is None else float(dt)
    return cfg


def run_case(dx: float, width: float = 0.02, length: float | None = None,
             nu: float = 1.0e-2, body: float = 9.81, rho0: float = 1000.0,
             sound_speed: float = 2.0, t_end: float = 0.08,
             sample_every: int = 100, steps_limit: int = 200000,
             stop_when_steady: bool = True):
    """Integrate the channel to ``t_end`` and score the steady profile."""
    t0 = time.time()
    cfg = make_config(dx, nu, body, rho0, sound_speed)
    h = cfg.h_factor * cfg.dx
    if length is None:
        # 16 spacings keep the two 4h-deep image strips (2 * 5.2 dx) disjoint.
        length = 16.0 * dx
    a = 0.5 * width
    pos, mass, wall = build_channel(dx, width, length, rho0,
                                    wall_layers=cfg.wall_layers, h=h)
    n_fluid = pos.shape[0]
    vel = torch.zeros_like(pos)
    dt = cfg.dt

    ref = analytic_scalars(a, nu, body)
    u_max = ref["u_max_m_per_s"]

    hist = {"t": [], "u_mean": [], "u_max": [], "rho_min": [], "rho_max": [],
            "mass": [], "p_max": [], "vmax": [], "xmax_over_a": []}
    steps = 0
    t = 0.0
    converged = False
    while steps < steps_limit and t < t_end:
        all_pos, all_vel, all_mass, nf = augment(pos, vel, mass, length, 2.0 * h)
        if wall is None:
            pairs = cell_linked_pairs(all_pos, 2.0 * h)
        else:
            pairs = cell_linked_pairs(torch.cat((all_pos, wall), dim=0), 2.0 * h)
            # the images are fluid particles for the purpose of the pair
            # filter: only wall-wall pairs may be dropped
            pairs = drop_wall_wall_pairs(pairs, all_pos.shape[0])
        rho, pressure, acc = rhs(all_pos, all_vel, all_mass, cfg, pairs=pairs,
                                 wall=wall, dt=dt)
        acc = acc[:nf]
        vel = vel + dt * acc
        pos = pos + dt * vel
        t += dt
        steps += 1

        if steps % sample_every == 0 or t >= t_end:
            u = -vel[:, 1]
            hist["t"].append(t)
            hist["u_mean"].append(float(u.mean()))
            hist["u_max"].append(float(u.max()))
            # only the real block matters: the image strip at its far end is
            # deliberately incomplete (it is never a neighbour of a real
            # particle), so scoring its density would be measuring the halo.
            hist["rho_min"].append(float(rho[:nf].min() / rho0))
            hist["rho_max"].append(float(rho[:nf].max() / rho0))
            hist["xmax_over_a"].append(float(pos[:, 0].abs().max()) / a)
            hist["mass"].append(float(mass.sum()))
            hist["p_max"].append(float(pressure.max()))
            hist["vmax"].append(float(torch.linalg.vector_norm(vel, dim=1).max()))
            if stop_when_steady and len(hist["u_mean"]) > 3:
                prev = hist["u_mean"][-2]
                if (abs(hist["u_mean"][-1] - prev) <= 5e-3 * abs(u_max)
                        and t > 0.5 * t_end):
                    converged = True
                    break

    wall_clock = time.time() - t0
    profile = mean_profile(pos, -vel[:, 1], dx, width)
    xb = [p["x_m"] for p in profile]
    ua = analytic_profile(xb, a, nu, body)
    err = profile_error([p["u_mean"] for p in profile], ua, u_max)
    max_err = (max(p["u_mean"] for p in profile) - u_max) / u_max * 100.0
    shear_fit = shear_stress_fit(pos, -vel[:, 1], dx, width, nu, rho0, body)
    wall_off = effective_wall_offset(profile, a, nu, body)
    pfit_const = parabolic_fit(profile, a, nu, body, rho0)
    pfit_eff = parabolic_fit(profile, a + wall_off["offset_m"], nu, body, rho0)
    # profile scored against the parabola through the *effective* wall plane
    ae = a + wall_off["offset_m"]
    ue = analytic_profile(xb, ae, nu, body)
    err_eff = profile_error([p["u_mean"] for p in profile], ue, u_max)
    # and against the cruder, non-fitted hypothesis delta = dx/2, i.e. the
    # no-slip plane is the first wall-particle row.  If the fit lands on
    # half a spacing this is the explanation a user can act on.
    a_half = a + 0.5 * dx
    err_half = profile_error([p["u_mean"] for p in profile],
                             analytic_profile(xb, a_half, nu, body),
                             analytic_scalars(a_half, nu, body)["u_max_m_per_s"])
    # Flow rate, compared on the *same* discrete sample points: the mean over
    # the fluid columns is the midpoint rule for the average over the fluid
    # extent, so the analytic side has to be averaged there too (averaging the
    # parabola over [-a_half, a_half] instead would compare two different
    # intervals and read 8-12 % high for free).
    ubar_sph = sum(q["u_mean"] for q in profile) / len(profile)
    ubar_ex = sum(ua) / len(ua)
    ubar_ex_eff = sum(ue) / len(ue)
    ubar_ex_half = sum(analytic_profile(xb, a_half, nu, body)) / len(xb)
    # steady window: how much the mean velocity still drifts at the end
    steady = None
    if len(hist["u_mean"]) >= 3:
        steady = abs(hist["u_mean"][-1] - hist["u_mean"][-2]) / abs(hist["u_mean"][-1])

    return {
        "setup": {
            "dx_m": dx, "h_m": h, "channel_width_m": width,
            "periodic_length_m": length, "n_fluid": n_fluid,
            "n_wall": int(0 if wall is None else wall.shape[0]),
            "viscosity_m2_per_s": nu, "body_acceleration_m_per_s2": body,
            "rho0_kg_per_m3": rho0, "sound_speed_m_per_s": sound_speed,
            "gamma": cfg.gamma, "dt_s": dt, "steps": steps,
            "physical_time_s": t, "t_over_viscous_time":
                t / ref["viscous_time_s"],
            "lattice": "square", "wall_friction": cfg.wall_friction,
            "wall_model": "static dummy particles, Adami et al. (2012)",
            "periodic_axis": "y (streamwise) via image particles",
        },
        "reference": {**ref,
                      "wall_shear_Pa": rho0 * body * 2.0 * a * 0.5,
                      "profile": [{"x_m": xx, "u_m_per_s": uu}
                                  for xx, uu in zip(xb, ua)]},
        "profile": profile,
        "gates": {
            # convention A: the wall plane is the fluid face, x = +-a
            "profile_l2_rel_pct": err["l2_rel_pct"],
            "u_max_rel_pct": max_err,
            "within_3pct": abs(err["l2_rel_pct"]) < 3.0 and abs(max_err) < 3.0,
            "within_15pct": abs(err["l2_rel_pct"]) < 15.0 and abs(max_err) < 15.0,
            # convention B: the wall plane is where the solver actually puts it
            "profile_l2_rel_pct_effective_wall": err_eff["l2_rel_pct"],
            "u_max_rel_pct_effective_wall":
                (max(p["u_mean"] for p in profile)
                 - analytic_scalars(ae, nu, body)["u_max_m_per_s"])
                / analytic_scalars(ae, nu, body)["u_max_m_per_s"] * 100.0,
            "within_3pct_effective_wall": abs(err_eff["l2_rel_pct"]) < 3.0,
            "within_15pct_effective_wall": abs(err_eff["l2_rel_pct"]) < 15.0,
            # convention B', the fitted delta replaced by the a-priori dx/2
            "profile_l2_rel_pct_half_spacing_wall": err_half["l2_rel_pct"],
            "within_3pct_half_spacing_wall": abs(err_half["l2_rel_pct"]) < 3.0,
            "flow_rate_rel_pct_nominal": (ubar_sph - ubar_ex) / ubar_ex * 100.0,
            "flow_rate_rel_pct_half_spacing":
                (ubar_sph - ubar_ex_half) / ubar_ex_half * 100.0,
            "flow_rate_rel_pct_effective_wall":
                (ubar_sph - ubar_ex_eff) / ubar_ex_eff * 100.0,
            "mass_drift_rel": abs(hist["mass"][-1] - hist["mass"][0])
                              / hist["mass"][0],
            "density_min": hist["rho_min"][-1],
            "density_max": hist["rho_max"][-1],
            "wall_penetration_over_dx":
                (float(pos[:, 0].abs().max()) - a) / dx,
            "steady_drift_last_window": steady,
            "max_profile_std_over_umax":
                max(p["u_std"] for p in profile) / u_max,
            "converged": converged,
        },
        "viscosity_fit": {
            "nominal_wall": pfit_const,
            "effective_wall": pfit_eff,
        },
        "wall_condition": {
            **shear_fit,
            "effective_wall_offset_over_dx": wall_off["offset_over_dx"],
            "effective_wall_offset_m": wall_off["offset_m"],
            "first_fluid_layer_u_over_local_analytic":
                profile[0]["u_mean"] / analytic_profile([profile[0]["x_m"]], a,
                                                        nu, body)[0],
        },
        "max_pressure_Pa": hist["p_max"][-1],
        "history": hist,
        "wall_clock_s": wall_clock,
    }


def mean_profile(pos, u, dx, width):
    """Streamwise-averaged profile per fluid column (the flow is x-only)."""
    a = 0.5 * width
    col = torch.round((pos[:, 0] + a - 0.5 * dx) / dx).long()
    out = []
    for c in sorted(set(col.tolist())):
        m = col == c
        x = float(pos[m, 0].mean())
        uu = u[m]
        out.append({
            "column": int(c),
            "x_m": x,
            "x_over_a": x / a,
            "u_mean": float(uu.mean()),
            "u_std": float(uu.std()) if int(m.sum()) > 1 else 0.0,
            "n": int(m.sum()),
        })
    return out


def profile_error(u_sph, u_ref, u_max):
    num = 0.0
    den = 0.0
    for us, ur in zip(u_sph, u_ref):
        num += (us - ur) ** 2
        den += ur ** 2
    return {"l2_abs": math.sqrt(num / max(len(u_sph), 1)),
            "l2_rel_pct": math.sqrt(num / den) * 100.0 if den > 0 else float("nan"),
            "u_max_m_per_s": u_max}


def parabolic_fit(profile, a_eff, nu, body, rho0):
    """Fit ``u = C (a_eff^2 - x^2)`` and read the viscosity back out.

    ``C`` is one number for the whole profile (a least-squares fit through the
    origin), and ``nu_eff = g / (2 C)`` is the viscosity the *solution* shows,
    as opposed to the one the configuration asked for.  This is the headline
    verification quantity: it separates "is the viscous operator right" from
    "where does the solver put the wall".
    """
    num = 0.0
    den = 0.0
    for p in profile:
        basis = a_eff * a_eff - p["x_m"] ** 2
        num += p["u_mean"] * basis
        den += basis * basis
    c = num / den if den > 0 else float("nan")
    nu_eff = body / (2.0 * c) if c != 0 else float("inf")
    return {
        "C_fit_per_m_s": c,
        "C_exact_per_m_s": body / (2.0 * nu),
        "nu_effective_m2_per_s": nu_eff,
        "nu_rel_pct": (nu_eff - nu) / nu * 100.0,
        "tau_wall_from_fit_Pa": rho0 * nu_eff * 2.0 * c * a_eff,
    }


def shear_stress_fit(pos, u, dx, width, nu, rho0, body):
    """Wall shear measured from the discrete profile, compared with ``rho g a``.

    For the exact solution the wall shear is ``tau_w = rho * g * a`` (momentum
    balance over a half channel).  The discrete version uses the *slope of the
    first two fluid columns* (a one-sided, first-order estimate), which is the
    quantity an SPH user would actually trust near a wall.
    """
    a = 0.5 * width
    col = torch.round((pos[:, 0] + a - 0.5 * dx) / dx).long()
    cols = sorted(set(col.tolist()))
    u_of = {c: float(u[col == c].mean()) for c in cols}
    x_of = {c: float(pos[col == c, 0].mean()) for c in cols}
    c0, c1 = cols[0], cols[1]
    slope = (u_of[c1] - u_of[c0]) / (x_of[c1] - x_of[c0])
    tau_sph = rho0 * nu * abs(slope)
    tau_exact = rho0 * body * a
    return {"tau_wall_discrete_Pa": tau_sph,
            "tau_wall_exact_Pa": tau_exact,
            "tau_wall_rel_pct": (tau_sph - tau_exact) / tau_exact * 100.0,
            "du_dx_first_pair": slope}


def effective_wall_offset(profile, a, nu, body):
    """Half-width ``a_eff`` that minimises the L2 error of the parabola fit.

    The static dummy wall does not put the no-slip plane exactly at ``x = +-a``:
    the admitted slip/no-slip blend and the kernel truncation shift it.  A
    half-spacing shift is a 2.5 % change of ``a`` here, i.e. a 5 % change of
    ``u_max``, so the offset has to be reported instead of hidden.
    """
    best = {"offset_m": 0.0, "offset_over_dx": 0.0, "l2_rel_pct": None}
    dx = profile[1]["x_m"] - profile[0]["x_m"]
    for k in range(-400, 401):
        off = 0.5 * k * dx / 199.0
        ae = a + off
        num = 0.0
        den = 0.0
        for p in profile:
            ur = body / (2.0 * nu) * (ae * ae - p["x_m"] ** 2)
            num += (p["u_mean"] - ur) ** 2
            den += ur * ur
        err = math.sqrt(num / den) * 100.0 if den > 0 else float("inf")
        if best["l2_rel_pct"] is None or err < best["l2_rel_pct"]:
            best = {"offset_m": off, "offset_over_dx": off / dx,
                    "l2_rel_pct": err}
    return best


# --------------------------------------------------------------------------- #
# figure + doc
# --------------------------------------------------------------------------- #
def make_figure(runs, out_dir, stability=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    colors = plt.cm.viridis([0.15 + 0.6 * i / max(len(runs) - 1, 1)
                             for i in range(len(runs))])

    for run, c in zip(runs, colors):
        p = run["profile"]
        a = 0.5 * run["setup"]["channel_width_m"]
        nu = run["setup"]["viscosity_m2_per_s"]
        body = run["setup"]["body_acceleration_m_per_s2"]
        ae = a + run["wall_condition"]["effective_wall_offset_m"]
        tag = f"dx={run['setup']['dx_m']*1e3:.1f} mm"
        axes[0, 0].plot([q["x_m"] * 1e3 for q in p], [q["u_mean"] for q in p],
                        "o", ms=5, color=c, label=f"SPH {tag}")
        axes[0, 0].plot([q["x_m"] * 1e3 for q in run["reference"]["profile"]],
                        [q["u_m_per_s"] for q in run["reference"]["profile"]],
                        "k--", lw=1.0)
        axes[0, 1].plot([q["x_m"] * 1e3 for q in p],
                        [abs(q["u_mean"] - u) / run["reference"]["u_max_m_per_s"]
                         for q, u in zip(p, analytic_profile(
                             [q["x_m"] for q in p], ae, nu, body))],
                        "o-", ms=4, color=c, label=tag)
    axes[0, 0].set_xlabel("x [mm]"); axes[0, 0].set_ylabel(r"$u_y$ [m/s]")
    axes[0, 0].set_title("(a) steady profile vs $g(a^2-x^2)/2\\nu$ (dashed)")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("x [mm]")
    axes[0, 1].set_ylabel(r"$|u_{SPH}-u_{exact}|/u_{max}$")
    axes[0, 1].set_title("(b) pointwise error against the effective-wall parabola")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3, which="both")

    dxs = [r["setup"]["dx_m"] * 1e3 for r in runs]
    axes[1, 0].plot(dxs, [r["wall_condition"]["effective_wall_offset_over_dx"]
                          for r in runs], "o-", color="tab:blue",
                    label=r"fitted wall offset $\delta/dx$")
    axes[1, 0].axhline(0.5, color="k", ls="--", label="half a spacing")
    axes[1, 0].set_xlabel("dx [mm]")
    axes[1, 0].set_ylabel(r"$\delta/dx$")
    ax2 = axes[1, 0].twinx()
    ax2.plot(dxs, [r["viscosity_fit"]["effective_wall"]["nu_rel_pct"]
                   for r in runs], "s--", color="tab:red",
             label=r"$\nu_{eff}/\nu-1$ [%]")
    ax2.set_ylabel(r"$\nu_{eff}/\nu-1$ [%]", color="tab:red")
    ax2.axhline(0.0, color="tab:red", lw=0.6, alpha=0.4)
    axes[1, 0].set_title("(c) wall-plane offset and recovered viscosity")
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1, 0].legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    axes[1, 0].grid(alpha=0.3)

    if stability:
        h = stability["history"]
        axes[1, 1].plot(h["t"], h["rho_min"], "-", color="tab:blue",
                        label=r"min $\rho/\rho_0$ (real particles)")
        axes[1, 1].plot(h["t"], h["xmax_over_a"], "-", color="tab:green",
                        label=r"$\max|x|/a$ (1.0 = nominal wall face)")
        axr = axes[1, 1].twinx()
        axr.semilogy(h["t"], [max(v, 1e-3) for v in h["p_max"]], "--",
                     color="tab:red", label=r"$\max p$ [Pa]")
        axr.set_ylabel("max p [Pa]", color="tab:red")
        axes[1, 1].set_xlabel("t [s]")
        axes[1, 1].set_title("(d) long run: when the channel stops holding")
        h1, l1 = axes[1, 1].get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        axes[1, 1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    path = out / "poiseuille.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"figure -> {path}")
    return path


def write_doc(report, out_dir):
    """Inject every number from the JSON; nothing here is typed by hand."""
    runs = report["runs"]
    best = min(runs, key=lambda r: r["setup"]["dx_m"])
    stab = report.get("stability_run")
    rows = "\n".join(
        f"| {r['setup']['dx_m']*1e3:.1f} | {r['setup']['n_fluid']} | "
        f"{r['setup']['steps']} | {r['gates']['profile_l2_rel_pct']:.2f} % | "
        f"{r['gates']['u_max_rel_pct']:+.2f} % | "
        f"{r['wall_condition']['effective_wall_offset_over_dx']:+.3f} | "
        f"{r['gates']['profile_l2_rel_pct_effective_wall']:.3f} % | "
        f"{r['gates']['profile_l2_rel_pct_half_spacing_wall']:.3f} % | "
        f"{r['viscosity_fit']['effective_wall']['nu_rel_pct']:+.3f} % |"
        for r in runs)
    doc = f"""# Plane Poiseuille flow — analytic verification of the viscous term and the no-slip wall

Generated by `benchmarks/poiseuille/run.py` from `results.json`
(every number below is injected from that file, none is typed by hand).

## 1. Case

Gravity-driven plane channel in a streamwise-periodic domain:

| quantity | value |
|---|---|
| channel width `H` | {runs[0]['setup']['channel_width_m']} m (walls at `x = ±a`, `a = H/2`) |
| periodic length `L_y` | {runs[0]['setup']['periodic_length_m']} m (`16 dx`) |
| kinematic viscosity `nu` | {runs[0]['setup']['viscosity_m2_per_s']} m²/s |
| body acceleration `g` | {runs[0]['setup']['body_acceleration_m_per_s2']} m/s² (along `-y`) |
| density `rho0` | {runs[0]['setup']['rho0_kg_per_m3']} kg/m³ |
| artificial sound speed `c0` | {runs[0]['setup']['sound_speed_m_per_s']} m/s (`Mach = {runs[0]['reference']['u_max_m_per_s']/runs[0]['setup']['sound_speed_m_per_s']:.4f}`) |
| wall model | {runs[0]['setup']['wall_model']}, `wall_friction = {runs[0]['setup']['wall_friction']}` |
| scoring time | {best['setup']['physical_time_s']:.4f} s `= {best['setup']['t_over_viscous_time']:.2f}` viscous times |

Exact steady state: `u(x) = g(a² − x²)/(2ν)`, `u_max = {runs[0]['reference']['u_max_m_per_s']:.5f} m/s`,
`ū = (2/3)u_max = {runs[0]['reference']['u_mean_m_per_s']:.5f} m/s`,
`Re = u_max·H/ν = {runs[0]['reference']['reynolds_channel']:.1f}`.
The pressure field is uniform, so this case isolates the viscous operator and
the wall condition from the Tait-EOS/density noise that limits the
free-surface benchmarks.

## 2. Two wall conventions

The static dummy wall does **not** put the no-slip plane on the fluid face
`x = ±a`. The particles in the first wall layer sit half a spacing outside that
face, and the no-slip plane ends up on them. The table therefore scores the
same run twice:

* **nominal** — `a` is the fluid face, i.e. what a user of the code would
  assume by default;
* **effective** — `a_eff = a + δ`, with `δ` fitted by least squares.

| dx [mm] | particles | steps | L2 error (nominal) | u_max error (nominal) | δ/dx | L2 error (effective) | L2 error at δ=dx/2 | recovered `ν_eff/ν − 1` |
|---|---|---|---|---|---|---|---|---|
{rows}

## 3. Verdict

* Recovered viscosity against the profile fit: **{best['viscosity_fit']['effective_wall']['nu_rel_pct']:+.3f} %**
  (`< 3 %`? {'yes' if abs(best['viscosity_fit']['effective_wall']['nu_rel_pct']) < 3 else 'no'}) at
  `dx = {best['setup']['dx_m']*1e3:.1f} mm`.
* Profile L2 error with the effective wall plane:
  **{best['gates']['profile_l2_rel_pct_effective_wall']:.3f} %**
  (`< 3 %`? {'yes' if best['gates']['within_3pct_effective_wall'] else 'no'}).
* Profile L2 error with the nominal wall plane:
  **{best['gates']['profile_l2_rel_pct']:.2f} %**
  (`< 15 %`? {'yes' if best['gates']['within_15pct'] else 'no'}) — the
  whole deviation is the half-spacing wall offset, not a viscosity error.
* Fitted offset `δ/dx = {best['wall_condition']['effective_wall_offset_over_dx']:.3f}`
  at every resolution (see the table): a **first-order** boundary
  discretisation error, constant in `dx`, so it does not vanish under
  refinement unless the wall convention is fixed. Putting the no-slip plane
  a-priori on the first wall-particle row (`δ = dx/2`, no fit) already gives
  `{best['gates']['profile_l2_rel_pct_half_spacing_wall']:.3f} %`, so the
  offset is half a spacing and not an extra calibration constant.
* Volume flow rate (mean velocity over the fluid columns): `{best['gates']['flow_rate_rel_pct_nominal']:+.2f} %`
  against the nominal wall, `{best['gates']['flow_rate_rel_pct_half_spacing']:+.2f} %` with the
  a-priori `δ = dx/2`, `{best['gates']['flow_rate_rel_pct_effective_wall']:+.2f} %` with the fitted `δ`.
* Internal consistency: `max(u_std)/u_max = {best['gates']['max_profile_std_over_umax']:.2e}`
  across the periodic seam, i.e. the image-by-hand periodicity leaves no seam
  artefact; mass drift `{best['gates']['mass_drift_rel']:.1e}`; no wall
  penetration (`max|x|/a = {1.0 + best['gates']['wall_penetration_over_dx']*best['setup']['dx_m']/(0.5*best['setup']['channel_width_m']):.4f}`).
"""
    if stab:
        doc += f"""
## 4. Limitation: the channel does not hold for ever

A long run at `dx = {stab['setup']['dx_m']*1e3:.1f} mm` (panel (d) of the
figure) reaches the same steady state and then degrades:

* `min ρ/ρ0` over the real particles falls to
  `{min(stab['history']['rho_min']):.4f}` by `t = {stab['history']['t'][-1]:.3f} s`;
* `max p` grows from `{stab['history']['p_max'][0]:.2f}` Pa to
  `{stab['history']['p_max'][-1]:.1f}` Pa;
* `max|x|/a` grows past 1.0, i.e. fluid particles walk into the wall layer.

Cause: in this configuration the pressure has no work to do (the body force is
balanced by shear), so the gauge pressure is ~0 everywhere and
`tait_pressure` clamps it at 0. The clamped EOS therefore provides **no
restoring force against rarefaction**, and the regularised viscous term is
free to drive the classic pairing/clumping instability; once a particle
crosses the nominal wall face it is compressed against the wall layer and the
run diverges. The same root cause (a zero/clamped pressure baseline) is what
keeps the Young-Laplace pressure jump WITHHELD in
`docs/verification_surface_tension_consistent.md`. A background pressure
`p0` in the Tait EOS is the single change that would address both; it is
recorded as P0 in `backlog_update_m2174.md` and was **not** implemented here.
Consequence for this benchmark: it is scored at the viscous time scale
`t ≈ (2a)²/ν`, and any configuration that needs a long-time confined run has to
be checked against this failure mode first.
"""
    doc += """
## 5. How to run

```bash
PYTHONPATH=src python benchmarks/poiseuille/run.py --dx 0.004,0.0025,0.002 --t-end 0.08
PYTHONPATH=src pytest tests/test_poiseuille.py -q
```
"""
    path = (Path(__file__).resolve().parents[2]
            / "docs" / "benchmarks" / "poiseuille.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    print(f"doc -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", default="0.004,0.0025,0.002")
    ap.add_argument("--width", type=float, default=0.02)
    ap.add_argument("--length", type=float, default=None,
                    help="streamwise periodic length; default 16*dx")
    ap.add_argument("--nu", type=float, default=1.0e-2)
    ap.add_argument("--body", type=float, default=9.81)
    ap.add_argument("--sound-speed", type=float, default=2.0)
    ap.add_argument("--t-end", type=float, default=0.08)
    ap.add_argument("--stability-dx", type=float, default=0.0025,
                    help="resolution of the long-time stability run")
    ap.add_argument("--stability-t-end", type=float, default=0.4)
    ap.add_argument("--steps-limit", type=int, default=200000)
    ap.add_argument("--out", default="benchmarks/poiseuille")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for dx in [float(v) for v in args.dx.split(",")]:
        print(f"--- dx = {dx*1e3:.2f} mm")
        r = run_case(dx=dx, width=args.width, length=args.length, nu=args.nu,
                     body=args.body, sound_speed=args.sound_speed,
                     t_end=args.t_end, steps_limit=args.steps_limit)
        print(json.dumps({k: r[k] for k in ("gates", "wall_condition",
                                            "viscosity_fit")}, indent=1))
        runs.append(r)
    stability = None
    if not args.no_stability:
        print(f"--- stability run dx = {args.stability_dx*1e3:.2f} mm")
        stability = run_case(dx=args.stability_dx, width=args.width,
                             length=args.length, nu=args.nu, body=args.body,
                             sound_speed=args.sound_speed,
                             t_end=args.stability_t_end, sample_every=50,
                             steps_limit=args.steps_limit,
                             stop_when_steady=False)
        stability.pop("profile", None)
        stability.pop("reference", None)
        print(f"    min rho/rho0 = {min(stability['history']['rho_min']):.4f}, "
              f"max p = {stability['history']['p_max'][-1]:.1f} Pa at "
              f"t = {stability['history']['t'][-1]:.3f} s")
    report = {
        "benchmark": "plane Poiseuille (gravity-driven, streamwise periodic)",
        "reference": "u(x) = g (a^2 - x^2) / (2 nu)",
        "runs": runs,
        "stability_run": stability,
    }
    (out / "results.json").write_text(json.dumps(report, indent=1))
    if not args.no_figure:
        make_figure(runs, out, stability)
    write_doc(report, out)
    print(f"json -> {out/'results.json'}")


if __name__ == "__main__":
    main()
