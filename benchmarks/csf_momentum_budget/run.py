"""CSF momentum budget: which density turns the interface traction into an acceleration?

What is checked, and why it is the right observable
--------------------------------------------------
The CSF traction (Brackbill, Kothe & Zemach 1992) is a force per unit *volume*

    f = -sigma kappa delta_s n ,      int delta_s dn = 1 ,

obtained by spreading a force per unit *area* over the interface shell.  The
particle that carries it is accelerated by ``a = f / rho``.  Which ``rho`` is
the question here:

``summation``  ``rho_i`` = kernel-summed density.  At a free surface the kernel
               is truncated, so ``rho_i`` is 15-25 % below ``rho0`` in the
               outermost layer; ``m_i a_i = rho0 V_i f_i / rho_i`` then exceeds
               the geometric momentum ``V_i f_i`` by the same factor.
``rho0``       the reference density: the mass that carries the traction is
               ``rho0 V_i`` with the geometric volume ``V_i = m_i/rho0``, so
               ``m_i a_i = V_i f_i`` exactly -- a surface integral, independent
               of how badly the free-surface kernel sum is biased.

The three analytic reference values used below do **not** depend on R, on the
resolution, or on the curvature operator, so none of them can be satisfied by
calibration:

* net momentum injection ``sum_i m_i a_i`` on a **closed** interface in vacuum
  must be exactly 0 (Newton's third law: the fluid cannot pull itself by the
  bootstraps).  This is the momentum-budget gate proper.
* net torque ``sum_i r_i x m_i a_i`` about the centre of mass must be 0
  (the traction of a circular interface is radial, so it can carry no torque).
* the cut force ``sum_{x>0} V_i f_i,x = -2 sigma`` (2-D, force per unit depth).

For the geometric assembly the first two are exact by construction up to the
discretisation of ``f`` itself; for the summation assembly they are **not**,
because ``rho0/rho_i`` is a particle-wise, non-antisymmetric factor that
multiplies the traction.  Comparing the two is therefore a quantitative
statement about which assembly conserves momentum, not a tautology: the
difference is exactly the free-surface density deficit of the CSF shell.

Usage::

    PYTHONPATH=src python benchmarks/csf_momentum_budget/run.py            # full
    PYTHONPATH=src python benchmarks/csf_momentum_budget/run.py --quick    # CI
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.csf_consistent import csf_fields, surface_tension_acceleration
from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
SCHEMES = ("summation", "rho0")


# --------------------------------------------------------------------------- #
# geometry (identical to benchmarks/laplace_law so the numbers are comparable)
# --------------------------------------------------------------------------- #
def hexagonal_droplet(radius: float, dx: float, rho0: float):
    """Hexagonally packed circular drop centred at the origin."""
    dy = dx * math.sqrt(3.0) / 2.0
    mass = rho0 * dx * dy
    pts = []
    j = 0
    y = -radius
    while y <= radius:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), mass, dtype=torch.float64)


def make_config(dx: float, sigma: float, rho0: float, sound_speed: float,
                accel_density: str, curvature: str = "fit_quadratic") -> Config:
    return Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                  gamma=7.0, gravity=0.0, viscosity=0.01,
                  surface_tension_coefficient=sigma, csf_scheme="consistent",
                  csf_curvature=curvature, csf_accel_density=accel_density)


# --------------------------------------------------------------------------- #
# static momentum budget on the initial lattice
# --------------------------------------------------------------------------- #
def static_budget(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0,
                  sound_speed=20.0, curvature="fit_quadratic"):
    """Traction field ``f`` and both accelerations on the same configuration."""
    cfg = make_config(dx, sigma, rho0, sound_speed, "summation", curvature)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    vol = mass / rho0

    acc_sum, state = surface_tension_acceleration(
        pos, rho, mass, pairs, h, sigma, rho0=rho0,
        curvature_scheme=curvature, accel_density="summation")
    acc_ref, _ = surface_tension_acceleration(
        pos, rho, mass, pairs, h, sigma, rho0=rho0,
        curvature_scheme=curvature, accel_density="rho0")
    # f = a_rho0 * rho0 is the traction exactly as the solver builds it.
    force = acc_ref * rho0

    interface = state.delta_s > 0.05 * state.delta_s.max()
    rho_ratio = (rho[interface] / rho0)

    # momentum injection: sum m a inside a closed interface in vacuum is 0
    scale = float((vol * torch.linalg.vector_norm(force, dim=1)).sum())
    com = (mass[:, None] * pos).sum(0) / mass.sum()
    lever = pos - com
    torque_scale = scale * float(torch.linalg.vector_norm(lever, dim=1).max())

    half = pos[:, 0] > 0.0
    rows = {}
    for scheme, acc in (("summation", acc_sum), ("rho0", acc_ref)):
        p_dot = (mass[:, None] * acc).sum(0)
        torque = float((lever[:, 0] * mass * acc[:, 1]
                        - lever[:, 1] * mass * acc[:, 0]).sum())
        cut = float((vol[half, None] * force[half]).sum(0)[0])
        rows[scheme] = {
            "net_force_N": [float(p_dot[0]), float(p_dot[1])],
            "net_force_magnitude_N": float(torch.linalg.vector_norm(p_dot)),
            "net_force_over_traction": float(torch.linalg.vector_norm(p_dot)) / scale,
            "torque_Nm": torque,
            "torque_over_reference": abs(torque) / max(torque_scale, 1e-30),
            "cut_force_N_per_m": cut,
            "cut_force_over_2sigma": cut / (-2.0 * sigma),
            "traction_scale_N": scale,
        }

    return {
        "dx_m": dx,
        "h_m": h,
        "n_particles": int(pos.shape[0]),
        "n_interface": int(interface.sum()),
        "curvature": curvature,
        "interface_density": {
            "mean_rho_over_rho0": float(rho_ratio.mean()),
            "min_rho_over_rho0": float(rho_ratio.min()),
            "momentum_injection_factor": float((rho0 / rho[interface]).mean()),
            "min_momentum_injection_factor": float((rho0 / rho[interface]).max()),
        },
        "assemblies": rows,
    }


# --------------------------------------------------------------------------- #
# dynamic budget: does the drop's centre of mass stay put?
# --------------------------------------------------------------------------- #
def dynamic_budget(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0,
                   sound_speed=20.0, t_over_tsigma=2.8, accel_density="rho0",
                   curvature="fit_quadratic", output_every=20):
    """Relax a static drop and track the centre of mass and the kinetic energy."""
    cfg = make_config(dx, sigma, rho0, sound_speed, accel_density, curvature)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    steps = max(1, int(round(t_over_tsigma * t_sigma / dt)))
    v_ref = math.sqrt(sigma / (rho0 * radius))

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vel = torch.zeros_like(pos)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                     cell_linked_pairs(pos, 2.0 * h))

    com0 = (mass[:, None] * pos).sum(0) / mass.sum()
    hist = {"t_over_tsigma": [], "com_drift_over_h": [],
            "vmax_over_vref": [], "ke_cm_fraction": []}
    e_cm0 = None
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % output_every == 0 or step == steps - 1:
            com = (mass[:, None] * pos).sum(0) / mass.sum()
            momentum = mass.sum() * ((mass[:, None] * vel).sum(0) / mass.sum())
            ke_cm = float((momentum ** 2).sum() / (2.0 * mass.sum()))
            m = mass[:, None]
            ke_internal = float((0.5 * m * (vel - momentum / mass.sum()) ** 2).sum())
            if e_cm0 is None:
                e_cm0 = ke_cm
            hist["t_over_tsigma"].append((step + 1) * dt / t_sigma)
            hist["com_drift_over_h"].append(
                float(torch.linalg.vector_norm(com - com0)) / h)
            hist["vmax_over_vref"].append(
                float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref)
            hist["ke_cm_fraction"].append(
                ke_cm / max(ke_cm + ke_internal, 1e-30))

    r = torch.linalg.vector_norm(pos, dim=1)
    core = r < 0.4 * radius
    com = (mass[:, None] * pos).sum(0) / mass.sum()
    return {
        "accel_density": accel_density,
        "dx_m": dx,
        "h_m": h,
        "steps": steps,
        "dt_s": dt,
        "t_end_over_tsigma": steps * dt / t_sigma,
        "v_ref_m_per_s": v_ref,
        "com_drift_final_over_h": hist["com_drift_over_h"][-1],
        "com_drift_max_over_h": max(hist["com_drift_over_h"]),
        "vmax_over_vref_final": hist["vmax_over_vref"][-1],
        "vmax_over_vref_max": max(hist["vmax_over_vref"]),
        "ke_cm_fraction_max": max(hist["ke_cm_fraction"]),
        "p_interior_Pa": float(tait_pressure(rho, cfg)[core].mean()),
        "p_interior_over_law": float(tait_pressure(rho, cfg)[core].mean()
                                    / (sigma / radius)),
        "rho_core_over_rho0": float(rho[core].mean() / rho0),
        "history": hist,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(static_rows, dynamic_rows, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    colours = {"summation": "tab:red", "rho0": "tab:blue"}

    ax = axes[0, 0]
    for scheme in SCHEMES:
        y = [row["assemblies"][scheme]["net_force_over_traction"] for row in static_rows]
        ax.loglog([row["dx_m"] * 1e3 for row in static_rows], y, "o-",
                  color=colours[scheme], label=f"a = f/rho ({scheme})")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel(r"$|\sum m\,a| \,/\, \sum V|f|$")
    ax.set_title("net momentum injection (exact 0)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    width = 0.35
    xs = [row["dx_m"] * 1e3 for row in static_rows]
    for k, scheme in enumerate(SCHEMES):
        y = [row["assemblies"][scheme]["cut_force_over_2sigma"] for row in static_rows]
        ax.bar([x + (k - 0.5) * width for x in xs], y, width,
               color=colours[scheme], label=scheme)
    ax.axhline(-1.0, color="k", ls="--", lw=1)
    ax.axhspan(-1.15, -0.85, color="grey", alpha=0.2)
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel(r"$F_{cut} / (2\sigma)$")
    ax.set_title("cut force identity (target -1, 15 % band)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    for row in dynamic_rows:
        hist = row["history"]
        ax.plot(hist["t_over_tsigma"], hist["com_drift_over_h"], "-",
                color=colours[row["accel_density"]],
                label=f"a = f/rho ({row['accel_density']})")
    ax.set_xlabel(r"$t / t_\sigma$")
    ax.set_ylabel(r"$|\Delta x_{cm}| / h$")
    ax.set_title("centre-of-mass drift")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for row in dynamic_rows:
        hist = row["history"]
        ax.semilogy(hist["t_over_tsigma"], hist["vmax_over_vref"], "-",
                    color=colours[row["accel_density"]],
                    label=f"a = f/rho ({row['accel_density']})")
    ax.axhline(0.1, color="k", ls="--", lw=1, label="Adami 0.1")
    ax.set_xlabel(r"$t / t_\sigma$")
    ax.set_ylabel(r"$|v|_{max} / \sqrt{\sigma/\rho R}$")
    ax.set_title("spurious current")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle("CSF momentum budget: acceleration assembly f/rho_sum vs f/rho0")
    fig.tight_layout()
    path = out_dir / "csf_momentum_budget.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--sigma", type=float, default=7.2)
    parser.add_argument("--rho0", type=float, default=1000.0)
    parser.add_argument("--sound-speed", type=float, default=20.0)
    parser.add_argument("--curvature", default="fit_quadratic")
    parser.add_argument("--dx-list", nargs="+", type=float,
                        default=[0.003, 0.002, 0.0015])
    parser.add_argument("--dynamic-dx", type=float, default=0.002)
    parser.add_argument("--t-over-tsigma", type=float, default=2.8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--figure", action="store_true")
    parser.add_argument("--json-out", default=str(HERE / "validation_report.json"))
    args = parser.parse_args()

    if args.quick:
        args.dx_list = [0.002]
        args.t_over_tsigma = 0.4

    started = time.perf_counter()
    static_rows = []
    for dx in args.dx_list:
        row = static_budget(args.radius, dx, args.sigma, args.rho0,
                            args.sound_speed, args.curvature)
        static_rows.append(row)
        print(f"dx={dx*1e3:.2f} mm  N={row['n_particles']:4d}  "
              f"rho_int/rho0={row['interface_density']['mean_rho_over_rho0']:.3f}")
        for scheme in SCHEMES:
            a = row["assemblies"][scheme]
            print(f"    {scheme:9s}  |sum m a|/scale={a['net_force_over_traction']:.4f}"
                  f"  torque/ref={a['torque_over_reference']:.2e}"
                  f"  F_cut/2sigma={a['cut_force_over_2sigma']:+.4f}")

    dynamic_rows = []
    for scheme in SCHEMES:
        row = dynamic_budget(args.radius, args.dynamic_dx, args.sigma, args.rho0,
                             args.sound_speed, args.t_over_tsigma, scheme,
                             args.curvature)
        dynamic_rows.append(row)
        print(f"relax {scheme:9s}  |dx_cm|/h={row['com_drift_final_over_h']:.3f}  "
              f"|v|max/vref={row['vmax_over_vref_max']:.3f}  "
              f"p_int/(sigma/R)={row['p_interior_over_law']:+.3f}")

    report = {
        "benchmark": "csf_momentum_budget",
        "reference": {
            "net_force_N": 0.0,
            "torque_Nm": 0.0,
            "cut_force_over_2sigma": -1.0,
            "laplace_delta_p_Pa": args.sigma / args.radius,
            "gate_pct": 15.0,
            "adami_spurious_current": 0.1,
        },
        "setup": {
            "radius_m": args.radius,
            "sigma_N_per_m": args.sigma,
            "rho0": args.rho0,
            "sound_speed": args.sound_speed,
            "curvature": args.curvature,
            "dx_list": args.dx_list,
            "dynamic_dx": args.dynamic_dx,
            "t_over_tsigma": args.t_over_tsigma,
        },
        "static": static_rows,
        "dynamic": dynamic_rows,
        "wall_clock_s": time.perf_counter() - started,
    }

    out = Path(args.json_out)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")

    if args.figure:
        fig = make_figures(static_rows, dynamic_rows)
        print(f"wrote {fig}")


if __name__ == "__main__":
    main()