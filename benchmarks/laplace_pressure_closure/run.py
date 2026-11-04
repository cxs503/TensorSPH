"""Close the free-surface surface-delta function and the Young-Laplace jump.

Why this benchmark exists
-------------------------
``benchmarks/laplace_pressure_floor/`` (m2971) decomposed the WITHHELD
Young-Laplace check into four error sources and handed the next mission a P0:

    "add shell-pressure extrapolation (Adami/Ihmsen/Adams 2012 §3.3) for
     free-surface particles; acceptance = every case inside 15 %".

This benchmark measures the momentum budget *before* implementing that, and the
measurement says the P0 was aimed one step too late.  On the relaxed droplet the
radial balance reads

    dp/dr = rho a_r ,  p(R) = 0    =>    p_core = int_0^R (rho a)_r dr .

Splitting the integral into the two forces that act (gravity = 0, the drop is at
rest so viscosity contributes nothing):

    int (rho a_pressure)_r dr   =  the measured ``p_core`` (self-consistent)
    int (rho a_csf     )_r dr   =  0.44 sigma/R      <-- the force itself is short

so the interior pressure cannot rise to ``sigma/R`` no matter how the pressure
field is closed at the surface.  The missing factor is in the CSF surface delta
function ``delta_s = |grad c|``:

    sum_i V_i delta_s,i / (2 pi R) = 0.53  instead of 1

The diffuse-interface colour profile spans both sides of the free surface, but
the particle set only covers the fluid side: the outer half of the profile
(c < 0.5) lives in vacuum and is never sampled, so ``delta_s`` integrates to
~0.5 of unit and the surface tension force is short by the same factor.  This is
a property of free-surface SPH, not of the EOS or of the pressure closure: on the
un-relaxed lattice the same ratio is 0.89, i.e. it degrades as the interface
blurs.

The fix
-------
``Config.csf_delta_scale`` multiplies ``delta_s``.  2.0 completes the missing
outer half of a *planar* profile (the ``c(n) + c(-n) = 1`` symmetry of a diffuse
interface); for a convex interface of thickness 2h the exact 2-D volume ratio is
(1 - h/R)/(1 + h/R) = 0.88 at h/R = 0.13, so 2.0 slightly over-completes and the
residual stays negative, which is what the numbers show.  The factor is exposed
as a knob with a legacy default of 1.0 so that every previously recorded result
stays bit-reproducible; free-surface cases should opt in.

Two exact identities are used as the *reference* for the force model, so the
closure is validated independently of the pressure field:

  1. unit normalisation     (1 / 2 pi R) sum_i V_i delta_s,i = 1
  2. 2-D virial            -(1 / 2 sigma) sum_{x>0} V_i F_x,i = 1
                           (the surface tension force on a half disk is 2 sigma
                            per unit depth, exact for any radius)

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_closure/run.py
    PYTHONPATH=src python benchmarks/laplace_pressure_closure/run.py --quick
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.csf_consistent import csf_fields
from tensorsph.droplet_impact import (Config, density_from_neighbor_pairs, rhs,
                                     stable_dt, tait_pressure)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Reuse the floor benchmark's verified droplet construction and read-outs.
_spec = importlib.util.spec_from_file_location(
    "laplace_pressure_floor_run", HERE.parent / "laplace_pressure_floor" / "run.py")
_floor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_floor)

RADIUS = _floor.RADIUS
SIGMA = _floor.SIGMA
RHO0 = _floor.RHO0
GAMMA = _floor.GAMMA
VISCOSITY = _floor.VISCOSITY
DP_ANALYTIC = _floor.DP_ANALYTIC      # sigma / R = 360 Pa (2-D droplet)
T_SIGMA = _floor.T_SIGMA
PERIMETER = 2.0 * math.pi * RADIUS


def make_config(dx: float, c0: float, delta_scale: float) -> Config:
    return Config(dx=dx, h_factor=_floor.H_FACTOR, rho0=RHO0, sound_speed=c0,
                  gamma=GAMMA, gravity=0.0, viscosity=VISCOSITY,
                  surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                  csf_curvature="fit_quadratic", csf_delta_scale=delta_scale)


# --------------------------------------------------------------------------- #
# the two exact identities, measured on a particle state
# --------------------------------------------------------------------------- #
def force_audit(pos, mass, rho, cfg):
    """Normalisation of the surface delta and the exact 2-D virial."""
    n = pos.shape[0]
    h = cfg.h_factor * cfg.dx
    near = cell_linked_pairs(pos, 2.0 * h)
    state = csf_fields(pos, mass, near, h, RHO0, wall=None, contact_angle=None,
                       curvature_scheme=cfg.csf_curvature,
                       curvature_stencil=4.0)
    delta_s = state.delta_s.clone()
    flat = delta_s <= 1.0e-3 * float(delta_s.max())
    delta_s[flat] = 0.0
    kappa = state.kappa
    iface = ~flat
    volume = mass / rho

    scale = float(cfg.csf_delta_scale)
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r[:, None].clamp_min(1e-30)

    # 1. unit normalisation: sum V delta_s / perimeter
    norm_scale = float((volume * delta_s * scale).sum()) / PERIMETER

    # 2. virial: force on the half disk x > 0 is 2 sigma per unit depth
    acc = compute_surface_tension_force(pos, rho, mass, near, cfg)
    force = acc * rho[:, None]                    # N/m^3
    half = (pos[:, 0] > 0.0) & (r < 1.15 * RADIUS)
    virial = -float((volume[half] * force[half, 0]).sum()) / (2.0 * SIGMA)

    # radial force-density integral: int (rho a)_r dr, equals p_core in balance
    f_r = (force * rhat).sum(dim=1)
    edges = torch.linspace(0.0, RADIUS, 61)
    radial_integral = 0.0
    for k in range(60):
        band = (r >= edges[k]) & (r < edges[k + 1])
        if bool(band.any()):
            radial_integral += float(f_r[band].mean()) * float(
                edges[k + 1] - edges[k])
    return {
        "delta_scale": scale,
        "interface_particles": int(iface.sum()),
        "kappa_mean_interface": float(kappa[iface].mean()),
        "kappa_ratio_to_1_over_R": float(kappa[iface].mean()) * RADIUS,
        "delta_s_unit_integral": norm_scale,
        "virial_ratio": virial,
        "radial_force_integral_Pa": -radial_integral,
        "radial_ratio_to_sigma_over_R": -radial_integral / DP_ANALYTIC,
    }


# --------------------------------------------------------------------------- #
# dynamics
# --------------------------------------------------------------------------- #
def relax(dx: float, c0: float, delta_scale: float,
          t_end_over_t_sigma: float = 1.5, max_steps: int = 4000,
          sample_every: int = 50, keep_state: bool = False):
    cfg = make_config(dx, c0, delta_scale)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_end = t_end_over_t_sigma * T_SIGMA
    steps = min(max_steps, int(math.ceil(t_end / dt)))
    dt = t_end / steps

    pos, mass, h, _, bias, bias_after = _floor.calibrated_droplet(dx)
    vel = torch.zeros_like(pos)
    v_ref = math.sqrt(SIGMA / (RHO0 * RADIUS))

    trace = {"t": [], "p_core": [], "vmax_over_vref": [], "rho_core_rel": []}
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % sample_every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
            trace["t"].append((step + 1) * dt / T_SIGMA)
            trace["p_core"].append(float(pressure[r < 0.5].mean()))
            trace["vmax_over_vref"].append(
                float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref)
            trace["rho_core_rel"].append(float(rho[r < 0.5].mean() / RHO0 - 1.0))

    r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
    p_core = float(pressure[r < 0.5].mean())
    n_tail = max(1, int(0.4 * len(trace["p_core"])))
    p_tail = sum(trace["p_core"][-n_tail:]) / n_tail

    def score(value):
        err = (value - DP_ANALYTIC) / DP_ANALYTIC
        return {"delta_p_Pa": value, "rel_error_pct": err * 100.0,
                "within_3pct": abs(err) < 0.03, "within_15pct": abs(err) < 0.15}

    run = {
        "dx_m": dx, "sound_speed": c0, "delta_scale": delta_scale,
        "h_m": h, "n_particles": int(pos.shape[0]), "steps": steps, "dt_s": dt,
        "mass_calibration_bias": bias, "mass_calibration_after": bias_after,
        "p_core_final_Pa": p_core,
        "absolute_time_average_tail": score(p_tail),
        "absolute_final": score(p_core),
        "density": {
            "rho_core_over_rho0": float(rho[r < 0.5].mean() / RHO0),
            "compression_required": (1.0 + DP_ANALYTIC
                                     / (RHO0 * c0 ** 2 / GAMMA)) ** (1.0 / GAMMA) - 1.0,
        },
        "spurious_current": {"vmax_over_vref_final": trace["vmax_over_vref"][-1]},
        "finite": bool(torch.isfinite(pos).all() and torch.isfinite(vel).all()),
        "pressure_profile": _floor.radial_profile(pos, pressure),
        "frame": {"pos": pos.tolist(), "pressure": pressure.tolist()},
        "trace": trace,
    }
    run["force_audit"] = force_audit(pos, mass, rho, cfg)
    if keep_state:
        run["_state"] = (pos, mass, rho, cfg)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    dxs = [0.002] if args.quick else [0.002, 0.0015]
    c0s = [20.0] if args.quick else [20.0, 5.0]

    # ---- static audit on the calibrated lattice (no dynamics) --------------
    lattice_audit = []
    for dx in dxs:
        pos, mass, h, _, bias, _ = _floor.calibrated_droplet(dx)
        rho = density_from_neighbor_pairs(pos, mass, h,
                                         cell_linked_pairs(pos, 2.0 * h))
        row = {"dx_m": dx, "n_particles": int(pos.shape[0]),
               "lattice_bias_rel_density": bias, "scales": {}}
        for scale in (1.0, 2.0):
            row["scales"][f"{scale:g}"] = force_audit(
                pos, mass, rho, make_config(dx, 20.0, scale))
        lattice_audit.append(row)
        a1 = row["scales"]["1"]
        a2 = row["scales"]["2"]
        print(f"lattice dx={dx:g}: delta_s integral {a1['delta_s_unit_integral']:.3f}"
              f" -> {a2['delta_s_unit_integral']:.3f};  virial"
              f" {a1['virial_ratio']:.3f} -> {a2['virial_ratio']:.3f}", flush=True)

    # ---- dynamics: 4 acceptance points with the closure, 2 legacy controls -
    runs = []
    for dx in dxs:
        for c0 in c0s:
            run = relax(dx, c0, delta_scale=2.0)
            runs.append(run)
            t = run["absolute_time_average_tail"]
            a = run["force_audit"]
            print(f"closure dx={dx:g} c0={c0:5g}: tail {t['delta_p_Pa']:7.2f} Pa "
                  f"({t['rel_error_pct']:+6.1f} %)  virial {a['virial_ratio']:.3f}"
                  f"  delta_s {a['delta_s_unit_integral']:.3f}", flush=True)

    controls = []
    for c0 in c0s:
        run = relax(dxs[0], c0, delta_scale=1.0)
        controls.append(run)
        t = run["absolute_time_average_tail"]
        print(f"legacy  dx={dxs[0]:g} c0={c0:5g}: tail {t['delta_p_Pa']:7.2f} Pa "
              f"({t['rel_error_pct']:+6.1f} %)", flush=True)

    for run in runs + controls:
        run.pop("_state", None)
        run["frame"] = {"pos": run["frame"]["pos"][:0],
                        "pressure": run["frame"]["pressure"][:0]}

    report = {
        "benchmark": "laplace_pressure_closure",
        "reference": {"delta_p_Pa": DP_ANALYTIC, "sigma_N_per_m": SIGMA,
                      "radius_m": RADIUS, "rho0": RHO0,
                      "t_sigma_s": T_SIGMA,
                      "identities": {
                          "surface_delta_unit_integral": 1.0,
                          "virial_ratio": 1.0,
                          "note": "2-D droplet: sigma/R; half-disk virial 2 sigma "
                                  "per unit depth; delta_s int dn = 1"}},
        "lattice_audit": lattice_audit,
        "runs": runs,
        "controls": controls,
        "acceptance": {
            "criterion": "|rel_error_pct| < 15 on every closure run",
            "measured_max_abs_rel_error_pct": max(
                abs(r["absolute_time_average_tail"]["rel_error_pct"]) for r in runs),
            "passed": all(r["absolute_time_average_tail"]["within_15pct"]
                          for r in runs),
            "legacy_max_abs_rel_error_pct": max(
                abs(r["absolute_time_average_tail"]["rel_error_pct"])
                for r in controls),
        },
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    report["figures"] = make_figures(report)
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print("acceptance:", report["acceptance"], flush=True)
    print("wrote", HERE / "report.json", report["figures"], flush=True)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(report, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, controls = report["runs"], report["controls"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

    ax = axes[0][0]
    for run, style in ((controls[0], "s--"), (runs[0], "o-")):
        r = [p["r_over_R_mid"] for p in run["pressure_profile"]]
        p = [p["mean"] for p in run["pressure_profile"]]
        ax.plot(r, p, style, ms=3,
                label=rf"$\delta_s\times${run['delta_scale']:g}")
    ax.axhline(DP_ANALYTIC, color="k", ls=":", lw=1.2,
               label=rf"$\sigma/R$ = {DP_ANALYTIC:.0f} Pa")
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$p$ [Pa]")
    ax.set_title(rf"(a) pressure profile, $c_0$={runs[0]['sound_speed']:g} m/s,"
                 rf" $\Delta x$={runs[0]['dx_m']*1000:.1f} mm")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    labels, legacy, closed = [], [], []
    for ctrl, run in zip(controls, runs[:len(controls)]):
        labels.append(rf"$c_0$={ctrl['sound_speed']:g}")
        legacy.append(ctrl["absolute_time_average_tail"]["rel_error_pct"])
        closed.append(run["absolute_time_average_tail"]["rel_error_pct"])
    idx = torch.arange(len(labels)).numpy()
    ax.bar(idx - 0.2, legacy, 0.4, label=r"legacy $\delta_s$ (scale 1)")
    ax.bar(idx + 0.2, closed, 0.4, label=r"closed $\delta_s$ (scale 2)")
    ax.axhspan(-15, 15, color="0.85", zorder=0, label="±15 % criterion")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"error in $p_{\rm core}$ vs $\sigma/R$ [%]")
    ax.set_title(rf"(b) jump error, $\Delta x$={runs[0]['dx_m']*1000:.1f} mm")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][0]
    cases = []
    for run in controls + runs:
        a = run["force_audit"]
        cases.append((f"{run['dx_m']*1000:.1f}mm\nc0={run['sound_speed']:g}\n"
                      f"x{a['delta_scale']:g}", a["delta_s_unit_integral"],
                      a["virial_ratio"]))
    idx = torch.arange(len(cases)).numpy()
    ax.bar(idx - 0.2, [c[1] for c in cases], 0.4, label=r"$\sum V\delta_s/2\pi R$")
    ax.bar(idx + 0.2, [c[2] for c in cases], 0.4, label="virial / 2$\\sigma$")
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="exact 1")
    ax.set_xticks(idx)
    ax.set_xticklabels([c[0] for c in cases], fontsize=6)
    ax.set_title("(c) surface-delta normalisation and exact 2-D virial")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    for run, style in ((controls[0], "s--"), (runs[0], "o-")):
        ax.plot(run["trace"]["t"], run["trace"]["p_core"], style, lw=1, ms=2.5,
                label=rf"$\delta_s\times${run['delta_scale']:g}")
    ax.axhline(DP_ANALYTIC, color="k", ls=":", lw=1.2,
               label=rf"$\sigma/R$={DP_ANALYTIC:.0f} Pa")
    ax.set_xlabel(r"$t/t_\sigma$")
    ax.set_ylabel(r"$p_{\rm core}$ [Pa]")
    ax.set_title("(d) relaxation history")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / "laplace_pressure_closure.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [{"name": path.name,
             "caption": "Free-surface surface-delta closure: pressure profile, "
                        "jump error against the 15 % criterion, exact surface-delta "
                        "normalisation and half-disk virial, relaxation history."}]


if __name__ == "__main__":
    main()