"""Does the free-surface pressure closure close the Young-Laplace gate?

Context
-------
Four rounds of work have failed the same end-to-end check, and the last one
(``backlog_update_m2971.md``) concluded that the only remaining gate is *C3*:
``tait_pressure`` clamps to ``p >= 0``, so every particle within ``2h`` of the
free surface sits at exactly zero, the interface cannot hold ``sigma/R``, and
the interior settles at a fraction of the analytic jump.  It handed the next
mission one P0: implement the shell-pressure extrapolation of Adami, Hu & Adams
(2012) Sec. 3.3 for *fluid* free-surface particles (the wall particles already
had it).

This benchmark answers the question the P0 could not: **does the closure
actually move the number?**  It is an attribution experiment, not a tuning
exercise:

  arm A  ``free_surface_pressure="clamp"``  - the state before this mission.
  arm B  ``free_surface_pressure="shell"``  - the P0 closure, 8 lines of
         ``rhs`` plus ``tensorsph/fs_pressure.py``.

  grid   ``{clamp, shell} x {dx = 2.0 mm, 1.5 mm} x {c0 = 20, 5 m/s}``

  sigma ladder  ``sigma in {3.6, 7.2, 14.4} N/m`` at ``dx = 2 mm, c0 = 20``.
         If the residual is a *pressure-closure* defect, doubling ``sigma``
         doubles ``Delta p`` and the relative error is unchanged; if the
         discretisation saturates (force is under-resolved), the relative error
         falls with ``sigma``.  This separates "wrong boundary condition" from
         "wrong surface force magnitude".

Reference.  Tait is normalised so that ``p(rho0) = 0``, and the drop sits in
vacuum, so the physical jump is ``p_core - 0 = sigma/R = 360 Pa``.  The reading
is the absolute pressure, time-averaged over the last 40 % of the window
(``absolute_time_average_tail`` in ``laplace_pressure_floor``) - the band
definitions of the earlier rounds are contaminated by the density deficit they
were measuring.

Usage
-----
    PYTHONPATH=src python benchmarks/fs_pressure/run.py --quick
    PYTHONPATH=src python benchmarks/fs_pressure/run.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, rhs, stable_dt, tait_pressure
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.fs_pressure import free_surface_mask

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load_floor_module():
    """Reuse the m2971 initialisation and measurement, do not re-implement it."""
    path = ROOT / "benchmarks" / "laplace_pressure_floor" / "run.py"
    spec = importlib.util.spec_from_file_location("laplace_pressure_floor_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FLOOR = _load_floor_module()
RADIUS = FLOOR.RADIUS
SIGMA = FLOOR.SIGMA
RHO0 = FLOOR.RHO0
GAMMA = FLOOR.GAMMA
H_FACTOR = FLOOR.H_FACTOR
VISCOSITY = FLOOR.VISCOSITY
T_SIGMA = FLOOR.T_SIGMA
DP_ANALYTIC = FLOOR.DP_ANALYTIC


def relax(dx: float, c0: float, closure: str, sigma: float = SIGMA,
          t_end_over_t_sigma: float = 1.5, max_steps: int = 4000,
          sample_every: int = 50) -> dict:
    """Relax a calibrated drop and read ``Delta p`` as the absolute core pressure."""
    cfg = Config(dx=dx, h_factor=H_FACTOR, rho0=RHO0, sound_speed=c0,
                 gamma=GAMMA, gravity=0.0, viscosity=VISCOSITY,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 csf_curvature="fit_quadratic",
                 free_surface_pressure=closure)
    h = cfg.h_factor * cfg.dx
    t_sigma = math.sqrt(RHO0 * RADIUS ** 3 / sigma)
    dp_ref = sigma / RADIUS
    dt = float(stable_dt(cfg))
    t_end = t_end_over_t_sigma * t_sigma
    steps = min(max_steps, int(math.ceil(t_end / dt)))
    dt = t_end / steps

    pos, mass, h, _, bias, _ = FLOOR.calibrated_droplet(dx)
    vel = torch.zeros_like(pos)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = FLOOR.density_from_neighbor_pairs(pos, mass, h, pairs)
    v_ref = math.sqrt(sigma / (RHO0 * RADIUS))

    trace = {"t": [], "delta_p": [], "vmax_over_vref": [], "rho_core_rel": [],
             "n_free": [], "mean_shell_lift": []}
    pressure = tait_pressure(rho, cfg)
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % sample_every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
            trace["t"].append((step + 1) * dt / t_sigma)
            trace["delta_p"].append(float(pressure[r < 0.5].mean()))
            trace["vmax_over_vref"].append(
                float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref)
            trace["rho_core_rel"].append(float(rho[r < 0.5].mean() / RHO0 - 1.0))
            free = free_surface_mask(rho, RHO0)
            trace["n_free"].append(int(free.sum()))
            trace["mean_shell_lift"].append(
                float((pressure[free]).mean()) if bool(free.any()) else 0.0)

    n_tail = max(1, int(0.4 * len(trace["delta_p"])))
    tail = sum(trace["delta_p"][-n_tail:]) / n_tail
    r = torch.linalg.vector_norm(pos, dim=1) / RADIUS
    err = (tail - dp_ref) / dp_ref

    def score(value):
        e = (value - dp_ref) / dp_ref
        return {"delta_p_Pa": value, "rel_error_pct": e * 100.0,
                "within_3pct": abs(e) < 0.03, "within_15pct": abs(e) < 0.15}

    return {
        "dx_m": dx, "sound_speed": c0, "sigma_N_per_m": sigma,
        "closure": closure, "n_particles": int(pos.shape[0]),
        "lattice_bias": bias, "dt_s": dt, "steps": steps,
        "t_end_s": t_end, "t_over_t_sigma": t_end / t_sigma,
        "reference": {"delta_p_Pa": dp_ref, "t_sigma_s": t_sigma},
        "absolute_time_average_tail": score(tail),
        "final_absolute": score(float(trace["delta_p"][-1])),
        "spurious_current": {
            "vmax_over_vref_final": trace["vmax_over_vref"][-1],
            "passes_adami_0p1": trace["vmax_over_vref"][-1] < 0.1},
        "free_surface": {"n_particles": trace["n_free"][-1],
                         "fraction": trace["n_free"][-1] / int(pos.shape[0])},
        "density": {
            "rho_core_over_rho0": float(rho[r < 0.5].mean() / RHO0),
            "compression_required": (1.0 + dp_ref / (RHO0 * c0 ** 2 / GAMMA))
            ** (1.0 / GAMMA) - 1.0,
        },
        "finite": bool(torch.isfinite(pos).all() and torch.isfinite(vel).all()),
        "pressure_profile": FLOOR.radial_profile(pos, pressure),
        "trace": trace,
    }


def figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = report["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    for case in cases:
        if case["sound_speed"] != 20.0 or case["sigma_N_per_m"] != SIGMA:
            continue
        prof = case["pressure_profile"]
        xs = [p["r_over_R_mid"] for p in prof]
        ys = [p["mean"] for p in prof]
        es = [p["std"] for p in prof]
        label = f"{case['closure']}, dx={case['dx_m']*1e3:.1f} mm"
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, lw=1.2, capsize=2,
                    label=label)
    ax.axhline(DP_ANALYTIC, color="k", ls="--", lw=1.0,
               label=r"analytic $\sigma/R=360$ Pa")
    ax.set_xlabel("r / R")
    ax.set_ylabel("pressure [Pa]")
    ax.set_title("radial pressure profile (c0 = 20 m/s)")
    ax.legend(fontsize=7)

    ax = axes[1]
    labels, errors, colors = [], [], []
    for case in cases:
        if case["sigma_N_per_m"] != SIGMA:
            continue
        labels.append(f"{case['closure'][:5]}\ndx={case['dx_m']*1e3:.1f}/c0={case['sound_speed']:g}")
        errors.append(case["absolute_time_average_tail"]["rel_error_pct"])
        colors.append("#1f77b4" if case["closure"] == "clamp" else "#d62728")
    ax.bar(range(len(errors)), errors, color=colors)
    ax.set_xticks(range(len(errors)))
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    ax.axhspan(-15, 15, color="green", alpha=0.12, label="+/-15 % gate")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("relative error of $\\Delta p$ [%]")
    ax.set_title("absolute time-averaged jump vs $\\sigma/R$")
    ax.legend(fontsize=7)

    ax = axes[2]
    for closure, style in (("clamp", "o-"), ("shell", "s--")):
        pts = [(c["sigma_N_per_m"], c["absolute_time_average_tail"]["rel_error_pct"])
               for c in cases if c["sigma_N_per_m"] != SIGMA
               and c["closure"] == closure and c["sound_speed"] == 20.0]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=closure)
    ax.axhspan(-15, 15, color="green", alpha=0.12)
    ax.set_xlabel(r"$\sigma$ [N/m]")
    ax.set_ylabel("relative error [%]")
    ax.set_title("sigma ladder (dx = 2 mm, c0 = 20)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--t-end", type=float, default=1.5)
    ap.add_argument("--out", default=str(HERE / "report.json"))
    args = ap.parse_args()

    if args.quick:
        grid = [(0.002, 20.0), (0.002, 5.0)]
        sigmas = [SIGMA]
    else:
        grid = [(0.002, 20.0), (0.0015, 20.0), (0.002, 5.0), (0.0015, 5.0)]
        sigmas = [3.6, 14.4]

    cases = []
    for dx, c0 in grid:
        for closure in ("clamp", "shell"):
            case = relax(dx, c0, closure, SIGMA, args.t_end)
            cases.append(case)
            print(f"dx={dx} c0={c0:g} {closure:5s} -> "
                  f"{case['absolute_time_average_tail']['delta_p_Pa']:8.2f} Pa "
                  f"({case['absolute_time_average_tail']['rel_error_pct']:+6.1f} %) "
                  f"n={case['n_particles']} steps={case['steps']}", flush=True)

    for sigma in sigmas:
        for closure in ("clamp", "shell"):
            case = relax(0.002, 20.0, closure, sigma, args.t_end)
            cases.append(case)
            print(f"sigma={sigma:g} {closure:5s} -> "
                  f"{case['absolute_time_average_tail']['delta_p_Pa']:8.2f} Pa "
                  f"({case['absolute_time_average_tail']['rel_error_pct']:+6.1f} %)",
                  flush=True)

    def best(closure, dx=None, c0=None):
        rows = [c for c in cases if c["closure"] == closure
                and c["sigma_N_per_m"] == SIGMA
                and (dx is None or c["dx_m"] == dx)
                and (c0 is None or c["sound_speed"] == c0)]
        return min(rows, key=lambda c: abs(
            c["absolute_time_average_tail"]["rel_error_pct"]), default=None)

    report = {
        "benchmark": "fs_pressure_attribution",
        "question": "does the shell-pressure closure (Adami 2012 Sec. 3.3) close "
                    "the Young-Laplace gate identified as C3 in m2971?",
        "reference": {"delta_p_Pa": DP_ANALYTIC, "sigma_N_per_m": SIGMA,
                      "radius_m": RADIUS, "reading": "absolute core pressure, "
                      "time-averaged over the last 40 % of 1.5 t_sigma"},
        "cases": cases,
        "summary": {
            "best_clamp": best("clamp"),
            "best_shell": best("shell"),
            "gate_15pct_closed": bool(
                best("shell") is not None
                and abs(best("shell")["absolute_time_average_tail"]["rel_error_pct"]) < 15.0),
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2))
    figure(report, HERE / "fs_pressure.png")
    print("wrote", out)


if __name__ == "__main__":
    main()