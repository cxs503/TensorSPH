"""Young-Laplace equilibrium protocol for the single-phase CSF drop (m2545).

Why this harness exists
-----------------------
``benchmarks/csf_consistent`` scored the Young-Laplace jump of the CSF model at
``Delta p = -75.6 %`` and ``benchmarks/laplace_law`` at ``-77 % .. -118 %``.  Two
questions were left open there:

1. is the number a property of the *model*, or of the *protocol*?
2. the standing P0 item of ``backlog_update_m2174.md`` ("add the Adami 2012
   background pressure ``p0 = sigma/R`` to the Tait EOS, otherwise the drop
   cannot build up over-pressure") -- does it actually change the measured jump?

This harness answers both with three experiments on the same drop:

A. **initial-condition control**.  ``lattice_droplet`` (as shipped) puts
   ``rho/rho0 = 1.00588`` in the core, i.e. ``p = 2395 Pa`` at ``t = 0`` -- 6.6x
   the Laplace pressure that is supposed to be measured.  The drop therefore
   starts with an *expansion* wave.  ``density_consistent_droplet`` rescales the
   particle masses by ``rho0 / <rho>`` measured on the same lattice, so the core
   starts at exactly ``p = 0`` and surface tension is the only driving force.

B. **estimator control**.  ``p_interface_band`` (the shipped estimator, band
   ``0.80 R < r < 0.95 R``) lives inside the kernel-deficiency layer, where
   density summation returns ``rho/rho0 ~ 0.91`` and the Tait pressure is clamped
   to zero.  The Young-Laplace law with a vacuum outside is instead the
   *absolute* interior pressure: ``Delta p = <p>_(r<0.7R) - 0``.

C. **p0-bias negative control**.  ``p = p0 + B[(rho/rho0)^gamma - 1]`` with
   ``p0 = sigma/R`` (Adami et al. 2012) is switched on through a monkeypatched
   EOS and the same measurement is repeated.

Every run also records the *discrete delta-function normalisation*
``N_delta = sum_i delta_s,i A_i / (2 pi R)`` (``A_i = m_i/rho0``), which is the
mesh-consistency condition of Brackbill et al. (1992): a correct CSF delta
function must satisfy ``N_delta = 1``.

Usage::

    PYTHONPATH=src python benchmarks/laplace_equilibrium/run.py            # all
    PYTHONPATH=src python benchmarks/laplace_equilibrium/run.py --quick    # dx=0.003 only
    PYTHONPATH=src python benchmarks/laplace_equilibrium/run.py --figure   # redraw only
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / "equilibrium_report.json"

torch.set_num_threads(4)

SIGMA = 7.2          # N/m, water
RHO0 = 1000.0        # kg/m^3
C0 = 20.0            # m/s, Mach ~ 0.03
GAMMA = 7.0
RADIUS = 0.02        # m
T_END = 0.0936       # s = 2.8 t_sigma, the window of the shipped benchmark
T_SIGMA = math.sqrt(RHO0 * RADIUS ** 3 / SIGMA)


# --------------------------------------------------------------------------- #
# imports of the shipped code (kept lazy so --figure works without torch work)
# --------------------------------------------------------------------------- #
def _csf_run_module():
    spec = importlib.util.spec_from_file_location(
        "csf_consistent_run", ROOT / "benchmarks" / "csf_consistent" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_config(dx: float):
    from tensorsph.droplet_impact import Config
    return Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=C0, gamma=GAMMA,
                  gravity=0.0, viscosity=0.01, surface_tension_coefficient=SIGMA,
                  csf_scheme="consistent")


# --------------------------------------------------------------------------- #
# initial conditions
# --------------------------------------------------------------------------- #
def lattice_droplet(dx: float, consistent: bool = False):
    """Hexagonal drop; ``consistent=True`` rescales the masses (see module doc)."""
    from tensorsph.droplet_impact import density_from_neighbor_pairs
    from tensorsph.neighbor_search import cell_linked_pairs

    mod = _csf_run_module()
    pos, mass = mod.hexagonal_droplet(RADIUS, dx, RHO0)
    h = 1.3 * dx
    r = torch.linalg.vector_norm(pos, dim=1)
    core = r < 0.5 * RADIUS
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho_lat = density_from_neighbor_pairs(pos, mass, h, pairs)
    info = {"rho_core_over_rho0_raw": float(rho_lat[core].mean() / RHO0)}
    if consistent:
        scale = RHO0 / float(rho_lat[core].mean())
        mass = mass * scale
        rho_lat = density_from_neighbor_pairs(pos, mass, h, pairs)
        info["mass_scale"] = scale
    info["rho_core_over_rho0"] = float(rho_lat[core].mean() / RHO0)
    info["p_core_initial_Pa"] = float(_tait(rho_lat, 0.0)[core].mean())
    return pos, mass, info


# --------------------------------------------------------------------------- #
# equation of state (with optional background pressure, Adami et al. 2012)
# --------------------------------------------------------------------------- #
def _tait(rho: torch.Tensor, p0: float) -> torch.Tensor:
    bulk = RHO0 * C0 ** 2 / GAMMA
    ratio = (rho / RHO0).clamp_min(0.01)
    return (p0 + bulk * (ratio.pow(GAMMA) - 1.0)).clamp_min(0.0)


# --------------------------------------------------------------------------- #
# relaxation
# --------------------------------------------------------------------------- #
def relax(dx: float, consistent: bool = True, p0: float = 0.0,
          steps: int | None = None, output_every: int = 50,
          tau_damp: float | None = None) -> dict:
    """March the drop and record the interior pressure history.

    ``p0`` != 0 patches ``tensorsph.droplet_impact.tait_pressure`` for the run
    (restored afterwards), which is the Adami background-pressure fix exactly as
    the backlog asks for it.
    """
    import tensorsph.droplet_impact as di
    from tensorsph.csf_consistent import csf_fields
    from tensorsph.droplet_impact import stable_dt
    from tensorsph.neighbor_search import cell_linked_pairs

    cfg = make_config(dx)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    if steps is None:
        steps = int(round(T_END / dt))
    t_sigma = T_SIGMA
    v_ref = math.sqrt(SIGMA / (RHO0 * RADIUS))
    area = dx * dx * math.sqrt(3.0) / 2.0

    pos, mass, init = lattice_droplet(dx, consistent=consistent)
    vel = torch.zeros_like(pos)

    original = di.tait_pressure
    if p0 != 0.0:
        di.tait_pressure = lambda rho, cfg_: _tait(rho, p0)
    try:
        hist = []
        for step in range(steps + 1):
            pairs = cell_linked_pairs(pos, 2.0 * h)
            rho, pressure, acc = di.rhs(pos, vel, mass, cfg, pairs=pairs,
                                        wall=None, dt=dt)
            vel = vel + dt * acc
            pos = pos + dt * vel
            if tau_damp:
                vel = vel * math.exp(-dt / tau_damp)
            if step % output_every == 0 or step == steps:
                state = csf_fields(pos, mass, pairs, h, RHO0)
                hist.append(_sample(pos, rho, pressure, vel, state, area,
                                    (step + 1) * dt, v_ref, p0))
    finally:
        di.tait_pressure = original

    rel = (hist[-1]["p_interior_Pa"] - SIGMA / RADIUS) / (SIGMA / RADIUS) * 100.0
    # quasi-equilibrium window: the last 25 % of the run
    tail = [x for x in hist if x["t"] > 0.75 * hist[-1]["t"]]
    p_tail = sum(x["p_interior_Pa"] for x in tail) / len(tail)
    return {
        "protocol": {"dx_m": dx, "h_m": h, "dt_s": dt, "steps": steps,
                     "t_end_s": hist[-1]["t"], "t_end_over_t_sigma": hist[-1]["t"] / t_sigma,
                     "consistent_init": consistent, "p0_Pa": p0,
                     "tau_damp_s": tau_damp, "n_particles": int(pos.shape[0])},
        "initial_condition": init,
        "reference": {"delta_p_Pa": SIGMA / RADIUS, "t_sigma_s": t_sigma},
        "final": hist[-1],
        "quasi_static": {"p_interior_mean_Pa": p_tail,
                         "rel_error_pct": (p_tail - SIGMA / RADIUS) / (SIGMA / RADIUS) * 100.0,
                         "n_delta_mean": sum(x["n_delta"] for x in tail) / len(tail)},
        "history": hist,
    }


def _sample(pos, rho, pressure, vel, state, area, t, v_ref, p0=0.0) -> dict:
    r = torch.linalg.vector_norm(pos, dim=1)
    interior = r < 0.7 * RADIUS
    shell = (r > 0.80 * RADIUS) & (r < 0.95 * RADIUS)
    bands = []
    for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85),
                   (0.85, 0.95), (0.95, 1.05)]:
        sel = (r > lo * RADIUS) & (r <= hi * RADIUS)
        if int(sel.sum()):
            bands.append({"r_lo": lo, "r_hi": hi, "n": int(sel.sum()),
                          "p_Pa": float(pressure[sel].mean()),
                          "rho_over_rho0": float((rho[sel] / RHO0).mean())})
    n_delta = float((state.delta_s * area).sum()) / (2 * math.pi * RADIUS)
    # fraction of particles whose raw Tait pressure is negative, i.e. the ones
    # the zero-clamp of ``tait_pressure`` masks (they carry NO pressure at all).
    bulk = RHO0 * C0 ** 2 / GAMMA
    raw = p0 + bulk * ((rho / RHO0).clamp_min(0.01).pow(GAMMA) - 1.0)
    clamped = raw < 0.0
    frac_clamped = float(clamped.float().mean())
    frac_clamped_shell = float(clamped[shell].float().mean()) if int(shell.sum()) else float("nan")
    return {
        "t": t, "p_interior_Pa": float(pressure[interior].mean()),
        "p_shell_band_Pa": float(pressure[shell].mean()),
        "rho_interior_over_rho0": float((rho[interior] / RHO0).mean()),
        "rho_shell_band_over_rho0": float((rho[shell] / RHO0).mean()),
        "delta_p_vacuum_ref_Pa": float(pressure[interior].mean()),
        "delta_p_shellband_estimator_Pa": float(pressure[interior].mean()
                                                - pressure[shell].mean()),
        "rmax_m": float(r.max()),
        "vmax_over_v_ref": float(torch.linalg.vector_norm(vel, dim=1).max()) / v_ref,
        "n_delta": n_delta,
        "frac_clamped": frac_clamped,
        "frac_clamped_shell_band": frac_clamped_shell,
        "bands": bands,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(report: dict, out_png: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = out_png or HERE / "laplace_equilibrium.png"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ref = SIGMA / RADIUS

    ax = axes[0][0]
    for key, label, style in (("shipped_init", "shipped init (p0_lat = 2395 Pa)", "-"),
                              ("consistent_init", "density-consistent init", "-")):
        run = report["experiments"].get(key)
        if not run:
            continue
        t = [x["t"] for x in run["history"]]
        p = [x["p_interior_Pa"] for x in run["history"]]
        ax.plot(t, p, style, label=label, lw=1.6)
    ax.axhline(ref, color="k", ls="--", lw=1.2, label=r"$\sigma/R = 360$ Pa")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"interior pressure $\langle p\rangle_{r<0.7R}$ [Pa]")
    ax.set_title("(a) initial-condition control, dx = 2 mm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot([x["t"] for x in report["experiments"]["consistent_init"]["history"]],
            [x["n_delta"] for x in report["experiments"]["consistent_init"]["history"]],
            "o-", ms=3, lw=1.4, label=r"$N_\delta=\sum_i\delta_{s,i}A_i/2\pi R$")
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="Brackbill normalisation = 1")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$N_\delta$ [-]")
    ax.set_title(r"(b) discrete $\delta_s$ normalisation decays (interface diffusion)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    hist = report["experiments"]["consistent_init"]["history"]
    nd = [x["n_delta"] for x in hist]
    pr = [x["p_interior_Pa"] / ref for x in hist]
    ax.plot(nd, pr, "o-", ms=3, lw=1.4, label="dx = 2 mm")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label=r"$\Delta p/(\sigma/R) = N_\delta$")
    ax.set_xlabel(r"$N_\delta$ [-]")
    ax.set_ylabel(r"$\Delta p/(\sigma/R)$ [-]")
    ax.set_title("(c) jump tracks the delta-function normalisation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    hist = report["experiments"]["consistent_init"]["history"]
    for row in [hist[0], hist[len(hist) // 3], hist[-1]]:
        rr = [0.5 * (b["r_lo"] + b["r_hi"]) for b in row["bands"]]
        ax.plot(rr, [b["p_Pa"] for b in row["bands"]], "o-", ms=3,
                label=f"t = {row['t']:.3f} s")
    ax.axhline(ref, color="k", ls="--", lw=1.2, label=r"$\sigma/R$")
    ax.set_xlabel("r/R [-]")
    ax.set_ylabel("p [Pa]")
    ax.set_title("(d) radial pressure profile (shaded: deficiency layer r/R>0.85)")
    ax.axvspan(0.85, 1.05, color="0.85", alpha=0.6)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("TensorSPH m2545 - Young-Laplace equilibrium protocol (CSF consistent)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    if args.figure:
        report = json.loads(REPORT.read_text())
        print("figure ->", make_figures(report))
        return

    report = {
        "benchmark": "csf_laplace_equilibrium_protocol",
        "mission": "m2545",
        "reference": {"delta_p_Pa": SIGMA / RADIUS, "kappa_1_per_m": 1.0 / RADIUS,
                      "t_sigma_s": T_SIGMA, "sigma_N_per_m": SIGMA,
                      "rho0": RHO0, "radius_m": RADIUS, "sound_speed": C0},
        "experiments": {},
        "resolution_scan": [],
        "p0_control": [],
    }

    # A: initial-condition control (one resolution)
    report["experiments"]["shipped_init"] = relax(0.002, consistent=False)
    report["experiments"]["consistent_init"] = relax(0.002, consistent=True)

    # B: resolution scan with the corrected protocol
    for dx in ([0.003] if args.quick else [0.003, 0.002, 0.0015]):
        run = relax(dx, consistent=True)
        report["resolution_scan"].append({
            "dx_m": dx, "n_particles": run["protocol"]["n_particles"],
            "p_interior_final_Pa": run["final"]["p_interior_Pa"],
            "rel_error_pct": (run["final"]["p_interior_Pa"] - SIGMA / RADIUS)
                             / (SIGMA / RADIUS) * 100.0,
            "n_delta_initial": run["history"][0]["n_delta"],
            "n_delta_final": run["final"]["n_delta"],
            "vmax_over_v_ref_final": run["final"]["vmax_over_v_ref"],
            "rho_interior_over_rho0_final": run["final"]["rho_interior_over_rho0"],
        })
        report.setdefault("resolution_history", {})[str(dx)] = run["history"]

    # C: p0-bias negative control
    for p0 in ([0.0] if args.quick else [0.0, SIGMA / RADIUS]):
        run = relax(0.002, consistent=True, p0=p0)
        report["p0_control"].append({
            "p0_Pa": p0,
            "p_interior_final_Pa": run["final"]["p_interior_Pa"],
            "rel_error_pct": (run["final"]["p_interior_Pa"] - SIGMA / RADIUS)
                             / (SIGMA / RADIUS) * 100.0,
            "p_center_time_series": [round(x["p_interior_Pa"], 2)
                                     for x in run["history"][::2]],
            "p_shell_band_Pa": run["final"]["p_shell_band_Pa"],
            "rho_shell_band_over_rho0": run["final"]["rho_shell_band_over_rho0"],
            "frac_clamped_final": run["final"]["frac_clamped"],
            "frac_clamped_shell_band_final": run["final"]["frac_clamped_shell_band"],
        })

    # damped quasi-static control
    report["experiments"]["damped_consistent"] = relax(0.002, consistent=True,
                                                       tau_damp=5e-4)

    REPORT.write_text(json.dumps(report, indent=1))
    print("report ->", REPORT)

    shipped = report["experiments"]["shipped_init"]
    cons = report["experiments"]["consistent_init"]
    print(f"shipped init : p_core(t=0) = {shipped['initial_condition']['p_core_initial_Pa']:.0f} Pa, "
          f"final dp = {shipped['final']['p_interior_Pa']:.1f} Pa "
          f"({(shipped['final']['p_interior_Pa'] - SIGMA / RADIUS) / (SIGMA / RADIUS) * 100:.1f} %), "
          f"shell-band estimator = {shipped['final']['delta_p_shellband_estimator_Pa']:.1f} Pa")
    print(f"consistent   : p_core(t=0) = {cons['initial_condition']['p_core_initial_Pa']:.1f} Pa, "
          f"final dp = {cons['final']['p_interior_Pa']:.1f} Pa "
          f"({(cons['final']['p_interior_Pa'] - SIGMA / RADIUS) / (SIGMA / RADIUS) * 100:.1f} %)")
    for row in report["p0_control"]:
        print(f"p0 = {row['p0_Pa']:6.1f} Pa -> dp = {row['p_interior_final_Pa']:.1f} Pa "
              f"({row['rel_error_pct']:.1f} %)")
    for row in report["resolution_scan"]:
        print(f"dx = {row['dx_m']}: dp = {row['p_interior_final_Pa']:.1f} Pa "
              f"({row['rel_error_pct']:.1f} %), N_delta {row['n_delta_initial']:.3f} -> "
              f"{row['n_delta_final']:.3f}")
    print("figure ->", make_figures(report))


if __name__ == "__main__":
    main()