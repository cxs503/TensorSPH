"""Corrected Young-Laplace pressure-jump verification of the CSF surface model.

Why this benchmark exists
-------------------------
``benchmarks/laplace_law`` (m2174) and ``benchmarks/csf_consistent`` (m2588)
score the end-to-end Laplace jump as

    Delta p = p(r < 0.5 R) - p(0.80 R < r < 0.95 R)                  (LEGACY)

and recorded a -75.6 % error for the consistent CSF scheme, which was booked as
"end-to-end pressure jump not met; cause = EOS resolution floor, not the force".
That definition is geometrically wrong for a **single-phase** drop: the region
``0.80 R < r < 0.95 R`` is still *inside* the liquid, and the 2-D Young-Laplace
law says the pressure is **uniform** throughout the interior,

    p(r) = sigma / R      for r < R,        p = 0 (vacuum)  for r > R,

so the legacy estimator measures the *radial non-uniformity of the interior
pressure*, i.e. the numerical noise of the pressure field, and **not** the jump.
The jump has to be taken against the ambient, which in this single-phase model
is the free-surface condition ``p = 0`` imposed by the clamped Tait law.

This harness therefore

1. measures the jump with the corrected definition
   ``Delta p = p(r < 0.7 R_eff) - p_amb``, where ``p_amb = 0`` is the free
   surface condition and ``R_eff`` is the equivalent radius from the particle
   area (so a shrinking / expanding drop is scored against ``sigma / R_eff``);
2. reports the legacy definition side by side, so the definition error is
   quantified rather than argued;
3. time-averages the pressure over a whole capillary period.  The drop is an
   extremely under-damped oscillator (viscous time ``rho R^2 / mu = 40 s`` is
   ~10^3 capillary times), so a single snapshot at ``t = 0.7 t_sigma`` -- the
   window used by the earlier verdicts -- is not an equilibrium measurement;
4. tests two competing explanations for the residual:
   * *EOS resolution floor*: if true, the error must grow as ``c0^2`` (larger
     ``c0`` -> smaller required compression -> larger lumped error).  Verified
     by a sound-speed ladder ``c0 in {20, 10, 5}`` at fixed ``dx``.
   * *under-dense free surface*: the density-summation deficit over the two
     surface layers (``rho/rho0`` down to ~0.90) clamps the pressure to zero in
     exactly the shell where the jump must form.  Verified by switching on the
     delta-SPH density diffusion of Antuono et al. (2010), which is already
     implemented as ``cfg.density_diffusion`` and default-off.
5. reports the time-averaged radial pressure profile, which localises the
   residual: a flat interior plateau below ``sigma / R_eff`` plus a
   pressure-free shell of ~2 layers at the free surface.

References
    Young (1805); Laplace (1806).  Brackbill, Kothe & Zemach (1992) JCP 100:335.
    Antuono, Colagrossi & Marrone (2010) Comput. Phys. Commun. 183:2570.
    Adami, Hu & Adams (2012) JCP 231:7057 (spurious currents).

Usage::

    PYTHONPATH=src python benchmarks/laplace_pressure_jump/run.py --quick
    PYTHONPATH=src python benchmarks/laplace_pressure_jump/run.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    stable_dt,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: area of one particle of a triangular (hexagonal) lattice with spacing dx
HEX_AREA = math.sqrt(3.0) / 2.0


def _load_sibling(filename: str):
    path = HERE.parent / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law/run.py")
hexagonal_droplet = _law.hexagonal_droplet


# --------------------------------------------------------------------------- #
# one relaxation run + the two pressure-jump readings
# --------------------------------------------------------------------------- #
def run_case(radius=0.02, dx=0.002, sigma=7.2, rho0=1000.0, sound_speed=20.0,
             viscosity=0.01, density_diffusion=0.0, scheme="consistent",
             t_end_in_capillary_times=2.2):
    """Relax a drop and read the Laplace jump with both definitions."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme=scheme,
                 density_diffusion=density_diffusion)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    t_end = t_end_in_capillary_times * t_sigma
    steps = max(1, int(t_end / dt))

    pos, mass = hexagonal_droplet(radius, dx, rho0)
    vel = torch.zeros_like(pos)
    rho = density_from_neighbor_pairs(pos, mass, h,
                                      cell_linked_pairs(pos, 2.0 * h))

    # time-average the pressure and density over the last full capillary period
    p_sum = torch.zeros(len(pos))
    rho_sum = torch.zeros(len(pos))
    n_avg = 0
    vmax = 0.0
    v_ref = math.sqrt(sigma / (rho0 * radius))
    t0 = time.time()
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if (steps - step) * dt <= t_sigma:
            p_sum += pressure
            rho_sum += rho
            n_avg += 1
            vmax = max(vmax, float(torch.linalg.vector_norm(vel, dim=1).max()))
    wall = time.time() - t0

    p_bar = p_sum / max(n_avg, 1)
    rho_bar = rho_sum / max(n_avg, 1)

    # equivalent radius: the drop conserves particle number, not area
    area = len(pos) * HEX_AREA * dx * dx
    r_eff = math.sqrt(area / math.pi)
    dp_theory = sigma / r_eff

    r = torch.linalg.vector_norm(pos, dim=1)
    core = r < 0.5 * r_eff
    interior = r < 0.7 * r_eff
    inner_band = (r > 0.80 * r_eff) & (r < 0.95 * r_eff)
    shell = (r > 0.85 * r_eff) & (r < r_eff)

    p_core = float(p_bar[core].mean())
    p_interior = float(p_bar[interior].mean())
    p_inner_band = float(p_bar[inner_band].mean())

    # LEGACY definition (m2174 / m2588): core minus a band that is *inside* the
    # drop -- kept only to quantify the definition error.
    dp_legacy = p_core - p_inner_band
    # CORRECTED definition: the ambient pressure of a single-phase drop in
    # vacuum is the free-surface condition p = 0.
    dp_corrected = p_interior - 0.0

    def _err(value):
        return 100.0 * (value / dp_theory - 1.0) if dp_theory else float("nan")

    # time-averaged radial pressure profile: shows (i) the interior plateau,
    # (ii) the pressure-free shell that the clamped Tait law creates, and
    # (iii) how far the plateau sits below sigma/R_eff.
    edges = [i * 0.1 for i in range(0, 13)]
    profile = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo * r_eff) & (r < hi * r_eff)
        if int(m.sum()) == 0:
            continue
        profile.append({
            "r_over_r_eff_lo": lo, "r_over_r_eff_hi": hi,
            "n": int(m.sum()),
            "p_mean_Pa": float(p_bar[m].mean()),
            "p_std_Pa": float(p_bar[m].std()),
            "rho_over_rho0_mean": float((rho_bar[m] / rho0).mean()),
        })

    return {
        "setup": {
            "radius_m": radius, "dx_m": dx, "h_m": h, "sigma_N_per_m": sigma,
            "rho0": rho0, "sound_speed": sound_speed, "viscosity": viscosity,
            "gamma": cfg.gamma, "scheme": scheme,
            "density_diffusion": density_diffusion,
            "n_particles": int(len(pos)), "dt_s": dt, "steps": steps,
            "t_end_s": steps * dt, "t_over_t_sigma": steps * dt / t_sigma,
            "averaging_window": "last full capillary period",
            "averaging_samples": n_avg, "wall_time_s": wall,
        },
        "reference": {
            "r_eff_m": r_eff, "dp_theory_Pa": dp_theory,
            "note": "2-D Young-Laplace dp = sigma/R_eff, R_eff from particle area",
        },
        "legacy_definition": {
            "p_core_r_lt_0.5R_Pa": p_core,
            "p_inner_band_0.80-0.95R_Pa": p_inner_band,
            "delta_p_Pa": dp_legacy,
            "rel_error_pct": _err(dp_legacy),
            "inner_band_over_core": p_inner_band / p_core if p_core else float("nan"),
            "valid_as_jump": False,
        },
        "corrected_definition": {
            "p_interior_r_lt_0.7R_Pa": p_interior,
            "p_ambient_Pa": 0.0,
            "delta_p_Pa": dp_corrected,
            "rel_error_pct": _err(dp_corrected),
            "within_3pct": abs(_err(dp_corrected)) < 3.0,
            "within_15pct": abs(_err(dp_corrected)) < 15.0,
            "interior_std_over_mean_pct": 100.0 * float(p_bar[interior].std())
            / max(abs(p_interior), 1e-30),
        },
        "free_surface": {
            "rho_over_rho0_shell_min": float((rho_bar[shell] / rho0).min())
            if int(shell.sum()) else float("nan"),
            "rho_over_rho0_shell_mean": float((rho_bar[shell] / rho0).mean())
            if int(shell.sum()) else float("nan"),
            "rho_over_rho0_interior_mean": float((rho_bar[interior] / rho0).mean()),
            "compression_required": dp_theory / (rho0 * sound_speed ** 2),
        },
        "achieved_over_theory": dp_corrected / dp_theory,
        "pressure_profile": profile,
        "spurious_current": {
            "v_ref_m_per_s": v_ref,
            "vmax_over_vref": vmax / v_ref,
            "passes_adami_0p1": vmax / v_ref < 0.1,
        },
    }


# --------------------------------------------------------------------------- #
# the experiment matrix
# --------------------------------------------------------------------------- #
def experiment(quick=False):
    """Sound-speed ladder x density-diffusion, plus a resolution spot check."""
    t0 = time.time()
    rows = []
    # (a) sound-speed ladder at fixed dx: tests the "EOS resolution floor"
    #     explanation.  The required compression is  dp / (rho0 c0^2).
    for c0 in (20.0, 10.0, 5.0):
        rows.append(run_case(sound_speed=c0))
    # (b) delta-SPH density diffusion: tests the "under-dense free surface"
    #     explanation.  Same c0 and dx, only the diffusion coefficient changes.
    for delta in (0.05, 0.1, 0.2):
        rows.append(run_case(sound_speed=20.0, density_diffusion=delta))
    if not quick:
        # (c) resolution spot check at the baseline settings
        for dx in (0.003, 0.0015):
            rows.append(run_case(dx=dx))
    return {
        "benchmark": "laplace_pressure_jump_corrected",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_wall_time_s": time.time() - t0,
        "reference_law": "2-D Young-Laplace: p_in - p_out = sigma / R, uniform inside",
        "acceptance": {"platform_standard_pct": 15.0, "target_pct": 3.0},
        "cases": rows,
    }


def make_figure(report, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = report["cases"]
    baseline = [c for c in cases if c["setup"]["sound_speed"] == 20.0
                and c["setup"]["density_diffusion"] == 0.0
                and abs(c["setup"]["dx_m"] - 0.002) < 1e-9][0]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    ax = axes[0]
    labels = ["legacy def.\n(core - inner band)", "corrected def.\n(interior - ambient)"]
    vals = [baseline["legacy_definition"]["rel_error_pct"],
            baseline["corrected_definition"]["rel_error_pct"]]
    bars = ax.bar(labels, vals, color=["C3", "C0"])
    ax.axhline(15, color="r", ls="--", lw=1, label="15 % gate")
    ax.axhline(-15, color="r", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:+.1f} %", (bar.get_x() + bar.get_width() / 2, v),
                    ha="center", va="bottom" if v > 0 else "top",
                    fontsize=9, fontweight="bold")
    ax.set_ylabel("Laplace-jump error [%]")
    ax.set_title("(a) definition of the jump decides the verdict")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    c0s = [c["setup"]["sound_speed"] for c in cases
           if c["setup"]["density_diffusion"] == 0.0]
    errs = [c["corrected_definition"]["rel_error_pct"] for c in cases
            if c["setup"]["density_diffusion"] == 0.0]
    ax.plot(c0s, errs, "o-", label="corrected jump error")
    ax.axhline(15, color="r", ls="--", label="15 % gate")
    ax.axhline(-15, color="r", ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("sound speed c0 [m/s]")
    ax.set_ylabel("error [%]")
    ax.set_title("(b) sound-speed ladder (EOS-floor test)")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    deltas = [0.0] + [c["setup"]["density_diffusion"] for c in cases
                      if c["setup"]["density_diffusion"] > 0.0]
    derrs = [baseline["corrected_definition"]["rel_error_pct"]] + [
        c["corrected_definition"]["rel_error_pct"] for c in cases
        if c["setup"]["density_diffusion"] > 0.0]
    ax.plot(deltas, derrs, "s-", color="C2", label="corrected jump error")
    ax.axhline(15, color="r", ls="--", label="15 % gate")
    ax.axhline(-15, color="r", ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\delta$-SPH density diffusion $\delta$ [-]")
    ax.set_ylabel("error [%]")
    ax.set_title("(c) free-surface density deficit test")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    for c, label in ((baseline, "baseline (c0=20, delta=0)"),):
        prof = c["pressure_profile"]
        centres = [0.5 * (b["r_over_r_eff_lo"] + b["r_over_r_eff_hi"]) for b in prof]
        ax.plot(centres, [b["p_mean_Pa"] for b in prof], "o-", label="measured $\\bar p(r)$")
        ax.fill_between(centres,
                        [b["p_mean_Pa"] - b["p_std_Pa"] for b in prof],
                        [b["p_mean_Pa"] + b["p_std_Pa"] for b in prof],
                        alpha=0.2)
    ax.axhline(baseline["reference"]["dp_theory_Pa"], color="k", ls="--",
               label=r"analytic $\sigma/R_{\rm eff}$")
    ax.axvline(1.0, color="gray", ls=":")
    ax.set_xlabel(r"$r / R_{\rm eff}$")
    ax.set_ylabel("time-averaged pressure [Pa]")
    ax.set_title("(d) radial pressure profile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = Path(out_dir) / "laplace_pressure_jump.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the resolution spot check")
    ap.add_argument("--out", default=str(HERE / "validation_report.json"))
    args = ap.parse_args()
    report = experiment(quick=args.quick)
    Path(args.out).write_text(json.dumps(report, indent=1))
    fig = make_figure(report)
    print(f"wrote {args.out}")
    print(f"wrote {fig}")
    print(f"wall time {report['total_wall_time_s']:.1f} s")
    for c in report["cases"]:
        s, corr, leg = c["setup"], c["corrected_definition"], c["legacy_definition"]
        print(f"  dx={s['dx_m']:.4f} c0={s['sound_speed']:5.1f} "
              f"delta={s['density_diffusion']:4.2f} N={s['n_particles']:5d} "
              f"legacy={leg['rel_error_pct']:+7.1f}%  "
              f"corrected={corr['rel_error_pct']:+7.1f}%  "
              f"rho_shell={c['free_surface']['rho_over_rho0_shell_min']:.3f}  "
              f"achieved={c['achieved_over_theory']:.2f} sigma/R")


if __name__ == "__main__":
    main()