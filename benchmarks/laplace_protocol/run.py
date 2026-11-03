"""Young-Laplace pressure jump: measurement protocol and density-summation bias.

Two defects made the end-to-end Laplace criterion unreadable in earlier rounds:

1. **Measurement band.** ``benchmarks/csf_consistent`` reported
   ``Delta p = p(r < 0.5 R) - p(0.80 R < r < 0.95 R)``. For a static drop the
   Laplace pressure is *uniform inside the liquid*, so that second band is
   inside the drop: what it measures is the smeared CSF force ramp, not the
   jump across the interface. The physically defined jump is
   ``p_inside - p_outside``; in vacuum with no particles outside, the exterior
   pressure is the Tait value of ``rho = 0``, i.e. exactly 0 Pa.
2. **Density-summation lattice bias.** The discrete sum
   ``rho_i = sum_j m_j W_ij`` on a hexagonal lattice returns
   ``rho0 * S(h/dx)`` with ``S(1.3) = 1.00588`` (+0.59 %). The ``gamma = 7``
   Tait law amplifies that into ``p = 2395 Pa`` on a resting, uncompressed
   lattice -- 6.6x the 360 Pa the same drop must hold. The initial state is
   therefore *not* a Young-Laplace equilibrium; the drop relaxes a 2.4 kPa
   offset, and the reading becomes a snapshot of that relaxation. ``S`` is a
   property of the lattice and kernel only, so the summation can be calibrated
   by it (``Config.density_normalization = "lattice"``).

This harness measures both effects on one 2-D drop and reports, per case, the
jump under the historical and the corrected definition, at its nominal radius
``R`` and at the radius the drop actually has (``R_surf``, from the 90th
percentile particle radius, ``r_90 = sqrt(0.9) R`` for a uniform disc).

Usage::

    PYTHONPATH=src python benchmarks/laplace_protocol/run.py            # full
    PYTHONPATH=src python benchmarks/laplace_protocol/run.py --quick    # 1 case
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    lattice_kernel_sum,
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RADIUS = 0.02
SIGMA = 7.2
RHO0 = 1000.0
SOUND_SPEED = 20.0
# r_90 / R of a uniform disc: the 90th percentile of r for a density
# proportional to r is sqrt(0.9). Used to turn the percentile radius into the
# radius of the material surface.
R90_OVER_R = math.sqrt(0.9)


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")
hexagonal_droplet = _law.hexagonal_droplet


def band_mean(pressure: torch.Tensor, r_over_R: torch.Tensor, lo: float, hi: float):
    mask = (r_over_R > lo) & (r_over_R < hi)
    if not bool(mask.any()):
        return float("nan"), 0
    return float(pressure[mask].mean()), int(mask.sum())


def surface_radius(pos: torch.Tensor) -> float:
    """Material radius of the drop from the 90th percentile particle radius."""
    r = torch.linalg.vector_norm(pos, dim=1)
    r_sorted = torch.sort(r).values
    return float(r_sorted[int(0.9 * r_sorted.numel())]) / R90_OVER_R


def initial_state(radius=RADIUS, dx=0.002, normalization="lattice", viscosity=0.05):
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=SOUND_SPEED, gamma=7.0,
                 gravity=0.0, viscosity=viscosity, surface_tension_coefficient=SIGMA,
                 csf_scheme="consistent", csf_curvature="fit_quadratic",
                 density_normalization=normalization)
    h = cfg.h_factor * cfg.dx
    pos, mass = hexagonal_droplet(radius, dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    if normalization == "lattice":
        rho = rho / lattice_kernel_sum(h, cfg.dx)
    return cfg, h, pos, mass, vel_like(pos), rho


def vel_like(pos: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(pos)


def radial_profile(pos: torch.Tensor, pressure: torch.Tensor, radius: float,
                   nbins: int = 14, hi: float = 1.15):
    """Mean pressure in equal-width radial bins, ``r`` in units of ``R``."""
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    rows = []
    edges = [hi * k / nbins for k in range(nbins + 1)]
    for lo, up in zip(edges[:-1], edges[1:]):
        mean, count = band_mean(pressure, r, lo, up)
        if count:
            rows.append({"r_over_R": round(0.5 * (lo + up), 4),
                         "p_Pa": mean, "count": count})
    return rows


def relax(radius=RADIUS, dx=0.002, normalization="lattice", viscosity=0.05,
          t_over_t_sigma=2.6, sample_every=0.1, quiet=False):
    """Relax a 2-D drop and record the two pressure-jump definitions."""
    cfg, h, pos, mass, vel, rho = initial_state(radius, dx, normalization, viscosity)
    pressure = tait_pressure(rho, cfg)
    dt = float(stable_dt(cfg))
    steps = int(round(t_over_t_sigma * math.sqrt(RHO0 * radius ** 3 / SIGMA) / dt))
    v_ref = math.sqrt(SIGMA / (RHO0 * radius))
    dp_ref = SIGMA / radius
    r0 = torch.linalg.vector_norm(pos, dim=1) / radius
    p_in_0, _ = band_mean(pressure, r0, 0.0, 0.5)
    record_every = max(1, int(round(sample_every * steps / t_over_t_sigma)))

    history = {"t_over_t_sigma": [], "p_inside": [], "p_legacy_band": [],
               "p_outside": [], "delta_p_corrected": [], "delta_p_legacy": [],
               "r_surf_over_R": [], "spurious": []}
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % record_every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1) / radius
            p_in, _ = band_mean(pressure, r, 0.0, 0.5)
            p_leg, n_leg = band_mean(pressure, r, 0.80, 0.95)
            p_out, n_out = band_mean(pressure, r, 0.98, 1.10)
            history["t_over_t_sigma"].append(round((step + 1) * dt
                                                   / math.sqrt(RHO0 * radius ** 3 / SIGMA), 4))
            history["p_inside"].append(p_in)
            history["p_legacy_band"].append(p_leg)
            history["p_outside"].append(p_out)
            history["delta_p_corrected"].append(p_in - (0.0 if math.isnan(p_out) else p_out))
            history["delta_p_legacy"].append(p_in - p_leg)
            history["r_surf_over_R"].append(surface_radius(pos) / radius)
            history["spurious"].append(float(torch.linalg.vector_norm(vel, dim=1).max())
                                       / v_ref)

    r = torch.linalg.vector_norm(pos, dim=1) / radius
    p_in, _ = band_mean(pressure, r, 0.0, 0.5)
    p_leg, n_leg = band_mean(pressure, r, 0.80, 0.95)
    p_out, n_out = band_mean(pressure, r, 0.98, 1.10)
    p_out_used = 0.0 if math.isnan(p_out) else p_out
    r_surf = surface_radius(pos) / radius
    dp_corr = p_in - p_out_used
    dp_leg = p_in - p_leg

    def rel(value, ref):
        return (value - ref) / ref * 100.0

    # Late-window average (last 30 % of the trace) and its drift.
    n_hist = len(history["t_over_t_sigma"])
    late = slice(int(0.7 * n_hist), n_hist)
    t_late = history["t_over_t_sigma"][late]
    dp_late = history["delta_p_corrected"][late]
    dp_mean = sum(dp_late) / len(dp_late)
    dp_std = (sum((x - dp_mean) ** 2 for x in dp_late) / len(dp_late)) ** 0.5
    slope = 0.0
    if len(t_late) > 1:
        t_mean = sum(t_late) / len(t_late)
        num = sum((t - t_mean) * (d - dp_mean) for t, d in zip(t_late, dp_late))
        den = sum((t - t_mean) ** 2 for t in t_late)
        slope = num / den if den > 0 else 0.0

    case = {
        "density_normalization": normalization,
        "dx_m": dx,
        "h_over_dx": cfg.h_factor,
        "n_particles": int(pos.shape[0]),
        "dt_s": dt,
        "steps": steps,
        "t_end_s": steps * dt,
        "t_over_t_sigma_end": history["t_over_t_sigma"][-1],
        "viscosity_m2_per_s": viscosity,
        "initial": {"p_inside_Pa": p_in_0},
        "final": {
            "p_inside_Pa": p_in,
            "p_legacy_band_Pa": p_leg,
            "p_outside_Pa": p_out_used,
            "n_legacy_band": n_leg,
            "n_outside": n_out,
            "delta_p_corrected_Pa": dp_corr,
            "rel_error_corrected_pct": rel(dp_corr, dp_ref),
            "delta_p_legacy_Pa": dp_leg,
            "rel_error_legacy_pct": rel(dp_leg, dp_ref),
            "r_surf_over_R": r_surf,
            "delta_p_corrected_vs_sigma_over_r_surf_pct": rel(dp_corr, SIGMA / (radius * r_surf)),
            "spurious_vmax_over_vref": history["spurious"][-1],
        },
        "late_window": {
            "t_over_t_sigma_span": [t_late[0], t_late[-1]],
            "delta_p_corrected_mean_Pa": dp_mean,
            "delta_p_corrected_std_Pa": dp_std,
            "rel_error_mean_pct": rel(dp_mean, dp_ref),
            "drift_Pa_per_t_sigma": slope,
        },
        "history": history,
    }
    if not quiet:
        print(f"{normalization:10s} dx={dx:.4f} n={case['n_particles']:4d} "
              f"t/t_sig={case['t_over_t_sigma_end']:.2f} | "
              f"dp_corr={dp_corr:7.1f} Pa ({case['final']['rel_error_corrected_pct']:+6.1f}%) "
              f"dp_legacy={dp_leg:7.1f} Pa ({case['final']['rel_error_legacy_pct']:+6.1f}%) "
              f"R_surf/R={r_surf:.3f} p_in(0)={p_in_0:7.1f}", flush=True)
    return case, history, pos, pressure, radial_profile(pos, pressure, radius)


def lattice_bias_table():
    """Spurious Tait pressure of a resting lattice vs h/dx."""
    rows = []
    dx = 0.002
    for h_factor in (1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0):
        cfg = Config(dx=dx, h_factor=h_factor, rho0=RHO0, sound_speed=SOUND_SPEED,
                     gamma=7.0, gravity=0.0)
        s = lattice_kernel_sum(h_factor * dx, dx)
        p = float(tait_pressure(torch.tensor([RHO0 * s], dtype=torch.float64), cfg)[0])
        bulk = RHO0 * SOUND_SPEED ** 2 / 7.0
        p_unclamped = bulk * (s ** 7 - 1.0)
        rows.append({"h_over_dx": h_factor, "rho_sum_over_rho0": s,
                     "p_offset_Pa": p, "p_offset_unclamped_Pa": p_unclamped,
                     "sign": "compressive" if p_unclamped >= 0 else "tensile (clamped to 0)",
                     "rel_error_vs_laplace_pct":
                     (p - SIGMA / RADIUS) / (SIGMA / RADIUS) * 100.0})
    return rows


def make_figures(cases, profiles, bias, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dp_ref = SIGMA / RADIUS
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    ax = axes[0, 0]
    ax.plot([r["h_over_dx"] for r in bias], [r["p_offset_Pa"] for r in bias],
            "o-", color="C3", label="spurious pressure of resting lattice")
    ax.axhline(dp_ref, color="k", ls="--", label=r"Young-Laplace $\sigma/R$ = 360 Pa")
    ax.set_xlabel("h / dx")
    ax.set_ylabel("Tait pressure [Pa]")
    ax.set_title("(a) density-summation lattice bias vs smoothing length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for case, style in zip(cases, _styles(len(cases))):
        label = f"{case['density_normalization']}, dx={case['dx_m']*1e3:.1f} mm"
        ax.plot(case["history"]["t_over_t_sigma"],
                case["history"]["delta_p_corrected"], linestyle=style, label=label)
    ax.axhline(dp_ref, color="k", ls="--", lw=1, label=r"$\sigma/R$")
    ax.set_xlabel(r"$t / t_\sigma$")
    ax.set_ylabel(r"$\Delta p = p_{in} - p_{out}$ [Pa]")
    ax.set_title("(b) corrected jump: inside vs exterior")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for case, style in zip(cases, _styles(len(cases))):
        label = f"{case['density_normalization']}, dx={case['dx_m']*1e3:.1f} mm"
        ax.plot(case["history"]["t_over_t_sigma"],
                case["history"]["delta_p_legacy"], linestyle=style,
                label=label + " (legacy band)")
    ax.axhline(dp_ref, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$t / t_\sigma$")
    ax.set_ylabel(r"$\Delta p = p_{in} - p(0.80\text{–}0.95R)$ [Pa]")
    ax.set_title("(c) historical band definition understates the jump")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    labels = []
    corrected, legacy = [], []
    for case in cases:
        labels.append(f"{case['density_normalization']}\n{case['dx_m']*1e3:.1f} mm")
        corrected.append(case["final"]["delta_p_corrected_Pa"])
        legacy.append(case["final"]["delta_p_legacy_Pa"])
    x = torch.arange(len(cases)).numpy()
    ax.bar(x - 0.2, corrected, 0.4, label="corrected (in vs out)", color="C0")
    ax.bar(x + 0.2, legacy, 0.4, label="legacy band", color="C1")
    ax.axhline(dp_ref, color="k", ls="--", label=r"$\sigma/R$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$\Delta p$ [Pa]")
    ax.set_title("(d) final reading, both definitions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = out_dir / "laplace_protocol.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _styles(count: int):
    pool = ["-", (0, (4, 2)), ":", (0, (1, 1))]
    return pool[:count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="single case, short run")
    ap.add_argument("--t-over-t-sigma", type=float, default=2.6)
    args = ap.parse_args()

    if args.quick:
        cases_def = [("lattice", 0.003)]
        t_end = 1.2
    else:
        cases_def = [("summation", 0.003), ("lattice", 0.003),
                     ("summation", 0.002), ("lattice", 0.002)]
        t_end = args.t_over_t_sigma

    cases, profiles = [], []
    for normalization, dx in cases_def:
        case, _hist, _pos, _press, prof = relax(dx=dx, normalization=normalization,
                                                t_over_t_sigma=t_end)
        case["radial_profile"] = prof
        cases.append(case)
        profiles.append(prof)

    bias = lattice_bias_table()
    for row in bias:
        print(f"h/dx={row['h_over_dx']:.1f} rho_sum/rho0={row['rho_sum_over_rho0']:.6f} "
              f"p_offset={row['p_offset_Pa']:9.1f} Pa (raw {row['p_offset_unclamped_Pa']:+9.1f}) "
              f"{row['sign']}")

    dp_ref = SIGMA / RADIUS
    by_key = {(c["density_normalization"], c["dx_m"]): c for c in cases}
    report = {
        "benchmark": "laplace_pressure_protocol",
        "reference": {"delta_p_Pa": dp_ref, "radius_m": RADIUS,
                      "sigma_N_per_m": SIGMA, "rho0": RHO0,
                      "sound_speed_m_per_s": SOUND_SPEED},
        "definition": {
            "corrected": "delta p = mean p(r < 0.5 R) - mean p(0.98 R < r < 1.10 R); "
                         "the exterior band is empty for a free drop, in which case "
                         "the exterior pressure is the Tait value of rho = 0, i.e. 0 Pa",
            "legacy": "delta p = mean p(r < 0.5 R) - mean p(0.80 R < r < 0.95 R); "
                      "the second band lies INSIDE the liquid, so it measures the "
                      "smeared CSF force ramp instead of the jump across the interface",
            "r_surf": "material radius from the 90th percentile particle radius / sqrt(0.9)",
        },
        "lattice_bias": bias,
        "cases": cases,
        "verdicts": {
            f"{norm}_dx{dx_m*1e3:.1f}mm": {
                "corrected_within_15pct": abs(c["final"]["rel_error_corrected_pct"]) < 15.0,
                "corrected_within_3pct": abs(c["final"]["rel_error_corrected_pct"]) < 3.0,
                "legacy_within_15pct": abs(c["final"]["rel_error_legacy_pct"]) < 15.0,
                "late_window_corrected_within_15pct":
                    abs(c["late_window"]["rel_error_mean_pct"]) < 15.0,
            }
            for (norm, dx_m), c in by_key.items()
        },
    }
    (HERE / "validation_report.json").write_text(json.dumps(report, indent=2))
    print("figure ->", make_figures(cases, profiles, bias))
    print("report ->", HERE / "validation_report.json")


if __name__ == "__main__":
    main()