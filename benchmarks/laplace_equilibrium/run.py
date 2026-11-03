"""End-to-end Young-Laplace law on a static drop: is the measurement falsifiable?

The repository has one end-to-end check of the surface-tension model: relax a
2-D drop of radius ``R`` and compare its interior pressure with
``Delta p = sigma/R`` (``benchmarks/csf_consistent/run.py :: relax``).  It
reports -76 % ... -107 % and the previous mission filed the cause as "the Tait
compression needed for 360 Pa sits on the summation-density noise floor".
This script tests that explanation rather than restating it, and it finds
**two measurement defects on top of the physical question**:

D1 -- gauge.  ``relax`` measures ``p_core - p_band`` with the band at
     ``0.80 R < r < 0.95 R``, i.e. *inside* the drop.  The CSF force is a
     volume force localised in the interface shell, so in equilibrium the
     interior pressure is a plateau and the jump lives between the outermost
     particle layer and the void, where the Tait clamp gives ``p = 0`` by
     construction.  Subtracting two interior points measures the residual
     plateau tilt, not the Laplace jump.

D2 -- initial condition.  ``hexagonal_droplet`` places particles on a hex
     lattice of spacing ``dx`` with ``mass = rho0 * dx * dy``, which assumes the
     *discrete* partition of unity of that lattice is 1.  With ``h/dx = 1.3`` it
     is not: the core summation density is 1.00588 ``rho0``, so the initial Tait
     pressure is ``B[(1.00588)^7 - 1] = 2395 Pa`` -- **6.7x the 360 Pa signal** --
     and it sits behind a free surface, so it is released immediately as an
     acoustic transient.  Any sample of ``p_core`` is then a small difference of
     two large numbers dominated by that transient.

The fix tested here is a **volume-consistent initialisation**: set the particle
mass (equivalently the particle volume, the lattice spacing is untouched) so
that the discrete partition of unity equals 1, i.e.
``m* = rho0 dx dy / (sum_j V_j W_ij)``, which is exact in one step because the
summation density is linear in the mass.  Nothing else changes -- same kernel,
same ``dx``, same ``h/dx = 1.3``, same CSF assembly, same EOS, same limiter.
With that initial state:

* ``sigma = 0`` becomes a genuine quiescence control (``p_core`` must stay ~0
  and the drop must not move), which the baseline initialisation cannot pass;
* every subsequent interior compression is caused by the CSF force alone, so
  ``p_core`` is a direct measurement of ``sigma/R``;
* the law is additionally checked in *slope* form, ``d p_core / d sigma = 1/R``,
  which is immune to any constant offset that survives.

Usage::

    PYTHONPATH=src python benchmarks/laplace_equilibrium/run.py
    PYTHONPATH=src python benchmarks/laplace_equilibrium/run.py --steps 2000 --quick-isolation
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
    rhs,
    stable_dt,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

H_FACTOR = 1.3          # kernel ratio the rest of the repo is calibrated for
BASELINE_MASS_SCALE = 1.0


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_law = _load_sibling("laplace_law_run", "laplace_law/run.py")
hexagonal_droplet = _law.hexagonal_droplet


# --------------------------------------------------------------------------- #
# D2: the discrete partition of unity of the initial lattice
# --------------------------------------------------------------------------- #
def core_density_ratio(radius: float, dx: float, rho0: float = 1000.0,
                       h_factor: float = H_FACTOR,
                       mass_scale: float = 1.0) -> float:
    """Summation density of the *initial* drop, averaged over ``r < 0.5 R``.

    This is the quantity ``hexagonal_droplet`` implicitly assumes is 1.0.  It is
    a property of the discretisation (the lattice and ``h/dx``), not of the
    physics, so it is measured here and used to calibrate the initial state.
    """
    h = h_factor * dx
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass * mass_scale, h, pairs)
    r = torch.linalg.vector_norm(pos, dim=1)
    core = r < 0.5 * radius
    return float(rho[core].mean() / rho0)


def calibrate_mass_scale(radius: float, dx: float, rho0: float = 1000.0,
                         h_factor: float = H_FACTOR) -> tuple[float, dict]:
    """Mass scale that makes the discrete partition of unity exactly 1.

    The summation density is linear in the particle mass, so the calibration is
    one division -- and it is then *verified* on the full initial state rather
    than trusted.
    """
    ratio = core_density_ratio(radius, dx, rho0, h_factor)
    scale = 1.0 / ratio
    achieved = core_density_ratio(radius, dx, rho0, h_factor, scale)
    return scale, {
        "baseline_core_density_ratio": ratio,
        "mass_scale": scale,
        "achieved_core_density_ratio": achieved,
        "linearity": "summation density is linear in mass, so scale = 1/ratio",
    }


def initial_core_pressure(radius: float, dx: float, rho0: float = 1000.0,
                          sound_speed: float = 20.0, gamma: float = 7.0,
                          h_factor: float = H_FACTOR,
                          mass_scale: float = 1.0) -> float:
    """Tait pressure of the *unforced* initial state (the D2 artefact)."""
    ratio = core_density_ratio(radius, dx, rho0, h_factor, mass_scale)
    return float(rho0 * sound_speed ** 2 / gamma * (ratio ** gamma - 1.0))


# --------------------------------------------------------------------------- #
# the relaxation run
# --------------------------------------------------------------------------- #
def radial_pressure_profile(pos: torch.Tensor, pressure: torch.Tensor,
                            radius: float, bins: int = 10) -> list[dict]:
    """Mean pressure per ``r/R`` bin plus the particle count in each bin."""
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    edges = torch.linspace(0.0, 1.05, bins + 1, dtype=pos.dtype)
    out = []
    for k in range(bins):
        lo, hi = float(edges[k]), float(edges[k + 1])
        sel = (r >= lo) & (r < hi)
        cnt = int(sel.sum())
        out.append({
            "r_over_R": 0.5 * (lo + hi),
            "p_Pa": float(pressure[sel].mean()) if cnt else float("nan"),
            "n": cnt,
        })
    return out


def run_case(dx: float = 0.002, sigma: float = 7.2,
             mass_scale: float = BASELINE_MASS_SCALE, steps: int = 6000,
             viscosity: float = 0.01, sound_speed: float = 20.0,
             radius: float = 0.02, rho0: float = 1000.0, gamma: float = 7.0,
             h_factor: float = H_FACTOR, output_every: int = 50,
             label: str = "") -> dict:
    """Relax a drop and record the interior-pressure history.

    Gauge (the D1 fix): ``p_core`` is the mean over ``r < 0.5 R``, the interior
    plateau.  The ambient pressure is zero by the free-surface clamp, so the
    Laplace jump is ``p_core - 0``.  ``p_shell`` (``0.97 R < r <= 1.02 R``) is
    reported only as a diagnostic of how much of the jump is already resolved
    inside the outermost layer; it is *not* used as the ambient value.
    """
    cfg = Config(dx=dx, h_factor=h_factor, rho0=rho0, sound_speed=sound_speed,
                 gamma=gamma, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent")
    h = h_factor * dx
    dt = float(stable_dt(cfg))
    pos, mass = hexagonal_droplet(radius, dx, rho0)
    mass = mass * mass_scale
    vel = torch.zeros_like(pos)

    rho_init = density_from_neighbor_pairs(pos, mass, h,
                                           cell_linked_pairs(pos, 2.0 * h))
    pressure = tait_pressure(rho_init, cfg)
    core0 = torch.linalg.vector_norm(pos, dim=1) < 0.5 * radius
    p_start = float(pressure[core0].mean())

    v_ref = math.sqrt(sigma / (rho0 * radius)) if sigma > 0 else 1.0
    hist = {"t": [], "p_core": [], "p_shell": [], "rho_core": [],
            "r_mean": [], "r95": [], "vmax": [], "vmax_over_vref": []}
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % output_every == 0 or step == steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            hist["t"].append((step + 1) * dt)
            hist["p_core"].append(float(pressure[r < 0.5 * radius].mean()))
            hist["p_shell"].append(float(pressure[(r > 0.97 * radius)
                                                  & (r <= 1.02 * radius)].mean()))
            hist["rho_core"].append(float(rho[r < 0.5 * radius].mean() / rho0))
            hist["r_mean"].append(float(r.mean()))
            hist["r95"].append(float(torch.quantile(r, 0.95)))
            vmax = float(torch.linalg.vector_norm(vel, dim=1).max())
            hist["vmax"].append(vmax)
            hist["vmax_over_vref"].append(vmax / v_ref)

    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma) if sigma > 0 else float("nan")
    capillary_period = (2.0 * math.pi / math.sqrt(6.0 * sigma / (rho0 * radius ** 3))
                        if sigma > 0 else float("nan"))
    return {
        "label": label,
        "setup": {
            "radius_m": radius, "dx_m": dx, "h_over_dx": h_factor, "h_m": h,
            "sigma_N_per_m": sigma, "rho0": rho0, "sound_speed": sound_speed,
            "gamma": gamma, "viscosity": viscosity, "steps": steps, "dt_s": dt,
            "mass_scale": mass_scale, "t_end_s": steps * dt,
            "t_over_t_sigma": (steps * dt / t_sigma) if sigma > 0 else None,
            "capillary_period_s": capillary_period if sigma > 0 else None,
        },
        "reference": {
            "delta_p_Pa": sigma / radius,
            "compression_required": (sigma / radius) / (rho0 * sound_speed ** 2 / gamma),
            "kappa_1_per_m": 1.0 / radius,
        },
        "initial_state": {
            "core_density_ratio": core_density_ratio(radius, dx, rho0, h_factor,
                                                     mass_scale),
            "core_pressure_Pa": p_start,
        },
        "history": hist,
        "profile": radial_pressure_profile(pos, pressure, radius),
    }


# --------------------------------------------------------------------------- #
# statistics and gates
# --------------------------------------------------------------------------- #
def _linfit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Least-squares ``y = a + b x`` (torch has no ``polyfit``)."""
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    if denom == 0.0:
        raise ValueError("degenerate abscissa in least-squares fit")
    b = float((dx * (y - ym)).sum()) / denom
    return b, float(ym) - b * float(xm)


def plateau(series: list[float], times: list[float], window: float = 0.5) -> dict:
    """Time mean / fluctuation / range over the last ``window`` fraction.

    The drop is an inviscid oscillator once the transient has decayed, so a
    single final sample is not a measurement.  A measurement counts as stable
    when the relative fluctuation is at most 15 % -- the same shape of criterion
    as the hydrostatic benchmark's pressure-fluctuation gate.
    """
    if len(series) != len(times) or len(series) < 4:
        raise ValueError("need >=4 paired samples")
    k = max(2, int(round(window * len(series))))
    vals = torch.tensor(series[-k:], dtype=torch.float64)
    tt = torch.tensor(times[-k:], dtype=torch.float64)
    mean = float(vals.mean())
    std = float(vals.std(unbiased=True))
    span = float(tt[-1] - tt[0])
    slope = _linfit(tt, vals)[0] if k > 2 else float("nan")
    drift = slope * span if math.isfinite(slope) else float("nan")
    rel_drift = drift / abs(mean) if mean else float("inf")
    return {
        "window_fraction": window, "samples": k,
        "mean": mean, "std": std,
        "sem": std / math.sqrt(k),
        "min": float(vals.min()), "max": float(vals.max()),
        "relative_fluctuation": std / abs(mean) if mean else float("inf"),
        "drift_per_window": drift,
        "relative_drift": rel_drift,
        # a window is a measurement only if it neither rings nor drifts:
        # <=15 % fluctuation (the hydrostatic benchmark's gate) and <=10 %
        # residual trend over the window.
        "stable": bool(mean != 0.0 and std / abs(mean) <= 0.15
                       and abs(rel_drift) <= 0.10),
    }


def laplace_verdict(measured: float, theory: float) -> dict:
    """Relative error of a pressure measurement against ``sigma/R``."""
    rel = (measured - theory) / theory if theory else float("nan")
    return {
        "measured_Pa": measured, "theory_Pa": theory,
        "rel_error_pct": 100.0 * rel,
        "within_3pct": abs(rel) < 0.03, "within_15pct": abs(rel) < 0.15,
    }


def slope_estimator(sigmas: list[float], pressures: list[float]) -> dict:
    """Least-squares ``d p / d sigma`` -- the offset-free form of the law.

    For a static 2-D drop ``p_core = sigma/R + offset``; a constant offset (a
    residual density bias, for instance) drops out of the slope, which is why
    the slope is reported next to the absolute verdict rather than instead of it.
    """
    x = torch.tensor(sigmas, dtype=torch.float64)
    y = torch.tensor(pressures, dtype=torch.float64)
    if x.numel() < 2 or float(x.std()) == 0.0:
        raise ValueError("need at least two distinct sigma values")
    slope, offset = _linfit(x, y)
    return {"slope_Pa_per_N_per_m": slope, "offset_Pa": offset,
            "n": int(x.numel())}


def slope_verdict(slope: float, radius: float) -> dict:
    theory = 1.0 / radius
    rel = (slope - theory) / theory
    return {
        "measured_1_per_m": slope, "theory_1_per_m": theory,
        "rel_error_pct": 100.0 * rel,
        "within_3pct": abs(rel) < 0.03, "within_15pct": abs(rel) < 0.15,
    }


def summarise(case: dict) -> dict:
    """Fold a raw case into gates: plateau statistics + Laplace verdict."""
    hist = case["history"]
    stats = plateau(hist["p_core"], hist["t"])
    verdict = laplace_verdict(stats["mean"], case["reference"]["delta_p_Pa"])
    sp = plateau(hist["vmax_over_vref"], hist["t"])
    return {
        "label": case["label"],
        "setup": case["setup"], "reference": case["reference"],
        "initial_state": case["initial_state"],
        "plateau": stats,
        "laplace": verdict,
        "shell_diagnostic": plateau(hist["p_shell"], hist["t"]),
        "spurious": {
            "vmax_over_vref_plateau": sp["mean"],
            "vmax_over_vref_final": hist["vmax_over_vref"][-1],
            "passes_adami_0p1": sp["mean"] < 0.1,
        },
        "quiescence": {
            "sigma_zero": case["setup"]["sigma_N_per_m"] == 0.0,
            "max_abs_p_core_Pa": max(abs(v) for v in hist["p_core"]),
            "max_vmax": max(hist["vmax"]),
            "passes": bool(max(abs(v) for v in hist["p_core"]) < 1.0
                           and max(hist["vmax"]) < 1e-6),
        },
        "density": {
            "core_density_ratio_plateau": plateau(hist["rho_core"], hist["t"])["mean"],
            "compression_required": case["reference"]["compression_required"],
        },
        "profile": case["profile"],
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(summaries: list[dict], raw: list[dict], out_dir: Path = HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    color_base, color_cal, color_ctrl = "#c0392b", "#1f77b4", "#7f8c8d"

    def _color(label: str) -> str:
        if label.startswith("baseline"):
            return color_base
        if "control" in label:
            return color_ctrl
        return color_cal

    theory = raw[0]["reference"]["delta_p_Pa"]

    # (a) p_core history
    ax = axes[0, 0]
    for case in raw:
        ax.plot(case["history"]["t"], case["history"]["p_core"], lw=1.1,
                color=_color(case["label"]), alpha=0.9, label=case["label"])
    ax.axhline(theory, color="k", ls="--", lw=1.0, label=r"$\sigma/R$")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$p_{core}$ [Pa]")
    ax.set_title("(a) interior pressure history")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    # (b) radial pressure profile
    ax = axes[0, 1]
    for case, summ in zip(raw, summaries):
        pts = [(b["r_over_R"], b["p_Pa"]) for b in summ["profile"] if b["n"]]
        if not pts:
            continue
        ax.plot([q[0] for q in pts], [q[1] for q in pts],
                "--" if case["label"].startswith("baseline") else "-",
                marker="o", ms=3, lw=1.2, color=_color(case["label"]),
                label=case["label"])
    ax.axhline(theory, color="k", ls=":", lw=1.0, label=r"$\sigma/R$")
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel("p [Pa]")
    ax.set_title(r"(b) radial pressure profile at $t_{end}$")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    # (c) absolute verdict
    ax = axes[1, 0]
    xpos = list(range(len(summaries)))
    ax.bar(xpos, [s["plateau"]["mean"] for s in summaries],
           yerr=[s["plateau"]["std"] for s in summaries], capsize=4,
           color=[_color(s["label"]) for s in summaries])
    ax.axhline(theory, color="k", ls="--", lw=1.0, label=r"$\sigma/R$")
    ax.axhspan(0.85 * theory, 1.15 * theory, color="green", alpha=0.10,
               label="15 % gate")
    ax.set_xticks(xpos)
    ax.set_xticklabels([s["label"] for s in summaries], rotation=20,
                       ha="right", fontsize=6)
    ax.set_ylabel(r"time-mean $p_{core}$ [Pa]")
    ax.set_title("(c) absolute Young-Laplace verdict")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # (d) sigma scaling
    ax = axes[1, 1]
    pts = [(s["setup"]["sigma_N_per_m"], s["plateau"]["mean"],
            s["plateau"]["std"]) for s in summaries
           if s["setup"]["sigma_N_per_m"] > 0 and not s["label"].startswith("baseline")]
    if pts:
        ax.errorbar([q[0] for q in pts], [q[1] for q in pts],
                    yerr=[q[2] for q in pts], fmt="o", capsize=3,
                    color=color_cal, label="measured (volume-consistent)")
        smax = max(q[0] for q in pts)
        ax.plot([0, smax * 1.15], [0, (smax * 1.15) / raw[0]["setup"]["radius_m"]],
                "k--", lw=1.0, label=r"$\sigma/R$")
        ax.set_xlim(0, smax * 1.15)
        ax.set_ylim(0, max(q[1] + q[2] for q in pts) * 1.3)
    ax.set_xlabel(r"$\sigma$ [N/m]")
    ax.set_ylabel(r"$p_{core}$ [Pa]")
    ax.set_title(r"(d) $\sigma$ scaling (slope $= 1/R$)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "laplace_equilibrium.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--quick-isolation", action="store_true",
                    help="baseline + calibrated + sigma=0 control only")
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args(argv)

    scale, cal = calibrate_mass_scale(a.radius, a.dx)
    p_lat_base = initial_core_pressure(a.radius, a.dx)
    p_lat_cal = initial_core_pressure(a.radius, a.dx, mass_scale=scale)
    print(f"baseline      : core density ratio {cal['baseline_core_density_ratio']:.6f}"
          f" -> initial p = {p_lat_base:.1f} Pa")
    print(f"volume-consistent: mass_scale {scale:.6f} -> ratio "
          f"{cal['achieved_core_density_ratio']:.6f}, initial p = {p_lat_cal:.1f} Pa")

    # the sigma = 0 controls only have to demonstrate (in)stability, so they
    # run for a quarter of the relaxation window
    cases = [("baseline sigma=7.2", BASELINE_MASS_SCALE, 7.2, a.steps),
             ("volume-consistent sigma=7.2", scale, 7.2, a.steps),
             ("volume-consistent sigma=0 quiescence control", scale, 0.0,
              max(500, a.steps // 4))]
    if not a.quick_isolation:
        cases += [("volume-consistent sigma=14.4", scale, 14.4, a.steps),
                  ("volume-consistent sigma=21.6", scale, 21.6, a.steps),
                  ("baseline sigma=0 control", BASELINE_MASS_SCALE, 0.0,
                   max(500, a.steps // 4))]

    raw, summaries = [], []
    for label, ms, sigma, steps in cases:
        case = run_case(dx=a.dx, sigma=sigma, mass_scale=ms, steps=steps,
                        radius=a.radius, label=label)
        s = summarise(case)
        raw.append(case)
        summaries.append(s)
        extra = (f" | quiescent={s['quiescence']['passes']}"
                 if s["quiescence"]["sigma_zero"] else "")
        print(f"{label:46s} p_core={s['plateau']['mean']:8.2f} +- "
              f"{s['plateau']['std']:7.2f} Pa | rel={s['laplace']['rel_error_pct']:+8.2f}%"
              f" | <15%: {s['laplace']['within_15pct']!s:5s} | "
              f"stable={s['plateau']['stable']!s:5s} | "
              f"vmax/vref={s['spurious']['vmax_over_vref_final']:.3f}{extra}")

    pts = [(s["setup"]["sigma_N_per_m"], s["plateau"]["mean"]) for s in summaries
           if s["setup"]["sigma_N_per_m"] > 0 and not s["label"].startswith("baseline")]
    scaling = None
    if len(pts) >= 2:
        fit = slope_estimator([q[0] for q in pts], [q[1] for q in pts])
        scaling = {"fit": fit, "verdict": slope_verdict(
            fit["slope_Pa_per_N_per_m"], a.radius)}
        print(f"slope dp/dsigma = {fit['slope_Pa_per_N_per_m']:.2f} 1/m vs 1/R = "
              f"{1.0 / a.radius:.2f} 1/m ({scaling['verdict']['rel_error_pct']:+.2f}%)")

    report = {
        "benchmark": "laplace_equilibrium",
        "headline": ("Young-Laplace law on a static 2-D drop measured with an "
                     "ambient-zero gauge and a volume-consistent initial state"),
        "initialisation": {
            "h_over_dx": H_FACTOR,
            "baseline_mass_scale": BASELINE_MASS_SCALE,
            **cal,
            "baseline_initial_core_pressure_Pa": p_lat_base,
            "calibrated_initial_core_pressure_Pa": p_lat_cal,
        },
        "scaling": scaling,
        "cases": summaries,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "raw_cases.json").write_text(json.dumps(raw))
    (a.out / "validation_report.json").write_text(json.dumps(report, indent=2))
    png = make_figures(summaries, raw, a.out)
    print("figure ->", png)
    print("report ->", a.out / "validation_report.json")
    return report


if __name__ == "__main__":
    main()