"""Is the beta = 30 deg peak-pressure coefficient measurable, or is it binning?

Motivation (measured, not assumed)
----------------------------------
``run.py --stage accept`` reported ``cp_band_peak`` = 32.01 / 28.05 at
``dx = 4 / 2 mm`` against the Wagner anchor 8.4022 (rel. err. +281 % / +234 %),
while the same trajectory gives ``cp_centre_mean`` ~ 4.9 (10 % below the
analytic stagnation value 5.4414) and a wetted half-width that tracks
``C V t`` to a few percent.  A solution whose stagnation line and wetted width
are within ~10 % cannot simultaneously be 280 % wrong *and* noise free at one
abscissa, so before touching the solver the metric itself has to be probed.

The suspicion is sampling, not physics: the wall profile is binned in the
self-similar abscissa ``tau = |x| / (C V t)`` over ``[0, 1.2]`` with
``cfg.profile_bins = 120`` (``run.py`` BASE), while the wetted surface carries
only ``2 c / dx`` nodes (~20 at ``dx = 4 mm``, ``c = 40 mm``).  With more bins
than samples a bin *is* a particle, the ensemble mean over frames mixes
different particles, and the maximum over bins is a maximum over single-particle
pressure noise.  This script measures that directly: one trajectory, the wall
samples taken once per frame, re-binned at several bin counts and with a
sample-count-driven bin width.  Nothing here changes the solver, the case or a
threshold; it perturbs only the estimator.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/metric_resolution.py \
        --dx 0.002 --delta 0.0 --out artifacts/case_sph_wedge_entry_2d_metric

Writes one JSON artefact per (dx, delta, form) with the re-binned estimators,
the density excursion, and the reference values the estimators are compared to.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tensorsph.wedge_entry import (  # noqa: E402
    WedgeConfig,
    WedgeImpact,
    wagner_c_coeff,
    wagner_p_over_q,
    wagner_peak_over_q,
)

BIN_COUNTS = (12, 20, 30, 40, 60, 120)
TAU_MAX = 1.2
ENSEMBLE_FRAMES = 5


def wall_samples(sim: WedgeImpact) -> tuple[list[float], list[float], int]:
    """``(tau, cp, n_surface_nodes)`` of the wetted wall nodes, for the current state.

    Same definitions as ``WedgeImpact.diagnostics`` (tau normalised by the
    *analytic* wetted edge ``C V t``, pressure from the Adami extrapolation,
    contact = kernel support above ``cfg.contact_support``), but returned per
    node so the binning can be varied afterwards.
    """
    cfg = sim.cfg
    state = sim._evaluate(sim._last_dt)
    p_wall = state["p_wall"][: sim.n_wedge]
    den = state["den"][: sim.n_wedge]
    surf = sim.is_first
    x = (sim.wedge_local[:, 0] + sim.body_offset[0])[surf]
    p = p_wall[surf]
    contact = den[surf] > cfg.contact_support * sim.support_ref
    c_w = wagner_c_coeff(cfg.beta) * sim.V * sim.time
    if c_w <= 0.0 or not bool(contact.any()):
        return [], [], int(surf.sum().item())
    tau = (x.abs()[contact] / c_w).tolist()
    cp = (p[contact] / sim.q).tolist()
    return tau, cp, int(surf.sum().item())


def binned_profile(taus: list[float], cps: list[float], n_bins: int):
    edges = [TAU_MAX * k / n_bins for k in range(n_bins + 1)]
    sums = [0.0] * n_bins
    cnts = [0] * n_bins
    for tau, cp in zip(taus, cps):
        if not (tau == tau) or not (cp == cp):
            continue
        k = min(int(tau / TAU_MAX * n_bins), n_bins - 1)
        if k < 0:
            continue
        sums[k] += cp
        cnts[k] += 1
    centres = [0.5 * (edges[k] + edges[k + 1]) for k in range(n_bins)]
    means = [sums[k] / cnts[k] if cnts[k] else float("nan") for k in range(n_bins)]
    return centres, means, cnts


def adaptive_bins(tau_width: float):
    """Bin count for a prescribed bin width in tau (``n/2`` rounding, >= 4)."""
    return max(4, int(round(TAU_MAX / max(tau_width, 1.0e-9))))


def band_peak(centres, means, cnts, tau_hi=1.0):
    best, best_tau, best_count = float("nan"), float("nan"), 0
    for c, m, n in zip(centres, means, cnts):
        if not (m == m) or c > tau_hi:
            continue
        if not (best == best) or m > best:
            best, best_tau, best_count = m, c, n
    return best, best_tau, best_count


def value_at(centres, means, tau0, half_width):
    vals = [m for c, m in zip(centres, means) if m == m and abs(c - tau0) <= half_width]
    return statistics.mean(vals) if vals else float("nan")


def run_case(dx: float, delta: float, form: str, velocity: float,
             frames: int, out_dir: Path, ensemble_frames: int = ENSEMBLE_FRAMES) -> dict:
    cfg = WedgeConfig(beta_deg=30.0, velocity=velocity, dx=dx, frames=frames,
                      alpha_visc=0.03, sound_speed_factor=20.0,
                      max_courant=0.35, profile_bins=120,
                      density_diffusion=delta, density_diffusion_form=form)
    cfg.validate()
    t0 = time.time()
    sim = WedgeImpact(cfg)
    C = wagner_c_coeff(cfg.beta)
    t_end = cfg.resolved_t_end()
    times = [t_end * k / (frames - 1) for k in range(frames)]

    shots = []          # per frame: (tau, cp, n_surface, t, c_sph)
    series = []
    step = 0
    target = 0
    while sim.time < t_end - 1.0e-15:
        dt = min(sim.dt(), t_end - sim.time)
        if target < len(times):
            dt = min(dt, max(times[target] - sim.time, 1.0e-18))
        sim.advance(dt)
        step += 1
        if target < len(times) and sim.time >= times[target] - 1.0e-15:
            diag = sim.diagnostics()
            tau, cp, n_surf = wall_samples(sim)
            shots.append({"t": sim.time, "tau": tau, "cp": cp,
                          "n_surface": n_surf, "c_sph": diag["c"],
                          "max_rho_ratio": diag["max_rho_ratio"]})
            series.append({k: diag[k] for k in
                           ("t", "c", "p_max_footprint", "p_max_interior",
                            "p_centre", "max_rho_ratio")})
            target += 1
    wall = time.time() - t0
    q = sim.q

    # ---- estimators on the last ENSEMBLE_FRAMES frames -------------------- #
    sel = shots[-ensemble_frames:]
    mean_samples = sum(len(s["tau"]) for s in sel) / max(len(sel), 1)
    c_last = sel[-1]["c_sph"]
    dx_tau = (2.0 * dx) / (C * velocity * sel[-1]["t"])
    bins = {"counts": {}, "adaptive": {}}
    for n_bins in BIN_COUNTS:
        per_frame = [binned_profile(s["tau"], s["cp"], n_bins) for s in sel]
        centres = per_frame[-1][0]
        means = []
        for k in range(n_bins):
            vals = [pf[1][k] for pf in per_frame
                    if pf[1][k] == pf[1][k]]
            means.append(statistics.mean(vals) if vals else float("nan"))
        counts = [max(pf[2][k] for pf in per_frame) for k in range(n_bins)]
        peak, tau_at, n_at = band_peak(centres, means, counts)
        bins["counts"][str(n_bins)] = {
            "bin_width_tau": TAU_MAX / n_bins,
            "bin_width_m": TAU_MAX / n_bins * c_last,
            "samples_per_bin": round(mean_samples * TAU_MAX / 1.2 / n_bins, 2),
            "cp_band_peak": peak, "tau_at_peak": tau_at, "samples_at_peak": n_at,
            "cp_at_wagner_tau_peak": value_at(centres, means,
                                              math.sqrt(1 - 1 / C ** 2), 0.05),
            "profile_tau": centres, "profile_cp": means,
        }
    n_ad = adaptive_bins(dx_tau)
    per_frame = [binned_profile(s["tau"], s["cp"], n_ad) for s in sel]
    centres = per_frame[-1][0]
    means = []
    for k in range(n_ad):
        vals = [pf[1][k] for pf in per_frame if pf[1][k] == pf[1][k]]
        means.append(statistics.mean(vals) if vals else float("nan"))
    counts = [max(pf[2][k] for pf in per_frame) for k in range(n_ad)]
    peak, tau_at, n_at = band_peak(centres, means, counts)
    bins["adaptive"] = {
        "rule": "bin width = 2 dx in x, converted to tau at the last frame",
        "bin_width_tau": dx_tau, "n_bins": n_ad,
        "cp_band_peak": peak, "tau_at_peak": tau_at, "samples_at_peak": n_at,
        "cp_at_wagner_tau_peak": value_at(centres, means,
                                          math.sqrt(1 - 1 / C ** 2), 0.05),
        "profile_tau": centres, "profile_cp": means,
    }

    refs = {"C": C, "cp_max_wagner": wagner_peak_over_q(cfg.beta),
            "cp_centre_wagner": wagner_p_over_q(0.0, cfg.beta)}
    rho = [s["max_rho_ratio"] for s in shots]
    artefact = {
        "case": {"dx": dx, "delta": delta, "form": form, "velocity": velocity,
                 "beta_deg": 30.0, "frames": frames,
                 "ensemble_frames": ensemble_frames,
                 "c0_over_V": cfg.sound_speed_factor},
        "derived": {"steps": step, "wall_seconds": round(wall, 1),
                    "q": q, "t_end": t_end,
                    "n_fluid": sim.n_fluid, "n_wedge_surface": sim.n_wedge,
                    "wetted_nodes_last": len(sel[-1]["tau"]),
                    "mean_samples_per_frame": mean_samples},
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "machine": platform.machine(),
                        "cpu_count": __import__("os").cpu_count(),
                        "torch_threads": torch.get_num_threads()},
        "series": series,
        "shot_metrics": {
            "max_rho_ratio": max(rho) if rho else float("nan"),
            "mean_max_rho_ratio": statistics.mean(rho) if rho else float("nan"),
            "cp_centre_mean": statistics.mean(
                [s["p_centre"] / q for s in series[-ensemble_frames:]]),
            "cp_footprint_max": max(s["p_max_footprint"] / q for s in series),
            "c_over_Vt_last": c_last / (velocity * sel[-1]["t"]),
            "c_over_Vt_reference": C,
        },
        "estimators": bins,
        "references": refs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"dx{int(round(dx * 1e4))}e-4_dd{str(delta).replace('.', 'p')}_{form}"
    (out_dir / f"metric_{tag}.json").write_text(
        json.dumps(artefact, indent=1, sort_keys=True))
    return artefact


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dx", type=float, default=4.0e-3)
    ap.add_argument("--delta", type=float, default=0.0,
                    help="density-diffusion coefficient (0 disables)")
    ap.add_argument("--form", default="molteni", choices=("molteni", "antuono"))
    ap.add_argument("--velocity", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--ensemble-frames", type=int, default=ENSEMBLE_FRAMES,
                    help="last N frames averaged into the profile (self-similar)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "artifacts"
                                         / "case_sph_wedge_entry_2d_metric"))
    args = ap.parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    art = run_case(args.dx, args.delta, args.form, args.velocity, args.frames,
                   Path(args.out), args.ensemble_frames)
    est = art["estimators"]
    print(f"dx={art['case']['dx']} delta={art['case']['delta']} "
          f"form={art['case']['form']} steps={art['derived']['steps']} "
          f"wall={art['derived']['wall_seconds']}s "
          f"wetted_nodes={art['derived']['wetted_nodes_last']}")
    print("bins : " + "  ".join(
        f"{k}:{v['cp_band_peak']:.2f}" for k, v in est["counts"].items()))
    print(f"adaptive(bin=2dx, n={est['adaptive']['n_bins']}): "
          f"{est['adaptive']['cp_band_peak']:.2f}")
    print(f"reference Wagner cp_max = {art['references']['cp_max_wagner']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())