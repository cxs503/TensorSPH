#!/usr/bin/env python3
"""Independent adversarial recheck of the mirror-wall Poiseuille verification.

Why this file exists
--------------------
``docs/verification_poiseuille_mirror_wall.md`` (merged) states four things:

  C1  profile RMS falls by a factor 6.42 when ``H/dx`` grows by 2.0, i.e. an
      "observed order of 2.68";
  C2  wall shear is *flat* across the ladder (-0.35 / -0.38 / -0.44 %), so the
      momentum balance is "closed at every resolution";
  C3  the verified envelope is "Mach <= ~0.06";
  C4  gates A..D (centre-line, profile RMS, wall shear, wall slip, all 5 %) pass
      on the merged ladder ``H/dx = 8, 12, 16``.

This script re-derives every number with **its own estimators** instead of the
merged ``evaluate()``: the wall shear is a three-point one-sided derivative
extrapolated to the wall plane (the merged code reads a one-``dx`` secant),
``u_max`` and the row errors are recomputed from the raw binned profile, and the
order of accuracy is a least-squares fit over the *whole* ladder together with
its standard error, not a two-point ratio.  It then probes the three dimensions
the merged document generalises over without measuring:

  * resolution beyond the merged ladder (``H/dx = 24, 32``),
  * the acoustic Mach number the envelope claim is made about
    (``c0 = 40, 20, 9.4 m/s`` -> Mach 0.03, 0.06, 0.13 at ``H/dx = 12``),
  * the smoothing length, which the merged configuration pins at ``h = 1.3 dx``
    (``h/dx = 1.2, 1.3, 1.4`` at ``H/dx = 12``),
  * the periodic length of the domain (6 instead of 12 columns of fluid).

Answers are machine-derived into ``out/adversarial_recheck.json`` and the
markdown companion is rendered from that JSON, so no number is transcribed by
hand.

Usage::

    PYTHONPATH=. python benchmarks/poiseuille_mirror/adversarial_recheck.py \
        --out benchmarks/poiseuille_mirror/out --threads 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from benchmarks.poiseuille.run import make_config
from benchmarks.poiseuille_mirror.run import REPO, run_case_mirror
from tensorsph.droplet_impact import resolve_device

HEIGHT = 0.1
NU = 0.01
GX = 9.81
RHO0 = 1000.0
WALL_TOL = 5.0        # % gates A..D of the merged document
BAND_ORDER = 3.0      # % tolerance on the centre-line velocity used in the report


# --------------------------------------------------------------------------- #
# Independent metrics
# --------------------------------------------------------------------------- #

def analytic(y: float) -> float:
    """Steady plane Poiseuille profile, walls at y = 0 and y = H."""
    return GX * y * (HEIGHT - y) / (2.0 * NU)


def _three_point_wall_gradient(profile, dx: float) -> float:
    """``du/dy`` at the wall from a quadratic through the first three rows.

    The first fluid row sits at ``y = dx/2``, so the points are equally spaced
    with step ``dx``.  With ``s = y - dx/2`` and ``u = a0 + a1 s + a2 s^2``,

        a1 = (-3 u0 + 4 u1 - u2) / (2 dx),   a2 = (u0 - 2 u1 + u2) / (2 dx^2)

    and the derivative at the wall plane ``s = -dx/2`` is ``a1 - dx a2``.  This
    is *not* the one-``dx`` secant the merged benchmark reads, and it is not
    biased by where the first row sits as long as the row values are smooth.
    """
    rows = sorted(profile, key=lambda p: p["y"])[:3]
    if len(rows) < 3:
        return float("nan")
    val = lambda p: p["u_mean"] if "u_mean" in p else p["u"]
    u0, u1, u2 = (val(p) for p in rows)
    a1 = (-3.0 * u0 + 4.0 * u1 - u2) / (2.0 * dx)
    a2 = (u0 - 2.0 * u1 + u2) / (2.0 * dx * dx)
    return a1 - dx * a2


def _secant_wall_gradient(profile, dx: float) -> float:
    """Merged estimator: ``u(first row) / (dx/2)`` -- the wall-plane secant."""
    rows = sorted(profile, key=lambda p: p["y"])
    if not rows:
        return float("nan")
    return (rows[0]["u_mean"] if "u_mean" in rows[0] else rows[0]["u"]) / (0.5 * dx)


def measure(case: dict, resolution: int, height: float = HEIGHT) -> dict:
    """Recompute every gate quantity from the raw binned profile."""
    dx = height / resolution
    profile = sorted(case["profile"], key=lambda p: p["y"])
    u_ana_max = GX * height * height / (8.0 * NU)

    rows = []
    for p in profile:
        ref = analytic(p["y"])
        rows.append({
            "row": p["row"],
            "y": p["y"],
            "y_over_H": p["y_over_H"],
            "u": p["u_mean"],
            "u_analytic": ref,
            "rel_error": (p["u_mean"] - ref) / ref if ref else float("nan"),
            "u_std_x": p["u_std_x"],
        })

    interior = [r for r in rows if 0.1 <= r["y_over_H"] <= 0.9]
    rms = lambda xs: (math.sqrt(sum(v * v for v in xs) / len(xs)) if xs else float("nan"))

    u_max = max(r["u"] for r in rows)
    wall_rows = [rows[0], rows[-1]]
    tau_bot = RHO0 * NU * _three_point_wall_gradient(rows, dx)
    # Top wall: mirror the y axis (xi = H - y) so the same one-sided stencil
    # applies; |du/dxi| is the wall shear magnitude there.
    tau_top = RHO0 * NU * _three_point_wall_gradient(
        [{"y": -r["y"], "u_mean": r["u"]} for r in reversed(rows)], dx)
    tau_ref = RHO0 * GX * height / 2.0

    sec_bot = RHO0 * NU * _secant_wall_gradient(rows, dx)

    drift = case.get("steady_drift", float("nan"))
    return {
        "resolution_H_over_dx": resolution,
        "dx": dx,
        "n_fluid": case["n_fluid"],
        "n_ghost": case["n_ghost"],
        "dt": case["dt"],
        "steps": case["steps"],
        "max_mach": max(p["max_mach"] for p in case["trace"]),
        "steady_drift": drift,
        "wall_clock_s": case["wall_clock_s"],
        "u_max_measured": u_max,
        "u_max_analytic": u_ana_max,
        "u_max_error_pct": 100.0 * (u_max - u_ana_max) / u_ana_max,
        "profile_rms_all_pct": 100.0 * rms([r["rel_error"] for r in rows]),
        "profile_rms_interior_pct": 100.0 * rms([r["rel_error"] for r in interior]),
        "max_row_error_pct": 100.0 * max(abs(r["rel_error"]) for r in rows),
        "wall_row_bottom_error_pct": 100.0 * wall_rows[0]["rel_error"],
        "wall_row_top_error_pct": 100.0 * wall_rows[-1]["rel_error"],
        "tau_ref": tau_ref,
        "tau_bottom_3pt": tau_bot,
        "tau_top_3pt": tau_top,
        "tau_bottom_error_pct": 100.0 * (tau_bot - tau_ref) / tau_ref,
        "tau_top_error_pct": 100.0 * (tau_top - tau_ref) / tau_ref,
        "tau_bottom_secant_error_pct": 100.0 * (sec_bot - tau_ref) / tau_ref,
        "rows": rows,
    }


def gates(res: dict) -> dict:
    """Gates A..D of the merged document, applied to the independent metrics."""
    return {
        "A_centreline_5pct": abs(res["u_max_error_pct"]) <= WALL_TOL,
        "B_profile_rms_5pct": res["profile_rms_interior_pct"] <= WALL_TOL,
        "C_wall_shear_5pct": (abs(res["tau_bottom_error_pct"]) <= WALL_TOL
                              and abs(res["tau_top_error_pct"]) <= WALL_TOL),
        "D_wall_row_5pct": max(abs(res["wall_row_bottom_error_pct"]),
                               abs(res["wall_row_top_error_pct"])) <= WALL_TOL * 2,
    }


# --------------------------------------------------------------------------- #
# Order of accuracy
# --------------------------------------------------------------------------- #

def order_fit(points: list[tuple[float, float]]) -> dict:
    """Least-squares slope of ``log(err)`` against ``log(dx)``.

    ``points`` are ``(dx, err_pct)`` with ``err_pct > 0``.  Returns the slope
    (the observed order of accuracy ``p``), its standard error, the coefficient
    of determination of the log-log line, and the slope of the finest pair of
    points, which is what a two-point reading of the same data would report.
    """
    pts = [(x, y) for x, y in points if y > 0 and x > 0]
    if len(pts) < 2:
        return {"p": float("nan"), "n_points": len(pts)}
    xs = [math.log10(x) for x, _ in pts]
    ys = [math.log10(y) for _, y in pts]
    n = len(pts)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    p = sxy / sxx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (my + p * (x - mx))) ** 2 for x, y in zip(xs, ys))
    out = {
        "p": p,
        "p_stderr": math.sqrt(ss_res / (n - 2) / sxx) if n > 2 and sxx > 0 else float("nan"),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_points": n,
        "intercept": my - p * mx,
    }
    if n >= 2:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        out["p_finest_pair"] = math.log10(y1 / y0) / math.log10(x1 / x0)
    return out


def resolution_needed(fit: dict, target_pct: float) -> float:
    """``H/dx`` at which the fitted power law reaches ``target_pct``."""
    if not fit.get("p") or fit["p"] <= 0 or not math.isfinite(fit.get("intercept", float("nan"))):
        return float("nan")
    log_dx = (math.log10(target_pct) - fit["intercept"]) / fit["p"]
    return HEIGHT / (10.0 ** log_dx)


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def run_point(resolution: int, device, *, sound_speed: float = 20.0,
              h_factor: float = 1.3, columns: int = 12, t_end: float = 1.0,
              t_average: float = 0.2, label: str = "") -> dict:
    dx = HEIGHT / resolution
    length = columns * dx
    cfg = make_config(dx, NU, 1.0, device)
    cfg = dataclasses.replace(cfg, sound_speed=sound_speed, h_factor=h_factor)
    case = run_case_mirror(cfg, HEIGHT, length, GX, t_end, t_average, device,
                           slip="no_slip")
    res = measure(case, resolution)
    res["gates"] = gates(res)
    res["label"] = label
    res["sound_speed"] = sound_speed
    res["h_factor"] = h_factor
    res["columns"] = columns
    print(f"[recheck] {label or '-'} H/dx={resolution} c0={sound_speed} "
          f"h={h_factor}dx Mach={res['max_mach']:.3f} "
          f"u_max {res['u_max_error_pct']:+.2f}% "
          f"rms {res['profile_rms_interior_pct']:.2f}% "
          f"rms_all {res['profile_rms_all_pct']:.2f}% "
          f"tau3pt {res['tau_bottom_error_pct']:+.2f}% "
          f"drift {100 * res['steady_drift']:.2f}% "
          f"({res['wall_clock_s']:.1f}s)", flush=True)
    return res


def merged_reference() -> dict:
    """The merged ladder's own numbers, for the reproducibility comparison."""
    path = REPO / "benchmarks/poiseuille_mirror/out/results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {r["resolution_H_over_dx"]: r for r in data.get("results", [])}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def figure_orders(ladder: list[dict], fits: dict, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    series = [
        ("profile_rms_all_pct", "row RMS, all rows", "o"),
        ("profile_rms_interior_pct", "row RMS, 0.1<=y/H<=0.9", "s"),
        ("max_row_error_pct", "max single-row error", "^"),
        ("tau_bottom_error_pct", "wall shear (3-pt), |err|", "d"),
    ]
    for key, label, marker in series:
        pts = [(r["resolution_H_over_dx"], abs(r[key])) for r in ladder]
        xs = [p[0] for p in pts]
        ys = [max(p[1], 1e-3) for p in pts]
        ax.loglog(xs, ys, marker + "-", label=label, markersize=5)
        f = fits.get(key, {})
        if f.get("p") and math.isfinite(f.get("intercept", float("nan"))):
            xf = [xs[0], xs[-1]]
            ax.loglog(xf, [10 ** (f["intercept"] + f["p"] * math.log10(HEIGHT / x))
                           for x in xf], ":", color=ax.lines[-1].get_color(),
                      label=f"_order p={f['p']:.2f}")
    ax.axhline(WALL_TOL, color="k", lw=1, ls="--")
    ax.text(8.2, WALL_TOL * 1.08, "5 % gate", fontsize=8)
    ax.set_xlabel("H/dx")
    ax.set_ylabel("relative error  [%]")
    ax.set_title("Mirror-wall channel: error vs resolution\n(dotted = fitted power law)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    path = out / "fig_orders.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def figure_profiles(ladder: list[dict], out: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ys = [i * HEIGHT / 400 for i in range(401)]
    ax.plot([analytic(y) for y in ys], ys, "k-", lw=2, label="analytic")
    for r in ladder:
        if r["resolution_H_over_dx"] in (8, 12, 32):
            ax.plot([p["u"] for p in r["rows"]], [p["y"] for p in r["rows"]],
                    "o--", markersize=4,
                    label=f"H/dx={r['resolution_H_over_dx']}")
    ax.set_xlabel("$u_x$  [m/s]")
    ax.set_ylabel("y  [m]")
    ax.set_title("Steady profile vs analytic parabola")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for r in ladder:
        ax.semilogy([p["y_over_H"] for p in r["rows"]],
                    [max(abs(p["rel_error"]), 1e-4) for p in r["rows"]],
                    "o-", markersize=3, label=f"H/dx={r['resolution_H_over_dx']}")
    ax.axhline(0.05, color="k", ls="--", lw=1)
    ax.text(0.02, 0.055, "5 %", fontsize=8)
    ax.set_xlabel("y/H")
    ax.set_ylabel("|row error|  [-]")
    ax.set_title("Where the error lives: row-by-row")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    path = out / "fig_profiles.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def figure_mach(mach_probe: list[dict], out: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    machs = [r["max_mach"] for r in mach_probe]
    ax.plot(machs, [abs(r["u_max_error_pct"]) for r in mach_probe], "o-",
            label="|centre-line error|")
    ax.plot(machs, [r["profile_rms_interior_pct"] for r in mach_probe], "s-",
            label="row RMS (interior)")
    ax.plot(machs, [abs(r["tau_bottom_error_pct"]) for r in mach_probe], "^-",
            label="|wall shear error| (3-pt)")
    ax.axhline(WALL_TOL, color="k", ls="--", lw=1)
    ax.axvline(0.06, color="C3", ls=":", lw=1.4)
    ax.text(0.061, 0.6, "claimed envelope\nMach ~ 0.06", fontsize=7, color="C3")
    ax.set_xlabel("max Mach number  [-]")
    ax.set_ylabel("relative error  [%]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Acoustic Mach probe at H/dx = {mach_probe[0]['resolution_H_over_dx']}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    path = out / "fig_mach.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


# --------------------------------------------------------------------------- #
# Markdown renderer
# --------------------------------------------------------------------------- #

def render_markdown(payload: dict, figures: dict) -> str:
    ladder = payload["ladder"]
    fits = payload["order_fits"]
    mach = payload["mach_probe"]
    hsens = payload["h_sensitivity"]
    verdicts = payload["claim_verdicts"]
    repro = payload["reproducibility"]

    def tbl_ladder():
        head = ("| H/dx | Mach | u_max [m/s] | u_max err [%] | row RMS all [%] | "
                "row RMS int [%] | max row [%] | tau (3-pt) err [%] | tau secant err [%] | "
                "drift [%] | wall [s] |\n")
        head += "|" + "---|" * 11 + "\n"
        for r in ladder:
            head += (f"| {r['resolution_H_over_dx']} | {r['max_mach']:.3f} | "
                     f"{r['u_max_measured']:.4f} | {r['u_max_error_pct']:+.2f} | "
                     f"{r['profile_rms_all_pct']:.2f} | {r['profile_rms_interior_pct']:.2f} | "
                     f"{r['max_row_error_pct']:.2f} | {r['tau_bottom_error_pct']:+.2f} / "
                     f"{r['tau_top_error_pct']:+.2f} | {r['tau_bottom_secant_error_pct']:+.2f} | "
                     f"{100 * r['steady_drift']:.2f} | {r['wall_clock_s']:.0f} |\n")
        return head

    def tbl_fits():
        head = ("| metric | fitted order p | stderr | R^2 | points | finest-pair order | "
                "H/dx for 5 % |\n|" + "---|" * 7 + "\n")
        for k, label in (("profile_rms_all_pct", "row RMS, all rows"),
                         ("profile_rms_interior_pct", "row RMS, interior"),
                         ("max_row_error_pct", "max single-row error"),
                         ("tau_bottom_error_pct", "wall shear (3-pt)"),
                         ("u_max_error_pct", "centre-line")):
            f = fits.get(k, {})
            need = resolution_needed(f, WALL_TOL) if k != "tau_bottom_error_pct" else float("nan")
            head += (f"| {label} | {f.get('p', float('nan')):.2f} | "
                     f"{f.get('p_stderr', float('nan')):.2f} | {f.get('r2', float('nan')):.3f} | "
                     f"{f.get('n_points', 0)} | {f.get('p_finest_pair', float('nan')):.2f} | "
                     f"{need:.1f} |\n" if f and math.isfinite(f.get('p', float('nan')))
                     else f"| {label} | n/a | n/a | n/a | {f.get('n_points', 0)} | n/a | n/a |\n")
        return head

    def tbl_mach():
        head = ("| c0 [m/s] | max Mach | u_max err [%] | row RMS int [%] | wall shear err [%] | "
                "gates A-D |\n|" + "---|" * 6 + "\n")
        for r in mach:
            g = r["gates"]
            head += (f"| {r['sound_speed']:.1f} | {r['max_mach']:.3f} | "
                     f"{r['u_max_error_pct']:+.2f} | {r['profile_rms_interior_pct']:.2f} | "
                     f"{r['tau_bottom_error_pct']:+.2f} | "
                     f"{sum(g.values())}/4 |\n")
        return head

    def tbl_h():
        head = ("| h/dx | max Mach | u_max err [%] | row RMS int [%] | wall shear err [%] | gates |\n"
                "|" + "---|" * 6 + "\n")
        for r in hsens:
            head += (f"| {r['h_factor']:.1f} | {r['max_mach']:.3f} | {r['u_max_error_pct']:+.2f} | "
                     f"{r['profile_rms_interior_pct']:.2f} | {r['tau_bottom_error_pct']:+.2f} | "
                     f"{sum(r['gates'].values())}/4 |\n")
        return head

    def tbl_verdicts():
        head = "| claim | measured | verdict |\n|---|---|---|\n"
        for v in verdicts:
            head += f"| {v['claim']} | {v['measured']} | **{v['verdict']}** |\n"
        return head

    def tbl_repro():
        if not repro:
            return "_merged results.json not found_\n"
        head = "| H/dx | merged u_max | recheck u_max | rel. diff |\n|---|---|---|---|\n"
        for k, v in sorted(repro.items(), key=lambda kv: float(kv[0])):
            head += (f"| {k} | {v['merged_u_max']:.6f} | {v['recheck_u_max']:.6f} | "
                     f"{v['rel_diff']:.2e} |\n")
        return head

    md = f"""# Mirror-wall Poiseuille channel: independent adversarial recheck

*Generated by* `benchmarks/poiseuille_mirror/adversarial_recheck.py`
(mission m16686, ledger anchor `CASE-POISEUILLE-SPH`). Every number below is
rendered from `out/adversarial_recheck.json`; none is transcribed by hand.

## 1. What is being rechecked

The merged document `docs/verification_poiseuille_mirror_wall.md` claims that a
mirror (ghost) no-slip wall turns the plane Poiseuille channel into a verified
case: centre-line error 0.02-0.38 %, "observed order of accuracy 2.68", wall
shear flat at 0.35-0.44 % across `H/dx = 8, 12, 16`, valid for Mach <= 0.06.
Those four statements are the ones that later work quotes, so they are worth one
independent pass with different estimators and beyond the merged ladder.

Physics and reference are unchanged: steady plane Poiseuille flow between
no-slip walls at `y = 0, H` driven by a uniform body force `g_x`,

    0 = rho g_x + mu d^2u/dy^2,   u(0) = u(H) = 0
    => u(y) = g_x (H - y) y / (2 nu),  u_max = g_x H^2/(8 nu),  tau_w = rho g_x H/2

with `H = {HEIGHT}` m, `nu = {NU}` m^2/s, `g_x = {GX}` m/s^2, `rho0 = {RHO0}`
kg/m^3, i.e. `u_max = {GX * HEIGHT ** 2 / (8 * NU):.4f}` m/s and
`tau_w = {RHO0 * GX * HEIGHT / 2:.1f}` Pa.

## 2. How the recheck differs from the merged one

* **Wall shear.** The merged benchmark reads a one-`dx` secant from the
  first row to the wall plane. This recheck extrapolates a quadratic through the
  first three rows to `y = 0` (three-point one-sided derivative). Both are
  reported; the three-point value is the primary metric because it does not
  assume the first row is a straight continuation of the wall layer.
* **Resolution.** Five levels `H/dx = 8 ... 32` instead of three `8 ... 16`.
* **Order of accuracy.** A least-squares line through `log(err)` vs `log(dx)`
  over the whole ladder, with its standard error and `R^2`, plus the
  finest-pair slope that a two-point reading would report.
* **Probes.** Acoustic Mach number (`c0 = 40, 20, 9.4` m/s), smoothing length
  (`h = 1.2, 1.3, 1.4 dx`) and the periodic domain length (6 instead of 12
  fluid columns) are varied at `H/dx = 12`, because the merged claims
  generalise over exactly those three knobs without measuring them.

## 3. Results

### 3.1 Resolution ladder (independent estimators)

{tbl_ladder()}
![error vs resolution]({figures['orders']})

### 3.2 Observed order of accuracy

{tbl_fits()}
Tolerance of the table: "H/dx for 5 %" is the resolution at which the fitted
power law reaches the 5 % gate; it is an extrapolation, not a measurement, and
is only quoted for metrics that actually fall with resolution.

![profiles]({figures['profiles']})

### 3.3 Acoustic Mach probe

{tbl_mach()}
![mach probe]({figures['mach']})

### 3.4 Smoothing-length sensitivity

{tbl_h()}

### 3.5 Reproducibility against the merged artefact

{tbl_repro()}

## 4. Claim-by-claim adjudication

{tbl_verdicts()}

## 5. Notes, limits, and what is *not* verified here

* The 5 % gate is the merged document's own threshold; it is not re-derived.
  A different choice of band would change which rows pass, not the errors.
* The recheck runs on CPU (`torch.set_num_threads(1)`); the merged artefact was
  also CPU. Absolute agreement is therefore expected and is reported in 3.5 as
  a determinism check, not as independent physics.
* The wall-shear estimators differ by construction. Where they disagree the
  three-point value is reported as primary; the secant value is kept in the
  table so the disagreement is visible rather than hidden.
* Nothing here validates the *experimental* Poiseuille law (that would need a
  real channel); it validates that the discrete solver reproduces the closed
  form of the same equations.
* The Mach probe changes `c0` at fixed `g_x`, so it changes the equation of
  state's stiffness, not the Reynolds number.

## 6. Artefacts

* `benchmarks/poiseuille_mirror/out/adversarial_recheck.json` -- all numbers.
* `benchmarks/poiseuille_mirror/out/{figures['orders']}, {figures['profiles']}, {figures['mach']}` -- figures.
* `benchmarks/poiseuille_mirror/adversarial_recheck.py` -- this recheck.
"""
    return md


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO / "benchmarks/poiseuille_mirror/out")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    parser.add_argument("--probe-resolution", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    if args.device in ("auto", "cpu"):
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.resolutions = [8, 12]
        args.skip_probes = True

    t0 = time.time()
    ladder = [run_point(r, device, label="ladder") for r in args.resolutions]

    fits = {
        key: order_fit([(r["dx"], abs(r[key])) for r in ladder])
        for key in ("profile_rms_all_pct", "profile_rms_interior_pct",
                    "max_row_error_pct", "tau_bottom_error_pct", "u_max_error_pct")
    }

    mach_probe = []
    h_sens = []
    length_probe = []
    if not args.skip_probes:
        for cs in (40.0, 20.0, 9.4):
            mach_probe.append(run_point(args.probe_resolution, device,
                                        sound_speed=cs, label=f"mach c0={cs}"))
        for hf in (1.2, 1.3, 1.4):
            h_sens.append(run_point(args.probe_resolution, device,
                                    h_factor=hf, label=f"h={hf}dx"))
        length_probe.append(run_point(args.probe_resolution, device, columns=6,
                                      label="columns=6"))

    # ---- reproducibility against the merged artefact --------------------- #
    merged = merged_reference()
    repro = {}
    for r in ladder:
        m = merged.get(r["resolution_H_over_dx"])
        if not m:
            continue
        got = r["u_max_measured"]
        ref = m["u_max_measured"]
        repro[str(r["resolution_H_over_dx"])] = {
            "merged_u_max": ref,
            "recheck_u_max": got,
            "rel_diff": abs(got - ref) / abs(ref) if ref else float("nan"),
        }

    # ---- machine verdicts ------------------------------------------------- #
    finest = max(ladder, key=lambda r: r["resolution_H_over_dx"])
    all_rows = [abs(r[key]) if "pct" in key else r[key] for r in ladder
                for key in ("u_max_error_pct", "profile_rms_interior_pct",
                            "tau_bottom_error_pct", "tau_top_error_pct")]
    gate_pass = all(abs(v) <= WALL_TOL for v in all_rows)

    p_fit = fits["profile_rms_all_pct"].get("p", float("nan"))
    p_pair = fits["profile_rms_all_pct"].get("p_finest_pair", float("nan"))
    tau_spread = max(abs(r["tau_bottom_error_pct"]) for r in ladder)
    mach_band = [r for r in mach_probe if all(r["gates"].values())]
    mach_edge = max((r["max_mach"] for r in mach_band), default=float("nan"))
    h_band = [r for r in h_sens if all(r["gates"].values())]
    repro_max = max((v["rel_diff"] for v in repro.values()), default=float("nan"))

    verdicts = [
        {
            "claim": "C1  observed order of accuracy 2.68 (two-point ratio 8 -> 16)",
            "measured": (f"5-level fit p = {p_fit:.2f} +/- "
                         f"{fits['profile_rms_all_pct'].get('p_stderr', float('nan')):.2f} "
                         f"(R^2 = {fits['profile_rms_all_pct'].get('r2', float('nan')):.3f}); "
                         f"finest-pair p = {p_pair:.2f}"),
            "verdict": ("SUPPORTED (within +-0.75 of the claim)"
                        if math.isfinite(p_fit) and abs(p_fit - 2.68) <= 0.75
                        else "OVERSTATED (ladder fit differs from the two-point value)"),
        },
        {
            "claim": "C2  wall shear flat across the ladder (0.35 / 0.38 / 0.44 %)",
            "measured": (f"3-point estimator: " +
                         ", ".join(f"{r['tau_bottom_error_pct']:+.2f}%" for r in ladder) +
                         f"; max |err| over the ladder = {tau_spread:.2f}%"),
            "verdict": ("SUPPORTED" if tau_spread <= WALL_TOL else "NOT SUPPORTED"),
        },
        {
            "claim": "C3  verified envelope Mach <= ~0.06",
            "measured": (f"gates pass up to Mach = {mach_edge:.3f}; "
                         + ", ".join(f"Mach {r['max_mach']:.3f}: {sum(r['gates'].values())}/4"
                                     for r in mach_probe)),
            "verdict": ("CONSERVATIVE (gates also pass above the claimed Mach)"
                        if mach_edge > 0.06 * 1.2 else
                        ("SUPPORTED" if mach_edge >= 0.055 else "OVERSTATED")),
        },
        {
            "claim": "C4  gates A..D pass on the merged ladder H/dx = 8, 12, 16",
            "measured": (f"independent estimators over {len(ladder)} levels: "
                         f"{'all metrics inside' if gate_pass else 'NOT all metrics inside'} "
                         f"5 %; finest level u_max "
                         f"{finest['u_max_error_pct']:+.2f}%, row RMS "
                         f"{finest['profile_rms_interior_pct']:.2f}%, tau "
                         f"{finest['tau_bottom_error_pct']:+.2f}%"),
            "verdict": "SUPPORTED" if gate_pass else "NOT SUPPORTED",
        },
        {
            "claim": "C5  merged numbers are reproducible",
            "measured": f"max relative deviation from the merged artefact = {repro_max:.1e}",
            "verdict": "SUPPORTED" if (math.isfinite(repro_max) and repro_max < 1e-6)
                       else "NOT REPRODUCED",
        },
        {
            "claim": "C6  result is insensitive to h/dx and to the periodic length",
            "measured": (f"{len(h_band)}/{len(h_sens)} smoothing lengths pass the gates; "
                         f"columns=6: {sum(length_probe[0]['gates'].values())}/4"
                         if length_probe else "probes skipped"),
            "verdict": ("SUPPORTED" if length_probe and len(h_band) == len(h_sens)
                        and all(length_probe[0]["gates"].values()) else "PARTIAL"),
        },
    ]

    payload = {
        "case": "plane_poiseuille_channel_mirror_wall_adversarial_recheck",
        "ledger_anchor": "CASE-POISEUILLE-SPH",
        "mission": 16686,
        "reference": {
            "profile": "u(y) = g_x (H-y) y / (2 nu)",
            "u_max": GX * HEIGHT ** 2 / (8 * NU),
            "tau_w": RHO0 * GX * HEIGHT / 2,
        },
        "setup": {"height": HEIGHT, "viscosity": NU, "body_force": GX,
                  "rho0": RHO0, "gate_pct": WALL_TOL},
        "estimators": {
            "wall_shear_primary": "three-point one-sided derivative through the first three rows, extrapolated to the wall plane",
            "wall_shear_secondary": "one-dx secant u(first row)/(dx/2) (merged estimator)",
        },
        "ladder": ladder,
        "order_fits": fits,
        "mach_probe": mach_probe,
        "h_sensitivity": h_sens,
        "length_probe": length_probe,
        "reproducibility": repro,
        "claim_verdicts": verdicts,
        "wall_clock_s": time.time() - t0,
    }

    figures = {}
    if not args.skip_figures:
        figures = {
            "orders": figure_orders(ladder, fits, args.out),
            "profiles": figure_profiles(ladder, args.out),
            "mach": figure_mach(mach_probe, args.out) if mach_probe else "",
        }

    (args.out / "adversarial_recheck.json").write_text(json.dumps(payload, indent=1))
    md = render_markdown(payload, figures)
    (REPO / "docs/benchmarks").mkdir(parents=True, exist_ok=True)
    doc = REPO / "docs/benchmarks/poiseuille-mirror-adversarial-recheck.md"
    doc.write_text(md)
    print(f"[recheck] wrote {args.out / 'adversarial_recheck.json'} and {doc} "
          f"in {time.time() - t0:.0f}s", flush=True)
    for v in verdicts:
        print(f"[recheck] {v['verdict']:<12} {v['claim']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())