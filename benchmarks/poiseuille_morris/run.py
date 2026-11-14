"""Morris low-Reynolds plane Poiseuille channel: reference-vs-solution ladder.

Anchor
------
CASE-SPH-POISEUILLE-MORRIS.  Body-force-driven flow between two no-slip plates,
periodic in ``x``, integrated to a steady state and compared with the exact
steady solution.  The case authority is

    Morris, J.P., Fox, P.J., Zhu, Y. (1997). "Modeling Low Reynolds Number
    Incompressible Flows Using SPH."  J. Comput. Phys. 136(1), 214-226,
    doi:10.1006/jcph.1997.5776  -- the plane-Poiseuille validation section.

Reference numbers, and what is *not* transcribed
------------------------------------------------
The journal article is closed access (Unpaywall reports ``oa_status = closed``
and no repository copy, checked 2026-11-14), so the paper's own printed profile
/ table cannot be read here and no digit from it is reproduced from memory --
that would be a hand copy, which this benchmark forbids.  The reference curve
the paper validates against is the exact steady solution of the
Navier-Stokes equations for this geometry, and every reference number below is
*evaluated programmatically* from those equations (see ``reference_profile``):

    u(y)  = g y (H - y) / (2 nu),   u_max = g H^2 / (8 nu),
    tau_w = rho g H / 2             (momentum balance, d p/d x = -rho g)

which is the same closed form as Batchelor (1967, "An Introduction to Fluid
Dynamics", Sec. 4.2) and the steady limit of the Boussinesq (1868) start-up
series already implemented in ``tensorsph.isph_channel``.

Routes compared
---------------
``isph``  : ``tensorsph.isph_channel`` projection solver (no Tait EOS, no
            acoustic time-step limit) -- the route used for the headline claim.
``wcsp``  : ``benchmarks.poiseuille.run`` production WCSPH path
            (``tensorsph.droplet_impact.rhs``, Morris-Laplacian viscous term) --
            the incumbent baseline / negative control.

Design (predeclared in ``DESIGN.md``)
-------------------------------------
* resolution ladder  ``H/dx in {8, 12, 16, 24}`` at ``Re_c in {1.0, 12.26}``
* Reynolds ladder    ``Re_c in {0.1, 1.0, 12.26}`` at ``H/dx = 16``
* support ladder     ``h/dx in {1.5, 1.9, 2.3}`` at ``H/dx = 16``
* domain ladder      ``columns in {6, 12}`` (``L/H = 0.6, 1.2``) at ``H/dx = 16``

``Re_c = u_max H / nu`` is the Reynolds number built on the *analytical* peak
velocity, so it is known before the run and cannot be tuned afterwards.

Usage
-----
    PYTHONPATH=src:. python benchmarks/poiseuille_morris/run.py \
        --out runs/poiseuille_morris [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import resolve_device
from tensorsph.isph_channel import ChannelConfig, gates, run_case, fit_profile

HEIGHT = 0.1          # m, channel height H
RHO0 = 1000.0         # kg/m^3
RE_BASE = 12.26       # u_max H / nu of the repository's canonical channel
                      # (nu = 0.01 m^2/s, g = 9.81 m/s^2, H = 0.1 m)
N_DIFFUSIVE = 10.0    # t_end in units of the diffusive time H^2/(nu pi^2)

REFERENCE = {
    "case": "plane Poiseuille, body force driven, periodic in x, no-slip walls",
    "authority": ("Morris, J.P., Fox, P.J., Zhu, Y. (1997) Modeling Low Reynolds "
                  "Number Incompressible Flows Using SPH, J. Comput. Phys. "
                  "136(1):214-226, doi:10.1006/jcph.1997.5776 (Poiseuille section)"),
    "authority_access": {
        "status": "WITHHELD",
        "why": ("journal article closed access: Unpaywall api reports "
                "oa_status=closed, has_repository_copy=false, no OA location "
                "(checked 2026-11-14); the printed profile/table is therefore not "
                "readable in this environment"),
        "policy": ("zero hand copy: no number from the paper is reproduced from "
                   "memory; only the closed-form reference the paper compares to "
                   "is used, and it is evaluated programmatically"),
    },
    "secondary": [
        "Batchelor (1967) An Introduction to Fluid Dynamics, Sec. 4.2",
        "Boussinesq (1868) steady limit of the start-up series",
    ],
    "equations": {
        "u": "g * y * (H - y) / (2 * nu)",
        "u_max": "g * H^2 / (8 * nu)",
        "tau_wall": "rho * g * H / 2",
    },
}


# --------------------------------------------------------------------------- #
# Reference solution (evaluated, never transcribed)
# --------------------------------------------------------------------------- #

def reference_profile(y, height: float, body_force: float, viscosity: float):
    """Exact steady velocity of the plane channel; numpy-free scalar form."""
    return body_force * y * (height - y) / (2.0 * viscosity)


def case_from_reynolds(re_c: float, height: float = HEIGHT):
    """(viscosity, body_force, u_max_ref) for a target Re_c = u_max H / nu.

    The peak velocity is held at 1 m/s and the forcing is then fixed by
    ``g = 8 nu u_max / H^2``, so the Reynolds number is a *chosen* parameter and
    not something read off a finished run.
    """
    u_max = 1.0
    nu = u_max * height / re_c
    g = 8.0 * nu * u_max / height ** 2
    return nu, g, u_max


def make_config(re_c: float, resolution: int, h_factor: float = 1.90,
                columns: int = 12, height: float = HEIGHT) -> ChannelConfig:
    nu, g, _ = case_from_reynolds(re_c, height)
    t_diff = height ** 2 / (nu * math.pi ** 2)
    t_end = N_DIFFUSIVE * t_diff
    return ChannelConfig(
        height=height,
        columns=columns,
        resolution=resolution,
        h_factor=h_factor,
        rho0=RHO0,
        viscosity=nu,
        body_force=g,
        wall_layers=5,
        t_end=t_end,
        t_average=0.1 * t_end,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def profile_metrics(profile, cfg: ChannelConfig, lo: float = 0.15,
                    hi: float = 0.85) -> dict:
    """Row-wise comparison against the exact solution.

    Errors are normalised by the analytical peak ``u_max = g H^2 / (8 nu)``,
    which stays finite at the walls (a per-row relative error would divide by
    zero there and is reported separately as a diagnostic only).
    """
    height, g, nu = cfg.height, cfg.body_force, cfg.viscosity
    u_max_ref = g * height ** 2 / (8.0 * nu)

    rows = []
    for p in profile:
        y = p["y"]
        u_ref = reference_profile(y, height, g, nu)
        rows.append({
            "row": p["row"],
            "y": y,
            "y_over_H": p["y_over_H"],
            "u_measured": p["u_mean"],
            "u_reference": u_ref,
            "u_over_umax_measured": p["u_mean"] / u_max_ref,
            "u_over_umax_reference": u_ref / u_max_ref,
            "err_over_umax": (p["u_mean"] - u_ref) / u_max_ref,
        })

    def _band(band_lo, band_hi):
        sel = [r for r in rows if band_lo <= r["y_over_H"] <= band_hi]
        if not sel:
            return {}
        e = [r["err_over_umax"] for r in sel]
        num = sum(x * x for x in e)
        den = sum(r["u_over_umax_reference"] ** 2 for r in sel)
        return {
            "n_rows": len(sel),
            "l2_over_umax_pct": 100.0 * math.sqrt(num / len(sel)),
            "l2_relative_pct": 100.0 * math.sqrt(num / den) if den else float("nan"),
            "max_abs_over_umax_pct": 100.0 * max(abs(x) for x in e),
        }

    centre_row = min(rows, key=lambda r: abs(r["y_over_H"] - 0.5)) if rows else None
    return {
        "u_max_reference": u_max_ref,
        "u_centre_measured": centre_row["u_measured"] if centre_row else float("nan"),
        "u_centre_reference": centre_row["u_reference"] if centre_row else float("nan"),
        "centre_error_pct": (
            100.0 * centre_row["err_over_umax"] / (centre_row["u_over_umax_reference"])
            if centre_row and centre_row["u_reference"] else float("nan")
        ),
        "all_rows": _band(0.0, 1.0),
        "interior": _band(lo, hi),
        "rows": rows,
    }


def run_isph(re_c: float, resolution: int, h_factor: float = 1.90,
             columns: int = 12, device=None) -> dict:
    device = device if device is not None else resolve_device("cpu")
    cfg = make_config(re_c, resolution, h_factor=h_factor, columns=columns)
    result = run_case(cfg, device)
    fit = fit_profile(result["profile"], cfg.height, cfg.body_force,
                      cfg.viscosity, lo=0.05, hi=0.95)
    gate = gates(result, fit, cfg)
    metrics = profile_metrics(result["profile"], cfg)
    return {
        "route": "isph_projection",
        "reynolds_c": re_c,
        "resolution_H_over_dx": resolution,
        "h_over_dx": cfg.h_factor,
        "columns": columns,
        "length_over_height": cfg.length / cfg.height,
        "dx": cfg.dx,
        "h": cfg.h,
        "viscosity": cfg.viscosity,
        "body_force": cfg.body_force,
        "t_end": cfg.t_end,
        "t_end_over_diffusive_time": cfg.t_end / (cfg.height ** 2 / (cfg.viscosity * math.pi ** 2)),
        "dt": result["trace"][-1]["dt"] if result["trace"] else float("nan"),
        "steps": result["steps"],
        "n_fluid": result["n_fluid"],
        "n_wall": result["n_wall"],
        "wall_clock_s": result["wall_clock_s"],
        "divergence_reduction": result["divergence_reduction"],
        "density_error_max": result["density_error_max"],
        "metrics": metrics,
        "gates": gate,
    }


def run_wcsp(re_c: float, resolution: int, device=None, columns: int = 12) -> dict:
    """Incumbent WCSPH production route, scored with the *same* metric."""
    from benchmarks.poiseuille.run import run_one

    device = device if device is not None else resolve_device("cpu")
    nu, g, _ = case_from_reynolds(re_c)
    height = HEIGHT
    t_diff = height ** 2 / (nu * math.pi ** 2)
    t_end = N_DIFFUSIVE * t_diff
    dx = height / resolution
    length = columns * dx
    raw = run_one(resolution, nu, g, 1.0, length, height, t_end,
                  0.1 * t_end, device)

    # re-score against the same reference as the ISPH route
    rows = raw["rows"]
    u_max_ref = g * height ** 2 / (8.0 * nu)
    scored = []
    for p in rows:
        u_ref = reference_profile(p["y"], height, g, nu)
        scored.append({
            "row": p["row"],
            "y": p["y"],
            "y_over_H": p["y_over_H"],
            "u_measured": p["u_mean"],
            "u_reference": u_ref,
            "u_over_umax_measured": p["u_mean"] / u_max_ref,
            "u_over_umax_reference": u_ref / u_max_ref,
            "err_over_umax": (p["u_mean"] - u_ref) / u_max_ref,
        })
    sel = [r for r in scored if 0.15 <= r["y_over_H"] <= 0.85]
    l2 = 100.0 * math.sqrt(sum(r["err_over_umax"] ** 2 for r in sel) / len(sel))
    centre = min(scored, key=lambda r: abs(r["y_over_H"] - 0.5))
    return {
        "route": "wcsp_production",
        "reynolds_c": re_c,
        "resolution_H_over_dx": resolution,
        "h_over_dx": 1.3,
        "columns": columns,
        "dx": dx,
        "viscosity": nu,
        "body_force": g,
        "t_end": t_end,
        "steps": raw["steps"],
        "n_fluid": raw["n_fluid"],
        "n_wall": raw["n_wall"],
        "wall_clock_s": raw["wall_clock_s"],
        "metrics": {
            "u_max_reference": u_max_ref,
            "u_centre_measured": centre["u_measured"],
            "u_centre_reference": centre["u_reference"],
            "centre_error_pct": 100.0 * centre["err_over_umax"] / centre["u_over_umax_reference"],
            "interior": {
                "n_rows": len(sel),
                "l2_over_umax_pct": l2,
                "max_abs_over_umax_pct": 100.0 * max(abs(r["err_over_umax"]) for r in sel),
            },
            "rows": scored,
        },
        "nu_eff_over_nu": raw["fit"]["nu_eff_over_nu"],
        "wall_offset_over_dx": raw["fit"]["delta"] / dx,
        "u_max_ratio": raw["u_max_ratio"],
    }


# --------------------------------------------------------------------------- #
# Convergence analysis
# --------------------------------------------------------------------------- #

def convergence_order(points, key="l2_over_umax_pct"):
    """Least-squares slope of log(err) vs log(dx); < 0 means converging."""
    pts = [(math.log(p["dx"]), math.log(p["metrics"]["interior"][key]))
           for p in points if p["metrics"]["interior"].get(key, 0) > 0]
    if len(pts) < 2:
        return {"order": float("nan"), "n_points": len(pts)}
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return {"order": -slope, "n_points": n,
            "fit": "log(e) = a + p log(dx), p = -slope"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/poiseuille_morris"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    device = resolve_device("cpu")
    t0 = time.time()

    def _env_list(name, default, cast):
        raw = os.environ.get(name, "").strip()
        if raw in ("skip", "none"):
            return []
        return [cast(v) for v in raw.split()] if raw else list(default)

    # The sweep can be sliced into separately launched chunks (PM_RES / PM_RE /
    # PM_H / PM_COLS); each chunk merges into an existing results.json instead of
    # overwriting it, so a long sweep fits inside a short scheduler slot.
    resolutions = _env_list("PM_RES", [8, 12, 16, 24], int)
    re_list = _env_list("PM_RE", [0.1, 1.0, RE_BASE], float)
    h_list = _env_list("PM_H", [1.5, 1.90, 2.3], float)
    columns_list = _env_list("PM_COLS", [6, 12], int)
    do_wcsp = os.environ.get("PM_WCSP", "1") != "0"
    do_controls = os.environ.get("PM_CONTROLS", "1") != "0"
    if args.quick:
        resolutions = [8, 12]
        re_list = [1.0, RE_BASE]
        h_list = [1.5, 1.90]
        columns_list = [12]

    isph = []
    wcsp = []

    def _log(tag, r):
        m = r["metrics"]
        print(f"[morris] {tag:34s} Re_c={r['reynolds_c']:<6g} "
              f"H/dx={r['resolution_H_over_dx']:<3d} "
              f"centre={m['centre_error_pct']:+6.2f}% "
              f"L2(int)={m['interior']['l2_over_umax_pct']:5.2f}% "
              f"({r['wall_clock_s']:.1f}s)", flush=True)

    for re_c in re_list:
        for res in resolutions:
            r = run_isph(re_c, res, device=device)
            isph.append(r)
            _log("isph res-ladder", r)

    # support-radius sensitivity at H/dx = 16, Re_c = 1
    for hf in h_list:
        r = run_isph(1.0, 16, h_factor=hf, device=device)
        isph.append(r)
        _log(f"isph h/dx={hf}", r)

    # domain-length sensitivity (periodic image adequacy) at H/dx = 16, Re_c = 1
    for cols in columns_list:
        r = run_isph(1.0, 16, columns=cols, device=device)
        isph.append(r)
        _log(f"isph columns={cols}", r)

    # controls: (i) zero forcing must give a zero field, (ii) a repeat of the
    # headline configuration must be bit-identical (deterministic solver).
    null_control, repeat_control = {}, {}
    if do_controls:
        ctrl_cfg = make_config(1.0, 16)
        ctrl_cfg.body_force = 0.0
        null_run = run_case(ctrl_cfg, device)
        null_control = {
            "case": "zero body force, same geometry and run length as Re_c = 1, H/dx = 16",
            "max_abs_u_measured": max(abs(p["u_mean"]) for p in null_run["profile"]),
            "steps": null_run["steps"],
            "wall_clock_s": null_run["wall_clock_s"],
            "pass": max(abs(p["u_mean"]) for p in null_run["profile"]) < 1e-12,
        }
        _log("control zero-forcing", {"metrics": {"centre_error_pct": 0.0,
                                                 "interior": {"l2_over_umax_pct": 0.0}},
                                      "reynolds_c": 0.0, "resolution_H_over_dx": 16,
                                      "wall_clock_s": null_run["wall_clock_s"]})

    if not do_wcsp:
        resolutions_for_wcsp = []
    # incumbent route, same metric
    for re_c in ((1.0, RE_BASE) if do_wcsp else ()):
        for res in ([8, 16] if args.quick else [8, 12, 16, 24] if not resolutions
                    else sorted(set(resolutions))):
            r = run_wcsp(re_c, res, device=device)
            wcsp.append(r)
            _log("wcsp production", r)

    # merge a previous chunk, if one exists, before the ladders are summarised
    previous = {}
    if args.out.joinpath("results.json").exists():
        previous = json.loads(args.out.joinpath("results.json").read_text())

    def _key(c):
        return (c["route"], c["reynolds_c"], c["resolution_H_over_dx"],
                round(c["h_over_dx"], 4), c["columns"])

    def _merge(current, old):
        seen = {_key(c) for c in current}
        return current + [c for c in old if _key(c) not in seen]

    if previous:
        isph = _merge(isph, previous.get("cases", {}).get("isph_projection", []))
        wcsp = _merge(wcsp, previous.get("cases", {}).get("wcsp_production", []))
        for k, v in (previous.get("controls") or {}).items():
            if v:
                if k == "zero_forcing":
                    null_control = null_control or v
                elif k == "repeat_headline":
                    repeat_control = repeat_control or v

    # determinism check: rebuild and re-run the headline configuration
    headline = [p for p in isph if p["reynolds_c"] == 1.0
                and p["resolution_H_over_dx"] == 16 and p["h_over_dx"] == 1.90
                and p["columns"] == 12]
    repeat_control = repeat_control or {}
    if headline and do_controls:
        repeat = run_isph(1.0, 16, device=device)
        repeat_control = {
            "case": "repeat of isph Re_c=1, H/dx=16, h/dx=1.90, columns=12",
            "centre_error_pct_first": headline[0]["metrics"]["centre_error_pct"],
            "centre_error_pct_repeat": repeat["metrics"]["centre_error_pct"],
            "profiles_identical": all(
                abs(a["u_measured"] - b["u_measured"]) == 0.0
                for a, b in zip(headline[0]["metrics"]["rows"], repeat["metrics"]["rows"])),
        }
        _log("control repeat headline", repeat)

    # group ladders
    def ladder(route, re_c, key="resolution_H_over_dx"):
        pts = [p for p in route if p["reynolds_c"] == re_c
               and p["h_over_dx"] == (1.90 if route is isph else 1.3)
               and p["columns"] == 12]
        pts.sort(key=lambda p: p[key])
        return pts

    summary = {}
    for re_c in re_list:
        pts = ladder(isph, re_c)
        if len(pts) >= 2:
            summary[f"isph_re{re_c}"] = {
                "points": [{"H_over_dx": p["resolution_H_over_dx"],
                            "centre_pct": p["metrics"]["centre_error_pct"],
                            "l2_interior_pct": p["metrics"]["interior"]["l2_over_umax_pct"],
                            "max_interior_pct": p["metrics"]["interior"]["max_abs_over_umax_pct"]}
                           for p in pts],
                "convergence": convergence_order(pts),
            }
    for re_c in (1.0, RE_BASE):
        pts = [p for p in wcsp if p["reynolds_c"] == re_c]
        pts.sort(key=lambda p: p["resolution_H_over_dx"])
        if len(pts) >= 2:
            summary[f"wcsp_re{re_c}"] = {
                "points": [{"H_over_dx": p["resolution_H_over_dx"],
                            "centre_pct": p["metrics"]["centre_error_pct"],
                            "l2_interior_pct": p["metrics"]["interior"]["l2_over_umax_pct"]}
                           for p in pts],
                "convergence": convergence_order(pts),
            }

    # headline verdict: finest ISPH run of the canonical low-Re case
    def verdict_for(route, re_c, limit_pct=5.0):
        pts = [p for p in route if p["reynolds_c"] == re_c and p["columns"] == 12]
        if not pts:
            return None
        finest = max(pts, key=lambda p: p["resolution_H_over_dx"])
        m = finest["metrics"]
        e_c = abs(m["centre_error_pct"])
        e_l2 = m["interior"]["l2_over_umax_pct"]
        return {
            "route": finest["route"],
            "reynolds_c": re_c,
            "H_over_dx": finest["resolution_H_over_dx"],
            "centre_error_pct": m["centre_error_pct"],
            "l2_interior_pct": e_l2,
            "limit_pct": limit_pct,
            "pass_profile": bool(e_l2 < limit_pct),
            "pass_centre": bool(e_c < limit_pct),
            "pass": bool(e_l2 < limit_pct and e_c < limit_pct),
        }

    payload = {
        "anchor": "CASE-SPH-POISEUILLE-MORRIS",
        "mission_id": 20084,
        "generated_by": "benchmarks/poiseuille_morris/run.py",
        "reference": REFERENCE,
        "settings": {
            "height": HEIGHT,
            "rho0": RHO0,
            "reynolds_base": RE_BASE,
            "n_diffusive_times": N_DIFFUSIVE,
            "resolutions": resolutions,
            "reynolds_ladder": re_list,
            "h_over_dx_ladder": h_list,
            "columns_ladder": columns_list,
            "reference_normalisation": "errors / u_max = g H^2/(8 nu)",
            "interior_band": [0.15, 0.85],
        },
        "cases": {"isph_projection": isph, "wcsp_production": wcsp},
        "controls": {"zero_forcing": null_control, "repeat_headline": repeat_control},
        "summary": summary,
        "verdict": {
            "isph": {str(re_c): verdict_for(isph, re_c) for re_c in re_list},
            "wcsp": {str(re_c): verdict_for(wcsp, re_c) for re_c in (1.0, RE_BASE)},
        },
        "totals": {
            "wall_clock_s": time.time() - t0,
            "n_cases": len(isph) + len(wcsp),
            "torch": torch.__version__,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[morris] wrote {args.out / 'results.json'} in {payload['totals']['wall_clock_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())