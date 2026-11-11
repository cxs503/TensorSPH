"""Round 2 of the capillary-wave open-surface audit: *what would it take?*

Why this file exists
--------------------
``benchmarks/capillary_wave/open_surface_audit.py`` turned the old evidence debt
("harness only, no artifact", E1) into a measurement: the linear capillary
standing wave cannot be carried at HEAD, the harness aborts at step 336, and the
verdict is WITHHELD.  That audit answered *whether* the claim is gradable.  It
did not answer *what would have to change* -- and it never separated the two
candidate causes it points at:

1. the discrete surface-tension force on the free surface, or
2. the boundary treatment of the pool, which in that audit is a bare ``clamp``
   box with **no wall particles at all** (every other pool benchmark in this
   repository -- hydrostatic, dam break, sloshing, conservation -- runs with
   ``wall_lattice``), so a fluid particle next to the clamp misses half its
   kernel and the kernel-sum colour function there looks like a *free surface*.

This study measures the decomposition instead of assuming it.  Four independent
readings, each with numbers:

``force_budget``
    Of the initial CSF acceleration budget ``sum |a|``, how much sits on
    wall-adjacent particles, on the free-surface shell, and in the interior?
    A wall-truncated colour function would concentrate it on the walls.
``wall_bc``
    The *same* case marched to the same horizon twice: with the clamp box of the
    original audit (no wall particles) and with static wall particles
    (``droplet_impact.wall_lattice``, ``boundary="wall"``).  If the abort is a
    boundary artefact, the wall run completes.
``curvature_floor``
    The estimator's noise floor: on a **perfectly flat** interface (``A = 0``)
    the discrete curvature must be zero.  Whatever it reports is the floor that
    a wave of curvature signal ``k^2 A`` has to clear.  Reported against
    resolution (20/40/80 points per wavelength) and against the signal of the
    amplitudes actually used by the harness.
``ladder``
    ``(lambda/dx, A/dx)`` cells marched with the wall boundary condition, each
    scored against the predeclared criteria below.  This is the part that can
    actually grade the claim; it is also the part that costs wall clock, so the
    cell list is a profile (``--profile quick`` for a sub-2-minute local check).

Predeclared analysis (fixed before the runs, per the experimental-design rule)
------------------------------------------------------------------------------
* dispersion criterion: ``|omega_measured - omega_Lamb| / omega_Lamb < 3 %``
  with the damped-oscillation fit residual ``< 0.2`` (below that the fit has no
  frequency to report);
* admissibility: Adami, Ihmsen & Adams (2012) spurious-current limit, ``v_max /
  (A omega) < 0.10``;
* a cell is **graded** only when both hold; a completed-but-inadmissible run is
  reported as a reading, not as a claim;
* stopping rule: horizon = 2 capillary periods, or the first boundary escape
  (whichever comes first), identical for every cell;
* no threshold is adjusted after the numbers are seen.  If no cell is graded the
  verdict stays WITHHELD and the shortfall is reported as a resolution
  criterion (the amplitude a wave needs on this discretisation to clear the
  curvature floor).

References: Lamb (1932) Art. 267/348; Adami, Ihmsen & Adams (2012) JCP
231:7057; Adami, Hu & Adams (2012) JCP 231:7057-7075 (wall BC); Brackbill,
Kothe & Zemach (1992) JCP 100:335.

Usage::

    PYTHONPATH=src python benchmarks/capillary_wave/resolution_criterion.py
    PYTHONPATH=src python benchmarks/capillary_wave/resolution_criterion.py \
        --profile quick --out /tmp        # local sanity check, < 2 min

The script always exits 0 for a completed study: it reports a state, the gates
live in ``tests/test_capillary_wave_resolution_criterion.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # run as a script: ``validation`` is next to us
    sys.path.insert(0, str(HERE))

import open_surface_audit as A  # noqa: E402  (path set above)
import validation as V  # noqa: E402

from tensorsph.csf_consistent import csf_fields  # noqa: E402
from tensorsph.droplet_impact import (  # noqa: E402
    apply_boundary,
    rhs,
    wall_lattice,
)
from tensorsph.neighbor_search import PairListCache  # noqa: E402

#: The harness case of the round-1 audit (``open_surface_audit.CASE``).
CASE = dict(A.CASE)
#: Predeclared criteria (see the module docstring).  Not tuned after the fact.
THRESHOLDS = {
    "dispersion_rel_error_pct": 3.0,
    "fit_residual_rel": 0.2,
    "adami_spurious_current": 0.10,
}
#: Horizon and the sampling used for the frequency fit (same for every cell).
HORIZON_PERIODS = 2.0
SAMPLES_PER_PERIOD = 120
#: ``lambda/dx`` and ``A/dx`` cells of the march ladder, per profile.
PROFILES = {
    #: ``quick`` is the profile that fits a single local invocation (< 2 min):
    #: an amplitude ladder at the harness resolution, one period.
    "quick": {"ppw": [20], "amp_over_dx": [1.0, 2.0, 4.0],
              "renormalization": [False], "periods": 1.0},
    "full": {"ppw": [20, 40], "amp_over_dx": [2.0, 4.0],
             "renormalization": [False, True], "periods": HORIZON_PERIODS},
}
#: Resolutions of the curvature-floor sweep (cheap: one ``rhs`` call each).
FLOOR_PPW = [20, 40, 80]
#: Amplitudes whose analytic curvature signal ``k^2 A`` is compared to the floor.
FLOOR_AMP_OVER_DX = [0.5, 1.0, 2.0, 4.0]


def _case(ppw: float, amp_over_dx: float, **over) -> dict:
    """Harness case at ``lambda/dx = ppw`` and ``A = amp_over_dx * dx``."""
    wavelength = CASE["wavelength"]
    return dict(wavelength=wavelength, depth=wavelength,
                dx=wavelength / ppw, amp_over_dx=amp_over_dx,
                rho0=CASE["rho0"], sigma=CASE["sigma"],
                viscosity=CASE["viscosity"], sound_speed=CASE["sound_speed"],
                **over)


# --------------------------------------------------------------------------- #
# 1. where the spurious force lives
# --------------------------------------------------------------------------- #
def force_budget(kw: dict = CASE) -> dict:
    """Decompose the initial CSF acceleration over walls / free surface / interior.

    ``initial_fields`` of the round-1 audit is reused unchanged, so the numbers
    are directly comparable with ``open_surface_audit.json``.
    """
    pos, mass, cfg, rho, p, acc = A.initial_fields(kw)
    a = torch.linalg.vector_norm(acc, dim=1)
    h = cfg.h_factor * cfg.dx
    width = kw["wavelength"] / 2.0
    x = pos[:, 0]
    y = pos[:, 1]
    top = float(y.max())
    wall_zone = (x < 2.5 * h) | (x > width - 2.5 * h)
    surface_zone = (~wall_zone) & (y > top - 3.0 * cfg.dx)
    interior = ~(wall_zone | surface_zone)
    total = float(a.sum())
    omega = V.angular_frequency(kw["sigma"], 2.0 * math.pi / kw["wavelength"],
                                kw["rho0"], kw["depth"])
    a_physical = omega ** 2 * kw["amp_over_dx"] * kw["dx"]
    out = {
        "a_physical_omega2A_m_per_s2": a_physical,
        "a_max_m_per_s2": float(a.max()),
        "a_max_over_physical": float(a.max()) / a_physical,
        "zones": {},
    }
    for name, mask in (("wall_adjacent", wall_zone),
                       ("free_surface", surface_zone),
                       ("interior", interior)):
        sub = a[mask]
        out["zones"][name] = {
            "n_particles": int(mask.sum()),
            "share_of_total_abs_a": (float(sub.sum()) / total) if total else 0.0,
            "a_max_m_per_s2": float(sub.max()) if sub.numel() else 0.0,
            "a_median_m_per_s2": float(sub.median()) if sub.numel() else 0.0,
        }
    out["wall_share"] = out["zones"]["wall_adjacent"]["share_of_total_abs_a"]
    # The corners are the worst case: two truncated directions at once.
    corner = ((x <= cfg.dx) | (x >= width - cfg.dx)) & (
        (y <= cfg.dx) | (y >= top - cfg.dx))
    out["corner"] = {
        "n_particles": int(corner.sum()),
        "a_max_m_per_s2": float(a[corner].max()) if corner.any() else 0.0,
        "share_of_total_abs_a": (float(a[corner].sum()) / total) if total else 0.0,
    }
    return out


# --------------------------------------------------------------------------- #
# 2. the boundary condition: clamp box vs static wall particles
# --------------------------------------------------------------------------- #
def march(kw: dict = CASE, wall_bc: bool = True, periods: float = HORIZON_PERIODS,
          **over) -> dict:
    """March the standing wave with or without static wall particles.

    ``wall_bc=False`` reproduces the round-1 audit exactly: a clamp box and
    ``wall=None`` in ``rhs``.  ``wall_bc=True`` adds ``wall_lattice`` particles
    and switches ``cfg.boundary`` to ``"wall"`` (the configuration every other
    pool benchmark in this repository uses).
    """
    over = dict(over)
    pos, mass, cfg = A.build(kw, **over)
    k = 2.0 * math.pi / kw["wavelength"]
    amp = kw["amp_over_dx"] * kw["dx"]
    omega = V.angular_frequency(kw["sigma"], k, kw["rho0"], kw["depth"])
    u_ref = amp * omega
    top = kw["depth"] + amp + 4.0 * cfg.dx
    dt = float(cfg.dt)
    steps = int(math.ceil(periods * 2.0 * math.pi / omega / dt))
    every = max(1, int(round(steps / (periods * SAMPLES_PER_PERIOD))))
    n_bins = max(8, int(round((kw["wavelength"] / 2.0) / (2.0 * kw["dx"]))))

    cfg.boundary = "wall" if wall_bc else "clamp"
    wall = (wall_lattice(cfg, pos.device, top=top).float() if wall_bc else None)
    h = cfg.h_factor * cfg.dx
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                          fluid_count=pos.shape[0])
    vel = torch.zeros_like(pos)

    times: list[float] = []
    amps: list[float] = []
    peak = 0.0
    rho_min = 1.0
    status, escape_step = "completed", None
    t0 = time.perf_counter()
    for step in range(steps + 1):
        if wall is None:
            pairs, _ = cache.get(pos)
        else:
            pairs, _ = cache.get(torch.cat((pos, wall), dim=0))
        rho, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall, dt=dt)
        rho_min = min(rho_min, float((rho / cfg.rho0).min()))
        if step % every == 0 or step == steps:
            times.append(step * dt)
            amps.append(V.column_elevation(pos, mass, kw["wavelength"] / 2.0,
                                           kw["depth"], kw["rho0"], k, n_bins))
        vel = vel + dt * acc
        pos = pos + dt * vel
        peak = max(peak, float(torch.linalg.vector_norm(vel, dim=1).max()))
        if apply_boundary(pos, vel, cfg, top=top):
            status, escape_step = "aborted", step
            break
    row = {
        "boundary": "static_wall_particles" if wall_bc else "clamp_box_no_wall",
        "status": status,
        "escape_step": escape_step,
        "steps_horizon": steps,
        "escape_fraction_of_horizon": (None if escape_step is None
                                       else escape_step / steps),
        "n_fluid_particles": int(pos.shape[0]),
        "n_wall_particles": 0 if wall is None else int(wall.shape[0]),
        "dt_s": dt,
        "horizon_periods": periods,
        "v_max_over_u_ref": peak / u_ref,
        "rho_min_over_rho0": rho_min,
        "admissible_adami": peak / u_ref < THRESHOLDS["adami_spurious_current"],
        "wall_clock_s": time.perf_counter() - t0,
    }
    if status == "completed":
        fit = V.fit_damped_oscillation(np.asarray(times), np.asarray(amps),
                                       omega)
        row.update({
            "omega_analytic_rad_per_s": omega,
            "omega_relative_error_pct": 100.0 * (fit["omega"] - omega) / omega,
            "fit_residual_rel": fit["residual_rel"],
            "amplitude_initial_measured_m": float(amps[0]) if amps else None,
            "amplitude_imposed_m": amp,
            "amplitude_history_m": amps,
            "t_history_s": times,
        })
    return row


def wall_bc_comparison(kw: dict = CASE, periods: float = HORIZON_PERIODS) -> dict:
    """Same case, same horizon, only the wall treatment differs."""
    clamp = march(kw, wall_bc=False, periods=periods)
    wall = march(kw, wall_bc=True, periods=periods)
    if clamp["status"] == "aborted" and wall["status"] == "completed":
        cause = ("the round-1 abort is a boundary artefact: with the same CSF "
                 "model and the same horizon, static wall particles carry the "
                 "run to the end while the bare clamp box loses a particle")
    elif clamp["status"] == "aborted" and wall["status"] == "aborted":
        cause = ("the wall boundary condition does not rescue the case; the "
                 "spurious force on the open free surface alone destroys it")
    elif clamp["status"] == "completed" and wall["status"] == "completed":
        cause = ("both boundary treatments complete: the round-1 abort has to be "
                 "re-measured and the round-1 page rewritten")
    else:
        cause = "the wall run aborts where the clamp run completes: investigate"
    return {"clamp_box_no_wall": clamp, "static_wall_particles": wall,
            "conclusion": cause}


# --------------------------------------------------------------------------- #
# 3. the curvature noise floor
# --------------------------------------------------------------------------- #
def curvature_floor(kw: dict = CASE) -> dict:
    """Curvature of a *flat* interface: the floor any wave signal must clear.

    A perfect ``eta = const`` surface has ``kappa = 0``, so every non-zero value
    here is estimator noise.  The wave signal is ``k^2 A``; the table reports
    both so the required amplitude follows from arithmetic, not from a fit.
    """
    k = 2.0 * math.pi / kw["wavelength"]
    width = kw["wavelength"] / 2.0
    cells = []
    for ppw in FLOOR_PPW:
        flat = dict(kw, dx=kw["wavelength"] / ppw, amp_over_dx=0.0)
        pos, mass, cfg = A.build(flat)
        h = cfg.h_factor * cfg.dx
        cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                              fluid_count=pos.shape[0])
        pairs, _ = cache.get(pos)
        state = csf_fields(pos, mass, pairs, h, kw["rho0"])
        shell = state.delta_s > 0.5 * state.delta_s.max()
        x = pos[:, 0]
        inner = shell & (x > 3.0 * h) & (x < width - 3.0 * h)
        kappa = state.kappa[shell]
        kappa_inner = state.kappa[inner]
        row = {
            "points_per_wavelength": ppw,
            "dx_m": cfg.dx,
            "n_interface_particles": int(shell.sum()),
            "n_interior_interface_particles": int(inner.sum()),
            "kappa_flat_mean_1_per_m": float(kappa.mean()),
            "kappa_flat_rms_1_per_m": float(torch.sqrt((kappa ** 2).mean())),
            "kappa_flat_abs_p95_1_per_m": float(torch.quantile(kappa.abs(), 0.95)),
            "kappa_flat_interior_mean_1_per_m": (float(kappa_inner.mean())
                                                 if kappa_inner.numel() else None),
            "kappa_flat_interior_rms_1_per_m": (float(torch.sqrt(
                (kappa_inner ** 2).mean())) if kappa_inner.numel() else None),
            "signals": {},
        }
        for amp_over_dx in FLOOR_AMP_OVER_DX:
            amp = amp_over_dx * cfg.dx
            signal = k * k * amp
            row["signals"][f"A_over_dx={amp_over_dx:g}"] = {
                "amplitude_m": amp,
                "kappa_signal_k2A_1_per_m": signal,
                "floor_over_signal": row["kappa_flat_rms_1_per_m"] / signal,
            }
        cells.append(row)
    return {"k_1_per_m": k, "cells": cells}


def amplitude_criterion(floor: dict) -> dict:
    """Smallest ``A/dx`` whose signal clears the measured curvature floor.

    Uses the **interior** floor where it exists (it is the free-surface number,
    free of the clamp walls); falls back to the all-shell floor at coarse
    resolution where the interior sample is too small to trust.
    """
    k = floor["k_1_per_m"]
    picked = []
    for cell in floor["cells"]:
        usable = cell["n_interior_interface_particles"] >= 20
        rms = (cell["kappa_flat_interior_rms_1_per_m"] if usable
               else cell["kappa_flat_rms_1_per_m"])
        need = rms / (k * k * cell["dx_m"])       # A/dx for floor == signal
        picked.append({"points_per_wavelength": cell["points_per_wavelength"],
                       "floor_rms_used_1_per_m": rms,
                       "floor_source": "interior" if usable else "all_shell",
                       "required_A_over_dx_for_parity": need,
                       "required_A_over_dx_for_10x": 10.0 * need})
    worst = max(p["required_A_over_dx_for_10x"] for p in picked)
    return {"cells": picked,
            "required_A_over_dx_10x_snr": worst,
            "note": ("A/dx multiplies the signal but not the floor, so refining "
                     "the lattice at fixed A/dx does not improve the ratio")}


# --------------------------------------------------------------------------- #
# 4. the march ladder
# --------------------------------------------------------------------------- #
def ladder(profile: dict) -> list[dict]:
    rows = []
    for ppw in profile["ppw"]:
        for amp_over_dx in profile["amp_over_dx"]:
            for renorm in profile["renormalization"]:
                over = {"density_renormalization": True} if renorm else {}
                row = march(_case(ppw, amp_over_dx),
                            wall_bc=True, periods=profile["periods"], **over)
                row.update({
                    "variant": (f"ppw={ppw}/A={amp_over_dx:g}dx"
                                + ("/renorm" if renorm else "")),
                    "points_per_wavelength": ppw,
                    "amplitude_over_dx": amp_over_dx,
                    "density_renormalization": bool(renorm),
                    "steepness_kA": 2.0 * math.pi * amp_over_dx / ppw,
                })
                row["graded"] = bool(
                    row["status"] == "completed"
                    and row["admissible_adami"]
                    and abs(row["omega_relative_error_pct"])
                    < THRESHOLDS["dispersion_rel_error_pct"]
                    and row["fit_residual_rel"] < THRESHOLDS["fit_residual_rel"])
                rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": torch.get_num_threads(),
        "deterministic": True,
        "random_seed": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--profile", default="full", choices=sorted(PROFILES))
    args = ap.parse_args()
    profile = PROFILES[args.profile]

    t0 = time.perf_counter()
    print(f"[1/4] force budget ({args.profile}) ...")
    budget = force_budget()
    print(f"      wall-adjacent share of |a| = {budget['wall_share']:.2f}, "
          f"a_max/omega^2A = {budget['a_max_over_physical']:.0f}x")

    print("[2/4] wall boundary condition: clamp box vs static wall particles ...")
    bcs = wall_bc_comparison(periods=profile["periods"])
    for name, row in (("clamp_no_wall", bcs["clamp_box_no_wall"]),
                      ("static_wall", bcs["static_wall_particles"])):
        if row["status"] == "aborted":
            print(f"      {name:14s} aborted at step {row['escape_step']}, "
                  f"v/u_ref = {row['v_max_over_u_ref']:.1f}")
        else:
            print(f"      {name:14s} completed, v/u_ref = "
                  f"{row['v_max_over_u_ref']:.1f}, omega error "
                  f"{row['omega_relative_error_pct']:+.1f}%")

    print("[3/4] curvature noise floor on a flat interface ...")
    floor = curvature_floor()
    crit = amplitude_criterion(floor)
    for cell in floor["cells"]:
        print(f"      ppw={cell['points_per_wavelength']:3d} "
              f"kappa_flat mean {cell['kappa_flat_mean_1_per_m']:+.1f} "
              f"rms {cell['kappa_flat_rms_1_per_m']:.1f} 1/m "
              f"(interior n={cell['n_interior_interface_particles']})")
    print(f"      required A/dx for 10x SNR: {crit['required_A_over_dx_10x_snr']:.1f}")

    print(f"[4/4] march ladder ({args.profile}) ...")
    rows = ladder(profile)
    for row in rows:
        if row["status"] == "aborted":
            print(f"      {row['variant']:26s} aborted at step "
                  f"{row['escape_step']}, v/u_ref {row['v_max_over_u_ref']:.1f}")
        else:
            print(f"      {row['variant']:26s} omega error "
                  f"{row['omega_relative_error_pct']:+6.2f}%, resid "
                  f"{row['fit_residual_rel']:.2f}, v/u_ref "
                  f"{row['v_max_over_u_ref']:5.2f}"
                  + ("  -> GRADED" if row["graded"] else ""))

    graded = [r for r in rows if r["graded"]]
    admissible_only = [r for r in rows if r.get("admissible_adami")]
    completed = [r for r in rows if r["status"] == "completed"]
    if graded:
        verdict = {
            "grade": "REAL",
            "reason": (f"{len(graded)} ladder cell(s) reproduce Lamb within "
                       f"{THRESHOLDS['dispersion_rel_error_pct']:g} % while "
                       "staying inside the Adami spurious-current limit"),
            "graded_variants": [r["variant"] for r in graded],
        }
    elif completed:
        best = min(completed, key=lambda r: abs(r["omega_relative_error_pct"]))
        verdict = {
            "grade": "WITHHELD",
            "reason": ("No ladder cell satisfies both predeclared criteria "
                       "(3 % dispersion AND Adami admissibility); the closest "
                       f"completed cell {best['variant']} reports "
                       f"{best['omega_relative_error_pct']:+.1f} % with "
                       f"v_max/(A omega) = {best['v_max_over_u_ref']:.1f} "
                       f"(limit {THRESHOLDS['adami_spurious_current']:g})"),
            "closest_cell": best["variant"],
        }
    else:
        verdict = {"grade": "WITHHELD",
                   "reason": "Every ladder cell aborted; nothing was measured"}

    payload = {
        "benchmark": "capillary_wave_resolution_criterion",
        "question": ("which of the two candidate causes destroys the capillary "
                     "standing wave at HEAD -- the CSF force on the open free "
                     "surface or the wall treatment -- and what resolution or "
                     "amplitude would make the Lamb (1932) dispersion claim "
                     "gradable?"),
        "references": {
            "dispersion": ("Lamb (1932) Hydrodynamics 6th ed. Art. 267/348: "
                           "omega^2=(g k + sigma k^3/rho) tanh(kH), g=0 here"),
            "admissibility": ("Adami, Ihmsen & Adams (2012) JCP 231:7057: the "
                              "spurious current must stay below 10 % of "
                              "u_ref = A omega"),
            "wall_bc": ("Adami, Hu & Adams (2012) JCP 231:7057-7075: static "
                        "wall particles restore the kernel support a fluid "
                        "particle misses at a wall"),
            "csf": "Brackbill, Kothe & Zemach (1992) JCP 100:335",
        },
        "predeclared": {
            "thresholds": THRESHOLDS,
            "horizon_periods": profile["periods"],
            "samples_per_period": SAMPLES_PER_PERIOD,
            "stopping_rule": "fixed horizon or first boundary escape",
            "graded_rule": ("|d omega|/omega < 3 % AND fit residual < 0.2 AND "
                            "v_max/(A omega) < 0.1"),
            "profile": args.profile,
            "tuned_after_results": False,
        },
        "environment": _environment(),
        "case": _case(CASE["wavelength"] / CASE["dx"], CASE["amp_over_dx"]),
        "force_budget": budget,
        "wall_bc": bcs,
        "curvature_floor": floor,
        "amplitude_criterion": crit,
        "ladder": rows,
        "summary": {
            "n_cells": len(rows),
            "n_aborted": sum(1 for r in rows if r["status"] == "aborted"),
            "n_completed": len(completed),
            "n_admissible": len(admissible_only),
            "n_graded": len(graded),
            "wall_share_of_abs_a": budget["wall_share"],
            "wall_bc_rescues_case": (bcs["clamp_box_no_wall"]["status"] == "aborted"
                                     and bcs["static_wall_particles"]["status"]
                                     == "completed"),
            "wall_clock_s": time.perf_counter() - t0,
        },
        "verdict": verdict,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resolution_criterion.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"report -> {path}  ({payload['summary']['wall_clock_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
