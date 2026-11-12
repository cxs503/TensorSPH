#!/usr/bin/env python3
"""Controlled audit of the density-diffusion (delta-SPH) term in the 3-D WCSPH path.

Question
--------
``ledger.entry`` ``threed_hydrostatic.profile.max_rel_error_core.adami_best``
grades FAIL (43 % worst core layer).  The per-row pressure field in the shipped
artifact alternates layer to layer (row 9: -11 %, row 10: +1 %, row 11: -17 %,
row 12: +4 % at ``adami-c0_20-nz16``), i.e. it carries an odd-even mode that the
symmetric SPH gradient cannot damp on a regular lattice.  The 2-D solver damps it
with the diffusive density term of :mod:`tensorsph.density_diffusion`; the 3-D
operator set had no such term until this change.  This script measures whether
adding it moves the profile error, holding every other parameter fixed.

Protocol (pre-declared)
-----------------------
* Same lattice, same wall model (``adami``), same body force, same viscosity,
  same time step rule, same averaging window for every delta.
* ``delta = 0.0`` is the *control*: it must reproduce the pre-change numbers of
  the same configuration bit-for-bit (the term defaults to off).
* Reported meters (per run): bottom pressure error, worst/mean core layer error
  on both binning schemes, odd-even amplitude, depth-integrated load error,
  worst ``|rho/rho0 - 1|``.
* Verdict rule: the term is credited only if the worst core-layer error on the
  *lattice* rows (the binning-invariant one) falls by >= 5 percentage points at
  some delta without raising the bottom-pressure error above 15 %.

Outputs: one JSON per run plus ``summary.json`` in ``--out``.  Figure and the
report are generated from those files, never from hand-typed numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.threed import acoustic_dt, run_hydrostatic  # noqa: E402


def run_one(*, sound_speed: float, delta: float, nz: int, nx: int, ny: int,
            dx: float, viscosity: float, t_end: float,
            form: str = "antuono") -> dict:
    h = 1.3 * dx
    dt = acoustic_dt(dx, h, sound_speed)
    steps = int(round(t_end / dt))
    t0 = time.time()
    result = run_hydrostatic(nx=nx, ny=ny, nz=nz, dx=dx, sound_speed=sound_speed,
                             viscosity=viscosity, wall_model="adami",
                             steps=steps, dt=dt,
                             sample_every=max(1, steps // 40),
                             settle_fraction=0.5,
                             density_diffusion=delta,
                             density_diffusion_form=form)
    profile = result.profile
    scale = profile["rho0"] * result.gravity * result.height
    bottom = profile["rows"][0]
    rho_excess = max((abs(r["rho_measured"] / profile["rho0"] - 1.0)
                      for r in profile["rows"] if r.get("p_analytic")))
    out = {
        "case": f"adami-c0_{sound_speed:g}-nz{nz}-delta{delta:g}",
        "sound_speed": sound_speed,
        "density_diffusion": delta,
        "density_diffusion_form": form,
        "nz": nz, "nx": nx, "ny": ny, "dx": dx, "h": h,
        "viscosity": viscosity,
        "dt": dt, "steps": steps, "t_end": steps * dt,
        "fluid_particles": result.fluid_particles,
        "wall_particles": result.wall_particles,
        "bottom_rel_error": bottom["p_rel_error"],
        "max_rel_error_core": profile["max_rel_error_core"],
        "mean_rel_error_core": profile["mean_rel_error_core"],
        "max_rel_error_core_lattice": profile["max_rel_error_core_lattice"],
        "mean_rel_error_core_lattice": profile["mean_rel_error_core_lattice"],
        "layers_within_15pct_lattice": profile["layers_within_15pct_lattice"],
        "core_layers_lattice": profile["core_layers_lattice"],
        "odd_even_amplitude_over_scale_lattice": profile[
            "odd_even_amplitude_over_scale_lattice"],
        "integral_rel_error": profile["integral_rel_error"],
        "max_abs_error_over_rho0_g_H": profile["max_abs_error_over_scale"],
        "abs_error_scale_Pa": scale,
        "max_rho_excess": rho_excess,
        "wall_clock_s": time.time() - t0,
        "profile_lattice": profile["rows_by_lattice"],
        "profile_eulerian": profile["rows"],
    }
    if profile["odd_even_amplitude_over_scale_lattice"] is not None:
        out["odd_even_amplitude_over_scale_eulerian"] = profile[
            "odd_even_amplitude_over_scale"]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound-speeds", default="20")
    parser.add_argument("--deltas", default="0,0.1,0.2")
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--viscosity", type=float, default=0.02)
    parser.add_argument("--t-end", type=float, default=0.3)
    parser.add_argument("--form", default="antuono")
    parser.add_argument("--out", default=str(REPO / "runs" / "threed_delta_sph"))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for c0 in (float(x) for x in args.sound_speeds.split(",") if x):
        for delta in (float(x) for x in args.deltas.split(",") if x):
            row = run_one(sound_speed=c0, delta=delta, nz=args.nz, nx=args.nx,
                          ny=args.ny, dx=args.dx, viscosity=args.viscosity,
                          t_end=args.t_end, form=args.form)
            (out_dir / f"{row['case']}.json").write_text(json.dumps(row, indent=1))
            runs.append(row)
            print(f"{row['case']:28s} bottom={row['bottom_rel_error']:+.4f} "
                  f"max_core={row['max_rel_error_core']:+.4f} "
                  f"max_core_lat={row['max_rel_error_core_lattice']:+.4f} "
                  f"oe={row['odd_even_amplitude_over_scale_lattice']:.4f} "
                  f"integral={row['integral_rel_error']:+.4f} "
                  f"({row['wall_clock_s']:.1f}s)", flush=True)

    summary = {
        "case": "threed_hydrostatic_delta_sph_audit",
        "protocol": {
            "control": "delta = 0.0 (term off, the pre-change behaviour)",
            "fixed": {"wall_model": "adami", "viscosity": args.viscosity,
                      "dx": args.dx, "nz": args.nz, "nx": args.nx,
                      "ny": args.ny, "t_end": args.t_end,
                      "form": args.form},
            "credit_rule": ("worst core-layer error on the lattice rows falls by "
                            ">= 0.05 absolute at some delta while the bottom "
                            "pressure error stays within 15 %"),
            "environment": {
                "torch": torch.__version__,
                "device": "cpu",
                "threads": torch.get_num_threads(),
                "dtype": "float64",
            },
        },
        "runs": runs,
    }
    by_speed: dict[str, dict] = {}
    for row in runs:
        control = next(r for r in runs
                       if r["sound_speed"] == row["sound_speed"]
                       and r["density_diffusion"] == 0.0)
        entry = by_speed.setdefault(f"c0={row['sound_speed']:g}", {})
        entry[row["case"]] = {
            "delta": row["density_diffusion"],
            "max_rel_error_core_lattice": row["max_rel_error_core_lattice"],
            "delta_vs_control": (row["max_rel_error_core_lattice"]
                                 - control["max_rel_error_core_lattice"]),
            "odd_even_amplitude_over_scale_lattice": row[
                "odd_even_amplitude_over_scale_lattice"],
            "bottom_rel_error": row["bottom_rel_error"],
        }
    summary["comparison"] = by_speed
    credited = any(
        v["delta_vs_control"] is not None and v["delta_vs_control"] <= -0.05
        and abs(v["bottom_rel_error"]) <= 0.15
        for group in by_speed.values() for v in group.values())
    summary["verdict"] = "CREDITED" if credited else "NOT_CREDITED"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("verdict:", summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
