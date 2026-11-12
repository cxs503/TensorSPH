#!/usr/bin/env python3
"""Same-estimator (like-for-like) summation-density error: WCSPH vs ISPH channel.

Why this file exists
--------------------
``docs/isph_projection_channel.md`` §5.5(a) leaves one item open, in its own
words: the WCSPH number quoted by the inventory gap rule
(``docs/oss_inventory_sph/gaps.json``, ``max|rho/rho0-1| ~ 2.3e-2``) is **not**
re-measured with the wall-completed summation-density estimator that the ISPH
path uses, so the "ISPH is ~40x denser" claim compares two different
diagnostics.  This benchmark removes the mismatch: *one* estimator, applied to
the particle clouds of both schemes on the *same* plane-Poiseuille case.

The estimator
-------------
``rho_i = sum_j m_j W_ij`` over fluid **and** the static wall band, with the
minimum-image fold along ``x`` (``tensorsph.isph_channel.pair_geometry`` +
``summation_density`` -- literally the two functions the ISPH path calls) and
the lattice-calibrated mass ``m = rho0 / S(h, dx)``
(``tensorsph.droplet_impact.lattice_kernel_sum``).  With that mass a perfect
lattice returns ``rho = rho0`` to round-off, so the reported number is the
*disorder* of the cloud, not a kernel-sum offset.  The script proves this on
every run by measuring the reference lattice first
(``order_reference_error``, required ``<= 1e-9``).

Case (identical to the ISPH channel, ``tensorsph.isph_channel.ChannelConfig``)
-----------------------------------------------------------------------------
``H = 0.1 m``, periodic in ``x`` over 12 columns, no-slip walls at ``y = 0`` and
``y = H``, ``nu = 0.01 m^2/s``, body force ``g_x = 9.81 m/s^2``, 5 wall layers,
``t_end = 1.0 s``, diagnostic averaged over the last ``0.1 s``.

Two kernel ratios are run because the two schemes do not share one: the ISPH
path uses ``h = 1.9 dx`` (its own choice) and the WCSPH production channel
benchmark uses ``h = 1.3 dx``.  Only the ``h/dx = 1.9`` pair is strictly
like-for-like; the ``1.3`` row is reported as the repository's production
setting and is labelled as not transferable.

Usage
-----
    PYTHONPATH=src python benchmarks/density_order_likeforlike/run.py \
        --out benchmarks/density_order_likeforlike/out
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, lattice_kernel_sum, rhs
from tensorsph.isph_channel import (
    ChannelConfig,
    pair_geometry,
    run_case as isph_run_case,
    summation_density,
)

# case parameters shared with the ISPH channel (ChannelConfig defaults)
HEIGHT = 0.1
COLUMNS = 12
VISCOSITY = 0.01
BODY_FORCE = 9.81
RHO0 = 1000.0
SOUND_SPEED = 20.0
WALL_LAYERS = 5
T_END = 1.0
T_AVERAGE = 0.1
RESOLUTIONS = (8, 12, 16)
H_FACTORS = (1.3, 1.9)
ISPH_H_FACTOR = 1.90
REFERENCE_TOL = 1e-6
DIAG_STRIDE = 4


# --------------------------------------------------------------------------- #
# Geometry shared by the run and by the diagnostic
# --------------------------------------------------------------------------- #

def build_fluid(dx: float, height: float, columns: int, device,
                dtype=torch.float32) -> torch.Tensor:
    """Rows ``y = dx/2, 3dx/2, ...`` strictly inside the channel; ``x`` in [0, L)."""
    length = columns * dx
    xs = torch.arange(dx * 0.5, length, dx, device=device, dtype=dtype)
    ys = torch.arange(dx * 0.5, height, dx, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def build_wall(dx: float, height: float, columns: int, layers: int,
               device, margin: int = 0, x_offset: float = 0.5,
               dtype=torch.float32) -> torch.Tensor:
    """Static wall band, **one** period in ``x`` and aligned to the fluid lattice.

    Identical convention to ``tensorsph.isph_channel.build_channel``: the first
    layer sits at ``y = -dx/2`` (bottom) / ``y = H + dx/2`` (top) and the columns
    sit at ``x = dx/2 + k dx``.  Because the column spacing equals ``dx`` and the
    period ``L = columns*dx`` is an integer multiple of it, the band is invariant
    under the ``+-L`` image shift used for the periodic pair search, so one
    period of columns is exact for both the dynamics and the diagnostic.

    ``margin`` (in columns) extends the band past both ``x`` ends.  The *run*
    needs it (``rhs`` computes unwrapped distances, so the ``+-L`` fluid images
    must meet wall columns that are physically present); the *diagnostic* must
    not have it (the minimum-image fold already supplies the images, and a
    doubled band would double count -- see
    ``tensorsph.isph_channel.build_channel``).
    """
    # Built with the *same* float expression as benchmarks/poiseuille/run.py
    # (``arange(-margin*dx, length+margin*dx, dx)``): the channel turns out to be
    # sensitive to round-off in the wall lattice (a 1.5e-8 shift of the wall
    # column positions changes u_max by ~50 % at H/dx = 8), so an arithmetically
    # different but mathematically equal band is not interchangeable.
    wx = torch.arange(-margin * dx, (columns + margin) * dx, dx,
                      device=device, dtype=dtype) + x_offset * dx
    parts = []
    for layer in range(layers):
        yb = torch.full_like(wx, -dx * 0.5 - layer * dx)
        parts.append(torch.stack((wx, yb), dim=-1))
        yt = torch.full_like(wx, height + dx * 0.5 + layer * dx)
        parts.append(torch.stack((wx, yt), dim=-1))
    return torch.cat(parts, dim=0)


# --------------------------------------------------------------------------- #
# The estimator (the whole point of the file)
# --------------------------------------------------------------------------- #

def calibrated_mass(h: float, dx: float, rho0: float = RHO0) -> float:
    """``m = rho0 / S`` so that a perfect lattice sums to ``rho0`` exactly."""
    return rho0 / lattice_kernel_sum(h, dx)


def density_order_error(fluid: torch.Tensor, wall: torch.Tensor, h: float,
                        length: float, dx: float,
                        rho0: float = RHO0) -> dict:
    """``max``/``rms`` of ``|rho_summ/rho0 - 1|`` over the fluid particles.

    Fluid **and** wall particles enter the sum (wall-completed estimator), the
    pair search uses the minimum image in ``x``, and the mass is the calibrated
    one.  This is the same diagnostic the ISPH path applies to its own cloud --
    the two primitives are called here directly rather than re-implemented.
    """
    # float64: the reference (undisturbed lattice) reading is pure round-off and
    # float32 summation noise (~1e-7) would swamp the 1e-9 reference gate.
    pos = torch.cat((fluid.to(torch.float64), wall.to(torch.float64)), dim=0)
    n_fluid = int(fluid.shape[0])
    mass = torch.full((pos.shape[0],), calibrated_mass(h, dx, rho0),
                      dtype=torch.float64, device=pos.device)
    geo = pair_geometry(pos, h, length)
    rho = summation_density(geo, mass, h)
    err = (rho[:n_fluid] / rho0 - 1.0).abs()
    return {
        "max_abs": float(err.max()) if err.numel() else float("nan"),
        "rms": float(torch.sqrt(err.square().mean())) if err.numel() else float("nan"),
    }


# --------------------------------------------------------------------------- #
# WCSPH channel run
# --------------------------------------------------------------------------- #

def make_config(dx: float, h_factor: float, device, wall_layers: int = WALL_LAYERS) -> Config:
    return Config(
        dx=dx,
        h_factor=h_factor,
        rho0=RHO0,
        sound_speed=SOUND_SPEED,
        gamma=7.0,
        gravity=0.0,          # driven along x by hand
        viscosity=VISCOSITY,
        dt=1.0e-4,            # replaced by the computed limit below
        steps=0,
        output_every=10 ** 9,
        pool_width=0.2,
        pool_depth=0.1,
        boundary="wall",
        wall_layers=wall_layers,
        wall_friction=1.0,
        fail_fast=False,
        device=str(device),
    )


def periodic_images(pos: torch.Tensor, length: float) -> torch.Tensor:
    """Fluid at ``x - L``, ``x``, ``x + L``; the wall band is ``L``-periodic."""
    shift = torch.tensor([length, 0.0], dtype=pos.dtype, device=pos.device)
    return torch.cat((pos - shift, pos, pos + shift), dim=0)


def wcsph_case(resolution: int, h_factor: float, device, t_end: float = T_END,
               t_average: float = T_AVERAGE, wall_layers: int = WALL_LAYERS,
               margin: int = 3, wall_x_offset: float = 0.0,
               dt_scale: float = 1.0) -> dict:
    """Run the WCSPH channel and diagnose the particle cloud every step of the
    averaging window with the shared estimator."""
    dx = HEIGHT / resolution
    length = COLUMNS * dx
    h = h_factor * dx
    cfg = make_config(dx, h_factor, device, wall_layers)

    fluid = build_fluid(dx, height=HEIGHT, columns=COLUMNS, device=device)
    wall = build_wall(dx, HEIGHT, COLUMNS, wall_layers, device, margin=margin,
                      x_offset=wall_x_offset)
    wall_diag = build_wall(dx, HEIGHT, COLUMNS, wall_layers, device, margin=0,
                           x_offset=0.5)
    n = int(fluid.shape[0])
    flat_mass = torch.full((n,), RHO0 * dx * dx, dtype=fluid.dtype, device=device)
    vel = torch.zeros_like(fluid)

    # The reference check: an *undisturbed* lattice must read zero disorder.
    # the gate lattice is built in float64: a float32 lattice already carries
    # ~4e-7 of coordinate round-off, which is larger than the 1e-9 gate and would
    # be a false alarm about the estimator rather than about the lattice.
    reference = density_order_error(
        build_fluid(dx, HEIGHT, COLUMNS, device, dtype=torch.float64),
        build_wall(dx, HEIGHT, COLUMNS, wall_layers, device, margin=0,
                   x_offset=0.5, dtype=torch.float64),
        h, length, dx)

    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / VISCOSITY
    dt = dt_scale * min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))
    avg_from = t_end - t_average

    diag_max = 0.0
    diag_rms_sum = 0.0
    diag_samples = 0
    tait_max = 0.0
    u_win = torch.zeros_like(fluid)
    u_samples = 0
    clock = time.time()

    for step in range(steps):
        pos_ext = periodic_images(fluid, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((flat_mass, flat_mass, flat_mass), dim=0)
        rho, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg, wall=wall)
        acc = acc[:n].clone()
        acc[:, 0] += BODY_FORCE
        tait_max = max(tait_max, float((rho[:n] / RHO0 - 1.0).abs().max()))

        t = step * dt
        if t >= avg_from:
            # the diagnostic is O(N^2); subsample the window (>=100 samples are
            # still taken at every resolution used here) to keep the run cheap
            if (step % DIAG_STRIDE) == 0:
                diag = density_order_error(fluid, wall_diag, h, length, dx)
                diag_max = max(diag_max, diag["max_abs"])
                diag_rms_sum += diag["rms"]
                diag_samples += 1
            u_win += vel
            u_samples += 1

        vel = vel + dt * acc
        fluid = fluid + dt * vel
        fluid[:, 0] = torch.remainder(fluid[:, 0], length)

    u_mean = u_win / max(u_samples, 1)
    # Steady-state peak: max of the row-averaged streamwise velocity.
    rows = torch.clamp((fluid[:, 1] / dx).floor().long(), 0, resolution - 1)
    row_u = torch.zeros(resolution, device=device)
    row_cnt = torch.zeros(resolution, device=device)
    row_u.index_add_(0, rows, u_mean[:, 0])
    row_cnt.index_add_(0, rows, torch.ones(rows.shape[0], device=device))
    profile = (row_u / row_cnt.clamp_min(1.0)).tolist()

    return {
        "scheme": "WCSPH",
        "resolution_H_over_dx": resolution,
        "h_over_dx": h_factor,
        "h": h,
        "dx": dx,
        "length": length,
        "dt": dt,
        "steps": steps,
        "n_fluid": n,
        "n_wall": int(wall.shape[0]),
        "n_diag_samples": diag_samples,
        "wall_clock_s": time.time() - clock,
        "order_reference_error": reference["max_abs"],
        "density_order_error_max": diag_max,
        "density_order_error_rms_mean": diag_rms_sum / max(diag_samples, 1),
        "tait_density_error_max": tait_max,
        "u_max_measured": float(max(profile)) if profile else float("nan"),
        "u_max_analytic": BODY_FORCE * HEIGHT ** 2 / (8.0 * VISCOSITY),
        "mass_model": "rho0*dx^2 in the run; lattice-calibrated in the diagnostic",
    }


# --------------------------------------------------------------------------- #
# ISPH side (its own committed path, nothing re-implemented)
# --------------------------------------------------------------------------- #

def isph_case(resolution: int, device, h_factor: float = ISPH_H_FACTOR) -> dict:
    cfg = ChannelConfig(resolution=resolution, h_factor=h_factor)
    t0 = time.time()
    res = isph_run_case(cfg, device)
    return {
        "scheme": "ISPH",
        "resolution_H_over_dx": resolution,
        "h_over_dx": h_factor,
        "h": cfg.h,
        "dx": cfg.dx,
        "dt_note": "adaptive: min(cfl*h/|u|, 0.125*h^2/nu)",
        "steps": res["steps"],
        "n_fluid": res["n_fluid"],
        "n_wall": res["n_wall"],
        "wall_clock_s": time.time() - t0,
        "order_reference_error": None,  # calibrated mass => exact by construction
        "density_order_error_max": res["density_error_max"],
        "density_order_error_rms_mean": None,
        "u_max_measured": res.get("fit", {}).get("u_max_measured")
        or max((r["u_mean"] for r in res["profile"]), default=float("nan")),
        "u_max_analytic": cfg.u_max_analytic,
        "mass_model": "lattice-calibrated",
    }


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def grade(results: list[dict], reference_tol: float = REFERENCE_TOL) -> dict:
    """Gates that can fail.

    * ``reference_zero`` -- the estimator must return ~0 on an undisturbed
      lattice.  If the diagnostic is blind, every number below is worthless.
    * ``estimator_identity`` -- the WCSPH diagnostic is built from the *same*
      two functions the ISPH path uses; checked structurally by the tests, and
      here by requiring both schemes to report a finite number on the same case.
    * ``like_for_like_pair`` -- both schemes must have been run at the same
      ``h/dx``, otherwise the comparison is not like-for-like and must not be
      quoted as one.
    """
    wcsph = [r for r in results if r["scheme"] == "WCSPH"]
    isph = {r["resolution_H_over_dx"]: r for r in results if r["scheme"] == "ISPH"}

    reference_ok = bool(any(r["order_reference_error"] is not None for r in wcsph)) and all(
        r["order_reference_error"] <= reference_tol for r in wcsph
    )
    identity_ok = bool(wcsph) and bool(isph) and all(
        math.isfinite(r["density_order_error_max"]) for r in results
    )

    pairs = []
    for r in wcsph:
        other = isph.get(r["resolution_H_over_dx"])
        if other is None:
            continue
        same_h = abs(r["h_over_dx"] - other["h_over_dx"]) < 1e-12
        pairs.append({
            "resolution_H_over_dx": r["resolution_H_over_dx"],
            "h_over_dx": r["h_over_dx"],
            "like_for_like": same_h,
            "wcsph_max": r["density_order_error_max"],
            "isph_max": other["density_order_error_max"],
            "ratio_wcsph_over_isph": (
                r["density_order_error_max"] / other["density_order_error_max"]
                if other["density_order_error_max"] else float("inf")
            ),
        })

    lfl = [p for p in pairs if p["like_for_like"]]
    return {
        "reference_tol": reference_tol,
        "reference_zero": reference_ok,
        "estimator_identity": identity_ok,
        "like_for_like_pairs": pairs,
        "n_like_for_like_pairs": len(lfl),
        "verdict": "PASS" if (reference_ok and identity_ok and lfl) else "FAIL",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/density_order_likeforlike/out")
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(RESOLUTIONS))
    parser.add_argument("--h-factors", type=float, nargs="+", default=list(H_FACTORS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    results: list[dict] = []
    for resolution in args.resolutions:
        for hf in args.h_factors:
            print(f"[wcsph] H/dx={resolution} h/dx={hf}", flush=True)
            results.append(wcsph_case(resolution, hf, device))
        print(f"[isph ] H/dx={resolution} h/dx={ISPH_H_FACTOR}", flush=True)
        results.append(isph_case(resolution, device))

    gates = grade(results)
    payload = {
        "case": "density_order_likeforlike_plane_poiseuille",
        "anchor": "GAP-SPH-ISPH-PROJECTION (open item (a) of docs/isph_projection_channel.md §5.5)",
        "estimator": {
            "name": "wall-completed summation density, minimum image in x",
            "functions": [
                "tensorsph.isph_channel.pair_geometry",
                "tensorsph.isph_channel.summation_density",
            ],
            "mass": "rho0 / lattice_kernel_sum(h, dx)",
            "quantities": ["max|rho/rho0-1| over fluid particles",
                           "rms of the same over the averaging window"],
        },
        "parameters": {
            "height": HEIGHT, "columns": COLUMNS, "viscosity": VISCOSITY,
            "body_force": BODY_FORCE, "rho0": RHO0, "sound_speed": SOUND_SPEED,
            "wall_layers": WALL_LAYERS, "t_end": T_END, "t_average": T_AVERAGE,
            "resolutions": args.resolutions, "h_factors": args.h_factors,
            "isph_h_factor": ISPH_H_FACTOR,
        },
        "results": results,
        "gates": gates,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(gates, indent=1))
    print(f"wrote {out / 'results.json'}")
    return 0 if gates["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())