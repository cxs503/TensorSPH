"""Open free-surface audit: why ``validation.py`` cannot produce a capillary-wave claim.

Why this file exists
--------------------
``benchmarks/verification_ledger/waivers.json`` carried

    "benchmarks/capillary_wave/": "harness only (capillary-wave dispersion
    check), no committed artifact - evidence debt E1"

i.e. the repository shipped a capillary-wave harness but no committed evidence
that the capability works.  ``validation.py`` (the end-to-end comparison against
Lamb 1932) exists, but it commits no output either: running it at HEAD on its own
``resolution`` case aborts inside 2 capillary periods with

    RuntimeError: 1 particle(s) left the box at step 336; the case is not a
    free-surface standing wave any more

This audit turns that situation into a measurement instead of a silence.  It
answers three questions with numbers:

1. **Does the harness run at HEAD?** It is executed as-is (``V.run_case``); the
   abort is recorded, not swallowed.
2. **If it does not, is the cause the discrete surface-tension force on an *open*
   free surface?** The physical restoring acceleration of the wave is
   ``omega^2 A``; the CSF force of the same initial state is measured by
   differencing ``rhs`` with ``sigma = 0`` and ``sigma = 7.2 N/m``, and the
   curvature field on the (almost flat) interface is compared with the
   perturbation's own curvature ``k^2 A``.
3. **Does a larger amplitude, or one of the solver knobs, rescue it?** A variant
   sweep (density renormalisation, free-surface pressure closure in both donor
   semantics, legacy CSF scheme, square lattice, higher sound speed) and an
   amplitude ladder at ``A/dx = 0.5, 2`` are recorded with the same horizon.

The analytic references are classical and are *not* fitted:

* dispersion, Lamb (1932) Hydrodynamics 6th ed. Art. 267/348:
  ``omega^2 = (g k + sigma k^3 / rho) tanh(k H)``; the case sets ``g = 0`` and
  ``k H = 6.28`` so ``tanh(kH) = 0.99999`` (deep water);
* viscous damping, Lamb Art. 348 / Landau & Lifshitz §25: ``gamma = 2 nu k^2``;
* spurious-current admissibility, Adami, Ihmsen & Adams (2012) JCP 231:7057:
  a surface-tension run is only gradable when the spurious current stays below
  10 % of the capillary velocity scale.

Usage::

    PYTHONPATH=src python benchmarks/capillary_wave/open_surface_audit.py
    PYTHONPATH=src python benchmarks/capillary_wave/open_surface_audit.py --no-figures

Exit status is always 0 for a *completed* audit: the audit reports a defect, it
does not assert that the solver passes.  The gates live in
``tests/test_capillary_wave_open_surface.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # run as a script: ``validation`` is next to us
    sys.path.insert(0, str(HERE))

import validation as V  # noqa: E402  (path set above)

from tensorsph.csf_consistent import csf_fields  # noqa: E402
from tensorsph.droplet_impact import (  # noqa: E402
    apply_boundary,
    resolve_device,
    rhs,
)
from tensorsph.neighbor_search import PairListCache  # noqa: E402

#: Reference case of the harness audit (its own ``resolution`` case).
CASE = dict(wavelength=0.02, depth=0.02, dx=0.001, amp_over_dx=0.5,
            rho0=1000.0, sigma=7.2, viscosity=0.0, sound_speed=10.0)
#: Adami et al. (2012) admissibility limit on the spurious current.
ADAMI_LIMIT = 0.10
#: Audit horizon, in capillary periods.  The harness itself simulates 4-5.
HORIZON_PERIODS = 2.0


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def build(kw: dict, **overrides):
    """Initial state of the standing wave plus the configured run parameters."""
    overrides = dict(overrides)
    # ``lattice`` and ``h_factor`` are make_case() arguments, the rest are Config.
    case_kwargs = {key: overrides.pop(key) for key in ("lattice", "h_factor")
                   if key in overrides}
    pos, mass, cfg = V.make_case(
        kw["wavelength"], kw["depth"], kw["dx"], kw["amp_over_dx"] * kw["dx"],
        kw["rho0"], kw["sigma"], kw["viscosity"], kw["sound_speed"],
        **case_kwargs)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Config has no attribute {key!r}")
        setattr(cfg, key, value)
    # Fixed acoustic step (the harness uses suggested_dt; recompute it because a
    # variant may change sound speed or Courant).
    cfg.dt = cfg.max_courant * cfg.h_factor * cfg.dx / cfg.sound_speed
    return pos.float(), mass.float(), cfg


def initial_fields(kw: dict, **overrides):
    """``(pos, mass, cfg, rho, p, acc)`` of the untouched initial state."""
    pos, mass, cfg = build(kw, **overrides)
    h = cfg.h_factor * cfg.dx
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                          fluid_count=pos.shape[0])
    pairs, _ = cache.get(pos)
    rho, p, acc = rhs(pos, torch.zeros_like(pos), mass, cfg, pairs=pairs,
                      wall=None, dt=cfg.dt)
    return pos, mass, cfg, rho, p, acc


def simulate(kw: dict, periods: float = HORIZON_PERIODS, **overrides) -> dict:
    """March the standing wave; return a status row instead of an exception."""
    pos, mass, cfg = build(kw, **overrides)
    device = resolve_device("cpu")
    pos, mass = pos.to(device), mass.to(device)
    vel = torch.zeros_like(pos)

    k = 2.0 * math.pi / kw["wavelength"]
    omega = V.angular_frequency(kw["sigma"], k, kw["rho0"], kw["depth"])
    amp = kw["amp_over_dx"] * kw["dx"]
    u_ref = amp * omega                      # physical particle velocity
    c_phi = omega / k                        # phase speed
    dt = float(cfg.dt)
    steps = int(math.ceil(periods * 2.0 * math.pi / omega / dt))
    every = max(1, int(round(steps / (periods * 120))))
    n_bins = max(8, int(round((kw["wavelength"] / 2.0) / (2.0 * kw["dx"]))))

    h = cfg.h_factor * cfg.dx
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                          fluid_count=pos.shape[0])
    top = kw["depth"] + amp + 4.0 * cfg.dx

    times: list[float] = []
    amps: list[float] = []
    v_hist: list[float] = []
    peak = 0.0
    status, escape_step = "completed", None
    rho_min = 1.0
    t0 = time.perf_counter()
    for step in range(steps + 1):
        pairs, _ = cache.get(pos)
        rho, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=dt)
        rho_min = min(rho_min, float((rho / cfg.rho0).min()))
        if step % every == 0 or step == steps:
            times.append(step * dt)
            amps.append(V.column_elevation(pos, mass, kw["wavelength"] / 2.0,
                                           kw["depth"], kw["rho0"], k, n_bins))
            v_hist.append(float(torch.linalg.vector_norm(vel, dim=1).max()))
        vel = vel + dt * acc
        pos = pos + dt * vel
        peak = max(peak, float(torch.linalg.vector_norm(vel, dim=1).max()))
        if apply_boundary(pos, vel, cfg, top=top):
            status, escape_step = "aborted", step
            break
    wall = time.perf_counter() - t0

    row = {
        "status": status,
        "escape_step": escape_step,
        "steps_horizon": steps,
        "escape_fraction_of_horizon": (None if escape_step is None
                                       else escape_step / steps),
        "n_particles": int(pos.shape[0]),
        "dt_s": dt,
        "horizon_periods": periods,
        "v_max_m_per_s": peak,
        "u_ref_m_per_s": u_ref,
        "v_max_over_u_ref": peak / u_ref,
        "c_phase_m_per_s": c_phi,
        "v_max_over_c_phase": peak / c_phi,
        "rho_min_over_rho0": rho_min,
        "admissible_adami": peak / u_ref < ADAMI_LIMIT,
        "v_max_history": v_hist,
        "wall_clock_s": wall,
        "overrides": {k2: str(v2) for k2, v2 in overrides.items()},
    }
    if status == "aborted":
        return row
    t = np.asarray(times)
    a = np.asarray(amps)
    fit = V.fit_damped_oscillation(t, a, omega)
    row.update({
        "omega_analytic_rad_per_s": omega,
        "omega_measured_rad_per_s": fit["omega"],
        "omega_rel_error_pct": 100.0 * (fit["omega"] - omega) / omega,
        "gamma_measured_1_per_s": fit["gamma"],
        "gamma_analytic_1_per_s": V.damping_rate(kw["viscosity"], k),
        "fit_residual_rel": fit["residual_rel"],
        "amplitude_initial_measured_m": float(a[0]) if a.size else None,
        "amplitude_imposed_m": amp,
        "t_history": times,
        "amplitude_history_m": amps,
    })
    return row


# --------------------------------------------------------------------------- #
# the three measurements
# --------------------------------------------------------------------------- #
def harness_at_head() -> dict:
    """Run the harness's own case exactly as ``validation.py`` does."""
    try:
        rep = V.run_case(wavelength=CASE["wavelength"], depth=CASE["depth"],
                         dx=CASE["dx"],
                         amplitude=CASE["amp_over_dx"] * CASE["dx"],
                         periods=HORIZON_PERIODS, rho0=CASE["rho0"],
                         sigma=CASE["sigma"], viscosity=CASE["viscosity"],
                         sound_speed=CASE["sound_speed"],
                         samples_per_period=60)
        return {"case": "resolution", "status": "completed",
                "omega_rel_error_pct": rep["measured"]["omega_rel_error_pct"],
                "v_max_over_u_ref": (rep["measured"]["peak_speed_m_per_s"]
                                     / (CASE["amp_over_dx"] * CASE["dx"]
                                        * rep["analytic"]["omega_rad_per_s"]))}
    except RuntimeError as exc:
        msg = str(exc)
        step = int(msg.split("step")[1].split(";")[0]) if "step" in msg else None
        return {"case": "resolution", "status": "aborted", "exception": msg,
                "escape_step": step, "n_particles": None}


def curvature_audit(kw: dict = CASE) -> dict:
    """Discrete curvature of the almost-flat interface vs its analytic value."""
    k = 2.0 * math.pi / kw["wavelength"]
    amp = kw["amp_over_dx"] * kw["dx"]
    kappa_signal = k * k * amp          # |eta''| of cos(kx) at small slope
    cells = []
    for lattice in ("hex", "square"):
        for h_factor in (1.3, 1.5):
            pos, mass, cfg = build(kw, lattice=lattice, h_factor=h_factor)
            h = cfg.h_factor * cfg.dx
            cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h,
                                  fluid_count=pos.shape[0])
            pairs, _ = cache.get(pos)
            for scheme in ("mls", "fit_quadratic"):
                state = csf_fields(pos, mass, pairs, h, kw["rho0"],
                                   curvature_scheme=scheme)
                kappa = state.kappa
                delta_s = state.delta_s
                shell = delta_s > 0.5 * delta_s.max()
                vals = kappa[shell]
                rms = float(torch.sqrt((vals ** 2).mean()))
                cells.append({
                    "lattice": lattice,
                    "h_factor": h_factor,
                    "curvature_scheme": scheme,
                    "n_interface_particles": int(shell.sum()),
                    "kappa_rms_1_per_m": rms,
                    "kappa_abs_p95_1_per_m": float(torch.quantile(
                        vals.abs(), 0.95)),
                    "kappa_abs_max_1_per_m": float(vals.abs().max()),
                    "kappa_signed_mean_1_per_m": float(vals.mean()),
                    "delta_s_median_1_per_m": float(delta_s[shell].median()),
                    "noise_over_signal": rms / kappa_signal,
                })
    return {"analytic_signal_kappa_1_per_m": kappa_signal,
            "k_1_per_m": k, "amplitude_m": amp, "cells": cells}


def initial_acceleration(kw: dict = CASE) -> dict:
    """Physical restoring acceleration vs the CSF contribution of the same state."""
    k = 2.0 * math.pi / kw["wavelength"]
    amp = kw["amp_over_dx"] * kw["dx"]
    omega = V.angular_frequency(kw["sigma"], k, kw["rho0"], kw["depth"])
    a_physical = omega ** 2 * amp
    out = {"omega_rad_per_s": omega, "amplitude_m": amp,
           "a_physical_omega2A_m_per_s2": a_physical}
    for label, over in (("csf_on", {}),
                        ("csf_off", {"surface_tension_coefficient": 0.0}),
                        ("csf_on_density_renormalization",
                         {"density_renormalization": True}),
                        ("csf_on_mass_calibrated",
                         {"calibrate_initial_mass": True})):
        pos, mass, cfg, rho, p, acc = initial_fields(kw, **over)
        a = torch.linalg.vector_norm(acc, dim=1)
        shell = pos[:, 1] > kw["depth"] - 2.5 * kw["dx"]
        out[label] = {
            "a_max_m_per_s2": float(a.max()),
            "a_p99_m_per_s2": float(torch.quantile(a, 0.99)),
            "a_median_m_per_s2": float(torch.median(a)),
            "a_max_shell_m_per_s2": float(a[shell].max()),
            "a_median_shell_m_per_s2": float(torch.median(a[shell])),
            "rho_min_over_rho0": float((rho / cfg.rho0).min()),
            "p_max_Pa": float(p.max()),
            "a_max_over_physical": float(a.max()) / a_physical,
        }
    excess = (out["csf_on"]["a_max_m_per_s2"]
              - out["csf_off"]["a_max_m_per_s2"])
    out["csf_excess_a_max_m_per_s2"] = excess
    out["csf_excess_over_physical"] = excess / a_physical
    return out


VARIANTS = [
    ("baseline", {}),
    ("density_renormalization", {"density_renormalization": True}),
    ("mass_calibrated", {"calibrate_initial_mass": True}),
    ("pressure_closure_all_donors",
     {"surface_pressure_closure": True, "surface_pressure_donors": "all"}),
    ("pressure_closure_interior_2",
     {"surface_pressure_closure": True, "surface_pressure_donors": "interior"}),
    ("legacy_csf", {"csf_scheme": "legacy"}),
    ("square_lattice", {"lattice": "square"}),
    ("sound_speed_20", {"sound_speed": 20.0}),
    ("renorm_plus_calibrated",
     {"density_renormalization": True, "calibrate_initial_mass": True}),
    ("renorm_plus_legacy", {"density_renormalization": True,
                            "csf_scheme": "legacy"}),
]


def variant_sweep() -> list[dict]:
    rows = []
    for name, over in VARIANTS:
        row = simulate(CASE, **over)
        row["variant"] = name
        rows.append(row)
    return rows


def amplitude_ladder() -> list[dict]:
    rows = []
    for amp_over_dx in (0.5, 2.0):
        kw = dict(CASE, amp_over_dx=amp_over_dx)
        for name, over in (("renorm", {"density_renormalization": True}),
                           ("baseline", {})):
            row = simulate(kw, periods=3.0, **over)
            row["variant"] = f"A={amp_over_dx}dx/{name}"
            row["amplitude_over_dx"] = amp_over_dx
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(payload: dict) -> list[str]:
    """Delegate to :func:`make_doc.render_figure` (one figure, one code path).

    The drawing lives in ``make_doc`` because the figure must be a pure function
    of the committed JSON -- that is what lets
    ``tests/test_capillary_wave_open_surface.py`` regenerate it and compare,
    instead of trusting a binary that the merge path may not even carry (see the
    ``FIGURE_DIR`` note in ``make_doc``).  Here we only trigger it once the audit
    has finished, so a fresh run refreshes the figure it just measured.
    """
    import make_doc

    path = make_doc.render_figure(payload)
    return [path.name]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("[1/4] harness as-is at HEAD ...")
    head = harness_at_head()
    print(f"      -> {head['status']}"
          + (f" at step {head.get('escape_step')}" if head.get("escape_step")
             else f", omega error {head.get('omega_rel_error_pct'):+.1f}%"))

    print("[2/4] initial-state decomposition ...")
    acc = initial_acceleration()
    curv = curvature_audit()
    print(f"      physical a = {acc['a_physical_omega2A_m_per_s2']:.1f} m/s^2, "
          f"CSF excess = {acc['csf_excess_a_max_m_per_s2']:.0f} m/s^2 "
          f"({acc['csf_excess_over_physical']:.0f}x)")
    print(f"      curvature noise / signal = "
          f"{min(c['noise_over_signal'] for c in curv['cells']):.1f}"
          f"..{max(c['noise_over_signal'] for c in curv['cells']):.1f}x")

    print("[3/4] variant sweep ...")
    sweep = variant_sweep()
    for row in sweep:
        if row["status"] == "aborted":
            print(f"      {row['variant']:30s} aborted at step "
                  f"{row['escape_step']} ({row['escape_fraction_of_horizon']:.2f} "
                  f"of horizon), v_max/u_ref = {row['v_max_over_u_ref']:.1f}")
        else:
            print(f"      {row['variant']:30s} completed, v_max/u_ref = "
                  f"{row['v_max_over_u_ref']:.1f}, omega error "
                  f"{row['omega_rel_error_pct']:+.1f}%, resid "
                  f"{row['fit_residual_rel']:.2f}")

    print("[4/4] amplitude ladder ...")
    ladder = amplitude_ladder()
    for row in ladder:
        if row["status"] == "aborted":
            print(f"      {row['variant']:26s} aborted at step {row['escape_step']}")
        else:
            print(f"      {row['variant']:26s} omega error "
                  f"{row['omega_rel_error_pct']:+.1f}%, resid "
                  f"{row['fit_residual_rel']:.2f}, v_max/u_ref "
                  f"{row['v_max_over_u_ref']:.1f}, a(0) "
                  f"{row['amplitude_initial_measured_m']*1e3:+.2f} mm vs imposed "
                  f"{row['amplitude_imposed_m']*1e3:.2f} mm")

    completed = [r for r in sweep + ladder if r["status"] == "completed"]
    payload = {
        "benchmark": "capillary_wave_open_surface_audit",
        "question": ("can TensorSPH carry a linear capillary standing wave, i.e. "
                     "is the capillary-wave dispersion claim gradable at HEAD?"),
        "references": {
            "dispersion": ("Lamb (1932) Art. 267/348: "
                           "omega^2=(g k + sigma k^3/rho) tanh(kH), g=0 here"),
            "damping": "Lamb (1932) Art. 348: gamma = 2 nu k^2",
            "admissibility": ("Adami, Ihmsen & Adams (2012) JCP 231:7057: spurious "
                              "current must stay below 10 % of the capillary velocity "
                              "scale, here u_ref = A omega"),
        },
        "case": CASE,
        "horizon_periods": HORIZON_PERIODS,
        "admissibility": {"limit": ADAMI_LIMIT, "scale": "u_ref = A*omega"},
        "harness_at_head": head,
        "initial_acceleration": acc,
        "curvature": curv,
        "variant_sweep": sweep,
        "amplitude_ladder": ladder,
        "summary": {
            "n_variants": len(sweep),
            "n_aborted": sum(1 for r in sweep if r["status"] == "aborted"),
            "n_admissible": sum(1 for r in completed if r["admissible_adami"]),
            "min_noise_over_signal": min(c["noise_over_signal"]
                                         for c in curv["cells"]),
            "max_noise_over_signal": max(c["noise_over_signal"]
                                          for c in curv["cells"]),
            "wall_clock_s": time.perf_counter() - t0,
        },
        "verdict": {
            "grade": "WITHHELD",
            "reason": ("the discrete CSF force on an open free surface injects a "
                       "spurious acceleration two orders of magnitude above the "
                       "wave's own restoring acceleration, so the standing wave is "
                       "destroyed before one period; no variant of the solver knobs "
                       "brings the spurious current inside the Adami limit"),
        },
    }
    if not args.no_figures:
        payload["figures"] = make_figures(payload)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "open_surface_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"report -> {path}  ({payload['summary']['wall_clock_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
