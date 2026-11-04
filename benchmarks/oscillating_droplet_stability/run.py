"""Why the mode-2 (oscillating drop) verdict is not a physics verdict.

Motivation
----------
``docs/verification_oscillating_droplet.md`` records the 2-D Rayleigh-Lamb
mode-2 frequency error of the release case as **-43.5 %** (FAIL against the
15 % bar) and the CI companion case as -31.8 %, while GitHub's x86_64 CI
measures **+8.6 %** on *identical code* (see the strict xfail in
``tests/test_oscillating_droplet.py``).  Three numbers that disagree by 40
percentage points cannot all describe one surface-tension model.

This study tests the hypothesis that none of them does: the stored trace is a
*diverging* run, and the frequency estimator is applied to it anyway.

What is measured
----------------
1. **Physical sanity of the trajectory.**  Per step the harness records
   ``max|v|/c0`` (Mach) and the area-equivalent radius drift ``r_eff/R - 1``.
   A weak-compressibility SPH result is only admissible if the flow stays
   subsonic and the drop keeps its radius; the platform's own gates are
   Mach <= 0.1 and a few % radius drift.  The first time either gate trips is
   ``t_gate``: from there on the trace is numerical debris and every estimator
   computed on it is meaningless.
2. **What the published pipeline reports.**  ``damped_cosine_fit`` on the full
   stored trace -- reproduced bit-for-bit from ``trace_data.json`` without any
   simulation (the -43.5 % number) and, for comparison, on the pre-gate window.
3. **Window and estimator sensitivity.**  The same fit on windows
   ``[0, w T]`` for ``w in {0.5, 0.75, 1.0, 1.25, 1.5, 1.8}`` and the two
   independent estimators (kinetic energy, zero crossings) on the valid window.
4. **Spectrum.**  FFT of ``s(t)`` over the full trace vs the valid window: a
   well-posed measurement puts one peak at ``omega_2``; the full trace puts
   comparable energy at several frequencies, which is where the x86 / loong64
   split comes from (the least-squares fit then selects among near-degenerate
   basins).

Usage::

    PYTHONPATH=src python benchmarks/oscillating_droplet_stability/run.py \
        --stage diag        # integrate the case, save per-step diagnostics
    PYTHONPATH=src python benchmarks/oscillating_droplet_stability/run.py \
        --stage fit --plot  # estimators, windows, spectrum, figures, report.json
    PYTHONPATH=src python benchmarks/oscillating_droplet_stability/run.py \
        --stage all --plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "oscillating_droplet"))

from tensorsph.droplet_impact import Config, rhs, stable_dt  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

import run as od  # noqa: E402  (benchmarks/oscillating_droplet/run.py)

torch.set_num_threads(2)

MODE = 2
#: Platform gates: a weak-compressibility SPH trajectory must stay subsonic and
#: must not change the drop's area-equivalent radius.  Exceeding either makes
#: the run inadmissible as a *physical* measurement.
MACH_GATE = 0.10
RADIUS_DRIFT_GATE_PCT = 5.0


# --------------------------------------------------------------------------- #
# integration with per-step sanity diagnostics
# --------------------------------------------------------------------------- #
def integrate(radius: float = 0.02, dx: float = 0.002, sigma: float = 7.2,
              rho0: float = 1000.0, sound_speed: float = 20.0,
              viscosity: float = 5e-5, eps: float = 0.10,
              periods: float = 1.8, fixed_dt: bool = True) -> dict:
    """Release the elliptical drop and record s(t) *plus* admissibility history.

    The integration loop mirrors ``benchmarks/oscillating_droplet/run.run_case``
    line for line (same ``Config``, same ``rhs`` call, same ``stable_dt``), so
    the trajectory is the benchmark's; the only addition is the per-step
    ``max|v|/c0`` and ``r_eff`` history that the production recorder keeps only
    as a run-level maximum.
    """
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma, csf_scheme="consistent",
                 csf_curvature="fit_quadratic", density_diffusion=0.0)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    dt0 = dt
    omega_ref = od.analytic_omega(radius, sigma, rho0)
    period_ref = 2.0 * math.pi / omega_ref
    steps = int(math.ceil(periods * period_ref / dt))

    pos, mass = od.elliptical_drop(radius, dx, rho0, eps)
    vel = torch.zeros_like(pos)
    m0 = float(mass.sum())
    initial_state = od.mode_amplitude(pos, mass)

    ts, s_mom, ke_hist, mach_hist, reff_hist, dt_hist = [], [], [], [], [], []
    for step in range(steps):
        if not fixed_dt:
            dt = min(dt, float(stable_dt(cfg, float(torch.linalg.vector_norm(
                vel, dim=1).max()))))
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        amp = od.mode_amplitude(pos, mass)
        ts.append((step + 1) * dt)
        s_mom.append(amp["s_moment"])
        ke_hist.append(float(0.5 * (mass * vel.square().sum(dim=1)).sum()))
        mach_hist.append(float(torch.linalg.vector_norm(vel, dim=1).max())
                         / sound_speed)
        reff_hist.append(amp["r_eff"])
        dt_hist.append(dt)
        # stop early once the run is unambiguously debris (cost control)
        if mach_hist[-1] > 50.0 or not math.isfinite(mach_hist[-1]):
            break

    return {
        "setup": {"radius_m": radius, "dx_m": dx, "h_m": h, "dt0_s": dt0,
                  "n_particles": int(pos.shape[0]), "sigma_N_per_m": sigma,
                  "rho0": rho0, "sound_speed_m_per_s": sound_speed,
                  "viscosity_m2_per_s": viscosity, "initial_eps": eps,
                  "csf_scheme": "consistent", "csf_curvature": "fit_quadratic",
                  "steps": len(ts), "physical_time_s": ts[-1] if ts else 0.0,
                  "fixed_dt": fixed_dt},
        "reference": {"omega_rad_per_s": omega_ref, "period_s": period_ref},
        "trace": {"t": ts, "s_second_moment": s_mom, "kinetic_energy_J": ke_hist,
                  "mach": mach_hist, "r_eff_m": reff_hist, "dt_s": dt_hist},
        "mass_relative_drift": abs(float(mass.sum()) - m0) / m0,
        "initial_state": initial_state,
    }


# --------------------------------------------------------------------------- #
# admissibility
# --------------------------------------------------------------------------- #
def gate_time(case: dict) -> dict:
    """First time the trajectory violates a physical admissibility gate."""
    R = case["setup"]["radius_m"]
    t = case["trace"]["t"]
    mach = case["trace"]["mach"]
    reff = case["trace"]["r_eff_m"]
    t_mach = t_reff = None
    for ti, m in zip(t, mach):
        if m > MACH_GATE:
            t_mach = ti
            break
    for ti, r in zip(t, reff):
        if abs(r / R - 1.0) * 100.0 > RADIUS_DRIFT_GATE_PCT:
            t_reff = ti
            break
    cands = [x for x in (t_mach, t_reff) if x is not None]
    t_gate = min(cands) if cands else float("nan")
    return {"t_mach_gate_s": t_mach if t_mach is not None else float("nan"),
            "t_radius_gate_s": t_reff if t_reff is not None else float("nan"),
            "t_gate_s": t_gate,
            "gate_tripped": bool(cands),
            "mach_max": max(mach), "mach_at_end": mach[-1],
            "reff_drift_pct_max": max(abs(r / R - 1.0) * 100.0 for r in reff),
            "admissible_fraction_of_periods":
                (t_gate / case["reference"]["period_s"]) if cands else
                case["trace"]["t"][-1] / case["reference"]["period_s"]}


def fit_window(case: dict, w_periods: float | None = None,
               t_end: float | None = None) -> dict:
    """Damped-cosine fit on ``[0, min(w T, t_end)]`` of the stored trace."""
    T = case["reference"]["period_s"]
    om_ref = case["reference"]["omega_rad_per_s"]
    t = case["trace"]["t"]
    s = case["trace"]["s_second_moment"]
    stop = math.inf
    if w_periods is not None:
        stop = min(stop, w_periods * T)
    if t_end is not None:
        stop = min(stop, t_end)
    idx = [i for i, ti in enumerate(t) if ti <= stop]
    if len(idx) < 8:
        return {"n_samples": len(idx), "omega_rad_per_s": float("nan"),
                "rel_error_pct": float("nan")}
    tt = [t[i] for i in idx]
    ss = [s[i] for i in idx]
    fit = od.damped_cosine_fit(tt, ss, om_ref)
    zc = od.zero_crossing_period(tt, ss)
    ke = [case["trace"]["kinetic_energy_J"][i] for i in idx]
    fit_ke = od.damped_cosine_fit(tt, ke, 2.0 * om_ref)
    return {
        "window_periods": (tt[-1] - tt[0]) / T, "n_samples": len(idx),
        "omega_rad_per_s": fit["omega"],
        "rel_error_pct": (fit["omega"] - om_ref) / om_ref * 100.0,
        "r2": fit["r2"], "rms": fit["rms"], "lambda_per_s": fit["lambda_per_s"],
        "omega_zero_crossing_rad_per_s": zc["omega"],
        "zero_crossing_rel_error_pct":
            (zc["omega"] - om_ref) / om_ref * 100.0 if math.isfinite(zc["omega"])
            else float("nan"),
        "omega_kinetic_energy_over2_rad_per_s": 0.5 * fit_ke["omega"],
        "ke_rel_error_pct": (0.5 * fit_ke["omega"] - om_ref) / om_ref * 100.0,
    }


def fft_peaks(case: dict, t_end: float | None = None, n_keep: int = 5) -> list:
    """Dominant angular frequencies of ``s(t)`` (Hann-windowed, zero-padded)."""
    t = case["trace"]["t"]
    s = case["trace"]["s_second_moment"]
    if t_end is not None:
        idx = [i for i, ti in enumerate(t) if ti <= t_end]
        t = [t[i] for i in idx]
        s = [s[i] for i in idx]
    if len(s) < 16:
        return []
    dt = (t[-1] - t[0]) / (len(t) - 1)
    n = 1 << max(8, (len(s) - 1).bit_length() + 3)  # zero-pad x8 for peak loc.
    y = torch.tensor(s, dtype=torch.float64)
    y = (y - y.mean()) * torch.hann_window(len(s), dtype=torch.float64)
    y = torch.cat([y, torch.zeros(n - len(s), dtype=torch.float64)])
    sp = torch.fft.rfft(y).abs()
    fr = torch.fft.rfftfreq(n, d=dt)
    omega = 2.0 * math.pi * fr
    band = (omega > 0.15 * case["reference"]["omega_rad_per_s"]) & \
           (omega < 4.0 * case["reference"]["omega_rad_per_s"])
    idx = torch.nonzero(band).flatten()
    vals = sp[idx]
    order = torch.argsort(vals, descending=True)
    out, used = [], []
    for o in order.tolist():
        w = float(omega[idx[o]])
        if any(abs(w - u) < 0.25 * w for u in used):
            continue
        used.append(w)
        out.append({"omega_rad_per_s": w,
                    "rel_error_pct": (w - case["reference"]["omega_rad_per_s"])
                    / case["reference"]["omega_rad_per_s"] * 100.0,
                    "amplitude": float(vals[o])})
        if len(out) >= n_keep:
            break
    return out


def boundary_pin_check(omega_ref: float) -> dict:
    """Is the reported optimum the *edge* of the estimator's search box?

    ``damped_cosine_fit`` searches ``omega in [0.6, 1.6] * omega_guess`` and then
    runs three refinements whose half-width is +-2 % of the current best.  A
    minimum outside the box therefore walks the answer down to
    ``0.6 * omega_ref * 0.98**3``, the lowest value the ladder can express.  If
    the reported frequency lands there, the fit is boundary-pinned and the
    number says nothing about the model's frequency.
    """
    edge = 0.6 * omega_ref
    return {"lower_grid_edge_rad_per_s": edge,
            "edge_after_three_refinements_rad_per_s": edge * 0.98 ** 3,
            "refinement_shrink_factor": 0.98 ** 3}


def published_evidence() -> dict:
    """Admissibility of the *published* release case, from its own artefacts.

    ``benchmarks/oscillating_droplet/validation_report.json`` records the
    run-level maximum speed and the final drop state, and ``trace_data.json``
    stores the very trace the -43.5 % was fitted to.  Re-fitting that stored
    trace is how this study proves it reproduces the published number, and the
    run-level fields are how it proves the trace is debris -- no simulation and
    no arch assumption involved.
    """
    src = REPO / "benchmarks" / "oscillating_droplet"
    rep = json.loads((src / "validation_report.json").read_text())["frequency"]
    trace = json.loads((src / "trace_data.json").read_text())["frequency"]
    R = rep["setup"]["radius_m"]
    c0 = rep["setup"]["sound_speed"]
    case = {"setup": rep["setup"], "reference": rep["reference"],
            "trace": {"t": trace["t"],
                      "s_second_moment": trace["s_second_moment"],
                      "kinetic_energy_J": trace["kinetic_energy_J"]}}
    om_ref = rep["reference"]["omega_rad_per_s"]
    fit = od.damped_cosine_fit(trace["t"], trace["s_second_moment"], om_ref)
    return {
        "source_files": [
            "benchmarks/oscillating_droplet/validation_report.json",
            "benchmarks/oscillating_droplet/trace_data.json"],
        "recorded_verdict": json.loads(
            (src / "validation_report.json").read_text())["verdict"],
        "recorded_rel_error_pct": rep["frequency"]["fit_second_moment_rel_error_pct"],
        "refit_from_stored_trace": {
            "omega_rad_per_s": fit["omega"],
            "rel_error_pct": (fit["omega"] - om_ref) / om_ref * 100.0,
            "r2": fit["r2"], "rms": fit["rms"],
            "lambda_per_s": fit["lambda_per_s"],
            "reproduces_recorded": abs(
                (fit["omega"] - om_ref) / om_ref * 100.0
                - rep["frequency"]["fit_second_moment_rel_error_pct"]) < 1e-6},
        # admissibility, straight from the production recorder
        "max_speed_m_per_s": rep["diagnostics"]["max_speed_m_per_s"],
        "max_mach": rep["diagnostics"]["max_speed_m_per_s"] / c0,
        "final_reff_over_R": rep["final_state"]["r_eff"] / R,
        "final_semi_major_m": rep["final_state"]["semi_major"],
        "initial_eps": rep["setup"]["initial_eps"],
        "final_s_moment": rep["final_state"]["s_moment"],
        "admissible": (rep["diagnostics"]["max_speed_m_per_s"] / c0 <= MACH_GATE
                       and abs(rep["final_state"]["r_eff"] / R - 1.0) * 100.0
                       <= RADIUS_DRIFT_GATE_PCT),
        "boundary_pin": boundary_pin_check(om_ref),
        "fft_full_trace": fft_peaks(case),
    }


def study_case(case: dict, label: str) -> dict:
    """All estimators for one integrated case."""
    g = gate_time(case)
    om_ref = case["reference"]["omega_rad_per_s"]
    full = fit_window(case)
    valid = fit_window(case, t_end=None if not g["gate_tripped"]
                       else g["t_gate_s"])
    ladder = [fit_window(case, w_periods=w)
              for w in (0.5, 0.75, 1.0, 1.25, 1.5, 1.8)]
    return {
        "label": label, "setup": case["setup"], "reference": case["reference"],
        "admissibility": g,
        "fit_full_trace": full,
        "fit_valid_window": valid,
        "fit_window_ladder": ladder,
        "boundary_pin": boundary_pin_check(om_ref),
        "fft_full_trace": fft_peaks(case),
        "fft_valid_window": fft_peaks(case, t_end=None if not g["gate_tripped"]
                                      else g["t_gate_s"]),
        "trace": case["trace"],
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
CASES = {
    "release_R20mm_eps0.10": dict(radius=0.02, dx=0.002, eps=0.10, periods=1.8),
    "release_R20mm_eps0.02": dict(radius=0.02, dx=0.002, eps=0.02, periods=1.8),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["diag", "fit", "all"])
    ap.add_argument("--cases", default="release_R20mm_eps0.10",
                    help="comma separated keys of CASES")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    keys = [k.strip() for k in args.cases.split(",") if k.strip()]
    diag_path = HERE / "diagnostics.json"
    report_path = HERE / "report.json"

    if args.stage in ("diag", "all"):
        diag = {}
        if diag_path.exists() and args.stage == "diag":
            diag = json.load(open(diag_path))
        for k in keys:
            if k in diag:
                print(f"[diag] {k}: cached")
                continue
            print(f"[diag] {k}: integrating ...", flush=True)
            diag[k] = integrate(**CASES[k])
            g = gate_time(diag[k])
            print(f"[diag] {k}: steps={diag[k]['setup']['steps']} "
                  f"mach_max={g['mach_max']:.3g} t_gate={g['t_gate_s']}", flush=True)
        diag_path.write_text(json.dumps(diag))

    if args.stage in ("fit", "all"):
        diag = json.load(open(diag_path))
        out = {"platform_gates": {"mach": MACH_GATE,
                                  "radius_drift_pct": RADIUS_DRIFT_GATE_PCT},
               "published_run": published_evidence(),
               "cases": {}}
        for k in keys:
            if k not in diag:
                continue
            print(f"[fit] {k} ...", flush=True)
            case = study_case(diag[k], k)
            case.pop("trace", None)   # full history lives in diagnostics.json
            out["cases"][k] = case
        report_path.write_text(json.dumps(out, indent=1))
        pub = out["published_run"]
        print(f"[published] recorded {pub['recorded_rel_error_pct']:.2f}% "
              f"refit {pub['refit_from_stored_trace']['rel_error_pct']:.2f}% "
              f"r2={pub['refit_from_stored_trace']['r2']:.2f} | "
              f"max Mach {pub['max_mach']:.0f}, final r_eff/R "
              f"{pub['final_reff_over_R']:.0f} -> admissible={pub['admissible']}",
              flush=True)
        for k, c in out["cases"].items():
            g, f, v = c["admissibility"], c["fit_full_trace"], c["fit_valid_window"]
            print(f"{k}: t_gate={g['t_gate_s']:.4g}s "
                  f"({g['admissible_fraction_of_periods']:.2f} T) "
                  f"full={f['rel_error_pct']:.1f}% r2={f['r2']:.2f} | "
                  f"valid={v['rel_error_pct']:.1f}% r2={v['r2']:.2f}", flush=True)

    if args.plot:
        import plot
        plot.render(json.load(open(report_path)), HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())