"""Turn a ``run_body_force_transient.py`` artefact into the Turek-Hron verdict.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why this exists
---------------
The transient driver writes a per-step history plus (on a clean finish) a
``summary`` block.  A job killed by its wall-clock limit only has the history
(checkpointed every N steps), and the anchor's headline numbers -- ``Cd`` mean
and amplitude, ``Cl`` amplitude, beam tip amplitudes -- were carried as
WITHHELD for exactly that reason.  This script recomputes the summary from the
history alone, so a checkpointed run is still usable, and it refuses to report
numbers it cannot defend:

* the averaging window is reported explicitly (start/end time, samples) and the
  number of VIV periods it spans is *measured* from the signal, not assumed;
* ``Cd`` is decomposed into the pressure arm and the friction (viscous) arm,
  because the driver's ``body_drag_fit`` excludes the viscous load while the
  benchmark ``Cd`` is the total.  A deficit is then attributable rather than
  merely observed;
* every gate statement carries the window it was computed on.  If the window
  covers less than one VIV period the verdict is ``WITHHELD``, not PASS/FAIL.

Usage::

    python benchmarks/xval_fsi2/verdict_body_force_transient.py \
        --artifact artifacts/xval_fsi2/body_force_transient_fix.json \
        --out docs/xval_fsi2/body_force_verdict.json

Exit status is 0 for PASS / FAIL / WITHHELD alike (the verdict is data, not an
error); a non-zero status means the artefact was unusable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.fsi2_reference import (  # noqa: E402
    BENCHMARK_FORCES,
    BENCHMARK_TIP,
    compare_body_forces,
    compare_to_benchmark,
    summarise_body_forces,
)

GATE_ACCEPT = 0.15   # owner standard: > 15 % is not acceptable
GATE_REPORT = 0.03   # report discipline target


# --------------------------------------------------------------------------- #
# signal analysis
# --------------------------------------------------------------------------- #
def dominant_period(t, signal, min_lag_frac: float = 0.02) -> float | None:
    """Period of the strongest oscillatory component, from the autocorrelation.

    Returns ``None`` when the signal is too short or has no interior
    autocorrelation peak (a monotone start-up transient has none), which is the
    honest answer for a run that has not entered the periodic regime yet.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(signal, dtype=float)
    if x.size < 16 or not np.all(np.isfinite(x)):
        return None
    x = x - x.mean()
    var = float(np.dot(x, x))
    if var <= 0.0:
        return None
    # unbiased autocorrelation via FFT
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(F * np.conj(F), nfft)[:n]
    ac = ac / var
    min_lag = max(1, int(min_lag_frac * n))
    # first local maximum strictly after the zero-lag lobe
    for i in range(min_lag + 1, n - 1):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.05:
            dt = float(np.mean(np.diff(t)))
            return float(i * dt)
    return None


#: Strouhal band used *only* as a plausibility band for the identified period.
#: A vortex street behind a cylinder at Re ~ 100 sheds at ``St ~ 0.13-0.30``; the
#: FSI2 beam follows the shedding, so a "period" outside ``D/(St U)`` cannot be
#: the VIV period.  The band is deliberately wide (a factor 2.3) -- it is there
#: to reject start-up ringing (measured 0.153 s against a 1.7-3.8 s band), not
#: to score the answer.
VIV_STROUHAL_BAND = (0.13, 0.30)


def viv_period_band(u_mean: float, diameter: float) -> tuple | None:
    """Physically admissible VIV period range [s] for ``(U_mean, D)``."""
    if not u_mean or u_mean <= 0 or not diameter or diameter <= 0:
        return None
    lo, hi = VIV_STROUHAL_BAND
    return (float(diameter / (hi * u_mean)), float(diameter / (lo * u_mean)))


def _envelope_stationarity(x, min_ratio: float = 0.8,
                           max_ratio: float = 1.25) -> dict:
    """Peak-to-peak of the second half over the first half of the window.

    An oscillation whose *amplitude* is still growing (a start-up transient, or
    the acoustic ringing that follows the inlet ramp) can have a clean interior
    autocorrelation peak without being periodic.  This ratio is the cheap
    discriminator: the previous round's artefact was judged ``FAIL`` off a
    0.153 s "period" that was ringing, i.e. a start-up transient was scored as
    if it were a periodic solution.  A periodic window has ``ratio ~ 1``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 32 or not np.all(np.isfinite(x)):
        return {"ratio": None, "stationary": False}
    h = n // 2
    a = float(np.ptp(x[:h]))
    b = float(np.ptp(x[h:]))
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0:
        return {"ratio": None, "stationary": False}
    ratio = b / a
    return {"ratio": float(ratio),
            "stationary": bool(min_ratio <= ratio <= max_ratio)}


def periods_in_window(t, signal, warmup: int,
                      band: tuple | None = None) -> dict:
    t = np.asarray(t, dtype=float)
    x = np.asarray(signal, dtype=float)[warmup:]
    tt = t[warmup:]
    period = dominant_period(tt, x)
    span = float(tt[-1] - tt[0]) if tt.size > 1 else 0.0
    env = _envelope_stationarity(x)
    in_band = None
    if band is not None and period is not None:
        in_band = bool(band[0] <= period <= band[1])
    return {
        "period_s": period,
        "window_s": span,
        "periods_in_window": (span / period) if period else None,
        "amplitude_ratio_second_over_first_half": env["ratio"],
        "amplitude_stationary": env["stationary"],
        "period_band_s": list(band) if band is not None else None,
        "period_in_band": in_band,
    }


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def _arm(summary: dict) -> dict:
    return summary if summary.get("n_samples", 0) else {}


def _gate_row(name, ref, got, gate=GATE_ACCEPT):
    if got is None:
        return {"quantity": name, "reference": ref, "computed": None,
                "rel_error": None, "pass_15": False, "pass_3": False}
    rel = (got - ref) / ref
    return {"quantity": name, "reference": ref, "computed": float(got),
            "rel_error": float(rel), "pass_15": abs(rel) <= gate,
            "pass_3": abs(rel) <= GATE_REPORT}


def build_verdict(payload: dict, warmup_frac: float = 0.5) -> dict:
    hist = payload["history"]
    t = np.asarray(hist["t"], dtype=float)
    n = t.size
    if n == 0:
        raise SystemExit("artefact has no time samples")
    warm = int(warmup_frac * n)
    case = payload.get("case", {})
    rho_f = float(case.get("rho0", 1000.0))
    u_mean = float(case.get("inlet_u_max", 0.2)) / 1.5  # parabolic mean = 2/3 U_max
    D = float(case.get("cylinder_d", 0.1))
    dt = float(payload.get("dt", float("nan")))

    # --- summaries recomputed from the history (works for partial runs) -----
    fit = summarise_body_forces(hist["body_drag_fit"], hist["body_lift_fit"],
                                rho_f, u_mean, D, warmup=warm)
    avg = summarise_body_forces(hist["body_drag"], hist["body_lift"],
                                rho_f, u_mean, D, warmup=warm)

    # --- viscous decomposition: the benchmark Cd is pressure + friction -----
    visc_x = np.asarray(hist.get("drag_viscous", []), dtype=float)[warm:]
    scale = 0.5 * rho_f * u_mean ** 2 * D
    cd_visc = float(visc_x.mean() / scale) if visc_x.size else None
    cd_fit_total = None
    if fit.get("cd_mean") is not None and cd_visc is not None:
        cd_fit_total = float(fit["cd_mean"] + cd_visc)

    tip_x = float(0.5 * (max(hist["tip_x"][warm:]) - min(hist["tip_x"][warm:])))
    tip_y = float(0.5 * (max(hist["tip_y"][warm:]) - min(hist["tip_y"][warm:])))

    # --- how much of the periodic regime does the window actually cover? ----
    band = viv_period_band(u_mean, D)
    per = periods_in_window(t, hist["body_drag_fit"], warm, band=band)
    n_periods = per["periods_in_window"]
    # A window is only "periodic" if it both spans >= 1 cycle *and* the cycle
    # amplitude is stationary; otherwise the numbers are start-up, whatever the
    # autocorrelation says.  The Strouhal band is reported alongside but is not
    # part of the gate (unit-test signals are not physical cases).
    regime_ok = (n_periods is not None and n_periods >= 1.0
                 and bool(per["amplitude_stationary"]))

    # --- gates --------------------------------------------------------------
    fit_gate = [_gate_row(k, v, {"drag_mean": fit.get("cd_mean"),
                                 "drag_amplitude": fit.get("cd_amplitude"),
                                 "lift_amplitude": fit.get("cl_amplitude")}[k])
                for k, v in BENCHMARK_FORCES.items()]
    avg_gate = [_gate_row(k, v, {"drag_mean": avg.get("cd_mean"),
                                 "drag_amplitude": avg.get("cd_amplitude"),
                                 "lift_amplitude": avg.get("cl_amplitude")}[k])
                for k, v in BENCHMARK_FORCES.items()]
    tip_gate = [_gate_row(k, v, {"tip_x_amplitude": tip_x,
                                 "tip_y_amplitude": tip_y}[k])
                for k, v in BENCHMARK_TIP.items()]

    def verdict(rows):
        if not regime_ok:
            return "WITHHELD"
        return "PASS" if all(r["pass_15"] for r in rows) else "FAIL"

    worst = lambda rows: max((abs(r["rel_error"]) for r in rows
                              if r["rel_error"] is not None), default=None)

    converged = hist.get("coupling_converged", [])
    iters = payload.get("iterations", [])
    return {
        "artifact": payload.get("_source"),
        "n_steps": int(n),
        "dt": dt,
        "t_end": float(t[-1]),
        "dx": payload.get("dx"),
        "aborted": bool(payload.get("aborted", False)),
        "warmup_frac": warmup_frac,
        "warmup_samples": warm,
        "window_start_t": float(t[warm]) if warm < n else None,
        "window_end_t": float(t[-1]),
        "window": per,
        "regime_ok": bool(regime_ok),
        "coefficients": {
            "scale_0.5_rho_U2_D": float(scale),
            "u_mean": u_mean,
            "rho_f": rho_f,
            "D": D,
            "fit_arm": fit,
            "avg_arm": avg,
            "cd_viscous": cd_visc,
            "cd_fit_plus_viscous": cd_fit_total,
        },
        "tip": {"tip_x_amplitude": tip_x, "tip_y_amplitude": tip_y},
        "gates": {
            "fit_arm": {"rows": fit_gate, "worst_rel_error": worst(fit_gate),
                        "verdict": verdict(fit_gate)},
            "avg_arm": {"rows": avg_gate, "worst_rel_error": worst(avg_gate),
                        "verdict": verdict(avg_gate)},
            "tip": {"rows": tip_gate, "worst_rel_error": worst(tip_gate),
                    "verdict": verdict(tip_gate)},
            "fit_plus_viscous_cd_mean": (
                _gate_row("drag_mean", BENCHMARK_FORCES["drag_mean"],
                          cd_fit_total) if cd_fit_total is not None else None),
        },
        "coupling": {
            "mean_iterations": float(np.mean(iters)) if iters else None,
            "converged_fraction": float(np.mean(converged)) if converged else None,
        },
        "cross_check_compare_body_forces": compare_body_forces(fit),
        "cross_check_compare_to_benchmark": compare_to_benchmark(
            {"tip_x_amplitude": tip_x, "tip_y_amplitude": tip_y}),
        "thresholds": {"accept": GATE_ACCEPT, "report": GATE_REPORT},
    }


def _print_table(v: dict) -> None:
    print(f"steps={v['n_steps']}  dt={v['dt']:.3e}s  t_end={v['t_end']:.3f}s  "
          f"dx={v['dx']}  aborted={v['aborted']}")
    w = v["window"]
    print(f"window: t=[{v['window_start_t']:.3f}, {v['window_end_t']:.3f}]s  "
          f"period={w['period_s']}  periods={w['periods_in_window']}")
    print(f"{'quantity':22s} {'ref':>10s} {'got':>12s} {'rel':>9s}  gate")
    for group in ("fit_arm", "avg_arm", "tip"):
        for r in v["gates"][group]["rows"]:
            got = "n/a" if r["computed"] is None else f"{r['computed']:.5g}"
            rel = "n/a" if r["rel_error"] is None else f"{100*r['rel_error']:+.2f}%"
            print(f"{group+'/'+r['quantity']:22s} {r['reference']:10.5g} "
                  f"{got:>12s} {rel:>9s}  "
                  f"{'PASS' if r['pass_15'] else 'FAIL'}"
                  f"{' (<3%)' if r['pass_3'] else ''}")
    print("verdicts:", {g: v["gates"][g]["verdict"]
                        for g in ("fit_arm", "avg_arm", "tip")})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", default="docs/xval_fsi2/body_force_verdict.json")
    ap.add_argument("--warmup-frac", type=float, default=0.5)
    args = ap.parse_args()

    src = Path(args.artifact)
    if not src.is_file():
        print(f"artefact not found: {src}", file=sys.stderr)
        return 2
    payload = json.loads(src.read_text())
    if "history" not in payload:
        print("artefact has no 'history'; rerun the driver", file=sys.stderr)
        return 2
    payload["_source"] = str(src)
    v = build_verdict(payload, warmup_frac=args.warmup_frac)
    _print_table(v)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())