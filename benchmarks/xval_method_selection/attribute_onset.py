"""Attribute the onset of the wave-flume instability to a time, not a guess.

The repair mission (17906) established *that* method A diverges on CASE-WAVE-FLUME
(``max|v|/c0 = 3.55``, ``max compression 30 %``) but left open *when* it starts and
therefore *which mechanism* is admissible.  This script reads a stored record only
(no solver run, no tuning) and answers three questions with numbers:

1. **Is the wavemaker itself the driver?**  Compare the recorded paddle offset
   with the analytic first-order piston motion ``s(t) = S/2 sin(omega t)`` that the
   transfer function prescribes, and compare the largest physical orbital velocity
   with the observed ``max|v|``.  A generator that pushes harder than theory would
   show up as a stroke amplitude mismatch; a numerical instability shows up as
   ``max|v|`` orders of magnitude above the orbital velocity.
2. **When does ``max|v|`` leave the physical band?**  First crossing times of
   ``max|v|`` through a ladder of thresholds, plus a log-linear fit of the growth
   of the peak speed (an exponential growth rate is the fingerprint of a numerical
   instability, not of a wave).
3. **Does the onset coincide with the wave reaching the sponge?**  Compare the
   first crossing times with the analytic wave-front arrival times at the probe
   stations and at the sponge entrance, ``t = x / c``.  Onset *before* the front
   arrives cannot be blamed on the absorbing layer.

Nothing here changes a verdict: it produces the timing evidence the report needs,
and it can be re-run on any record of the same schema.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/attribute_onset.py \
        --record docs/reports/artifacts_wave_flume/wf_D20_unstable.json \
        --out docs/reports/xval_method_selection_onset.json \
        --figure docs/figures/xval_method_selection_m18598/onset.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

THRESHOLDS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def _orbital_velocity(height: float, k: float, depth: float, omega: float) -> float:
    """Maximum linear orbital speed at the still-water surface.

    ``u_max = (H/2) omega cosh(k(z+h))/sinh(k h)`` evaluated at ``z = 0``
    (Dean & Dalrymple 1991, eq. 4.9), i.e. ``(H/2) omega coth(kh)``.
    """
    return 0.5 * height * omega / math.tanh(k * depth)


def _first_crossings(times: np.ndarray, series: np.ndarray, thresholds) -> dict:
    out = {}
    for level in thresholds:
        above = np.nonzero(series > level)[0]
        out[f"{level:g}"] = (float(times[above[0]]) if above.size else None)
    return out


def _growth_rate(times: np.ndarray, series: np.ndarray, lo: float, hi: float) -> dict:
    """Least-squares slope of ``ln(max|v|)`` over ``lo <= max|v| <= hi``."""
    mask = (series >= lo) & (series <= hi)
    if mask.sum() < 5:
        return {"points": int(mask.sum()), "rate_per_s": None, "r2": None,
                "e_folding_s": None}
    x = times[mask]
    y = np.log(series[mask])
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    ss_res = float(((y - prediction) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {"points": int(mask.sum()), "rate_per_s": float(slope),
            "r2": float(r2), "e_folding_s": float(1.0 / slope) if slope > 0 else None}


def analyse(record: dict) -> dict:
    cfg = record["config"]
    spec = record["spec"]
    times = np.asarray(record["times"], dtype=float)
    speed = np.asarray(record["speed_history"], dtype=float)
    compression = np.asarray(record["compression_history"], dtype=float)
    paddle = np.asarray(record["paddle_offset"], dtype=float)

    c0 = float(cfg["sound_speed"])
    omega = float(spec["omega"])
    k = float(spec["k"])
    depth = float(cfg["depth"])
    stroke = float(spec["stroke"])
    height = float(cfg["wave_height"])
    celerity = float(spec["celerity"])

    # (1) wavemaker fidelity: after the ramp the piston offset must be a pure
    # harmonic of amplitude S/2 at the wave frequency.  Fit
    # s = C + A cos(omega t) + B sin(omega t) on the settled part and report the
    # amplitude and phase against the analytic S/2 sin(omega t); a generator that
    # pushes harder than the transfer function would show up as an amplitude bias.
    settle = float(cfg.get("settle_time", 0.0)) + 2.0 * float(cfg["wave_period"])
    mask = times >= settle
    fidelity = {"settle_cut_s": settle, "samples": int(mask.sum())}
    if mask.sum() >= 10:
        basis = np.stack([np.ones(mask.sum()),
                          np.cos(omega * times[mask]),
                          np.sin(omega * times[mask])], axis=1)
        coeffs, *_ = np.linalg.lstsq(basis, paddle[mask], rcond=None)
        fitted = basis @ coeffs
        amplitude = float(math.hypot(coeffs[1], coeffs[2]))
        # phase of the fitted harmonic relative to the analytic sin(omega t)
        phase = float(math.atan2(coeffs[1], coeffs[2]))
        fidelity.update({
            "fitted_amplitude_m": amplitude,
            "analytic_amplitude_m": 0.5 * stroke,
            "amplitude_ratio": amplitude / (0.5 * stroke),
            "offset_bias_m": float(coeffs[0]),
            "phase_error_rad": phase,
            # the driver evaluates the prescribed motion at
            # wave_time = elapsed - settle_time (wave_flume.run_case), so the
            # analytic motion carries a designed phase lag omega*settle_time;
            # the observation is checked against that, not against zero.
            "phase_expected_rad": (-omega * float(cfg.get("settle_time", 0.0))) % (2 * math.pi),
            "phase_matches_designed_lag": bool(abs(
                (phase - (-omega * float(cfg.get("settle_time", 0.0))))
                % (2 * math.pi)) < 1e-6 or abs(
                (phase - (-omega * float(cfg.get("settle_time", 0.0))))
                % (2 * math.pi) - 2 * math.pi) < 1e-6),
            "relative_residual": float(np.abs(paddle[mask] - fitted).max()
                                       / max(amplitude, 1e-12)),
        })
    orbital = _orbital_velocity(height, k, depth, omega)

    # (2) threshold crossings and the exponential growth rate of the peak speed.
    crossings = _first_crossings(times, speed, THRESHOLDS)
    growth = _growth_rate(times, speed, 1.0, 20.0)

    # (3) analytic wave-front arrival times: the front travels at the linear
    # celerity, so a station at x is reached at t = x / c.
    probes = [float(x) for x in record["probe_x"]]
    sponge_start = float(cfg["flume_length"]) - float(cfg["sponge_length"])
    arrivals = {
        "probes_s": [x / celerity for x in probes],
        "sponge_entrance_s": sponge_start / celerity,
        "downstream_wall_s": float(cfg["flume_length"]) / celerity,
    }

    # compression is the second, independent instability marker: report when the
    # pre-registered 10 % gate is first exceeded.
    compression_cross = {}
    for level in (0.01, 0.05, 0.10, 0.20):
        above = np.nonzero(compression > level)[0]
        compression_cross[f"{level:.2f}"] = (float(times[above[0]])
                                             if above.size else None)

    onset_1mach = None
    if crossings.get(f"{c0:g}") is not None:
        onset_1mach = crossings[f"{c0:g}"]
    else:
        above = np.nonzero(speed > c0)[0]
        onset_1mach = float(times[above[0]]) if above.size else None

    return {
        "record": {
            "simulated_time_s": float(times[-1]),
            "samples": int(times.size),
            "particles": int(cfg["particle_count"]),
            "dt_requested_s": float(cfg["dt_requested"]),
            "dt_final_s": float(cfg["dt_final"]),
            "sound_speed_m_s": c0,
            "depth_m": depth,
            "wave_height_m": height,
            "wavelength_m": float(spec["wavelength"]),
            "celerity_m_s": celerity,
        },
        "wavemaker_fidelity": dict(
            fidelity,
            stroke_target_m=stroke,
            recorded_peak_offset_m=float(np.abs(paddle).max()),
            note="amplitude_ratio = fitted harmonic amplitude / (S/2); "
                 "phase_error_rad = fitted phase offset against "
                 "S/2 sin(omega t); both are measured after the ramp",
        ),
        "speed_scale": {
            "linear_orbital_speed_m_s": orbital,
            "observed_max_speed_m_s": float(speed.max()),
            "ratio_observed_over_orbital": float(speed.max() / orbital),
            "mach_observed": float(speed.max() / c0),
        },
        "threshold_crossings_s": crossings,
        "peak_speed_growth": growth,
        "compression_crossings_s": compression_cross,
        "wave_front_arrivals_s": arrivals,
        "onset_vs_front": {
            "first_supersonic_s": onset_1mach,
            "sponge_entrance_s": arrivals["sponge_entrance_s"],
            "supersonic_before_sponge_arrival":
                (onset_1mach is not None
                 and onset_1mach < arrivals["sponge_entrance_s"]),
        },
    }


def figure(record: dict, result: dict, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                    # pragma: no cover
        return False
    times = np.asarray(record["times"], dtype=float)
    speed = np.asarray(record["speed_history"], dtype=float)
    compression = np.asarray(record["compression_history"], dtype=float)
    c0 = result["record"]["sound_speed_m_s"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.semilogy(times, speed, "-", lw=1.4, color="#1f4e79", label="max|v| (recorded)")
    ax.axhline(c0, color="#c00000", ls="--", lw=1.1, label=f"c0 = {c0:g} m/s (Mach 1)")
    ax.axhline(result["speed_scale"]["linear_orbital_speed_m_s"], color="#548235",
               ls=":", lw=1.1, label="linear orbital velocity")
    for label, key, colour in (("sponge entrance", "sponge_entrance_s", "#7f6000"),
                              ("downstream wall", "downstream_wall_s", "#7030a0")):
        ax.axvline(result["wave_front_arrivals_s"][key], color=colour, ls="-.",
                   lw=1.0)
        ax.annotate(label, (result["wave_front_arrivals_s"][key], speed.max() * 0.35),
                    rotation=90, fontsize=7, color=colour, ha="right")
    if result["onset_vs_front"]["first_supersonic_s"] is not None:
        ax.axvline(result["onset_vs_front"]["first_supersonic_s"], color="#c00000",
                   lw=0.8, alpha=0.6)
    ax.set_ylabel("peak speed [m/s]")
    ax.set_title("CASE-WAVE-FLUME · onset of the instability vs the wave front")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    ax2.plot(times, 100.0 * compression, "-", lw=1.3, color="#833c00")
    ax2.axhline(10.0, color="#c00000", ls="--", lw=1.0, label="10 % gate")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("max density compression [%]")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", default="docs/reports/artifacts_wave_flume/"
                                            "wf_D20_unstable.json")
    parser.add_argument("--out", default="docs/reports/"
                                       "xval_method_selection_onset.json")
    parser.add_argument("--figure", default="docs/figures/"
                                            "xval_method_selection_m18598/onset.png")
    args = parser.parse_args(argv)

    record = json.loads(Path(args.record).read_text())
    result = analyse(record)
    result["source_record"] = args.record
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    made = figure(record, result, Path(args.figure))
    print(f"wrote {out}: Mach 1 at {result['onset_vs_front']['first_supersonic_s']} s, "
          f"sponge front arrives at "
          f"{result['wave_front_arrivals_s']['sponge_entrance_s']:.2f} s, "
          f"observed/orbital speed ratio "
          f"{result['speed_scale']['ratio_observed_over_orbital']:.0f}")
    print(f"figure {args.figure}: {'ok' if made else 'skipped (no matplotlib)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())