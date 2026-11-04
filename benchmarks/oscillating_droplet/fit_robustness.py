"""Validate the mode-2 frequency *instrument* before quoting its reading.

Why this exists
---------------
The release benchmark measures the mode-2 (Rayleigh-Lamb) frequency of an
elliptical drop and reports the error against ``omega^2 = 8 sigma / (rho R^3)``.
Two CI runs of *identical, deterministic* code recorded ``+8.64 %`` (x86_64)
and ``-31.78 %`` (loong64); that pair is currently pinned as strict-xfail in
``tests/test_oscillating_droplet.py``.  A 40-point spread between two machines
cannot be physics, so before re-baselining the pin the instrument itself has to
be examined.  This benchmark does that on three levels:

  A. calibration against closed-form traces.  Exact damped cosines with a known
     ``omega``, sampled exactly like the recorded trace: noiseless, noisy, and
     *rectified* (``|cos|``, which is what the legacy observable measures).
     This separates the resolution floor from the bias of the observable.
  B. diagnosis of the recorded trace (``--trace``, the committed CI fixture).
     Goodness of the damped-cosine model (``r2``), the sign of the fitted
     damping, the number of sign changes, and the agreement of five independent
     estimators: damped-cosine fit of ``|s|``, fit of ``s_signed``,
     zero-crossing of both, half the kinetic-energy frequency, and the spectral
     peak.
  C. stability of the point estimate.  Refits on nested windows, and ``1e-12``
     relative perturbations that mimic the last-bit differences between BLAS
     builds -- i.e. the mechanism by which one machine could report ``+8.64 %``
     while another reports ``-31.78 %``.

The observable fix (signed mode-2 amplitude, ``s_signed``) and the basin-aware
fit are in ``run.py``; this benchmark is the evidence that they were needed and
that they work.  Nothing outside ``benchmarks/oscillating_droplet/`` is used to
produce it.

Usage
-----
    PYTHONPATH=src python benchmarks/oscillating_droplet/fit_robustness.py
    PYTHONPATH=src python benchmarks/oscillating_droplet/fit_robustness.py --quick
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUN = HERE / "run.py"
DEFAULT_TRACE = HERE / "trace_ci_dx0p001.json"

#: The two numbers CI recorded for identical code (tests/test_oscillating_droplet.py).
RECORDED_X86_PCT = 8.64
RECORDED_LOONG_PCT = -31.78


def load_harness():
    spec = importlib.util.spec_from_file_location("oscillating_droplet_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# A. calibration against closed-form traces
# --------------------------------------------------------------------------- #
def synthetic_trace(t: list[float], omega: float, lam: float, phase: float = 0.0,
                    amplitude: float = 0.10, offset: float = 0.0,
                    rectified: bool = False, noise: float = 0.0,
                    seed: int = 0) -> list[float]:
    """``offset + amp e^{-lam t} cos(omega t + phase)``, optionally rectified.

    ``rectified=True`` returns ``|...|``, which is what ``s_moment`` measures
    (``A >= B`` by construction, see ``run.mode_amplitude``).  ``noise`` is
    relative to the amplitude and seeded, so every number is reproducible.
    """
    ts = torch.tensor(t, dtype=torch.float64)
    wave = amplitude * torch.exp(-lam * ts) * torch.cos(omega * ts + phase)
    if rectified:
        wave = wave.abs()
    signal = offset + wave
    if noise > 0.0:
        generator = torch.Generator().manual_seed(seed)
        signal = signal + noise * amplitude * torch.randn(
            signal.shape, generator=generator, dtype=torch.float64)
    return [float(v) for v in signal]


def calibrated_error(run, t, omega, lam, *, rectified=False, noise=0.0,
                     phase=0.0, seed=0, starts=None) -> dict:
    """Fit a synthetic trace whose ``omega`` is known; return the error."""
    starts = run.FIT_STARTS if starts is None else starts
    s = synthetic_trace(t, omega, lam, phase=phase, rectified=rectified,
                        noise=noise, seed=seed)
    fit = run.damped_cosine_fit(t, s, omega, starts=starts)
    return {"error_pct": (fit["omega"] - omega) / omega * 100.0,
            "omega": fit["omega"], "r2": fit["r2"],
            "lambda_per_s": fit["lambda_per_s"],
            "ambiguity_rms_gap": fit["ambiguity_rms_gap"]}


def calibration(run, t, omega, lam, phases, noise_levels, seeds) -> dict:
    """A: resolution floor, rectification bias, noise scatter."""
    exact = [calibrated_error(run, t, omega, lam, phase=p) for p in phases]
    rectified = [calibrated_error(run, t, omega, lam, phase=p, rectified=True)
                 for p in phases]
    noise = []
    for level in noise_levels:
        errs = [calibrated_error(run, t, omega, lam, noise=level, seed=k)["error_pct"]
                for k in range(seeds)]
        noise.append({"noise_relative": level, "n": seeds,
                      "error_pct_min": min(errs), "error_pct_max": max(errs),
                      "error_pct_mean": statistics.fmean(errs),
                      "error_pct_sd": statistics.pstdev(errs) if seeds > 1 else 0.0})
    return {
        "exact_error_pct": [row["error_pct"] for row in exact],
        "exact_r2": [row["r2"] for row in exact],
        "rectified_error_pct": [row["error_pct"] for row in rectified],
        "rectified_r2": [row["r2"] for row in rectified],
        "noise_sweep": noise,
    }


# --------------------------------------------------------------------------- #
# B. diagnosis of a recorded trace
# --------------------------------------------------------------------------- #
def sign_changes(values: list[float]) -> int:
    """Number of sign changes about the mean -- 0 means the trace is rectified."""
    mean = statistics.fmean(values)
    centred = [v - mean for v in values]
    return sum(1 for a, b in zip(centred, centred[1:]) if a * b < 0.0)


def fft_dominant_ratio(t: list[float], s: list[float], omega_ref: float) -> float:
    """Dominant non-DC spectral peak, in units of ``omega_ref``.

    No fit is involved, so this is the estimator that cannot be talked into a
    basin: for a sinusoidal oscillation it reads 1.0, for a rectified one 2.0.
    """
    ts = torch.tensor(t, dtype=torch.float64)
    ss = torch.tensor(s, dtype=torch.float64)
    ss = ss - ss.mean()
    window = torch.hann_window(ss.numel(), dtype=torch.float64)
    power = torch.fft.rfft(ss * window).abs() ** 2
    freqs = torch.fft.rfftfreq(ss.numel(), d=float(ts[1] - ts[0]))
    power[0] = 0.0
    peak = float(freqs[int(torch.argmax(power))])
    return 2.0 * math.pi * peak / omega_ref


def estimators(run, t, s_moment, s_signed, ke, omega_ref) -> dict:
    """Five independent readings of the same oscillation."""
    fit_mom = run.damped_cosine_fit(t, s_moment, omega_ref)
    fit_sig = run.damped_cosine_fit(t, s_signed, omega_ref)
    zc_mom = run.zero_crossing_period(t, s_moment)
    zc_sig = run.zero_crossing_period(t, s_signed)
    fit_ke = run.damped_cosine_fit(t, ke, 2.0 * omega_ref)

    def pct(omega):
        if not math.isfinite(omega):
            return float("nan")
        return (omega - omega_ref) / omega_ref * 100.0

    return {
        "fit_moment": {"omega": fit_mom["omega"], "error_pct": pct(fit_mom["omega"]),
                       "r2": fit_mom["r2"], "lambda_per_s": fit_mom["lambda_per_s"],
                       "ambiguity_rms_gap": fit_mom["ambiguity_rms_gap"],
                       "basins": fit_mom["starts"]},
        "fit_signed": {"omega": fit_sig["omega"], "error_pct": pct(fit_sig["omega"]),
                       "r2": fit_sig["r2"], "lambda_per_s": fit_sig["lambda_per_s"],
                       "ambiguity_rms_gap": fit_sig["ambiguity_rms_gap"],
                       "basins": fit_sig["starts"]},
        "zero_crossing_moment": {"omega": zc_mom["omega"],
                                 "error_pct": pct(zc_mom["omega"])},
        "zero_crossing_signed": {"omega": zc_sig["omega"],
                                 "error_pct": pct(zc_sig["omega"])},
        "kinetic_energy_half": {"omega": 0.5 * fit_ke["omega"],
                                "error_pct": pct(0.5 * fit_ke["omega"]),
                                "r2": fit_ke["r2"]},
        "spectral_peak_ratio": fft_dominant_ratio(t, s_moment, omega_ref),
    }


def window_sensitivity(run, t, s, omega_ref, fractions) -> list[dict]:
    """Refit on nested windows: a real frequency does not depend on the window."""
    ts = torch.tensor(t, dtype=torch.float64)
    span = float(ts[-1] - ts[0])
    out = []
    for fraction in fractions:
        keep = [i for i, value in enumerate(t) if value - t[0] <= fraction * span]
        if len(keep) < 8:
            continue
        window = [s[i] for i in keep]
        fit = run.damped_cosine_fit([t[i] for i in keep], window, omega_ref)
        out.append({"window_fraction": fraction, "n_samples": len(keep),
                    "omega": fit["omega"],
                    "error_pct": (fit["omega"] - omega_ref) / omega_ref * 100.0,
                    "r2": fit["r2"],
                    "ambiguity_rms_gap": fit["ambiguity_rms_gap"]})
    return out


def perturbation_sweep(run, t, s, omega_ref, trials, relative=1.0e-12) -> list[dict]:
    """``1e-12`` relative perturbations: the size of a BLAS last-bit difference."""
    base = torch.tensor(s, dtype=torch.float64)
    rows = []
    for trial in range(trials):
        generator = torch.Generator().manual_seed(7000 + trial)
        jitter = relative * torch.randn(base.numel(), generator=generator,
                                        dtype=torch.float64)
        perturbed = [float(b + j) for b, j in zip(base, jitter)]
        row = {"trial": trial}
        for label, starts in (("single_start", 1), ("multi_start", run.FIT_STARTS)):
            fit = run.damped_cosine_fit(t, perturbed, omega_ref, starts=starts)
            row[f"{label}_error_pct"] = (fit["omega"] - omega_ref) / omega_ref * 100.0
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# C. verdict
# --------------------------------------------------------------------------- #
def verdict(trace, est, calib, windows_moment, windows_signed, perturbation,
            omega_ref) -> dict:
    readings = [est["fit_moment"]["error_pct"], est["zero_crossing_moment"]["error_pct"],
                est["kinetic_energy_half"]["error_pct"],
                est["fit_signed"]["error_pct"], est["zero_crossing_signed"]["error_pct"]]
    finite = [r for r in readings if math.isfinite(r)]
    single = [row["single_start_error_pct"] for row in perturbation]
    multi = [row["multi_start_error_pct"] for row in perturbation]
    recorded_spread = RECORDED_X86_PCT - RECORDED_LOONG_PCT
    return {
        "periods_recorded": trace["periods"],
        "n_samples": len(trace["t"]),
        "rayleigh_relative_to_omega_pct":
            100.0 * (2.0 * math.pi / (trace["t"][-1] - trace["t"][0])) / omega_ref,
        "sign_changes_moment": sign_changes(trace["s_moment"]),
        "sign_changes_signed": sign_changes(trace["s_signed"]),
        "moment_model_r2": est["fit_moment"]["r2"],
        "signed_model_r2": est["fit_signed"]["r2"],
        "moment_fitted_damping_per_s": est["fit_moment"]["lambda_per_s"],
        "signed_fitted_damping_per_s": est["fit_signed"]["lambda_per_s"],
        "estimator_error_pct_spread": max(finite) - min(finite),
        "estimator_errors_pct": {
            "fit_moment": est["fit_moment"]["error_pct"],
            "zero_crossing_moment": est["zero_crossing_moment"]["error_pct"],
            "kinetic_energy_half": est["kinetic_energy_half"]["error_pct"],
            "fit_signed": est["fit_signed"]["error_pct"],
            "zero_crossing_signed": est["zero_crossing_signed"]["error_pct"],
        },
        "window_error_pct_moment": [row["error_pct"] for row in windows_moment],
        "window_error_pct_signed": [row["error_pct"] for row in windows_signed],
        "window_spread_pct_moment": (max(r["error_pct"] for r in windows_moment)
                                     - min(r["error_pct"] for r in windows_moment)),
        "window_spread_pct_signed": (max(r["error_pct"] for r in windows_signed)
                                     - min(r["error_pct"] for r in windows_signed)),
        "round_off_spread_single_pct": max(single) - min(single),
        "round_off_spread_multi_pct": max(multi) - min(multi),
        "recorded_cross_arch_spread_pct": recorded_spread,
        "recorded_pair_pct": [RECORDED_LOONG_PCT, RECORDED_X86_PCT],
        "recorded_spread_inside_estimator_spread": bool(
            abs(recorded_spread) <= max(finite) - min(finite)),
        "forward_model_error_pct_exact": calib["exact_error_pct"],
        "forward_model_error_pct_rectified": calib["rectified_error_pct"],
        "rectification_bias_pct": statistics.fmean(calib["rectified_error_pct"]),
    }


def plot(trace, est, calib, windows_moment, windows_signed, perturbation, out, omega_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.4))
    period = 2.0 * math.pi / omega_ref
    tt = [value / period for value in trace["t"]]

    ax = axes[0][0]
    ax.plot(tt, trace["s_moment"], lw=1.1, label="$s_{moment}=|\\varepsilon|$ (legacy)")
    ax.plot(tt, trace["s_signed"], lw=1.1, label="$s_{signed}$ (this work)")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("t / T (analytic)")
    ax.set_ylabel("mode-2 amplitude")
    ax.set_title("recorded CI trace: the legacy observable is rectified")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    labels = ["fit |s|", "zero-cross\n|s|", "KE/2", "fit signed", "zero-cross\nsigned"]
    values = [est["fit_moment"]["error_pct"], est["zero_crossing_moment"]["error_pct"],
              est["kinetic_energy_half"]["error_pct"], est["fit_signed"]["error_pct"],
              est["zero_crossing_signed"]["error_pct"]]
    colors = ["tab:red", "tab:orange", "tab:purple", "tab:blue", "tab:cyan"]
    ax.bar(range(len(values)), values, color=colors)
    for value, label in ((RECORDED_X86_PCT, "+8.64"), (RECORDED_LOONG_PCT, "-31.78")):
        ax.axhline(value, color="grey", ls=":", lw=1.0)
        ax.annotate(label, (2.6, value), fontsize=7, color="grey")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("$\\omega$ error vs analytic [%]")
    ax.set_title("same trace, five estimators")

    ax = axes[0][2]
    ax.semilogx([r["noise_relative"] for r in calib["noise_sweep"]],
                [r["error_pct_mean"] for r in calib["noise_sweep"]], "o-",
                label="exact, noise sweep")
    ax.axhline(statistics.fmean(calib["rectified_error_pct"]), color="tab:red", ls="--",
               label=f"rectified signal ({statistics.fmean(calib['rectified_error_pct']):+.1f} %)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("relative noise amplitude")
    ax.set_ylabel("fitted $\\omega$ error [%]")
    ax.set_title("A: calibration on closed-form traces")
    ax.legend(fontsize=8)

    ax = axes[1][0]
    ax.plot([r["window_fraction"] for r in windows_moment],
            [r["error_pct"] for r in windows_moment], "x-", label="|s| (legacy)")
    ax.plot([r["window_fraction"] for r in windows_signed],
            [r["error_pct"] for r in windows_signed], "o-", label="signed")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("window length [periods of the record]")
    ax.set_ylabel("fitted $\\omega$ error [%]")
    ax.set_title("C: window sensitivity")
    ax.legend(fontsize=8)

    ax = axes[1][1]
    ax.plot([r["trial"] for r in perturbation],
            [r["single_start_error_pct"] for r in perturbation], "x-",
            label="single start")
    ax.plot([r["trial"] for r in perturbation],
            [r["multi_start_error_pct"] for r in perturbation], "o-",
            label="multi-start (shipped)")
    for value in (RECORDED_X86_PCT, RECORDED_LOONG_PCT):
        ax.axhline(value, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("round-off perturbation ($\\pm 10^{-12}$ relative)")
    ax.set_ylabel("fitted $\\omega$ error [%]")
    ax.set_title("C: sensitivity to last-bit differences")
    ax.legend(fontsize=8)

    ax = axes[1][2]
    ax.plot(tt, trace["kinetic_energy_at_start"] if False else trace["ke"], lw=1.1,
            color="tab:green")
    ax.set_xlabel("t / T (analytic)")
    ax.set_ylabel("kinetic energy [J]")
    ax.set_title("kinetic energy: monotone growth, not an oscillation")

    fig.suptitle(
        "mode-2 frequency instrument, CI release trace "
        f"({trace['periods']:.2f} periods, Rayleigh resolution "
        f"{verdict_rayleigh(trace, omega_ref):.1f} % of $\\omega$)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=125)
    plt.close(fig)


def verdict_rayleigh(trace, omega_ref):
    return 100.0 * (2.0 * math.pi / (trace["t"][-1] - trace["t"][0])) / omega_ref


def load_trace(path: Path, omega_ref: float) -> dict:
    data = json.loads(Path(path).read_text())
    trace = data["trace"]
    t = [float(v) for v in trace["t"]]
    return {"t": t,
            "s_moment": [float(v) for v in trace["s_second_moment"]],
            "s_signed": [float(v) for v in trace["s_signed"]],
            "ke": [float(v) for v in trace["kinetic_energy_J"]],
            "periods": t[-1] / (2.0 * math.pi / omega_ref),
            "setup": data.get("setup", {}),
            "frequency_reported": data.get("frequency", {})}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--out", default=str(HERE / "fit_robustness_report.json"))
    parser.add_argument("--figure", default=str(HERE / "fit_robustness.png"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    run = load_harness()
    omega_ref = run.analytic_omega(0.01, 7.2, 1000.0)
    trace = load_trace(Path(args.trace), omega_ref)
    t = trace["t"]
    lam = 2.0 * 5.0e-5 * 4.0 / 0.01 ** 2

    phases = [0.0, math.pi / 4, math.pi / 2] if args.quick else \
        [0.0, math.pi / 8, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    trials = 6 if args.quick else 12
    seeds = 3 if args.quick else 5
    fractions = [0.4, 0.6, 0.8, 1.0]

    calib = calibration(run, t, omega_ref, lam, phases, [0.0, 1e-3, 1e-2], seeds)
    est = estimators(run, t, trace["s_moment"], trace["s_signed"], trace["ke"], omega_ref)
    windows_moment = window_sensitivity(run, t, trace["s_moment"], omega_ref, fractions)
    windows_signed = window_sensitivity(run, t, trace["s_signed"], omega_ref, fractions)
    perturbation = perturbation_sweep(run, t, trace["s_moment"], omega_ref, trials)
    summary = verdict(trace, est, calib, windows_moment, windows_signed,
                      perturbation, omega_ref)

    report = {"trace": {k: v for k, v in trace.items() if k != "t"},
              "calibration": calib, "estimators": est,
              "window_sensitivity_moment": windows_moment,
              "window_sensitivity_signed": windows_signed,
              "round_off_perturbation": perturbation,
              "verdict": summary}
    Path(args.out).write_text(json.dumps(report, indent=2))
    plot(trace, est, calib, windows_moment, windows_signed, perturbation,
         Path(args.figure), omega_ref)
    print(json.dumps(summary, indent=2))
    print(f"report -> {args.out}")
    print(f"figure -> {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())