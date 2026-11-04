"""Figures for the mode-2 admissibility study (matplotlib only, no sim)."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render(report: dict, outdir, only: str | None = None) -> None:
    import json
    import pathlib

    # full per-step histories live in diagnostics.json (report.json is trace-free)
    diag_path = pathlib.Path(outdir) / "diagnostics.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
        for key, case in report["cases"].items():
            if "trace" not in case and key in diag:
                case["trace"] = diag[key]["trace"]
    gates = report["platform_gates"]
    T_ref = None
    for key, case in report["cases"].items():
        if only and key != only:
            continue
        tr = case["trace"]
        T = case["reference"]["period_s"]
        T_ref = T
        t = tr["t"]
        R = case["setup"]["radius_m"]
        g = case["admissibility"]

        # ---- figure 1: what the fit was applied to -------------------------
        fig, ax = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True)
        ax[0].plot(t, tr["s_second_moment"], lw=1.0, color="#1f4e79")
        ax[0].axhline(case["setup"]["initial_eps"], ls=":", c="0.5", lw=0.8)
        ax[0].set_ylabel(r"$s(t)=(A-B)/(A+B)$")
        ax[0].set_title(f"mode-2 release, {key}: the published fit was applied "
                        f"to a diverging trajectory", fontsize=10)
        if g["gate_tripped"]:
            ax[0].axvline(g["t_gate_s"], color="#c00000", lw=1.2,
                          label=f"admissibility gate t={g['t_gate_s']*1e3:.1f} ms")
            ax[0].legend(loc="upper left", fontsize=8)

        ax[1].semilogy(t, tr["mach"], lw=1.0, color="#7f6000")
        ax[1].axhline(gates["mach"], ls="--", c="#c00000", lw=1.0,
                      label=f"Mach gate {gates['mach']}")
        ax[1].set_ylabel(r"$\max|v|/c_0$")
        ax[1].legend(loc="lower right", fontsize=8)

        ax[2].semilogy(t, tr["r_eff_m"], lw=1.0, color="#375623")
        ax[2].axhline(R * (1 + gates["radius_drift_pct"] / 100), ls="--",
                      c="#c00000", lw=1.0, label=f"+{gates['radius_drift_pct']:.0f}% radius")
        ax[2].axhline(R, ls=":", c="0.4", lw=0.9, label="R")
        ax[2].set_ylabel("area-equivalent radius [m]")
        ax[2].set_xlabel("t [s]")
        ax[2].legend(loc="lower right", fontsize=8)
        for a in ax:
            a.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / "admissibility_trace.png", dpi=130)
        plt.close(fig)

        # ---- figure 2: estimator comparison on a log-frequency axis --------
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        om_ref = case["reference"]["omega_rad_per_s"]
        rows = [("full trace\n(published)", case["fit_full_trace"], "#c00000"),
                ("valid window", case["fit_valid_window"], "#2e75b6")]
        xs = [0, 1]
        ax.bar([x - 0.18 for x in xs],
               [r[1]["rel_error_pct"] for r in rows], width=0.34,
               color=[r[2] for r in rows], label="damped-cosine fit")
        ax.bar([x + 0.18 for x in xs],
               [r[1].get("ke_rel_error_pct", float("nan")) for r in rows],
               width=0.34, color="#a6a6a6", label="kinetic-energy fit /2")
        for x, r in zip(xs, rows):
            ax.text(x - 0.18, r[1]["rel_error_pct"] + 1.2,
                    f"{r[1]['rel_error_pct']:.1f}%\n$R^2$={r[1]['r2']:.2f}",
                    ha="center", fontsize=8)
        ax.axhspan(-15, 15, color="#c6e0b4", alpha=0.45, label="15% gate")
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([r[0] for r in rows])
        ax.set_ylabel("frequency error vs Rayleigh-Lamb [%]")
        ax.set_title(f"{key}: same estimators, admissible vs diverged data",
                     fontsize=10)
        ax.legend(fontsize=7, ncol=3, loc="lower left")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(outdir / "estimator_comparison.png", dpi=130)
        plt.close(fig)

        # ---- figure 3: window ladder vs spectrum ---------------------------
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
        lad = case["fit_window_ladder"]
        a1.plot([r["window_periods"] for r in lad],
                [r["rel_error_pct"] for r in lad], "o-", color="#7030a0")
        a1.axhspan(-15, 15, color="#c6e0b4", alpha=0.45)
        a1.axhline(0, c="k", lw=0.8)
        if g["gate_tripped"]:
            a1.axvline(g["t_gate_s"] / T, ls="--", c="#c00000", lw=1.0,
                       label="gate")
            a1.legend(fontsize=8)
        a1.set_xlabel("fit window [analytic periods]")
        a1.set_ylabel("frequency error [%]")
        a1.set_title("window ladder", fontsize=10)
        a1.grid(alpha=0.25)

        for label, peaks, col in (("full trace", case["fft_full_trace"], "#c00000"),
                                  ("valid window", case["fft_valid_window"], "#2e75b6")):
            a2.bar([p["rel_error_pct"] for p in peaks],
                   [p["amplitude"] / max(q["amplitude"] for q in peaks)
                    for p in peaks], width=1.1, alpha=0.65, color=col,
                   label=label)
        a2.axvline(0, c="k", lw=0.9)
        a2.axvspan(-15, 15, color="#c6e0b4", alpha=0.45)
        a2.set_xlabel(r"$\omega$ relative to $\omega_2$ [%]")
        a2.set_ylabel("normalised spectral amplitude")
        a2.set_title(r"FFT of $s(t)$: full trace is multi-modal", fontsize=10)
        a2.legend(fontsize=8)
        a2.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(outdir / "window_and_spectrum.png", dpi=130)
        plt.close(fig)
    if T_ref is not None:
        print(f"figures written to {outdir}")