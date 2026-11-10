"""Figures for the Rayleigh-Plateau verification report (matplotlib -> PNG).

Reads ``report.json`` produced by ``benchmarks/rayleigh_plateau/run.py`` so every
plotted number is the machine-written one (no hand transcription).

    python benchmarks/rayleigh_plateau/figures.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def fig_dispersion(report, path):
    from tensorsph import rayleigh_plateau as rp
    sigma, rho, radius = 0.072, 1000.0, 0.006
    x = [i * 0.005 for i in range(1, 201)]
    omega = [rp.dispersion_rate(sigma, rho, radius, xi / radius) for xi in x]
    naive = [math.sqrt(max(sigma / (rho * radius ** 3) * xi * (1 - xi * xi), 0.0))
             for xi in x]
    x_star = report["analytic"]["most_unstable_kR"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, omega, lw=2, label="Rayleigh (Bessel factor)")
    ax.plot(x, naive, "--", lw=1.4, label=r"naive $x(1-x^2)$")
    ax.axvline(x_star, color="crimson", ls=":", lw=1.2)
    ax.annotate(f"$kR^*={x_star:.4f}$\n$\\lambda^*/R=9.02$",
                xy=(x_star, max(omega) * 0.35), xytext=(x_star + 0.12,
                                                       max(omega) * 0.45),
                color="crimson", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1))
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_xlabel("$kR$")
    ax.set_ylabel(r"$\omega$ [1/s]")
    ax.set_title("Inviscid capillary dispersion of a liquid cylinder ($R=6$ mm)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_curvature(report, path):
    rows = report["curvature"]
    estimators = ["profile", "mls", "axisym"]
    resolutions = sorted({r["radius_over_dx"] for r in rows})
    width = 0.8 / max(1, len(estimators))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for idx, est in enumerate(estimators):
        vals = [100.0 * next((r["relative_error"] for r in rows
                              if r["estimator"] == est
                              and r["radius_over_dx"] == res), float("nan"))
                for res in resolutions]
        ax.bar([i + idx * width for i in range(len(resolutions))], vals,
               width=width, label=est)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(15.0, color="grey", ls="--", lw=0.8)
    ax.axhline(-15.0, color="grey", ls="--", lw=0.8)
    ax.set_xticks([i + width for i in range(len(resolutions))])
    ax.set_xticklabels([f"{int(r)}" for r in resolutions])
    ax.set_xlabel("$R/\\Delta x$")
    ax.set_ylabel(r"$\kappa R$ error vs $1/R$ [%]")
    ax.set_title("CSF curvature on a static cylinder (15 % gate dashed)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_initial_condition(report, path):
    rows = report["initial_condition"]
    labels = [f"{int(r['radius_over_dx'])}" for r in rows]
    measured = [r["epsilon_measured"] for r in rows]
    target = [r["epsilon_target"] for r in rows]
    null = [r["null_mode"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    idx = range(len(rows))
    ax.bar([i - 0.25 for i in idx], target, width=0.25, label="target $\\epsilon$")
    ax.bar(list(idx), measured, width=0.25, label="measured mode")
    ax.bar([i + 0.25 for i in idx], null, width=0.25, label="non-resonant mode")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels)
    ax.set_xlabel("$R/\\Delta x$")
    ax.set_ylabel("relative radial amplitude")
    ax.set_title("Initial condition carries the requested mode (3-D lattice)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_growth(report, path):
    run = report["column_run"]
    t = run["times"]
    amp = run["amplitudes"]
    null = run["null_amplitudes"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(t, amp, "o-", ms=3.5, lw=1.4, label="mode amplitude")
    ax.semilogy(t, null, "s--", ms=3, lw=1.1, label="non-resonant (3.5 k)")
    omega = run["omega_analytic"]
    a0 = amp[0]
    ax.semilogy([0, t[-1]], [a0, a0 * math.exp(omega * t[-1])], "k:",
                lw=1.2, label=f"Rayleigh $e^{{\\omega t}}$, $\\omega$={omega:.2f}/s")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("relative radial amplitude")
    title = (f"Column run $R/\\Delta x$={run['radius_over_dx']:.0f}, "
             f"$kR$={run['kR']:.4f}, $\\epsilon$={run['metadata']['epsilon']}")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)
    note = (f"measured $\\omega$ = {run['omega_measured']:.2f}/s, "
            f"Rayleigh = {omega:.2f}/s\n"
            f"max $|\\rho/\\rho_0-1|$ = {run['max_density_error']:.2f}, "
            f"stability gate: "
            f"{'pass' if run['stability']['pass'] else 'FAIL'}")
    ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    report = json.loads((HERE / "report.json").read_text())
    OUT.mkdir(exist_ok=True)
    fig_dispersion(report, OUT / "dispersion.png")
    fig_curvature(report, OUT / "curvature.png")
    fig_initial_condition(report, OUT / "initial_condition.png")
    fig_growth(report, OUT / "growth.png")
    print("wrote", *(str(p) for p in sorted(OUT.glob("*.png"))), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
