"""Figure for the m4659 re-validation: three assertions, three measurements.

Reads ``docs/reports/m4659_probe.json`` (written by ``m4659_probe.py``) so every
plotted number is machine-injected, and writes
``docs/figures/m4659_revalidation.png``.

    PYTHONPATH=$PWD/src python docs/reports/m4659_make_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = json.loads((HERE / "m4659_probe.json").read_text())
OUT = ROOT / "docs" / "figures" / "m4659_revalidation.png"
GATE = 1e-4
MACHINE_EPS = np.finfo(float).eps


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # ---- (a) kernel-sum bias vs h/dx, raw and calibrated ---------------------
    ax = axes[0]
    for kind, marker in (("square", "o"), ("hex", "s")):
        rows = [r for r in DATA["sweep"] if r["lattice"] == kind]
        hf = np.array([r["h_over_dx"] for r in rows])
        raw = np.abs(np.array([r["bias_raw"] for r in rows]))
        cal = np.abs(np.array([r["bias_calibrated"] for r in rows]))
        ax.semilogy(hf, np.maximum(raw, 1e-17), marker + "-", label=f"{kind} raw")
        ax.semilogy(hf, np.maximum(cal, 1e-17), marker + ":", alpha=0.7,
                    label=f"{kind} calibrated")
    ship = DATA["shipped_square"]
    ax.semilogy([ship["h_over_dx"]], [abs(ship["bias_pool_interior"])], "k*", ms=14,
                label=f"shipped make_particles ({abs(ship['bias_pool_interior']):.2e})")
    ax.axhline(GATE, color="crimson", ls="--", lw=1.2, label="m2971 gate 1e-4")
    ax.axhline(MACHINE_EPS, color="grey", ls="-", lw=0.8,
               label=f"float64 eps {MACHINE_EPS:.1e}")
    ax.set_xlabel("h / dx")
    ax.set_ylabel("|C - 1|  (kernel-sum bias)")
    ax.set_title("(a) initial-condition bias: calibration kills it")
    ax.set_ylim(1e-17, 1e-1)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower left")

    # ---- (b) wall pressure: solver vs closed form ---------------------------
    ax = axes[1]
    ms = np.array([r["mass_scale"] for r in DATA["wall"]])
    rhs = np.array([r["acc_rhs"] for r in DATA["wall"]])
    cf = np.array([r["acc_closed_form"] for r in DATA["wall"]])
    ax.plot(ms, cf, "k-", lw=1.4, label="closed form  p_f + rho_f g dy")
    ax.plot(ms, rhs, "o", ms=9, mfc="none", mew=1.8, label="tensorsph rhs()")
    for r in DATA["wall"]:
        ax.annotate(f"rel.err {r['relerr']:.1e}",
                    (r["mass_scale"], r["acc_rhs"]), textcoords="offset points",
                    xytext=(8, 10), fontsize=8)
    ax.set_xlabel("wall-particle mass / (rho0 dx^2)")
    ax.set_ylabel("fluid vertical acceleration  [m/s^2]")
    ax.set_title("(b) Adami eq. (11): two-term wall pressure")
    ax.set_xticks(ms)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (c) initial mass: uncalibrated branch and factor agreement ---------
    ax = axes[2]
    dens = DATA["density"]
    labels = ["m_off / (rho0 dx^2)", "|m_on/m_off - k| / k"]
    values = [abs(dens["m_off_over_rho0_dx2"] - 1.0) + 1e-17, dens["factor_relerr"]]
    ax.bar(labels, values, color=["tab:blue", "tab:green"], width=0.55)
    ax.axhline(1e-5, color="crimson", ls="--", lw=1.2,
               label="torch.allclose default rtol 1e-5")
    ax.set_yscale("log")
    ax.set_ylabel("relative deviation")
    ax.set_title("(c) density_calibration is explicit in the test")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.2e}", (i, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle(
        "m4659 re-validation on main: kernel-sum bias, wall pressure and initial mass"
        f"  [{DATA['arch']}, torch {DATA['torch']}, {DATA['dtype']}]",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
