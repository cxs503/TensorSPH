"""Figure for the B5 report, built from ``results.json`` alone."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main() -> None:
    data = json.loads((HERE / "results.json").read_text())
    cases = data["cases"]
    series = {(s["mode"], s["dx_m"]): s for s in data["series"]}
    colors = {1: "tab:blue", 2: "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9))

    ax = axes[0]
    for (mode, dx), s in sorted(series.items()):
        if dx != data["finest_dx"]:
            continue
        ax.plot(s["t"], [q * 1e3 for q in s["q"]], color=colors[mode], lw=1.1,
                label=f"mode {mode} (column volume)")
        ql = s.get("q_level_set") or []
        lv = [(t, q * 1e3) for t, q in zip(s["t"], ql) if q == q]
        if lv:
            ax.plot([v[0] for v in lv], [v[1] for v in lv], ".", ms=1.8,
                    color=colors[mode], alpha=0.45, label=f"mode {mode} (level set)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$q_n(t)$ [mm]")
    ax.set_title("(a) modal coordinate", fontsize=9)
    ax.legend(fontsize=5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for c in cases:
        ax.plot(c["dx_m"], c["omega_measured"], "o", color=colors[c["mode"]])
    for mode in sorted({c["mode"] for c in cases}):
        th = next(c["omega_theory"] for c in cases if c["mode"] == mode)
        ax.axhline(th, color=colors[mode], ls=":", lw=0.9,
                   label=f"mode {mode} analytic")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.set_title(r"(b) $\omega^2=gk\tanh(kH)$ vs measured", fontsize=9)
    ax.legend(fontsize=5)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for mode in sorted({c["mode"] for c in cases}):
        sub = sorted((c for c in cases if c["mode"] == mode), key=lambda c: c["dx_m"])
        ax.plot([c["dx_m"] for c in sub],
                [abs(c["frequency_error_pct"]) for c in sub], "o-",
                color=colors[mode], label=f"mode {mode}")
    ax.axhline(15.0, color="r", ls="--", lw=0.9, label="15 % acceptance")
    ax.axhline(3.0, color="g", ls=":", lw=0.9, label="3 % target")
    if all(c["dx_m"] > 0 for c in cases):
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("|error| [%]")
    ax.set_title("(c) frequency error", fontsize=9)
    ax.legend(fontsize=5)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = HERE / "slosh.png"
    fig.savefig(out, dpi=95)
    plt.close(fig)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
