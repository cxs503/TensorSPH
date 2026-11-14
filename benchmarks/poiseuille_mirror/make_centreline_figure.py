#!/usr/bin/env python3
"""Figures for the gate-A centre-line sampling-bias study.

Reads ``benchmarks/poiseuille_mirror/out/centreline_sampling.json`` and writes

* ``figures/centreline_estimators.png`` -- the four centre-line estimators
  against ``H/dx``, with odd resolutions (a lattice row on ``y = H/2``)
  highlighted.  The legacy metric zig-zags; the corrected ones do not.
* ``figures/centreline_sampling_factor.png`` -- log-log view: legacy error,
  fit-based error, and the closed-form sampling factor ``1 - (dx/H)^2``.
* ``figures/centreline_compensated.png`` -- the sampling hypothesis as a
  one-line test: ``(1 + A_legacy) / (1 - (dx/H)^2)`` vs ``1 + A_fit_full``.

Usage:
    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_centreline_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks/poiseuille_mirror/out"
FIGS = REPO / "benchmarks/poiseuille_mirror/figures"


def main() -> int:
    data = json.loads((OUT / "centreline_sampling.json").read_text())
    series = sorted(data["series"], key=lambda e: e["resolution_H_over_dx"])
    n = np.array([e["resolution_H_over_dx"] for e in series], dtype=float)
    odd = np.array([e["row_sits_on_centreline"] for e in series])
    legacy = 100 * np.array([e["A_legacy"] for e in series])
    same_row = 100 * np.array([e["A_same_row"] for e in series])
    fit_full = 100 * np.array([e["A_fit_full"] for e in series])
    fit_mid_vals = [e["A_fit_mid"] for e in series]
    fit_mid = (100 * np.array([v if v is not None else np.nan
                               for v in fit_mid_vals]))
    bias = np.array([e["bias_pred_factor"] for e in series])

    FIGS.mkdir(parents=True, exist_ok=True)

    # -- 1: estimators vs resolution ------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhspan(-5, 5, color="0.85", zorder=0, label="gate A band (±5 %)")
    ax.plot(n, legacy, "o-", color="#c0392b", label="A legacy (row-max vs continuum)")
    ax.plot(n, same_row, "s--", color="#2980b9", label="A same row (analytic at y_k)")
    ax.plot(n, fit_full, "^-", color="#27ae60", label="A parabola fit at y=H/2 (all rows)")
    ax.plot(n, fit_mid, "v:", color="#8e44ad", label="A parabola fit at y=H/2 (0.25–0.75)")
    for xi, yi, o in zip(n, legacy, odd):
        ax.annotate("row on\ny=H/2" if o else "", (xi, yi),
                    textcoords="offset points", xytext=(0, -26),
                    ha="center", fontsize=7, color="#c0392b")
    ax.set_xlabel("H/dx")
    ax.set_ylabel("centre-line error  (%)")
    ax.set_title("Gate A: the legacy metric alternates with lattice parity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "centreline_estimators.png", dpi=160)
    plt.close(fig)

    # -- 2: log-log, sampling factor ------------------------------------- #
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    dxh = 1.0 / n
    ax.loglog(dxh, np.abs(legacy), "o-", color="#c0392b", label="|A legacy|")
    ax.loglog(dxh, np.abs(fit_full), "^-", color="#27ae60", label="|A fit at y=H/2|")
    ax.loglog(dxh, 1.0 - bias, "k--",
              label=r"sampling factor $(dx/H)^2$")
    ax.loglog(dxh, np.abs(fit_full) + 1.0 - bias, ":", color="0.4",
              label="fit ⊕ sampling")
    for xf in (2.0, 1.0):
        ax.loglog(dxh, 100 * dxh ** xf / 100, alpha=0.0)
    ax.set_xlabel("dx/H")
    ax.set_ylabel("relative error")
    ax.set_title("The (dx/H)² sampling factor explains the even-rung collapse")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "centreline_sampling_factor.png", dpi=160)
    plt.close(fig)

    # -- 3: compensation test -------------------------------------------- #
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    pred = (1.0 + legacy / 100.0) / bias
    meas = 1.0 + fit_full / 100.0
    ax.plot([0.98, 1.02], [0.98, 1.02], "k--", lw=1, label="y = x")
    ax.plot(meas, pred, "o", color="#c0392b", ms=7)
    for xi, yi, ni in zip(meas, pred, n):
        ax.annotate(f"{int(ni)}", (xi, yi), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)
    ax.set_xlabel("1 + A_fit (sampling-free)")
    ax.set_ylabel(r"$ (1 + A_{legacy}) / (1 - (dx/H)^2) $")
    ax.set_title("Gate A corrected by the sampling factor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "centreline_compensated.png", dpi=160)
    plt.close(fig)

    print(f"wrote 3 figures to {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())