#!/usr/bin/env python3
"""Regenerate the three *analytic* figures of the slamming evidence bundle.

The committed bundle (``docs/slamming_campaign_report.json``) pointed its three
figures at ``artifacts/slamming_campaign/*.png`` -- a git-ignored directory that
was never committed, so the report shipped with three broken images and three
empty payloads.  All three figures are closed-form (Wagner), so they cost
milliseconds and belong in a tracked path: ``benchmarks/slamming/results/figures``.

Nothing here is a numerical result: these are the *reference* curves
(``C_s = pi^2/tan beta``, the self-similar pressure shape, ``m_a ~ t^2``) drawn
from ``tensorsph.slamming`` itself, so a figure cannot disagree with the code.

    PYTHONPATH=src python benchmarks/slamming/make_reference_figures.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.slamming import (  # noqa: E402
    added_mass_per_length,
    added_mass_law,
    slamming_force_coefficient,
    wagner_c_coeff,
    wagner_shape,
)

FIGS = Path(__file__).resolve().parent / "results" / "figures"
DOC = REPO / "docs" / "slamming_campaign_report.json"


def fig_pressure_shape() -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = torch.linspace(-1.0, 1.0, 601, dtype=torch.float64)
    for beta_deg in (10.0, 20.0, 30.0):
        beta = math.radians(beta_deg)
        ax.plot(x.numpy(), wagner_shape(x, 1.0, beta).numpy(),
                label=fr"$\beta$ = {beta_deg:g}$\degree$")
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$p/p_{max}$")
    ax.set_title("Wagner self-similar pressure shape (analytic)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = FIGS / "wagner_pressure_shape.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_force_coefficient() -> Path:
    betas = [b for b in range(5, 61)]
    cs = [slamming_force_coefficient(1000.0, 2.0, math.radians(b), 1.0e-3) for b in betas]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogy(betas, cs)
    for b in (20, 30):
        ax.plot([b], [slamming_force_coefficient(1000.0, 2.0, math.radians(b), 1e-3)],
                "o", color="C3")
        ax.annotate(fr"$\pi^2/\tan {b}\degree$ = "
                    f"{math.pi**2/math.tan(math.radians(b)):.2f}",
                    (b, math.pi ** 2 / math.tan(math.radians(b))),
                    textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax.set_xlabel(r"deadrise angle $\beta$ [deg]")
    ax.set_ylabel(r"$C_s = F/(\frac{1}{2}\rho V^2 c)$")
    ax.set_title(r"Wagner force coefficient $C_s = \pi^2/\tan\beta$")
    ax.grid(alpha=0.3, which="both")
    out = FIGS / "force_coefficient_vs_beta.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_added_mass() -> Path:
    beta = math.radians(20.0)
    ts = [10.0 ** (-4 + 4 * k / 200) for k in range(201)]
    ma = [added_mass_per_length(1000.0, 2.0, beta, t) for t in ts]
    c = [wagner_c_coeff(beta) * 2.0 * t for t in ts]
    law = added_mass_law(V=2.0, beta=beta)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(ts, ma, label=r"$m_a = \frac{\pi}{2}\rho c^2$ (Wagner)")
    ax.loglog(ts, [x / ma[0] * ma[0] for x in c], "--", label=r"$c = C V t$")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel("per-unit-length quantity")
    ax.set_title(fr"Straight wedge $\beta=20\degree$: $c\propto t^{{{law['c_exponent']:.2f}}}$, "
                 fr"$m_a\propto t^{{{law['added_mass_exponent']:.2f}}}$")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    out = FIGS / "added_mass_scaling.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def repoint_doc() -> None:
    blob = json.loads(DOC.read_text())
    md = blob.get("markdown", "")
    # The bundle linked artifacts/slamming_campaign/fig<N>_<shorthand>.png, whose
    # shorthands do not match the figure names; remap by position, in the order
    # the three figures appear in the document.
    names = ["wagner_pressure_shape", "force_coefficient_vs_beta",
             "added_mass_scaling"]
    links = list(re.finditer(r"artifacts/slamming_campaign/[^)\s]*\.png", md))
    for m, name in reversed(list(zip(links, names))):
        md = md[:m.start()] + f"benchmarks/slamming/results/figures/{name}.png" + md[m.end():]
    blob["markdown"] = md
    DOC.write_text(json.dumps(blob, indent=1, ensure_ascii=False))


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    outs = [fig_pressure_shape(), fig_force_coefficient(), fig_added_mass()]
    repoint_doc()
    for o in outs:
        print(f"{o.relative_to(REPO)} {o.stat().st_size} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
