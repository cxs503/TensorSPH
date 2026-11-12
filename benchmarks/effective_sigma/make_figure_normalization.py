"""Figure for the delta-normalisation record (reads ``decompose.decompose``).

Writes ``effective_sigma_delta_normalization.png`` next to this file.  All
numbers come from the same static droplet evaluation the test pins, so the
figure cannot drift away from the tables.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import decompose  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    cases = [(0.005, "fit_quadratic"), (0.0025, "fit_quadratic"),
             (0.005, "mls"), (0.0025, "mls")]
    labels, legacy, local, dlen_legacy, dlen_local = [], [], [], [], []
    for dx, curv in cases:
        a = decompose.decompose(dx, curv, "legacy")
        b = decompose.decompose(dx, curv, "local_range")
        labels.append(f"{curv}\ndx={dx * 1e3:.1f} mm")
        legacy.append(a["force_ratio_right_half"])
        local.append(b["force_ratio_right_half"])
        dlen_legacy.append(a["int_delta_s_over_area"])
        dlen_local.append(b["int_delta_s_over_area"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = range(len(labels))
    w = 0.38
    ax = axes[0]
    ax.bar([i - w / 2 for i in x], legacy, w, label=r"legacy  $2|\nabla c|$")
    ax.bar([i + w / 2 for i in x], local, w, label=r"local_range  $|\nabla c|/\Delta c$")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="exact  $|F_{right}|/(2\\sigma)=1$")
    ax.axhline(0.85, color="0.55", ls=":", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$|F_{right}|/(2\sigma)$")
    ax.set_title("integrated CSF traction on a circular droplet")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar([i - w / 2 for i in x], dlen_legacy, w, label="legacy")
    ax.bar([i + w / 2 for i in x], dlen_local, w, label="local_range")
    ax.axhline(1.0, color="k", ls="--", lw=1, label=r"exact  $\int\delta_s dV = 2\pi R$")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\int \delta_s\,dV \,/\, (2\pi R)$")
    ax.set_title("surface delta function: recovered interface length")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = HERE / "effective_sigma_delta_normalization.png"
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()