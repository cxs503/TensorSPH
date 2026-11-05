"""Figure for the free-surface pressure closure grid record.

Two panels, both read from the evidence files (no hand-entered numbers):

(a) the radial pressure profile of the dx = 2 mm drop at the end of the window,
    closure off vs closure on (``raw/relax_ladder.json``), against the analytic
    Young-Laplace jump ``sigma/R``;
(b) the tail time-average error of every grid case (``raw/closure_grid.json``)
    with the 3 % / 15 % criteria drawn in.

    PYTHONPATH=src python benchmarks/laplace_pressure_closure/make_closure_grid_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
GRID = HERE / "raw" / "closure_grid.json"
LADDER = HERE / "raw" / "relax_ladder.json"
OUT = HERE / "closure_grid.png"

SHORT = {
    "grid_dx2.0mm_off": "2.0 mm\noff",
    "grid_dx2.0mm_on": "2.0 mm\non",
    "grid_dx1.5mm_off": "1.5 mm\noff",
    "grid_dx1.5mm_on": "1.5 mm\non",
    "grid_dx2.0mm_c0_10_on": "2.0 mm\nc0=10 on",
    "grid_dx1.5mm_c0_10_on": "1.5 mm\nc0=10 on",
}


def main() -> None:
    grid = json.loads(GRID.read_text())
    ladder = json.loads(LADDER.read_text()) if LADDER.exists() else {"cases": []}
    dp = grid["reference"]["analytic_dp_Pa"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    for case in ladder.get("cases", []):
        if case["case"] not in ("closure_off", "closure_on_interior_layers2"):
            continue
        prof = case["pressure_profile"]
        r = [row["r_over_R_mid"] for row in prof]
        p = [row["mean"] for row in prof]
        e = [row["std"] for row in prof]
        label = ("closure off" if case["case"] == "closure_off"
                 else "closure on (interior, 2 layers)")
        ax.errorbar(r, p, yerr=e, marker="o", ms=3, lw=1.2, capsize=2, label=label)
    ax.axhline(dp, color="k", ls="--", lw=1.0, label=r"analytic $\sigma/R$")
    ax.axhline(0.0, color="r", ls=":", lw=0.8, label="Tait clamp floor")
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"pressure [Pa]")
    ax.set_title(r"(a) final radial profile, $dx$ = 2.0 mm")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    cases = grid["cases"]
    labels = [SHORT.get(c["case"], c["case"]) for c in cases]
    errs = [c["err_tail_pct"] for c in cases]
    colors = ["#c44e52" if not c["closure"]["enabled"] else "#4c72b0" for c in cases]
    ax.bar(range(len(cases)), errs, color=colors)
    ax.axhline(15.0, color="r", ls=":", lw=1, label="+15 %")
    ax.axhline(-15.0, color="r", ls=":", lw=1)
    ax.axhline(3.0, color="g", ls=":", lw=0.8, label="+3 %")
    ax.axhline(-3.0, color="g", ls=":", lw=0.8)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$(\Delta p_{\rm tail} - \sigma/R)/(\sigma/R)$ [%]")
    ax.set_title("(b) closure grid: tail-average error")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("TensorSPH: free-surface pressure closure on a dx x c0 grid")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
