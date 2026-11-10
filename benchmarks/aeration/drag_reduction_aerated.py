"""Benchmark 3/3 -- aerated friction reduction vs the Chisholm multiplier (C3).

Reference state: a bubbly mixture at fixed *mixture* volumetric flux ``u_m`` in a
channel of hydraulic diameter ``D``.  The **reduction rate** is

    R(alpha) = 1 - (dp/dz)_aerated / (dp/dz)_liquid-only at the same u_m,

so ``R > 0`` means the aerated flow drags less than the same volume flux of
clear water.  Two independent closures give ``R``:

* ``chisholm_reduction_rate``: Lockhart-Martinelli two-phase multiplier with
  ``C = 20`` (turbulent-turbulent), liquid friction factors from the log law;
* ``loglaw_reduction_rate``: mixture log-law friction factor with the mixture
  density and the mixture Reynolds number.

The campaign criterion is 20% agreement between the two over the bubbly band.
The band is **predeclared as alpha in [0.12, 0.30]**: below it both rates tend to
zero and the *relative* gap diverges even though the absolute gap stays near one
percentage point (measured here for alpha = 0.02 .. 0.10 and reported in the
rows so the limitation is on the record rather than hidden by the band).

The Wood (1991) von-Karman modification ``kappa_eff = kappa_0 (1 - gamma alpha)``
is carried through the model with ``gamma = 0``: no drag-reduction calibration
dataset is bundled, and fitting ``gamma`` against the Chisholm answer would make
this cross-check circular.
"""
from __future__ import annotations

from tensorsph.aeration import (
    chisholm_multiplier,
    chisholm_reduction_rate,
    lockhart_martinelli_x,
    loglaw_reduction_rate,
)

MIXTURE_FLUX = 5.0
DIAMETER = 0.5
ALPHAS = (0.02, 0.05, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35)
BAND = (0.12, 0.30)


def evaluate(alphas=ALPHAS, mixture_flux: float = MIXTURE_FLUX, diameter: float = DIAMETER):
    rows = []
    for alpha in alphas:
        x = lockhart_martinelli_x((1.0 - alpha) * mixture_flux, alpha * mixture_flux)
        reference = chisholm_reduction_rate(alpha, mixture_flux, diameter)
        model = loglaw_reduction_rate(alpha, mixture_flux, diameter)
        rows.append(
            {
                "alpha": alpha,
                "lockhart_martinelli_x": x,
                "chisholm_multiplier": chisholm_multiplier(x),
                "reduction_chisholm": reference,
                "reduction_loglaw": model,
                "abs_gap": abs(model - reference),
                "rel_deviation": abs(model - reference) / reference,
                "in_band": BAND[0] <= alpha <= BAND[1],
            }
        )
    band_rows = [row for row in rows if row["in_band"]]
    worst = max(row["rel_deviation"] for row in band_rows)
    worst_row = max(band_rows, key=lambda row: row["rel_deviation"])
    outside = [row for row in rows if not row["in_band"]]
    return {
        "closure": {"mixture_flux": mixture_flux, "diameter": diameter, "band": BAND},
        "rows": rows,
        "worst_rel_deviation_in_band": worst,
        "worst_alpha_in_band": worst_row["alpha"],
        "low_alpha_note": {
            "worst_rel_deviation_outside_band": max(row["rel_deviation"] for row in outside),
            "alpha_of_worst": max(outside, key=lambda row: row["rel_deviation"])["alpha"],
            "abs_gap_of_worst": max(outside, key=lambda row: row["rel_deviation"])["abs_gap"],
        },
    }
