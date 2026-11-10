"""Benchmark 2/3 -- single-bubble rise velocity, Stokes limit to Mendelson (C4).

The campaign criterion is "single-bubble rise velocity against the Mendelson
formula, error <= 10% over R = 0.5 .. 5 mm".  Two things are measured here and
they are deliberately kept apart:

* the **model vs Mendelson** error over the band: this is a check on the regime
  switching of ``rise_velocity`` (which closure is used where), *not*
  independent validation of Mendelson -- no digitized bubble-rise dataset is
  bundled, and the model *is* Mendelson above ``R = 0.5 mm`` by construction;
* the **Stokes limit** of the drag-balance solver, which *is* an independent
  check: the rigid-sphere force balance must reproduce ``2 g R^2 drho / (9 mu)``
  to under 1% once ``Re -> 0``.

The gap between the two branches is also reported, because it is the physical
reason for the switch: at ``R = 0.5 mm`` the rigid-sphere balance gives a rise
velocity three to four times *below* the wave-theory value, i.e. the
rigid-sphere closure is already outside its range of validity.
"""
from __future__ import annotations

import torch

from tensorsph.aeration import (
    drag_balance_rise_velocity,
    mendelson_rise_velocity,
    rise_velocity,
    stokes_rise_velocity,
)

BAND = (5.0e-4, 5.0e-3)  # R = 0.5 .. 5 mm
SAMPLES = 12
STOKES_RADIUS = 1.0e-5
SWITCH_RADIUS = 5.0e-4


def evaluate():
    radii = torch.linspace(BAND[0], BAND[1], SAMPLES, dtype=torch.float64)
    model = rise_velocity(radii)
    mendelson = mendelson_rise_velocity(radii)
    rigid = torch.tensor(
        [drag_balance_rise_velocity(float(r)) for r in radii], dtype=torch.float64
    )
    rows = []
    for index in range(radii.numel()):
        rows.append(
            {
                "radius_mm": float(radii[index]) * 1.0e3,
                "model": float(model[index]),
                "mendelson": float(mendelson[index]),
                "rigid_sphere": float(rigid[index]),
                "rel_error": float(abs(model[index] - mendelson[index]) / mendelson[index]),
            }
        )
    worst = max(row["rel_error"] for row in rows)

    stokes = float(stokes_rise_velocity(STOKES_RADIUS))
    balanced = drag_balance_rise_velocity(STOKES_RADIUS)
    stokes_error = abs(balanced - stokes) / stokes

    mendelson_switch = float(mendelson_rise_velocity(SWITCH_RADIUS))
    rigid_switch = drag_balance_rise_velocity(SWITCH_RADIUS)

    return {
        "band": {"radius_min_m": BAND[0], "radius_max_m": BAND[1], "samples": SAMPLES},
        "rows": rows,
        "worst_rel_error": worst,
        "stokes_check": {
            "radius": STOKES_RADIUS,
            "stokes_velocity": stokes,
            "drag_balance_velocity": balanced,
            "rel_error": stokes_error,
        },
        "rigid_sphere_gap_at_switch": {
            "radius": SWITCH_RADIUS,
            "rigid_sphere_velocity": rigid_switch,
            "mendelson_velocity": mendelson_switch,
            "ratio": mendelson_switch / rigid_switch,
        },
    }


def curves(radii=None):
    """Rise velocity vs radius for the figure (model, Mendelson, rigid sphere)."""
    if radii is None:
        radii = torch.logspace(-5, -2, 60, dtype=torch.float64)
    model = rise_velocity(radii)
    mendelson = mendelson_rise_velocity(radii)
    rigid = torch.tensor(
        [drag_balance_rise_velocity(float(r)) for r in radii], dtype=torch.float64
    )
    return {"radius": radii, "model": model, "mendelson": mendelson, "rigid_sphere": rigid}
