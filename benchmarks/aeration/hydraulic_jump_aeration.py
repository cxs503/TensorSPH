"""Benchmark 1/3 -- water-jump aeration layer vs the erf analytic profile (C2).

A hydraulic jump at inflow Froude number ``Fr_1 = 4 .. 8`` entrains air at the
roller surface.  The local void fraction below the (air-entraining) free surface
is modelled with the advective-diffusion equation already implemented in
``tensorsph.aeration``::

    dC/dt = D d2C/dy2 + w dC/dy,   C(0, t) = 1,  C(y, 0) = 0,

with ``y`` the depth below the surface, ``D`` the turbulent diffusivity of the
roller and ``w`` the buoyant drift of the bubble population back towards the
surface.  The analytic anchor of the campaign is

    C(y) = 1 - erf(y / (Y_90 / sqrt(2))),

which is exactly the drift-free solution ``erfc(y / (2 sqrt(D t)))`` with
``Y_90 = 2 sqrt(2) sqrt(D t)``.

Declared closures (no experimental table is bundled, so every coefficient is
stated rather than fitted):

* roller diffusivity ``D = 0.05 m^2/s`` -- inside the 0.01-0.1 m^2/s band
  measured in hydraulic-jump rollers, taken at its centre;
* station ``x = 3 d_2`` downstream of the toe, so ``t = x / U_2``;
* bubbles on the rigid-sphere branch (``R = 0.15 mm``) so the buoyant drift
  ``w`` stays small against the mixing velocity and the erf anchor remains the
  right comparison.  Larger bubbles leave the layer within ``~d_2`` and are
  beyond the scope of this single-source configuration.

The reported error is the relative L2 norm over the aeration layer,
``y <= 1.5 Y_90`` (predeclared window: further down the profile is zero to
within round-off, so including it would only dilute the error).
"""
from __future__ import annotations

import math


from tensorsph.aeration import (
    analytic_void_fraction,
    erf_anchor,
    l2_relative_error,
    rise_velocity,
    solve_void_fraction_field,
    y90_for_diffusion,
)

GRAVITY = 9.81
DIFFUSIVITY = 0.05
BUBBLE_RADIUS = 1.5e-4
STATION_FACTOR = 3.0
DEPTH = 2.0
CELLS = 300
WINDOW_FACTOR = 1.5


def jump_geometry(froude: float, inflow_depth: float, gravity: float = GRAVITY) -> dict:
    """Bélanger conjugate depth, inflow velocity and roller velocity."""
    v1 = froude * math.sqrt(gravity * inflow_depth)
    d2 = 0.5 * inflow_depth * (math.sqrt(1.0 + 8.0 * froude * froude) - 1.0)
    return {
        "froude": froude,
        "inflow_depth": inflow_depth,
        "inflow_velocity": v1,
        "conjugate_depth": d2,
        "roller_velocity": v1 * inflow_depth / d2,
    }


def evaluate(
    froude_values=(4.0, 5.0, 6.0, 8.0),
    inflow_depth: float = 0.1,
    diffusivity: float = DIFFUSIVITY,
    bubble_radius: float = BUBBLE_RADIUS,
    station_factor: float = STATION_FACTOR,
):
    """Return per-Froude rows and the worst-case L2 error of the erf comparison."""
    drift = float(rise_velocity(bubble_radius))
    rows = []
    for froude in froude_values:
        geometry = jump_geometry(froude, inflow_depth)
        transit = station_factor * geometry["conjugate_depth"] / geometry["roller_velocity"]
        y, drift_free = solve_void_fraction_field(
            depth=DEPTH,
            cells=CELLS,
            diffusivity=diffusivity,
            drift=0.0,
            transit_time=transit,
        )
        _, with_drift = solve_void_fraction_field(
            depth=DEPTH,
            cells=CELLS,
            diffusivity=diffusivity,
            drift=drift,
            transit_time=transit,
        )
        dy = float(y[1] - y[0])
        y90 = y90_for_diffusion(transit, diffusivity)
        anchor = erf_anchor(y - 0.5 * dy, y90)
        closed_form = analytic_void_fraction(y - 0.5 * dy, transit, diffusivity, drift)
        window = y <= WINDOW_FACTOR * y90
        rows.append(
            {
                "froude": froude,
                "conjugate_depth": geometry["conjugate_depth"],
                "roller_velocity": geometry["roller_velocity"],
                "transit_time": transit,
                "y90": y90,
                "drift_velocity": drift,
                "l2_vs_erf_anchor_drift_free": l2_relative_error(
                    drift_free[window], anchor[window]
                ),
                "l2_vs_erf_anchor_with_drift": l2_relative_error(
                    with_drift[window], anchor[window]
                ),
                "l2_vs_closed_form_with_drift": l2_relative_error(
                    with_drift[window], closed_form[window]
                ),
                "max_void_fraction": float(with_drift.max()),
            }
        )
    worst = max(
        max(row["l2_vs_erf_anchor_drift_free"], row["l2_vs_erf_anchor_with_drift"])
        for row in rows
    )
    return {
        "closure": {
            "diffusivity": diffusivity,
            "bubble_radius": bubble_radius,
            "drift_velocity": drift,
            "station_factor": station_factor,
            "depth": DEPTH,
            "cells": CELLS,
            "l2_window_factor": WINDOW_FACTOR,
        },
        "rows": rows,
        "worst_l2_vs_erf_anchor": worst,
        "profile_drift_free": None,  # filled by run.py when it needs the curves
    }


def profiles(froude: float = 6.0, inflow_depth: float = 0.1, diffusivity: float = DIFFUSIVITY,
             bubble_radius: float = BUBBLE_RADIUS, station_factor: float = STATION_FACTOR):
    """Curves for the figure: (y, drift-free, with-drift, erf anchor, closed form)."""
    geometry = jump_geometry(froude, inflow_depth)
    transit = station_factor * geometry["conjugate_depth"] / geometry["roller_velocity"]
    drift = float(rise_velocity(bubble_radius))
    y, drift_free = solve_void_fraction_field(
        depth=DEPTH, cells=CELLS, diffusivity=diffusivity, drift=0.0, transit_time=transit
    )
    _, with_drift = solve_void_fraction_field(
        depth=DEPTH, cells=CELLS, diffusivity=diffusivity, drift=drift, transit_time=transit
    )
    dy = float(y[1] - y[0])
    y90 = y90_for_diffusion(transit, diffusivity)
    return {
        "y": y,
        "drift_free": drift_free,
        "with_drift": with_drift,
        "anchor": erf_anchor(y - 0.5 * dy, y90),
        "closed_form": analytic_void_fraction(y - 0.5 * dy, transit, diffusivity, drift),
        "y90": y90,
        "transit": transit,
        "drift": drift,
        "froude": froude,
    }
