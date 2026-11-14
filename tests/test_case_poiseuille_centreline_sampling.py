#!/usr/bin/env python3
"""Invariants for the gate-A centre-line sampling-bias study.

Two layers:

1. **Synthetic (always runs).**  A parabola sampled on the case's lattice rows
   ``y_k = dx/2 + k dx`` must under-read the continuum centre-line value by
   exactly ``1 - (dx/H)^2`` when ``H/dx`` is even and must reproduce it exactly
   when ``H/dx`` is odd.  Feeding such a profile through
   :func:`benchmarks.poiseuille_mirror.study_centreline_sampling.centreline_estimators`
   must return ``A_fit_* ~ 0`` and ``A_legacy ~ bias_pred - 1``.  This pins the
   algebra of the metric independent of any solver run.

2. **Artifact (skips when the ladder has not been run in this checkout).**
   On ``benchmarks/poiseuille_mirror/out/centreline_sampling.json``:
   the measured legacy error must be reproduced from the sampling-free fit
   value times the sampling factor to within 0.5 percentage points, an odd
   ``H/dx`` must report ``row_sits_on_centreline`` and identical legacy /
   same-row errors, and every gate A variant must stay inside the 5 % band.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.poiseuille_mirror.study_centreline_sampling import (
    centreline_estimators,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "benchmarks/poiseuille_mirror/out/centreline_sampling.json"

HEIGHT = 0.1
BODY_FORCE = 9.81
VISCOSITY = 0.01
U_MAX = BODY_FORCE * HEIGHT ** 2 / (8.0 * VISCOSITY)


def parabola(y: float) -> float:
    return BODY_FORCE * (HEIGHT - y) * y / (2.0 * VISCOSITY)


def synth_case(resolution: int, u_of_y=parabola) -> dict:
    dx = HEIGHT / resolution
    rows = []
    for k in range(resolution):
        y = dx * 0.5 + k * dx
        rows.append({"row": k, "y": y, "y_over_H": y / HEIGHT,
                     "u_mean": u_of_y(y), "u_std_x": 0.0})
    peak = max(rows, key=lambda r: r["u_mean"])
    return {
        "resolution_H_over_dx": resolution,
        "rows": rows,
        "u_max_measured": peak["u_mean"],
        "profile_rms_interior_pct": 0.0,
        "profile_rms_all_rows_pct": 0.0,
        "wall_shear_bottom_error_pct": 0.0,
        "steady_drift": 0.0,
        "max_mach": 0.0,
        "n_fluid": resolution * 12,
        "steps": 1,
        "dt": 1.0,
        "wall_clock_s": 0.0,
        "gates": {},
    }


@pytest.mark.parametrize("resolution", [8, 12, 16, 24])
def test_even_resolution_row_max_underreads_by_dx_over_h_squared(resolution):
    est = centreline_estimators(synth_case(resolution), HEIGHT, BODY_FORCE,
                                VISCOSITY)
    dx_over_h = 1.0 / resolution
    assert not est["row_sits_on_centreline"]
    assert est["bias_pred_factor"] == pytest.approx(1.0 - dx_over_h ** 2,
                                                    rel=1e-12)
    # exact parabola: the legacy error is *only* the sampling factor
    assert est["A_legacy"] == pytest.approx(-(dx_over_h ** 2), rel=1e-9,
                                            abs=1e-12)
    # the corrected estimators see the exact solution
    assert est["A_same_row"] == pytest.approx(0.0, abs=1e-12)
    assert est["A_fit_full"] == pytest.approx(0.0, abs=1e-9)
    assert est["A_fit_mid"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("resolution", [9, 15, 17, 25])
def test_odd_resolution_has_a_row_on_the_centreline(resolution):
    est = centreline_estimators(synth_case(resolution), HEIGHT, BODY_FORCE,
                                VISCOSITY)
    assert est["row_sits_on_centreline"]
    assert est["nearest_row_y_over_H"] == pytest.approx(0.5, abs=1e-12)
    assert est["A_legacy"] == pytest.approx(0.0, abs=1e-12)
    assert est["A_legacy"] == pytest.approx(est["A_same_row"], abs=1e-12)


def test_sampling_factor_is_the_only_even_odd_difference_on_exact_parabola():
    """Adjacent even/odd rungs differ by exactly the sampling factor."""
    even = centreline_estimators(synth_case(24), HEIGHT, BODY_FORCE, VISCOSITY)
    odd = centreline_estimators(synth_case(25), HEIGHT, BODY_FORCE, VISCOSITY)
    assert even["A_legacy"] - odd["A_legacy"] == pytest.approx(
        even["bias_pred_factor"] - 1.0, rel=1e-9, abs=1e-12)


def _load_artifact():
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT.name} not produced in this checkout")
    return json.loads(ARTIFACT.read_text())


def test_artifact_legacy_metric_is_predicted_by_sampling_factor():
    data = _load_artifact()
    assert data["anchor"] == "CASE-POISEUILLE-SPH"
    assert len(data["series"]) >= 3
    for entry in data["series"]:
        assert entry["A_legacy_from_fit_full"] is not None
        # the sampling mechanism must explain the legacy metric to 0.5 pp
        assert abs(entry["A_legacy"] - entry["A_legacy_from_fit_full"]) < 0.005, entry
        assert abs(entry["A_fit_full"]) <= 0.05, entry
        if entry["A_fit_mid"] is not None:
            assert abs(entry["A_fit_mid"]) <= 0.05, entry
        assert abs(entry["A_same_row"]) <= 0.05, entry


def test_artifact_odd_resolutions_have_no_sampling_penalty():
    data = _load_artifact()
    for entry in data["series"]:
        if entry["resolution_H_over_dx"] % 2 == 1:
            assert entry["row_sits_on_centreline"], entry
            assert entry["A_legacy"] == pytest.approx(entry["A_same_row"],
                                                      abs=1e-12), entry