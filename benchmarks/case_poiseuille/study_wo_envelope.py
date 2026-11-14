#!/usr/bin/env python3
"""CASE-POISEUILLE-SPH -- where does the harmonic (<5 %) certificate actually end?

Why this study exists
---------------------
``run.py`` certifies the *harmonic* (Womersley) regime of the plane channel at
two frequencies only: ``omega = 13.0`` (``Wo = 1.80``, ``delta/dx = 3.1..6.3``)
and ``omega = 31.4`` (``Wo = 2.80``, used for the frequency-ratio gate G6).
The verification note then states its scope as *"Laminar, ``Wo`` below ~3,
``delta/dx`` between 3 and 7 ... Nothing here tests turbulent or ``Wo >> 10``
pulsatile flow"* -- the upper edge of that scope was **asserted, not measured**.
A user who pulses the same channel at ``Wo = 5`` has no evidence whether the
<5 % band still holds, or how much resolution it costs.

This study measures the edge instead of asserting it, and separates the two
mechanisms that can put an edge there: **spatial** (Stokes layer thinner than
the particle spacing) and **temporal** (the harness samples
``SAMPLES_PER_PERIOD = 64`` points per period, but ``dt`` is CFL/viscosity
limited, so above a frequency there are fewer steps per period than samples --
the measurement, not the physics, breaks first).

Stage-1 hypotheses (predeclared before any point of this study was run)
----------------------------------------------------------------------
H1 (Wo-free / unconditional)
    The harmonic error is set by the boundary operator and the harmonic fit
    alone: every frequency in the sweep passes G1..G4 at ``H/dx = 16``.
H2 (Stokes-layer resolution rule) -- the *reading of the existing scope note*
    The error is controlled by ``delta/dx``: the band is met iff
    ``delta/dx >= 3``.  At ``H/dx = 16`` (``dx = 6.25 mm``) that is
    ``Wo <= 3.77``, so ``Wo = 5.0`` (``delta/dx = 2.32``) and ``Wo = 6.5``
    (``delta/dx = 1.67``) must **fail**.
H3 (transient / averaging artefact)
    Any off-band point is an unconverged periodic state or a too-short
    measurement window rather than a spatial-resolution limit.
H4 (signal-to-noise floor)
    Off-band error is measurement noise: at high ``Wo`` the centre-line
    amplitude collapses as ``~g0/omega`` while the pressure noise does not.

Stage-1 probe, and what it did to H2
------------------------------------
The two cheapest points (``W3_wo500_r16``, ``W4_wo650_r16``) were run first.
**Both passed** (interior amplitude RMS 0.80 % and 0.60 % at
``delta/dx = 2.26`` and ``1.74``), which falsifies H2 as predeclared: the
interior-band gate is *not* limited by ``delta/dx >= 3``.  H1 was still
standing.  The probe is reported, not hidden (``probe_before_stage2`` in the
artefact), and it forced a better question, which is stage 2.

Stage-2 hypotheses (predeclared after the probe, before the sweep below)
-----------------------------------------------------------------------
H5 (two-ceiling / measurement-limited)
    The certificate does not end at a physics limit in the graded direction:
    every point whose temporal sampling is adequate
    (``steps_per_period >= SAMPLES_PER_PERIOD``) stays inside G1..G4, and the
    first point that breaks the *harness* (fewer steps per period than
    samples) is invalid rather than failing.  Corollary: the same frequency
    re-run at a resolution with a smaller ``dt`` regains adequate sampling and
    returns to the band.
H6 (the interior gate is blind to the wall layer)
    Because G2/G4 average over ``0.15 <= y/H <= 0.85``, they cannot see the
    Stokes layer: the interior amplitude RMS stays inside 5 % even for
    ``delta/dx < 1``, while the near-wall rows (``y/H <= 0.15``) degrade
    monotonically with ``dx/delta``.  The discriminator for "is the Stokes
    layer resolved" is therefore the wall-row metric, not the interior gate --
    which is exactly why the declared scope ("``delta/dx`` between 3 and 7")
    was never tested by the gate it was attached to.

Design (``g0 = 9.81``, ``nu = 0.01``, ``H = 0.1``, ``c_s = 20``, ``x`` periodic
with 8 columns: all ``run.py`` defaults; ``omega`` is never hand-picked but
derived from the declared ``Wo`` by ``omega = (2 Wo / H)**2 * nu``)
-------------------------------------------------------------------
A. frequency sweep at ``H/dx = 16`` (the finest committed rung):
   ``Wo = 1.80 / 2.80 / 3.77 / 5.00 / 6.50 / 8.00 / 10.0 / 12.5``
   (``delta/dx = 6.30 / 4.00 / 3.00 / 2.26 / 1.74 / 1.40 / 1.13 / 0.90``).
   ``W0``/``W1`` are controls that must reproduce the committed rungs.
B. resolution ladder at fixed ``Wo = 6.50``: ``H/dx = 8 / 16 / 32``
   (``delta/dx = 0.87 / 1.74 / 3.48``) -- H6's monotone-in-``dx/delta`` test.
C. sampling dissociation at fixed ``Wo = 16``: ``H/dx = 16``
   (``dt = 1.22e-4``, ``~50`` steps per period for 64 samples -> invalid) vs
   ``H/dx = 32`` (``~101`` steps per period -> valid).  H5's corollary.
D. artefact controls: ``W3_wo500_r16`` re-run with 4 settle periods instead of
   2 (H3), and the H6 ladder itself (H4: refining at fixed ``Wo`` either fixes
   the error -- spatial -- or does not -- noise).

Usage
-----
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/study_wo_envelope.py \
        --points W0_wo180_r16,W1_wo280_r16 --out benchmarks/case_poiseuille/out/wo_envelope_part1.json
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/study_wo_envelope.py \
        --merge out/wo_envelope_part*.json --out benchmarks/case_poiseuille/out/wo_envelope.json

The runner is a sharded study: each invocation writes a self-contained artefact
carrying its own hypotheses and predictions, and :func:`merge` folds the parts.
Predictions are static text in ``PREDICTIONS``, never derived from measurements.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import run as caserun  # noqa: E402
from tensorsph import womersley as wm  # noqa: E402
from tensorsph.droplet_impact import resolve_device  # noqa: E402

TOL = 5.0                       # the case-pool tolerance (feature_ledger)
DELTA_OVER_DX_RULE = 3.0        # H2's predeclared (and falsified) rule
WALL_BAND = (0.0, 0.15)         # near-wall rows for H6

#: ``(id, Wo, H/dx, settle periods, window periods, role)``.
POINTS = (
    ("W0_wo180_r16",          1.80, 16, 2, 2, "A:control_committed_rung"),
    ("W1_wo280_r16",          2.80, 16, 2, 2, "A:control_G6_fast"),
    ("W2_wo377_r16",          3.77, 16, 2, 2, "A:H2_boundary"),
    ("W3_wo500_r16",          5.00, 16, 2, 2, "A:H2_predicted_fail"),
    ("W4_wo650_r16",          6.50, 16, 2, 2, "A:H2_predicted_fail"),
    ("W5_wo800_r16",          8.00, 16, 2, 2, "A:sweep"),
    ("W6_wo1000_r16",        10.00, 16, 2, 2, "A:sweep"),
    ("W7_wo1250_r16",        12.50, 16, 2, 2, "A:sweep"),
    ("W4_wo650_r8",           6.50,  8, 2, 2, "B:H6_ladder"),
    ("W4_wo650_r32",          6.50, 32, 2, 2, "B:H6_ladder"),
    ("W8_wo1600_r16",        16.00, 16, 2, 2, "C:H5_sampling_inadequate"),
    ("W8_wo1600_r32",        16.00, 32, 2, 2, "C:H5_sampling_recovered"),
    ("W3_wo500_r24",          5.00, 24, 2, 2, "B:refinement_at_fixed_Wo"),
    ("W3_wo500_r16_settle4",  5.00, 16, 4, 2, "D:H3_settling_control"),
)

#: Frozen predictions.  ``A:`` rows are the stage-1 H2 prediction, kept as
#: written so the falsification is visible rather than retconned; ``B:``/``C:``
#: rows are the stage-2 (post-probe) predictions.
PREDICTIONS = {
    "W0_wo180_r16": "pass, and reproduces the committed rung (profile RMS +0.39 %)",
    "W1_wo280_r16": "pass (committed G6 fast frequency)",
    "W2_wo377_r16": "pass (delta/dx = 3.00 is exactly H2's rule)",
    "W3_wo500_r16": "FAIL by H2 (delta/dx = 2.26 < 3) -- measured: PASS",
    "W4_wo650_r16": "FAIL by H2 (delta/dx = 1.74 < 3) -- measured: PASS",
    "W5_wo800_r16": "H6: interior band passes (delta/dx = 1.40); wall rows degrade",
    "W6_wo1000_r16": "H6: interior band passes (delta/dx = 1.13); wall rows degrade further",
    "W7_wo1250_r16": "H6: interior band passes (delta/dx = 0.90); wall rows worst",
    "W4_wo650_r8": "H6: interior band passes at delta/dx = 0.87; wall rows worse than r16",
    "W4_wo650_r32": "H6: interior band passes; wall rows best of the ladder (delta/dx = 3.48)",
    "W8_wo1600_r16": "H5: sampling inadequate (~50 steps per period for 64 samples) "
                     "-> flagged invalid, verdict not graded",
    "W8_wo1600_r32": "H5: sampling recovered (~101 steps per period) -> interior band passes",
    "W3_wo500_r24": "pass (refinement at fixed Wo cannot change a passing verdict)",
    "W3_wo500_r16_settle4": "pass; same verdict as W3_wo500_r16 (H3 predicts an "
                            "unchanged verdict means the band is not a transient artefact)",
}

HYPOTHESES = {
    "H1_wo_free": "every H/dx = 16 sweep point meets G1..G4; the band is "
                  "independent of Wo",
    "H2_layer_resolution": "band met iff delta/dx >= 3 (the reading of the "
                           "existing scope note); FALSIFIED at the stage-1 probe",
    "H3_transient_artefact": "off-band points are an unconverged periodic state "
                             "or a short window, not a spatial limit",
    "H4_noise_floor": "off-band error is measurement noise that refining dx "
                      "cannot remove",
    "H5_two_ceiling": "points with steps_per_period >= samples_per_period stay "
                      "in the band; a sampling-starved point is invalid, not a "
                      "physics failure, and the same Wo at finer dx recovers",
    "H6_interior_gate_blind_to_wall_layer": "G2/G4 (0.15 <= y/H <= 0.85) stay "
                                            "inside 5 % even for delta/dx < 1, "
                                            "while wall-row error grows "
                                            "monotonically with dx/delta",
    "H7_row_mean_floor_share": "POST-HOC: the closed-form row-averaging floor "
                               "accounts for at most half of the wall-row error "
                               "(so the wall rows are not merely a metric artefact)",
    "H8_wall_error_collapse": "POST-HOC: the wall-row error is single-valued in "
                              "delta/dx -- points with delta/dx within 5 % but "
                              "different (Wo, H/dx) agree within 2 percentage points",
}

PREDECLARATION = {
    "stage1": "H1-H4, written before any point of this study ran",
    "stage1_probe": ["W3_wo500_r16", "W4_wo650_r16"],
    "stage1_probe_outcome": "both passed -> H2 falsified before the sweep; H1 intact",
    "stage2": "H5-H6, written after the probe and before the sweep / ladder / "
              "sampling points below",
    "stage3_post_hoc": "H7-H8, formulated *after* the sweep, when the wall-row "
                       "errors were seen to saturate near 26-31 % at high Wo.  They "
                       "are descriptive tests of an explanation, not predeclared "
                       "predictions, and are labelled as such in the verdict.",
    "note": "the H2 prediction text is preserved verbatim in PREDICTIONS so the "
            "falsification stays visible",
}


# --------------------------------------------------------------------------- #
# Reference-side floor: what a row-mean metric can resolve at all
# --------------------------------------------------------------------------- #

FLOOR_SUBDIV = 1025  # sub-points per lattice row (converged: the complex
                    # amplitude rotates fast inside a row when delta < dx, so the
                    # integral is cancellation-sensitive and needs a fine rule)


def row_averaging_floor(point: dict, height: float = caserun.HEIGHT,
                        viscosity: float = caserun.VISCOSITY,
                        body_force: float = caserun.BODY_FORCE) -> dict:
    """Error the *row-mean definition* alone produces, before any solver error.

    ``bin_rows`` reports the mean of ``u_x`` over one lattice row
    ``[k dx, (k+1) dx]`` at ``y = (k+1/2) dx``.  The reference used by the gate
    is the *point* value ``Uhat(y)``.  Averaging a curved harmonic profile over
    a row whose width is a sizeable fraction of the Stokes layer therefore
    introduces an error that no solver can remove -- it is a property of the
    metric, not of the physics.  Because the average of a harmonic in time stays
    a harmonic in time, the row mean is exactly ``(1/dx) Int Uhat dy``, so the
    floor is a closed-form integral, computed here by Gauss-Legendre-free fine
    sub-sampling of the sympy-verified closed form.

    Rows return ``amp_pct`` (relative to ``|Uhat(y_k)|``) and ``phase_deg``
    (relative to ``arg Uhat(y_k)``), in exactly the units the gate uses.
    """
    dx = height / point["resolution_H_over_dx"]
    omega = point["omega"]
    out = []
    for r in point["rows"]:
        k = r["row"]
        y = r["y"]
        edges = np.linspace(k * dx, (k + 1) * dx, FLOOR_SUBDIV)
        u_bar = wm.complex_amplitude(edges, height, body_force, viscosity, omega).mean()
        u_pt = wm.complex_amplitude(np.array([y]), height, body_force, viscosity,
                                    omega)[0]
        amp_pct = 100.0 * (abs(u_bar) / abs(u_pt) - 1.0) if abs(u_pt) else float("nan")
        dphi = np.angle(u_bar) - np.angle(u_pt)
        pha_deg = float(np.degrees((dphi + np.pi) % (2 * np.pi) - np.pi))
        out.append({"row": k, "y_over_H": r["y_over_H"],
                    "floor_amp_pct": float(amp_pct),
                    "floor_phase_deg": pha_deg,
                    # same units/keys as the measured rows so one summariser serves both
                    "floor_amplitude_rel_error": float(amp_pct) / 100.0,
                    "floor_phase_error_deg": pha_deg})
    return {"rows": out,
            "all_rows": _rows_metrics(out, 0.0, 1.0, prefix="floor_"),
            "interior_rows": _rows_metrics(out, caserun.INTERIOR_LO,
                                           caserun.INTERIOR_HI, prefix="floor_"),
            "wall_rows": _rows_metrics(out, *WALL_BAND, prefix="floor_")}


def attach_floors(doc: dict) -> dict:
    """Attach the row-averaging floor and the measured-minus-floor excess."""
    for p in doc["points"]:
        fl = row_averaging_floor(p)
        p["row_averaging_floor"] = fl
        p["wall_floor_fraction"] = (fl["wall_rows"]["amp_worst_pct"]
                                   / p["wall_rows"]["amp_worst_pct"]
                                   if p["wall_rows"]["amp_worst_pct"] else float("nan"))
        p["wall_excess_pct"] = (p["wall_rows"]["amp_worst_pct"]
                                - fl["wall_rows"]["amp_worst_pct"])
        p["interior_floor_rms_pct"] = fl["interior_rows"]["amp_rms_pct"]
        p["interior_excess_pct"] = (p["amp_profile_rms_pct"]
                                    - fl["interior_rows"]["amp_rms_pct"])
    return doc


# --------------------------------------------------------------------------- #
# One point
# --------------------------------------------------------------------------- #

def omega_from_wo(wo: float, height: float = caserun.HEIGHT,
                  viscosity: float = caserun.VISCOSITY) -> float:
    """``omega`` implied by a declared ``Wo = (H/2) sqrt(omega/nu)``."""
    return (2.0 * wo / height) ** 2 * viscosity


def _rows_metrics(rows, y_lo: float, y_hi: float, prefix: str = "") -> dict:
    """RMS/worst amplitude and phase error over the rows in a band."""
    sel = [r for r in rows if y_lo <= r["y_over_H"] <= y_hi]
    amp = np.array([r[prefix + "amplitude_rel_error"] for r in sel])
    pha = np.degrees(np.array([r[prefix + "phase_error_deg"] for r in sel]))
    return {
        "n_rows": len(sel),
        "amp_rms_pct": float(100.0 * np.sqrt(np.mean(amp ** 2))),
        "amp_worst_pct": float(100.0 * np.max(np.abs(amp))),
        "phase_worst_deg": float(np.max(np.abs(pha))),
        "phase_rms_pct_of_quarter_turn":
            float(100.0 * np.sqrt(np.mean(np.radians(pha) ** 2)) / caserun.QUARTER_TURN),
    }


def run_point(point_id: str, wo: float, resolution: int, settle: int, window: int,
              role: str, device) -> dict:
    omega = omega_from_wo(wo)
    t0 = time.time()
    res = caserun.run_harmonic(resolution, omega, device,
                               settle_periods=settle, window_periods=window)
    gates = res["gates"]
    graded = [v for k, v in gates.items()
              if k.startswith(("G1_", "G2_", "G3_", "G4_"))]
    passes = bool(all(g is True for g in graded))
    steps_per_period = res["period"] / res["dt"]
    sampling_ok = steps_per_period >= caserun.SAMPLES_PER_PERIOD - 1e-9

    all_rows = _rows_metrics(res["rows"], 0.0, 1.0)
    wall_rows = _rows_metrics(res["rows"], *WALL_BAND)

    out = {
        "id": point_id,
        "role": role,
        "prediction": PREDICTIONS[point_id],
        "womersley_requested": wo,
        "womersley_number": res["womersley_number"],
        "resolution_H_over_dx": resolution,
        "dx": caserun.HEIGHT / resolution,
        "omega": omega,
        "period": res["period"],
        "stokes_layer": res["stokes_layer"],
        "stokes_layer_over_dx": res["stokes_layer_over_dx"],
        "cells_per_stokes_layer": res["stokes_layer"] / (caserun.HEIGHT / resolution),
        "settle_periods": settle,
        "window_periods": window,
        "dt": res["dt"],
        "steps": res["steps"],
        "steps_per_period": steps_per_period,
        "samples_per_period": caserun.SAMPLES_PER_PERIOD,
        "sampling_ok": bool(sampling_ok),
        "n_fluid": res["n_fluid"],
        "n_ghost": res["n_ghost"],
        "max_mach": res["max_mach"],
        "amp_centre": res["amp_centre"],
        "amp_centre_analytic": res["amp_centre_analytic"],
        "amp_centre_error_pct": res["amp_centre_error_pct"],
        "amp_profile_rms_pct": res["amp_profile_rms_pct"],
        "amp_worst_row_pct": res["amp_worst_row_pct"],
        "phase_centre_error_deg": res["phase_centre_error_deg"],
        "phase_centre_pct_of_quarter_turn": res["phase_centre_pct_of_quarter_turn"],
        "phase_profile_rms_pct_of_quarter_turn": res["phase_profile_rms_pct_of_quarter_turn"],
        "window_amplitude_drift_pct": res["window_amplitude_drift_pct"],
        "window_phase_drift_deg": res["window_phase_drift_deg"],
        "fit_residual_max": res["fit_residual_max"],
        "all_rows": all_rows,
        "wall_rows": wall_rows,
        "gates": dict(gates),
        "passes_case_gates": passes,
        "verdict": ("pass" if passes else "fail") if sampling_ok else "invalid_sampling",
        "rows": [{k: r[k] for k in ("row", "y", "y_over_H", "amplitude",
                                    "amplitude_analytic", "amplitude_rel_error",
                                    "phase", "phase_analytic", "phase_error_deg",
                                    "fit_residual")} for r in res["rows"]],
        "wall_clock_s": time.time() - t0,
        "solver_wall_clock_s": res["wall_clock_s"],
    }
    out["worst_metric_pct"] = max(abs(out["amp_centre_error_pct"]),
                                  out["amp_profile_rms_pct"],
                                  out["phase_centre_pct_of_quarter_turn"],
                                  out["phase_profile_rms_pct_of_quarter_turn"])
    return out


# --------------------------------------------------------------------------- #
# Verdicts -- computed, never typed
# --------------------------------------------------------------------------- #

def _graded(points):
    """Points whose measurement is admissible (sampling adequate)."""
    return [p for p in points if p["sampling_ok"]]


def _spearman(xs, ys) -> float:
    """Rank correlation without scipy (average ranks for ties)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return float(num / den) if den else float("nan")


def evaluate_hypotheses(points: list[dict]) -> dict:
    by_id = {p["id"]: p for p in points}
    graded = _graded(points)
    invalid = [p for p in points if not p["sampling_ok"]]

    sweep = sorted([p for p in graded if p["role"].startswith("A:")],
                   key=lambda p: p["womersley_number"])
    h1 = bool(sweep) and all(p["passes_case_gates"] for p in sweep)

    # H2 (falsified rule): agreement of "delta/dx >= 3" with the measured gate
    h2_rows = []
    for p in graded:
        predicted = p["stokes_layer_over_dx"] >= DELTA_OVER_DX_RULE
        h2_rows.append({"id": p["id"],
                        "delta_over_dx": p["stokes_layer_over_dx"],
                        "rule_says_pass": predicted,
                        "measured_pass": p["passes_case_gates"],
                        "agrees": predicted == p["passes_case_gates"]})
    h2 = all(r["agrees"] for r in h2_rows) if h2_rows else None

    base, ctrl = by_id.get("W3_wo500_r16"), by_id.get("W3_wo500_r16_settle4")
    h3 = None
    if base and ctrl:
        h3 = bool(base["passes_case_gates"] == ctrl["passes_case_gates"])

    # H5: every admissible-sampling point stays in band, and the sampling
    # starvation is what disqualifies the invalid point.
    h5 = (bool(graded) and all(p["passes_case_gates"] for p in graded)
          and all(not p["sampling_ok"] for p in invalid))
    h5_pairs = []
    for p in invalid:
        alt = [q for q in graded
               if abs(q["womersley_requested"] - p["womersley_requested"]) < 1e-9
               and q["resolution_H_over_dx"] != p["resolution_H_over_dx"]]
        h5_pairs.append({"invalid": p["id"],
                         "steps_per_period": p["steps_per_period"],
                         "recovered_by": [q["id"] for q in alt],
                         "recovered_sampling": [q["steps_per_period"] for q in alt],
                         "recovered_pass": [q["passes_case_gates"] for q in alt]})
    h5 = h5 and all(any(rp) for rp in (x["recovered_pass"] for x in h5_pairs)) \
        if h5_pairs else h5

    # H6: interior band blind to the layer, wall rows monotone in dx/delta
    ladder = sorted([p for p in graded
                     if abs(p["womersley_requested"] - 6.50) < 1e-9],
                    key=lambda p: p["stokes_layer_over_dx"])
    h6_blind = bool(ladder) and all(p["amp_profile_rms_pct"] <= TOL for p in ladder)
    h6_thin = bool(ladder) and min(p["stokes_layer_over_dx"] for p in ladder) < 1.0
    h6_monotone = None
    if len(ladder) >= 3:
        rho = _spearman([p["stokes_layer_over_dx"] for p in ladder],
                        [-p["wall_rows"]["amp_worst_pct"] for p in ladder])
        h6_monotone = bool(rho >= 0.99)
        h6_rho = rho
    else:
        h6_rho = None
    h6 = bool(h6_blind and h6_thin and (h6_monotone is not False))

    # H4: does refining at fixed Wo reduce the wall-row error (spatial) or not
    # (noise)?  Compare the coarsest and finest rung of the ladder.
    h4 = None
    if len(ladder) >= 2:
        coarse = ladder[0]["wall_rows"]["amp_worst_pct"]     # thinnest layer
        fine = ladder[-1]["wall_rows"]["amp_worst_pct"]      # resolved layer
        # H4 (noise floor) is supported only if refining does NOT reduce the
        # wall-row error; if it does, the limit is spatial and H4 is refuted.
        h4 = bool(coarse <= fine)

    # H7 (post-hoc): share of the wall-row error that the row-mean definition
    # alone produces.  Supported iff that share stays below 50 % everywhere.
    floor_share = [{"id": p["id"],
                    "delta_over_dx": p["stokes_layer_over_dx"],
                    "wall_amp_worst_pct": p["wall_rows"]["amp_worst_pct"],
                    "floor_amp_worst_pct": p["row_averaging_floor"]["wall_rows"]["amp_worst_pct"],
                    "floor_fraction": p.get("wall_floor_fraction")}
                   for p in (_graded(points) if all("row_averaging_floor" in p
                                                    for p in points) else [])]
    h7 = (all(x["floor_fraction"] <= 0.5 for x in floor_share)
          if floor_share else None)

    # H8 (post-hoc): collapse of the wall-row error onto delta/dx alone.
    collapse = []
    if floor_share:
        for i, a in enumerate(floor_share):
            for b in floor_share[i + 1:]:
                da, db = a["delta_over_dx"], b["delta_over_dx"]
                if abs(da - db) / max(da, db) <= 0.05:
                    collapse.append({
                        "ids": [a["id"], b["id"]],
                        "delta_over_dx": [da, db],
                        "wall_amp_worst_pct": [a["wall_amp_worst_pct"],
                                               b["wall_amp_worst_pct"]],
                        "gap_pp": abs(a["wall_amp_worst_pct"] - b["wall_amp_worst_pct"]),
                        "within_2pp": bool(abs(a["wall_amp_worst_pct"]
                                               - b["wall_amp_worst_pct"]) <= 2.0),
                    })
    h8 = all(c["within_2pp"] for c in collapse) if collapse else None

    # metric-floor sanity inside the graded band: does the gate measure the
    # solver or the row-mean definition?
    interior_floor = [{"id": p["id"],
                       "measured_rms_pct": p["amp_profile_rms_pct"],
                       "floor_rms_pct": p["row_averaging_floor"]["interior_rows"]["amp_rms_pct"],
                       "solver_dominates": bool(p["amp_profile_rms_pct"]
                                                >= p["row_averaging_floor"]["interior_rows"]["amp_rms_pct"])}
                      for p in (_graded(points) if all("row_averaging_floor" in p
                                                       for p in points) else [])]

    boundary = None
    for prev, nxt in zip(sweep, sweep[1:]):
        if prev["passes_case_gates"] and not nxt["passes_case_gates"]:
            boundary = {"between_wo": [prev["womersley_number"], nxt["womersley_number"]],
                        "between_delta_over_dx": [prev["stokes_layer_over_dx"],
                                                  nxt["stokes_layer_over_dx"]],
                        "between_ids": [prev["id"], nxt["id"]]}

    failing = [p for p in graded if not p["passes_case_gates"]]
    return {
        "tolerance_pct": TOL,
        "rule_delta_over_dx": DELTA_OVER_DX_RULE,
        "H1_wo_free": {"supported": h1,
                       "n_sweep_points": len(sweep),
                       "max_wo_in_sweep": max((p["womersley_number"] for p in sweep),
                                              default=None),
                       "note": "every admissible H/dx = 16 sweep point is inside "
                               "G1..G4" if h1 else
                               "at least one admissible sweep point left the band"},
        "H2_layer_resolution": {"supported": h2, "per_point": h2_rows,
                                "note": "the predeclared delta/dx >= 3 rule does "
                                        "not describe the interior gate; it failed "
                                        "at the first probe" if h2 is False else
                                        "rule agrees with every graded point"},
        "H3_transient_artefact": {
            "supported": h3,
            "note": None if h3 is None else
                    ("doubling the settle time does not change the verdict, so the "
                     "band is not a settling artefact" if h3 else
                     "settling changes the verdict; the point was transient-limited")},
        "H4_noise_floor": {
            "supported": h4,
            "note": None if h4 is None else
                    ("wall-row error does not fall with refinement -> not a "
                     "resolution limit" if h4 else
                     "wall-row error falls with refinement -> spatial limit, not noise")},
        "H5_two_ceiling": {"supported": h5, "pairs": h5_pairs,
                           "invalid_points": [p["id"] for p in invalid],
                           "note": "every point with adequate temporal sampling is "
                                   "inside the band; the only out-of-scope point is "
                                   "sampling-starved (steps_per_period < 64) and is "
                                   "recovered by refining dx at the same Wo" if h5 else
                                   "an admissible-sampling point left the band, so the "
                                   "ceiling is not purely a sampling artefact"},
        "H6_interior_gate_blind_to_wall_layer": {
            "supported": h6,
            "ladder_ids": [p["id"] for p in ladder],
            "interior_band_all_below_tol": h6_blind,
            "ladder_reaches_delta_over_dx_below_1": h6_thin,
            "wall_row_error_monotone_in_delta": h6_monotone,
            "spearman_rho": h6_rho,
            "note": "G2/G4 average over 0.15 <= y/H <= 0.85 and therefore cannot see "
                    "the Stokes layer; the wall rows are the sensitive metric"},
        "H7_row_mean_floor_share": {
            "supported": h7, "per_point": floor_share,
            "note": None if h7 is None else
                    ("the row-mean definition explains at most half of the wall-row "
                     "error; the rest is solver smearing of the sub-dx layer" if h7 else
                     "the row-mean floor explains most of the wall-row error, so the "
                     "wall metric measures the metric, not the solver")},
        "H8_wall_error_collapse": {
            "supported": h8, "pairs": collapse,
            "note": None if h8 is None else
                    ("pairs matched in delta/dx but differing in (Wo, H/dx) agree "
                     "within 2 percentage points: the wall-row error is a function of "
                     "delta/dx alone" if h8 else
                     "matched-delta/dx pairs disagree by more than 2 pp, so the "
                     "wall-row error is not single-valued in delta/dx")},
        "interior_metric_floor": interior_floor,
        "measured_boundary_H_over_dx_16": boundary,
        "min_delta_over_dx_passing": min((p["stokes_layer_over_dx"] for p in graded),
                                         default=None),
        "n_points": len(points),
        "n_pass": sum(1 for p in graded if p["passes_case_gates"]),
        "n_fail": len(failing),
        "n_invalid": len(invalid),
    }


def build_document(points: list[dict], extra: dict | None = None) -> dict:
    points = sorted(points, key=lambda p: (p["womersley_requested"],
                                           p["resolution_H_over_dx"],
                                           p["id"]))
    doc = {
        "case": caserun.CASE_KEY,
        "tag": caserun.TAG,
        "study": "wo_envelope",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tolerance_pct": TOL,
        "hypotheses": HYPOTHESES,
        "predeclaration": PREDECLARATION,
        "predictions": PREDICTIONS,
        "parameters": {"height": caserun.HEIGHT, "viscosity": caserun.VISCOSITY,
                       "rho0": caserun.RHO0, "body_force": caserun.BODY_FORCE,
                       "sound_speed": 20.0, "columns": caserun.COLUMNS,
                       "samples_per_period": caserun.SAMPLES_PER_PERIOD,
                       "interior_band": [caserun.INTERIOR_LO, caserun.INTERIOR_HI],
                       "wall_band": list(WALL_BAND)},
        "points": points,
    }
    attach_floors(doc)
    doc["verdict"] = evaluate_hypotheses(doc["points"])
    if extra:
        doc.update(extra)
    return doc


def merge(docs: list[dict]) -> dict:
    by_id: dict[str, dict] = {}
    for doc in docs:
        for p in doc["points"]:
            by_id[p["id"]] = p
    extra = {}
    for doc in docs:
        for k in ("wall_clock_total_s", "provenance"):
            if k in doc:
                extra[k] = doc[k]
    if "wall_clock_total_s" in extra:
        extra["wall_clock_total_s"] = sum(d.get("wall_clock_total_s", 0.0) for d in docs)
    return build_document(list(by_id.values()), extra=extra or None)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", default="all",
                        help="comma-separated point ids, or 'all'")
    parser.add_argument("--out", type=Path,
                        default=REPO / "benchmarks/case_poiseuille/out/wo_envelope.json")
    parser.add_argument("--merge", type=Path, nargs="+", default=None,
                        help="merge existing part artefacts instead of running")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    if args.merge:
        docs = [json.loads(p.read_text()) for p in args.merge]
        doc = merge(docs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=2))
        v = doc["verdict"]
        print(f"merged {len(docs)} artefacts ({len(doc['points'])} points) -> {args.out}")
        print(f"H1 {v['H1_wo_free']['supported']} H2 {v['H2_layer_resolution']['supported']} "
              f"H5 {v['H5_two_ceiling']['supported']} "
              f"H6 {v['H6_interior_gate_blind_to_wall_layer']['supported']}")
        return 0

    if args.points == "all":
        wanted = [p[0] for p in POINTS]
    else:
        wanted = [s.strip() for s in args.points.split(",") if s.strip()]
    table = {p[0]: p for p in POINTS}
    missing = [w for w in wanted if w not in table]
    if missing:
        raise SystemExit(f"unknown point id(s): {missing}")

    import torch
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    points, t0 = [], time.time()
    for pid in wanted:
        point_id, wo, res, settle, window, role = table[pid]
        print(f"[{pid}] Wo={wo} H/dx={res} settle={settle}", flush=True)
        rec = run_point(point_id, wo, res, settle, window, role, device)
        print(f"    delta/dx={rec['stokes_layer_over_dx']:.2f} "
              f"spp={rec['steps_per_period']:.0f} "
              f"amp_centre={rec['amp_centre_error_pct']:+.2f}% "
              f"amp_rms={rec['amp_profile_rms_pct']:.2f}% "
              f"wall_worst={rec['wall_rows']['amp_worst_pct']:.2f}% "
              f"-> {rec['verdict'].upper()} ({rec['wall_clock_s']:.0f}s)", flush=True)
        points.append(rec)

    doc = build_document(points, extra={
        "wall_clock_total_s": time.time() - t0,
        "provenance": {
            "runner": "benchmarks/case_poiseuille/study_wo_envelope.py",
            "harness": "benchmarks/case_poiseuille/run.py:run_harmonic",
            "solver": "tensorsph.droplet_impact.rhs + tensorsph.wall_mirror",
            "reference": "tensorsph.womersley (sympy-verified closed form)",
        }})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"\n{len(points)} point(s), wall clock {doc['wall_clock_total_s']:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())