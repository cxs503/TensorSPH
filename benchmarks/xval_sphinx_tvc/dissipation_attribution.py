"""Attribution: is the dam-break gain of TVC the operator, or just dissipation?

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

The question this closes
------------------------
``run.py`` / ``uncalibrated.py`` grade the ported transport-velocity correction
by the mean error of ``Z/a(tau)`` against Martin & Moyce (1952).  Two claims are
made from that number:

* ``docs/xval_sphinx_tvc/VALIDATION_REPORT.md``: "TVC reduces MAE from 0.621 to
  0.143 (4.3x)";
* ``UNCALIBRATED_DEFAULTS.md``: with the coefficients SPHinXsys actually ships
  (0.2 dual / 0.05 single acoustic step) the gate is *not* met.

Both read the improvement as an *operator* result.  But ``Z/a`` is a single
integral measure of a flow whose late-time front speed is controlled by how
much momentum the floor takes away.  The published runs all use
``wall_friction = 0`` (**free-slip floor**) and the default laminar viscosity,
i.e. the tank Martin & Moyce used with a **no-slip** floor is not reproduced.
A front that overshoots because the floor is slippery can be slowed down by
*any* dissipation -- including the numerical dissipation a position correction
adds.  So "TVC fixes ``Z/a``" is confounded unless a dissipation-only path to
the same error is exhibited.

What is measured here
---------------------
Three axes, all through the *same* solver path as ``run.py`` (``run_variant``
is imported, not re-implemented):

``tvc_alpha``       ``None`` (off) / 0.05 / 0.10 / 0.20
``viscosity``       laminar viscosity coefficient (default 0.01)
``wall_friction``   0 = free slip (the published setting) / 1 = no-slip floor

Predeclared questions and decision rule (fixed before the runs):

Q1 *physical fix*: does the **no-slip** floor, with TVC **off**, meet the owner
   gate (< 15 %) on its own?  If yes, the published free-slip baseline is a
   configuration defect, not evidence about TVC.
Q2 *dissipation substitution*: is there a laminar-viscosity value that, with TVC
   **off**, matches or beats the calibrated ``alpha = 0.10`` error?  Reported as
   the resolved viscosity and the error margin.
Q3 *unique signature*: does TVC do something dissipation cannot -- i.e. is the
   lattice-regularity residual (``grad1_residual_mean_h``, ``median_nn_over_dx``)
   better for TVC at equal ``Z/a`` error?  This is the only channel in which the
   operator can be credited without confounding.

A claim of the form "TVC improves ``Z/a``" is only *attributable* if Q1 and Q2
both answer no.  Otherwise the honest statement is that the ``Z/a`` gain is a
dissipation effect that TVC happens to supply, and the operator's own merit has
to be read off Q3.

Note on the fidelity of "no-slip": ``wall_friction`` blends the viscosity
between fluid and *static* wall particles (zero velocity), which is a no-slip
tangential condition imposed through the viscous term -- not a wall boundary
layer model.  It is the closest thing this solver has to Martin & Moyce's
floor, and it is reported as such.

Usage
-----
    python benchmarks/xval_sphinx_tvc/dissipation_attribution.py \
        --out docs/xval_sphinx_tvc/data/dissipation_attribution.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import torch  # noqa: E402

from xval_sphinx_tvc.run import (OWNER_GATE_REL, LEDGER_GATE_ABS,  # noqa: E402
                 clumping_metrics, load_experiment, nondim_front, score)

from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402

ANCHOR = "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION"
#: Digitisation band of the reference (``Z/a``), propagated as in run.py.
DIGITISATION_BAND = 0.05

#: The calibrated coefficient the anchor reports (``VALIDATION_REPORT.md``).
CALIBRATED_ALPHA = 0.10
#: The two coefficients SPHinXsys ships (see UNCALIBRATED_DEFAULTS.md).
UPSTREAM_SINGLE_STEP = 0.05
UPSTREAM_DUAL_CRITERIA = 0.20
#: Solver default laminar viscosity (``droplet_impact.Config.viscosity``).
DEFAULT_VISCOSITY = 0.01


def _tvc(alpha):
    return {} if alpha is None else dict(tvc_every=1, tvc_alpha=float(alpha),
                                         tvc_scope="all")


# --------------------------------------------------------------------------
# Predeclared grid.  Every row is (label, tvc_alpha, viscosity, wall_friction).
# The rows are chosen so each of Q1/Q2/Q3 is answered by measurement:
#
#   * ``slip_base`` / ``slip_a005`` / ``slip_a010`` / ``slip_a020`` replicate the
#     published free-slip rows, so this payload can be checked against
#     ``data/grid_dx003.json`` without trusting it;
#   * ``noslip_base`` is Q1;
#   * ``slip_visc*`` is Q2 (dissipation substituted for the correction);
#   * ``slip_a010_visc10`` is the additive control: dissipation *and* the
#     calibrated correction -- if the two are complements the error should fall
#     further, if they are the same mechanism it should not;
#   * ``noslip_visc10`` is the physical-fix-plus-dissipation corner.
# --------------------------------------------------------------------------
GRID = (
    ("slip_base", None, DEFAULT_VISCOSITY, 0.0),
    ("slip_a005", UPSTREAM_SINGLE_STEP, DEFAULT_VISCOSITY, 0.0),
    ("slip_a010", CALIBRATED_ALPHA, DEFAULT_VISCOSITY, 0.0),
    ("slip_a020", UPSTREAM_DUAL_CRITERIA, DEFAULT_VISCOSITY, 0.0),
    ("noslip_base", None, DEFAULT_VISCOSITY, 1.0),
    ("noslip_a010", CALIBRATED_ALPHA, DEFAULT_VISCOSITY, 1.0),
    ("slip_visc05", None, 0.05, 0.0),
    ("slip_visc20", None, 0.20, 0.0),
    ("slip_visc50", None, 0.50, 0.0),
    ("slip_a010_visc10", CALIBRATED_ALPHA, 0.10, 0.0),
    ("noslip_visc10", None, 0.10, 1.0),
    ("noslip_visc20", None, 0.20, 1.0),
)


def run_row(label: str, alpha, viscosity: float, wall_friction: float,
            dx: float, seconds: float) -> dict:
    """One grid point, through ``run.py``'s own ``nondim_front``/``score``."""
    dt = DamBreakConfig(dx=dx).dt_from_courant()
    steps = int(round(seconds / dt))
    cfg = DamBreakConfig(dx=dx, dt=dt, steps=steps,
                         output_every=max(1, steps // 60), device="cpu",
                         viscosity=viscosity, wall_friction=wall_friction,
                         **_tvc(alpha))
    t0 = time.time()
    run = run_case(cfg)
    wall = time.time() - t0
    tau, za = nondim_front(run.times, run.front, cfg.column_width, cfg.gravity)
    sc = score(tau, za, load_experiment())
    return {
        "label": label,
        "tvc_alpha": alpha,
        "viscosity": viscosity,
        "wall_friction": wall_friction,
        "dx": dx,
        "dt": dt,
        "steps": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": wall,
        "tau": tau,
        "Z_over_a": za,
        # A front that runs away late is the signature of a slippery floor;
        # record it so the mechanism claim is checkable from the payload.
        "front_over_a_final": za[-1] if za else None,
        "mae": sc["mae"],
        "mean_rel_error": sc["mean_rel_error"],
        "max_abs_error": sc["max_abs_error"],
        "gate_abs_pass": sc["gate_abs_pass"],
        "gate_rel_pass": sc["gate_rel_pass"],
        "rows": sc["rows"],
        "max_density_compression": run.max_density_compression,
        "mass_drift": run.mass_drift,
        "max_tvc_shift_over_h": run.max_tvc_shift / (cfg.h_factor * cfg.dx),
        "clumping": clumping_metrics(run.final_positions, dx,
                                     cfg.h_factor * dx),
    }


def band_verdict(record: dict, exp: dict) -> str:
    """``PASS`` / ``PASS_marginal`` / ``FAIL`` under the digitisation band.

    Same rule as ``uncalibrated.py``: the gate has to hold for the nominal
    reference *and* for the reference shifted by ``+-band``, otherwise the
    verdict is labelled marginal instead of a pass.  Computed from the stored
    ``tau``/``Z_over_a`` samples so it is reproducible from the payload alone.
    """
    verdicts = []
    for shift in (-DIGITISATION_BAND, 0.0, DIGITISATION_BAND):
        shifted = json.loads(json.dumps(exp))
        for d in shifted["data"]:
            d["Z_over_a"] = d["Z_over_a"] + shift
        sc = score(record["tau"], record["Z_over_a"], shifted)
        ok = bool(sc["gate_rel_pass"]) and bool(sc["gate_abs_pass"])
        verdicts.append(ok)
    if all(verdicts):
        return "PASS"
    if any(verdicts):
        return "PASS_marginal"
    return "FAIL"


def summarise(records: list, exp: dict) -> dict:
    """Answer Q1/Q2/Q3 from the measured records (no hand-typed numbers)."""
    by = {r["label"]: r for r in records}

    def best(labels):
        cand = [by[l] for l in labels if l in by]
        return min(cand, key=lambda r: r["mean_rel_error"]) if cand else None

    slip_only = [l for l, a, v, w in GRID if w == 0.0 and a is None]
    noslip_only = [l for l, a, v, w in GRID if w == 1.0 and a is None]
    q2 = best(slip_only)
    q1 = best(noslip_only)
    calibrated = by.get("slip_a010")

    def margin(rec):
        if rec is None or calibrated is None:
            return None
        return rec["mean_rel_error"] - calibrated["mean_rel_error"]

    # Q3: compare lattice regularity at the two points that reach the same
    # Z/a error by different means.
    def regularity(rec):
        c = (rec or {}).get("clumping", {})
        return {"grad1_residual_mean_h": c.get("grad1_residual_mean_h"),
                "median_nn_over_dx": c.get("median_nn_over_dx"),
                "close_pair_fraction": c.get("close_pair_fraction")}

    return {
        "q1_noslip_physical_fix": {
            "best_noslip_no_tvc": q1["label"] if q1 else None,
            "mean_rel_error": q1["mean_rel_error"] if q1 else None,
            "gate_met_without_tvc": bool(q1 and q1["gate_rel_pass"]),
            "note": "Q1: no-slip floor, correction off",
        },
        "q2_dissipation_substitution": {
            "best_slip_no_tvc": q2["label"] if q2 else None,
            "mean_rel_error": q2["mean_rel_error"] if q2 else None,
            "calibrated_tvc": calibrated["label"] if calibrated else None,
            "calibrated_mean_rel_error": (calibrated["mean_rel_error"]
                                          if calibrated else None),
            "error_difference_pp": (None if margin(q2) is None
                                    else 100.0 * margin(q2)),
            "dissipation_reaches_calibrated_within_1pp": (
                None if margin(q2) is None else abs(margin(q2)) <= 0.01),
            "note": "Q2: laminar viscosity alone, correction off",
        },
        "q3_unique_signature": {
            "at_calibrated_tvc": regularity(calibrated),
            "at_matched_dissipation": regularity(q2),
            "note": "Q3: distributional regularity cannot be produced by the "
                    "viscous term, so this is the only channel that credits "
                    "the operator itself",
        },
        "records": len(records),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=0.03,
                    help="resolution (0.03 = the anchor's headline row)")
    ap.add_argument("--seconds", type=float, default=0.45,
                    help="simulated time, same as grid_dx003.json")
    ap.add_argument("--rows", nargs="+", default=None,
                    help="subset of labels (default: the predeclared grid)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "xval_sphinx_tvc" / "data"
                    / "dissipation_attribution.json")
    args = ap.parse_args(argv)

    rows = [g for g in GRID if args.rows is None or g[0] in args.rows]
    records = []
    for label, alpha, visc, friction in rows:
        rec = run_row(label, alpha, visc, friction, args.dx, args.seconds)
        rec["band_verdict"] = band_verdict(rec, load_experiment())
        records.append(rec)
        print(f"{label:<18} alpha={str(alpha):<5} visc={visc:<5} "
              f"fric={friction:<4} mae={rec['mae']:.4f} "
              f"rel={rec['mean_rel_error']:.4f} "
              f"front/a_end={rec['front_over_a_final']:.3f} "
              f"{rec['band_verdict']}  [{rec['wall_clock_s']:.1f}s]")

    payload = {
        "anchor": ANCHOR,
        "question": "is the Z/a gain of the ported TVC attributable to the "
                    "operator, or is it dissipation the operator happens to add?",
        "gates": {"owner_rel": OWNER_GATE_REL, "ledger_abs": LEDGER_GATE_ABS},
        "digitisation_band": DIGITISATION_BAND,
        "dx": args.dx,
        "seconds": args.seconds,
        "grid_declared": [
            {"label": l, "tvc_alpha": a, "viscosity": v, "wall_friction": w}
            for l, a, v, w in GRID],
        "reference": load_experiment().get("reference", {}),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
        },
        "records": records,
        "summary": summarise(records, load_experiment()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    s = payload["summary"]

    def _fmt(value, spec=".4f"):
        return "n/a" if value is None else format(value, spec)

    q1, q2 = s["q1_noslip_physical_fix"], s["q2_dissipation_substitution"]
    print(f"Q1 no-slip, no TVC : {_fmt(q1['mean_rel_error'])}"
          f"  gate_met={q1['gate_met_without_tvc']}")
    print(f"Q2 viscosity only  : {_fmt(q2['mean_rel_error'])}"
          f"  vs calibrated {_fmt(q2['calibrated_mean_rel_error'])}"
          f"  diff={_fmt(q2['error_difference_pp'], '+.1f')}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())