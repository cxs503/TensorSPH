"""CASE-SPH-WEDGE-ENTRY-2D -- deadrise response axis (beta = 20 / 30 / 40 deg).

Why this module exists
----------------------
Every reading of this case so far (m17548 ... m23165) lives at a **single**
deadrise angle, beta = 30 deg.  A case that is only ever evaluated at one angle
cannot separate "the solver reproduces wedge-impact physics" from "the solver
was tuned to one geometry", and -- more concretely -- it cannot say anything
about the reference-family question that dominates the residual: at 30 deg the
in-repo analytic anchor (Wagner asymptotic, ``Cp_max = C^2 + 1``) sits 21.4 %
above the mission-designated composite solution (Zhao & Faltinsen 1993, ``COM``)
and above the exact similarity solution (``EIM``).  Both published families are
tabulated in ``refs.json`` (Jain et al. 2022, JFM 934 A26, Table 1) at several
deadrise angles, so a second angle turns that tie into a test.

Two products, in this order:

``envelope``
    Reference side, zero SPH compute.  For every tabulated deadrise angle,
    compare the in-repo Wagner anchor against ``EIM`` (exact nonlinear
    similarity solution) and ``COM`` (the mission's metric owner).  This is a
    *range-of-validity* statement for the anchor the case is graded against: the
    Wagner peak is asymptotically exact (beta -> 0) and degrades monotonically
    with beta, so at 30 deg it can never be a 15 % reference.

``response``
    Solver side.  Run the same estimator the case uses
    (``by_frames["61"].resolution_band.peak_median``) at beta = 20/30/40 deg,
    dx = 4 mm, three sub-cell realisations per angle (the m22826 discrete-phase
    axis) and grade the **ratio** ``R(beta) = Cp(beta) / Cp(30 deg)`` against the
    same ratio taken from each reference family.

Why the ratio and not the absolute peak
---------------------------------------
At dx = 4 mm the absolute peak carries a large grid-dependent error (the
committed readings scatter between 6.8 and 13.4 q on this grid family).  That
error is, to leading order, a property of the *operator* (finite resolution,
pressure extrapolation, artificial viscosity) and multiplies the same way at
neighbouring angles, so it cancels in ``R``.  ``R`` therefore tests the
deadrise *scaling* independently of the resolution bias -- it does not replace
the acceptance metric and it is never reported as one.

Predeclared design (frozen before the first solver run of this axis)
-------------------------------------------------------------------
* angles: ``ANGLES = (20, 30, 40)`` deg, all three are rows of the published
  table in ``refs.json`` (no interpolation of a literature table);
* grid: ``DX = 4 mm`` (the coarse-but-affordable screen; the case's own ladder
  keeps 1 mm as its judgement grid -- this axis is a scaling test, not a
  convergence test);
* realisations: ``SEEDS = (1, 2, 3)`` through ``WedgeConfig.phase_x_dx`` /
  ``gap_jitter_dx`` (m22826);
* statistic: ``by_frames["61"].resolution_band.peak_median`` at that angle,
  identical to the committed estimator; median over seeds is the reading;
* reference ratios: ``COM`` (mission-designated) is primary, ``EIM`` is the
  independent exact-solution cross-check, the in-repo Wagner asymptotic is the
  diagnostic;
* tolerance: ``GATE = 0.15`` -- the case tolerance, reused, not widened;
* verdict rule (per non-reference angle):
  ``PASS_COM`` if ``|R_meas/R_COM - 1| <= GATE``, else
  ``EQUIVOCAL`` if ``|R_meas/R_Wagner - 1| <= GATE`` (the reading then
  discriminates between reference families instead of confirming one), else
  ``FAIL``;
* overall: ``RESPONSE_CONSISTENT`` iff every non-reference angle is
  ``PASS_COM``; anything else is reported as-is, honest FAIL included.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/deadrise_response.py envelope
    python benchmarks/case_sph_wedge_entry_2d/deadrise_response.py \
        run --beta 20 --dx 0.004 --seed 1 --threads 4 \
        --out docs/artifacts/case_sph_wedge_entry_2d_deadrise
    python benchmarks/case_sph_wedge_entry_2d/deadrise_response.py analyse
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import ensemble_peak as ep  # noqa: E402
import literature_refs as lr  # noqa: E402
from tensorsph.wedge_entry import (  # noqa: E402
    WedgeConfig,
    wagner_c_coeff,
    wagner_peak_over_q,
    with_realisation,
)

# --- predeclared design (do not edit after a run exists) ------------------- #
ANGLES = (20.0, 30.0, 40.0)
REFERENCE_ANGLE = 30.0
DX = 4.0e-3
SEEDS = (1, 2, 3)
GATE = 0.15                       # the case tolerance, reused
STATISTIC = 'by_frames["61"].resolution_band.peak_median'
FRAME_COUNT = "61"
PRIMARY = "COM"                   # mission-designated model (Zhao & Faltinsen 1993)
CROSSCHECK = "EIM"                # exact nonlinear similarity solution
DIAGNOSTIC = "wagner_1932"        # the in-repo asymptotic anchor

OUT = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_deadrise"


# --------------------------------------------------------------------------- #
# Reference side (no solver involved)
# --------------------------------------------------------------------------- #

def wagner_cp_max(beta_deg: float) -> float:
    """In-repo Wagner asymptotic peak in q units (``C^2 + 1``), recomputed."""
    return float(wagner_peak_over_q(math.radians(beta_deg)))


def reference_cp_max(beta_deg: float, model: str, refs: dict | None = None) -> float:
    """One published point-pressure model at a *tabulated* angle, in q units.

    ``model == "wagner_1932"`` is answered from the in-repo closed form instead
    of the table (the table does not list it); every other model is read out of
    ``refs.json`` so that no literature number is ever typed into this file.
    """
    if model == DIAGNOSTIC:
        return wagner_cp_max(beta_deg)
    return float(lr.cp_max_model(model, beta_deg, refs))


def reference_envelope(refs: dict | None = None) -> dict:
    """Wagner anchor vs the published exact/designated solutions, all angles.

    Returns ``{"rows": [...], "asymptote": {...}, "validity": {...}}`` where
    ``rows`` holds, per tabulated angle, the wetting coefficient ``C``, the
    in-repo Wagner peak and the published models, plus the relative gaps.
    """
    refs = refs or lr.load_refs()
    rows = []
    for key in lr.sorted_rows(refs):
        beta = float(key)
        C = wagner_c_coeff(math.radians(beta))
        w = wagner_cp_max(beta)
        eim = reference_cp_max(beta, CROSSCHECK, refs)
        com = reference_cp_max(beta, PRIMARY, refs)
        rows.append({
            "beta_deg": beta,
            "C": C,
            "cp_wagner": w,
            "cp_eim": eim,
            "cp_com": com,
            "wagner_vs_eim_rel": w / eim - 1.0,
            "wagner_vs_com_rel": w / com - 1.0,
            "com_vs_eim_rel": com / eim - 1.0,
        })
    smallest = rows[0]
    # Range of validity of the anchor *at the case tolerance*: the largest
    # tabulated angle whose Wagner-vs-EIM gap still fits in GATE.
    inside = [r["beta_deg"] for r in rows if abs(r["wagner_vs_eim_rel"]) <= GATE]
    return {
        "rows": rows,
        "asymptote": {
            "beta_deg": smallest["beta_deg"],
            "wagner_vs_eim_rel": smallest["wagner_vs_eim_rel"],
            "note": ("smallest tabulated deadrise; the Wagner asymptotic peak is "
                     "expected to coincide with the exact similarity solution "
                     "here, which is what makes the anchor usable at all"),
        },
        "validity": {
            "gate": GATE,
            "angles_inside_gate": inside,
            "max_angle_inside_gate": max(inside) if inside else None,
            "case_angle": REFERENCE_ANGLE,
            "case_angle_wagner_vs_eim_rel":
                next(r["wagner_vs_eim_rel"] for r in rows
                     if r["beta_deg"] == REFERENCE_ANGLE),
        },
    }


def ratio_reference(beta_deg: float, model: str,
                    refs: dict | None = None) -> float:
    """``Cp_ref(beta) / Cp_ref(30 deg)`` for one reference family."""
    refs = refs or lr.load_refs()
    return (reference_cp_max(beta_deg, model, refs)
            / reference_cp_max(REFERENCE_ANGLE, model, refs))


def reference_ratios(beta_deg: float, refs: dict | None = None) -> dict:
    refs = refs or lr.load_refs()
    return {m: ratio_reference(beta_deg, m, refs)
            for m in (PRIMARY, CROSSCHECK, DIAGNOSTIC)}


# --------------------------------------------------------------------------- #
# Solver side
# --------------------------------------------------------------------------- #

def run_one(beta_deg: float, dx: float, seed: int, threads: int,
            frames: int = 61, c0_factor: float = 20.0,
            velocity: float = 1.0) -> dict:
    """One realisation at one deadrise angle -> artefact payload."""
    t0 = time.time()
    case = ep.collect(dx, velocity, frames, c0_factor, threads,
                      seed=seed, beta_deg=beta_deg)
    result = ep.analyse(case, ep.BIN_GRID)
    rb = result["by_frames"][FRAME_COUNT]["resolution_band"]
    return {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "axis": "deadrise_response",
        "script": "benchmarks/case_sph_wedge_entry_2d/deadrise_response.py",
        "beta_deg": float(beta_deg),
        "realisation": {"seed": int(seed),
                        "phase_x_dx": case["cfg"].get("phase_x_dx"),
                        "gap_jitter_dx": case["cfg"].get("gap_jitter_dx")},
        "design": {"statistic": STATISTIC, "frame_count": FRAME_COUNT,
                   "bin_grid": list(ep.BIN_GRID), "gate": GATE,
                   "dx": dx, "c0_factor": c0_factor},
        "observed": {"peak_median": rb["peak_median"],
                     "spread_rel": rb["spread_rel"],
                     "n_bins_used": rb.get("n_bins_used"),
                     "judgeable": rb.get("judgeable")},
        "reference": {"cp_wagner": wagner_cp_max(beta_deg),
                      "cp_com": reference_cp_max(beta_deg, PRIMARY),
                      "cp_eim": reference_cp_max(beta_deg, CROSSCHECK),
                      "C": wagner_c_coeff(math.radians(beta_deg))},
        "config": case["cfg"], "steps": case["steps"],
        "wall_seconds": round(case["wall_seconds"], 1),
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform(), "threads": threads},
    }


def artefact_name(beta_deg: float, dx: float, seed: int) -> str:
    return f"deadrise_b{beta_deg:g}_dx{dx * 1e3:g}mm_seed{int(seed)}.json"


def load_artefacts(out_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(out_dir).glob("deadrise_b*_seed*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


# --------------------------------------------------------------------------- #
# Aggregation + predeclared judgement
# --------------------------------------------------------------------------- #

def _median(values):
    vals = [v for v in values if v == v]
    return statistics.median(vals) if vals else float("nan")


def _rel_sd(values):
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return float("nan")
    m = statistics.mean(vals)
    return statistics.stdev(vals) / m if m else float("nan")


def analyse(arts: list[dict], gate: float = GATE) -> dict:
    """Per-grid, per-angle readings, ratios and the predeclared verdict.

    Grids are analysed **separately**: a ratio may only be formed between
    readings taken on the same dx, otherwise a grid error would enter as a
    deadrise error.

    The estimator uncertainty of a reading is ``max(rel_sd over realisations,
    median band spread)`` -- the same "two denominators, take the larger"
    convention the case used in m22826/m23165 -- and a reading whose
    uncertainty exceeds ``gate`` is ``NOT_JUDGEABLE``: the axis then reports the
    number but refuses to grade it.
    """
    grids: dict[float, list[dict]] = {}
    for a in arts:
        grids.setdefault(float(a["design"]["dx"]), []).append(a)

    out = {}
    for dx in sorted(grids):
        by_angle: dict[float, list[dict]] = {}
        for a in grids[dx]:
            by_angle.setdefault(float(a["beta_deg"]), []).append(a)
        readings = {}
        for beta in sorted(by_angle):
            group = by_angle[beta]
            obs = [a["observed"]["peak_median"] for a in group]
            spreads = [a["observed"]["spread_rel"] for a in group]
            rel_sd = _rel_sd(obs)
            band = _median(spreads)
            cand = [v for v in (rel_sd, band) if v == v]
            unc = max(cand) if cand else float("nan")
            readings[beta] = {
                "n_realisations": len(obs),
                "seeds": sorted(a["realisation"]["seed"] for a in group),
                "peak_median_per_seed": obs,
                "peak_median": _median(obs),
                "rel_sd": rel_sd,
                "band_spread_median": band,
                "estimator_uncertainty": unc,
                "judgeable": bool(unc == unc and unc <= gate),
                "reference": group[0]["reference"],
            }
        if REFERENCE_ANGLE not in readings:
            raise KeyError(
                f"dx={dx:g}: no artefact at the reference angle "
                f"{REFERENCE_ANGLE} deg, cannot form a ratio")
        ref = readings[REFERENCE_ANGLE]
        rows = []
        for beta in sorted(readings):
            if beta == REFERENCE_ANGLE:
                continue
            rd = readings[beta]
            r_meas = rd["peak_median"] / ref["peak_median"]
            r_refs = reference_ratios(beta)
            parts = [v for v in (rd["estimator_uncertainty"],
                                 ref["estimator_uncertainty"]) if v == v]
            u_r = math.sqrt(sum(v * v for v in parts)) if parts else float("nan")
            resid = {m: (r_meas / r - 1.0) for m, r in r_refs.items()}
            if not rd["judgeable"] or not ref["judgeable"]:
                verdict = "NOT_JUDGEABLE"
            elif abs(resid[PRIMARY]) <= gate:
                verdict = "PASS_COM"
            elif abs(resid[DIAGNOSTIC]) <= gate:
                verdict = "EQUIVOCAL"
            else:
                verdict = "FAIL"
            rows.append({
                "beta_deg": beta,
                "cp_measured": rd["peak_median"],
                "cp_measured_rel_sd": rd["rel_sd"],
                "cp_measured_band_spread": rd["band_spread_median"],
                "ratio_measured": r_meas,
                "ratio_uncertainty": u_r,
                "ratio_reference": r_refs,
                "residual_vs_ratio": resid,
                "resolvable_outside_own_uncertainty":
                    bool(resid[PRIMARY] == resid[PRIMARY]
                         and abs(resid[PRIMARY]) > u_r),
                "verdict": verdict,
            })
        judged = [r for r in rows if r["verdict"] != "NOT_JUDGEABLE"]
        out[f"{dx:g}"] = {
            "dx": dx,
            "readings": {f"{b:g}": v for b, v in readings.items()},
            "ratios": rows,
            "verdict": ("NO_JUDGEABLE_ANGLE" if not judged else
                        "RESPONSE_CONSISTENT"
                        if all(r["verdict"] == "PASS_COM" for r in judged)
                        else "RESPONSE_NOT_CONSISTENT"),
        }

    return {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "axis": "deadrise_response",
        "design": {"angles": list(ANGLES), "reference_angle": REFERENCE_ANGLE,
                   "seeds": list(SEEDS), "gate": gate,
                   "statistic": STATISTIC,
                   "primary": PRIMARY, "crosscheck": CROSSCHECK,
                   "diagnostic": DIAGNOSTIC},
        "grids": out,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("run", "analyse", "envelope"))
    ap.add_argument("--beta", type=float, default=REFERENCE_ANGLE)
    ap.add_argument("--dx", type=float, default=DX)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    if args.action == "run":
        payload = run_one(args.beta, args.dx, args.seed, args.threads,
                          frames=args.frames)
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / artefact_name(args.beta, args.dx, args.seed)
        path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(f"{path}  obs={payload['observed']['peak_median']:.3f}  "
              f"band_spread={payload['observed']['spread_rel'] * 100:.1f}%  "
              f"steps={payload['steps']}  {payload['wall_seconds']}s")
        return 0

    if args.action == "envelope":
        payload = reference_envelope()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "reference_envelope.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True))
        print(f"{'beta':>5} {'C':>8} {'Wagner':>9} {'EIM':>9} {'COM':>9} "
              f"{'w/EIM-1':>9} {'w/COM-1':>9}")
        for r in payload["rows"]:
            print(f"{r['beta_deg']:5.0f} {r['C']:8.3f} {r['cp_wagner']:9.3f} "
                  f"{r['cp_eim']:9.3f} {r['cp_com']:9.3f} "
                  f"{r['wagner_vs_eim_rel'] * 100:+8.2f}% "
                  f"{r['wagner_vs_com_rel'] * 100:+8.2f}%")
        v = payload["validity"]
        print(f"anchor inside {v['gate'] * 100:g}% up to beta = "
              f"{v['max_angle_inside_gate']:g} deg; at the case angle "
              f"{v['case_angle']:g} deg it is "
              f"{v['case_angle_wagner_vs_eim_rel'] * 100:+.2f}%")
        return 0

    summary = analyse(load_artefacts(args.out))
    (Path(args.out) / "deadrise_response.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True))
    for tag, grid in summary["grids"].items():
        for beta, rd in grid["readings"].items():
            print(f"dx={tag} beta={beta:>2}  Cp={rd['peak_median']:8.3f}  "
                  f"u={rd['estimator_uncertainty'] * 100:5.1f}%  "
                  f"{'judgeable' if rd['judgeable'] else 'NOT_JUDGEABLE'}")
        for r in grid["ratios"]:
            print(f"dx={tag} beta={r['beta_deg']:g}  R={r['ratio_measured']:.3f} "
                  f"(u={r['ratio_uncertainty'] * 100:.1f}%) "
                  f"R_com={r['ratio_reference'][PRIMARY]:.3f} "
                  f"R_wag={r['ratio_reference'][DIAGNOSTIC]:.3f} "
                  f"res_com={r['residual_vs_ratio'][PRIMARY] * 100:+.1f}%  "
                  f"{r['verdict']}")
        print(f"dx={tag}: {grid['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())