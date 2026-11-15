#!/usr/bin/env python3
"""Discretization-uncertainty / validation layer for CASE-POISEUILLE-SPH.

The case (``benchmarks/case_poiseuille``) grades nine gates against the analytic
Womersley / Boussinesq references on a three-rung ladder ``H/dx = 8, 12, 16``.
Its verification note reports the *point* errors at the finest rung and, for the
ladder, only that "the convergence can be read" from the table.  Whether those
point errors lie inside the *numerical uncertainty* -- i.e. whether the residual
is discretization error that will keep falling, or a floor no refinement will
remove -- was never stated.  That is the verification debt this script pays.

It reads the committed ``out/result.json`` (no solver run) and, for each graded
functional, reports:

* the measured error on every rung against the *per-rung* analytic reference
  (the row centres move with the resolution, so one reference value would be
  wrong);
* the observed order of accuracy from the rung pairs **with a status**: a ladder
  whose error changes sign, stalls, or whose pair orders disagree has no order,
  and none is quoted;
* the Grid Convergence Index of the finest rung (Roache 1994, Fs = 1.25),
  keeping the conservative ``p = 1`` band as the floor of the claim;
* the ASME V&V 20-2009 comparison ``E`` vs ``U_V = hypot(U_num, U_input)``,
  ``U_input`` being the run's own between-window repeatability;
* the repeatability floor, so a residual sitting on the noise floor is
  recognised as such instead of being sold as "converging at order 3";
* whether the uncertainty band is narrow enough to *support* the declared 5 %
  tolerance (gate U3) -- the question a point error cannot answer.

Functionals are defined on a **common y grid**: the row values of every rung are
line-interpolated onto the same interior grid before the norm is taken.  Taking
the norm over each rung's own rows instead makes the functional itself
resolution-dependent (the band edges move), which inflates the pair differences
by an order of magnitude and turns the GCI into noise -- that mistake is
reproducible with ``--rowwise-norm`` and is why the common grid is the default.

Outputs
-------
``out/uncertainty.json``                          all numbers, no hand copying
``docs/case_poiseuille_uncertainty.md``            generated note
``docs/figures/case_poiseuille_uncertainty.png``   error vs dx + bands

Usage
-----
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/uncertainty.py [--no-doc]

Exit status is non-zero when a declared gate (U1/U2/U3) fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph import womersley as wom  # noqa: E402
from tensorsph.verification.uncertainty import (  # noqa: E402
    CONSERVATIVE_ORDER,
    DEFAULT_SAFETY,
    Rung,
    ladder_analysis,
    repeatability,
)

RESULT = HERE / "out" / "result.json"
OUT_JSON = HERE / "out" / "uncertainty.json"
DOC = REPO / "docs" / "case_poiseuille_uncertainty.md"
FIG = REPO / "docs" / "figures" / "case_poiseuille_uncertainty.png"

TOL_PERCENT = 5.0          # feature-ledger acceptance for the case
INTERIOR = (0.15, 0.85)    # y/H band used by gates G2 and G4
GRID_N = 41                # points of the common y grid on that band
QUARTER = math.pi / 2.0


# --------------------------------------------------------------------------- #
# functionals
# --------------------------------------------------------------------------- #
def _common_grid(n: int = GRID_N) -> list[float]:
    lo, hi = INTERIOR
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _lagrange(xs: list[float], vs: list[float], grid: list[float], k: int = 5) -> list[float]:
    """Interpolate row values onto the common grid with ``k``-point Lagrange.

    The interpolation is part of the measurement protocol, so its order matters:
    linear interpolation between rows leaves an O(dx^2) residual that *floors* the
    profile norm at about 0.1 % whatever the solver does (observed: the band-L2
    error stalls at 0.1069 % between H/dx=12 and 16, i.e. the functional stops
    measuring the solver and starts measuring the interpolant).  A 5-point
    Lagrange stencil on the same rows restores a clean decay of the norm error
    (1.04 -> 0.44 -> 0.19 %), so that is the default.  Node count is clipped to
    the available rows near the band edges.
    """
    out: list[float] = []
    for x in grid:
        idx = sorted(range(len(xs)), key=lambda i: abs(xs[i] - x))[:k]
        total = 0.0
        for i in idx:
            basis = 1.0
            for j in idx:
                if j != i:
                    basis *= (x - xs[j]) / (xs[i] - xs[j])
            total += vs[i] * basis
        out.append(total)
    return out


def _on_grid(rows: list[dict], key: str, grid: list[float], k: int = 5) -> list[float]:
    ys = [r["y_over_H"] for r in rows]
    vs = [r[key] for r in rows]
    return _lagrange(ys, vs, grid, k)


def _l2(xs: list[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def _rowwise_norm(rows: list[dict], key: str) -> float:
    """The resolution-dependent variant, kept only to expose the pitfall."""
    band = [r[key] for r in rows if INTERIOR[0] <= r["y_over_H"] <= INTERIOR[1]]
    return _l2(band)


def analytic_profile(result: dict, run: dict, grid: list[float]) -> tuple[list[float], list[float]]:
    """Exact closed-form amplitude/phase at the fixed common-grid points.

    The reference must *not* be interpolated from the rung's own rows: that makes
    the reference move with the resolution and leaves an O(dx^2) interpolation
    residual in the comparison, which then masquerades as solver error (0.19 % at
    H/dx=16 for a solver whose centre-line error is 0.07 %).  `tensorsph.womersley`
    supplies the closed form the case itself uses, so the reference is a fixed
    scalar for every rung.
    """
    g, par = result["geometry"], result["parameters"]
    ys = [yH * g["height"] for yH in grid]
    amp = wom.amplitude(ys, height=g["height"], body_force=g["body_force"],
                        viscosity=g["viscosity"], omega=run["omega"])
    pha = wom.phase(ys, height=g["height"], body_force=g["body_force"],
                    viscosity=g["viscosity"], omega=run["omega"])
    return [float(a) for a in amp], [float(p) for p in pha]


def reference_consistency(result: dict) -> dict:
    """Do the stored analytic row values equal the library closed form?

    Guards the reference itself: if the case ever evaluates the analytic profile
    differently from `tensorsph.womersley`, every error in this note would be
    measured against the wrong reference.  Cheap (no solver), so it is checked
    for every rung and both rows arrays.
    """
    g = result["geometry"]
    worst = 0.0
    for run in result["harmonic_runs"]:
        ys = [r["y"] for r in run["rows"]]
        amp = wom.amplitude(ys, height=g["height"], body_force=g["body_force"],
                            viscosity=g["viscosity"], omega=run["omega"])
        pha = wom.phase(ys, height=g["height"], body_force=g["body_force"],
                        viscosity=g["viscosity"], omega=run["omega"])
        for ref, calc in ((a["amplitude_analytic"], b) for a, b in zip(run["rows"], amp)):
            worst = max(worst, float(abs(ref / calc - 1.0)))
        for ref, calc in ((a["phase_analytic"], b) for a, b in zip(run["rows"], pha)):
            worst = max(worst, float(abs(ref / calc - 1.0) if calc else abs(ref - calc)))
    worst = float(worst)
    return {"max_relative_deviation": worst,
            "pass": bool(worst < 1e-9),
            "note": "stored row analytic values vs tensorsph.womersley closed form"}


def quantities(result: dict, common_grid: bool = True) -> dict:
    """The four graded functionals, each normalised so its tolerance is a plain %.

    Amplitude functionals are relative (scale = |reference|); phase functionals
    are expressed in quarter turns and carry ``scale = 1.0`` so that "5 %" means
    5 % of a quarter turn, exactly as the case declares it.
    """
    runs = result["harmonic_runs"]
    grid = _common_grid()
    keys = ("amp_centre", "amp_profile_l2", "phase_centre", "phase_profile_l2")
    q = {k: {"spacings": [], "values": [], "truth": [], "labels": [], "scale": None}
         for k in keys}

    for run in runs:
        dx = result["geometry"]["height"] / run["resolution_H_over_dx"]
        label = f"H/dx={run['resolution_H_over_dx']}"
        rows = run["rows"]

        if common_grid:
            amp_m = _l2(_on_grid(rows, "amplitude", grid))
            ph_m = _l2(_on_grid(rows, "phase", grid))
            amp_ref, ph_ref = analytic_profile(result, run, grid)
            amp_a, ph_a = _l2(amp_ref), _l2(ph_ref)
        else:  # the pitfall variant: functional defined on each rung's own rows
            band = [r for r in rows if INTERIOR[0] <= r["y_over_H"] <= INTERIOR[1]]
            amp_m = _l2([r["amplitude"] for r in band])
            ph_m = _l2([r["phase"] for r in band])
            amp_a = _l2([r["amplitude_analytic"] for r in band])
            ph_a = _l2([r["phase_analytic"] for r in band])

        entries = {
            "amp_centre": (run["amp_centre"], run["amp_centre_analytic"],
                           "relative, |S-T|/|T|"),
            "amp_profile_l2": (amp_m, amp_a,
                               "relative L2 norm of the amplitude profile on the "
                               f"common grid over {INTERIOR[0]}<=y/H<={INTERIOR[1]}"),
            "phase_centre": (run["phase_centre_rad"] / QUARTER,
                             run["phase_centre_analytic_rad"] / QUARTER,
                             "rad/(pi/2): per cent of a quarter turn"),
            "phase_profile_l2": (ph_m / QUARTER, ph_a / QUARTER,
                                 "L2 norm of the phase profile on the common grid, "
                                 "per cent of a quarter turn"),
        }
        for key, (val, ref, scale_text) in entries.items():
            q[key]["spacings"].append(dx)
            q[key]["values"].append(val)
            q[key]["truth"].append(ref)
            q[key]["labels"].append(label)
            q[key]["scale"] = (1.0 if key.startswith("phase") else None)
            q[key]["scale_text"] = scale_text
    return q


def input_uncertainty(result: dict) -> dict:
    """Repeatability of the finest-rung measurement (``U_input``).

    The two measurement windows of the periodic state are two independent
    estimates of the same Fourier coefficient, so their half-range is the run's
    own statistical scatter -- the honest ``U_input`` here: the case is
    deterministically forced and laminar, so the only non-determinism is the
    sampling window.  Phase scatter is expressed in quarter turns.
    """
    finest = max(result["harmonic_runs"], key=lambda r: r["resolution_H_over_dx"])
    amps = [w["amplitude"] for w in finest["windows"]]
    phases = [w["phase"] / QUARTER for w in finest["windows"]]
    ph = repeatability(phases)
    return {
        "resolution_H_over_dx": finest["resolution_H_over_dx"],
        "amplitude": repeatability(amps),
        "phase_quarter_turns": {**ph, "half_range_percent": 100.0 * ph["half_range"]},
        "fit_residual_max": finest["fit_residual_max"],
        "note": ("half-range of the two measurement windows of the finest rung, "
                 "in per cent of the measured value (amplitude) or of a quarter "
                 "turn (phase); U_input, not U_num"),
    }


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyse(result: dict, common_grid: bool = True) -> dict:
    q = quantities(result, common_grid=common_grid)
    u_in = input_uncertainty(result)
    u_input_pct = {
        "amp_centre": u_in["amplitude"]["half_range_percent_of_mean"],
        "amp_profile_l2": u_in["amplitude"]["half_range_percent_of_mean"],
        "phase_centre": u_in["phase_quarter_turns"]["half_range_percent"],
        "phase_profile_l2": u_in["phase_quarter_turns"]["half_range_percent"],
    }

    quantities_out: dict[str, dict] = {}
    for key, spec in q.items():
        rungs = [Rung(h, v, lab)
                 for h, v, lab in zip(spec["spacings"], spec["values"], spec["labels"])]
        rep = ladder_analysis(
            rungs,
            truth=spec["truth"],
            tolerance_percent=TOL_PERCENT,
            u_input_percent=u_input_pct[key],
            safety=DEFAULT_SAFETY,
            scale=spec["scale"],
        )
        # Errors must follow the *report* order (fine -> coarse), which the
        # analysis sorted; zipping the unsorted input lists with the sorted
        # values silently labels the finest rung with the coarsest error.
        scale_denom = spec["scale"]
        errs_pct = [
            100.0 * abs(v - t) / (abs(scale_denom) if scale_denom else abs(t))
            for v, t in zip(rep["values"], rep["truth_per_rung"])
        ]
        rep["scale_text"] = spec["scale_text"]
        rep["error_percent_per_rung"] = errs_pct
        rep["worst_rung_error_percent"] = max(errs_pct)
        rep["worst_rung_within_tolerance"] = max(errs_pct) <= TOL_PERCENT
        rep["repeatability_floor_percent"] = u_input_pct[key]
        rep["error_over_floor"] = (
            errs_pct[0] / u_input_pct[key] if u_input_pct[key] > 0 else None)
        quantities_out[key] = rep

    const = result["constant_run"]
    single = {
        "steady_profile_rms": {
            "resolution_H_over_dx": const["resolution_H_over_dx"],
            "error_percent": const["steady_profile_rms_pct"],
            "within_tolerance": const["steady_profile_rms_pct"] <= TOL_PERCENT,
        },
        "startup_profile_rms_at_0.40s": {
            "resolution_H_over_dx": const["resolution_H_over_dx"],
            "error_percent": const["startup_rms_pct_at_040"],
            "within_tolerance": const["startup_rms_pct_at_040"] <= TOL_PERCENT,
        },
        "status": "insufficient",
        "reason": ("the committed R1 artifacts contain a single resolution "
                   "(H/dx=12): neither an order nor a GCI can be estimated, so the "
                   "R1 anchor is graded against the tolerance only"),
    }

    ref = reference_consistency(result)
    gates = {
        "U0_reference_matches_closed_form": ref["pass"],
        "U1_all_rungs_within_5pct": all(
            v["worst_rung_within_tolerance"] for v in quantities_out.values()),
        "U1b_R1_anchor_within_5pct": all(
            v["within_tolerance"] for v in single.values() if isinstance(v, dict)
            and "within_tolerance" in v),
        "U2_validation_E_leq_U_V": all(
            (v.get("validation") or {}).get("verdict") == "validated"
            for v in quantities_out.values()),
        "U3_band_supports_5pct": all(
            (v.get("validation") or {}).get("supports_tolerance") is True
            for v in quantities_out.values()),
    }

    return {
        "case": result["case"],
        "generated_by": "benchmarks/case_poiseuille/uncertainty.py",
        "source_artifact": str(RESULT.relative_to(REPO)),
        "source_sha256": hashlib.sha256(RESULT.read_bytes()).hexdigest(),
        "tolerance_percent": TOL_PERCENT,
        "safety_factor": DEFAULT_SAFETY,
        "conservative_order": CONSERVATIVE_ORDER,
        "interior_band_y_over_H": list(INTERIOR),
        "method": ("per-rung analytic reference from tensorsph.womersley; common "
                   "grid + 5-point Lagrange interpolation for the norm functionals; "
                   "GCI finest pair with Fs=1.25, selected band = max(observed-order, "
                   "conservative p=1); ASME V&V 20-2009 E vs U_V"),
        "common_grid_points": GRID_N if common_grid else None,
        "interpolation": "5-point Lagrange on the rung rows" if common_grid else "none (own rows)",
        "reference_consistency": ref,
        "input_uncertainty": u_in,
        "quantities": quantities_out,
        "single_rung": single,
        "order_status": {k: (v.get("order_analysis") or {}).get("status")
                         for k, v in quantities_out.items()},
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "honest_limits": [
            "three rungs with unequal ratios (H/dx = 8,12,16 -> r = 1.5 and 1.333) "
            "leave the observed order only loosely identified: a monotone ladder can "
            "still show pair orders 1.9 and 4.2 because each pair samples a different "
            "interval of a staircase error, so no order is quoted when they disagree",
            "the GCI band is a bound on the graded *functional*, not a pointwise field "
            "bound; the pointwise profile error is larger near the walls",
            "U_input is the between-window repeatability of one run, not a run-to-run "
            "ensemble: it captures the sampling window, not reproducibility across "
            "hardware or thread counts",
            "the phase functional is oscillatory (the error changes sign between "
            "H/dx=12 and 16), so its band uses the conservative p=1 convention and no "
            "order of accuracy is claimed for the phase",
        ],
    }


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_doc(payload: dict) -> Path:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    A = L.append
    A("# CASE-POISEUILLE-SPH — discretization uncertainty and validation")
    A("")
    A(f"Generated by `{payload['generated_by']}` from "
      f"`{payload['source_artifact']}`; every number below is read from "
      f"`out/uncertainty.json` (no hand copying, no solver run).")
    A("")
    A("## 1. Why this note exists")
    A("")
    A("Gates G1–G9 grade the *point* error at the finest rung against the analytic "
      "reference (tolerance 5 %). That answers \"is the number close\". It does not "
      "answer \"is the residual discretization error, or a floor\", nor \"is the "
      "uncertainty band narrow enough to certify 5 %\". Those are the two questions "
      "the standard verification/validation machinery answers (Roache 1994 GCI; "
      "ASME V&V 20-2009).")
    A("")
    A("## 2. Method")
    A("")
    A("* Errors per rung against the **per-rung** analytic reference — the row "
      "centres move with the resolution, so a single reference value would invent "
      "an order the data does not contain.")
    A(f"* Norm functionals are taken on a common y grid "
      f"({payload['common_grid_points']} points over "
      f"{payload['interior_band_y_over_H'][0]} ≤ y/H ≤ "
      f"{payload['interior_band_y_over_H'][1]}); taking the norm over each rung's "
      f"own rows makes the functional resolution-dependent.")
    A(f"* GCI of the finest rung with Fs = {payload['safety_factor']:g}; the "
      f"selected band is the larger of the observed-order and the conservative "
      f"p = {payload['conservative_order']:g} estimate.")
    A("* Validation: `E = |S-T|/scale`, `U_V = hypot(U_num, U_input)`; "
      "`validated` iff `E ≤ U_V`; `U_input` = between-window repeatability of the "
      "finest run.")
    A("* Scale: amplitude functionals are relative (scale = |T|); phase "
      "functionals are in quarter turns (scale = π/2), so \"5 %\" means 5 % of a "
      "quarter turn, exactly as declared.")

    A("")
    A("## 3. Ladder")
    A("")
    A("| functional | H/dx | value | reference | error % |")
    A("|---|---|---|---|---|")
    for key, v in payload["quantities"].items():
        for lab, val, t, e in zip(v["labels"], v["values"], v["truth_per_rung"],
                                  v["error_percent_per_rung"]):
            A(f"| `{key}` | {lab} | {val:.6g} | {t:.6g} | {e:.4f} |")
    A("")
    A("## 4. Uncertainty and validation")
    A("")
    A("| functional | order status | pair orders (r:p) | p used | U_num % | "
      "U_input % | E % | U_V % | E/U_V | ≤5 %? | band ≤5 %? | verdict |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, v in payload["quantities"].items():
        oa = v.get("order_analysis") or {}
        pairs = ", ".join(f"{p['r']:.3g}:{p['p']:.2f}"
                          for p in oa.get("pair_orders", []) if p.get("p") is not None
                          ) or "none"
        val = v.get("validation") or {}
        A(f"| `{key}` | {oa.get('status')} | {pairs} | {v['p_used']:.2f} | "
          f"{v.get('U_num_percent', float('nan')):.3f} | "
          f"{val.get('U_input_percent', float('nan')):.3f} | "
          f"{val.get('E_percent', float('nan')):.4f} | "
          f"{val.get('U_V_percent', float('nan')):.3f} | "
          f"{(val.get('E_over_U_V') or float('nan')):.2f} | "
          f"{val.get('within_tolerance')} | {val.get('supports_tolerance')} | "
          f"{val.get('verdict')} |")
    A("")
    A("### Status narrative (honest)")
    A("")
    for key, v in payload["quantities"].items():
        oa = v.get("order_analysis") or {}
        ratio = v["error_over_floor"]
        A(f"* **`{key}`** — {oa.get('status')}: {oa.get('reason')}; finest-rung "
          f"error {v['error_percent_per_rung'][0]:.4f} % vs repeatability floor "
          f"{v['repeatability_floor_percent']:.4f} % (ratio "
          f"{ratio:.2f})." if ratio is not None else
          f"* **`{key}`** — {oa.get('status')}: {oa.get('reason')}.")
    A("")
    A("Reading: an error several times the repeatability floor is real "
      "discretization error and a finer rung would reduce it; an error at the "
      "floor is measurement scatter and refinement cannot fix it.")
    A("")
    A("## 5. Gates")
    A("")
    for k, ok in payload["gates"].items():
        A(f"* `{k}`: {'PASS' if ok else 'FAIL'}")
    A("")
    A("`U1` = the acceptance holds at *every* rung, not only the finest one chosen "
      "for grading; `U2` = the residual is inside the uncertainty band; `U3` = the "
      "band itself is narrower than the 5 % tolerance, so the case can certify "
      "5 % rather than merely happen to hit it.")
    A("")
    A("## 6. R1 anchors (single rung)")
    A("")
    s = payload["single_rung"]
    A(f"* steady profile RMS: {s['steady_profile_rms']['error_percent']:.4f} % "
      f"(≤ 5 %: {s['steady_profile_rms']['within_tolerance']})")
    A(f"* start-up profile RMS at t = 0.40 s: "
      f"{s['startup_profile_rms_at_0.40s']['error_percent']:.4f} % "
      f"(≤ 5 %: {s['startup_profile_rms_at_0.40s']['within_tolerance']})")
    A(f"* status: **{s['status']}** — {s['reason']}")
    A("")
    A("## 7. Limits")
    A("")
    for lim in payload["honest_limits"]:
        A(f"* {lim}")
    A("")
    A("## 8. Figure")
    A("")
    A("![uncertainty](../docs/figures/case_poiseuille_uncertainty.png)")
    A("")
    DOC.write_text("\n".join(L) + "\n")
    return DOC


def write_figure(payload: dict) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[figure] skipped: {exc}", file=sys.stderr)
        return None

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    panels = ((axes[0], ["amp_centre", "amp_profile_l2"], "amplitude functionals"),
              (axes[1], ["phase_centre", "phase_profile_l2"], "phase functionals"))
    for ax, keys, title in panels:
        for key in keys:
            v = payload["quantities"][key]
            err = [max(e, 1e-8) for e in v["error_percent_per_rung"]]
            ax.loglog(v["spacings"], err, "o-", label=f"{key} error")
            floor = v["repeatability_floor_percent"]
            if floor > 0:
                ax.axhline(floor, ls=":", lw=1.0, label=f"{key} repeatability")
            band = (v.get("validation") or {}).get("U_V_percent")
            if band:
                ax.axhline(band, ls="-.", lw=0.8, alpha=0.7,
                           label=f"{key} U_V")
        ax.axhline(payload["tolerance_percent"], color="k", ls="--", lw=1.0,
                   label="acceptance 5 %")
        ax.set_xlabel(r"$\Delta x = H/(H/\Delta x)$  [m]")
        ax.set_ylabel("error / band  [% of declared scale]")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("CASE-POISEUILLE-SPH: error ladders, repeatability floor, "
                 "uncertainty band, acceptance")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-doc", action="store_true",
                    help="skip doc/figure generation")
    ap.add_argument("--rowwise-norm", action="store_true",
                    help="use each rung's own rows for the norm (the pitfall variant)")
    args = ap.parse_args(argv)

    result = json.loads(RESULT.read_text())
    payload = analyse(result, common_grid=not args.rowwise_norm)
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"{'functional':18s} {'status':18s} {'E%':>8s} {'U_num%':>8s} "
          f"{'U_V%':>8s} {'E/U_V':>6s} {'≤5%':>5s} {'band≤5%':>7s} verdict")
    for key, v in payload["quantities"].items():
        val = v.get("validation") or {}
        print(f"{key:18s} {str((v.get('order_analysis') or {}).get('status')):18s} "
              f"{val.get('E_percent', float('nan')):8.4f} "
              f"{val.get('U_num_percent', float('nan')):8.3f} "
              f"{val.get('U_V_percent', float('nan')):8.3f} "
              f"{(val.get('E_over_U_V') or float('nan')):6.2f} "
              f"{str(val.get('within_tolerance')):>5s} "
              f"{str(val.get('supports_tolerance')):>7s} {val.get('verdict')}")
    print("gates:", payload["gates"])
    print("all_gates_pass:", payload["all_gates_pass"])

    if not args.no_doc:
        write_doc(payload)
        print("doc:", DOC.relative_to(REPO), "figure:", FIG.relative_to(REPO))

    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())