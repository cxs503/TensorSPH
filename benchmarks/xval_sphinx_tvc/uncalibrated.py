"""Grade the ported TVC with the coefficients SPHinXsys *ships*, not a fitted one.

Why this script exists
----------------------
Rounds 1-2 of ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION`` (§``README.md``,
``RESOLUTION_LADDER.md``) grade the ported operator with ``alpha = 0.10``.
That value was **calibrated against the same Martin & Moyce curve it is then
scored against** (round 1 swept ``alpha`` and kept the best).  Read
adversarially, the headline "10.6 % at ``dx = 0.03``" is a *fit*, not a
prediction: an operator whose only tuned constant is fitted to the target
cannot fail the target for a trivial reason.

The upstream sources in ``docs/xval_sphinx_tvc/oss_source/`` ship two
coefficients and neither is 0.10:

* ``coefficient = 0.2`` -- ``transport_velocity_correction.h`` constructor
  default, stated for the **dual-criteria** time stepping of Zhang, Rezavand &
  Hu (2020);
* ``coefficient = 0.2 / 4 = 0.05`` -- the same header, "if single (acoustic)
  time step is used, the coefficient should be decreased to about 1/4 of the
  default".  TensorSPH's ``dam_break.py`` *is* a single acoustic step, so
  ``0.05`` is the coefficient a user of the port gets without touching
  anything.

This script therefore re-grades the scheme with ``alpha = 0.05`` and
``alpha = 0.20`` over the full resolution ladder, with **no calibration step**
(the variant table of ``run.py`` fixes them before any run), and reports:

1. the accuracy of each uncalibrated variant against the owner 15 % standard
   and the ledger ``MAE < 0.15`` gate;
2. a **digitisation-robustness band**: the reference ``Z/a`` carries
   ``+-0.05`` uncertainty, so every error is recomputed with the reference
   shifted by ``-band`` and ``+band``.  A gate verdict that flips inside that
   band is not a verdict and is labelled as such;
3. the comparison against the calibrated ``alpha = 0.10`` and the baseline, so
   the value added by the operator and the value added by the *fit* are
   separated;
4. for the finest level, the observed order of the uncalibrated variants from
   the Cauchy differences of ``Z/a(tau)`` (no experimental input at all).

Everything is post-processing of payloads written by
``benchmarks/xval_sphinx_tvc/run.py``; no number in the generated report is
typed by hand.

Usage
-----
::

    python benchmarks/xval_sphinx_tvc/uncalibrated.py \
        --inputs docs/xval_sphinx_tvc/data/ladder_dense.json \
                 docs/xval_sphinx_tvc/data/ladder_upstream_defaults.json \
                 docs/xval_sphinx_tvc/data/level_dx0005.json \
        --out docs/xval_sphinx_tvc/data/uncalibrated_defaults.json \
        --doc docs/xval_sphinx_tvc/UNCALIBRATED_DEFAULTS.md
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

DEFAULT_EXP = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952.json"
DEFAULT_FIG = ROOT / "docs" / "xval_sphinx_tvc" / "figure_uncalibrated_defaults.png"

#: Owner accuracy standard and ledger absolute gate (identical numbers, kept
#: separate so a change to one is visible).
OWNER_GATE_REL = 0.15
LEDGER_GATE_ABS = 0.15

#: Digitisation uncertainty of the reference ``Z/a`` (from
#: ``martin_moyce_1952.json`` -> ``digitization.uncertainty``).
DEFAULT_BAND = 0.05


def upstream_default_alphas() -> dict[str, float]:
    """The two coefficients upstream ships, read from the port module."""
    from tensorsph.transport_velocity_correction import (
        SPHINXSYS_DEFAULT_COEFFICIENT,
        SPHINXSYS_SINGLE_STEP_FRACTION,
    )

    return {
        "dual_criteria": float(SPHINXSYS_DEFAULT_COEFFICIENT),
        "single_acoustic_step": float(
            SPHINXSYS_DEFAULT_COEFFICIENT * SPHINXSYS_SINGLE_STEP_FRACTION),
    }


# --------------------------------------------------------------------------- #
# payload handling
# --------------------------------------------------------------------------- #
def load_rows(paths: Iterable[Path]) -> dict[tuple[str, float], dict]:
    """Merge payload files into ``{(variant, dx): record}`` (later files win)."""
    rows: dict[tuple[str, float], dict] = {}
    missing = []
    for path in map(Path, paths):
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text())
        for rec in payload.get("records", []):
            rows[(rec["variant"], float(rec["dx"]))] = rec
    if missing:
        raise FileNotFoundError(
            "payload file(s) not written yet: " + ", ".join(missing))
    return rows


def interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear interpolation clamped at both ends (run.py convention)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for k in range(1, len(xs)):
        if xs[k] >= x:
            w = (x - xs[k - 1]) / (xs[k] - xs[k - 1])
            return ys[k - 1] + w * (ys[k] - ys[k - 1])
    return ys[-1]


# --------------------------------------------------------------------------- #
# scoring with the digitisation band
# --------------------------------------------------------------------------- #
def error_at(taus: Sequence[float], za: Sequence[float], exp: dict,
             band: float = 0.0) -> dict:
    """Errors of a front history against the reference, reference shifted by ``band``.

    ``band > 0`` moves every reference point *up*, ``band < 0`` *down*; the
    relative error is ``|got - ref| / ref``, so both directions change it and
    the two signed evaluations bracket the plausible truth.
    """
    data = exp["data"]
    rows = []
    for d in data:
        tau_ref = d["tau"]
        if tau_ref < taus[0] or tau_ref > taus[-1]:
            continue
        got = interp(taus, za, tau_ref)
        ref = d["Z_over_a"] + band
        abs_e = abs(got - ref)
        # ``ref == 0`` (the ``tau = 0`` anchor) has no relative error; it still
        # counts in the absolute statistics, exactly as ``run.py:score`` does.
        rows.append({"tau": tau_ref, "ref": d["Z_over_a"],
                     "ref_shifted": ref, "got": got,
                     "abs_error": abs_e,
                     "rel_error": abs_e / ref if ref > 0.0 else None})
    if not rows:
        return {"points": 0}
    rel = [r["rel_error"] for r in rows if r["rel_error"] is not None]
    return {
        "points": len(rows),
        "mae": sum(r["abs_error"] for r in rows) / len(rows),
        "mean_rel_error": (sum(rel) / len(rel)) if rel else None,
        "max_abs_error": max(r["abs_error"] for r in rows),
        "max_rel_error": max(rel) if rel else None,
        "rows": rows,
    }


def band_bracket(taus: Sequence[float], za: Sequence[float], exp: dict,
                 band: float = DEFAULT_BAND, n: int = 21) -> dict:
    """Sensitivity of the mean error to a uniform bias in the digitised curve.

    The reference ``Z/a(tau)`` is reported digitised to ``+-band``.  The
    physically meaningful reading of that number is a **bias of the whole
    curve**, not independent noise on each point, so the mean error is swept
    over a uniform offset ``delta in [-band, +band]`` and the interval is
    reported: "if the digitised curve is systematically high/low by up to
    ``band``, the mean error lies here".  ``delta = 0`` is part of the sweep, so
    the nominal value is inside the interval by construction.

    The ``tau = 0`` anchor (``Z/a = 0``) has no relative error and is excluded
    from both statistics, which is the same convention as ``run.py:score``.
    """
    data = exp["data"]
    samples = []
    for d in data:
        tau_ref = d["tau"]
        if tau_ref < taus[0] or tau_ref > taus[-1]:
            continue
        samples.append((d["Z_over_a"], interp(taus, za, tau_ref)))
    if not samples:
        return {"abs_band": [None, None], "rel_band": [None, None],
                "worst_delta_rel": None, "samples": 0}
    rel_curve, mae_curve = [], []
    deltas = [-band + 2.0 * band * k / (n - 1) for k in range(n)]
    for delta in deltas:
        rel = [(abs(got - (ref + delta)) / (ref + delta))
               for ref, got in samples if ref > 0.0 and ref + delta > 0.0]
        mae = [abs(got - (ref + delta)) for ref, got in samples]
        rel_curve.append(sum(rel) / len(rel) if rel else float("nan"))
        mae_curve.append(sum(mae) / len(mae))
    finite = [(d, r) for d, r in zip(deltas, rel_curve) if math.isfinite(r)]
    return {
        "abs_band": [min(mae_curve), max(mae_curve)],
        "rel_band": [min(r for _, r in finite), max(r for _, r in finite)],
        "worst_delta_rel": (min(finite, key=lambda p: p[1])[0]
                            if finite else None),
        "samples": len(samples),
    }


def grade(taus: Sequence[float], za: Sequence[float], exp: dict,
          band: float = DEFAULT_BAND) -> dict:
    """Nominal errors, the ``+-band`` bracket and the resulting verdict."""
    nominal = error_at(taus, za, exp, 0.0)
    if nominal.get("points", 0) == 0:
        return {"points": 0, "verdict": "no_overlap"}
    rows = nominal["rows"]
    bracket = band_bracket(taus, za, exp, band=band)
    within = [r for r in rows if r["abs_error"] <= band]
    late = [r for r in rows if r["ref"] >= 0.5 and r["rel_error"] is not None]
    early = [r for r in rows if r["ref"] < 0.5]
    rel_band = bracket["rel_band"]
    mae_band = bracket["abs_band"]
    rel_pass = (nominal["mean_rel_error"] is not None
                and nominal["mean_rel_error"] < OWNER_GATE_REL)
    mae_pass = nominal["mae"] < LEDGER_GATE_ABS
    robust_pass = (rel_band[1] is not None and rel_band[1] < OWNER_GATE_REL
                   and mae_band[1] < LEDGER_GATE_ABS)
    robust_fail = (rel_band[0] is None or rel_band[0] >= OWNER_GATE_REL
                   or mae_band[0] >= LEDGER_GATE_ABS)
    if rel_pass and mae_pass and robust_pass:
        verdict = "PASS"
    elif not rel_pass or not mae_pass:
        verdict = "FAIL" if robust_fail else "FAIL_marginal"
    else:
        verdict = "PASS_marginal"
    return {
        "points": nominal["points"],
        "mae": nominal["mae"],
        "mean_rel_error": nominal["mean_rel_error"],
        "max_abs_error": nominal["max_abs_error"],
        "max_rel_error": nominal["max_rel_error"],
        # ``Z/a -> 0`` at early ``tau`` makes the relative error a small
        # denominator problem; the late-window mean is reported next to it so
        # the reader can see how much of the error is that effect.  The gate
        # itself stays the full-sample mean (the predeclared one, identical to
        # ``run.py:score``).
        "mean_rel_error_late": (sum(r["rel_error"] for r in late) / len(late)
                                if late else None),
        "samples_late": len(late),
        "samples_early": len(early),
        "within_band_fraction": len(within) / len(rows),
        "mae_band": mae_band,
        "rel_band": rel_band,
        "band_worst_delta": bracket["worst_delta_rel"],
        "gate_rel_pass": bool(rel_pass),
        "gate_abs_pass": bool(mae_pass),
        "robust_pass": bool(robust_pass),
        "robust_fail": bool(robust_fail),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Cauchy / observed order (no experimental input)
# --------------------------------------------------------------------------- #
def cauchy_pairs(levels: dict[float, dict], n_grid: int = 401) -> list[dict]:
    """Successive-level ``L2`` differences of ``Z/a(tau)`` on a common grid."""
    ordered = sorted(levels, reverse=True)
    if len(ordered) < 2:
        return []
    lo = max(levels[dx]["tau"][0] for dx in ordered)
    hi = min(levels[dx]["tau"][-1] for dx in ordered)
    grid = [lo + (hi - lo) * k / (n_grid - 1) for k in range(n_grid)]
    curves = {dx: [interp(levels[dx]["tau"], levels[dx]["Z_over_a"], t)
                   for t in grid] for dx in ordered}
    out = []
    for coarse, fine in zip(ordered[:-1], ordered[1:]):
        diff = [a - b for a, b in zip(curves[coarse], curves[fine])]
        l2 = math.sqrt(sum(d * d for d in diff) / len(diff))
        out.append({"dx_coarse": coarse, "dx_fine": fine,
                    "r": coarse / fine, "cauchy_l2": l2,
                    "order": None})
    for a, b in zip(out[:-1], out[1:]):
        if a["cauchy_l2"] > 0 and b["cauchy_l2"] > 0:
            b["order"] = (math.log(a["cauchy_l2"] / b["cauchy_l2"])
                          / math.log(math.sqrt(a["r"] * b["r"])))
    return out


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f} %"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def build_report(rows: dict[tuple[str, float], dict], exp: dict,
                 band: float = DEFAULT_BAND,
                 calibration_alpha: float = 0.10,
                 pending: str | None = None) -> dict:
    """Grade every ``(variant, dx)`` and answer the uncalibrated question."""
    alphas = upstream_default_alphas()
    variants = sorted({v for v, _ in rows})
    levels = sorted({dx for _, dx in rows})
    table = []
    for variant in variants:
        for dx in levels:
            rec = rows.get((variant, dx))
            if rec is None:
                continue
            g = grade(rec["tau"], rec["Z_over_a"], exp, band=band)
            alpha = rec.get("params", {}).get("tvc_alpha")
            table.append({
                "variant": variant,
                "dx": dx,
                "alpha": alpha,
                "alpha_source": (
                    "none (no correction)" if alpha is None else
                    "upstream dual-criteria default"
                    if alpha == alphas["dual_criteria"] else
                    "upstream single-step default"
                    if alpha == alphas["single_acoustic_step"] else
                    f"calibrated ({alpha})"),
                "wall_clock_s": rec.get("wall_clock_s"),
                "steps": rec.get("steps"),
                "max_tvc_shift_over_h": rec.get("max_tvc_shift_over_h"),
                "max_density_compression": rec.get("max_density_compression"),
                "mass_drift": rec.get("mass_drift"),
                "clumping": rec.get("clumping"),
                "grade": {k: v for k, v in g.items() if k != "rows"},
            })

    def find(variant: str, dx: float) -> dict | None:
        for row in table:
            if row["variant"] == variant and row["dx"] == dx:
                return row
        return None

    default_variants = {
        name: alpha for name, alpha in (
            ("tvc_a005_all", alphas["single_acoustic_step"]),
            ("tvc_a020_all", alphas["dual_criteria"]))
    }
    best_default = None
    for variant in default_variants:
        for dx in levels:
            row = find(variant, dx)
            if row is None:
                continue
            rel = row["grade"].get("mean_rel_error")
            if rel is None:
                continue
            if best_default is None or rel < best_default["mean_rel_error"]:
                best_default = {"variant": variant, "dx": dx,
                                "mean_rel_error": rel,
                                "mae": row["grade"]["mae"],
                                "verdict": row["grade"]["verdict"]}
    passes = [(row["variant"], row["dx"])
              for row in table
              if row["variant"] in default_variants
              and row["grade"].get("gate_rel_pass")
              and row["grade"].get("gate_abs_pass")]
    robust_passes = [(row["variant"], row["dx"])
                     for row in table
                     if row["variant"] in default_variants
                     and row["grade"].get("robust_pass")]
    all_robust = [(row["variant"], row["dx"], row["alpha"])
                  for row in table
                  if row["grade"].get("robust_pass")]
    all_nominal = [(row["variant"], row["dx"], row["alpha"])
                   for row in table
                   if row["grade"].get("gate_rel_pass")
                   and row["grade"].get("gate_abs_pass")]

    calib = {dx: find(f"tvc_a{int(round(calibration_alpha * 100)):03d}_all", dx)
             for dx in levels}
    base = {dx: find("baseline", dx) for dx in levels}

    # Cauchy order per variant over the levels it has
    cauchy = {}
    for variant in variants:
        have = {dx: rows[(variant, dx)] for dx in levels
                if (variant, dx) in rows}
        cauchy[variant] = cauchy_pairs(have)

    return {
        "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
        "question": "does the ported TVC meet the owner 15 % standard with the "
                    "coefficients SPHinXsys ships, without calibrating alpha?",
        "upstream_defaults": alphas,
        "calibration_alpha": calibration_alpha,
        "gates": {"owner_rel": OWNER_GATE_REL, "ledger_abs": LEDGER_GATE_ABS,
                  "reference_band": band},
        "reference": {
            "name": exp.get("source", {}).get("paper"),
            "digitization_uncertainty": exp.get("digitization", {}).get("uncertainty"),
        },
        "environment": {"python": platform.python_version(),
                        "machine": platform.machine()},
        "levels": levels,
        "pending": pending,
        "table": table,
        "summary": {
            "best_upstream_default": best_default,
            "uncalibrated_plain_passes": [list(p) for p in passes],
            "uncalibrated_robust_passes": [list(p) for p in robust_passes],
            "all_nominal_passes": [list(p) for p in all_nominal],
            "all_robust_passes": [list(p) for p in all_robust],
            "calibrated_by_dx": {str(dx): (
                None if calib[dx] is None else
                {"mean_rel_error": calib[dx]["grade"]["mean_rel_error"],
                 "mae": calib[dx]["grade"]["mae"],
                 "verdict": calib[dx]["grade"]["verdict"]})
                for dx in levels},
            "baseline_by_dx": {str(dx): (
                None if base[dx] is None else
                {"mean_rel_error": base[dx]["grade"]["mean_rel_error"],
                 "mae": base[dx]["grade"]["mae"],
                 "verdict": base[dx]["grade"]["verdict"]})
                for dx in levels},
        },
        "cauchy": cauchy,
    }


def write_figure(report: dict, path: Path) -> None:
    """``mean relative error vs dx`` for every variant, with the owner gate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover - display
        print(f"[warn] matplotlib unavailable ({exc}); figure skipped",
              file=sys.stderr)
        return
    order = ["baseline", "tvc_a005_all", "tvc_a010_all", "tvc_a020_all"]
    style = {
        "baseline": dict(color="0.45", marker="o", ls="--"),
        "tvc_a005_all": dict(color="C2", marker="s", ls="-"),
        "tvc_a010_all": dict(color="C0", marker="^", ls="-"),
        "tvc_a020_all": dict(color="C3", marker="D", ls="-"),
    }
    label = {
        "baseline": "no correction",
        "tvc_a005_all": r"TVC $\alpha=0.05$ (upstream single-step default)",
        "tvc_a010_all": r"TVC $\alpha=0.10$ (calibrated — round 1/2)",
        "tvc_a020_all": r"TVC $\alpha=0.20$ (upstream dual-criteria default)",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    for variant in order:
        pts = [(row["dx"], row["grade"]["mean_rel_error"])
               for row in report["table"]
               if row["variant"] == variant
               and row["grade"].get("mean_rel_error") is not None]
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [100.0 * p[1] for p in pts]
        ax.plot(xs, ys, lw=1.6, ms=5.5, label=label[variant],
                **style.get(variant, {}))
    ax.axhline(100.0 * report["gates"]["owner_rel"], color="k", lw=1.2, ls=":")
    ax.text(min(dx for dx, _ in [(r["dx"], 0) for r in report["table"]]),
            100.0 * report["gates"]["owner_rel"] + 1.0,
            "owner standard 15 %", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"resolution $\Delta x$ [m]")
    ax.set_ylabel(r"mean relative error of $Z/a(\tau)$ [%]")
    ax.set_title("TVC graded with the upstream default coefficients\n"
                 "(no calibration against the reference)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_doc(report: dict, path: Path, fig_path: Path,
              pending: str | None = None) -> None:
    """Generate the markdown report — every number comes from ``report``."""
    t = {(row["variant"], row["dx"]): row for row in report["table"]}
    levels = report["levels"]
    pending = pending or report.get("pending")
    gate_rel = 100.0 * report["gates"]["owner_rel"]
    gate_abs = report["gates"]["ledger_abs"]
    band = report["gates"]["reference_band"]

    def cell(variant: str, dx: float) -> tuple[str, str]:
        row = t.get((variant, dx))
        if row is None:
            return "—", "—"
        g = row["grade"]
        return (f"{_num(g['mae'])} / {_pct(g['mean_rel_error'])}",
                g["verdict"])

    lines = [
        "# Grading the ported TVC with the coefficients SPHinXsys actually ships",
        "",
        "Anchor: `XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION`.",
        "Generated by `benchmarks/xval_sphinx_tvc/uncalibrated.py` from the raw",
        "payloads under `docs/xval_sphinx_tvc/data/` — no number below is typed",
        "by hand.",
        "",
        "## 1. The gap this closes",
        "",
        "Rounds 1-2 (`README.md`, `RESOLUTION_LADDER.md`) grade the ported",
        f"operator with `alpha = {report['calibration_alpha']}`, a value picked by",
        "sweeping `alpha` **against the same Martin & Moyce curve it is then",
        "scored against**. The honest reading is that 10-12 % is a *fit*: a tuned",
        "constant always converges on its own target. Upstream ships two",
        "coefficients and neither is that value:",
        "",
        "| upstream coefficient | value | where it comes from |",
        "|---|---|---|",
        "| dual-criteria default | "
        f"{report['upstream_defaults']['dual_criteria']} | "
        "`transport_velocity_correction.h`, `coefficient = 0.2` |",
        "| single acoustic step | "
        f"{report['upstream_defaults']['single_acoustic_step']} | "
        "same header: \"decreased to about 1/4\" for a single step |",
        "",
        "TensorSPH's `dam_break.py` is a single acoustic step, so",
        f"`alpha = {report['upstream_defaults']['single_acoustic_step']}` is what a",
        "user of the port gets by default and `alpha =",
        f"{report['upstream_defaults']['dual_criteria']}` is the dual-criteria value.",
        "Both were **fixed in the variant table before any run of this round**",
        "(`VARIANTS` in `run.py`); nothing here is re-tuned.",
        "",
        "## 2. Method",
        "",
        "| item | value |",
        "|---|---|",
        "| case | Martin & Moyce (1952) dam break, `a = 0.2 m`, `2a = 0.4 m`, dry bed |",
        "| solver | in-repo WCSPH `src/tensorsph/dam_break.py`, single acoustic step |",
        "| correction | `src/tensorsph/transport_velocity_correction.py` (ported) |",
        "| resolutions | " + ", ".join(f"`dx = {dx}` m" for dx in levels) + " |",
        "| variants | `baseline`, `tvc_a005_all`, `tvc_a010_all` (calibrated control), `tvc_a020_all` |",
        "| metric | mean relative error of `Z/a(tau)` at the experimental `tau` samples |",
        f"| gate | owner `{gate_rel:.0f} %` relative **and** ledger `MAE < {gate_abs}` |",
        f"| reference band | `+-{band}` in `Z/a` (digitisation uncertainty), propagated |",
        "",
        "Each error is reported three times: at the nominal reference, and with the",
        "reference shifted by `-band` and `+band`. A verdict that flips inside that",
        "band is labelled `*_marginal` instead of `PASS`/`FAIL`, so a `PASS` here",
        "means the gate holds for the whole digitisation band.",
        "",
        "## 3. Results",
        "",
        "`MAE / mean rel.` and the gate verdict:",
        "",
        "| `dx` [m] | baseline | TVC a=0.05 (default) | TVC a=0.10 (calibrated) | TVC a=0.20 (default) |",
        "|---|---|---|---|---|",
    ]
    for dx in levels:
        cells = [cell(v, dx) for v in
                 ("baseline", "tvc_a005_all", "tvc_a010_all", "tvc_a020_all")]
        lines.append(
            f"| {dx} | {cells[0][0]} ({cells[0][1]}) | {cells[1][0]} "
            f"({cells[1][1]}) | {cells[2][0]} ({cells[2][1]}) | "
            f"{cells[3][0]} ({cells[3][1]}) |")
    lines += [
        "",
        f"![mean relative error vs dx]({fig_path.name})",
        "",
        "### 3a. How much of the error is the small-denominator effect",
        "",
        "`Z/a -> 0` at early `tau` inflates a *relative* error without any",
        "physical content. The late window (`Z/a >= 0.5`) is reported beside the",
        "predeclared full-sample mean, and the band column sweeps a uniform",
        f"offset `delta in [-{band}, +{band}]` over the whole digitised curve:",
        "the interval is where the mean error lands if the reference is",
        "systematically high or low by up to its stated uncertainty",
        "(`delta = 0` is in the sweep, so the nominal value is inside it).",
        "",
        "| `dx` [m] | variant | full-sample rel. | late-window rel. | samples with abs err <= band | rel. bracket ±band |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["table"]:
        g = row["grade"]
        band_txt = ("n/a" if g.get("rel_band") is None
                    or g["rel_band"][0] is None else
                    f"{_pct(g['rel_band'][0])} … {_pct(g['rel_band'][1])}")
        lines.append(
            f"| {row['dx']} | `{row['variant']}` | {_pct(g.get('mean_rel_error'))} | "
            f"{_pct(g.get('mean_rel_error_late'))} "
            f"({g.get('samples_late')}/{g.get('points')} samples) | "
            f"{_pct(g.get('within_band_fraction'))} | {band_txt} |")
    lines += [
        "",
        "Per-run integrity diagnostics (measured, from the payloads):",
        "",
        "| `dx` | variant | `alpha` | steps | wall [s] | max shift/h | max compression | mass drift |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["table"]:
        lines.append(
            f"| {row['dx']} | `{row['variant']}` | {_num(row['alpha'])} | "
            f"{row['steps']} | {row['wall_clock_s']:.1f} | "
            f"{_num(row['max_tvc_shift_over_h'])} | "
            f"{_pct(row['max_density_compression'])} | {row['mass_drift']:.2e} |")
    summary = report["summary"]
    best = summary["best_upstream_default"]
    if pending:
        lines += [
            "",
            "> **Measured-status of the uncalibrated leg.** " + pending,
        ]
    lines += [
        "",
        "## 4. Verdict",
        "",
        "| question | answer |",
        "|---|---|",
        "| best uncalibrated result | " + (
            "none" if best is None else
            f"`{best['variant']}` at `dx = {best['dx']}`: MAE {_num(best['mae'])}, "
            f"{_pct(best['mean_rel_error'])}, verdict `{best['verdict']}`") + " |",
        "| uncalibrated levels passing both gates (nominal) | " + (
            "none" if not summary["uncalibrated_plain_passes"] else
            ", ".join(f"`{v}` @ {dx}" for v, dx in
                      summary["uncalibrated_plain_passes"])) + " |",
        "| uncalibrated levels robust to the digitisation band | " + (
            "none" if not summary["uncalibrated_robust_passes"] else
            ", ".join(f"`{v}` @ {dx}" for v, dx in
                      summary["uncalibrated_robust_passes"])) + " |",
        "| any variant passing both gates at its nominal value | " + (
            "none" if not summary["all_nominal_passes"] else
            ", ".join(f"`{v}` (alpha {_num(a)}) @ {dx}"
                      for v, dx, a in summary["all_nominal_passes"])) + " |",
        "| any variant whose pass survives a uniform `±"
        + f"{band}" + "` bias of the reference | " + (
            "none" if not summary["all_robust_passes"] else
            ", ".join(f"`{v}` (alpha {_num(a)}) @ {dx}"
                      for v, dx, a in summary["all_robust_passes"])) + " |",
    ]
    for variant, pairs in report["cauchy"].items():
        if not pairs:
            continue
        desc = ", ".join(
            f"{p['dx_coarse']}->{p['dx_fine']}: L2 {p['cauchy_l2']:.3f}"
            + ("" if p["order"] is None else f" (p {p['order']:+.2f})")
            for p in pairs)
        lines.append(f"| self-convergence `{variant}` (no experiment) | {desc} |")
    lines += [
        "",
        "Reading, unsoftened:",
        "",
        f"* The ported operator with the **shipped default** coefficient is worth",
        "  substantially more than nothing at every level, but the accuracy it",
        "  reaches *without calibration* is the honest number for the ledger, and",
        "  it is the one graded above.",
        "* The calibrated `alpha = 0.10` column is kept in the same table on",
        "  purpose: the difference between it and the default column is the size",
        "  of the fit, i.e. the part of rounds 1-2 that does not transfer to a new",
        "  case.",
        "",
        "## 5. Limits of this round",
        "",
        "1. One case (dam break), one metric (front position `Z/a`). The bulk",
        "   diagnostics (`grad-1` residual, clumping) are recorded per run but are",
        "   not turned into a gate here; see `DISTRIBUTION_LADDER.md`.",
        "2. `alpha` is still not resolution independent (round 2 §3): a default",
        "   coefficient graded over four levels inherits that, which is exactly",
        "   why the per-level table, not a single number, is the deliverable.",
        f"3. The reference is digitised with `+-{band}` uncertainty; the band",
        "   bracket is propagated, but a *systematic* digitisation bias in the",
        "   front definition would not be.",
        "4. No live SPHinXsys binary comparison (no Boost/TBB in the image); the",
        "   operator leg is the transcription test in",
        "   `tests/test_transport_velocity_correction.py`.",
        "",
        "## 6. Reproduce",
        "",
        "```",
        "# uncalibrated coefficients over the ladder (~3 min single core)",
        "python benchmarks/xval_sphinx_tvc/run.py --resolutions 0.04 0.03 0.02 0.01 \\",
        "    --variants tvc_a005_all tvc_a020_all --seconds 0.45 \\",
        "    --output-samples 240 --out docs/xval_sphinx_tvc/data/ladder_upstream_defaults.json",
        "",
        "# extra level, to put the 0.02->0.01 pair into a 3-level group",
        "python benchmarks/xval_sphinx_tvc/run.py --resolutions 0.005 \\",
        "    --variants baseline tvc_a010_all tvc_a005_all --seconds 0.45 \\",
        "    --output-samples 240 --out docs/xval_sphinx_tvc/data/level_dx0005.json",
        "",
        "# grading + figure + this document",
        "python benchmarks/xval_sphinx_tvc/uncalibrated.py \\",
        "    --inputs docs/xval_sphinx_tvc/data/ladder_dense.json \\",
        "             docs/xval_sphinx_tvc/data/ladder_upstream_defaults.json \\",
        "             docs/xval_sphinx_tvc/data/level_dx0005.json \\",
        "    --out docs/xval_sphinx_tvc/data/uncalibrated_defaults.json \\",
        "    --doc docs/xval_sphinx_tvc/UNCALIBRATED_DEFAULTS.md",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--experiment", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/xval_sphinx_tvc/data/uncalibrated_defaults.json")
    ap.add_argument("--doc", type=Path,
                    default=ROOT / "docs/xval_sphinx_tvc/UNCALIBRATED_DEFAULTS.md")
    ap.add_argument("--figure", type=Path, default=DEFAULT_FIG)
    ap.add_argument("--band", type=float, default=DEFAULT_BAND)
    ap.add_argument("--pending", type=str, default=None,
                    help="status note written into the report when part of the "
                         "measurement has not run yet (honest-FAIL discipline)")
    args = ap.parse_args(argv)

    exp = json.loads(args.experiment.read_text())
    rows = load_rows(args.inputs)
    report = build_report(rows, exp, band=args.band, pending=args.pending)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    write_figure(report, args.figure)
    write_doc(report, args.doc, args.figure, pending=args.pending)
    for row in report["table"]:
        g = row["grade"]
        print(f"dx={row['dx']:<6} {row['variant']:<14} alpha={_num(row['alpha'])} "
              f"mae={_num(g.get('mae'))} rel={_pct(g.get('mean_rel_error'))} "
              f"{g['verdict']}")
    print(f"wrote {args.out}, {args.doc}, {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())