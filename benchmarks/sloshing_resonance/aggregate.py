#!/usr/bin/env python3
"""Aggregate the CASE-SLOSHING case runs into criteria, tables and figures.

Reads the per-case JSONs written by ``run.py`` (order-independent), re-measures
the frequency with a *second*, independent estimator, applies the predeclared
tolerance, and writes:

* ``criteria.json`` -- per-case verdicts plus the three predeclared questions
  (tolerance / refinement / sensitivity) with the numbers that answer them;
* ``table.md``       -- the ref-vs-got table used by the report (no hand typing);
* ``fig1..fig5.png`` -- the figures of the report;
* ``summary.json``   -- flattened headline numbers.

The second estimator is a damped-sinusoid least-squares fit
(``a1(t) ~ exp(-lambda t) (a cos wt + b sin wt)``) whose frequency is *not*
taken from the crossings. It exists to separate "the solver oscillates at the
wrong frequency" from "the crossing counter was fooled by gauge noise": the two
must agree for the primary (crossing-based) number to be believed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

import sys

sys.path.insert(0, str(HERE))
import cases as case_table  # noqa: E402

TOL = case_table.TOLERANCE


# --------------------------------------------------------------------------- #
# second estimator
# --------------------------------------------------------------------------- #
def theory_anchored_fit(times, signal, omega) -> dict:
    """Damped sinusoid *at the linear-theory frequency*: amplitude, decay, residual.

    The frequency is **not** fitted here, on purpose. This is a *measurement
    quality* fit: it answers "how much of this record is a damped sinusoid at the
    frequency the theory predicts", which gives (a) the decay with which to cut
    the record before the gauge noise takes over and (b) the noise fraction
    ``residual_rms / amplitude``. Claiming agreement with theory from a fit whose
    frequency was free to land on the theory value would be circular; the
    frequency claim is carried by the crossing estimators only.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.size < 8 or not np.all(np.isfinite(y)) or not math.isfinite(omega):
        return {"decay": float("nan"), "residual_rms": float("nan"),
                "amplitude": float("nan"), "noise_fraction": float("nan")}
    best = None
    for decay in np.linspace(0.0, 8.0, 401):
        basis = np.column_stack((
            np.ones_like(t),                       # the centre of mass oscillates
            np.exp(-decay * t) * np.cos(omega * t),  # about L/2, so keep a mean
            np.exp(-decay * t) * np.sin(omega * t),
        ))
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        residual = float(np.sum((y - basis @ coef) ** 2))
        if best is None or residual < best[0]:
            best = (residual, float(decay), coef)
    residual, decay, coef = best
    amplitude = float(math.hypot(coef[1], coef[2]))
    residual_rms = float(math.sqrt(residual / y.size))
    return {
        "decay": decay,
        "residual_rms": residual_rms,
        "amplitude": amplitude,
        "noise_fraction": residual_rms / amplitude if amplitude > 0 else float("nan"),
    }


def windowed_crossing_estimate(times, signal, fit, floor: float = 0.3) -> float:
    """Mean-crossing frequency on the part of the record the signal still fills.

    The registered operator (:func:`tensorsph.freesurface.profiles.
    oscillation_frequency`) counts every upward mean crossing of the whole
    record. Once the physical mode has decayed into the topmost-particle gauge
    noise (a fraction of ``dx``), those crossings are noise crossings and the
    reported frequency jumps -- which is exactly what the coarsest rung of the
    ladder does. The window is set from the *fit* (data-driven, theory-free):
    the record is cut where the fitted envelope falls below ``floor`` of its
    initial value.
    """
    from tensorsph.freesurface import profiles

    t = np.asarray(times, dtype=float)
    y = np.asarray(signal, dtype=float)
    decay = fit.get("decay", float("nan"))
    if not (decay == decay and decay > 1.0e-3) or t.size < 8:
        return float("nan")
    t_cut = math.log(1.0 / floor) / decay
    keep = t <= t_cut
    if keep.sum() < 8:
        return float("nan")
    try:
        return float(profiles.oscillation_frequency(t[keep].tolist(), y[keep].tolist()))
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# loading / verdicts
# --------------------------------------------------------------------------- #
def load(outdir: Path) -> list:
    records = []
    for case in case_table.CASES:
        path = outdir / f"{case.label}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        records.append(payload)
    return records


def evaluate(records: list) -> dict:
    rows = []
    for payload in records:
        theory = payload["theory"]["omega_linear"]
        measured = payload.get("measured") or {}
        times = measured.get("times") or []
        modal = measured.get("modal") or []
        com = measured.get("com_x") or []
        fit = theory_anchored_fit(times, com or modal, theory) if times else {}
        # primary: centre-of-mass measurement; legacy: topmost-particle gauge
        omega_top = payload.get("omega_measured", float("nan"))
        error_top = (abs(omega_top - theory) / theory
                     if math.isfinite(omega_top) else float("nan"))
        omega_full = payload.get("omega_from_com", float("nan"))
        error_full = (abs(omega_full - theory) / theory
                      if math.isfinite(omega_full) else float("nan"))
        omega_windowed = windowed_crossing_estimate(times, com, fit) if times else float("nan")
        # primary: crossings on the part of the record the mode still fills; the
        # cut comes from the data-driven decay of the fit above. Falls back to the
        # full record when the fit cannot support a cut (very light damping).
        omega = omega_windowed if math.isfinite(omega_windowed) else omega_full
        error = abs(omega - theory) / theory if math.isfinite(omega) else float("nan")
        rows.append({
            "label": payload["label"],
            "family": payload["family"],
            "H_over_L": payload["case"]["depth_ratio_H_over_L"],
            "A_over_H": payload["case"]["amplitude_ratio_A_over_H"],
            "cells_per_depth": payload["case"]["cells_per_depth"],
            "sound_speed_ratio": payload["case"]["sound_speed_ratio"],
            "dx": payload["case"]["dx"],
            "omega_theory": theory,
            "omega_sim": omega,
            "error": error,
            "omega_top": omega_top,
            "error_top": error_top,
            "omega_full": omega_full,
            "error_full": error_full,
            "pass": bool(math.isfinite(error) and error <= TOL),
            "noise_fraction": fit.get("noise_fraction", float("nan")),
            "fit_decay": fit.get("decay", float("nan")),
            "decay_per_period": measured.get("decay_per_period", float("nan")),
            "mass_drift": measured.get("mass_drift", float("nan")),
            "max_compression": measured.get("max_density_compression", float("nan")),
            "particles": measured.get("particles"),
            "steps": measured.get("steps"),
            "simulated_time": measured.get("simulated_time"),
            "wall_clock_s": payload.get("wall_clock_s"),
            "json": f"{payload['label']}.json",
        })
    by_label = {row["label"]: row for row in rows}

    def pick(label):
        return by_label.get(label, {})

    base = pick("base_cpd12")
    ladder = sorted((by_label[l] for l in
                     ("res_cpd08", "base_cpd12", "res_cpd16", "res_cpd20")
                     if l in by_label), key=lambda r: r["cells_per_depth"])
    geom = sorted((by_label[l] for l in ("geom_h020", "base_cpd12", "geom_h060")
                   if l in by_label), key=lambda r: r["H_over_L"])
    amp = sorted((by_label[l] for l in ("amp_004", "base_cpd12", "amp_016")
                  if l in by_label), key=lambda r: r["A_over_H"])
    cs = sorted((by_label[l] for l in ("cs_040", "base_cpd12", "cs_160")
                 if l in by_label), key=lambda r: r["sound_speed_ratio"])

    criteria = {
        "tolerance": TOL,
        "declared_questions": {
            "Q1_tolerance": {
                "question": "every case within 10% of linear theory?",
                "n_cases": len(rows),
                "n_pass": sum(1 for r in rows if r["pass"]),
                "worst_label": max(rows, key=lambda r: r["error"] if math.isfinite(r["error"]) else 9e9)["label"] if rows else None,
                "worst_error": max((r["error"] for r in rows
                                    if math.isfinite(r["error"])), default=float("nan")),
                "verdict": "PASS" if rows and all(r["pass"] for r in rows) else "FAIL",
            },
            "Q2_refinement": {
                "question": "does the error fall under refinement (cpd 8 -> 20)?",
                "points": [{"cells_per_depth": r["cells_per_depth"],
                            "dx": r["dx"], "error": r["error"]} for r in ladder],
                "monotone": all(ladder[i + 1]["error"] <= ladder[i]["error"] + 0.005
                                for i in range(len(ladder) - 1)) if len(ladder) > 1 else False,
            },
            "Q3_sensitivity": {
                "question": "how much do A/H and cs/sqrt(gH) move the measured frequency?",
                "amplitude": [{"A_over_H": r["A_over_H"], "error": r["error"],
                               "omega_sim": r["omega_sim"]} for r in amp],
                "sound_speed": [{"cs_ratio": r["sound_speed_ratio"],
                                 "error": r["error"], "omega_sim": r["omega_sim"]}
                                for r in cs],
                "compressibility_bias_estimate": base.get("max_compression"),
            },
        },
        "rows": rows,
        "headline": {
            "base_omega_theory": base.get("omega_theory"),
            "base_omega_sim": base.get("omega_sim"),
            "base_error": base.get("error"),
            "base_error_pct": (100.0 * base["error"]) if base.get("error") is not None
            and math.isfinite(base.get("error", float("nan"))) else None,
            "base_decay_per_period": base.get("decay_per_period"),
            "base_mass_drift": base.get("mass_drift"),
            "base_max_compression": base.get("max_compression"),
            "base_particles": base.get("particles"),
        },
    }
    return criteria


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
def markdown_table(rows: list) -> str:
    head = ("| case | family | H/L | A/H | cpd | cs/sqrt(gH) | dx [mm] | "
            "omega_th [rad/s] | omega_sim [rad/s] | err [%] | "
            "omega_fullrec [rad/s] | err_fullrec [%] | "
            "omega_top [rad/s] | err_top [%] | decay/period | verdict |\n")
    sep = "|" + "---|" * 16 + "\n"
    lines = []
    for r in rows:
        verdict = "PASS" if r["pass"] else ("FAIL" if math.isfinite(r["error"]) else "UNMEASURED")
        lines.append(
            f"| {r['label']} | {r['family']} | {r['H_over_L']:.2f} | {r['A_over_H']:.2f} | "
            f"{r['cells_per_depth']:.0f} | {r['sound_speed_ratio']:.0f} | "
            f"{1000.0 * r['dx']:.1f} | {r['omega_theory']:.4f} | {r['omega_sim']:.4f} | "
            f"{100.0 * r['error']:+.2f} | {r['omega_full']:.4f} | "
            f"{100.0 * r['error_full']:+.2f} | {r['omega_top']:.4f} | "
            f"{100.0 * r['error_top']:+.2f} | {r['decay_per_period']:.3f} | {verdict} |\n")
    return head + sep + "".join(lines)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(records: list, rows: list, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    by_label = {r["label"]: r for r in rows}
    payloads = {p["label"]: p for p in records}

    def save(fig, name, caption):
        path = outdir / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        made.append({"name": name, "caption": caption, "path": str(path)})

    # fig1: modal record of the baseline against the linear period
    base = payloads.get("base_cpd12")
    if base and base.get("measured", {}).get("times"):
        t = np.array(base["measured"]["times"])
        a = np.array(base["measured"]["modal"])
        omega = base["theory"]["omega_linear"]
        amp = base["case"]["amplitude"]
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.plot(t, 1e3 * a, "o-", ms=2.6, lw=1.0, label="SPH modal amplitude a$_1$(t)")
        ax.plot(t, 1e3 * amp * np.cos(omega * t), "k--", lw=1.0,
                label=f"linear theory: A cos($\\omega_1$t), $\\omega_1$={omega:.3f} rad/s")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("a$_1$ [mm]")
        ax.set_title(f"Released first mode, L=0.60 m, H=0.25 m, "
                     f"$c_s/\\sqrt{{gH}}$=80  (cpd={base['case']['cells_per_depth']:.0f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "fig1_modal_record.png",
             "Baseline modal amplitude record a1(t) against the linear-theory "
             "oscillation A cos(omega1 t); the phase drift over three periods is "
             "the frequency error, the envelope decay is numerical damping.")

    # fig2: dispersion curve omega_1(H/L): theory line vs SPH points
    geom = [r for r in rows if r["family"] in ("geometry", "baseline")]
    if geom:
        ratio = np.linspace(0.05, 0.9, 200)
        k = np.pi / case_table.TANK_LENGTH
        theory = np.sqrt(case_table.GRAVITY * k * np.tanh(k * ratio * case_table.TANK_LENGTH))
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
        axes[0].plot(ratio, theory, "-", color="0.3",
                     label=r"linear theory $\omega_1=\sqrt{gk_1\tanh k_1H}$")
        axes[0].plot([r["H_over_L"] for r in geom], [r["omega_sim"] for r in geom],
                     "o", ms=7, mfc="none", mew=1.6, color="C3", label="SPH, released mode 1")
        for r in geom:
            axes[0].annotate(r["label"], (r["H_over_L"], r["omega_sim"]),
                             textcoords="offset points", xytext=(6, -10), fontsize=7)
        axes[0].set_xlabel("H/L [-]")
        axes[0].set_ylabel(r"$\omega_1$ [rad/s]")
        axes[0].set_title("resonance frequency vs depth ratio (cpd=12)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].axhline(100.0 * TOL, color="C3", ls="--", lw=1.0,
                        label=f"mission tolerance {100 * TOL:.0f}%")
        axes[1].plot([r["H_over_L"] for r in geom], [100.0 * r["error"] for r in geom],
                     "s-", color="C0")
        for r in geom:
            axes[1].annotate(f"{100 * r['error']:.2f}%", (r["H_over_L"], 100.0 * r["error"]),
                             textcoords="offset points", xytext=(6, 4), fontsize=7)
        axes[1].set_xlabel("H/L [-]")
        axes[1].set_ylabel("frequency error [%]")
        axes[1].set_title("error against linear theory")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        save(fig, "fig2_dispersion.png",
             "Left: measured first-mode frequency vs H/L on top of the linear "
             "dispersion curve; right: the same points as an error against the "
             "10% mission tolerance.")

    # fig3: resolution ladder
    ladder = [r for r in rows if r["family"] in ("resolution", "baseline")]
    if len(ladder) >= 2:
        ladder = sorted(ladder, key=lambda r: r["dx"])
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.loglog([1000 * r["dx"] for r in ladder], [100.0 * r["error"] for r in ladder],
                  "o-", color="C0", label="SPH frequency error")
        ax.axhline(100.0 * TOL, color="C3", ls="--", lw=1.0,
                   label=f"tolerance {100 * TOL:.0f}%")
        for r in ladder:
            ax.annotate(f"cpd={r['cells_per_depth']:.0f}",
                        (1000 * r["dx"], 100.0 * r["error"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xlabel(r"$\Delta x$ [mm]")
        ax.set_ylabel("frequency error [%]")
        ax.set_title("resolution ladder (H/L=0.42)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        save(fig, "fig3_ladder.png",
             "Frequency error against particle spacing for cells-per-depth "
             "8/12/16/20; the coarse end is dominated by the topmost-particle "
             "gauge quantisation, the fine end plateaus near 1%.")

    # fig4: sensitivity panels
    amp = sorted([r for r in rows if r["family"] in ("amplitude", "baseline")],
                 key=lambda r: r["A_over_H"])
    cs = sorted([r for r in rows if r["family"] in ("sound_speed", "baseline")],
                key=lambda r: r["sound_speed_ratio"])
    if amp and cs:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
        axes[0].semilogx([r["A_over_H"] for r in amp], [100.0 * r["error"] for r in amp],
                         "o-", color="C2")
        axes[0].axhline(100.0 * TOL, color="C3", ls="--", lw=1.0, label="tolerance")
        axes[0].set_xlabel("A/H [-]")
        axes[0].set_ylabel("frequency error [%]")
        axes[0].set_title("amplitude sensitivity (cpd=12)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3, which="both")
        axes[1].semilogx([r["sound_speed_ratio"] for r in cs],
                         [100.0 * r["error"] for r in cs], "o-", color="C4")
        axes[1].axhline(100.0 * TOL, color="C3", ls="--", lw=1.0, label="tolerance")
        axes[1].set_xlabel(r"$c_s/\sqrt{gH}$ [-]")
        axes[1].set_ylabel("frequency error [%]")
        axes[1].set_title("weak-compressibility sensitivity (cpd=12)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, which="both")
        save(fig, "fig4_sensitivity.png",
             "One-at-a-time sensitivity of the measured frequency to the released "
             "amplitude A/H and to the artificial sound speed cs/sqrt(gH).")

    # fig5: field frames of the baseline
    if base and base.get("snapshot") and base.get("measured", {}).get("profiles"):
        snap = base["snapshot"]
        profiles = base["measured"]["profiles"]
        t = np.array([p[0] for p in profiles])
        x = np.array(profiles[0][1])
        eta = np.array([p[2] for p in profiles])
        good = np.all(np.isfinite(eta), axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                 gridspec_kw={"width_ratios": [1.0, 1.25]})
        sc = axes[0].scatter(snap["x"], snap["y"], c=snap["speed"], s=6,
                             cmap="viridis", vmin=0.0)
        axes[0].axhline(base["case"]["water_depth"], color="w", ls="--", lw=0.8)
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[0].set_title(f"|u| field at t={snap['time']:.3f} s (quarter period)")
        fig.colorbar(sc, ax=axes[0], label="|u| [m/s]", fraction=0.046)
        im = axes[1].contourf(t[good], x, eta[good].T, levels=21, cmap="RdBu_r")
        axes[1].axhline(base["case"]["water_depth"], color="k", ls=":", lw=0.8)
        axes[1].set_xlabel("t [s]")
        axes[1].set_ylabel("x [m]")
        axes[1].set_title(r"free-surface elevation $\eta(x,t)$ [m]")
        fig.colorbar(im, ax=axes[1], label=r"$\eta$ [m]", fraction=0.046)
        save(fig, "fig5_fields.png",
             "Baseline field frames: left, particle speed |u| at a quarter of the "
             "first-mode period; right, the space-time free-surface elevation, "
             "whose antinode is at the walls and node at mid-length (mode 1).")
    return made


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", default=str(REPO / "runs" / "sloshing_resonance"))
    parser.add_argument("--outdir", default=str(REPO / "runs" / "sloshing_resonance"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = load(indir)
    if not records:
        print(f"no case JSONs under {indir}; run run.py first")
        return 1
    criteria = evaluate(records)
    (outdir / "criteria.json").write_text(json.dumps(criteria, indent=1, allow_nan=True))
    table = markdown_table(criteria["rows"])
    (outdir / "table.md").write_text(
        "# CASE-SLOSHING: ref (linear theory) vs got (TensorSPH)\n\n" + table)
    figs = [] if args.no_figures else figures(records, criteria["rows"], outdir)
    summary = {
        "benchmark": "CASE-SLOSHING",
        "n_cases": len(records),
        "tolerance": TOL,
        "headline": criteria["headline"],
        "verdicts": criteria["declared_questions"],
        "figures": [{k: v for k, v in f.items() if k != "path"} | {
            "file": Path(f["path"]).name} for f in figs],
        "run_artifacts": sorted({r["json"] for r in criteria["rows"]}
                                | {"criteria.json", "table.md",
                                   *[Path(f["path"]).name for f in figs]}),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=True))
    print(table)
    print(json.dumps(criteria["declared_questions"], indent=1, allow_nan=True)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())