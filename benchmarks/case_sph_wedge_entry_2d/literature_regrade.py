"""Re-grade the CASE-SPH-WEDGE-ENTRY-2D grid ladder against the literature reference set.

Inputs (all committed, no solver run)
-------------------------------------
``docs/artifacts/case_sph_wedge_entry_2d_ladder/ladder_closure.json``
    the published grid ladder: for every ``dx`` the estimator
    ``by_frames[61].resolution_band.peak_median`` (in ``q = rho V^2/2`` units),
    its resolution spread and its jackknife spread over the 61 self-similar
    frames, plus the judgeability flag of the previously used rule.
``benchmarks/case_sph_wedge_entry_2d/refs.json``
    the reference set, now including the open-access Table 1 of Jain et al.
    (2022) with the Zhao & Faltinsen (1993) composite value and the exact
    self-similar value at ``beta = 30 deg``.

Outputs
-------
``docs/artifacts/case_sph_wedge_entry_2d_regrade/regrade.json``  numbers (zero hand-copy)
``docs/figures/case_sph_wedge_entry_2d_regrade_*.png``           three figures

Pre-declared decision rule (fixed before looking at the numbers)
----------------------------------------------------------------
gate ``G = 0.15`` (mission tolerance)

* a grid is *judgeable* iff its estimator uncertainty (the larger of the
  resolution spread and the jackknife spread reported by ``ladder_closure``) is
  ``<= G`` -- otherwise the measurement cannot resolve the gate and its residual
  is reported but not scored;
* a judgeable grid is PASS iff ``|obs/ref - 1| <= G``;
* the case is PASS iff at least two judgeable grids exist and all of them pass
  with the same sign; PARTIAL_SINGLE_GRID if exactly one judgeable grid passes;
  FAIL if a judgeable grid fails; UNDETERMINED if no grid is judgeable.

The reference is the *mission-designated* Zhao & Faltinsen (1993) composite
value in ``q`` units (Table 1 column COM).  The exact self-similar value
(Dobrovol'skaya 1969 via Wang & Faltinsen 2017, column EIM) and the other four
models are reported alongside as the model-form spread -- a residual inside that
spread is not solver evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from literature_refs import (
    DESIGNATED_MODEL, EXACT_MODEL, POINT_MODELS, band, cp_max_table,
    c_wagner, designated_cp_max, load_refs, sorted_rows, table_ref,
    wagner_eq10_over_q, wagner_peak_eq10,
)

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ladder/ladder_closure.json"
OUT_DIR = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_regrade"
FIG_DIR = ROOT / "docs/figures"
BETA_DEG = 30.0
BETA = math.radians(BETA_DEG)
GATE = 0.15
# fig filename suffix; empty for the historical (pre-bandfix) campaign so
# that committed figures keep their names (m23280)
FIG_SUFFIX = ""


def rel_err(obs: float, ref: float) -> float:
    return obs / ref - 1.0


def grade(ladder: dict, refs: dict, model: str) -> dict:
    ref_q = cp_max_table(BETA_DEG, refs)[model]
    rows = []
    for g in ladder["grids"]:
        unc = float(g["uncertainty"])
        err = rel_err(float(g["obs_peak"]), ref_q)
        judgeable = unc <= GATE
        scored = bool(judgeable)
        verdict = ("PASS" if abs(err) <= GATE else "FAIL") if scored else "UNSCORED"
        rows.append({
            "tag": g["tag"], "dx_m": float(g["dx"]),
            "obs_peak_q_units": float(g["obs_peak"]),
            "uncertainty_rel": unc,
            "residual_rel": err,
            "judgeable": judgeable, "scored": scored, "verdict": verdict,
        })
    scored = [r for r in rows if r["scored"]]
    passed = [r for r in scored if r["verdict"] == "PASS"]
    if not scored:
        case = "UNDETERMINED"
        why = "no grid resolves the 15 % gate (estimator uncertainty > gate everywhere)"
    elif len(passed) != len(scored):
        case = "FAIL"
        why = f"{len(scored) - len(passed)} judgeable grid(s) outside the gate"
    elif len(passed) >= 2 and len({math.copysign(1.0, r["residual_rel"]) for r in passed}) == 1:
        case = "PASS"
        why = f"{len(passed)} judgeable grids inside the gate with the same sign"
    else:
        case = "PARTIAL_SINGLE_GRID"
        why = (f"{len(passed)} judgeable grid(s) inside the gate; closure rule asks for "
               ">= 2 with the same sign")
    return {"model": model, "reference_q_units": ref_q, "grids": rows,
            "verdict": case, "reason": why}


def main(ladder_path=None, out_dir=None, fig_dir=None, fig_suffix="") -> dict:
    """Grade a ladder artefact against the literature reference set.

    Defaults reproduce the committed m21430 artefacts; the keyword arguments let a
    later campaign (m23280: the same runs re-measured on the fixed resolution band)
    be graded through *this* code path instead of a re-implementation.
    """
    global LADDER, OUT_DIR, FIG_DIR, FIG_SUFFIX
    if ladder_path is not None:
        LADDER = Path(ladder_path)
    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    if fig_dir is not None:
        FIG_DIR = Path(fig_dir)
    FIG_SUFFIX = fig_suffix
    refs = load_refs()
    ladder = json.loads(LADDER.read_text())
    band30 = band(BETA_DEG, refs)
    wagner_point = float(ladder["wagner_cp_max_point"])

    graded = {m: grade(ladder, refs, m) for m in POINT_MODELS}
    designation = graded[DESIGNATED_MODEL]
    exact = graded[EXACT_MODEL]

    # ---- reference-side audit: eq. (10) of the source vs the in-repo profile --
    from tensorsph.wedge_entry import wagner_p_over_q, wagner_peak_over_q

    tau = np.linspace(0.0, 0.98, 4001)
    eq10 = np.array([wagner_eq10_over_q(float(t), BETA) for t in tau])
    repo = np.array([float(wagner_p_over_q(float(t), BETA)) for t in tau])
    dev = repo - eq10
    # the variant that also charges the (1/2) rho V^2 of the uniform stream on
    # top of the jet term -- this is what the m17079 audit computed as "Euler"
    euler_double_count = eq10 - 1.0
    asym_point_rel = wagner_peak_eq10(BETA) / band30["models"][DESIGNATED_MODEL] - 1.0

    out = {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/literature_regrade.py",
        "beta_deg": BETA_DEG,
        "gate": GATE,
        "cp_units": "q = rho V^2 / 2 (the case's Cp = p/q); Table 1 values are P/(rho V^2) and are scaled by 2",
        "inputs": {"ladder": str(LADDER.relative_to(ROOT)),
                   "refs": "benchmarks/case_sph_wedge_entry_2d/refs.json"},
        "reference_family_beta30_q_units": band30,
        "wagner_asymptotic_point_q_units": wagner_point,
        "wagner_asymptotic_vs_designated_rel": asym_point_rel,
        "designation": designation,
        "exact_similarity": exact,
        "by_model": graded,
        "reference_side_audit": {
            "eq10_vs_repo_profile_max_abs_dev": float(np.max(np.abs(dev))),
            "eq10_vs_repo_profile_max_rel_dev": float(np.max(np.abs(dev) / np.abs(eq10))),
            "repo_minus_eq10_mean": float(np.mean(dev)),
            "audit_variant_minus_eq10": float(np.mean(euler_double_count - eq10)),
            "wagner_peak_over_q_module": float(wagner_peak_over_q(BETA)),
            "wagner_peak_eq10": float(wagner_peak_eq10(BETA)),
            "audit_variant_peak_q_units": float(np.max(euler_double_count)),
        },
        "ladder_reported_by_previous_rule": ladder.get("verdict"),
        "ladder_grids": ladder["grids"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "regrade.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    figs = [fig_ladder(out, refs), fig_boundary(refs), fig_profile(out, tau, repo, eq10)]
    out["figures"] = [str(p.relative_to(ROOT)) for p in figs]
    (OUT_DIR / "regrade.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    print(json.dumps({
        "designated_ref_q": designation["reference_q_units"],
        "exact_ref_q": exact["reference_q_units"],
        "band": band30["models"],
        "designation_verdict": designation["verdict"],
        "graded_residuals_vs_COM": [(r["tag"], round(r["residual_rel"] * 100, 2)) for r in designation["grids"]],
        "graded_residuals_vs_EIM": [(r["tag"], round(r["residual_rel"] * 100, 2)) for r in exact["grids"]],
        "wagner_asymptotic_vs_designated_rel": round(asym_point_rel * 100, 2),
        "eq10_vs_repo_max_abs_dev": out["reference_side_audit"]["eq10_vs_repo_profile_max_abs_dev"],
    }, indent=1))
    return out


def fig_ladder(out: dict, refs: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    tab = out["reference_family_beta30_q_units"]["models"]
    colors = {"COM": "#b2182b", "EIM": "#2166ac"}
    for m, v in tab.items():
        ax.axhline(v, ls="-" if m in colors else ":", lw=2.0 if m in colors else 1.0,
                   color=colors.get(m, "0.55"), zorder=1)
        ax.annotate(f"{m} = {v:.2f}", (0.985, v), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7.5,
                    color=colors.get(m, "0.35"))
    ax.axhline(out["wagner_asymptotic_point_q_units"], color="0.25", ls="--", lw=1.4, zorder=1)
    ax.annotate(f"repo Wagner asymptotic C^2+1 = {out['wagner_asymptotic_point_q_units']:.2f}",
                (0.015, out["wagner_asymptotic_point_q_units"]), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=7.5, color="0.25")
    dx = np.array([r["dx_m"] for r in out["designation"]["grids"]])
    obs = np.array([r["obs_peak_q_units"] for r in out["designation"]["grids"]])
    unc = np.array([r["uncertainty_rel"] for r in out["designation"]["grids"]])
    ax.errorbar(dx * 1e3, obs, yerr=obs * unc, fmt="o-", color="k", ms=6, capsize=4,
                lw=1.4, zorder=3, label="SPH peak estimator (61-frame ensemble)")
    for x, y, r in zip(dx * 1e3, obs, out["designation"]["grids"]):
        ax.annotate(f"{r['residual_rel']*100:+.1f}% vs COM", (x, y), textcoords="offset points",
                    xytext=(6, 8), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks(dx * 1e3)
    ax.set_xticklabels([f"{v:g}" for v in dx * 1e3])
    ax.set_xlabel("particle spacing dx [mm]")
    ax.set_ylabel("peak slamming pressure coefficient  $C_p = p_{max}/q$")
    ax.set_title(r"CASE-SPH-WEDGE-ENTRY-2D, $\beta=30^\circ$: SPH ladder vs literature band"
                 "\n(error bars = estimator uncertainty; error % is vs the designated Z&F 1993 value)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, bbox_to_anchor=(1.0, 0.86))
    fig.tight_layout()
    p = FIG_DIR / f"case_sph_wedge_entry_2d_regrade_ladder{FIG_SUFFIX}.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_boundary(refs: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    rows = sorted_rows(refs)
    betas = np.array([float(r) for r in rows])
    styles = {"COM": ("-", "o", "#b2182b"), "EIM": ("-", "s", "#2166ac"),
              "GWM": ("--", "^", "#1b7837"), "MLM": ("--", "v", "#762a83"),
              "ASM": (":", "D", "#e08214"), "OLM": (":", "x", "0.4")}
    for m in POINT_MODELS:
        vals = np.array([table_ref(refs)["cp_max_rho_v2"][r][m] for r in rows])
        ls, mk, col = styles[m]
        ax.plot(betas, vals, ls=ls, marker=mk, color=col, ms=4.5, lw=1.5, label=m)
    C = c_wagner(math.radians(30.0))
    ax.plot([30.0], [0.5 * (C * C + 1.0)], "k*", ms=13, zorder=5,
            label=r"in-repo Wagner $C^2+1$ (at 30$^\circ$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"deadrise angle $\beta$ [deg]")
    ax.set_ylabel(r"$C_{p,max} = P_{max}/\rho V^2$")
    ax.set_title("Literature reference band for wedge slamming\n"
                 "(Jain et al. 2022 JFM 934 A26, Table 1)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = FIG_DIR / f"case_sph_wedge_entry_2d_regrade_boundary{FIG_SUFFIX}.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_profile(out: dict, tau, repo, eq10) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    ax.plot(tau, eq10, "-", lw=3.2, color="0.75", label="Jain et al. eq. (10), recomputed")
    ax.plot(tau, repo, "--", lw=1.3, color="k", label="in-repo wagner_p_over_q")
    ax.plot(tau, eq10 - 1.0, ":", lw=1.5, color="#b2182b",
            label="m17079 'Euler' variant (extra $-q$)")
    ax.set_xlabel(r"$\tau = |x|/c$")
    ax.set_ylabel(r"$p/q$")
    ax.set_ylim(-2, 22)
    ax.set_title(r"Wagner pressure profile, $\beta=30^\circ$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(tau, repo - eq10, "-", color="k", lw=1.2, label="repo - eq.(10)")
    ax.set_xlabel(r"$\tau = |x|/c$")
    ax.set_ylabel(r"deviation [q units]")
    ax.set_title(f"max |repo - eq.(10)| = {out['reference_side_audit']['eq10_vs_repo_profile_max_abs_dev']:.1e} q")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / f"case_sph_wedge_entry_2d_regrade_profile{FIG_SUFFIX}.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="regrade the wedge ladder against refs")
    ap.add_argument("--ladder", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--fig-suffix", default="")
    a = ap.parse_args()
    main(a.ladder, a.out, a.fig_dir, a.fig_suffix)