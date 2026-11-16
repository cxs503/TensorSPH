"""CASE-SPH-WEDGE-ENTRY-2D -- do the refinement ladder *and* the realisation
ensemble agree about how big the error is?  (m23321)

Why this round exists
---------------------
Round m23165 measured the *design effect* of the realisation axis at ``dx=4mm``
(``rho_bar = 0.815``, ``N_eff ~ 1.19``) and therefore downgraded the
realisation route.  It left the *other* route -- grid refinement -- untouched,
and wrote down two things a next round had to answer:

* 7-2: "add realisations until ``U_ci < 15%``" is falsified if the fine grid
  does not reproduce the same pattern;
* 7-3: the **directional risk** "the symmetric lattice reads systematically
  high" (4mm: unperturbed 10.04 vs phase-median 8.11).

This module answers both with data that is already committed.  Nothing here runs
SPH, changes the solver, adds a criterion or relaxes a gate: it reads

* the four committed single-run grids
  ``docs/artifacts/case_sph_wedge_entry_2d_ensemble/ensemble_dx*_c0v20.json``
  (dx = 4 / 2 / 1 / 0.5 mm -- the refinement ladder),
* the committed phase realisations
  ``docs/artifacts/case_sph_wedge_entry_2d_realisations/real_dx{4,2}mm_*_seed[1-9].json``
  (``scale = 1`` only; the ``scale = 0.25`` members belong to the A2 amplitude
  study and are excluded from every ensemble statistic here),
* the same-operator truth
  ``docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json``
  (Wagner's field through *the same measurement operator*, seed independent),
* the reference side
  ``docs/artifacts/case_sph_wedge_entry_2d_regrade/regrade.json``
  (Z&F 1993 composite ``COM = 6.92`` in ``q`` units, gate ``G = 0.15``).

Predeclared decision rules (fixed before the numbers were computed)
-------------------------------------------------------------------
``r_g`` = ``obs_g / synth_g - 1``   same-operator residual of grid ``g``
``e_g`` = ``obs_g / COM - 1``       acceptance residual of grid ``g``
``r_i`` = the same formula for one phase realisation

* **P1 refinement monotonicity.** With ``dx`` decreasing 4 -> 2 -> 1 -> 0.5 mm
  the sequence ``|r_g|`` must be non-increasing and ``sign(r_g)`` must not
  change for the ladder to be a convergence sequence.  ``n_sign_changes >= 1``
  or ``max|r| > G`` on the finest grid means the ladder is **not** a convergence
  sequence at the measured range, and standard GCI/order analysis is not
  applicable (it presupposes monotone same-sign convergence).
* **P2 symmetric-lattice representativeness.** For every grid with ``N >= 3``
  phase realisations, ``z = (r_sym - mean_i r_i) / sd_i(r_i)``; ``|z| > 2``
  means the unperturbed lattice is an **outlier** of its own grid's phase
  distribution.  "Systematically high" is only supported if every outlier grid
  has ``z > 0``; mixed signs falsify it.
* **P3 chance-PASS.** Convert the measured realisation residuals to the
  acceptance denominator (``(1+r)*synth_g/COM - 1``) and fit ``N(mu, sd)``.
  Report ``P_chance = P(|e| <= G)`` (how cheap a hit is) *and* the 95 % CI of the
  mean.  A single grid inside the gate is **evidence** only if that CI lies
  inside the gate; otherwise it is a draw from a distribution that straddles or
  misses the gate and must not be scored as a PASS.
* **P4 honest floor.** Pool every *measured* decision uncertainty -- the
  within-run ``U`` of each grid and the between-realisation ``sd`` of each
  ensemble -- and report both the optimistic floor (``min``) and the typical
  value (``median``).  ``metric_can_meet_gate`` = the optimistic floor is inside
  ``G`` (the best case is not excluded); ``metric_resolution_stable`` = the
  *median* is inside ``G`` (the metric reliably delivers the tolerance).  A
  metric whose typical resolution exceeds the gate cannot be used to score the
  case, however favourable its best single number is -- that is a statement
  about the metric, not about the solver's accuracy.

Each rule is a pure function (no file IO) so the tests can feed synthetic
ladders, including the negative controls: a genuinely monotone ladder must come
out ``monotone=True`` and a symmetric sample that *is* representative must come
out ``outlier=False``.
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ENSEMBLE_DIR = REPO / "docs/artifacts/case_sph_wedge_entry_2d_ensemble"
REAL_DIR = REPO / "docs/artifacts/case_sph_wedge_entry_2d_realisations"
BIAS_PATH = REPO / "docs/artifacts/case_sph_wedge_entry_2d_bias/bias_report.json"
REGRADE_PATH = REPO / "docs/artifacts/case_sph_wedge_entry_2d_regrade/regrade.json"
OUT_DIR = REPO / "docs/artifacts/case_sph_wedge_entry_2d_phase_ladder"

GRID_TAGS = ("dx4mm", "dx2mm", "dx1mm", "dx0.5mm")
GRID_DX = {"dx4mm": 4e-3, "dx2mm": 2e-3, "dx1mm": 1e-3, "dx0.5mm": 5e-4}
GATE = 0.15


# --------------------------------------------------------------------------- #
# pure statistics -- no file IO, unit-testable
# --------------------------------------------------------------------------- #
def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_changes(values: list[float]) -> int:
    """Number of sign changes in the sequence (zeros are skipped)."""
    signs = [1 if v > 0 else -1 for v in values if v != 0.0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def ladder_verdict(residuals: list[float], gate: float = GATE) -> dict:
    """P1: is ``residuals`` (ordered coarse -> fine) a convergence sequence?"""
    if len(residuals) < 2:
        raise ValueError("need at least two grids")
    absr = [abs(r) for r in residuals]
    n_sign = sign_changes(residuals)
    non_increasing = all(b <= a + 1e-12 for a, b in zip(absr, absr[1:]))
    adaptive = non_increasing and n_sign == 0
    return {
        "residuals": list(residuals),
        "abs_residuals": absr,
        "n_sign_changes": n_sign,
        "abs_monotone_decreasing": non_increasing,
        # signed excursion: how far the reading swings, coarse high to fine low
        "span_pp": 100.0 * (max(residuals) - min(residuals)),
        "span_over_gate": (max(residuals) - min(residuals)) / gate,
        "finest_abs_residual": absr[-1],
        "finest_outside_gate": bool(absr[-1] > gate),
        "gci_applicable": adaptive,
        "monotone": adaptive,
        "coarsest_to_finest_ratio": absr[0] / absr[-1] if absr[-1] > 0 else float("inf"),
    }


def ensemble_position(sym_residual: float, phase_residuals: list[float],
                      z_crit: float = 2.0) -> dict:
    """P2: where does the unperturbed lattice sit inside its phase ensemble?"""
    n = len(phase_residuals)
    if n < 2:
        raise ValueError("need at least two realisations for a spread")
    mean = st.mean(phase_residuals)
    sd = st.stdev(phase_residuals)
    z = (sym_residual - mean) / sd if sd > 0 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "median": st.median(phase_residuals),
        "sd": sd,
        "cv_of_residual": sd / abs(mean) if mean else float("nan"),
        "reject_sd_of_mean": sd / math.sqrt(n),
        "symmetric_residual": sym_residual,
        "sym_minus_mean": sym_residual - mean,
        "z": z,
        "is_outlier": bool(abs(z) > z_crit) if sd > 0 else False,
        "direction": ("high" if z > 0 else "low") if z == z else "unknown",
    }


def direction_consistency(signs: list[int]) -> dict:
    """P2 (cont.): mixed outlier directions falsify 'systematically high'."""
    used = [s for s in signs if s != 0]
    return {
        "outlier_directions": used,
        "n_outlier_grids": len(used),
        "direction_consistent": bool(used) and len(set(used)) == 1,
        "systematically_high": bool(used) and set(used) == {1},
        "systematically_low": bool(used) and set(used) == {-1},
    }


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042}


def t95(dof: int) -> float:
    """Two-sided 95 % Student-t factor (table to 30 dof, normal beyond)."""
    if dof in _T95:
        return _T95[dof]
    keys = sorted(_T95)
    for k in reversed(keys):
        if dof >= k:
            return _T95[k]
    return 1.96


def chance_pass(draws_vs_com: list[float], gate: float = GATE) -> dict:
    """P3: can a single gate hit be read as evidence?

    Two separately reported quantities:

    * ``p_chance`` -- the probability that *one draw* from ``N(mu, sd)`` lands
      inside the gate.  Large means a hit is cheap.
    * ``pass_is_evidence`` -- whether a hit would support "this grid's
      representative value is inside the gate", i.e. whether the 95 % CI of the
      *mean* (``mu +/- t95(n-1) sd/sqrt(n)``) lies inside the gate.  This is the
      quantity that matters: it fails both when the spread is too wide and when
      the mean is biased outside the gate, even if single draws land in it.
    """
    n = len(draws_vs_com)
    if n < 3:
        raise ValueError("need at least three draws")
    mu = st.mean(draws_vs_com)
    sd = st.stdev(draws_vs_com)
    if sd <= 0:
        p = 1.0 if abs(mu) <= gate else 0.0
    else:
        p = _phi((gate - mu) / sd) - _phi((-gate - mu) / sd)
    half = t95(n - 1) * sd / math.sqrt(n) if sd > 0 else 0.0
    ci_lo, ci_hi = mu - half, mu + half
    return {
        "n": n,
        "mu": mu,
        "sd": sd,
        "p_chance": p,
        "mean_ci95": [ci_lo, ci_hi],
        "mean_ci_half_width": half,
        "pass_is_evidence": bool(ci_lo >= -gate and ci_hi <= gate),
        "draws_vs_com": list(draws_vs_com),
    }


def honest_floor(within_run_uncertainty: list[float],
                 between_realisation_sd: list[float]) -> dict:
    """P4: optimistic floor *and* typical resolution of the metric."""
    vals = [v for v in list(within_run_uncertainty) + list(between_realisation_sd)
            if v == v and v >= 0]
    if not vals:
        return {"u_floor": float("nan"), "u_typical": float("nan"),
                "metric_can_meet_gate": False, "metric_resolution_stable": False,
                "n_inputs": 0, "inputs": []}
    floor = min(vals)
    typical = st.median(vals)
    return {
        "u_floor": floor,
        "u_typical": typical,
        "floor_over_gate": floor / GATE,
        "typical_over_gate": typical / GATE,
        "metric_can_meet_gate": bool(floor <= GATE),
        "metric_resolution_stable": bool(typical <= GATE),
        "n_inputs": len(vals),
        "inputs": vals,
    }


# --------------------------------------------------------------------------- #
# artefact readers
# --------------------------------------------------------------------------- #
def _read_key(path: Path) -> dict:
    """Return the peak reading of the last frame set of a committed run."""
    d = json.loads(path.read_text())
    bf = d.get("by_frames") or {}
    if not bf:
        rb = d.get("resolution_band") or {}
        return {"peak_median": rb.get("peak_median", d.get("peak_median")),
                "spread_rel": rb.get("spread_rel", d.get("spread_rel", float("nan")))}
    key = sorted(bf, key=int)[-1]
    rb = bf[key]["resolution_band"]
    return {"peak_median": rb["peak_median"],
            "spread_rel": bf[key].get("spread_rel", float("nan")),
            "n_frames_key": int(key)}


def load_ladder() -> list[dict]:
    reg = json.loads(REGRADE_PATH.read_text())
    com = float(reg["designation"]["reference_q_units"])
    by_tag = {g["tag"]: g for g in reg["ladder_grids"]}
    rows = []
    for tag in GRID_TAGS:
        src = ENSEMBLE_DIR / f"ensemble_{tag}_c0v20.json"
        if not src.exists() or tag not in by_tag:
            continue
        k = _read_key(src)
        g = by_tag[tag]
        obs = float(k["peak_median"])
        synth = float(g["synth_peak"])
        rows.append({
            "tag": tag,
            "dx_m": GRID_DX[tag],
            "obs_peak": obs,
            "synth_peak": synth,
            "residual_same_operator": obs / synth - 1.0,
            "com": com,
            "residual_vs_com": obs / com - 1.0,
            "within_run_uncertainty": float(g["uncertainty"]),
            "judgeable": bool(g["judgeable"]),
            "wall_seconds": float(g.get("wall_seconds", float("nan"))),
            "steps": g.get("steps"),
            "n_bins_used": g.get("n_bins_used"),
            "pooled_samples": g.get("pooled_samples"),
        })
    rows.sort(key=lambda r: -r["dx_m"])
    return rows


def load_realisations(scale: float = 1.0) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(REAL_DIR.glob("real_dx*_c0v20_seed*.json")):
        d = json.loads(path.read_text())
        real = d.get("realisation") or {}
        if not real or real.get("scale") != scale:
            continue
        tag = ("dx4mm" if "4mm" in path.name else
               "dx2mm" if "2mm" in path.name else
               "dx1mm" if "1mm" in path.name else "unknown")
        bf = d.get("by_frames") or {}
        key = sorted(bf, key=int)[-1]
        rb = bf[key]["resolution_band"]
        out.setdefault(tag, []).append({
            "file": path.name,
            "seed": real.get("seed"),
            "phase_x_dx": real.get("phase_x_dx"),
            "gap_jitter_dx": real.get("gap_jitter_dx"),
            "peak_median": rb["peak_median"],
            "band_spread": bf[key].get("spread_rel", float("nan")),
        })
    for v in out.values():
        v.sort(key=lambda r: r["seed"])
    return out


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyse(ladder: list[dict], realisations: dict[str, list[dict]],
            gate: float = GATE) -> dict:
    p1 = ladder_verdict([r["residual_same_operator"] for r in ladder], gate)

    per_grid: dict[str, dict] = {}
    z_signs: list[int] = []
    dx_draws: list[float] = []
    between_sd: list[float] = []
    for row in ladder:
        tag = row["tag"]
        plist = realisations.get(tag, [])
        entry = {"tag": tag, "n_phase_realisations": len(plist), "ensemble": None,
                 "status": "NOT_RUN_at_this_grid"}
        if len(plist) >= 2:
            rr = [p["peak_median"] / row["synth_peak"] - 1.0 for p in plist]
            pos = ensemble_position(row["residual_same_operator"], rr)
            entry.update({"ensemble": pos, "status": "measured",
                          "realisation_residuals": rr,
                          "realisation_peaks": [p["peak_median"] for p in plist]})
            between_sd.append(pos["sd"])
            if pos["is_outlier"]:
                z_signs.append(1 if pos["z"] > 0 else -1)
            if len(plist) >= 3:
                dx_draws += [(1.0 + x) * row["synth_peak"] / row["com"] - 1.0 for x in rr]
        per_grid[tag] = entry

    p2 = direction_consistency(z_signs)
    p3 = (chance_pass(dx_draws, gate) if len(dx_draws) >= 3 else
          {"status": "WITHHELD", "n": len(dx_draws),
           "reason": "fewer than three phase realisations"})
    p4 = honest_floor([r["within_run_uncertainty"] for r in ladder], between_sd)

    scored = [r for r in ladder if r["judgeable"]]
    passed = [r for r in scored if abs(r["residual_vs_com"]) <= gate]
    pass_credible = False
    if passed:
        ens = (per_grid[passed[0]["tag"]] or {}).get("ensemble") or {}
        pass_credible = bool(p3.get("pass_is_evidence", False)
                             and abs(ens.get("z") or 0.0) <= 2.0)
    any_outlier = any((per_grid[t].get("ensemble") or {}).get("is_outlier")
                      for t in per_grid)
    verdict = {
        "ladder_is_convergence_sequence": p1["monotone"],
        "gci_applicable": p1["gci_applicable"],
        "symmetric_lattice_representative": not any_outlier,
        "symmetric_direction_consistent": p2["direction_consistent"],
        "chance_pass_p": p3.get("p_chance"),
        "single_grid_pass_is_evidence": bool(p3.get("pass_is_evidence", False)),
        "metric_can_meet_gate": p4["metric_can_meet_gate"],
        "metric_resolution_stable": p4["metric_resolution_stable"],
    }
    if not verdict["metric_can_meet_gate"]:
        verdict["decision"] = "METRIC_INSUFFICIENT_FOR_GATE"
    elif not verdict["ladder_is_convergence_sequence"]:
        verdict["decision"] = "NOT_DECIDABLE_BY_REFINEMENT_AT_MEASURED_RANGE"
    elif not verdict["single_grid_pass_is_evidence"]:
        verdict["decision"] = "SINGLE_GRID_PASS_NOT_EVIDENCE"
    else:
        verdict["decision"] = "SUPPORTED"
    return {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/phase_ladder_consistency.py",
        "gate": gate,
        "P1_ladder": p1,
        "P2_per_grid": per_grid,
        "P2_direction": p2,
        "P3_chance_pass": p3,
        "P4_honest_floor": p4,
        "closure": {"scored_grids": [r["tag"] for r in scored],
                    "passed_grids": [r["tag"] for r in passed],
                    "pass_credible": pass_credible},
        "verdict": verdict,
    }


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--no-doc", action="store_true")
    args = ap.parse_args(argv)

    ladder = load_ladder()
    realisations = load_realisations(scale=1.0)
    res = analyse(ladder, realisations)
    res["ladder"] = ladder
    res["realisations_used"] = {k: [p["file"] for p in v] for k, v in realisations.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phase_ladder_consistency.json").write_text(
        json.dumps(res, indent=1, default=float) + "\n")

    if not args.no_figure or not args.no_doc:
        import phase_ladder_doc
        fig = args.out / "figures" / "case_sph_wedge_entry_2d_phase_ladder_consistency.png"
        if not args.no_figure:
            phase_ladder_doc.make_figure(res, ladder, fig)
        if not args.no_doc:
            doc = phase_ladder_doc.render_doc(res, ladder, fig)
            (REPO / "docs/case_sph_wedge_entry_2d_phase_ladder_consistency_m23321.md").write_text(doc)

    v = res["verdict"]
    print(f"decision={v['decision']}  ladder_monotone={v['ladder_is_convergence_sequence']}  "
          f"outlier={not v['symmetric_lattice_representative']}  "
          f"p_chance={v['chance_pass_p']:.3f}  U_floor={res['P4_honest_floor']['u_floor']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())