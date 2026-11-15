#!/usr/bin/env python3
"""Render the independent-audit note and its figures from committed JSON.

Nothing in the produced markdown is typed by hand: every number is read from
``out/result.json`` (the case's self-report) and ``out/audit_independent.json``
(what :mod:`audit_independent` derives), so the note cannot drift from the data.

Usage
-----
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/make_audit_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CASE = REPO / "benchmarks" / "case_poiseuille"
OUT = CASE / "out"
DOC = REPO / "docs" / "case_poiseuille_independent_audit.md"
FIGDIR = REPO / "docs" / "figures" / "case_poiseuille_audit"


def _load():
    res = json.loads((OUT / "result.json").read_text())
    aud = json.loads((OUT / "audit_independent.json").read_text())
    return res, aud


def fig_profiles(res, aud):
    """Measured vs analytic amplitude/phase, plus the two independent references."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "case_poiseuille_audit", CASE / "audit_independent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    finest = max(res["harmonic_runs"], key=lambda r: r["resolution_H_over_dx"])
    y = np.array([r["y"] for r in finest["rows"]])
    amp = np.array([r["amplitude"] for r in finest["rows"]])
    pha = np.array([r["phase_deg"] for r in finest["rows"]])
    order = np.argsort(y)
    ubvp = mod.analytic_bvp(y, omega=float(finest["omega"]))
    ufd = mod.fd_marched(y, omega=float(finest["omega"]))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].plot(y[order] * 1e2, amp[order], "o", ms=5, label="SPH rows (committed)")
    ax[0].plot(y[order] * 1e2, np.abs(ubvp)[order], "-", lw=1.4,
               label="analytic (independent BVP)")
    ax[0].plot(y[order] * 1e2, np.abs(ufd)[order], "--", lw=1.2,
               label="Crank-Nicolson march (audit)")
    ax[0].set_xlabel("y [cm]")
    ax[0].set_ylabel("|U| [m/s]")
    ax[0].set_title(f"Womersley amplitude, H/dx={finest['resolution_H_over_dx']}")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(y[order] * 1e2, pha[order], "o", ms=5, label="SPH rows (committed)")
    ax[1].plot(y[order] * 1e2, np.degrees(np.angle(ubvp))[order], "-", lw=1.4,
               label="analytic (independent BVP)")
    ax[1].plot(y[order] * 1e2, np.degrees(np.angle(ufd))[order], "--", lw=1.2,
               label="Crank-Nicolson march (audit)")
    ax[1].set_xlabel("y [cm]")
    ax[1].set_ylabel("phase lag [deg]")
    ax[1].set_title("phase of the frequency response")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = FIGDIR / "profiles_vs_reference.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_errors(res, aud):
    """Gate metrics and the honest wall-band disclosure vs resolution."""
    runs = aud["harmonic_gate_audit"]
    hdx = np.array([r["resolution_H_over_dx"] for r in runs], dtype=float)
    amp_c = np.array([abs(r["recomputed_committed_analytic"]["amp_centre_pct"]) for r in runs])
    amp_r = np.array([r["recomputed_committed_analytic"]["amp_profile_rms_pct"] for r in runs])
    pha_c = np.array([r["recomputed_committed_analytic"]["phase_centre_pct_of_quarter_turn"]
                      for r in runs])
    wall = np.array([r["recomputed_committed_analytic"]["amp_wall_pct"] for r in runs])

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.loglog(hdx, amp_c, "o-", label="amp, centre row")
    ax.loglog(hdx, amp_r, "s-", label="amp, interior RMS (0.15<=y/H<=0.85)")
    ax.loglog(hdx, pha_c, "^-", label="phase, centre (% of pi/2)")
    ax.loglog(hdx, wall, "v--", color="crimson",
              label="amp, wall-adjacent row (NOT graded)")
    ax.axhline(5.0, color="k", ls=":", lw=1.2, label="5 % acceptance")
    ax.set_xlabel("H/dx")
    ax.set_ylabel("error [%]")
    ax.set_xticks(hdx)
    ax.set_xticklabels([f"{int(v)}" for v in hdx])
    ax.set_title("CASE-POISEUILLE-SPH: gate metrics vs resolution")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "errors_vs_resolution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_falsifiability(aud):
    """Mutation response: the gate stack must reject over-tolerance data."""
    muts = aud["falsifiability"]["mutations"]
    tags = list(muts)
    g1 = [1.0 if m["gates"]["G1_amp_centre_5pct"] else 0.0 for m in muts.values()]
    g2 = [1.0 if m["gates"]["G2_amp_profile_rms_5pct"] else 0.0 for m in muts.values()]
    g3 = [1.0 if m["gates"]["G3_phase_centre_5pct_of_quarter_turn"] else 0.0
          for m in muts.values()]
    g4 = [1.0 if m["gates"]["G4_phase_profile_rms_5pct_of_quarter_turn"] else 0.0
          for m in muts.values()]

    x = np.arange(len(tags))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for i, (lab, vals) in enumerate((("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4))):
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["rejected", "passes"])
    ax.set_title("falsifiability: gate response to published-data mutations")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "falsifiability.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def render(res, aud, figs):
    v = aud["verdict"]
    ref = aud["reference_provenance"]
    runs = aud["harmonic_gate_audit"]
    disc = aud["frequency_discrimination"]
    wall = aud["wall_band_disclosure"]
    fals = aud["falsifiability"]

    L = []
    A = L.append
    A("# CASE-POISEUILLE-SPH — independent audit of the evidence bundle")
    A("")
    A(f"> Generated by `benchmarks/case_poiseuille/make_audit_doc.py` from "
      f"`out/result.json` (sha256 `{aud['source_sha256'][:16]}`) and "
      f"`out/audit_independent.json`. Every number below is read from those files.")
    A("")
    A(f"**Disposition: {v['disposition']}.** "
      f"reference provenance confirmed = {v['reference_provenance_independently_confirmed']}, "
      f"gate arithmetic reproduced = {v['gate_arithmetic_reproduced']}, "
      f"gates reject degraded data = {v['gates_reject_degraded_data']}, "
      f"5 % acceptance holds on the graded band = {v['acceptance_5pct_holds_interior_band']}.")
    A("")
    A("## 1. What was audited")
    A("")
    A("The case-pool entry *算例池·泊肃叶流* (`CASE-POISEUILLE-SPH`) is a plane channel "
      "`0 <= y <= H` driven by a body force `g(t)`: constant (`R1`, steady parabola and "
      "Boussinesq start-up) and harmonic (`R2`, Womersley periodic state "
      "`g = g0 cos(w t)`). The evidence bundle is `out/result.json`: the analytic "
      "reference, the measured per-row amplitude and phase at three resolutions, and "
      "eleven gates, all produced by `run.py`.")
    A("")
    A("That artifact is a *self-report*: the reference, the finite-difference check, the "
      "harmonic fit and the gate arithmetic all live in code written in the same change. "
      "The audit therefore re-derives the case from the PDE and imports none of it.")
    A("")
    A("## 2. Method")
    A("")
    A("| route | equation solved | grid / scheme | shares code with the case |")
    A("|---|---|---|---|")
    A("| closed form (own algebra) | `U'' - a^2 U = -g0/nu`, `a = sqrt(i w/nu)` | analytic | no |")
    A("| complex BVP | `i w U - nu U'' = g0`, `U(0)=U(H)=0` | central FD, 4001 nodes, banded solve | no |")
    A("| time-domain march | `u_t = nu u_yy + g0 cos(w t)`, from rest | Crank-Nicolson, nx=801, 2048 steps/period, projection | no |")
    A("| per-row re-fit | `A cos(w t + phi) + c` | own normal equations | no |")
    A("")
    A("## 3. Reference provenance")
    A("")
    A("| omega [rad/s] | closed form vs BVP | committed `amplitude_analytic` vs BVP | committed `phase_analytic` vs BVP [deg] | marched FD vs BVP (amp) | marched FD vs BVP (phase) [deg] |")
    A("|---|---|---|---|---|---|")
    for tag, b in ref.items():
        A(f"| {b['omega']:.1f} | {b['closed_form_vs_bvp_rel']:.2e} | "
          f"{b['committed_amp_vs_bvp_rel']:.2e} | {b['committed_phase_vs_bvp_deg']:.2e} | "
          f"{b['marched_fd_vs_bvp_amp_rel']:.2e} | {b['marched_fd_vs_bvp_phase_deg']:.2e} |")
    A("")
    A("The time-domain march matters for one specific reason: a sign convention shared by "
      "the closed form and the harmonic fit would cancel in every comparison they make "
      "between themselves. The march knows nothing about either, and it reaches the same "
      "complex amplitude, so the phase convention (lag, not lead) is established, not "
      "assumed.")
    A("")
    A("![profiles](figures/case_poiseuille_audit/profiles_vs_reference.png)")
    A("")
    A("*Amplitude and phase of the frequency response: SPH rows as committed, analytic "
      "from the independent boundary-value solve, and the time-domain march.*")
    A("")
    A("## 4. Reproducing the published gates from the raw samples")
    A("")
    A("| H/dx | amp centre [%] published / recomputed | amp interior RMS [%] | phase centre [% of pi/2] | phase interior RMS [% of pi/2] | wall row [%] (not graded) |")
    A("|---|---|---|---|---|---|")
    for r in runs:
        p, c = r["published"], r["recomputed_committed_analytic"]
        A(f"| {r['resolution_H_over_dx']} | {p['amp_centre_pct']:+.3f} / {c['amp_centre_pct']:+.3f} | "
          f"{c['amp_profile_rms_pct']:.3f} | {c['phase_centre_pct_of_quarter_turn']:.3f} | "
          f"{c['phase_profile_rms_pct_of_quarter_turn']:.3f} | {c['amp_wall_pct']:.2f} |")
    A("")
    A("The re-fit of the committed centre-line series reproduces the published fit to "
      f"{max(r['refit_from_centre_series']['amplitude_vs_published_rel'] for r in runs):.1e} "
      "in amplitude and "
      f"{max(r['refit_from_centre_series']['phase_vs_published_deg'] for r in runs):.1e} deg "
      "in phase, so the published gate values are a consequence of the samples rather than "
      "of the post-processing code.")
    A("")
    A("![errors](figures/case_poiseuille_audit/errors_vs_resolution.png)")
    A("")
    A("*Gate metrics against the 5 % acceptance. The wall-adjacent row (dashed) is above the "
      "acceptance at every rung and is outside the graded band.*")
    A("")
    A("## 5. Falsifiability")
    A("")
    A("| mutation | G1 | G2 | G3 | G4 | all pass |")
    A("|---|---|---|---|---|---|")
    for tag, m in fals["mutations"].items():
        g = m["gates"]
        A(f"| {tag} | {g['G1_amp_centre_5pct']} | {g['G2_amp_profile_rms_5pct']} | "
          f"{g['G3_phase_centre_5pct_of_quarter_turn']} | "
          f"{g['G4_phase_profile_rms_5pct_of_quarter_turn']} | {m['all_pass']} |")
    A("")
    A("![falsifiability](figures/case_poiseuille_audit/falsifiability.png)")
    A("")
    A("## 6. Discrimination: this is a measurement, not a tautology")
    A("")
    A(f"The two frequencies separate an unsteady solver from a quasi-steady one by "
      f"**{disc['quasi_steady_rejection_pct']:.1f} %** in centre-line amplitude ratio "
      f"(measured {disc['ratio_measured']:.4f}, independent analytic "
      f"{disc['ratio_analytic_independent']:.4f}; a quasi-steady solver would return 1.0). "
      f"Compared against the steady parabola at the same forcing, the harmonic centre-line "
      f"amplitude differs by {disc['steady_would_be_error_pct_of_analytic_harmonic']:.1f} %. "
      f"The measured ratio is {disc['error_vs_analytic_pct']:.2f} % from the independent "
      f"analytic value (G6 tolerance 5 %).")
    A("")
    A("## 7. Caveats (read before quoting the acceptance)")
    A("")
    A(f"* {wall['statement']}.")
    for row in wall["rows"]:
        A(f"  * H/dx={row['resolution_H_over_dx']}: worst excluded row "
          f"{row['worst_excluded_amp_pct']:.2f} % amplitude error, "
          f"{row['excluded_rows']} rows outside the band.")
    A("* The measurement window is sampled at solver steps, so it spans "
      + ", ".join(f"{r['refit_from_centre_series']['window_span_periods']:.5f}"
                 for r in runs) +
      " periods rather than exactly 2. The fit carries a constant column and solves the "
      "full normal equations; the residual of the refit is below "
      f"{max(r['refit_from_centre_series']['residual'] for r in runs):.3f} relative.")
    A("* G5 (window stability) compares two consecutive measurement periods of the *same* "
      "run. It detects a run that has not reached the periodic state; it does not, by "
      "itself, establish convergence.")
    A("")
    A("## 8. Not run")
    A("")
    for item in aud["not_run"]:
        A(f"* {item}.")
    A("")
    A("## 9. The case's own suite at the audited commit")
    A("")
    unc = json.loads((OUT / "uncertainty.json").read_text())
    for line in render_status(suite_status(), unc):
        A(line)
    A("")
    A("## 10. Disposition")
    A("")
    A(f"**{v['disposition']}.** The reference is independently confirmed to "
      f"{max(b['committed_amp_vs_bvp_rel'] for b in ref.values()):.1e} in amplitude, the "
      "published gates are reproduced from the committed samples, and the gate stack "
      "rejects over-tolerance data while passing a sub-tolerance control. The notes that "
      "travel with the approval are sections 7 and 8: the 5 % acceptance is a statement "
      "about the predeclared interior band, and the solver itself was not re-executed by "
      "this audit.")
    A("")
    DOC.write_text("\n".join(L) + "\n")
    return DOC


def suite_status():
    """Run the case's own R1-ladder tests and record what fails, unedited.

    This is a *report*, not a repair: the failures below were present at the
    audited commit and the audit does not touch the assertions that carry them.
    """
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "pytest",
           "tests/test_case_poiseuille_r1_ladder.py",
           "-q", "--no-header", "-p", "no:cacheprovider"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": ".:src"},
                          timeout=600)
    failed = [line.split("::")[1].split(" ")[0]
              for line in proc.stdout.splitlines() if line.startswith("FAILED")]
    tail = [l for l in proc.stdout.splitlines() if " passed" in l or " failed" in l]
    return {"command": " ".join(cmd[1:]), "returncode": proc.returncode,
            "summary": tail[-1].strip() if tail else "", "failed": failed}


def render_status(status, unc):
    steady = unc["r1_quantities"]["r1_steady_profile_rms"]
    L = []
    if not status["failed"]:
        L.append("The case's own R1-ladder suite is green at the audited commit.")
        return L
    L.append("The case's own utility/uncertainty suite is **not green** at the audited "
             "commit, and the failures predate this audit (no file outside the audit's "
             "own paths was modified). `" + status["command"] + "` -> "
             f"`{status['summary']}`.")
    L.append("")
    for name in status["failed"]:
        L.append(f"* `{name}`")
    L.append("")
    L.append("Both assert that the R1 *steady* band does **not** support the 5 % acceptance "
             "(an honest FAIL recorded when the band was wider). The committed "
             "`out/uncertainty.json` now reports "
             f"`U_V = {steady['validation']['U_V_percent']:.2f} %` with the order "
             f"identified (`p = {steady['p_used']:.2f}`, spread "
             f"`{steady['order_analysis']['order_spread']:.2f}`, status "
             f"`{steady['order_analysis']['status']}`), so `U3b_R1_band_supports_5pct` is "
             f"`{unc['gates']['U3b_R1_band_supports_5pct']}` in the artifact while the "
             "pinned expectation is `False`.")
    L.append("")
    L.append("Which side is stale is a substantive question, not a formatting one: either "
             "the band genuinely narrowed after the honest FAIL was recorded (in which case "
             "the assertion needs new evidence attached, and the FAIL must not simply be "
             "flipped) or the artifact drifted. **This audit does not flip the assertion** "
             "and records the mismatch instead; the 5 % acceptance quoted above rests on "
             "the G-gates in `out/result.json`, which this audit does reproduce, and not "
             "on `U3b`.")
    return L


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    res, aud = _load()
    figs = {
        "profiles": fig_profiles(res, aud),
        "errors": fig_errors(res, aud),
        "falsifiability": fig_falsifiability(aud),
    }
    doc = render(res, aud, figs)
    for name, path in figs.items():
        print(f"[doc] figure {name}: {path.relative_to(REPO)} "
              f"({path.stat().st_size // 1024} kB)")
    print(f"[doc] wrote {doc.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())