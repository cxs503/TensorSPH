"""Generate ``docs/verification_mixture_surface_tension.md`` from committed JSON.

Every number in the document is injected here, never transcribed: the sources
are ``results.json`` (main ladder) and ``ablation_curvature.json``. Re-run after
re-running either harness::

    PYTHONPATH=src python benchmarks/xval_mixture_laplace/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parents[1] / "docs" / "verification_mixture_surface_tension.md"


def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def main() -> None:
    res = _load("results.json")
    abl = _load("ablation_curvature.json")
    s = res["setup"]
    summ = res["summary"]
    rows = res["rows"]
    cases = [c["label"] for c in s["rho_cases"]]
    dxs = sorted({r["dx"] for r in rows})
    finest = min(dxs)

    def get(case: str, dx: float) -> dict:
        return [r for r in rows if r["case"] == case and abs(r["dx"] - dx) < 1e-12][0]

    lines: list[str] = []
    A = lines.append
    A("# Mixture-material surface tension vs Young-Laplace (cross-validation)")
    A("")
    A("**Status: PASS** of the mixture-material chain, **FAIL** (honest negative "
      "result, kept) of the shipped single-material colour on mixtures.")
    A("Anchor: `XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION` (TensorSPH). "
      "Acceptance: <15 % error of the effective surface tension against an "
      "analytic/literature reference.")
    A("")
    A("Reproduce:")
    A("")
    A("```bash")
    A("PYTHONPATH=src python benchmarks/xval_mixture_laplace/run.py --plot")
    A("PYTHONPATH=src python benchmarks/xval_mixture_laplace/ablation_curvature.py")
    A("PYTHONPATH=src python benchmarks/xval_mixture_laplace/make_doc.py")
    A("PYTHONPATH=src python -m pytest tests/test_csf_mixture.py -v")
    A("```")
    A("")
    A("## 1. Case introduction")
    A("")
    A("Surface tension in a **mixture** (two materials, unequal rest density) is "
      "the first problem in which the colour function of an SPH code is a *phase "
      "indicator* rather than a *presence indicator*. The reference formulation "
      "for that case (Xiangyu-Hu/SPHinXsys, `SurfaceTension`, i.e. the "
      "C-correction of Adami, Hu & Adams 2010) normalises the colour function by "
      "the local kernel sum of the same per-phase volumes, so that the colour "
      "jump across the interface is exactly 1 whatever the density ratio.")
    A("")
    A("Geometry and parameters (all injected from `results.json`):")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| interface | circle, disc of material A in material B, particles on both sides |")
    A(f"| radius `R` | {s['radius_m']} m |")
    A(f"| surface tension `sigma_AB` | {s['sigma_N_per_m']} N/m |")
    A(f"| exact jump `sigma/R` | {res['rows'][0]['mixture']['dp_theory_Pa']} Pa |")
    A(f"| kernel | cubic spline, `h = {s['h_factor']} dx` |")
    A(f"| lattice | square, mass `m_i = rho0_phase dx^2` (volume-consistent) |")
    dxs_mm = ", ".join(f"{d*1e3:g} mm" for d in dxs)
    rdx = ", ".join(f"{s['radius_m']/d:g}" for d in dxs)
    A(f"| resolutions | `dx` = {dxs_mm} (`R/dx` = {rdx}) |")
    A(f"| rest-density cases | " + "; ".join(
        f"`{c['label']}`: rho_A = {c['rho_in']}, rho_B = {c['rho_out']}" for c in s["rho_cases"])
      + " |")
    A(f"| curvature | {s['curvature']} |")
    A("")
    A("## 2. Method")
    A("")
    A("Two arms are compared on **identical particles and identical operators**; "
      "only the colour function and the delta-function rule differ, which is the "
      "ablation that isolates the mixture-material question.")
    A("")
    A("`mixture` arm (`tensorsph.csf_mixture`, new):")
    A("")
    A("* per-phase volume `V_j = m_j / rho0_phase(j)`,")
    A("* normalised phase-indicator colour "
      "`c^A_i = sum_{j in A} V_j W_ij / sum_j V_j W_ij` -> `c = 1` in the bulk of "
      "A, `c = 0` in the bulk of B, for **any** density ratio;")
    A("* `delta_s = |grad c^A|` (no factor 2: the normalised colour completes the "
      "full 1 -> 0 transition, unlike the one-sided free-surface rule);")
    A("* `n = -grad c/|grad c|`, `kappa = div n` (MLS), force assembled as a "
      "**volume force density** `f = -sigma kappa delta_s n`; the resultant is "
      "`sum_i f_i V_i`, so the momentum bookkeeping does not depend on which "
      "phase a particle belongs to.")
    A("")
    A("`legacy` arm (shipped): `c = sum_j (m_j/rho0) W_ij` with one scalar "
      f"`rho0 = {s['legacy_rho0']}` (`tensorsph.csf_consistent.kernel_sum_color`), "
      "`delta_s = 2 |grad c|`, same normal smoothing and same `div n` curvature.")
    A("")
    A("Measured quantities, per configuration:")
    A("")
    A("* `sigma_eff/sigma = |sum_{x>0} f V| / (2 sigma)`, exact value 1: for a "
      "circular interface the resultant of `-sigma kappa delta_s n` over the half "
      "plane is `sigma kappa * int_arc n dl = 2 sigma`;")
    A("* `sum_i delta_s_i V_i / (2 pi R)`, exact value 1 (Brackbill identity);")
    A("* shell thickness of the force-carrying band, in `dx`;")
    A("* curvature error of the shell mean vs `1/R`;")
    A("* relative momentum residual `||sum_i f_i V_i|| / sum_i ||f_i|| V_i` on a "
      "cloud whose centre is **off** the lattice (`+0.37 dx, +0.21 dx`), so the "
      "cancellation is not lattice symmetry.")
    A("")
    A("## 3. Results")
    A("")
    A("### 3.1 Resolution and density-ratio ladder")
    A("")
    A("| case | dx [mm] | N | `sigma_eff/sigma` mixture | err [%] | int delta_s | "
      "kappa err [%] | `sigma_eff/sigma` legacy | err [%] | int delta_s | kappa err [%] |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in cases:
        for dx in dxs:
            r = get(case, dx)
            m, l = r["mixture"], r["legacy"]
            A(f"| {case} | {dx*1e3:g} | {r['n_particles']} | "
              f"{m['sigma_eff_over_sigma']:.4f} | {m['laplace_error_pct']:+.1f} | "
              f"{m['delta_integral_over_2piR']:.3f} | {m['kappa_error_pct']:+.1f} | "
              f"{l['sigma_eff_over_sigma']:.4f} | {l['laplace_error_pct']:+.1f} | "
              f"{l['delta_integral_over_2piR']:.3f} | {l['kappa_error_pct']:+.1f} |")
    A("")
    A("### 3.2 Density-ratio invariance")
    A("")
    A("| quantity | mixture arm | legacy arm |")
    A("|---|---|---|")
    A(f"| spread of `sigma_eff/sigma` over the three density cases at the finest "
      f"resolution | {summ['mixture_density_ratio_spread']:.2e} | "
      f"{summ['legacy_density_ratio_spread']:.4f} |")
    A(f"| finest `sigma_eff/sigma`, control | "
      f"{summ['mixture_finest']['control']:.4f} | {summ['legacy_finest']['control']:.4f} |")
    A(f"| finest `sigma_eff/sigma`, water/air | "
      f"{summ['mixture_finest']['water-air']:.4f} | {summ['legacy_finest']['water-air']:.4f} |")
    A(f"| finest `sigma_eff/sigma`, air/water | "
      f"{summ['mixture_finest']['air-water']:.4f} | {summ['legacy_finest']['air-water']:.4f} |")
    A(f"| within 15 % at the finest resolution (all three cases) | "
      f"{all(summ['mixture_within_15pct'].values())} | "
      f"{all(summ['legacy_within_15pct'].values())} |")
    A("")
    mom = [r["momentum_residual_offlattice"] for r in rows]
    A(f"Momentum residual of the closed interface, off-lattice centre: max over "
      f"all configurations = {max(mom):.2e} (relative).")
    A("")
    A("### 3.3 Curvature-operator ablation (same colour field, `R/dx = "
      f"{abl['R_over_dx']:g}`)")
    A("")
    A("| curvature | stencil [h] | kappa shell mean | kappa err [%] | `sigma_eff/sigma` |")
    A("|---|---|---|---|---|")
    for r in abl["rows"]:
        st = "-" if r["stencil_h"] is None else f"{r['stencil_h']:g}"
        A(f"| {r['method']} | {st} | {r['kappa_shell_mean']:.2f} | "
          f"{r['kappa_error_pct']:+.1f} | {r['sigma_eff_over_sigma']:.3f} |")
    A("")
    A(f"`div n` (library default) is the accurate operator here "
      f"({abl['rows'][0]['kappa_error_pct']:+.1f} %); the local interface fits of "
      "`csf_curvature.fit_curvature` degrade badly on this band and were therefore "
      "**not** used, rather than tuned.")
    A("")
    A("![mixture surface tension](verification_mixture_surface_tension.png)")
    A("")
    A("Left: `sigma_eff/sigma` vs `R/dx` for both arms and the three density "
      "cases (solid = mixture, dashed = legacy, grey = 15 % band). Middle: "
      "Brackbill delta-function mass `sum delta_s V /(2 pi R)`. Right: colour "
      "profile across the interface at `R/dx = 16`.")
    A("")
    A("## 4. Error judgement")
    A("")
    for case in cases:
        r = get(case, finest)
        m = r["mixture"]
        three = abs(m["laplace_error_pct"]) < 3.0
        fifteen = abs(m["laplace_error_pct"]) < 15.0
        A(f"* **{case}**, finest (`R/dx = {r['R_over_dx']:g}`): mixture error "
          f"{m['laplace_error_pct']:+.2f} % -> <3 %? **{'yes' if three else 'no'}**; "
          f"<15 %? **{'yes' if fifteen else 'no'}**.")
    r0 = get(cases[0], max(dxs))
    A(f"* coarsest (`R/dx = {r0['R_over_dx']:g}`), mixture: "
      f"{r0['mixture']['laplace_error_pct']:+.2f} % -> <15 %? "
      f"**{'yes' if abs(r0['mixture']['laplace_error_pct']) < 15 else 'no'}**, "
      f"<3 %? **{'yes' if abs(r0['mixture']['laplace_error_pct']) < 3 else 'no'}**.")
    lr = get(cases[1], min(dxs))["legacy"]
    A(f"* legacy arm, water/air, finest: {lr['laplace_error_pct']:+.2f} % -> "
      "**FAIL**, and the error does not decrease under refinement, so it is an "
      "estimator defect, not truncation.")
    A("")
    A("## 5. Caveats and scope")
    A("")
    A("* This verifies the **static equilibrium** (Laplace law) and the estimator "
      "properties (colour invariance, delta mass, zero resultant). It does not "
      "exercise a time-stepping solver, capillary waves or the Rayleigh-Plateau "
      "instability; those have their own harnesses in this repository.")
    A("* `tensorsph.csf_mixture` is a **new module**; the pipeline in "
      "`droplet_impact.py` still calls the single-material chain. Nothing in the "
      "shipped solver changed, so no existing number moves.")
    A("* The legacy arm is driven through the same `div n` curvature, so its "
      "failure is attributable to the colour function and the `2|grad c|` rule, "
      "not to the curvature operator. Part of its error is also the *free-surface* "
      "truncation of `c = sum V W` at the outer edge of the cloud "
      "(colour jump "
      f"{get('control', finest)['legacy']['color_jump']:.3f} instead of 1).")
    A("* `fit_curvature` (local fits) is inaccurate on this band — non-monotone "
      "in the stencil radius, with the same value for `stencil = 4` and `5`. That "
      "is a pre-existing defect of `csf_curvature`, outside the scope of this "
      "task; recorded as a follow-up below.")
    A("* 2-D only. The 3-D counterpart (`tensorsph.threed.csf3`) has its own chain "
      "and was not touched.")
    A("")
    A("## 6. Follow-up")
    A("")
    A("* Route `droplet_impact` / `threed.solver` through the mixture colour so "
      "that multi-material cases stop using a single `rho0`.")
    A("* Repair `csf_curvature.fit_curvature`: measure the stencil behaviour on an "
      "analytic circle with an analytic normal and fix the non-monotone response.")
    A("")
    (DOC.parent / "verification_mixture_surface_tension.png").write_bytes(
        (HERE / "figure.png").read_bytes())
    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC} ({len(lines)} lines)")


if __name__ == "__main__":
    main()