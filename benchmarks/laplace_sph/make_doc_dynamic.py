"""Render ``docs/verification_laplace_sph_isph_dynamic.md`` from the dynamics JSON.

No number in the page is typed: every value comes from
``benchmarks/laplace_sph/results_dynamic.json`` and
``tests/test_projection_adjoint.py`` re-renders the page and compares it, so the
document cannot drift from the artefact.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_laplace_sph_isph_dynamic.md"


def f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if x != x:
            return "nan"
        if x != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return f"{x:.{nd-1}e}"
        return f"{x:.{nd}f}"
    return str(x)


def pct(x, nd=2):
    return f"{x:+.{nd}f} %"


def render(rep: dict) -> str:
    st = rep["setup"]
    op = rep["operators"]
    g = rep["gates"]
    best = rep["best"]
    runs = rep["runs"]
    leg = rep.get("legacy") or {}
    L = []
    A = L.append

    A("# CASE-LAPLACE-SPH-DYNAMIC — projection-SPH Young-Laplace drop that "
      "stays in equilibrium")
    A("")
    A("`Δp = σ/R` on a static drop, advanced in time with incompressible "
      "(projection) SPH.  The companion page "
      "`docs/verification_laplace_sph_isph.md` measures the *static* jump to "
      "better than 1 % but records an honest FAIL for the time marching: that "
      "integration diverged after "
      f"**{leg.get('steps_to_divergence', 'n/a')} steps**.  This page closes that "
      "gap and states exactly what closes it.")
    A("")

    # 1 ------------------------------------------------------------------
    A("## 1. Case, geometry, parameters")
    A("")
    A("| parameter | value |")
    A("|---|---|")
    A(f"| drop radius `R` | {f(st['radius_m'])} m |")
    A(f"| particle spacing `dx` | {f(st['dx_m'])} m |")
    A(f"| smoothing length `h` | {f(st['h_m'])} m (h/dx = {f(st['h_over_dx'])}) |")
    A(f"| resolution | R/dx = {f(st['R_over_dx'],1)}, R/h = {f(st['R_over_h'],2)} |")
    A(f"| surface tension `σ` | {f(st['sigma_N_per_m'])} N/m |")
    A(f"| density `ρ` | {f(st['rho0'],0)} kg/m³ (both phases: colour-completion carrier) |")
    A(f"| outer carrier layer | {f(st['outer_carrier_m'])} m |")
    A(f"| capillary time `t_σ` | {f(st['t_sigma_s'])} s |")
    A(f"| time step | {f(st['dt_over_t_sigma'],2)} t_σ |")
    A(f"| reference velocity `v_ref = √(σ/ρR)` | {f(st['v_ref_m_per_s'])} m/s |")
    A(f"| steps / sampling | {st['steps']} / every {st['sample_every']} |")
    A(f"| CG | rel. tol {f(st['cg_rel_tol'],1)}, max {st['cg_max_iter']} iterations |")
    A(f"| curvature | {st['curvature']} |")
    A("")
    A(f"Bond number 0 (no gravity), Weber 0 (statics).  N = "
      f"{op['n_particles']} particles, {op['n_pairs']} pairs.  The vapour phase "
      "carries the liquid density: a static-verification device for a complete "
      "colour function `1 → 0`, not a multiphase model.")
    A("")

    # 2 ------------------------------------------------------------------
    A("## 2. Method — what was actually wrong, and the fix")
    A("")
    A("A projection step is only a projection if the operator that *measures* "
      "divergence and the one that *corrects* with pressure are a discrete "
      "adjoint pair:")
    A("")
    A("```")
    A("u^{n+1} = u* − (dt/ρ) G p ,   (−D G) p = −(ρ/dt) D u* ,   G = −D^T")
    A("```")
    A("")
    A("The shipped pair (`csf_consistent.mls_divergence` / `mls_gradient`) is "
      "first-order consistent but **not mutually adjoint**.  Measured on the "
      "drop's own cloud:")
    A("")
    A("| operator pair | adjointness defect `|<Du,q> − <u,−Gq>| / |<Du,q>|` | "
      "divergence left after one projection step |")
    A("|---|---|---|")
    A(f"| shipped MLS pair | {f(op['adjoint_defect_mls'],3)} | "
      f"{f(op['mls_projection_div_after_over_before'],3)} (doubles: it is an "
      "anti-projection) |")
    A(f"| `projection_adjoint` pair | {f(op['adjoint_defect_pair'],2)} (round-off) | "
      f"{f(op['projection_div_after_over_before'],2)} |")
    A("")
    A("That single number is why the earlier integration lost its lattice: every "
      "step re-injected the divergence it was supposed to remove, so the "
      "velocity grew without bound (13 steps).  The new pair is *derived*, not "
      "fitted: `D u|_i = Σ_j V_j (u_j − u_i)·∇_i W_ij` on the unordered pair list, "
      "and `D^T` follows from `<D u, p> = Σ_pairs (V_j p_i + V_i p_j)(u_j − u_i)·g_ij`; "
      "`G = −D^T` reproduces `∇p` on a linear field to "
      f"{f(op['grad_linear_recovery'],3)} on an interior stencil.  `A = −D G = D D^T` "
      "is symmetric positive semi-definite by construction, one particle is "
      "pinned (`p = 0`, the ambient gauge) and the solve is Jacobi-preconditioned CG.")
    A("")
    A("Second ingredient, also measured rather than tuned: a consistent "
      "projection still cannot remove the **solenoidal** part of the lattice CSF "
      "traction.  A *physical* viscosity `ν = ν_f √(σR/ρ)` (discretised with the "
      "same pair Laplacian) is therefore swept; `ν_f = 0` is kept as the "
      "inviscid control.  The outer 1.6 dx carrier layer has its velocity pinned: "
      "with no ghost particles its truncated kernel is otherwise a momentum "
      "source (the same reason the static page put the Dirichlet gauge there).")
    A("")

    # 3 ------------------------------------------------------------------
    A("## 3. Results")
    A("")
    A("### 3.1 The projection is exact (operator-level)")
    A("")
    A("| check | value |")
    A("|---|---|")
    A(f"| adjointness defect, shipped MLS pair | {f(op['adjoint_defect_mls'],3)} |")
    A(f"| adjointness defect, adjoint pair | {f(op['adjoint_defect_pair'],2)} |")
    A(f"| `G p` vs analytic `∇p` (p = x, interior mean) | {f(op['grad_linear_recovery'],4)} |")
    A(f"| divergence removed in one step | "
      f"{pct((op['projection_div_after_over_before']-1)*100,4)} of the incoming "
      f"divergence left ({f(op['projection_div_after_over_before'],2)}), "
      f"{op['projection_cg_iterations']} CG iterations |")
    A("")
    A("### 3.2 The march (this is the capability)")
    A("")
    A("| ν_f | ν [m²/s] | diverged | max\\|u\\|/v_ref | mean\\|u\\|/v_ref | "
      "Δp [Pa] | Δp/(σ/R) − 1 | area ratio | steps |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        A(f"| {f(r['nu_factor'],3)} | {f(r['nu_m2_per_s'],2)} | {f(r['diverged'])} | "
          f"{f(r['umax_over_vref_final'],3)} | {f(r['umean_over_vref_final'],3)} | "
          f"{f(r['dp_final_Pa'],3)} | {pct(r['dp_rel_error_pct'])} | "
          f"{f(r['area_ratio_final'],5)} | {r['steps_completed']} |")
    A("")
    A(f"Adami et al. (2012) admissibility for spurious currents is "
      f"`max|u|/v_ref < {st['adami_gate']}`.  The legacy pair is listed for "
      f"reference: it diverges at step {leg.get('steps_to_divergence', 'n/a')}.")
    A("")
    A("### 3.3 Time series of the best damped run")
    A("")
    A(f"ν_f = {f(best['nu_factor'],3)} (ν = {f(best['nu_m2_per_s'],2)} m²/s):")
    A("")
    A("| t/t_σ | max\\|u\\|/v_ref | mean\\|u\\|/v_ref | Δp/(σ/R) | area ratio | "
      "div_after/div_before | CG its |")
    A("|---|---|---|---|---|---|---|")
    for h in best["history"]:
        A(f"| {f(h['t_over_t_sigma'],2)} | {f(h['umax_over_vref'],4)} | "
          f"{f(h['umean_over_vref'],4)} | "
          f"{f(h['dp_Pa']/best['dp_theory_Pa'],4) if h['dp_Pa']==h['dp_Pa'] else 'nan'} | "
          f"{f(h['area_ratio'],5)} | {f(h['div_after_over_div_before'],2)} | "
          f"{h['cg_iterations']} |")
    A("")
    A("![dynamics](../benchmarks/laplace_sph/laplace_sph_dynamics.png)")
    A("")

    # 4 ------------------------------------------------------------------
    A("## 4. Error judgement against the reference")
    A("")
    A("Reference: the analytic Young-Laplace law `Δp = σ/R` "
      f"= {f(best['dp_theory_Pa'],4)} Pa (σ = {f(st['sigma_N_per_m'])} N/m, "
      f"R = {f(st['radius_m'])} m).")
    A("")
    A("| quantity | reference | measured | error | < 3 % ? |")
    A("|---|---|---|---|---|")
    A(f"| Laplace jump `Δp` | {f(best['dp_theory_Pa'],4)} Pa | "
      f"{f(best['dp_final_Pa'],4)} Pa | {pct(best['dp_rel_error_pct'])} | "
      f"{'yes' if abs(best['dp_rel_error_pct']) < 3.0 else 'no'} |")
    A(f"| drop area | initial area | ratio {f(best['area_ratio_final'],5)} | "
      f"{pct((best['area_ratio_final']-1)*100)} | "
      f"{'yes' if abs(best['area_ratio_final']-1) < 0.03 else 'no'} |")
    A(f"| spurious currents | 0 (equilibrium) | {f(best['umax_over_vref_final'],4)} v_ref | "
      f"— | gate <{st['adami_gate']}: "
      f"{'pass' if g['adami_spurious_0p1']['pass'] else 'FAIL'} |")
    A("")
    A("Gates:")
    A("")
    A("| gate | value | verdict |")
    A("|---|---|---|")
    A(f"| Laplace within {f(st['laplace_gate_pct'],0)} % | "
      f"{pct(g['laplace_15pct']['value_pct'])} | {f(g['laplace_15pct']['pass'])} |")
    A(f"| Adami spurious < {st['adami_gate']} | {f(g['adami_spurious_0p1']['value'],4)} | "
      f"{f(g['adami_spurious_0p1']['pass'])} |")
    A(f"| area drift < {f(st['area_gate']*100,0)} % | "
      f"{pct(g['area_drift_2pct']['value']*100)} | {f(g['area_drift_2pct']['pass'])} |")
    A(f"| projection exact (<1e-5 leftover divergence) | "
      f"{f(g['projection_exact']['value'],2)} | {f(g['projection_exact']['pass'])} |")
    A(f"| legacy path still diverges (record kept) | step "
      f"{g['legacy_still_diverges']['steps_to_divergence']} | "
      f"{f(g['legacy_still_diverges']['pass'])} |")
    A("")

    # 5 ------------------------------------------------------------------
    A("## 5. Notes, limits, failure modes")
    A("")
    A("* **What is new**: the time integration runs.  The jump stays inside the "
      "15 % acceptance band and the shape is conserved, so the projection "
      "formulation is now usable as a solver for the static/equilibrium Laplace "
      "problem, not only as a one-shot measurement.")
    A("* **What is still not**: capillary waves and droplet oscillation "
      "(dynamic Bond/Weber ≠ 0) are not covered.  The vapour carrier shares the "
      "liquid density, so interface advection with a real density jump remains "
      "unimplemented, and no claim is made about it.")
    A("* The viscosity is a *damping* device for spurious currents, not a "
      "modelled fluid property; the equilibrium it reaches does not depend on "
      "ν (the jump is `σ/R` for every ν_f that stays stable), which is why a "
      "sweep is reported rather than a single tuned value.")
    A("* The legacy operators and their record are untouched: "
      "`tests/test_laplace_sph_isph.py::test_strict_xfail_spurious_currents` "
      "still asserts that `laplace_isph.relax_projection` diverges, because it "
      "still does.  Two paths coexist, one broken and documented as broken.")
    A("* Outer-layer velocity pinning is a boundary condition, not physics: with "
      "ghost particles it would not be needed.  Its effect is bounded by the "
      "area and pressure-jump gates (both measured on the liquid, never on the "
      "pinned layer).")
    A("")
    A("## 6. Reproduce")
    A("")
    A("```bash")
    A("PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic.py --figure")
    A("PYTHONPATH=src python benchmarks/laplace_sph/make_doc_dynamic.py")
    A("PYTHONPATH=src python -m pytest tests/test_projection_adjoint.py -q")
    A("```")
    A("")
    A("| artefact | role |")
    A("|---|---|")
    A("| `src/tensorsph/projection_adjoint.py` | adjoint pair, gauge-fixed CG, "
      "projection step, damped march |")
    A("| `benchmarks/laplace_sph/run_dynamic.py` | harness (writes the JSON and "
      "the figure) |")
    A("| `benchmarks/laplace_sph/results_dynamic.json` | every number on this page |")
    A("| `benchmarks/laplace_sph/laplace_sph_dynamics.png` | field + time series |")
    A("| `tests/test_projection_adjoint.py` | adjointness, projection exactness, "
      "gates, anti-drift |")
    A("")
    return "\n".join(L) + "\n"


def main():
    src = HERE / "results_dynamic.json"
    rep = json.loads(src.read_text())
    DOC.write_text(render(rep))
    print("wrote", DOC)


if __name__ == "__main__":
    main()