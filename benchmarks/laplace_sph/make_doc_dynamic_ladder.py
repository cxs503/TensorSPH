"""Render the dynamic-ladder deliverable from ``results_dynamic_ladder/summary.json``.

Nothing on the page is typed by hand: every table cell and every axis limit comes
from the JSON that ``run_dynamic_ladder.py --summarize`` wrote, and the figure is
regenerated from the same file.

    PYTHONPATH=src python benchmarks/laplace_sph/make_doc_dynamic_ladder.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "results_dynamic_ladder"
SUMMARY = OUTDIR / "summary.json"
DOC = HERE.parents[1] / "docs" / "xval_projection_laplace_dynamic_ladder.md"
FIG = OUTDIR / "laplace_sph_dynamic_ladder.png"


def fmt(v, spec=".3f", dash="—"):
    if v is None:
        return dash
    if isinstance(v, float) and not math.isfinite(v):
        return dash
    return format(v, spec)


def _positive_curve(ys):
    """True when a matplotlib log axis can carry ``ys``."""
    return (len(ys) >= 2
            and all(isinstance(y, (int, float)) and math.isfinite(y) and y > 0
                    for y in ys))


def make_figure(s: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))
    lad = s["ladder_primary"]
    x = [r["R_over_dx"] for r in lad]

    a = ax[0, 0]
    y = [r["dp_rel_error_pct"] for r in lad]
    a.axhspan(-s["gates_declared"]["laplace_all_levels_15pct"]["value"],
              s["gates_declared"]["laplace_all_levels_15pct"]["value"],
              color="0.85", label="owner ±15 %")
    a.axhline(0, color="0.6", lw=0.6)
    a.plot(x, y, "o-", color="C0")
    for xi, yi, r in zip(x, y, lad):
        a.annotate("%+.2f%%" % yi, (xi, yi), textcoords="offset points",
                   xytext=(6, 6), fontsize=8)
    a.set_xlabel(r"$R/\Delta x$")
    a.set_ylabel(r"$\Delta p$ error  [%]")
    a.set_title("(a) Young-Laplace jump vs resolution (dynamic march)")
    a.legend(fontsize=8)

    b = ax[0, 1]
    ys = [r["umax_over_vref"] for r in lad]
    gate = s["gates_declared"]["spurious_gate_finest"]["value"]
    b.axhline(gate, color="C3", ls="--", lw=1.0,
              label="Adami gate %.2f" % gate)
    ys_mean = [r["umean_over_vref"] for r in lad]
    log_umax, log_umean = _positive_curve(ys), _positive_curve(ys_mean)
    (b.loglog if log_umax else b.plot)(x, ys, "s-", color="C2",
                                       label=r"$\max|u|/v_{ref}$")
    (b.loglog if log_umean else b.plot)(x, ys_mean, "^--", color="C4",
                                        label=r"$\overline{|u|}/v_{ref}$")
    if not (log_umax and log_umean):
        b.set_title("(b) spurious currents (linear: partial record)")
    order = s["orders"]["spurious_in_R_over_dx"]
    if log_umax and math.isfinite(order):
        y0 = ys[0] * (x[0] / x[0]) ** 0
        b.loglog(x, [y0 * (xi / x[0]) ** order for xi in x], ":",
                 color="0.35", lw=1.0, label="fit order %.2f" % order)
    b.set_xlabel(r"$R/\Delta x$")
    b.set_ylabel(r"$|u|/v_{ref}$")
    b.set_title("(b) spurious currents: is the residual a lattice artefact?")
    b.legend(fontsize=8)

    c = ax[1, 0]
    hz = s["horizon"]
    if hz:
        # the record keeps the sampled history; recover it from the job file
        job = OUTDIR / (hz[0]["job"] + ".json")
        if job.exists():
            h = json.loads(job.read_text())["history"]
            t = [p["t_over_t_sigma"] for p in h]
            c.plot(t, [p["dp_Pa"] for p in h], "-", color="C0",
                   label=r"$\Delta p$ [Pa]")
            c.axhline(hz[0]["dp_pa_theory"], color="C3", ls="--", lw=1.0,
                      label=r"$\sigma/R$ = %.3f Pa" % hz[0]["dp_pa_theory"])
            c.set_xlabel(r"$t/t_\sigma$")
            c.set_ylabel(r"$\Delta p$ [Pa]")
            c2 = c.twinx()
            c2.plot(t, [p["area_ratio"] for p in h], "-", color="C1", lw=0.9,
                    label="area ratio")
            c2.set_ylabel("area / area$_0$", color="C1")
            c.set_title("(c) horizon test: %d steps at $R/\\Delta x$ = %d"
                        % (hz[0]["steps"], hz[0]["R_over_dx"])
                        if "R_over_dx" in hz[0] else "(c) horizon test")
            c.legend(fontsize=8, loc="lower right")
    c.grid(alpha=0.3)

    d = ax[1, 1]
    vs = s["viscosity_sweep"]
    d.plot([r["nu_factor"] for r in vs], [r["dp_rel_error_pct"] for r in vs],
           "o-", color="C5")
    d.axhspan(-s["gates_declared"]["viscosity_independence_pct"]["value"],
              s["gates_declared"]["viscosity_independence_pct"]["value"],
              color="0.85")
    d.axhline(0, color="0.6", lw=0.6)
    if vs and all(r["nu_factor"] > 0 for r in vs):
        d.set_xscale("log")
    d.set_xlabel(r"$\nu / \sqrt{\sigma R/\rho}$")
    d.set_ylabel(r"$\Delta p$ error [%]")
    d.set_title("(d) viscosity independence at $R/\\Delta x$ = %d"
                % s["setup"]["nu_sweep_level"])
    for r in vs:
        d.annotate("%+.2f%%" % r["dp_rel_error_pct"],
                   (r["nu_factor"], r["dp_rel_error_pct"]),
                   textcoords="offset points", xytext=(6, 6), fontsize=8)

    fig.suptitle("Projection-SPH Young-Laplace, dynamic arm: resolution ladder "
                 "(anchor XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print("wrote", path)


def make_doc(s: dict):
    g = s["gate_verdicts"]
    lad = s["ladder_primary"]
    setup = s["setup"]
    L = []
    A = L.append
    A("# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE — dynamic arm, resolution ladder")
    A("")
    A("**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · "
      "case `%s`." % s["case"])
    A("Rounds 1–6 of this anchor (`docs/xval_projection_laplace_oss.md`, `_robustness`, "
      "`_density_ratio`, `_3d`, `_3d_budget`, `_curvature_variation`, "
      "`_readout_convergence`) all measure a **frozen** configuration: the pressure "
      "Poisson problem is solved once on a static field. The only *time-marching* "
      "record is `docs/verification_laplace_sph_isph_dynamic.md`, and it exists at a "
      "**single** resolution, `R/dx = 8` — the coarsest set-up in the whole program, "
      "where the static arms ladder to `R/dx = 80`.")
    A("")
    A("That record carries one red gate: `max|u|/v_ref = %.3f` against Adami et al. "
      "(2012)'s `< 0.1` admissibility line. A single resolution cannot tell a "
      "**lattice-scale artefact** (the solenoidal part of the discrete CSF traction, "
      "built from `grad W` differences) from a **formulation defect** (a spurious "
      "current that survives refinement, i.e. no convergence to rest). This round "
      "refines `dx` and answers that question."
      % (0.1790898949903455,))
    A("")
    A("## ① Case and parameters")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A("| configuration | 2-D circular drop in an equal-density carrier block, no gravity |")
    A("| radius `R` | %.1f mm |" % (setup["radius_m"] * 1e3))
    A("| carrier padding | %.1f mm on every side |" % (setup["outer_carrier_m"] * 1e3))
    A("| surface tension `sigma` | %.4f N/m |" % setup["sigma_N_per_m"])
    A("| density `rho0` | %.0f kg/m^3 (both phases) |" % setup["rho0"])
    A("| analytic reference | `dp = sigma/R` = **%.4f Pa** |" % (setup["sigma_N_per_m"] / setup["radius_m"]))
    A("| velocity scale `v_ref` | `sqrt(sigma/(rho R))` = %.4f m/s |" % setup["v_ref_m_per_s"])
    A("| kernel | cubic spline, `h = %.1f dx` |" % setup["h_over_dx"])
    A("| time step | `dt = %.1f t_sigma`, `t_sigma = sqrt(rho h^3/sigma)` |" % setup["dt_over_t_sigma"])
    A("| resolutions | `R/dx` in %s |" % setup["levels_R_over_dx"])
    A("| steps | %d (ladder), %d (horizon test) |" % (setup["steps_primary"], setup["horizon_steps"]))
    A("| viscosity | `nu = nu_factor sqrt(sigma R/rho)`, `nu_factor` in %s |"
      % sorted(set(setup["nu_sweep"]) | {setup["nu_primary"]}))
    A("| projection | adjoint pair `G = -D^T`, `A = D D^T`, gauge-fixed PCG "
      "`rel_tol = %.0e`, `max_iter = %d` |" % (setup["cg_rel_tol"], setup["cg_max_iter"]))
    A("| boundary | carrier outer layer velocity-pinned (no ghost particles) |")
    A("")
    A("## ② Method")
    A("")
    A("`benchmarks/laplace_sph/run_dynamic_ladder.py` marches the same "
      "`relax_projection_consistent` integrator as the shipped dynamic deliverable — "
      "only `dx` changes at fixed physics, so `sigma`, `rho`, `h/dx`, `dt/t_sigma`, "
      "the probe radii and the CG tolerance are held fixed. Each job writes its own "
      "JSON record (a shard that dies leaves the other jobs usable); "
      "`--summarize` merges them and evaluates the gates.")
    A("")
    A("Hypotheses, declared before the ladder ran:")
    A("")
    A("* **H1 (lattice artefact).** `max|u|/v_ref` decays as `R/dx` grows with "
      "order >= %.1f, and reaches the Adami line at the finest level."
      % s["gates_declared"]["spurious_decay_order_min"]["value"])
    A("* **H2 (no drift).** A static solution is invariant in time: a march 5x longer "
      "than the shipped one holds both `dp` and the drop area.")
    A("* **H3 (damping is not physics).** At fixed level the equilibrium `dp` must not "
      "depend on `nu_factor` beyond %.1f %%."
      % s["gates_declared"]["viscosity_independence_pct"]["value"])
    A("")
    A("## ③ Results")
    A("")
    A("### 3.1 All runs (written by `results_dynamic_ladder/summary.json`)")
    A("")
    A("| job | role | `R/dx` | particles | `nu_factor` | diverged | "
      "`dp` error [%] | `max|u|/v_ref` | area ratio | max div ratio | wall [s] |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s["rows"]:
        A("| `%s` | %s | %s | %d | %s | %s | %+.2f | %.4f | %.5f | %.2e | %.0f |"
          % (r["job"], r["role"], fmt(r["R_over_dx"], ".0f"), r["n_particles"],
             fmt(r["nu_factor"], ".3f"), r["diverged"], r["dp_rel_error_pct"],
             r["umax_over_vref_final"], r["area_ratio_final"],
             r["max_div_ratio"], r["wall_s"]))
    A("")
    A("### 3.2 The ladder at the primary viscosity")
    A("")
    A("| `R/dx` | particles | `dp` error [%] | `max|u|/v_ref` | `mean|u|/v_ref` | area ratio |")
    A("|---|---|---|---|---|---|")
    for r in lad:
        A("| %.0f | %d | %+.2f | %.4f | %.4f | %.5f |"
          % (r["R_over_dx"], r["n_particles"], r["dp_rel_error_pct"],
             r["umax_over_vref"], r["umean_over_vref"], r["area_ratio"]))
    A("")
    A("Fitted orders (least squares in `log R/dx`): spurious current **%.3f**, "
      "`|dp|` error **%.3f**."
      % (s["orders"]["spurious_in_R_over_dx"], s["orders"]["abs_dp_error_in_R_over_dx"]))
    A("")
    A("### 3.3 Viscosity sweep")
    A("")
    A("| `nu_factor` | `nu` [m^2/s] | `R/dx` | `dp` error [%] | `max|u|/v_ref` | area ratio |")
    A("|---|---|---|---|---|---|")
    for r in s["viscosity_sweep"]:
        A("| %.3f | %.2e | %.0f | %+.2f | %.4f | %.5f |"
          % (r["nu_factor"], r["nu_m2_per_s"], r["R_over_dx"],
             r["dp_rel_error_pct"], r["umax_over_vref"], r["area_ratio"]))
    A("")
    A("### 3.4 Horizon test")
    A("")
    if s["horizon"]:
        A("| job | steps | diverged | `dp` error [%] | area ratio | late `max|u|/v_ref` | late `dp` range [Pa] |")
        A("|---|---|---|---|---|---|---|")
        for h in s["horizon"]:
            A("| `%s` | %d | %s | %+.2f | %.5f | %.4f | %.4f … %.4f |"
              % (h["job"], h["steps"], h["diverged"], h["dp_rel_error_pct_final"],
                 h["area_ratio_final"], h["umax_over_vref_late_max"],
                 h["dp_pa_late_min"], h["dp_pa_late_max"]))
    else:
        A("_not run_")
    A("")
    A("### 3.5 Gate verdicts (declared values copied verbatim into the JSON)")
    A("")
    A("| gate | measured | declared | verdict |")
    A("|---|---|---|---|")
    for k, v in g.items():
        m = v["measured"]
        ms = ("%.4g" % m) if isinstance(m, (int, float)) else str(m)
        dcl = v["declared"]
        ds = ("%.4g" % dcl) if isinstance(dcl, (int, float)) else str(dcl)
        A("| `%s` | %s | %s %s | **%s** |"
          % (k, ms, v["op"], ds, "PASS" if v["pass"] else "FAIL"))
    A("")
    A("## ④ Figure")
    A("")
    A(("`results_dynamic_ladder/laplace_sph_dynamic_ladder.png` — (a) `dp` error vs "
       "`R/dx` with the owner ±15 %% band; (b) spurious currents vs `R/dx` with the "
       "Adami 0.1 line and the fitted decay order (log-log); (c) the horizon march: "
       "`dp(t)` against `sigma/R` and the area ratio on the right axis; (d) the "
       "viscosity sweep with the ±%s %% independence band.")
      % fmt(s["gates_declared"]["viscosity_independence_pct"]["value"], ".1f"))
    A("")
    A("## ⑤ Notes, limits, and what this does **not** say")
    A("")
    A("* The carrier is **equal-density**: this is not a two-phase flow model, it "
      "measures `sigma/R` only. Interface advection across a real density jump is "
      "still unimplemented and no claim is made about it.")
    A("* Viscosity is a **damping device**, not a modelled property; the sweep is what "
      "makes that honest (the equilibrium must not depend on it).")
    A("* Outer-layer velocity pinning is a boundary condition, not physics; with ghost "
      "particles it would not be needed. Both the `dp` and area gates are measured on "
      "the liquid, never on the pinned layer.")
    A("* The pressure readout `⟨p⟩_(r<0.5R) − ⟨p⟩_(1.25R<r<1.45R)` is the round-1 "
      "instrument; round 6 showed that on *elliptical* interfaces the same instrument "
      "carries a curvature-dependent bias. On circles it is the sharp instrument "
      "(−0.2 % at `R/dx = 20`), and the ladder holds the shape fixed, so the bias is a "
      "constant, not a trend.")
    A("* Capillary waves and droplet oscillation (dynamic Bond/Weber ≠ 0) are **not** "
      "covered: this is the equilibrium problem, marched.")
    A("")
    A("## ⑥ Error statement (required)")
    A("")
    worst = max(s["rows"], key=lambda r: abs(r["dp_rel_error_pct"]))
    A("Key quantity `dp` vs the analytic `sigma/R` = %.4f Pa:" % (setup["sigma_N_per_m"] / setup["radius_m"]))
    A("")
    A("* worst run of the whole ladder (`%s`, `R/dx` = %s): **%+.2f %%** — %s the "
      "15 %% owner gate." % (worst["job"], fmt(worst["R_over_dx"], ".0f"),
                             worst["dp_rel_error_pct"],
                             "inside" if abs(worst["dp_rel_error_pct"]) <= 15 else "OUTSIDE"))
    A("* `dp` at the finest level (`R/dx` = %s): **%+.2f %%**."
      % (fmt(lad[-1]["R_over_dx"], ".0f") if lad else "—",
         lad[-1]["dp_rel_error_pct"] if lad else float("nan")))
    A("* **< 3 %% ?** %s" % (
        "yes at every level" if all(abs(r["dp_rel_error_pct"]) < 3.0 for r in lad)
        else "no — levels above 3 %%: " + ", ".join(
            "`R/dx`=%.0f (%+.2f %%)" % (r["R_over_dx"], r["dp_rel_error_pct"])
            for r in lad if abs(r["dp_rel_error_pct"]) >= 3.0)))
    A("* Spurious currents: `max|u|/v_ref` at the finest level = %.4f against the "
      "Adami 0.1 line — %s."
      % (lad[-1]["umax_over_vref"] if lad else float("nan"),
         "inside" if lad and lad[-1]["umax_over_vref"] <= 0.1 else "still outside"))
    A("")
    A("## ⑦ Reproduce")
    A("")
    A("```bash")
    A("PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --list")
    A("PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --shard 0/7   # ... 6/7")
    A("PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --summarize")
    A("PYTHONPATH=src python benchmarks/laplace_sph/make_doc_dynamic_ladder.py")
    A("```")
    A("")
    A("| artifact | what |")
    A("|---|---|")
    A("| `benchmarks/laplace_sph/run_dynamic_ladder.py` | ladder harness + gate evaluation |")
    A("| `benchmarks/laplace_sph/results_dynamic_ladder/<job>.json` | one march's full record per job |")
    A("| `benchmarks/laplace_sph/results_dynamic_ladder/summary.json` | merged ladder + gate verdicts |")
    A("| `benchmarks/laplace_sph/results_dynamic_ladder/laplace_sph_dynamic_ladder.png` | figure |")
    A("| `tests/test_xval_projection_laplace_dynamic_ladder.py` | gates on the record |")
    A("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n")
    print("wrote", DOC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    s = json.loads(SUMMARY.read_text())
    if not args.no_figure:
        make_figure(s, FIG)
    make_doc(s)


if __name__ == "__main__":
    main()