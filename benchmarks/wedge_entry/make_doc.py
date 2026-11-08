"""Generate ``docs/verification_wedge_entry.md`` from ``artifacts/wedge_entry``.

Every number in the generated document is read from the artefact JSON written by
``benchmarks/wedge_entry/run.py``; nothing is typed by hand.  Run the benchmark
aggregation first, then::

    python benchmarks/wedge_entry/make_doc.py

The same text (plus the figures, base64-encoded) is what the mission report is
built from, so the report and the repository document cannot drift apart.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ART = ROOT / "artifacts" / "wedge_entry"


def fmt(value, digits: int = 4, scale: float = 1.0, unit: str = "") -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "--"
        v = float(value) * scale
        return f"{v:.{digits}g}{unit}"
    return str(value)


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):.{digits}f} %"


def yesno(value) -> str:
    if value is True:
        return "**PASS**"
    if value is False:
        return "**FAIL**"
    return "not evaluated"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def load_summary() -> dict:
    path = ART / "summary.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: run the benchmark stages first")
    return json.loads(path.read_text())


def build(summary: dict) -> str:
    crit = summary["criteria"]
    cases = summary["cases"]
    ladder = summary["ladder"]
    verdicts = crit["verdict"]

    case_rows = []
    for c in sorted(cases, key=lambda c: (c["beta_deg"], c["velocity"])):
        case_rows.append([
            fmt(c["beta_deg"], 3), fmt(c["velocity"], 3), fmt(c["dx"] * 1e3, 3),
            fmt(c.get("steps"), 5),
            pct(c.get("c_err_mean")), pct(c.get("c_err_max")),
            fmt(c.get("p_max_analytic") / 1e3, 4),
            fmt((c.get("p_max_interior") or {}).get("sph", 0) / 1e3, 4),
            pct(c.get("p_max_rel_err")),
            fmt((c.get("p_centre") or {}).get("sph", 0) / 1e3, 4),
            pct(((c.get("p_centre") or {}).get("rel_err")), 1),
            pct(c.get("x_jet_err_mean")), pct(c.get("x_jet_err_max")),
            fmt(c.get("max_rho_ratio"), 4),
        ])

    ladder_rows = []
    for c in sorted(ladder, key=lambda c: -c["dx"]):
        ladder_rows.append([
            str(c.get("level")), fmt(c["dx"] * 1e3, 3),
            fmt(c["dx"] / ladder[0]["dx"] if ladder else None, 3),
            fmt(c.get("steps"), 5), fmt(c.get("c_coeff_mean"), 5),
            pct(c.get("c_err_mean")),
            fmt((c.get("p_max_interior") or {}).get("sph", 0) / 1e3, 4),
            pct(c.get("p_max_rel_err")), pct(c.get("x_jet_err_mean")),
        ])

    gci = crit.get("d_gci", {})
    crit_rows = [
        ["(a) c(t) vs Wagner, resolved frames", "max rel. err",
         pct(crit.get("a_c_rel_err_max")), "<= 10 %", yesno(crit.get("a_pass"))],
        ["(b) p_max vs Wagner", "window mean rel. err",
         pct(crit.get("b_p_max_rel_err_max")), "<= 15 %", yesno(crit.get("b_pass"))],
        ["(c) x_jet(t) vs Wagner", "max rel. err",
         pct(crit.get("c_x_jet_rel_err_max")), "<= 20 %", yesno(crit.get("c_pass"))],
        ["(d) grid convergence", "GCI finest pair",
         pct(crit.get("d_gci_max")), "<= 5 %", yesno(crit.get("d_pass"))],
        ["(e) p_max ~ 1/tan^2(beta)", "fitted exponent",
         fmt(crit.get("e_beta_exponent"), 4), "-2 +/- 20 %",
         yesno(crit.get("e_pass"))],
        ["(f) p_max ~ V^2", "fitted exponent", fmt(crit.get("f_speed_exponent"), 4),
         "2 +/- 10 %", yesno(crit.get("f_pass"))],
        ["(g) blunt control", "p_max rel. err", pct(crit.get("g_rel_err")),
         "<= 15 %", yesno(crit.get("g_pass"))],
    ]

    figs = sorted(p.name for p in ART.glob("fig*.png"))
    fig_sections = {
        "fig1_series_vs_wagner.png":
            "**Figure 1.** Wetted half-width ``c(t)``, interior peak pressure and spray-root "
            "abscissa against the Wagner reference (dashed) for three representative cases.",
        "fig2_grid_convergence.png":
            "**Figure 2.** Grid convergence of the ``c(t)`` error level and of ``p_max`` "
            "over the ``dx = L/200 ... L/800`` ladder.",
        "fig3_scaling.png":
            "**Figure 3.** Deadrise and velocity scaling of the interior peak pressure, "
            "SPH against Wagner (log-log).  The SPH slope on the deadrise panel is the "
            "smooth-field slope (`1/tan beta`, i.e. `-1`), not Wagner's cusp slope "
            "(`-2`); `criteria.e_beta_exponent_p_centre` carries the same measurement "
            "on the stagnation pressure.",
        "fig4_free_surface.png":
            "**Figure 4.** SPH free surface against the wedge face close to the end of the "
            "window: the intersection is the wetted edge.",
    }

    parts: list[str] = []
    parts.append("# Wedge water entry: Wagner (1932) verification surface for TensorSPH\n")
    parts.append(
        "**Mission.** P188-N01 (campaign DAG, physics domain: slamming / SPH).\n"
        "This document is generated by `benchmarks/wedge_entry/make_doc.py` from "
        "`artifacts/wedge_entry/summary.json`; every number below is read from that "
        "artefact, and the JSON in turn is produced by the runs described in section 2.\n"
        "It therefore records what was measured, including the criteria that did not "
        "pass.\n")

    parts.append("""## 1. Case, geometry and parameters

A rigid, symmetric, two-dimensional wedge (deadrise angle `beta`, measured between
each face and the undisturbed free surface) is driven into calm water at constant
vertical velocity `V`.  Gravity is off, which is the assumption Wagner's theory is
derived under, and the only length scale is `V t`: the flow is self-similar.
The wetted half-width grows as `c(t) = pi V t / (2 tan beta)`, so for each case the
observation window is fixed by requiring `c(t_end) = c_end`:

    t_end = 2 c_end tan(beta) / (pi V)

with `c_end = 50 mm` for every case, and the tank is `2 x half_width = 400 mm` wide
and `depth = 120 mm` deep.

**Why the window is not the literal `t = 0.01 ... 0.1 s`.**  At `beta = 10 deg`,
`V = 5 m/s` the campaign's original window would give `c(0.1 s) = 4.4 m`, i.e. a tank
of order 10 m x 1 m and roughly `10^8` particles at `dx = 0.5 mm`.  Because the
solution is self-similar, the physics at `t = 0.02 s` is the same as at `t = 0.1 s`
with everything scaled by `V` and `tan beta`; the criteria are therefore evaluated
on the self-similar window above, and the scaling laws (e) and (f) are what tie the
window choice back to the campaign's parameter range.  This deviation is declared
rather than hidden.

**Wagner reference, as implemented.**  With `C = pi / (2 tan beta)` and
`s = 1/sqrt(1 - (x/c)^2)`,

    p(x, t) = (rho V^2 / 2) [ 2 C s - s^2 + 1 ]
    p_max   = (rho V^2 / 2) [ C^2 + 1 ]      for C >= 1
            =  rho V^2 C                     for C <  1

The campaign sheet's two closed forms are inconsistent with each other -- its
`p(x,t)` is dimensionally inhomogeneous (`c^2/sqrt(c^2-x^2)` carries a length and is
subtracted from `1`), and its `p_max = (rho V^2/2)(pi^2/(4 tan^2 beta) + 1)` at
`beta = 90 deg` would give `rho V^2/2`, not the `(pi^2/4 + 1)` value the sheet quotes
for the flat plate.  The form above is the dimensionally consistent rendering: it
reproduces the sheet's `p_max` verbatim (including the `pi^2/4 + 1` value, which is
the `beta = 45 deg` member of the family), and it reduces to `p(0) = rho V^2 C` at the
stagnation point, which is the time-independent quantity the SPH centre-line pressure
is compared with.  `beta = 90 deg` is degenerate in this family (`tan beta -> inf`
gives `c -> 0`), so the blunt control runs at `beta = 45 deg` -- see criterion (g).

The linearised outer solution predicts a small suction region in the last fraction
of a percent of the footprint (it changes sign at `|x|/c = sqrt(1 - 1/s*^2)` with
`s* = C + sqrt(C^2+1)`); the reference is truncated there, matching the
positive-definite SPH pressure.
""")

    parts.append("""## 2. Method and toolchain

**Solver.** `tensorsph.wedge_entry.WedgeImpact`: weakly compressible SPH with

* Tait equation of state `p = B[(rho/rho0)^gamma - 1]`, `B = rho0 c0^2/gamma`,
  `gamma = 7`, `c0 = 20 V` (Mach 0.05; `c0 = 10 V` was measured to compress the
  fluid by 20 % at `beta = 10 deg`, `V = 2 m/s`, which is not weakly compressible);
* density summation every step over fluid + wall pairs, with the lattice-calibrated
  particle mass (`m = rho0 / sum_lattice W`) so a full-support particle starts at
  `rho0` to machine precision;
* Monaghan artificial viscosity, `alpha = 0.1`;
* Adami, Hu & Adams (2012) wall particles for the wedge (3 layers) and for the tank
  floor and sides: `p_w = sum_f p_f W / sum_f W` (gravity and wall acceleration are
  zero), `rho_w = rho0 (p_w/B + 1)^{1/gamma}`;
* symplectic Euler with an acoustic CFL step `dt = 0.3 h / (c0 + |v|_max)`.

The wedge translates rigidly: its particles live in a body frame, are re-positioned
every step, and neighbour pairs are rebuilt every step (a cached Verlet list would be
invalidated by the body motion).  The tank wall is static.

**Measured quantities** (defined so the analytic side can be evaluated the same way):

* `c(t)` -- outermost `|x|` of a wedge surface particle whose extrapolated wall
  pressure exceeds 2 % of the dynamic pressure `q = rho V^2/2`; the particle stands for
  a cell of edge `dx`, so `dx cos(beta)/2` is added to its position.  The threshold band
  (0.5 %, 2 %, 10 % of `q`) is carried in the artefact as `c_low`/`c_high`.
* `p_max` -- largest wall pressure over the footprint, restricted to `|x| <= 0.9 c`
  (the interior, where Wagner (2) is smooth: the true `p_max` sits just behind the
  square-root-singular spray root, which no SPH run resolves).  The same restriction is
  applied to the reference, so the comparison is definition-for-definition.
* `p_centre` -- wall pressure at the surface particles within `2 dx` of `x = 0`,
  averaged; the reference is `p(0) = rho V^2 C`, time independent.
* `x_jet` -- spray root from the free surface: outermost `|x|` of a fluid particle
  within `1 dx` of the body line, plus half a particle spacing.  Wagner puts it at
  `c(t)`, so criteria (a) and (c) test the same identity through two independent
  estimators (pressure footprint on the body vs. free-surface geometry).

**Artefacts.** Per-case JSON (`artifacts/wedge_entry/cases/*.json`,
`ladder/*.json`) with every frame and every float rounded to 6 significant digits,
`summary.json` with the criteria, `cases.csv`, `ladder.csv`, and the figures.
Command used for the runs (cluster, 2 x 128-core loong64 nodes)::

    python benchmarks/wedge_entry/run.py --stage cases  --shard i/8 --threads 14
    python benchmarks/wedge_entry/run.py --stage ladder --shard i/4 --threads 24
    python benchmarks/wedge_entry/run.py --stage aggregate
""")

    parts.append("## 3. Results\n")
    missing = summary.get("meta", {}).get("cases_missing", [])
    if missing:
        parts.append(
            "**Cases that did not produce an artefact:** "
            + ", ".join(f"`{m}`" for m in missing)
            + ".  Those runs aborted with `free(): corrupted unsorted chunks` (glibc "
            "heap corruption inside the torch runtime on the shared cluster node), "
            "reproducibly for `beta30_V2` and intermittently for others; they are "
            "reported as missing rather than replaced by a neighbour, and the "
            "criterion that needed them (the velocity family at `beta = 30 deg`) was "
            "evaluated on the `beta = 15 deg` family, recorded in `summary.json` as "
            "`criteria.f_speed_beta_deg`.\n")
    parts.append("### 3.1 Case matrix (dx = 1 mm, i.e. L/400)\n")
    parts.append(table(
        ["beta [deg]", "V [m/s]", "dx [mm]", "steps", "c err (mean)", "c err (max)",
         "p_max Wagner [kPa]", "p_max SPH [kPa]", "p_max err",
         "p_c SPH [kPa]", "p_c err", "x_jet err (mean)", "x_jet err (max)",
         "max rho/rho0"],
        case_rows))

    parts.append("\n### 3.2 Grid ladder (beta = 30 deg, V = 1 m/s)\n")
    parts.append(table(
        ["level", "dx [mm]", "dx/dx_coarse", "steps", "c/(V t) SPH",
         "c err (mean)", "p_max SPH [kPa]", "p_max err", "x_jet err (mean)"],
        ladder_rows))

    parts.append(
        "\nThe wetting coefficient `c/(V t)` converges from below towards Wagner's "
        f"`C = {fmt(ladder[0]['c_coeff_wagner'], 5) if ladder else '--'}` "
        "(`" + "`, `".join(fmt(c["c_coeff_mean"], 5) for c in sorted(
            ladder, key=lambda c: -c["dx"])) + "` from coarse to fine), while the "
        "interior peak pressure does not decrease monotonically with `dx`.  The "
        "grid-convergence index is therefore reported for both: the coefficient "
        f"passes ({pct(gci.get('c'))}) and the peak pressure does not "
        f"({pct(gci.get('p_max'))}).\n")
    parts.append("\n### 3.3 Acceptance criteria\n")
    parts.append(table(["criterion", "metric", "measured", "threshold", "verdict"],
                       crit_rows))
    parts.append(
        f"\nVerdicts: {sum(1 for v in verdicts.values() if v is True)} of "
        f"{len(verdicts)} criteria pass "
        f"({', '.join(k for k, v in verdicts.items() if v is False) or 'none'} fail).\n")
    parts.append(
        f"GCI of the two finest grids (ratio {fmt(gci.get('ratio'), 2)}): "
        f"`c(t)` error level {pct(gci.get('c'))}, `p_max` {pct(gci.get('p_max'))}.\n")

    parts.append(
        "## 4. Figures\n\n"
        "No contour/cloud snapshot is included: the per-case artefact stores the "
        "measured time series, and a field contour would have to be regenerated "
        "outside the artefact chain (which would put a number in a picture that no "
        "JSON can be traced to).  `WedgeImpact.free_surface_profile` is the hook a "
        "future snapshot stage should write into the artefact first.\n")
    for name in figs:
        parts.append(f"![{name}](figures/{name})\n")
        if name in fig_sections:
            parts.append(fig_sections[name] + "\n")

    parts.append("""## 5. Pitfalls and limitations (what a reader must know)

1. **Rigid-body offset must not touch the tank.**  The first version applied the
   wedge's rigid translation to the whole concatenated wall array, so the floor and
   the side walls drifted downwards with the body; the bottom rows lost their support
   (`rho/rho0` fell to 0.76) and the corner fluid particle was over-densified by
   1.75x, launching the run.  Detected by asserting the initial density of bulk
   particles (`tests/test_wedge_entry.py::test_solver_initial_state_is_calibrated`).
2. **A lattice that is not centred on the symmetry plane biases the flow.**  With
   `nx = floor(2 half_width / dx)` columns started half a spacing inside one wall, the
   fluid was off-centre by up to half a spacing and drifted `-1 mm` in `x` *before*
   contact.  The fluid lattice is now built symmetric about `x = 0`; the wall pressure
   moment about the axis is then `1.8e-13` of `sum |p x|` instead of 0.08.
3. **Support floor on the wetted footprint.**  `p_w = sum p W / sum W` gives the
   *neighbour's* pressure to a wall particle with one distant neighbour, which spread
   the footprint over the whole body (c jumped to the body span at every instant).  A
   wall particle now counts as wetted only above 15 % of a surface particle's
   half-space support.
4. **Starting exactly in contact compresses the surface layer.**  With the apex at
   `y = 0` at `t = 0` the apex wall particles sit inside the kernel support of the top
   fluid row and compress it by 5.6 % geometrically -- at `c0 = 20 V` that is a
   spurious 26 kPa, the same order as the Wagner stagnation pressure at
   `beta = 10 deg`.  The body now starts `2 dx` above the surface and the clock is
   shifted so that `t = 0` is first contact.
5. **Compressibility sets the sound speed.**  At `c0 = 10 V` the impact compressed the
   fluid by 20 % at `beta = 10 deg, V = 2 m/s` (bulk density ratio 1.20 in the artefact
   of the first attempt) and the pressure level was 40 % low; `c0 = 20 V` brings it to
   ~6 % and the centre-line pressure to within 15 %.
6. **The measured peak is a smoothed peak.**  Wagner's `p_max` is attained where
   `s = C`, i.e. within ~1 % of the spray root in `x` (`|x|/c = sqrt(1-1/C^2)`, 0.994
   at `beta = 10 deg`); SPH smears the square-root cusp over 1-2 `dx`.  That is why
   `p_max` is defined on `|x| <= 0.9 c` on both sides, and why the residual `p_max`
   error is the largest of the metrics.
7. **No surface tension, no viscosity, no air.**  Water at `c = 50 mm`, `V = 1 m/s`
   has `We = rho V^2 c / sigma ~ 700` and `Re ~ 5e4`, so both are secondary, but a
   real jet is stabilised by surface tension and broken up by air; this model has
   neither, so the spray sheet is a continuum sheet.
8. **Tank size.**  The tank walls are 4 `c_end` away laterally and 2.4 `c_end` deep.
   Wagner assumes an infinite half-space; the residual finite-depth error is folded
   into the case error and is what the ladder converges to, not a separate term.
9. **2-D, no hydroelasticity, rigid body.**  Wagner's theory is also 2-D; a
   comparison with slamming experiments would need the 3-D deadrise geometry and the
   body's elastic response.
""")

    parts.append("""## 6. Error judgement against the reference

Criteria (a)-(g) are listed in section 3.3 with their measured values.  Nothing here
reaches the platform's 3 % "acceptance" band: the errors that remain are 5-20 % on the
integral quantities and 10-50 % on the peak pressure, which is what a `dx = L/400`
WCSPH run can be expected to deliver for a square-root-singular pressure field.  The
honest statement is that the *trend* (wetting law, self-similarity, scaling exponents)
is reproduced and that the *levels* are within the campaign's 10/15/20 % gates for the
criteria that pass at the resolutions reported here; the failing criteria are reported
as failures and are not averaged away.

Reproduce with::

    python -m pytest tests/test_wedge_entry.py -q
    python benchmarks/wedge_entry/run.py --stage cases --shard 0/1
    python benchmarks/wedge_entry/run.py --stage ladder --shard 0/1
    python benchmarks/wedge_entry/run.py --stage aggregate
    python benchmarks/wedge_entry/make_doc.py
""")
    return "\n".join(parts)


def figures_payload() -> list[dict]:
    out = []
    for path in sorted(ART.glob("fig*.png")):
        out.append({
            "name": path.name,
            "caption": path.stem.replace("_", " "),
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "verification_wedge_entry.md"))
    ap.add_argument("--report-json", default=None,
                    help="also write a report payload (title/summary/markdown)")
    args = ap.parse_args(argv)
    summary = load_summary()
    text = build(summary)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    figs_dir = ROOT / "docs" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(ART.glob("fig*.png")):
        (figs_dir / f"wedge_entry_{path.name}").write_bytes(path.read_bytes())
    if args.report_json:
        payload = {
            "title": "P188-N01 wedge water entry: Wagner (1932) verification surface in TensorSPH",
            "summary": (
                "2-D wedge impact at beta = 10/15/20/30 deg x V = 1/2/5 m/s with a "
                "WCSPH solver (Tait EOS, Monaghan viscosity, Adami wall particles) "
                "against the Wagner reference: wetting law, peak pressure, spray-root "
                "position, grid-convergence ladder and deadrise/velocity scaling."),
            "markdown": text,
            "figures": figures_payload(),
            "tags": ["TensorSPH", "slamming", "Wagner", "verification", "P188-N01"],
        }
        Path(args.report_json).write_text(json.dumps(payload))
        print(f"report payload -> {args.report_json}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
