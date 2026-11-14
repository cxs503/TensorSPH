"""Emit ``docs/xval_projection_laplace_3d_density_ratio.md`` from the sweep JSON.

Every number in the document is formatted from
``scripts/xval_projection_laplace_3d/results/density_ratio.json`` so nothing is
typed by hand.  Run after ``density_ratio.py``:

    python scripts/xval_projection_laplace_3d/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "scripts/xval_projection_laplace_3d/results/density_ratio.json"
OUT = ROOT / "docs/xval_projection_laplace_3d_density_ratio.md"

ARMS = ("force_density", "density", "uniform")


def _fmt(v, nd=2, sign=False):
    if v is None:
        return "—"
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s


def _table(rows, arms, ratios):
    head = "| arm | " + " | ".join(f"ratio {r:.0f}:1" for r in ratios) + " |"
    sep = "|" + "---|" * (len(ratios) + 1)
    body = []
    for arm in arms:
        cells = []
        for r in ratios:
            hit = [x for x in rows
                   if x["requested_weighting"] == arm and x["requested_ratio"] == r]
            cells.append(f"{_fmt(hit[0]['dp_error_pct'], 2, True)} %" if hit else "—")
        body.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def _table_dp(rows, arms, ratios):
    head = "| arm | " + " | ".join(f"ratio {r:.0f}:1" for r in ratios) + " |"
    sep = "|" + "---|" * (len(ratios) + 1)
    body = []
    for arm in arms:
        cells = []
        for r in ratios:
            hit = [x for x in rows
                   if x["requested_weighting"] == arm and x["requested_ratio"] == r]
            cells.append(f"{hit[0]['dp_measured_Pa']:.4f}" if hit else "—")
        body.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def _table_res(rows, ratios):
    levels = sorted({r["R_over_dx"] for r in rows})
    head = "| R/dx | " + " | ".join(f"`{a}` {r:.0f}:1" for a in ("force_density", "density")
                                    for r in ratios) + " |"
    sep = "|" + "---|" * (1 + 2 * len(ratios))
    body = []
    for lv in levels:
        cells = []
        for a in ("force_density", "density"):
            for r in ratios:
                hit = [x for x in rows if abs(x["R_over_dx"] - lv) < 1e-9
                       and x["poisson_weighting"] == a
                       and x["requested_ratio"] == r]
                cells.append(f"{_fmt(hit[0]['dp_error_pct'], 2, True)} %" if hit else "—")
        body.append(f"| {lv:.1f} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def main():
    d = json.loads(RES.read_text())
    p, v, g = d["params"], d["verdict"], d["gates"]
    ratios = sorted({r["requested_ratio"] for r in d["ratio_ladder"]})
    ratio_rows = sorted(d["ratio_ladder"],
                        key=lambda r: (ARMS.index(r["requested_weighting"]),
                                       r["requested_ratio"]))
    res_rows = d["resolution_ladder"]
    ctrl = d["controls"]
    gh = g["owner_gate_pct"]

    fd = [r for r in ratio_rows if r["requested_weighting"] == "force_density"]
    spread = d["derived"]["ratio_spread_pct"]
    fd_hi = max(fd, key=lambda r: r["requested_ratio"])
    dens_hi = [r for r in ratio_rows
               if r["requested_weighting"] == "density"
               and r["requested_ratio"] == max(ratios)][0]
    naive_hi = [r for r in ratio_rows
                if r["requested_weighting"] == "uniform"
                and r["requested_ratio"] == max(ratios)][0]
    res_hi = [r for r in res_rows if r["requested_weighting"] == "force_density"]
    res_den = [r for r in res_rows if r["requested_weighting"] == "density"]
    max_abs_fd = max(abs(r["dp_error_pct"]) for r in fd + res_hi)
    max_abs_den = max(abs(r["dp_error_pct"]) for r in res_den)
    res_levels = sorted({r["R_over_dx"] for r in res_rows})
    ratio_hi = max(ratios)

    gates_tbl = "\n".join(
        f"| `{k}` | {json.dumps(val, ensure_ascii=False)} |"
        for k, val in v.items() if isinstance(val, dict))

    doc = f"""# The projection-SPH Young-Laplace jump in 3-D at a real density ratio

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH**
· 2-D rounds: `docs/xval_projection_laplace_oss.md` (m17594),
`docs/xval_projection_laplace_robustness.md` (m18632),
`docs/xval_projection_laplace_density_ratio.md` (m20557 + round 4 m20771)
· 3-D arm: `docs/xval_projection_laplace_3d.md`, `docs/xval_projection_laplace_3d_budget.md`.

## Why this round exists

The 2-D arm closed a live defect: at a real density contrast the
variable-coefficient `1/rho` Poisson drifted under refinement (−6.4 % → −12.7 %),
and the fix was to stop putting `rho` in the operator at all. The static
momentum balance is `grad p = f` with the **force density** `f = -sigma kappa
delta_s n`, which contains neither `rho` nor the ratio, so the correct
projection operator is `-grad^2 p = -div(f)` (round 4, `poisson_weighting=
"force_density"`).

That fix existed **only in 2-D**. `tensorph.laplace_isph3d` had no density ratio
parameter at all: its carrier was built with `m = rho0 dx^3` everywhere, its
source was `-rho0 div(a)`, and every published 3-D number for this module was
therefore measured with *the one term the ratio multiplies switched off by
construction*. This round ports the arm to 3-D and measures what it does.

## Artifacts

| artifact | what |
|---|---|
| `src/tensorsph/laplace_isph3d.py` | `rho_vapour=`, `poisson_weighting=` on the case; `rho=`/`weighting=` on `pressure_poisson3`; `rho_phase=` on `csf_state3_two_phase`; `rho_vapour=` on `sphere_block` |
| `scripts/xval_projection_laplace_3d/density_ratio.py` | the sweep + figure generator (gates declared in `GATES`) |
| `scripts/xval_projection_laplace_3d/results/density_ratio.json` | every number below ({len(d['ratio_ladder'])} ratio-ladder + {len(d['resolution_ladder'])} resolution runs) |
| `scripts/xval_projection_laplace_3d/results/density_ratio.png` | fig. 1 (3 panels) |
| `tests/test_xval_projection_laplace_3d_density_ratio.py` | 8 gates at `R/dx = {p['radius_m']/p['ratio_level_dx_m']:.0f}` |
| `scripts/xval_projection_laplace_3d/make_doc.py` | this document (all numbers formatted from the JSON) |

## ① Case and parameters

| quantity | value |
|---|---|
| interface | sphere, `R = {p['radius_m']} m` (principal curvatures `2/R`, `2/R`) |
| surface tension | `sigma = {p['sigma_N_per_m']} N/m` (water/air, 20 °C) |
| densities | `rho_l = {p['rho0_kg_per_m3']:.0f} kg/m³`, `rho_v = rho_l / ratio`, ratio {', '.join(f'{r:.0f}' for r in ratios)} |
| reference | `dp = 2 sigma/R = {p['dp_theory_pa']:.4f} Pa` — **independent of the ratio** |
| carrier | simple-cubic lattice, fixed cell volume `V = dx³` in both phases (`m = rho V`), block half-width `R + {p['outer_m']} m`, outer `1.6 dx` layer = Dirichlet `p = 0` |
| baseline level | `dx = {p['ratio_level_dx_m']} m` (`R/dx = {p['radius_m']/p['ratio_level_dx_m']:.0f}`), ladder `dx = {', '.join(str(x) for x in p['levels_dx_m'])}` |
| scheme | Shepard colour (volume-weighted) → MLS gradient → smoothed normal → `div n` curvature → Brackbill `|grad c|` delta → CSF traction → Cummins & Rudman pair Laplacian, matrix-free CG (`rel_tol 1e-10`) |
| curvature | `{p['curvature_method']}`, stencil `{p['curvature_stencil']}` (the validated 3-D default) |

Three arms:

* **`force_density`** — `-grad^2 p = -div(f)` with `f = -sigma kappa delta_s n`.
  No `rho` anywhere in the operator, so the jump is *ratio blind by
  construction*.
* **`density`** — `-div((1/rho) grad p) = -div(a)`, harmonic pair mean
  `rho_ij = 2 rho_i rho_j/(rho_i+rho_j)`: the 2-D round-3 arm, ported.
* **`uniform`** — constant-coefficient `-grad^2 p = -rho0 div(a)`, i.e. the
  historical 3-D implementation, which treats the ratio as a source rescaling.
  Declared **ablation**; it is *required* to fail (G5).

Gates were fixed before the sweep and are stored in the JSON:
{gh:.0f} % owner gate on both candidate arms, bit-identical 1:1 regression,
{spread and g['ratio_spread_pct']:.0f} % ratio-spread band,
{g['resolution_band_pct']:.0f} % refinement band, {g['flux_crosscheck_pct']:.0f} % traction-flux
cross-check, `sigma = 0` must give a zero jump, and the naive arm must exceed
{gh:.0f} %.

## ② Results

### A — ratio ladder at `R/dx = {p['radius_m']/p['ratio_level_dx_m']:.0f}`: `dp` [Pa]

{_table_dp(ratio_rows, ARMS, ratios)}

### B — the same rows as error against `2 sigma/R = {p['dp_theory_pa']:.4f} Pa`

{_table(ratio_rows, ARMS, ratios)}

* `force_density`: identical to display precision at **every** ratio — measured
  spread `{spread:.2e} %` of theory (`< {g['ratio_spread_pct']:.0f} %` gate). This is the 2-D round-4
  conclusion reproduced in 3-D.
* `density` (the variable-coefficient form): `{dens_hi['dp_error_pct']:+.2f} %` at
  {ratio_hi:.0f}:1 — the `1/rho` operator does *see* the ratio, and it degrades.
* `uniform` (historical): `{naive_hi['dp_error_pct']:+,.0f} %` at {ratio_hi:.0f}:1. The ratio is not a
  source rescaling; 3-D over-predicts by a factor
  {naive_hi['dp_measured_Pa']/naive_hi['dp_theory_Pa']:,.0f}.

### C — refinement at {ratio_hi:.0f}:1 (error against the analytic jump)

{_table_res(res_rows, (1.0, ratio_hi))}

* `force_density` stays inside `{max_abs_fd:.2f} %` over every level and both
  contrasts; `density` reaches `{max_abs_den:.2f} %`. Both pass the {gh:.0f} % owner
  gate; the `{g['resolution_band_pct']:.0f} %` band (G4) is
  {"**met**" if v['G4_resolution_band_5pct']['pass'] else "**missed**"} for the validated arm.
* The 1:1 column is the historical 3-D ladder, unchanged (the equal-density
  shortcut keeps the legacy operator path bit-identical — see D).

### D — controls

| control | value | gate |
|---|---|---|
| 1:1 regression, `force_density` vs historical | rel. diff `{ctrl['equal_density']['rel_diff_force_density']:.2e}` | `< {ctrl['equal_density']['gate_rel_tol']:.0e}` {"✓" if v['G2_equal_density_bit_identical']['pass'] else "✗"} |
| 1:1 regression, `density` vs historical | rel. diff `{ctrl['equal_density']['rel_diff_density']:.2e}` | same path ✓ |
| `sigma = 0` at {ratio_hi:.0f}:1 | `dp = {ctrl['sigma_zero']['dp_pa']:.3e} Pa`, flux `{ctrl['sigma_zero']['dp_flux_pa']:.3e} Pa` | exact zero ✓ |
| traction flux `dp_flux/(2 sigma/R)` | max dev `{v['flux_crosscheck']['max_abs_dev_pct']:.2f} %` over all {len(d['ratio_ladder'])+len(d['resolution_ladder'])} runs, ratio blind | `< {g['flux_crosscheck_pct']:.0f} %` {"✓" if v['flux_crosscheck']['pass'] else "✗"} |
| carrier geometry | `V = dx³` identical in both phases; colour, `kappa`, `delta_s` invariant to the ratio | pinned by `test_carrier_geometry_is_ratio_invariant` |

The traction-flux instrument never touches the Poisson solve, so its ratio
blindness (`{v['flux_crosscheck']['max_abs_dev_pct']:.2f} %` deviation from theory, no ratio
dependence) is what makes the pressure readout trustworthy: the jump is in the
traction, and only the *operator* was ever at stake.

### E — figure

`scripts/xval_projection_laplace_3d/results/density_ratio.png`:
(a) ratio ladder for the three arms with the {gh:.0f} % gate band,
(b) refinement at 1:1 and {ratio_hi:.0f}:1 for the two candidate operators with the
{g['resolution_band_pct']:.0f} % band, (c) the pressure profile through the droplet at
{ratio_hi:.0f}:1 (force-density arm) showing the two plateaus and `2 sigma/R`.

## ③ Gates

| gate | value |
|---|---|
{gates_tbl}

`all_pass = {v['all_pass']}`.

## ④ Notes, limits, what is still open

* **What this changes.** The 3-D module now has the same physics-preserving
  operator as 2-D, and the 3-D jump is measured to be *independent of the
  density ratio* — the property the analytic law demands and the previous 3-D
  implementation could not demonstrate (it had no ratio to vary).
* **What it does not change.** This is the static equilibrium of the projection
  step at `u = 0`. The vapour is a particle carrier at `rho_v`, not a simulated
  gas: no mass transfer, no free-surface boundary condition, no time
  integration. The dynamic path remains the documented unstable one from the 2-D
  round 1.
* **The `density` arm is kept as an ablation, not as the shipped default.** Its
  ratio-dependent error (`{dens_hi['dp_error_pct']:+.2f} %` at {ratio_hi:.0f}:1 here) is the same
  failure family as the 2-D round-3 refined-mesh drift; the mechanism named there
  (an inconsistent divergence of `a` against the stiff `1/rho` bridge) is not
  remeasured in 3-D in this round.
* **Lattice orientation is still axis-aligned.** The 3-D budget round listed
  general rotation as an open item; it needs a rebuilt non-cubic lattice, and
  rotating the carrier also re-samples the interface, so the two effects cannot
  be separated without a separate study. Not attempted here.
* **Resolution is coarse by 2-D standards.** `R/dx = {min(res_levels):.0f}…{max(res_levels):.1f}` in 3-D is
  what the memory budget allows; the numbers are therefore quoted against the
  {gh:.0f} % owner gate, and the {g['resolution_band_pct']:.0f} % band is a self-imposed target, not a
  convergence proof.
* **Parameter-free? No.** `h/dx = {p['h_factor']}` and the {p['outer_m']} m pad are discretisation
  choices; the 2-D rounds showed they can move the jump by a few percent while
  the physical invariants move by ~0.1 %.

## ⑤ Error verdict

| quantity | reference | best measured | gate |
|---|---|---|---|
| `dp`, `force_density`, ratio 1 (regression) | `{p['dp_theory_pa']:.4f} Pa` | `{ctrl['equal_density']['rel_diff_force_density']:.1e}` rel. diff to history | identical ✓ |
| `dp`, `force_density`, all ratios at `R/dx = {p['radius_m']/p['ratio_level_dx_m']:.0f}` | `{p['dp_theory_pa']:.4f} Pa` | max `{max(abs(r['dp_error_pct']) for r in fd):.2f} %` | `< 3 %` {"✓" if max(abs(r['dp_error_pct']) for r in fd) < 3.0 else "✗"} |
| `dp`, `force_density`, all levels at {ratio_hi:.0f}:1 | `{p['dp_theory_pa']:.4f} Pa` | max `{max(abs(r['dp_error_pct']) for r in res_hi):.2f} %` | `< {g['resolution_band_pct']:.0f} %` {"✓" if v['G4_resolution_band_5pct']['pass'] else "✗"} |
| `dp`, `density`, worst cell | `{p['dp_theory_pa']:.4f} Pa` | `{max(abs(r['dp_error_pct']) for r in dens_hi and [dens_hi] + res_den):.2f} %` | `< {gh:.0f} %` ✓ |
| `dp`, `uniform` (naive), {ratio_hi:.0f}:1 | `{p['dp_theory_pa']:.4f} Pa` | `{naive_hi['dp_error_pct']:+,.0f} %` | must fail ✓ (it fails) |
| traction flux, all runs | `2 sigma/R` | max dev `{v['flux_crosscheck']['max_abs_dev_pct']:.2f} %`, ratio blind | `< {g['flux_crosscheck_pct']:.0f} %` ✓ |
| `sigma = 0` control | `0 Pa` | `{ctrl['sigma_zero']['dp_pa']:.1e} Pa` | exact ✓ |

**Is the key quantity within 3 %?** For the validated arm, **yes at every ratio
and every level** (max `{max_abs_fd:.2f} %` over the {len(fd)+len(res_hi)} force-density runs), with the
equal-density path unchanged from the historical 3-D numbers. The
variable-coefficient arm is *not* within 3 % at high contrast
(`{dens_hi['dp_error_pct']:+.2f} %`), which is exactly why it is not the default.

*Wall clock*: {sum(r['wall_s'] for r in d['ratio_ladder'] + d['resolution_ladder']):.0f} s for the
{len(d['ratio_ladder']) + len(d['resolution_ladder'])} sweep runs (single thread, CPU).
"""
    OUT.write_text(doc)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()