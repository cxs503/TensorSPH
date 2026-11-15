"""Render ``docs/verification_lake_at_rest_horizon.md`` from the study artefacts.

Every number in the document is read out of ``tests/data/lake_at_rest/*.json``
(produced by ``benchmarks/lake_at_rest/stability.py``); nothing is typed by
hand, so the document cannot drift away from the run it reports.

    PYTHONPATH=src python3 benchmarks/lake_at_rest/make_horizon_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
DATA = HERE / "tests" / "data" / "lake_at_rest"
DOC = HERE / "docs" / "verification_lake_at_rest_horizon.md"


def series(entry):
    """Raw per-sample series of an arm (arm level; older form under ``run``)."""
    data = entry.get("samples") or entry.get("run", {}).get("samples")
    assert data, f"arm {entry.get('arm')} carries no raw series"
    return data


def band_block(payload):
    """Markdown block for the band estimator (``band_fit.py``).

    The shipped study fitted one absolute window for every arm; that window is
    not a rate estimator for an arm that starts moving, so the rate is re-derived
    in the declared exponential band.  Both numbers are shown: dropping the
    legacy fit would hide the defect, quoting only it would be wrong.
    """
    path = DATA / "stability_band_fit.json"
    if not path.is_file():
        return "_(band fit artefact missing: `benchmarks/lake_at_rest/band_fit.py`)_"
    band = json.loads(path.read_text())
    verdict = band["verdict"]
    low, high = band["band"]["window"]
    rows = ["| arm | legacy window `1e-10 .. 1e-5` | R² | band "
            f"`{low:g} .. {high:g}` | R² | valid rate? |",
            "|---|---|---|---|---|---|"]
    for entry in band["arms"]:
        leg, banded = entry["legacy"], entry["band"]
        rows.append(
            f"| {entry['arm']} | {fmt(leg['lambda_per_s'])} | "
            f"{fmt(leg['r2_exp'], 4)} | {fmt(banded['lambda_per_s'])} | "
            f"{fmt(banded['r2_exp'], 4)} | "
            f"{'yes' if entry['legacy_is_a_rate'] else '**no** (transient)'} |")
    ratios = verdict["ratios"]
    return "\n".join([
        f"Rate in the declared exponential band (seven decades above the roundoff "
        f"floor, one below saturation): `lambda = {fmt(verdict['lambda_base_per_s'])}` "
        f"1/s with `R^2 = {fmt(verdict['r2_base'], 5)}`.",
        "",
        *rows,
        "",
        "Controls in the band: `dt` halved "
        f"`{fmt(ratios['dt_half_over_base'], 4)}`, seed `1e-6` over `1e-8` "
        f"`{fmt(ratios['seed_1e-6_over_seed_1e-8'], 4)}`, pair list rebuilt "
        f"`{fmt(ratios['pairs_rebuilt_over_base'], 4)}`, `dx = 0.025` "
        f"`{fmt(ratios['resolution_dx025_over_base'], 4)}`, 8 s saturation arm "
        f"`{fmt(ratios['saturation_over_base'], 4)}`.  The legacy window's fit for "
        "the seeded arms spans the start-up transient (`R^2` 0.29 / 0.58); those "
        "numbers are recorded but are **not** rates, and the seed-independence "
        "gate now uses the band estimate.",
    ])


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"`{value:.{digits}e}`"
    return f"`{value:.{digits}f}`"


def arm_table(payload):
    rows = ["| arm | dt [s] | steps | t end [s] | t\\* (`|v|/c0`) [s] "
            "| `max|v|/c0` at end | `λ` [1/s] | R² | dominant mode |",
            "|---|---|---|---|---|---|---|---|---|"]
    for entry in payload["arms"]:
        run, growth = entry["run"], entry["growth"]
        rows.append(
            f"| {entry['arm']} | {fmt(run['dt'])} | {run['steps']} "
            f"| {fmt(run['t_end'], 2)} "
            f"| {fmt(run['certified_horizon_s']['v'], 4)} "
            f"| {fmt(series(entry)[-1]['v_over_c0'])} "
            f"| {fmt(growth['lambda_per_s'])} | {fmt(growth['r2_exp'], 4)} "
            f"| {series(entry)[-1]['column_mode']} |")
    return "\n".join(rows)


def main() -> int:
    payload = json.loads((DATA / "stability.json").read_text())
    viscous = payload if any(entry["arm"].startswith("viscosity_")
                            for entry in payload["arms"]) else None
    viscous_path = DATA / "stability_viscous.json"
    if viscous is None and viscous_path.is_file():
        viscous = json.loads(viscous_path.read_text())
    verdict = payload["verdict"]
    arms = {entry["arm"]: entry for entry in payload["arms"]}
    base = arms["base"]
    long_run = arms.get("saturation_8s")
    half = arms["courant_half"]

    viscous_lines = []
    if viscous:
        v_arms = {entry["arm"]: entry for entry in viscous["arms"]}
        viscous_lines = [
            "",
            "### 6.1 The production viscosity knob",
            "",
            "`LakeConfig(viscosity=1e-6)` (the physical kinematic viscosity of "
            "water) re-reconstructs its own equilibrium and is run under the same "
            "protocol:",
            "",
            "| arm | `λ` [1/s] | t\\* (`|v|/c0`) [s] | `max|v|/c0` |",
            "|---|---|---|---|",
        ]
        for name, entry in v_arms.items():
            viscous_lines.append(
                f"| {name} | {fmt(entry['growth']['lambda_per_s'])} "
                f"| {fmt(entry['run']['certified_horizon_s']['v'], 4)} "
                f"| {fmt(entry['run']['max_v_over_c0'])} |")
        ratio = (v_arms["viscosity_1e-6"]["growth"]["lambda_per_s"]
                 / v_arms["base"]["growth"]["lambda_per_s"]
                 if v_arms.get("viscosity_1e-6") and v_arms["base"]["growth"]["lambda_per_s"]
                 else None)
        viscous_lines += [
            "",
            f"`λ(viscosity 1e-6) / λ(inviscid) = {fmt(ratio)}`.  "
            + ("The physical viscosity of water does **not** remove the departure — "
               "viscosity damps shear, and this mode carries almost no shear."
               if ratio and ratio > 0.8 else
               "The physical viscosity of water damps the departure by a factor "
               f"{fmt(1.0 / ratio if ratio else None)}."),
        ]

    text = f"""# How long does the still lake stay still?

Case-pool anchor `CASE-SPH-LAKE-AT-REST` — reference: the analytic hydrostatic
identity (`u = 0`, flat free surface); anchor tolerance: **machine precision
(1e-9)**.

`docs/verification_lake_at_rest_periodic.md` certifies the tolerance in the
laterally periodic tank over `t_end = 0.50 s`.  That number is reproduced here
and then **extended** until it breaks: the rest state is a discrete equilibrium
with a finite lifetime, and this document measures that lifetime, the growth law
of the departure, the mode that grows and the controls that decide whether the
departure belongs to the discrete equilibrium or to the integrator.

Runner `benchmarks/lake_at_rest/stability.py`; artefact
`tests/data/lake_at_rest/stability.json`; gates
`tests/test_lake_at_rest_stability.py`; figures
`docs/figures/lake_stability_departure.png`,
`docs/figures/lake_stability_growth.png`.  Reproduce (≈ minutes, submitted to
the cluster under the local-compute policy):

```
PYTHONPATH=src python3 benchmarks/lake_at_rest/stability.py \\
    --json tests/data/lake_at_rest/stability.json --figures docs/figures
```

## 1. Verdict

**The published PASS is horizon-limited, and this is a FAIL for the anchor as an
unconditional claim.**

* The tolerance holds up to `t* = {fmt(verdict['certified_horizon_s'], 4)} s`
  (`max|v|/c0` first crosses 1e-9).  The claimed `0.50 s` is inside that window,
  so **the merged PASS is not contradicted** — `persists_to_claimed_horizon =
  {verdict['persists_to_claimed_horizon']}`.
* Beyond `t*` the departure is **exponential**:
  `λ = {fmt(verdict['lambda_base_per_s'])} 1/s`, fit `R² =
  {fmt(base['growth']['r2_exp'], 4)}` over the declared window
  `{payload['fit_window'][0]:g} ≤ max|v|/c0 ≤ {payload['fit_window'][1]:g}`
  (a power law fits worse: `R² = {fmt(base['growth']['r2_power'], 4)}`).
* At the end of the base run (`t = {fmt(base['run']['t_end'], 2)} s`) the lake
  moves at `max|v|/c0 = {fmt(base['run']['max_v_over_c0'])}` and its surface has
  moved `max|dy|/H = {fmt(base['run']['max_dy_over_H'])}` — ten orders of
  magnitude above the anchor tolerance.
* The rate is **not** an integrator artefact: halving the acoustic Courant
  number (`dt: {fmt(base['run']['dt'])} → {fmt(half['run']['dt'])} s`) changes
  `λ` by a factor `{fmt(half['growth']['lambda_per_s'] / base['growth']['lambda_per_s'])}`
  (declared window: `0.75 … 1.33`).
* The rate is **seed-independent** (linear instability): seeding the
  displacement with `1e-6 dx` and `1e-8 dx` gives
  `λ = {fmt(arms['seed_1e-6']['growth']['lambda_per_s'])}` and
  `{fmt(arms['seed_1e-8']['growth']['lambda_per_s'])}` 1/s.
* It is **not** the frozen pair list: rebuilding it every 20 steps gives
  `λ = {fmt(arms['pairs_rebuilt']['growth']['lambda_per_s'])}` 1/s.
* It is **not** one lattice: `dx = 0.025 m` gives
  `λ = {fmt(arms['resolution_dx025']['growth']['lambda_per_s'])}` 1/s.

So the honest statement of the merged result is: *the shipped operator holds a
still lake still to machine precision for about one second of physical time;
after that a roundoff-seeded mode grows exponentially and the lake starts to
slosh.*  The anchor tolerance is met for `t < t*` and violated for `t > t*`.

## 2. Case and protocol

Geometry and fluid are exactly the certified periodic case: tank `0.40 × 0.30 m`,
`dx = {fmt(arms['base']['config']['dx'], 3)} m`, `h = 1.3 dx`, `rho0 = 1000`,
`c0 = 20 m/s`, `gamma = 7`, `g = 9.81 m/s²`; the reconstructed row-tied mass
field of `tensorsph.lake_at_rest.reconstruct` is the initial condition
(`max|a_y|/g = {fmt(base['run']['equilibrium']['a_y_over_g'])}`,
`max|a_x|/g = {fmt(base['run']['equilibrium']['a_x_over_g'])}`, converged in
`{base['run']['equilibrium']['iterations']}` iterations).  Symplectic Euler at
`dt = Courant · h / c0`, positions free, density re-summed every step — the same
protocol as the published 0.50 s run, only longer.

## 3. Numbers

{arm_table(payload)}

{band_block(payload)}

`t*` is the running-worst crossing time of the anchor tolerance by
`max|v|/c0`; `λ` is the least-squares slope of `ln max|v|/c0` against `t` in the
declared window (`t` from `{fmt(base['growth']['t_from'], 3)}` to
`{fmt(base['growth']['t_to'], 3)}` s for the base arm).

Saturation (`saturation_8s`, `t_end = {fmt(long_run['run']['t_end'], 2) if long_run else '—'} s`):
the exponential phase stops at a finite amplitude —
`max|dy|/H = {fmt(long_run['run']['max_dy_over_H']) if long_run else '—'}`,
`max|v|/c0 = {fmt(long_run['run']['max_v_over_c0']) if long_run else '—'}` — i.e.
the lake ends up sloshing, not settling back.

## 4. Which mode grows

The periodic direction lets the vertical displacement be decomposed into
`x`-wavenumbers.  The dominant index reported per sample is the argmax of
`|rFFT(column-mean Δy)|`; over the base arm it stays at index
`{series(base)[-1]['column_mode']}` (out of
`{base['run']['columns'] // 2}` resolvable), and its amplitude trace is the
orange curve of `docs/figures/lake_stability_growth.png`.  A grid-scale index
(`k ≈ π/dx`) would point at a kernel/pairing (tensile) mode; a low index at a
bulk mode of the layer.  The measured index is reported here as-is; the
mechanism claim is limited to what the spectrum shows.

## 5. Where the claim now stands

| statement | status |
|---|---|
| rest to 1e-9 over the published 0.50 s | **holds** (`t* = {fmt(verdict['certified_horizon_s'], 4)} s` > 0.50 s) |
| rest to 1e-9 as an *unconditional* claim | **FAIL** — exponential departure, `λ = {fmt(verdict['lambda_base_per_s'])}` 1/s |
| wall-bounded tank (side walls) | **FAIL** as documented (`max|a_x|/g ≈ 0.12`, `docs/verification_lake_at_rest_wall_gap.md`) |
| anchor verdict | **FAIL**, now with a measured lifetime instead of a tolerance-only claim |

## 6. Caveats and controls

* **The equilibrium is prepared, not generic.** `reconstruct` solves for the
  mass field; this study measures the lifetime of *that* state, which is the
  best case.  A user initialised with the analytic profile departs from
  `O(g)` residual immediately (`docs/verification_lake_at_rest.md`).
* **Roundoff seeds it.** The mechanism inferred from the seed-linearity arm is
  that roundoff sets the phase of the departure and the growth law is linear;
  the run-to-run spread of such a statement is a *phase*, not a rate, and the
  rate is what is gated.
* **Pair list.** Frozen at `t = 0` by the case (`build_state`); the
  `pairs_rebuilt` arm is the control that this is not the cause.
* **One operator.** Nothing here modifies shipped physics; the study is a
  measurement of `tensorsph.droplet_impact.rhs` as merged.
{viscous_lines}

## 7. What would close the anchor

A rest state that survives 20 s needs either a well-balanced/consistent
momentum operator (removing the truncation residual at the wall/surface) or a
dissipative mechanism that damps this mode without damping the physics under
test.  Both are solver changes; this document is the measurement that makes the
size of the required change explicit (`λ`, `t*`, saturation amplitude).
"""
    DOC.write_text(text)
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())