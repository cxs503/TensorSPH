"""Generate docs/benchmarks/laplace.md from artifacts/laplace/laplace_result.json.

Numbers are injected from the run artefact, never typed by hand.
"""
# ruff: noqa: E501  (long CJK/markdown template lines; see PR notes)
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "artifacts/laplace/laplace_result.json").read_text())
a, b, hd = d["stage_a"], d["stage_b"], d["stage_a"]["headline"]

rows_sweep = "\n".join(
    "| {w:.1f} | {ring} | {k:.3f} | {ke:.2f} % | {i:.3f} | {ie:.2f} % "
    "| {dp:.4f} | {dpe:.2f} % |".format(
        w=e["w_over_dx"], ring=e["ring_particles"], k=e["kappa_mean"],
        ke=100 * e["kappa_relative_error"], i=e["integral_grad_c_dr"],
        ie=100 * e["integral_relative_error"], dp=e["delta_p_operator"],
        dpe=100 * e["delta_p_relative_error"])
    for e in a["width_sweep"])

body = f"""# Laplace-law verification of the CSF surface-tension model

**Case** `benchmarks/laplace/run.py` · stage A = operator vs an analytic circle ·
stage B = free-surface droplet relaxation · {b['particles']} particles in stage B,
{a['particles']} quadrature points in the stage-A patch · torch {d['torch']} /
python {d['python']} · total wall {d['wall_seconds']:.0f} s (CPU, {d['nproc']} cores
visible, run under heavy machine load).

## 1. What is verified, and why

`src/tensorsph/surface_tension.py` (m277 / PR #13) implements the Continuum
Surface Force model (Brackbill, Kothe & Zemach 1992) and shipped ten unit tests.
None of them compares the model with an analytic surface-tension result: they
check shapes, the `sigma = 0` short circuit and unit-length normals. The
reference used here is the Laplace law for a 2-D static droplet:

    Delta p = p_in - p_out = sigma / R

Two independent measurements are reported: the **operator** (curvature and
interface delta function on a *prescribed* analytic interface) and the
**free-surface droplet** the solver actually carries in `rhs`.

## 2. A defect found on the way (fixed in this branch)

`compute_color_gradient` negated the j-side `index_add` and used `V_j` for it.
Each pair then contributed opposite signs to its two endpoints, so the two
roughly equal populations cancelled and the gradient of a *smooth* field came
out as noise around zero. Measured on the linear field `c = 3x`: the interior
mean gradient was `1e-18` before the fix and {a['grad_linear_measured'][0]:.4f}
after it, against the analytic `3.0` (rel. err
{100 * a['grad_linear_relative_error']:.2f} %). The CSF force was therefore
inert, and all ten original tests still passed -- a uniform colour field has a
zero gradient in either version.

## 3. Stage A -- operator verification (PASS)

`c(r) = 0.5 (1 - tanh((r - R)/w))` sampled on a full lattice patch of
half-width {1000 * a['lattice_half_width']:.0f} mm ({a['pairs']} pairs), `h = 1.3 dx`,
`R = 40 mm`, `dx = 2.5 mm`. `kappa` must equal `1/R = 25 1/m` and
`integral |grad c| dr` must equal the colour jump (= 1). Their product is the
Laplace jump implied by the discretised operator.

| w/dx | ring particles | kappa [1/m] | kappa err | int\\|grad c\\|dr | int err | Delta p [Pa] | Delta p err |
|---|---|---|---|---|---|---|---|
{rows_sweep}

Analytic `Delta p = sigma/R = {hd['delta_p_analytic']:.4f} Pa`. The error is flat
in `w/dx` ({100 * min(e['delta_p_relative_error'] for e in a['width_sweep']):.2f} -
{100 * max(e['delta_p_relative_error'] for e in a['width_sweep']):.2f} %), i.e. the
operator is *consistent* rather than tuned for one interface width. Headline at
`w = 2 dx`: kappa = {hd['kappa_mean']:.3f} 1/m ({100 * hd['kappa_relative_error']:.2f} %),
`int |grad c| dr` = {hd['integral_grad_c_dr']:.3f}
({100 * hd['integral_relative_error']:.2f} %), Delta p = {hd['delta_p_operator']:.4f} Pa,
i.e. **{100 * hd['delta_p_relative_error']:.2f} %**.

Method note that matters: evaluating the analytic colour field *only on the
fluid side* (the droplet lattice) removes the outer half of the tanh, drops
`integral |grad c| dr` to 0.35 instead of 1 and biases every Laplace number by a
factor ~3. An operator test must span both sides of the interface.

## 4. Stage B -- free-surface droplet (FAIL, {100 * b['relative_error']:.1f} %)

Static droplet, `gravity = 0`, no walls, `rho0 = {b['rho0']:.0f}`,
`sigma = {b['sigma']}`, `c0 = {b['sound_speed']}`, `nu = {b['viscosity']}`,
hex lattice `R = {1000 * b['R_lattice']:.0f} mm`, `R_eq = {1000 * b['R_equivalent']:.2f} mm`.

| quantity | value |
|---|---|
| dt | {b['dt']:.3e} s (binding limit: {b['dt_limit_source']}) |
| steps / sim time | {b['steps_run']} / {b['sim_time']:.3f} s = {b['sim_time'] / b['tau_capillary']:.2f} tau_sigma |
| tau_sigma = sqrt(rho R^3 / sigma) | {b['tau_capillary']:.3f} s |
| Delta p measured, relaxed | {b['delta_p_measured']:.4f} Pa |
| Delta p static force balance, t = 0 | {b['delta_p_static_force_balance']:.4f} Pa |
| Delta p analytic sigma/R_eq | {b['delta_p_analytic']:.4f} Pa |
| relative error, relaxed | **{100 * b['relative_error']:.2f} %** |
| relative error, static balance | **{100 * b['static_force_balance_relative_error']:.2f} %** |

Both measurements under-predict, so the routine is **not acceptable** under the
15 % criterion, and the 3 % figure is missed by an order of magnitude.

## 5. Why it under-predicts, and what would fix it

The jump equals `sigma * integral kappa * delta_s dA`, with `delta_s = |grad c|`
normalised so that `integral delta_s dr` across the interface equals the colour
jump. In a single-phase droplet the exterior is vacuum: there are no particles
outside `r = R`, so `|grad c|` is only integrated over the inner half of the
interface and `integral delta_s dr` reaches only ~0.25-0.35 of the jump. The
colour function used by `compute_surface_tension_force` -- neighbour counts with
a `0.7 * <count>` threshold, whose own jump is ~0.4 rather than 1 -- multiplies
the deficit. Both factors belong to the *indicator*; the curvature operator that
stage A exercises is accurate to ~0.3 %.

Remediation options, cheapest first:

1. normalise `delta_s` per interface so that its integral over the band equals
   the colour jump between the interior and the outermost shell;
2. use a density-ratio colour function `c = rho/rho0` (which the module
   docstring already claims) with an explicit free-surface threshold, and
   normalise `delta_s` by that jump;
3. two-phase CSF with a genuine exterior phase, which is what Adami, Hu &
   Adams (2010) use for their Laplace test and what removes the half-support
   factor entirely.

`tests/test_laplace_validation.py::test_free_surface_droplet_laplace_deficit`
records the current deficit as `xfail(strict=True)`: whoever fixes it turns the
suite red on purpose, so the fix cannot be merged silently.

## 6. Pitfalls recorded

- The first stage-B attempt used `dt = 0.2 h / c0` (acoustic limit only) and
  diverged within 40 steps to ~9 km/s. At `nu = 1e-2` the viscous limit
  `0.125 h^2 / nu` is five times tighter, and the same divergence occurs with
  `sigma = 0`: it is a base-solver instability, not a CSF one. The benchmark now
  takes the minimum of the library's `stable_dt` and `capillary_dt_limit`.
- A single instant of a relaxing droplet is not converged. The table therefore
  also reports the static momentum balance `Delta p = -integral rho (a . r_hat) dA`,
  which needs no time marching and is bit-reproducible.

## 7. Error verdict

| quantity | basis | error | < 3 %? | < 15 %? |
|---|---|---|---|---|
| colour-gradient consistency | analytic linear field | {100 * a['grad_linear_relative_error']:.2f} % | yes | yes |
| curvature kappa | analytic 1/R | {100 * hd['kappa_relative_error']:.2f} % | yes | yes |
| interface delta integral | analytic colour jump | {100 * hd['integral_relative_error']:.2f} % | yes | yes |
| Laplace jump, operator level | analytic sigma/R | {100 * hd['delta_p_relative_error']:.2f} % | yes | yes |
| Laplace jump, free-surface droplet | analytic sigma/R_eq | {100 * b['relative_error']:.2f} % | **no** | **no** |

The operator (after the sign fix) reproduces the Laplace law to better than 3 %.
The free-surface force routine does not, and this report states it rather than
rounding it away. Figure `artifacts/laplace/laplace.png`: (a) kappa and
`integral |grad c| dr` against `w/dx` with the analytic line, (b) the droplet
pressure jump history against `sigma/R_eq(t)`.
"""
out = ROOT / "docs/benchmarks/laplace.md"
out.write_text(body)
print(out, len(body))
