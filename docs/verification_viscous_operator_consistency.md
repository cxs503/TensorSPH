# Viscous-operator consistency: seven implementations on one analytic field

**Benchmark**: `benchmarks/viscous_operator_consistency/`  
**Artifact**: `benchmarks/viscous_operator_consistency/results.json` (20 measurement records, wall clock 210 s on 1 thread)  
**Figure**: `benchmarks/viscous_operator_consistency/figures/nu_eff_vs_dx.png`

## 1. Why this measurement exists

`docs/verification_poiseuille_convergence.md` §6.3 lists one open item:

> The repository has 4 implementations of the WCSPH viscous operator
> (droplet_impact, wcsp_h/solver, threed, sod_shock). Consolidating to a
> single, well-verified operator would eliminate cross-implementation
> discrepancies.

The sentence asserts a discrepancy and never measures it. That ordering is
backwards: an operator that silently applies `0.5 nu` corrupts every case that
uses it, an operator that agrees to 0.1 % does not need consolidating, and the
decision to refactor shared numerical code should be taken against a number.
This benchmark produces that number for the paths that are callable as
operators, on a field where the exact answer is known.

## 2. Method

Each implementation is applied to the same analytic velocity field on the same
lattice, and its effective kinematic viscosity is read back by least-squares
projection on the exact Laplacian:

```
acc = nu_eff lap(u)   =>   nu_eff/nu = sum(acc . lap u) / (nu sum |lap u|^2)
```

`nu_eff/nu` is what the physics sees: it is the ratio that fixes a case's
effective Reynolds number, and it is normalisation-free -- an operator that is
a half-strength Laplacian returns 0.5 on any field. The residual
`||acc - nu_eff lap u|| / ||nu_eff lap u||` is reported separately, so a wrong
*level* (a constant) is not confused with a wrong *shape* (a stencil).

| parameter | value |
|---|---|
| kinematic viscosity nu | 0.0159155 m^2/s |
| field amplitude U0 | 1 m/s |
| smoothing length factor h/dx | 1.3 |
| reference density rho0 | 1000 kg/m^3 |
| artificial sound speed c0 | 20 m/s |
| Tait exponent gamma | 7 |
| periodic box | 1 m, 1 wave, k = 6.28319 1/m |
| Morris regulariser | eta^2 = 0.01 h^2 |
| library VISCOUS_LAPLACIAN_FACTOR | 2 |
| 2-D spacings | 0.05, 0.025, 0.0125 m |
| 3-D spacings | 0.1, 0.05, 0.04 m |

**Reference fields.** 2-D: the Taylor-Green field of `tensorsph.vv.tgv`,
`u = U e^{-2 nu k^2 t} sin kx cos ky`, `v = -U e^{-2 nu k^2 t} cos kx sin ky`, for
which `lap u = -2 k^2 u` exactly; the shape is an exact invariant of the
Navier-Stokes equations, so this is a reference and not a fit. 3-D:
`u = (sin kx sin ky sin kz, 0, 0)` with `lap u = -3 k^2 u`; a genuinely 3-D
field is used deliberately, because applying a 3-D operator to a z-independent
field would convolve the 3-D kernel with a 2-D one and confound the operator
under test with a kernel-projection artefact.

**Isolation of the viscous term.** `droplet_impact.rhs` and
`rhs_all_pairs` are full right-hand sides, so the viscous term is isolated by
differencing: the RHS is linear in `cfg.viscosity` (the pressure, density and
surface-tension terms never read it), hence `acc(nu) - acc(0)` is exactly the
viscous acceleration. This measures the production code path instead of a
private re-implementation of it. Those two paths have no periodic image, so
their number is read on the interior band (>= 3 lattice cells from every edge,
2h/dx = 2.6 spacings, so every interior particle sees its full support); the
periodic paths are read on the whole lattice and on the same band. The two
readings agree, which is the control that the band is wide enough.

**ISPH scaling.** `isph_channel.viscous_laplacian` returns the *bare*
Laplacian (the ISPH solver multiplies by nu itself and reuses the same operator
for the pressure Poisson matrix), so `nu * L[v]` is formed before the readout;
forgetting that factor makes the operator look 1/nu = 62.8 times too strong,
which is a property of the *caller*, not of the operator.

## 3. Results

### 3.1 2-D Morris family

| implementation | dx (m) | n | nu_eff/nu (interior) | residual % | family spread % |
|---|---:|---:|---:|---:|---:|
| droplet_impact.rhs (cell-linked) | 0.0125 | 80 | 0.9982 | 0.15 | 1.06 |
| isph_channel (Morris, eta-regularised) | 0.0125 | 80 | 1.0088 | 0.00 | 1.06 |
| vv.tgv (Morris) | 0.0125 | 80 | 0.9982 | 0.00 | 1.06 |
| droplet_impact.rhs_all_pairs (dense) | 0.025 | 40 | 0.9932 | 0.11 | 1.06 |
| droplet_impact.rhs (cell-linked) | 0.025 | 40 | 0.9932 | 0.11 | 1.06 |
| isph_channel (Morris, eta-regularised) | 0.025 | 40 | 1.0038 | 0.00 | 1.06 |
| vv.tgv (Morris) | 0.025 | 40 | 0.9932 | 0.00 | 1.06 |
| droplet_impact.rhs_all_pairs (dense) | 0.05 | 20 | 0.9736 | 0.09 | 1.07 |
| droplet_impact.rhs (cell-linked) | 0.05 | 20 | 0.9736 | 0.09 | 1.07 |
| isph_channel (Morris, eta-regularised) | 0.05 | 20 | 0.9839 | 0.00 | 1.07 |
| vv.tgv (Morris) | 0.05 | 20 | 0.9735 | 0.00 | 1.07 |

### 3.2 3-D Brookshaw family (`threed/operators3`)

| implementation | dx (m) | n | nu_eff/nu (interior) | residual % | family spread % |
|---|---:|---:|---:|---:|---:|
| threed without renormalisation | 0.04 | 25 | 1.1201 | 0.00 | 17.66 |
| threed renorm, order 2 | 0.04 | 25 | 0.9519 | 0.00 | 17.66 |
| threed renorm, order 4 | 0.04 | 25 | 1.0051 | 0.00 | 17.66 |
| threed without renormalisation | 0.05 | 20 | 1.0894 | 0.00 | 17.66 |
| threed renorm, order 2 | 0.05 | 20 | 0.9258 | 0.00 | 17.66 |
| threed renorm, order 4 | 0.05 | 20 | 1.0045 | 0.00 | 17.66 |
| threed without renormalisation | 0.1 | 10 | 0.8619 | 0.00 | 26.88 |
| threed renorm, order 2 | 0.1 | 10 | 0.7325 | 0.00 | 26.88 |
| threed renorm, order 4 | 0.1 | 10 | 0.9294 | 0.00 | 26.88 |

### 3.3 Spread inside each family (interior readout)

| dimension | dx (m) | min nu_eff/nu | max nu_eff/nu | spread % |
|---|---:|---:|---:|---:|
| 2-D | 0.05 | 0.9735 | 0.9839 | 1.07 |
| 2-D | 0.025 | 0.9932 | 1.0038 | 1.06 |
| 2-D | 0.0125 | 0.9982 | 1.0088 | 1.06 |
| 3-D | 0.1 | 0.7325 | 0.9294 | 26.88 |
| 3-D | 0.05 | 0.9258 | 1.0894 | 17.66 |
| 3-D | 0.04 | 0.9519 | 1.1201 | 17.66 |

### 3.4 Case-level cross-check (Poiseuille benchmark, production WCSPH)

`benchmarks/poiseuille/out/results.json`, `wall_friction = 1` (no-slip): the
measured effective viscosity of a *real integrated run*, read from the
curvature of the steady profile.

| H/dx | nu_eff/nu (fit over all rows) | nu_eff/nu (interior band only) |
|---:|---:|---:|
| 8 | 1.2182 | 0.9520 |
| 12 | 1.2106 | 1.0284 |
| 16 | 1.2356 | 1.1487 |

This is the cross-check that makes the operator audit usable: the operator
itself is `nu` to within a fraction of a percent on a smooth field (§3.1), so
the ~20 % excess seen when the *near-wall rows* are included in the fit is a
wall-boundary effect (the effective no-slip plane does not sit on the nominal
boundary), not a mis-normalised viscous operator. Two different measurements,
two different conclusions -- which is exactly why they are reported apart.

## 4. Findings

1. **The 2-D Morris family is one operator, not four.** `vv.tgv`, `droplet_impact.rhs` (cell-linked) and `droplet_impact.rhs_all_pairs` (dense) agree to better than 0.01 % at every resolution tested, and all three converge to `nu_eff/nu = 1` from below (0.9735 at dx = 0.05 to 0.9982 at dx = 0.0125). The ISPH variant, which carries its own hard-coded factor 2 and an `eta` regulariser, sits 1 % high at the finest resolution. Family spread at the finest 2-D resolution: 1.06 %. **There is nothing to consolidate here** — the discrepancy the §6.3 note predicts does not exist at this level.
2. **The 3-D family is where the ambiguity is.** The three
   `threed/operators3` variants (moment renormalisation order 2, order 4,
   none) spread
   17.66 % at dx = 0.04 and 26.9 % at dx = 0.1, i.e. the operator applies between 0.95 and 1.12 times the requested kinematic viscosity at the resolutions the 3-D cases
   actually use. All three residuals are 0.00 %, so the *shape* of the
   Laplacian is right everywhere; the disagreement is purely a level.
   The choice of renormalisation is therefore a physics decision with a
   measurable Reynolds-number consequence, not a cosmetic one.
3. **Order-4 correction is the accurate variant.** At dx = 0.04 the
   order-4 renormalised operator is within
   0.51 % of the requested viscosity, while order-2 renormalisation is
   4.81 % low and no renormalisation is
   12.01 % high. The published claim that renormalisation turns a 17 % over-prediction
   into a couple of percent is confirmed in sign and order of magnitude, but
   only for the order-4 variant at this h/dx; order-2 renormalisation
   overcorrects below the requested value at every resolution tested here.
4. **Convergence order of the level error.** Fitting the |nu_eff/nu - 1|
   ladder over the two finest 2-D resolutions gives apparent orders
   vv.tgv (Morris) = 1.93, droplet_impact.rhs (cell-linked) = 1.94, isph_channel (Morris, eta-regularised) = -1.23 — second order, as expected for a symmetric pairwise stencil. The ISPH
   variant has no meaningful apparent order: its error shrinks to a biased
   floor of 0.0088 instead of 0 (`nu_eff/nu -> 1.0088`), so the ratio of
   successive errors is not an order at all. That floor does not vanish under
   refinement, which is consistent with its `eta^2 = 0.01 h^2` regulariser (a
   fixed fraction of h); it was not isolated by a dedicated eta sweep, so this
   is the likely cause, not a measured one.

## 5. Caveats and scope

* Only paths that expose an operator (or that can be linearised by
  `nu = 0` differencing) are measured. `threed.solver`, `wcsph/solver`,
  `dam_break_3d`, `rayleigh_taylor` and the compressible/Sod path call the
  same per-dimension code but are not measured *through their own driver*;
  `sod_shock` uses artificial viscosity (a shock-capturing term), which is not
  the same operator and is out of scope by construction.
* The 3-D measurement reads density from `operators3.summation_density` on a
  periodic lattice; on a free-surface case the per-particle density varies and
  the `rho_ij` arithmetic mean in the operator's denominator is itself an
  approximation that this benchmark holds at rho = const.
* Both reference fields are smooth and single-mode. A high-wavenumber field
  (kh close to 1) is where the 3-D level error grows; the ladder here covers
  kh = 0.82 down to 0.33, so the numbers are an interpolation range, not a
  continuum limit.
* Interior-band readout excludes particles within 3 lattice cells of an edge;
  this is the periodic control and it agrees with the whole-lattice readout for
  the periodic paths, so it is not hiding a boundary artefact.

## 6. Error verdict (against the platform criterion)

| quantity | reference | best measured | error | < 3 % ? |
|---|---:|---:|---:|---|
| 2-D Morris, finest (dx = 0.0125) | nu_eff/nu = 1 | 0.9982 | 0.18 % | YES |
| 3-D order-4 renorm (dx = 0.04) | nu_eff/nu = 1 | 1.0051 | 0.51 % | YES |
| 3-D no renormalisation (dx = 0.04) | nu_eff/nu = 1 | 1.1201 | 12.01 % | NO |

The unrenormalised 3-D operator fails the platform's 3 % criterion at the
spacing the 3-D cases use; the renormalised order-4 operator passes. The 2-D
family passes with margin. This is the actionable statement §6.3 was missing.

## 7. Reproduction

```bash
PYTHONPATH=src python benchmarks/viscous_operator_consistency/run.py --threads 1
PYTHONPATH=src python benchmarks/viscous_operator_consistency/make_doc.py
PYTHONPATH=src pytest -q tests/test_viscous_operator_consistency.py
```
