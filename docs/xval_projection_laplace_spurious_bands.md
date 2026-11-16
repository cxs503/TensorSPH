# Band attribution of the projection-SPH spurious current — instrument shipped, measurement WITHHELD

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · mission **m22813**
**Type** verification instrument (capability extension of the existing march) · **Status of the
number: WITHHELD** — this page carries no new physics reading and must not be cited as one.

## ① Why the single gated number is not enough

The dynamic ladder gates `max|u|/v_ref` over **all** particles against Adami et al. (2012)'s
`0.10`. That number cannot say *what* is being measured, and the two possibilities have opposite
consequences for the anchor:

* the residual lives in the **interface shell** — the curvature is a second derivative of the
  smoothed colour field, so its lattice noise is one order higher than the velocity it drives;
  such a residual is tied to the colour-field stencil, not to the flow;
* the residual lives in the **liquid bulk** — then the projection is not holding the drop at rest
  and the "static equilibrium" claim is not a property of the formulation.

## ② What was added

`src/tensorsph/projection_adjoint.py`:

| site | addition |
|---|---|
| `radial_band_stats(pos, u, radius, v_ref, edges, profile, profile_bins)` | splits `|u|/v_ref` into `liquid_core` (`r/R < 0.8`), `interface_shell` (`0.8 ≤ r/R ≤ 1.2`), `vapour_carrier` (`r/R > 1.2`); reports per-band `n`, max and mean (an empty band reads `n = 0`, not a zero speed); the geometry of the worst particle (`umax_r_over_R`) and whether it moves radially (`umax_radial_fraction`) — a radial mode changes the area, a tangential one does not; optional binned radial profile |
| `relax_projection_consistent(..., band_report=False)` | when switched on, every sampled history entry carries a `bands` dict. **Off by default**, so the shipped ladder records keep their schema and stay byte-reproducible |

`benchmarks/laplace_sph/diagnose_spurious_bands.py` runs the same march as the ladder (physics
constants are *imported* from `run_dynamic_ladder`, not re-typed) with `band_report=True`, writes
one record per level under `results_dynamic_ladder/spurious_bands/Rdx<N>.json`, and renders the
figure and a values-only page from those records.

## ③ Verification actually run

```
PYTHONPATH=src python -m pytest tests/test_projection_spurious_bands.py -q     # 6 passed
PYTHONPATH=src python -m pytest tests/test_projection_adjoint.py \
                                 tests/test_projection_balanced.py -q          # untouched, 26 passed
```

The gates pin the **instrument**, not the physics: the bands partition the cloud with inclusive
edges (a particle may not be dropped or double counted); exact rest reads `0.0` and an undefined
direction reads `nan` (never a fabricated `0.0`); a shell peak and a bulk peak with the *same*
global max are attributed differently (the reason the split exists); the radial and tangential
cases separate; the switch is off by default **and** the shipped ladder records carry no `bands`
key (checked against every `dx*_nu*.json` in the tree).

## ④ Why there is no number here

The measurement needs a marched run (120 projection steps on 780–7 000 particles). Every such run
is queued compute:

| job | what | state at hand-off |
|---|---|---|
| `cj4396` | this instrument, `R/dx` = 8, 12, 16 | **queued** (16 cores, submitted 02:58, still queued 23 min later) |
| `cj4390` | ladder `dx16_nu01` | queued |
| `cj4389` | ladder `dx24_nu01` | queued |
| `cj4391`…`cj4393` | horizon + `nu` sweep | queued |

A local attempt is refused by the compute policy (the box only allows < 8 cores and < 10 min of
interactive work, and this march is ~5 min *per level* on 6 threads plus the PCG solves), and the
foreground attempt was killed by the 120 s shell budget. So the honest state of the band
attribution is **WITHHELD**: instrument green, number absent.

The two-level evidence that *does* exist in the tree is analysed without any new solve on
`docs/xval_projection_laplace_residual_growth.md` (rise-then-plateau of the residual, Adami line
broken by a factor 2.9, Laplace jump inside the owner 15 % gate).

## ⑤ Reproduce (once the queue drains)

```bash
PYTHONPATH=src python benchmarks/laplace_sph/diagnose_spurious_bands.py \
    --levels 8,12,16 --threads 8 --render
```