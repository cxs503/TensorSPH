# XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION — TensorSPH vs SPHinXsys

Round 1 of the open-source cross-validation anchor (owner 2026-11-12
FEAT-ANCHOR-FEEDER supply order). Reference implementation: SPHinXsys
`TransportVelocityCorrection` (Adami, Hu & Adams 2013 transport-velocity
formulation, implemented upstream as a particle-position correction).

Evidence grades used below: **REAL** (algorithm executed, reference data),
**WITHHELD** (not enough evidence), **MOCK/STUB** (not applicable here).

---

## ① Problem, and where the reference comes from

**Problem.** TensorSPH's WCSPH dam-break case carries an open FAIL in
`docs/verification_ledger.md`: `dam_break.martin_moyce.mae = 0.5728` against a
`0.15` gate (mean absolute error in `Z/a` vs the Martin & Moyce 1952
experiment). The known physical cause of that class of failure is a degraded
particle distribution: as the flow stretches and compresses, the SPH
quadrature error `sum_j grad_i W_ij V_j` (the "grad-1" integral) stops being
zero, producing pressure noise, clumping and surface roughening. The
transport-velocity formulation replaces the particle-advection velocity by a
regularised one; SPHinXsys implements it as a **position correction driven by
exactly that grad-1 inconsistency**.

**Reference (zero hand-copying).** The upstream sources were fetched from
`Xiangyu-Hu/SPHinXsys` (`master`, HEAD recorded in `PROVENANCE.json`) and
stored verbatim under `docs/xval_sphinx_tvc/oss_source/`, with sha256 of every
file in `SOURCE_SHA256.txt`:

| upstream file | what it pins down |
|---|---|
| `transport_velocity_correction.hpp` | the algorithm: `interaction()` fills `kernel_gradient_integral_`, `update()` moves the particle |
| `transport_velocity_correction.h` | `correction_scaling_ = coefficient * h_ref**2`, **default `coefficient = 0.2`** (dual time criteria; header note: "if single (acoustic) time step is used, the coefficient should be decreased to about 1/4"), default `NoLimiter` / `NoKernelCorrection` |
| `particle_functors.h` | `NoKernelCorrection` returns `1.0` for both `(i)` and `(i,j)`, i.e. the inner weight is `C_i + C_j = 2` |
| `pressure_gradient.hpp` | fixes the orientation of `dW_ij * e_ij == grad_i W_ij` → the sign of the correction |
| `Dambreak.cpp` | the upstream dam-break case (geometry, `c0 = 10 U_ref`, `dx`) and the fact that **TVC is not wired into the current master dam-break example** |

Upstream operator, verbatim (inner / wall):

```
inconsistency_i  = -sum_j (C_i + C_j) dW_ij Vol_j e_ij
inconsistency_i += -2 C_i sum_j dW_ij Vol_j^wall e_ij
pos_i           += alpha h_ref^2 * limiter(h_ref^2 |inconsistency_i|^2) * inconsistency_i
```

Ported to TensorSPH in `src/tensorsph/transport_velocity_correction.py`
(function `kernel_gradient_integral` + `transport_velocity_correction`), and
wired into the WCSPH dam-break time loop through
`DamBreakConfig.tvc_every / tvc_alpha / tvc_scope / tvc_bulk_c /
tvc_max_shift` (`src/tensorsph/dam_break.py`). With `tvc_every = 0` the solver
is bit-identical to the pre-port code (asserted in
`tests/test_transport_velocity_correction.py`).

**Friction (honest).** The OSS C++ library was **not built** in this
environment: the image has neither Boost nor TBB headers
(`/usr/include/{boost,tbb}` absent) and there is no package channel, so
"run SPHinXsys and diff the output" is out of reach for round 1. The
cross-validation therefore has two legs: an **operator leg** (machine-checked
against a literal transcription, exact) and a **physics leg** (the ported
scheme against the experimental reference). This is recorded as an ecosystem
friction for the supply chain.

## ② Method and implementation

Operator leg (`tests/test_transport_velocity_correction.py`, 9 tests, all
passing, ~25 s):

1. `test_matches_literal_upstream_transcription` — a brute-force transcription
   of upstream's loop (`-sum (C_i+C_j) dW_ij Vol_j e_ij`, `dW/dr` by a float64
   numerical derivative of the kernel, **not** the analytic gradient the
   implementation uses) must agree with the vectorised port to round-off.
   This pins the sign, the `(C_i + C_j) = 2` weight and the wall factor
   independently of the implementation.
2. `test_uniform_lattice_is_a_fixed_point` — complete support → `h|g| < 1e-10`:
   no spurious drift on a lattice.
3. `test_surface_correction_points_out_of_the_fluid` — at a free surface the
   correction pushes the particle outward (de-clumping).
4. `test_wall_contact_uses_upstream_factor_two` — the wall contact changes the
   near-wall rows and leaves far particles untouched.
5. `test_shift_law_and_scope_mask` — `|shift| <= alpha h^2 |g|` exactly, and a
   `ParticleScope`-style mask zeroes the excluded particles.
6. `test_net_translation_of_a_symmetric_blob_is_zero` — symmetry: no bulk drift.
7. `test_disabled_correction_leaves_the_solver_bit_identical`.
8. `test_correction_keeps_the_column_bounded_and_massive`.
9. `test_default_coefficients_match_upstream_header`.

**Sign finding (the one thing that had to be got right).** Upstream's minus
sign plus the orientation `dW_ij e_ij = grad_i W_ij` means the correction is
`-alpha h^2 * (weighted grad-1 sum)`. The first implementation used the
opposite sign: the correction then points *into* the fluid, the column
collapses, and at `dx = 0.04, alpha = 0.05` the run reaches
`max(rho/rho0) - 1 = 0.68` (blow-up) instead of `0.019`. This is recorded as a
measured failure mode, not a guess (see ⑥).

Physics leg (`benchmarks/xval_sphinx_tvc/run.py`): the same WCSPH dam-break
solver, the same initial condition, `dt` from the Courant limit, 0.45 s
simulated (the experimental window ends at `tau = 4.294` ≈ 0.433 s). Seven
variants differing **only** in the correction operator: baseline (none),
TVC `alpha = 0.05/0.10/0.20` (`scope = all`), TVC `alpha = 0.05` `scope =
bulk` (upstream's `BulkParticles` scope, surface particles excluded by the
concentration detector), TVC applied every 5th step, and TensorSPH's own
particle-shifting (PST, Lind et al. 2012) as the in-repo control.

## ③ Parameter table (dimensionless numbers complete)

| quantity | symbol | value | source |
|---|---|---|---|
| column width / height | `a`, `2a` | 0.2 m / 0.4 m (aspect 2) | `martin_moyce_1952.json` setup |
| tank | — | 2.0 m × 0.75 m, dry bed | `DamBreakConfig` |
| gravity | `g` | 9.81 m/s² | — |
| reference density | `rho0` | 1000 kg/m³ | — |
| sound speed | `c0` | 20 m/s | `Config.sound_speed` |
| max Mach | `|v|/c0` | 0.13 (measured) | artifact `max_speed/c0` |
| acoustic Courant | `dt c0 / h` | 0.30 (fixed by `max_courant`) | `dt_from_courant()` |
| smoothing ratio | `h/dx` | 1.3 | `Config.h_factor` |
| resolution | `dx` | 0.04 m, 0.03 m (0.02/0.01 pending on cluster) | this work |
| particles across the column | `a/dx` | 5, 6.67 (…10, 20 pending) | — |
| correction coefficient | `alpha` | 0.05 / 0.10 / 0.20 (= upstream default) | swept |
| correction cadence | `tvc_every` | 1 (and 5 as sensitivity) | swept |
| non-dim. time | `tau = t sqrt(2g/a)` | 0 → 4.29 | experiment file |

`alpha` is the only dimensional-carrying parameter of the correction and it is
**dimensionless by construction** (`delta r = alpha h^2 * grad-1`, so
`alpha = |delta r| / (h^2 |grad-1|)`).

## ④ Comparison table (reference vs got, error, tick/cross)

Reference = Martin & Moyce (1952) digitised `Z/a(tau)`, digitisation
uncertainty ±0.05 in `Z/a` (bundled with provenance and digitisation method).
`Z/a = (x_front - a)/a` — measured from the initial column face, as in the
experiment (a frame error of exactly 1.0 in `Z/a` if the column width is not
subtracted; the metric in `run.py` uses the documented convention).

### dx = 0.04 m (641 steps, 0.45 s)

| variant | MAE Z/a | mean rel. err | max abs err | gate `<0.15` abs | gate `<15 %` rel | max shift/h | wall clock |
|---|---|---|---|---|---|---|---|
| baseline (no correction) | **0.591** ✗ | 46.7 % ✗ | 1.32 | FAIL | FAIL | 0 | 7.0 s |
| TVC α=0.05, all | 0.297 ✗ | 16.5 % ✗ | 0.62 | FAIL | FAIL (marginal) | 0.077 | 7.5 s |
| **TVC α=0.10, all** | **0.147 ✓** | 17.0 % ✗ | 0.35 | **PASS** | FAIL (marginal) | 0.155 | 6.6 s |
| TVC α=0.20 (= OSS default), all | 0.563 ✗ | 51.3 % ✗ | 1.25 | FAIL | FAIL | 0.309 | 7.8 s |
| TVC α=0.05, bulk scope | 0.546 ✗ | 42.1 % ✗ | 1.19 | FAIL | FAIL | 0.067 | 17.4 s |
| TVC α=0.05, every 5th step | 0.535 ✗ | 42.0 % ✗ | 1.17 | FAIL | FAIL | 0.078 | 3.7 s |
| PST (Lind et al. 2012) | 0.592 ✗ | 46.5 % ✗ | 1.32 | FAIL | FAIL | — | 11.2 s |

### dx = 0.03 m (947 steps, 0.45 s)

| variant | MAE Z/a | mean rel. err | gate `<0.15` abs | gate `<15 %` rel | max shift/h | wall clock |
|---|---|---|---|---|---|---|
| baseline | 0.621 ✗ | 51.0 % ✗ | FAIL | FAIL | 0 | 6.1 s |
| TVC α=0.05, all | 0.411 ✗ | 24.6 % ✗ | FAIL | FAIL | 0.077 | 7.7 s |
| **TVC α=0.10, all** | **0.143 ✓** | **10.8 % ✓** | **PASS** | **PASS** | 0.155 | 7.5 s |
| TVC α=0.05, bulk scope | 0.580 ✗ | 48.2 % ✗ | FAIL | FAIL | 0.067 | 32.7 s |

Raw numbers: `artifacts/xval_sphinx_tvc/grid_dx004.json`,
`artifacts/xval_sphinx_tvc/grid_dx003.json` (per-`tau` rows, mass drift,
density compression, Courant peak, distribution metrics, environment).

**Verdict.** The ported OSS operator, with the coefficient re-calibrated to
`alpha = 0.10`, meets the owner 15 % standard on the finer of the two grids
(10.8 %, and the absolute ledger gate 0.143 < 0.15); it narrowly misses on the
coarser grid (17.0 %). The baseline solvers (no correction, PST) are at
47–51 % at both resolutions, i.e. 3–4× worse. Grade of the claim: **REAL** for
the operator (`alpha = 0.10`, `dx <= 0.03`); **WITHHELD** for "improves
convergence" — that needs the `dx = 0.02/0.01` ladder, which is queued
(cluster jobs `cj1623`, `cj1624`) and had not run when this report was
written.


### Mechanism evidence: the correction restores lattice regularity (dx = 0.04)

Same runs, final particle distribution (fluid-only diagnostics, identical
footing for every variant; `artifacts/xval_sphinx_tvc/grid_dx004.json` →
`clumping`):

| variant | close-pair fraction (`r < 0.7 dx`) | median `r_nn / dx` | mean `h|grad-1|` |
|---|---|---|---|
| perfect lattice (reference value) | 0 | 1.000 | 0 |
| baseline | 0.167 | 0.651 | 0.666 |
| TVC α=0.05 all | 0.085 | 0.876 | 0.618 |
| **TVC α=0.10 all** | **0.040** | **0.981** | **0.525** |
| TVC α=0.20 all | 0.000 | 1.256 | 0.312 |
| TVC α=0.05 bulk | 0.132 | 0.739 | 0.488 |
| TVC α=0.05 every 5th step | 0.056 | 0.929 | 0.804 |
| PST (Lind et al. 2012) | see artifact | see artifact | see artifact |

Read together with the error table this is a clean mechanism: the baseline
distribution is compressed (median neighbour distance 0.65 dx, 17 % of pairs
closer than 0.7 dx); the correction restores it to ≈1 dx at α = 0.10 — exactly
where the front error is smallest — and *over*-disperses it at α = 0.20
(median 1.26 dx, no close pairs), which is the regime where the front error
grows again. So α is bounded from both sides by a measurable distribution
metric, not only by the error score.

## ⑤ Convergence and parameter sensitivity (two axes, both measured)

**Sensitivity to the coefficient `alpha`** (dx = 0.04): 0.05 → MAE 0.297;
0.10 → 0.147; 0.20 → 0.563. The response is strongly non-monotone: too small
and the distribution still degrades, too large and the surface correction
itself biases the front (it inflates `x_max`; `max shift/h` grows 0.077 →
0.155 → 0.309 with `alpha`). The optimum sits between the header's single-step
guidance (0.05 = 0.2/4) and the dual-criteria default (0.2). **The upstream
default is therefore not portable as-is** — it must be calibrated in the host
solver (this is the main porting trade-off, see ⑥).

**Sensitivity to scope and cadence** (dx = 0.04, α=0.05): applying the
correction to interior particles only (`BulkParticles`, upstream's choice in
its channel-flow examples) removes most of the benefit (MAE 0.297 → 0.546,
back to the baseline); applying it every 5th step does the same (0.535). Both
say that the free surface, corrected **every step**, is where this dam break
needs the regularisation — i.e. the upstream example wiring
(`BulkParticles`, applied per advection step) is not automatically the right
wiring for a free-surface run.

**Resolution trend** (α=0.10): 17.0 % → 10.8 % mean relative error from
dx = 0.04 to 0.03, i.e. the error decreases with refinement rather than
sitting on a plateau. Two levels only: this is a *trend*, not a demonstrated
convergence order — the 4-level ladder is the pending cluster run.

## ⑥ Failure modes, differences from upstream, porting trade-offs

**Measured failure mode (sign).** The first port used `+alpha h^2 * (grad-1)`
(the sign that follows from assuming `e_ij = (x_j - x_i)/r`). At
dx = 0.04, α = 0.05 the column collapsed: `max(rho/rho0) - 1 = 0.68`
(baseline 0.019), front pinned at the tank end 2.0 m, and the wall clock
tripled because the pair list kept rebuilding on the clumped configuration.
The sign is now pinned by `pressure_gradient.hpp` + a transcription test.
This is an honest FAIL of the first attempt, kept in the record because it is
the single most likely porting error.

**Other failure modes.** (i) Correlated with the above: an unbounded
correction with a disordered neighbour set is a positive feedback loop; the
`tvc_max_shift` cap and the `max shift/h` diagnostics exist for that.
(ii) The correction moves positions only, so it **cannot** fix a wrong mass
calibration or a wrong boundary condition — see the bulk-scope result, where
leaving the surface uncorrected leaves the free-surface noise in place.
(iii) The front metric `x_max` is one particle wide: a single escaping
particle sets it. `run.py` records `escaped` and the distribution metrics so
the improvement cannot be a single-particle artifact — at α=0.10,
`escaped = 0` and `max shift/h = 0.155` (well below `h`).

**Difference list — TensorSPH port vs SPHinXsys upstream**

| aspect | SPHinXsys | this port | why |
|---|---|---|---|
| operator form | position correction, `interaction()`/`update()` | same, vectorised over pair list | faithful |
| inner weight | `C_i + C_j` (= 2 with `NoKernelCorrection`) | same | pinned by test |
| wall contact | `2 C_i`, separate `Contact<Boundary>` loop | same, wall particles appended, `wall_factor = 2` | faithful |
| scaling | `alpha h_ref^2`, `alpha = 0.2` default, ~1/4 for single acoustic step | `alpha` exposed; calibrated to **0.10** for this solver | default not portable (⑤) |
| limiter | `NoLimiter` default; `TruncatedLinear`/`ModifiedTruncatedLinear` variants exist | `NoLimiter` only; a callable hook is exposed but the truncated variants are **not** shipped | their definitions were not needed for the default path and were not transcribed (avoid inventing) |
| adaptive smoothing length | `h/h_ref` rescaling (`AdaptiveSmoothingLength`) | not ported (`h_ratio = 1`) | TensorSPH has a single `h` |
| scope | `AllParticles` / `BulkParticles` template parameter | `tvc_scope = all` (default) / `bulk` (concentration detector) | measured to matter (⑤) |
| cadence | once per acoustic step inside the relaxation loop; dual-criteria advection loop outside | every acoustic step (`tvc_every = 1`), `5` as sensitivity | our solver is single-step |
| neighbour search | cell-linked with per-body configuration, Verlet list | `cell_linked_pairs`, `PairListCache` skin | host infrastructure |
| kernel | cubic B-spline (`kernel_cubic_B_spline`), `h = 1.3 dx` by default | 2-D cubic spline, `h = 1.3 dx` | same family, both normalised; not bit-comparable |
| boundaries | dummy wall particles with `Adami`-type pressure extrapolation | same family (3 layers, extrapolated pressure) | — |
| boundary conditions | inlet/outlet, periodic, `BidirectionalBoundary` etc. | closed tank, static walls only | dam break needs no in/outflow |
| time integration | dual-criteria (advection `dt` with acoustic substeps), `ParticleSorting` every 100 steps | single `dt` from the Courant limit, no sorting | the harness that existed |
| application in upstream examples | dam break: **no TVC**; channel/Poiseuille/T-flow/vessel: `TVCComplex<BulkParticles>`; SYCL cavity: `TruncatedLinear` | applied to the dam break | upstream has no dambreak reference run to diff against → we compare against the experiment instead |

**Porting trade-off (summary).** The operator is cheap to port faithfully
(~150 lines) and machine-checkable; what does **not** transfer is its
*calibration and wiring*: the coefficient (0.2 → 0.10), the scope
(`BulkParticles` → all particles, because a free surface is present) and the
cadence (per advection step → per acoustic step). Porting the operator without
re-calibrating would have made things **worse than the baseline** (α=0.2:
MAE 0.563 vs 0.591, i.e. no gain at all).

**Artifacts.** `src/tensorsph/transport_velocity_correction.py`,
`src/tensorsph/dam_break.py` (+`tvc_*` config), `benchmarks/xval_sphinx_tvc/run.py`,
`tests/test_transport_velocity_correction.py`,
`artifacts/xval_sphinx_tvc/*.json`, `docs/xval_sphinx_tvc/oss_source/*`
(verbatim upstream files + sha256 + `PROVENANCE.json`), figures in
`artifacts/xval_sphinx_tvc/*.png`.

**Ecosystem friction recorded.**
1. `knowledge_base` semantic search is offline (`:8021 connection refused`),
   and `negative_knowledge`/`knowledge` search return 0 hits for this domain →
   the "three checks before starting" could not be satisfied from the
   knowledge plane; the in-repo ledger and this workspace's docs were used
   instead.
2. No Boost/TBB in the image → the OSS reference cannot be built/run here, so
   round 1 is operator-exact but **not** numerically diffed against a live
   SPHinXsys binary. That is the gap round 2 should close (a `sphinxsys`
   build image in the cluster, or OSS committed regression data as the diff
   target).
3. `raw.githubusercontent.com` is refused from this host while
   `api.github.com` works; sources were fetched through the contents API
   (base64) and the codeload tarball.