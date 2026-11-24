# CASE-SPH-CONDUCTION-2D — second benchmark family for the conduction operator

**Anchor**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` · **Repo**: TensorSPH
**Artefact**: `benchmarks/case_conduction_2d/result.json` · **Harness**:
`benchmarks/case_conduction_2d/run.py` · **Figure**: `mode_decay.png`
*(every number below is rendered from the artefact by
`docs/gen_case_conduction_2d.py`; none is typed in)*

## Why this case exists

`heat_conduction_operator` entered the method-selection registry on a single
benchmark family — the 1-D `erfc` similarity solution
(`CASE-SPH-CONDUCTION-1D`). The fragility audit
(`docs/xval_method_selection_fragility_m21797.md`) flagged that upper bound as
*single-source* and asked for a second family. This case supplies one, chosen so
that it can fail for a reason the 1-D case is blind to: **dimensional
consistency of the pair sum, and transferability of the lattice calibration.**

## The case

Periodic square `[0, L]²`, one Fourier mode `T = cos(kx x) cos(ky y)` with
`kx = ky = 2π/L`, `L = 1.0 m`, `α = 1.0 m²/s`,
`ρ = c_p = 1` (so `k = α`). Exact solution
`exp(-α (kx²+ky²) t) cos(kx x) cos(ky y)` (substitute into `∂_t T = α ∇²T`; the
periodic solution of the linear heat equation, Carslaw & Jaeger 1959, Ch. II).
`t_end` is fixed by the amplitude: `A(t_end)/A(0) = 0.1`;
`h/dx = 1.3`, Courant `0.2`,
`η = 0.01 h`.

The operator is the shipped Cleary & Monaghan (1999) pair sum lifted to 2-D with
the vector pair distance (`tensorsph.heat_conduction_2d`):

    dT_i/dt = (2/ρ) Σ_j V_j · 2k_i k_j/(k_i + k_j) · F_ij · (T_i − T_j).

## Results

| arm | integrator | n=32 | n=64 |
|---|---|---|---|
| `none` | euler | 772 % | 833 % |
| `lattice_1d` | euler | 774 % | 835 % |
| `k0_2d` | rk4 | 2.42 % | 0.602 % |
| `k0_2d` | euler | 0.644 % | 0.159 % |
| `mean_2d` | rk4 | 2.33 % | 0.513 % |
| `mode_2d` | rk4 | 1.08e-07 % | 4.21e-10 % |

The **banked record** is the 2-D-calibrated RK4 arm at `n = 128`
(`dx = 0.007812 m`, 2389 steps):

| quantity | value |
|---|---|
| max normalised field error (vs amplitude left) | **0.1504 %** (gate 3 %) |
| same, vs initial amplitude | 0.01503 % |
| decay-rate error | 0.1504 % |
| calibration factor `α_eff/α` (2-D, k→0) | 0.0150452 |
| convergence order (`k0_2d`, RK4) | 2.01 |
| raw operator `α_eff/(α dx)` | n=32: 1.9258, n=64: 1.9258 |
| symbol isotropy spread | n=32: 6.16 %, n=64: 1.56 % |
| stencil vs dense pair sum (max abs) | 2.89e-15 |
| 2-D code vs shipped 1-D `conduction_rhs` | 0 (exact) |
| independent Crank–Nicolson vs closed form | n=32: 0.671 %, n=64: 0.181 % |

## Findings

1. **The raw operator is `O(dx)` too slow in 2-D.** `α_eff/(α dx)` is constant
   to four digits across the ladder (n=32: 1.9258, n=64: 1.9258), i.e. `α_eff/α → 0` as `dx → 0`: the
   factor `2` and the volume `V_j = dx` of the 1-D layer sum do not carry over to
   a 2-D pair sum. The 1-D form is *not* a 2-D Laplacian.
2. **The shipped 1-D calibration therefore does not transfer**: applied to the
   2-D operator it mis-scales the run by ~`1/dx` — the `lattice_1d` arm sits at
   835 % worst. It is banked as a
   **`refuted`** registry record with a precondition, so the trap is on the record
   and cannot be re-derived as an accuracy claim.
3. **Redoing in 2-D what the 1-D solver already does in 1-D restores
   consistency**: dividing the operator by the 2-D `k → 0` symbol gives
   0.1504 % at `n = 128` with second-order
   convergence (2.01) — inside the 3 % gate and the owner 15 % standard.
4. **At coarse `dx` the integrator, not the operator, dominates** if forward
   Euler is used: compare the `k0_2d` rows in the table — Euler's first-order time
   error at `dt = 0.2 dx²/α` is of the same size as, and
   opposite in sign to, the spatial error, so the Euler arm *looks* better than
   it is. The banked reading uses RK4 for exactly that reason, and both arms are
   on the record.
5. **The discrete symbol is isotropic to 2nd order**: the ratio spread over the
   mode table falls n=32: 6.16 %, n=64: 1.56 % — the square lattice's anisotropy is truncation error,
   not a floor.
6. **Equivalence checks** (why this reads the shipped operator and not a
   re-implementation): the `torch.roll` fast path matches the dense pair sum to
   2.9e-15; the degenerate `ny = 1` case matches the
   shipped 1-D `conduction_rhs` **exactly** (0.0); the product mode is an
   eigenfunction of the lattice operator with the analytically predicted rate
   (residual `< 1e-10` relative, asserted in `tests/test_conduction_2d.py`).
7. **Independent reference**: a periodic 5-point Crank–Nicolson solve of the same
   IVP reproduces the closed form to n=32: 0.671 %, n=64: 0.181 %, so the target is checked by a second
   discretisation rather than asserted.

## Limitations / not claimed

* uniform conductivity only (`k_i = k_j`): the harmonic pair mean is exercised
  trivially here; the multi-material question stays open in 1-D and 2-D alike;
* periodic domain — no wall/ghost-layer question in 2-D (the 1-D family owns the
  mirror-ghost wall treatment);
* the calibration is a *lattice* property: uniform square lattice only; irregular
  particle distributions are unmeasured;
* one mode `(1,1)` and one amplitude window (`A(t_end)/A(0) = 0.1`): the symbol
  error grows with `k dx`, so shorter wavelengths will read worse. The `mode_2d`
  arm's exactness is a construction (it *is* the symbol), not a measurement of the
  scheme — it is in the table only as the eigen-relation check.
* the ladder in the committed artefact covers `n = [32, 64]` (plus the `n = 128`
  main run); a deeper ladder was queued on the cluster (cj3762) but not returned
  before this record was banked — the missing rungs are marked `—`, not guessed.

## How to reproduce

    PYTHONPATH=src python3 benchmarks/case_conduction_2d/run.py --figure
    PYTHONPATH=src python3 docs/gen_case_conduction_2d.py    # regenerates this file
    PYTHONPATH=src python3 -m pytest tests/test_conduction_2d.py -q

### Fixed-temperature slab with periodic transverse direction

On a unit square with unit diffusivity, set the left and right wall temperatures
to 3 and 1. The exact transient is
`3 - 2x + sin(πx) cos(2πy) exp(-5π²t)`. The lattice uses the long-wave
diffusion calibration, h/dx=1.3, and explicit midpoint time integration.
At t=0.02 the absolute temperature errors were:

| Particles per side | Maximum over particles | Maximum in wall-adjacent columns |
|---:|---:|---:|
| 16 | 0.00945503 | 0.000931239 |
| 32 | 0.00240267 | 0.000118036 |
| 64 | 0.000603009 | 0.0000148030 |

At 64 particles per side, halving the time step from 0.0000390625 changes
the maximum error to 0.000602839. The zero-mean transverse perturbation keeps
the mean temperature at 2. Reproduce from the repository root with
`PYTHONPATH=src OMP_NUM_THREADS=1 python -m benchmarks.case_conduction_2d.wall_decay`.

Each x wall also accepts a vector of temperatures sampled at y cell centres.
For a unit square with left wall cos(2πy) and right wall 0.5 sin(2πy), the
exact harmonic field interpolates the two profiles with weights
sinh(2π(1−x))/sinh(2π) and sinh(2πx)/sinh(2π). Initialized from this
equilibrium and evolved to t=0.02, the maximum absolute errors at 16, 32 and
64 particles per side were 0.022435, 0.006413 and 0.001701. The maximum
occurred in the wall-adjacent columns at each resolution. Reproduce with
`PYTHONPATH=src OMP_NUM_THREADS=1 python -m benchmarks.case_conduction_2d.wall_profile`.

### Heat exchanged with each wall

`stencil_rhs_with_wall_power_2d` returns the temperature rate and the two
powers delivered to the slab, summed over particle pairs crossing the left and
right walls. For unit thickness and unit specific heat the sum of the powers
equals the rate of change of total thermal energy. Multiply both outputs by
the same calibration factor when calibrating the operator.

For the unit slab, T=5−4x+sin(πx) exp(−π²t) gives exact integrated inward
heats +0.02298074 and −0.13701926 through t=0.02 at unit conductivity.
On square grids of 16, 32 and 64 particles per side, the larger absolute error
of the two wall heats was 1.8191e−4, 4.5503e−5 and 1.1383e−5, respectively.
The discrepancy between total stored heat change and accumulated inward wall
heat was below 5.3e−16 on all three grids. Reproduce with
`PYTHONPATH=src OMP_NUM_THREADS=1 python -m benchmarks.case_conduction_2d.wall_heat_budget`.

### Periodically driven wall

The slab benchmark integrator accepts a wall-temperature function of time and
samples it at the beginning and midpoint of each explicit midpoint step.
For a unit-diffusivity slab with left wall cos(ωt) and right wall zero, the
time-periodic solution is the real part of
`exp(iωt) sinh(q(1−x))/sinh(q)`, where `q=sqrt(iω)`. Starting from
that field at t=0 and advancing to t=0.03 for a period of 0.08, the maximum
absolute temperature errors at 16, 32 and 64 particles along x were
0.040104, 0.010163 and 0.002535. The largest error occurred adjacent to
a wall at each resolution. The transverse field is uniform and periodic.
Reproduce with
`PYTHONPATH=src OMP_NUM_THREADS=1 python -m benchmarks.case_conduction_2d.wall_cycle`.

### Finite thermal contact between grid-aligned layers

For `layered_x=True`, pass the areal face resistances using
`contact_resistance_x` and the grid spacing `dx`. Every pair crossing a
contact includes its resistance once. With fixed walls the vector has nx−1
entries. Periodic x adds one entry for the seam. The normal series-resistance
model applies to transport across the layers.

A two-rod decay mode has zero wall temperatures, unit conductivity and heat
capacity, and areal resistance R=2/a at x=0.5, where a=3π/2. The initial
temperature perturbations sin(ax) and −sin(a(1−x)) decay by exp(−a²t);
the interface flux is continuous and the jump equals R times that flux.
At t=0.015 the maximum absolute temperature errors on 16, 32 and 64 cells
were 0.005860, 0.001543 and 0.0003935. Absolute errors in the jump using
one-sided linear extrapolation were 0.05310, 0.01278 and 0.003120.
Reproduce with
`PYTHONPATH=src OMP_NUM_THREADS=1 python -m benchmarks.case_conduction_2d.contact_decay`.

The lattice temperature-rate and wall-power entry points accept a positive
scalar or broadcastable particle field `specific_heat` in J/(kg K). Storage
is weighted by density times specific heat. Wall power depends on transfer
through the particle pairs and can be checked independently against the
capacity-weighted sum of the temperature rates.

The contact-decay command above also runs unequal materials: left (k,c)=(1,1)
and right (k,c)=(4,256/81) at unit density. The right half-wave uses b=4π/3,
amplitude a cos(a/2)/(4b cos(b/2)), so its decay rate and contact flux match
the left half-wave. At 16, 32 and 64 particles along x, the maximum temperature
errors were 0.005510, 0.001473 and 0.0003794 K; the interface-jump errors were
0.03573, 0.008680 and 0.002131 K. Absolute errors in the change of stored heat
per unit cross section were 4.866e-5, 1.158e-5 and 2.836e-6 J/m².

For a fixed-wall slab, `stencil_x_face_power_2d` sums the power delivered
from left to right across an interior column boundary, counting each
crossing pair once. The face index is the number of physical columns to its
left. Use the same stencil and material/contact data as the temperature
update. Divide by the represented wall area and apply the same diffusivity
calibration to compare with a continuum heat flux. The regression
`test_every_internal_cut_closes_the_left_slab_heat_balance` compares
every cut with left-wall input minus the rate of stored heat change on its
left, including cuts within the kernel radius of a wall.
