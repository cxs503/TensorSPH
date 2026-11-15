# The closed tank at rest: what the FAIL is, and why refinement cannot fix it

Case-pool anchor `CASE-SPH-LAKE-AT-REST` — reference: the analytic hydrostatic
identity (`u = 0`, flat free surface, `p(y) = rho0 g (H - y)`), anchor tolerance
**machine precision (1e-9)**. Companion documents:

* `docs/verification_lake_at_rest.md` — the wall-bounded tank, honest FAIL, the
  certified vertical balance;
* `docs/verification_lake_at_rest_periodic.md` — the same shipped operator with
  `periodic_x=True` meets the tolerance by 4–5 orders (`2.6e-14` in `|v|/c0`);
* this document — belongs to the FAIL: **why** the closed tank cannot meet the
  tolerance, with the candidate cures measured and rejected.

Runner: `benchmarks/lake_at_rest/wall_gap.py`; evidence
`tests/data/lake_at_rest/wall_gap.json`; gates
`tests/test_lake_at_rest_wall_gap.py`; figures
`docs/figures/lake_wall_gap_resolution.png`, `docs/figures/lake_wall_gap_columns.png`.

Reproduce (≈1 min, four resolutions + the closure matrix):

```
PYTHONPATH=src python3 benchmarks/lake_at_rest/wall_gap.py \
    --json tests/data/lake_at_rest/wall_gap.json --figures docs/figures
```

## 1. Verdict

The closed-tank residual is **not a discretisation error**. It is
resolution-invariant (spread 1.23x over `dx = 0.05 … 0.02`, no trend), it is
carried entirely by the **free-surface row of the two side-wall columns**, and at
that row the particle's own pressure is clamped to zero — so the mass field the
reconstruction solves for has *no handle* on it. Four wall-face pressure fields
and three wall-mass conventions were measured; none closes the gap, and the shipped
Adami extrapolation is already the best of the pressure variants.

Consequence for the anchor: the closed tank stays **FAIL**, and the failure is
**not** removable by refinement (8 orders of magnitude short of the tolerance at
`dx = 0.02`, and unchanged at `dx = 0.005` were it run — the defect is
self-similar in `dx/h`). A well-balanced/consistent momentum operator (kernel
gradient correction or a boundary-integral wall treatment) is required; that is a
solver change, not a case change, and is out of scope here.

## 2. Case

| quantity | value |
|---|---|
| tank | `0.40 m × 0.30 m`, still-water depth `0.30 m` |
| spacing / smoothing | `dx = 0.050, 0.040, 0.025, 0.020 m`, `h = 1.3 dx`, kernel support `2h = 2.6 dx` |
| fluid particles | 48 / 70 / 192 / 300 |
| wall | 3 lattice layers on the two sides and the bottom, static, `wall_layers = 3` |
| fluid | `rho0 = 1000 kg/m3`, `c0 = 20 m/s`, `gamma = 7`, `g = 9.81 m/s2`, `viscosity = 0` |
| operator | `tensorsph.droplet_impact.rhs` (shipped), `density_calibration=True`, Adami wall-pressure extrapolation |

## 3. Method

1. **Reconstruction** (unchanged, `tensorsph.lake_at_rest.reconstruct`): damped
   Gauss–Newton on `u = log(m/m_unit)` per mirror pair, residual `a_y` of every
   fluid particle from the shipped `rhs`, Tikhonov pull towards the analytic
   profile. It converges at every resolution (`a_y = 1.3e-13 … 3.5e-13`) — the
   vertical balance is closable, so the horizontal number is not an artefact of an
   unconverged solve.
2. **Faithful re-implementation of the horizontal sum**, used only to *split* the
   residual. It is validated against the shipped `residual_report` at every
   resolution (identical to `1e-12` relative; the live check is
   `test_recorded_numbers_come_from_the_shipped_operator`). It keeps the shipped
   pair list, masses, densities, pressures and wall-face pressure and re-orders
   nothing; the only thing it can vary is the wall-face pressure and the wall mass.
3. **Term split**: `a_x,i = -sum_j m_j (c_i + c_j) dW_ij/dx` written as
   `own_pressure_term` (`-c_i sum_j m_j dW_ij`) plus `neighbour_pressure_term`
   (`-sum_j m_j c_j dW_ij`), `c = p/rho^2`.
4. **Candidate cures measured**: wall-face pressure `adami` (shipped),
   `dry_clip` (a wall particle with no fluid neighbour at or above its own level
   carries `p = 0`), `mirror` (the fluid pressure at the mirrored position about
   the wall plane), `hydrostatic` (`rho0 g (H - y)`); wall mass `nominal`
   (lattice, shipped), `eos` (`m_unit rho_w/rho0`), `mirror_mass` / `row_mass`
   (the reconstructed fluid mass of the mirrored particle / of the wall column's
   own row); shipped operator options `boundary_consistent_density`,
   `density_renormalization`, `surface_pressure_closure`.

## 4. Results

### 4.1 Resolution ladder — the residual does not refine away

| `dx` [m] | fluid particles | `max a_y/g` | `max a_x/g` (wall band) | `max a_x/g` (interior) | vs tolerance |
|---|---|---|---|---|---|
| 0.050 | 48 | `1.34e-13` | **`1.0919e-01`** | `1.049e-02` | 8.0 orders short |
| 0.040 | 70 | `2.05e-13` | **`1.2029e-01`** | `7.169e-03` | 8.1 orders short |
| 0.025 | 192 | `3.16e-13` | **`1.0132e-01`** | `1.059e-02` | 8.0 orders short |
| 0.020 | 300 | `3.55e-13` | **`1.2418e-01`** | `9.831e-03` | 8.1 orders short |

Spread `max/min = 1.23`, no monotone trend: the wall-band residual is a property of
the *operator*, not of the grid.

### 4.2 Where it lives

Per-column maximum for `dx = 0.02` (20 columns, wall at `x = 0.01` and `x = 0.39`):

`0.1242, 0.0528, 0.0234, 0.0098, 0.0039, 0.0015, 0.0006, 0.0003, 0.0002, 0.0001,
0.0001, 0.0002, 0.0003, 0.0006, 0.0015, 0.0039, 0.0098, 0.0234, 0.0528, 0.1242`

— two orders of magnitude between the wall column and the tank centre, mirror
antisymmetric to `2.2e-12 … 9.8e-12` (`a_x(W-x) = -a_x(x)`), and the worst particle is always
on the **free-surface row** of the wall column (`y = 0.275 / 0.260 / 0.2875 /
0.290` for the four resolutions).

### 4.3 Why no mass field can cancel it

At the worst particle the pressure is *exactly zero* — the Tait clamp puts the
truncated surface density (`rho/rho0 = 0.924 … 0.966`) below `rho0`:

| `dx` | `p` at worst [Pa] | `rho/rho0` | `own_pressure_term/g` | `neighbour_pressure_term/g` | total/g |
|---|---|---|---|---|---|
| 0.050 | 0.0 | 0.944 | `0.000e+00` | `-1.092e-01` | `-1.092e-01` |
| 0.040 | 0.0 | 0.961 | `0.000e+00` | `-1.203e-01` | `-1.203e-01` |
| 0.025 | 0.0 | 0.925 | `0.000e+00` | `-1.013e-01` | `-1.013e-01` |
| 0.020 | 0.0 | 0.966 | `0.000e+00` | `-1.242e-01` | `-1.242e-01` |

The residual is 100 % the *neighbour* pressure sum: the sub-surface neighbours on
the interior side push the surface particle inward and the outside of the support
(the wall, where the free-surface condition also forces `p = 0`) cannot answer.
Since `c_i = p_i/rho_i^2 = 0`, changing `m_i` (the only unknown the reconstruction
has) cannot change this particle's horizontal force at all. This is the quantitative
content of finding F1 in `docs/verification_lake_at_rest.md` ("the reconstruction
has one degree of freedom per particle and the horizontal force is odd under
reflection").

### 4.4 The candidate cures, measured

Wall-face pressure, factor on `max a_x/g` (1.0 = no change; `<1` better):

| `dx` | `dry_clip` | `mirror` | `hydrostatic` |
|---|---|---|---|
| 0.050 | 1.000 | 1.176 | 0.898 |
| 0.040 | 1.000 | 1.222 | 1.805 |
| 0.025 | 1.000 | 1.191 | 0.847 |
| 0.020 | 1.000 | 1.231 | 1.022 |

* `dry_clip` is a **no-op**: the Adami extrapolation already returns `p = 0` for
  wall particles above the free surface (the body-force term makes the source
  negative and it clamps). The "wet wall above the surface" artefact is *not* the
  mechanism.
* `mirror` (the exact continuation a symmetric tank would produce) is **worse** by
  18–23 %: the imbalance is not a mis-set wall pressure; making the wall pressure
  *more* correct makes the residual larger, which is only possible if the defect is
  in how the sum is weighted, not in the values it is weighted by.
* `hydrostatic` swings both ways (0.85 … 1.81) and is not a cure.

Wall mass (`scripts/probe_wall_mass.py`, `PYTHONPATH=src:benchmarks/lake_at_rest`),
`dx = 0.04`: `nominal 1.2029e-01`, `eos 1.2085e-01`, `mirror_mass 1.3026e-01`,
`row_mass 1.2965e-01`. The reconstructed fluid mass reaches `1.50 m_unit` at the
wall column while the wall particle keeps `1.00 m_unit`, but closing that gap makes
the residual *worse*: the mass asymmetry is a symptom, not the cause. (The
`adami` volume convention `m_w = rho_w / sum_f W_fw` is **not** usable as stated:
for the outermost wall lane the fluid kernel sum underflows, giving masses of
`1e32 m_unit`; it is reported here only as a rejected attempt.)

Shipped operator options (`dx = 0.04`):

| option | `max a_y/g` | `max a_x/g` | converged | verdict |
|---|---|---|---|---|
| baseline | `2.05e-13` | `1.203e-01` | yes | reference |
| `boundary_consistent_density` (Adami two-pass) | `2.14e-13` | `1.181e-01` | yes | 1.8 % better, gap open |
| `density_renormalization` (Shepard) | `9.43e+01` | `2.21e+01` | yes | destroys the vertical balance |
| `surface_pressure_closure` | `2.11e-01` | `1.77e-01` | no | destroys the vertical solve |

## 5. Figures

| figure | content |
|---|---|
| `docs/figures/lake_wall_gap_resolution.png` | `max |a_x|/g` vs `dx` for the three wall-pressure variants and the anchor tolerance line: flat, four orders above tolerance, and the mirror/hydrostatic variants do not cross it |
| `docs/figures/lake_wall_gap_columns.png` | per-column `max |a_x|/g` at `dx = 0.04`: a two-decade wall-to-centre decay |

## 6. Error judgement against the anchor

| quantity | reference | measured | error | `< 3 %`? |
|---|---|---|---|---|
| `max a_y/g` (vertical equilibrium) | analytic `rho0 g (H-y)` | `2.05e-13` (dx 0.04) | `2e-13` | **yes** (machine precision) |
| `max a_x/g` (horizontal equilibrium) | 0 (analytic identity) | `1.20e-01` (dx 0.04) | 120 % of `g` | **no** |
| `max |a_x|/g` vs anchor tolerance | `1e-9` | `1.20e-01` | 8.1 orders short | **no** |

The anchor's acceptance (machine precision on the zero-velocity/zero-wave
identity) is **met only in the laterally periodic tank** and **not met in the
closed tank**; this document does not change either verdict, it explains the second
one and closes the question "would a finer grid help?" — it would not.

## 7. Notes and limitations

* The mechanism is identified at the *free-surface × side-wall corner*; the same
  self-similar truncation is what the periodic run removes. It is a defect of the
  shipped momentum operator's consistency near a wall, not of the case
  configuration: any wall-bounded SPH rest state with a free surface touching the
  wall will show it.
* The probe re-implementation reproduces the shipped horizontal residual to
  `1e-12` relative, but it is used only for *attribution*: every gate in
  `tests/test_lake_at_rest_wall_gap.py` also re-derives one entry live from
  `tensorsph.lake_at_rest` so the evidence cannot silently drift from `src/`.
* `interior max a_x/g` is `~1e-2`, i.e. the defect is not confined to the exact wall
  column but decays over `~2h`; that is why the wall-band gate (`2h` band) is the
  right instrument and why the periodic variant is the only configuration that
  certifies the anchor.
* Not done here (deliberately out of scope): the operator change that would close
  the gap — a first-order-consistent (kernel-gradient-corrected) or boundary-integral
  pressure gradient. It is a solver capability, needs its own conservation audit
  (the correction is not momentum-conserving), and should be a separate ledger
  entry: `FEAT-SPH-WELL-BALANCED-WALL-GRADIENT`, acceptance "closed-tank
  `CASE-SPH-LAKE-AT-REST` `max |a_x|/g < 1e-9` at `dx = 0.02`, with the dam-break /
  hydrostatic gates unchanged".

## Update (closure attempt): the named cure, implemented and measured

The "out of scope" sentence in section 1 promised that "a well-balanced /
consistent momentum operator (kernel gradient correction ...) is required".
That operator now exists (`src/tensorsph/wellbalanced.py`) and the promise has
been kept and measured (`docs/verification_lake_at_rest_wellbalanced.md`):

* on a **strictly linear** pressure field the consistent operator returns
  `a_x/g < 1e-15` in the wall band at every resolution, where the shipped pair
  form returns `5.4e-3 ... 1.0e-1` -- **the operator claim in this document is
  correct, and now exact rather than plausible**;
* on the **shipped equilibrium** (Tait pressure of the reconstructed mass
  field) the same operator is *worse* (`0.301` vs `0.109 g`), because the field
  it must differentiate departs from a plane by `0.49 ... 0.56 p_max` across
  the wall-band supports -- the free-surface `max(p,0)` clamp, not the gradient
  operator, is what re-introduces the force;
* a mass field fitted against *both* acceleration components still cannot
  close it (`a_x` `0.097 g` only by giving up `a_y` to `0.122 g`), a set-level
  confirmation of section 4.3.

So the FAIL below stands, and the next attack is the density/pressure field
(well-balanced density decomposition), not another gradient operator.
