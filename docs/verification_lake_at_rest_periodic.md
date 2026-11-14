# Lake at rest, laterally periodic: where the anchor tolerance is actually met

Case-pool anchor `CASE-SPH-LAKE-AT-REST` — *still water with a free surface must
stay at rest: no spurious current, no free-surface motion*. Reference: the
analytic hydrostatic identity (`u = 0`, `eta = const`); tolerance: machine
precision (the anchor has no experimental uncertainty and no fitted constant).

This document is the companion of `docs/verification_lake_at_rest.md`, which
keeps the **honest FAIL of the wall-bounded tank** and is not modified by this
work. Here the *side walls are removed instead of the physics*: the `x`
direction becomes periodic, the bottom wall stays, and the anchor tolerance is
met by 4–5 orders of magnitude in **both** acceleration components.

Artifacts (all produced by the commands in §2, no number below is typed in by
hand):

| file | content |
|------|---------|
| `artifacts/lake_at_rest/rest_dx020_periodic.json` | periodic, `dx = 0.02 m` — the certified run |
| `artifacts/lake_at_rest/rest_dx025_periodic.json` | periodic, `dx = 0.025 m` — second resolution |
| `artifacts/lake_at_rest/rest_dx020_wall.json` | wall-bounded, `dx = 0.02 m` — same-run reproduction of the documented FAIL |
| `artifacts/lake_at_rest/fieldmaps_dx020.npz` | per-particle residual/density fields behind the figures |

## 1. Verdict

**PASS in the laterally periodic tank** (`periodic_x=True`):

| gate | quantity | measured (`dx = 0.02`) | anchor tolerance | verdict |
|------|----------|------------------------|------------------|---------|
| G1 | equilibrium `max|a_y|/g` | `6.781e-13` | `< 1e-9` | pass |
| G2 | equilibrium `max|a_x|/g` | `1.377e-12` | `< 1e-9` | pass |
| G3 | rest `max|v|/c0` over 0.50 s | `2.569e-14` | `< 1e-9` | pass |
| G4 | surface `max|dy|/H` over 0.50 s | `9.904e-14` | `< 1e-9` | pass |
| G5 | density `max|drho|/rho0` over 0.50 s | `3.558e-13` | `< 1e-9` | pass |
| G6 | negative control: uniform lattice mass `max|a|/g` | `1.000` | `> 0.5` | pass |
| G7 | negative control: analytic profile `max|a_y|/g` | `0.2499` | `> 0.05` | pass |
| G8 | linear response `r(1e-4)/r(1e-3)` | `0.09982` | `0.05 … 0.15` | pass |

`ALL GATES PASSED: True`, both at `dx = 0.02 m` (300 fluid particles) and at
`dx = 0.025 m` (192 particles, §3). The **FAIL** verdict for the wall-bounded
tank stands unchanged (G2 `0.1242`, G3 `0.01644`, G4 `0.0836`, G5 `0.2353`,
reproduced bit for bit in `rest_dx020_wall.json`).

## 2. Method

1. **Operator capability (the only change to shipped code).**
   `tensorsph.droplet_impact.rhs(..., wrap_width=L)` reduces every pair
   separation to its minimum image in `x`, `dx - L round(dx/L)`, *before* the
   kernel is evaluated. Nothing else changes: the density summation, the wall
   pressure extrapolation and the viscous terms see the same wrapped
   separations. `wrap_width=None` is the previous code path (test:
   `test_wrap_none_is_the_previous_code_path`, bit-identical) and the wall-mode
   artefact reproduces the documented numbers exactly, so no published number
   moves.
2. **Pair list.** A periodic lattice cannot be enumerated by
   `cell_linked_pairs` (it cannot see the seam), so
   `lake_at_rest.periodic_pairs` enumerates every unordered pair whose
   *minimum-image* separation is inside `2h`, wall/wall pairs excluded. It
   requires `width >= 4h` so that one image per pair is exact (guarded).
3. **Same case, different unknown map.** The discrete problem is now
   translation invariant in `x`, so the equilibrium mass field cannot depend on
   the column: the Gauss-Newton unknowns are **one mass per row** (15 unknowns
   at `dx = 0.02`, 12 at `dx = 0.025`) instead of one per mirror pair (150). The
   tie is the content of the test, not a solver convenience — and it is
   falsifiable, see §4 F1.
4. **Reconstruction is unchanged.** Damped Gauss-Newton on `u = log(m/m_unit)`,
   residual `a_y` of every fluid particle from the *shipped* `rhs`, Tikhonov pull
   towards the analytic profile, backtracking line search. Converged in 14
   iterations at `dx = 0.02` (`4.67e-12 -> 6.78e-13` after the last step) and 29
   at `dx = 0.025`.
5. **Rest test.** Symplectic Euler at `dt = 3.900e-04 s`
   (`Courant = 0.30`, 1282 steps = 0.50 s), positions free to move, density
   re-summed from the moved positions every step.

Reproduce:

```
PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py --periodic-x \
    --json artifacts/lake_at_rest/rest_dx020_periodic.json
PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py --periodic-x --dx 0.025 \
    --json artifacts/lake_at_rest/rest_dx025_periodic.json
PYTHONPATH=src python3 benchmarks/lake_at_rest/make_figures.py
```

## 3. Numbers

Geometry and cost (periodic `dx = 0.02`): tank `0.4 x 0.3 m`, `h = 0.026 m`
(`1.3 dx`), 300 fluid particles, 60 bottom-wall particles (one column per fluid
column, three layers — the side walls are gone), 3000 pairs,
`rho0 = 1000 kg/m3`, `c0 = 20 m/s`, `gamma = 7`, `g = 9.81 m/s2`, acoustic Mach
`c0/sqrt(gH) = 11.66`.

Second resolution `dx = 0.025` (192 particles, 48 wall, 1920 pairs): G1
`3.330e-13`, G2 `6.874e-13`, G3 `2.080e-14`, G4 `3.368e-14`, G5 `1.007e-13`,
G6 `1.000`, G7 `0.2499`, G8 `0.09982` — all pass. The rounding-level result is
therefore not a property of one lattice.

Wall-bounded vs periodic at `dx = 0.02`, identical operator, identical lattice
rows, identical solver:

| quantity | side walls | x-periodic |
|----------|-----------|-----------|
| equilibrium `max|a_y|/g` | `2.944e-13` | `6.781e-13` |
| equilibrium `max|a_x|/g` | `1.242e-01` (wall band) | `1.377e-12` |
| rest `max|v|/c0` | `1.644e-02` | `2.569e-14` |
| rest `max|dy|/H` | `8.360e-02` | `9.904e-14` |
| rest `max|drho|/rho0` | `2.353e-01` | `3.558e-13` |

Auxiliary physics cross-check (the equilibrium that holds the lake still is not
the exact barotropic profile — it is a *discrete* equilibrium; interior band
`0.070 <= y <= 0.230 m`, 180 particles):

| comparison | periodic `dx=0.02` | periodic `dx=0.025` | side walls `dx=0.02` |
|------------|--------------------|---------------------|----------------------|
| `max|rho - rho_Tait(y)|/rho_Tait` | `2.53e-04` | `3.82e-04` | `3.20e-04` |
| `max|p - p_Tait(y)| / p_scale` | `4.54e-02` | `7.44e-02` | `5.75e-02` |
| `max|p - rho0 g (H-y)| / p_scale` | `4.56e-02` | `7.49e-02` | `5.77e-02` |

Figures: `docs/lake_at_rest_periodic_fieldmap_ax.png` (where the horizontal
residual lives — the two wall bands with opposite signs versus a field at
rounding level), `docs/lake_at_rest_periodic_profiles.png` (reconstructed density
against the two analytic profiles; Newton history of both unknown maps),
`docs/lake_at_rest_periodic_rest.png` (`max|v|/c0` over 0.5 s for both tanks).

## 4. Caveats (what this does *not* say)

* **F1 — the row tie is the statement under test.** The periodic operator is
  translation invariant in `x`, so a translation-invariant mass field gives
  `a_x = 0` by pair antisymmetry; tying the columns enforces exactly that. The
  tie is *not* an assumption that hides the difficulty: the same code path with
  the wall tie cannot close the horizontal equation, and the analytic profile
  (untied, `y`-dependent, never reconstructed) already gives
  `max|a_x|/g = 4.28e-13` in the periodic tank versus
  `6.63e-02` in the wall band of the wall tank. The test is falsifiable at
  G6/G7: the uniform lattice free-falls at `1.000 g` and the analytic profile
  leaves `0.2499 g`, so G1/G2 are not measuring a dead operator.
* **F2 — this is not the wall-bounded tank.** A real tank has lateral walls, and
  there the anchor cannot be met: the side-wall truncation pushes every near-wall
  particle away from its own wall, the residual is odd under reflection about the
  centreline, and no mass field that is even under that reflection can cancel it
  (`docs/verification_lake_at_rest.md`, F1). What this document certifies is the
  anchor's *physical* statement — a still body of water stays still — in the
  configuration where the discretisation is able to represent it. Both verdicts
  are reported side by side; neither replaces the other.
* **F3 — the discrete equilibrium is not the continuum one.** `rho` agrees with
  the exact Tait profile to `2.5e-4` relative, but the pressure agrees only to
  `4.5 %` of `p_scale` (interior band). That is the finite-`dx` error of the
  equilibrium and it is reported, not hidden; it also does not improve with the
  coarse-to-fine pair measured here (`4.54e-02` at `dx=0.02` vs `7.44e-02` at
  `dx=0.025`), i.e. two resolutions are not a convergence study and this
  document makes no order-of-accuracy claim for `p`.
* **F4 — scope of the capability.** `wrap_width` is a *pair-separation*
  condition, not a general periodic solver: the caller owns the pair list
  (`periodic_pairs` here), the wall must not duplicate columns that the seam
  already closes, and `apply_boundary`/the impact driver are untouched, so
  periodicity is not available in `tensorsph.run` yet.

## 5. Error judgement (per key quantity, explicit < 3 % answer)

| key quantity | reference | error | < 3 %? |
|--------------|-----------|-------|--------|
| velocity `max|v|/c0` | analytic identity `u = 0` | `2.57e-14` (absolute in `c0` units) | **yes** (`2.6e-12 %`) |
| free-surface elevation `max|dy|/H` | analytic identity `eta = const` | `9.90e-14` (absolute in `H` units) | **yes** (`9.9e-12 %`) |
| density invariance `max|drho|/rho0` | analytic identity | `3.56e-13` | **yes** |
| equilibrium residual `max|a|/g` | analytic identity `a = 0` | `1.4e-12` | **yes** |
| reconstructed `rho` vs exact Tait profile | `eos_hydrostatic_density` | `2.53e-04` relative | **yes** (`0.025 %`) |
| reconstructed `p` vs exact Tait profile | `eos_hydrostatic_pressure` | `4.54e-02` of `p_scale` | **no** (`4.5 %`) — discretisation error of the discrete equilibrium, F3 |
| reconstructed `p` vs `rho0 g (H-y)` | incompressible hydrostatics | `4.56e-02` of `p_scale` | **no** (`4.6 %`) — same cause, plus the `O(Ma^2) = 0.28 %` EOS-vs-incompressible gap |

The anchor's own acceptance quantities are all met at machine precision; the two
entries that exceed 3 % are auxiliary profile comparisons against continuum
references, not the anchor's identity, and they are reported as measured.