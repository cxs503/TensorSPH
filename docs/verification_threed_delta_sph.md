# 3-D WCSPH profile error: is the missing diffusive density term the cause?

**Anchor:** `FEAT-OSS-INVENTORY-SPH` (open gap `GAP-SPH-SPS-LES`-sibling work on the 3-D
right-hand side; ledger entry under test:
`threed_hydrostatic.profile.max_rel_error_core.adami_best`, grade **FAIL**).
**Code:** `src/tensorsph/threed/operators3.py::density_diffusion_rate_3d`,
`accelerations(..., density_diffusion=, dt=, density_diffusion_form=)`,
`src/tensorsph/threed/solver.py::run_hydrostatic(..., density_diffusion=)`.
**Artifacts:** `benchmarks/threed_hydrostatic/delta_sph/summary_nz8.json` (+ the two per-run
JSONs it was built from), `benchmarks/threed_hydrostatic/delta_sph_audit.py`.
**Tests:** `tests/test_threed_density_diffusion.py` (6 tests, all pass).

## 1. Case

A 3-D water column of height `H = nz*dx` resting on a static wall lattice, driven by
gravity `g = 9.81 m/s^2`, with a free surface. The exact answer is `p(z) = rho0 g (H - z)`
with `rho(z) = rho0 (1 + gamma g (H - z) / c0^2)^(1/gamma)`, so every number below is
compared with a closed form, not with another code. Parameters held fixed in the audit
(`protocol.fixed` of the summary):

| parameter | value |
|---|---|
| `dx` / `h` | 0.01 m / 1.3 dx |
| lattice `nx x ny x nz` | 5 x 5 x 8 (200 fluid + 772 wall particles) |
| `c0` | 20 m/s (`dt = 0.25 h / c0 = 1.625e-4 s`, 923 steps) |
| `nu` / `rho0` | 0.02 m^2/s / 1000 kg/m^3 |
| wall model | Adami et al. (2012) pressure extrapolation, 2 wall layers |
| `t_end` / averaging window | 0.15 s / last 50 % |
| density diffusion `delta` | 0.0 (control) and 0.1 |

## 2. What was done

The shipped 3-D artifact `adami-c0_20-nz16` carries a layer-to-layer (odd-even) pressure
mode: rows 9..12 read -11.1 %, +1.1 %, -17.4 %, +4.0 % against the analytic profile. A
checkerboard mode in `rho` is invisible to the symmetric SPH pressure gradient on a
regular lattice (for `p_j = -p_i` along `z`, the pair contributions cancel exactly), so
nothing in the 3-D right-hand side damps it. The 2-D solver does damp it, with the
diffusive density term of `tensorsph.density_diffusion`. The 3-D operator set had no such
term, so this change adds it:

1. `density_diffusion_rate_3d` — pairs the existing 2-D machinery
   (`rate_from_positions`, Molteni-Colagrossi or the hydrostatic-corrected Antuono /
   Fourtakas `psi` form) with the 3-D pair geometry, fluid/fluid pairs only, scale
   `2 delta c0 h`, and returns a *rate* (the 3-D path is a summation-density solver, it
   never integrates the continuity equation).
2. `accelerations(...)` applies `rho = rho + dt * rate` when `density_diffusion > 0`; with
   the default `0.0` the right-hand side is bit-for-bit the previous one (pinned by
   `test_accelerations_default_is_unchanged`).
3. `run_hydrostatic(...)` exposes the switch and records it on the result.

Then a pre-declared controlled pair (`delta = 0` vs `delta = 0.1`, every other parameter
identical, same step count, same averaging window) was run, with the credit rule written
into the script before the runs: *credit the term only if the binning-invariant
(worst core-layer, lattice-row) error falls by >= 5 percentage points at some delta while
the bottom-pressure error stays inside 15 %.*

## 3. Results

Programmatically copied from `benchmarks/threed_hydrostatic/delta_sph/summary_nz8.json`
(no hand-typed numbers):

| meter | `delta = 0` (control) | `delta = 0.1` | change |
|---|---|---|---|
| bottom pressure rel. error | -0.043690 | -0.043451 | +0.00024 |
| worst core layer, Eulerian bins | 0.106303 | 0.106110 | -0.00019 |
| worst core layer, lattice rows | 0.135095 | 0.134091 | **-0.00100** |
| mean core layer, lattice rows | -0.034194 | -0.034111 | +0.00008 |
| odd-even amplitude / `rho0 g H` | 0.221753 | 0.221037 | -0.00072 |
| depth-integrated load rel. error | -0.108664 | -0.108751 | -0.00009 |
| worst `|rho/rho0 - 1|` | 1.750e-3 | 1.750e-3 | ~0 |
| layers within 15 % (of 5 core rows) | 5 | 5 | 0 |
| wall clock | 39.5 s | 50.1 s | +27 % |

**Verdict: `NOT_CREDITED`** (the script's own rule, `summary_nz8.json:verdict`). The
diffusive term changes the worst core-layer error by 0.10 percentage points — 0.7 % of
the error itself — and leaves the odd-even amplitude, the integrated load and the density
excursion untouched, while costing 27 % wall clock. The hypothesis "the 3-D hydrostatic
profile failure is an undamped diffusive density mode that `delta = 0.1` removes" is
therefore **falsified for this configuration**.

No figure is produced: this is an operator-level A/B on two runs of a 200-particle column
with one spatial profile; a plot of two nearly identical curves would add no information
that the table does not already carry.

## 4. Notes, limits, and what the negative result means

1. **The odd-even mode is real but it is not diffusive at this amplitude.** The lattice-row
   density excess at `nz = 8` is `1.75e-3` (compressibility-limited, `c0 = 20` is below the
   `10 sqrt(gH)` rule). The diffusive flux is proportional to the *contrast* in `rho`; at a
   contrast of order `1e-5 rho0` between rows, `2 delta c0 h * psi` is far below the
   acoustic imbalance that sets the pressure, so the term cannot move the profile before
   the compressibility floor dominates. That is a mechanism-level explanation consistent
   with the measured null result, not an independent proof.
2. **Scope of the claim.** One `c0` (20), one `delta` (0.1), one coarse lattice (5x5x8,
   only 5 core rows), `t_end = 0.15 s` (the shipped benchmark uses 0.6 s). The `nz = 12`,
   `delta in {0, 0.1, 0.2}`, `t_end = 0.3 s` ladder was started and **did not finish** in
   this slot's compute allowance (node `load15` was 126-185 throughout); it is
   `not_run`, not `passed`. Re-running
   `python benchmarks/threed_hydrostatic/delta_sph_audit.py --deltas 0,0.1,0.2 --nz 12 --nx 6 --ny 6 --t-end 0.3`
   is the direct continuation and should be done before anyone widens this conclusion.
3. **What this rules out, and what it does not.** It rules out "just switch on the 2-D
   diffusion term and the 3-D profile gate passes". It does not rule out a *continuity-form*
   delta-SPH (where the term is part of the integrated density rather than a correction to a
   summation density), nor particle shifting / remeshing, nor the more likely explanation
   visible in the shipped artifact: the profile error grows towards the free surface
   (row 12: -41 %), which is where the support is truncated, i.e. a free-surface closure
   question rather than a bulk diffusive one.
4. **Engineering note.** The default is off, so nothing in the shipped 3-D artifacts,
   engines or gates changes; the term is opt-in and the new code is exercised by six tests
   (exact-zero at `delta = 0`, volume-weighted rate conservation, wall rows untouched,
   hydrostatic-corrected form < 10 % of the Molteni form on a truncated linear profile,
   bit-identical default, `ValueError` when `dt` is missing).

## 5. Error judgement

| quantity | reference | measured (control) | error | < 3 %? |
|---|---|---|---|---|
| bottom pressure | 1177.2 Pa (`rho0 g H`, `H = 0.12 m`) | 1124.8 Pa | -4.37 % | **no** |
| worst core-layer pressure (lattice) | analytic `p(z)` per row | - | 13.51 % | **no** |
| odd-even amplitude / pressure scale | 0 (exact hydrostatic) | 0.2218 | - | **no** |
| worst density excursion `|rho/rho0-1|` | 0 (incompressible limit) | 1.75e-3 | - | **no** (WCSPH at `c0 = 20` is below the acoustic rule) |
| effect of the change on the worst core layer | control value | 0.135095 -> 0.134091 | -0.74 % of the error | the *change* is < 3 %, i.e. indistinguishable |

The 15 % gate on the profile is not met by either arm of the pair; nothing here is
presented as a pass. The honest reading is that the 3-D hydrostatic profile failure
remains open, with one candidate cause (bulk density diffusion) now measured and rejected
at this resolution.
