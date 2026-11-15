# The evidence channel of `CASE-SPH-LAKE-AT-REST`: making the verdict decidable where it is read

Case-pool anchor `CASE-SPH-LAKE-AT-REST` — *a still lake stays still*: reference is
the analytic hydrostatic identity (`u = 0`, `eta = const`), tolerance is machine
precision (`1e-9`). No experimental band, no fitted constant anywhere.

`docs/verification_lake_at_rest.md` (honest FAIL, wall-bounded tank) and
`docs/verification_lake_at_rest_periodic.md` (PASS, x-periodic tank) established
the physics. What was missing was the **evidence channel**: the JSON records lived
in `artifacts/lake_at_rest/` (`.gitignore`'d — absent on a fresh checkout) and the
numbers lived in prose (not machine-decidable, free to drift from the code). So on
a fresh checkout nothing failed when the shipped numbers stopped matching the
library, i.e. the anchor's verdict was not decidable where it is read. This change
closes that hole, following the pattern already used by `CASE-HYDROSTATIC`.

## 1. What ships now

| path | role |
|---|---|
| `tests/data/lake_at_rest/periodic_dx020.json` | full record of the passing configuration (`dx = 0.02`, 300 particles, 3000 pairs) |
| `tests/data/lake_at_rest/wall_dx020.json` | full record of the honest FAIL (`dx = 0.02`, 300 particles, 3165 pairs) |
| `tests/data/lake_at_rest/report.json` | gate table re-graded from those records, anchor tolerance, verdicts, reproducibility map |
| `tests/data/lake_at_rest/MANIFEST.sha256` | `sha256  bytes  relpath`, sorted — tamper-evident |
| `tests/data/lake_at_rest/channel.json` | what is shipped, what is **not** (PNG figures are artifact-class; `artifacts/` is not pushed), and why |
| `benchmarks/lake_at_rest/freeze.py` | freezes the channel; `--verify` re-runs both configs and compares against it |
| `tests/test_lake_at_rest_evidence.py` | 10 tests that make the shipped bytes load-bearing (~4 s) |

The gate tables and the grader moved into the library
(`tensorsph.lake_at_rest.GATES_WALL` / `GATES_PERIODIC` / `evaluate_gates`); the
runner now imports them (`evaluate` is kept as an alias). Runner, freeze tool and
tests therefore grade the *same* thresholds — a test that re-graded a shipped
record with its own copy could drift from the runner and still pass.

## 2. What the tests assert (not merely execute)

1. **Tamper-evidence**: every shipped byte is checked against `MANIFEST.sha256`;
   no unmanifested `*.json` may appear in the channel.
2. **The report cannot launder a verdict**: `report.json` rows are re-graded from
   the shipped run record with the library's `evaluate_gates`; verdicts,
   thresholds, comparisons and measured values must agree to `1e-12`.
3. **The anchor is met by the published evidence**: in `periodic_dx020` all five
   acceptance gates (`G1..G5`) pass and each measured value is `< 1e-9`; the run
   really moved the particles (`t_end = 0.5 s`, 1282 symmetric-Euler steps) and the
   reconstruction reported `converged = True`.
4. **The honest FAIL is preserved**: `wall_dx020` fails exactly
   `{G2, G3, G4, G5, G3i, G4i}`, `G1` still holds at machine precision, and the
   failure magnitudes stay in `(1e-9, 1)`. A future accidental "fix" of the
   side-wall residual is a red test, not silently stale prose.
5. **The operator is alive**: uniform-lattice control free-falls at `1.000 g` with
   `p_max < 1e-6`; the analytic profile leaves `0.2499 g`; the residual responds
   linearly to a mass perturbation (`r(1e-4)/r(1e-3) = 0.0998`).
6. **The evidence is re-derivable from source**: the shipped geometry (particles,
   wall particles, pairs, `h = 1.3 dx`) and `G7` (the analytic-profile residual —
   one `rhs` call) are recomputed at `dx = 0.02` on every checkout and compared
   with the frozen numbers at `rtol = 1e-8`. The full 300-particle reconstruction
   stays reproducible with `freeze.py --verify` (~40 s).

## 3. Verdict of the frozen record

| configuration | `max\|a\|/g` | `max\|a_x\|/g` | `max\|v\|/c0` | `max\|dy\|/H` | `max\|drho\|/rho0` | verdict |
|---|---|---|---|---|---|---|
| x-periodic `dx = 0.02` | `5.760e-13` | `1.388e-12` | `2.213e-14` | `4.903e-14` | `1.112e-13` | **PASS** (all gates) |
| wall-bounded `dx = 0.02` | `3.578e-13` | `1.242e-01` (wall band) | `1.644e-02` | `8.360e-02` | `2.353e-01` | **FAIL** (honest, documented) |

The difference between the rows is the side-wall truncation, not the operator:
the same code path, the same lattice rows and the same solver are used, and the
x-periodic operator is translation invariant in `x` (so the horizontal equation
closes by pair antisymmetry).

## 4. Reproducibility: the anchor is a *scale* statement, and that is now written down

Five identical runs of `periodic_dx020` (same code, same inputs, `OMP_NUM_THREADS=1`,
14 Newton iterations each) give

| quantity | run 1 | run 2 | run 3 | run 4 | run 5 | spread |
|---|---|---|---|---|---|---|
| `max\|a\|/g` | `7.45e-13` | `7.09e-13` | `7.30e-13` | `5.74e-13` | `6.68e-13` | 1.30x |
| `max\|v\|/c0` | `3.51e-14` | `2.59e-14` | `1.97e-14` | `4.34e-14` | `2.48e-14` | 2.20x |
| `rho` vs exact Tait | `2.526271e-04` | `2.526271e-04` | `2.526271e-04` | `2.526271e-04` | `2.526271e-04` | bit-identical |

The rounding-level entries are cancellations of double-precision pair sums and are
**not** bit reproducible; the physics of the equilibrium is. The channel therefore
declares two classes (`report.json → reproducibility`): `stable` (one solve / one
sum / an integer — checked at `rtol = 1e-8`) and `rounding_level` (checked by
*scale*: both sides below the `1e-9` anchor tolerance, ratio inside `[0.1, 10]`).
Above the tolerance a quantity is a truncation force, not a cancellation, so it is
checked as a number (`rtol = 1e-2`) — and a quantity may not cross the tolerance
sideways in either direction. A verification that demanded bit equality here would
be flaky; one that ignored the digits would ship a number that no longer exists.

## 5. Adversarial check of the guard (performed)

Two tampering attempts on a copy of the channel — (a) raise the periodic
`max|v|/c0` to `1e-3` so the lake "sloshed", (b) set the wall record's
`all_passed = true` to launder the honest FAIL — are both rejected:

```
VERIFY FAILED (5):
  periodic_dx020:rest.max_v_over_c0 crossed the anchor tolerance (1.000e-03 -> 2.912e-14)
  periodic_dx020: report gates disagree with the record they claim to summarise
  wall_dx020: report gates disagree with the record they claim to summarise
  periodic_dx020.json: sha256 mismatch against MANIFEST.sha256
  report.json: sha256 mismatch against MANIFEST.sha256
```

The pytest layer rejects the same edits by (1) the manifest check and (2) the
re-grade check, so neither the record nor the summary can be edited alone.

## 6. Error judgement (`< 3 %?`, per key quantity, explicit)

| key quantity | reference | measured | `< 3 %`? |
|---|---|---|---|
| `max\|v\|/c0` (periodic, shipped) | identity `u = 0` | `2.213e-14` | **yes** (`2.2e-12 %`) |
| `max\|dy\|/H` | identity `eta = const` | `4.903e-14` | **yes** |
| `max\|drho\|/rho0` | identity | `1.112e-13` | **yes** |
| equilibrium `max\|a\|/g` | identity `a = 0` | `5.76e-13` | **yes** |
| reconstructed `rho` vs exact Tait profile | `eos_hydrostatic_density` | `2.53e-04` | **yes** (`0.025 %`) |
| reconstructed `p` vs exact Tait profile | `eos_hydrostatic_pressure` | `4.536e-02` of `p_scale` | **no** (`4.5 %`) — discretisation error of the discrete equilibrium, reported as measured |
| reconstructed `p` vs `rho0 g (H-y)` | incompressible hydrostatics | `4.56e-02` of `p_scale` | **no** (`4.6 %`) — same cause plus the `O(Ma^2) = 0.28 %` EOS gap |

The anchor's own acceptance quantities are all at machine precision; the two
entries above 3 % are auxiliary continuum-profile comparisons, not the anchor's
identity, and they are pinned by a test that asserts they stay **above** 3 %
(`test_error_judgement_is_pinned_to_the_shipped_numbers`) so the honest number
cannot be quietly rounded into a pass.

## 7. Caveats

* **The wall-bounded tank still FAILS.** This change does not touch the physics
  or the solver; that FAIL is the measurement of side-wall truncation and remains
  the honest negative result of the anchor.
* **The periodic tank is not a general periodic solver.** `wrap_width` is a
  pair-separation condition; the caller owns the pair list (F4 of
  `verification_lake_at_rest_periodic.md`).
* **`p` is not converged** (two resolutions, opposite direction), so no
  order-of-accuracy claim is made for the pressure profile.
* **The channel is at `dx = 0.02` only.** The second resolution (`dx = 0.025`)
  is quoted in the periodic document as supporting evidence but is not frozen;
  `freeze.py --dx`-style extension would be a separate addition.
* Figures are **not** shipped in the channel: they are artifact-class in the
  delivery rule and re-creatable from the shipped numbers with
  `benchmarks/lake_at_rest/make_figures.py`; the channel records that decision
  (`channel.json → not_shipped`).

## 8. Reproduce

```
PYTHONPATH=src python3 benchmarks/lake_at_rest/freeze.py            # re-freeze
PYTHONPATH=src python3 benchmarks/lake_at_rest/freeze.py --verify   # fresh run vs frozen, ~40 s
python3 -m pytest tests/test_lake_at_rest_evidence.py -q            # ~4 s, fresh-checkout decidability
```