# m4480 — PR #242 CI red: viscosity shear-wave reproducibility gate

## Symptom

PR #242 (branch `agent/m4135`, benchmark `benchmarks/viscosity_shear_wave`)
reported `pytest (3.11) = failure`.  The benchmark itself and six of the seven
new tests were fine; the red came from the last guard,
`tests/test_viscosity_shear_wave.py::test_coarse_case_is_reproducible`.

## Root cause

That test re-ran the cheapest ladder rung and required the measured viscosity to
match the value committed in `benchmarks/viscosity_shear_wave/report.json` to
`rel=1e-9`.  `nu_effective` is a float32 reduction over particle pairs, so its
trailing digits follow the summation order, which torch derives from the thread
count.  `report.json` was produced on a 128-thread host; a GitHub runner has a
handful of cores.  Reproduced locally by changing nothing but the thread count:

    OMP_NUM_THREADS=1 python -m pytest tests/test_viscosity_shear_wave.py -q

    assert redo["nu_effective"] == pytest.approx(case["nu_effective"], rel=1e-9)
    E  assert 0.00817607052026891 == 0.008176345173174338 +/- 8.2e-12

Relative disagreement: **3.36e-05** — five orders of magnitude outside the
`1e-9` band, although the physics is unchanged (the 15 % acceptance gate is
never in question, the ratio stays at 0.9966).  So the gate was a property of
the host hardware, not of the solver: it was guaranteed to be red on CI.

## Fix (test only, 1 file)

`tests/test_viscosity_shear_wave.py` now separates the two claims that had been
collapsed into one assertion:

1. **Same process, same threads → bit-identical.**  The case is run twice and
   the two `nu_effective` values must be *exactly* equal.  The solver has no
   stochastic step, so this catches a changed initial condition or lattice
   convention on any machine, and it is what CI can enforce.
2. **Against the committed report, which used another thread count →
   `CROSS_MACHINE_TOLERANCE = 1e-3`.**  That band is ~30x the measured hardware
   drift (3.36e-05) and still ~150x tighter than the 15 % acceptance criterion;
   `steps` and `n_fluid` remain exact integer checks.  The derivation and the
   measured drift are recorded in a code comment next to the constant, so the
   tolerance cannot be mistaken for a fudge factor.

No production code, benchmark harness or recorded number was touched
(`report.json`, `docs/verification_viscosity_shear_wave.md`,
`benchmarks/viscosity_shear_wave/run.py` are unchanged).

## Verification (run locally, Python 3.11.6, torch 2.7.1)

| command | before fix | after fix |
|---|---|---|
| `OMP_NUM_THREADS=1 pytest tests/test_viscosity_shear_wave.py -q` | 1 failed, 6 passed | **7 passed** |
| `OMP_NUM_THREADS=4 pytest tests/test_viscosity_shear_wave.py -q` | 7 passed | **7 passed** |
| `pytest tests/test_viscosity_shear_wave.py -q` (defaults, 128 threads) | 7 passed | **7 passed** |
| `python -m ruff check tests/test_viscosity_shear_wave.py` | pass | **pass** |

The three thread settings cover the hardware spread the gate has to survive;
before the fix the single-thread setting failed, which is the CI configuration.

## Limitations / notes

* The committed `report.json` still carries the 128-thread numbers (correctly:
  it records the run that was actually performed); the tolerance, not the
  artifact, now absorbs the hardware spread.  Regenerating on a low-core host
  would move the last digits and nothing else.
* The rest of the suite is simulation-heavy and this host was under heavy load
  (load average >130 on 128 cores), so the full-suite run was not completed
  within the mission; the changed file is self-contained and was verified in
  isolation at three thread counts, including a reproduction of the red state.
* Pre-existing `ruff` findings elsewhere in the tree (E501/W292 in unrelated
  benchmark scripts) were left untouched: this PR's files lint clean and the CI
  signal that failed was `pytest (3.11)`.

## Second finding: the verification ledger is red on main (out of scope here)

A local run of the suite also fails 6 tests in `tests/test_verification_ledger.py`.
That is **not** the defect this PR was red for, and it is not fixed here, but it
is recorded because it gates the merge:

    $ python benchmarks/verification_ledger/build_ledger.py --check
    DRIFT: ledger.json is stale (run build_ledger.py)
    DRIFT: docs/verification_ledger.md is stale (run build_ledger.py)

Regenerating the ledger adds 6 `unregistered_artifact` findings and 1
`unmarked_superseded_number` finding.  Five of the seven belong to artifacts
already merged into `main` before this branch was rebased
(`benchmarks/conservation/report.json`,
`benchmarks/contact_angle_geometry/results.json`,
`benchmarks/dam_break/reference_ladder.json`,
`benchmarks/init_consistency/report.json`,
`benchmarks/initial_density_bias/report.json`,
`benchmarks/laplace_pressure_closure/raw/relax_ladder.json`,
`benchmarks/laplace_pressure_jump/validation_report.json`) plus
`docs/verification_surface_tension_consistent.md:158`; only one is attributable
to this PR, namely that
`benchmarks/viscosity_shear_wave/report.json` is not yet a registered claim in
`benchmarks/verification_ledger/build_ledger.py`.

Registering that claim (and regenerating `ledger.json` +
`docs/verification_ledger.md`) would fold all of `main`'s accumulated ledger
drift into this PR, i.e. it would rewrite records that have nothing to do with
the viscosity benchmark.  That is why it is *not* done here: the regeneration
was produced, inspected, and reverted, and the change stayed scoped to the test
that made CI red.  The registration of the new artifact should be its own
mission, together with the `main` ledger refresh.
