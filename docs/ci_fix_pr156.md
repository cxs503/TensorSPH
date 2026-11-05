# CI fix for the viscosity shear-wave benchmark (PR #156)

CI job `pytest (3.10/3.11/3.12)` failed on the *Lint* step for head `3420d0a`:
`python -m ruff check .` reported five findings in the three Python files this
branch adds, so every later step of the job (unit tests, small end-to-end case,
benchmark gates) was skipped.  The fix is confined to those three files:

| File | Finding | Change |
|---|---|---|
| `benchmarks/viscosity_shear_wave/make_doc.py` | F841 unused `checks` | drop the unused dict |
| `benchmarks/viscosity_shear_wave/make_doc.py` | E501 108 > 100 | bind the negative-control fraction once (`control_pct`) and quote it twice |
| `benchmarks/viscosity_shear_wave/make_doc.py` | W292 | trailing newline |
| `benchmarks/viscosity_shear_wave/run.py` | W292 | trailing newline |
| `tests/test_viscosity_shear_wave.py` | W292 | trailing newline |

No behaviour changes: the rewritten `make_doc.py` regenerates
`docs/verification_viscosity_shear_wave.md` byte for byte (md5
`16cb4dd7653f0b80644bdc2840467e96` before and after), and
`tests/test_viscosity_shear_wave.py` still reports 7 passed.

Verification: `python -m ruff check .` → *All checks passed!* (was 5 findings);
`python -m pytest tests/test_viscosity_shear_wave.py -q` → 7 passed.
