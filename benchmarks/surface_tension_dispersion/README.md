# Surface-tension dispersion benchmark (superseded location)

The measurement now lives in the library and its run artefacts:

| what | where |
|---|---|
| implementation + audits | `src/tensorsph/surface_tension_dispersion.py` |
| entry point | `scripts/run_surface_tension_dispersion.py`, `python -m tensorsph.surface_tension_dispersion` |
| reference-vs-measured table | `runs/surface_tension_dispersion/benchmark.json` |
| full report / provenance | `runs/surface_tension_dispersion/{report,meta}.json`, `summary.md`, `figures/` |
| write-up | `docs/verification_surface_tension_dispersion.md` |
| tests | `tests/test_surface_tension_dispersion.py`, `tests/test_csf_dispersion.py` |

`dispersion.py` in this directory is a thin forwarding shim so old command lines keep
working.  The previous `validation_report.json` (m13895) was produced by a harness with
four structural defects (closed-block geometry with four free surfaces, a scale-invariant
`k` sweep, a flipped traction sign and no lattice-phase averaging) and has been removed
so its numbers cannot be quoted; the A/B control of the old geometry is still reproducible
via `--defective-control`.
