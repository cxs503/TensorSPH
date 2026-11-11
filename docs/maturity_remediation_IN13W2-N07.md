# Maturity remediation — IN13W2-N07 (ice thermal-saline-mechanical basis)

*Repair round 1* after the `maturity_gate` verdict on mission `m10431`:

> only L0 (not L2 independent capability): L3 missing `impl_files`, `entry`
> (artefact missing), `run_artifacts`, `integration_callers`; L2 missing
> `impl_files`, `entry` (artefact missing), `run_artifacts`.

The physics of `m10431` was already in the repository
(`src/tensorsph/ice_thermal/*`, 1 900 lines, 25 passing anchor tests). What was
missing was the *evidence shape*: the numbers lived in a gitignored `artifacts/`
directory, nothing outside the module called it, and no machine could re-derive
the claims from the repository.  This round does not restate the physics; it
turns the delivery into artefacts a gate can check.

## 1. Why the gate saw "artefact missing"

| what the gate looked for | what was in the repository before | why it read as missing |
|---|---|---|
| `impl_files` | `src/tensorsph/ice_thermal/*.py` (committed) | the module was committed but **not exported** from `tensorsph` and had no caller, so it was not addressable as a capability of the package |
| `entry` | `benchmarks/ice_thermal_multiphysics/run.py` (committed) | the only "entry" was a benchmark script defaulting to `artifacts/ice_thermal_multiphysics`, i.e. to a directory that **does not exist in a checkout** (`.gitignore: artifacts/`); there was no `python -m` entry, and the report JSON's `figures[*].path` pointed into that missing directory |
| `run_artifacts` | `artifacts/ice_thermal_multiphysics/*.png`, `results.json` | **gitignored** — 5 figures and the numbers existed only on the machine that ran them; a fresh clone had none of them |
| `integration_callers` | none | `wave_ice_interaction.py` used a hard-coded `SIGMA_F_DEFAULT = 0.7 MPa`; no solver path called `ice_thermal` |
| `benchmark` (L4) | `results.json` had `"anchors": {}` (empty placeholder, never filled) | the unit test suite compared numbers, but the committed run artefact carried no ref/got table, so the L4 numeric comparison had nothing to read |

Two root causes, both worth remembering:

1. **`artifacts/` is gitignored repo-wide.** Anything a mission writes there is
   invisible to every downstream gate and reviewer. Run outputs that a claim
   depends on must be written to a *tracked* path (`runs/<node>/` here) or next
   to the harness (`benchmarks/<name>/validation_report.json`, the convention of
   the rest of this repository).
2. **A capability with no caller is not a capability.** An export list plus a
   benchmark that imports a module is self-testing; the gate asks for a caller in
   the solver path.

## 2. What was added in this round

| level | requirement | artefact (all under tracked paths) | machine check |
|---|---|---|---|
| L2 | `impl_files` non-placeholder | `src/tensorsph/ice_thermal/{__init__,properties,enthalpy,sph_thermal,brine,darcy,thermal_stress,coupled,__main__}.py` | AST audit in `evidence.py`: no `NotImplementedError` / `TODO` / empty-`pass` body, ≥ 1 public function or `__all__` |
| L2 | `entry` | `python -m tensorsph.ice_thermal [--quick]` (`__main__.py`, `runpy` shim over the benchmark) and `benchmarks/ice_thermal_multiphysics/run.py` | both paths exist, sha256 recorded |
| L2 | `run_artifacts` | `runs/IN13W2-N07/{results.json,run.log,environment.json,fig1..fig5.png}` | existence + size + sha256, produced by the real run (`run_campaign.sh`) |
| L3 | `integration_callers` | `wave_ice_interaction.strength_from_ice_state` → `run_wave_ice_scenario` (fracture criterion, FSD evolution, K_MIZ); package export in `tensorsph/__init__.py` | grep of the symbol in the caller file; the evidence build fails if the call is removed |
| L4 | `benchmark` ref vs got | `benchmarks/ice_thermal_multiphysics/validation_report.json` (canonical, next to the harness) = byte-identical to `runs/IN13W2-N07/results.json` | 14 rows recomputed from the raw series in `evidence.py` and compared with the stored rows (< 1e-6), plus the byte-equality of the two copies |
| L4 | `repro_cmd` | `bash benchmarks/ice_thermal_multiphysics/run_campaign.sh runs/IN13W2-N07` | the script itself runs the benchmark, the evidence build and the environment snapshot |

## 3. The one physics change: strength from the thermal-saline state

`wave_ice_interaction.run_wave_ice_scenario` used a constant flexural strength
(0.7 MPa) in the fracture criterion `sigma_max / sigma_f`. It now accepts the
state a ship in ice can actually estimate —
`WaveIceScenario(temperature_c=..., salinity_psu=...)` — and derives

```
v_b  = brine_volume_fraction(T, S)                      # Cox & Weeks (1983)
sigma_f = 1.76 MPa (1 - sqrt(v_b / 0.27))               # Timco & Frederking (1990)
```

with the constants added to `ice_thermal.properties` (`SIGMA_F_MAX`,
`V_B_CLOSURE`, `flexural_strength`, `strength_band`). Consistency checks in
`tests/test_ice_thermal_integration.py`:

* inverting the collapse curve gives `v_b = 0.098` for 0.7 MPa, and a 10 psu
  column reaches that brine volume at ≈ −5.4 °C → the old constant is a
  *physical* first-year state, not a free parameter;
* −20 °C / 3 psu and −2 °C / 5 psu give different `sigma_f`, different fracture
  ratios and different floe statistics from the *same* wave;
* `v_b > 0.27` (ice above pore closure, e.g. −1.5 °C / 15 psu) returns
  `sigma_f = 0`, and the criterion now returns an infinite ratio instead of
  raising `ZeroDivisionError` — a real robustness defect found while wiring the
  coupling;
* the default scenario (no state given) is bit-for-bit unchanged.

## 4. Ledger registration

`benchmarks/verification_ledger/build_ledger.py::ice_thermal_claims` registers
three readings from the canonical report, each against an *independent*
reference:

| claim | reference | rel. error | grade |
|---|---|---|---|
| `ice_thermal.stefan_front.finest` | Neumann similarity solution (λ from `√π λ e^{λ²} erf λ = Ste`) | 0.197 % | PASS |
| `ice_thermal.darcy_benard.ra_c` | exact `4π²` | 0.0157 % | PASS |
| `ice_thermal.brine_floor.min_margin` | one-sided bound `S ≥ S_eq(T)`, `S_eq` from `v_b = v_irr` (input, not a fit) | min margin +0.614 psu | PASS |

This removes the `uncovered_benchmark` finding for
`benchmarks/ice_thermal_multiphysics` (findings 32 → 31, claims 44 → 47) and is
the reason the numbers now have to keep passing on every ledger regeneration.

## 5. Checks run

```
python benchmarks/ice_thermal_multiphysics/run.py --out runs/IN13W2-N07   # 401 s
python benchmarks/ice_thermal_multiphysics/evidence.py --run-dir runs/IN13W2-N07
python benchmarks/verification_ledger/build_ledger.py
python -m pytest tests/test_ice_thermal_multiphysics.py \
                 tests/test_ice_thermal_integration.py \
                 tests/test_wave_ice_interaction.py -q
```

Known pre-existing red in this repository (present before this round, unrelated):
`tests/test_verification_ledger.py` (`no_evidence_artifact_is_unregistered`,
`every_waiver_has_a_reason`, `stale_numbers_in_prose_are_marked`,
`no_benchmark_harness_is_uncovered` for other benchmarks) and
`tests/test_doc_pointers.py::test_every_docs_pointer_resolves_or_is_waived`
(9 failures on the parent commit → 5 with these changes).

## 6. What is still not covered (honest)

* **Greeley & Wilson (1987) brine-drainage data** remains `WITHHELD`: the table
  is not reachable offline, so the desalination anchor is an observation *band*
  (Cox & Weeks 1983, 3–8 psu) plus the closed-form floor, not a point-by-point
  comparison.
* **The flexural/compressive split**: the Timco & Frederking collapse is
  measured for strength vs brine volume; using it for the *flexural* strength
  inherits a ~1.5× level uncertainty (`strength_band` returns the band rather
  than a single number). No independent flexural dataset was fitted here.
* **`k_ref`** (brine permeability scale) is uncertain by an order of magnitude;
  the run scans ×0.1/×1/×10 instead of hiding the choice.
* The three-field column does not ablate its base, and the ocean is a single
  well-mixed layer — documented in `docs/verification_ice_thermal_multiphysics.md`
  §8.