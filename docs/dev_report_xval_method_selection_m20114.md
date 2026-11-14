# m20114 — closing the last WITHHELD hole in the SPH method-selection evidence gate

**Anchor**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` (feature ledger, status `in_progress`)
**Mission**: 20114 · **Domain**: TensorSPH · **Date**: 2026-11-14
**Increment class**: verification debt closure (evidence leg), not new physics.

---

## ① Case introduction

`tensorsph.method_selection` routes a case profile to an SPH scheme, but only
after checking the repository's own verification records: a scheme with no
machine-readable benchmark record comes back **WITHHELD**, a scheme whose records
are all outside the owner 15 % standard comes back **FAILED**. That gate is the
whole point of the anchor — it is what stops the selector from nominating a
method nobody has ever scored.

When this mission started the registry
(`docs/reports/method_selection_registry.json`) still listed **one** scheme as
un-evidenced:

```
methods_without_evidence: ["transport_velocity_correction"]
```

so the `tensile_stretching` modifier rule fired but had to answer WITHHELD —
i.e. TensorSPH ships a transport-velocity correction
(`src/tensorsph/transport_velocity_correction.py`, ported from SPHinXsys
`TransportVelocityCorrection` and validated in
`docs/xval_tvc_validation_report.md`, mission m18253) that the selector was not
allowed to recommend. The physics evidence existed as a **prose report**; the
selector only reads artefacts. That gap is the increment.

Physical setting of the evidence case: Martin & Moyce (1952) dam break against a
dry bed — column width `a = 0.2 m`, height `0.4 m` (aspect ratio 2), tank
`2.0 × 0.75 m`, `g = 9.81 m/s²`, `ρ0 = 1000 kg/m³`, `c0 = 20 m/s`, `h/dx = 1.3`,
front trajectory non-dimensionalised as `τ = t√(2g/a)`, `Z/a`. Reference: the
in-repo digitisation `benchmarks/dam_break/martin_moyce_1952.json`
(Koshizuka & Oka 1996 data, digitisation uncertainty ±0.05 in `Z/a`).

## ② What was done

1. **Two extractors added** to the registry builder
   (`benchmarks/method_selection/build_registry.py`):
   * `x_transport_velocity` → `docs/xval_sphinx_tvc/data/grid_dx003.json`,
     arm pinned **by name** (`tvc_a010_all`, the α = 0.10 coefficient calibrated
     in the m18253 report), read at `dx = 0.03 m` (finest committed rung);
   * `x_transport_velocity_coarse` → the same arm at `dx = 0.04 m`, kept in the
     registry **on purpose** although it fails the 15 % standard, so resolution
     sensitivity is visible in the decision's caveats instead of being dropped.
2. **Registry rebuilt** from committed artefacts (sha256 per source, no hand
   typed numbers): `methods_without_evidence` is now empty, 10 schemes /
   12 records.
3. **Figure + table generator added**
   (`benchmarks/method_selection/plot_method_selection_m20114.py`): recomputes
   the error from the raw `tau`/`Z_over_a` curves against the experimental
   points (ignoring the artefact's stored `score` block), asserts equality with
   the banked registry number, and emits the tables below.
4. **Tests**: the old assertion
   `test_tensile_stretching_modifier_is_withheld_without_record` was inverted into
   `test_tensile_stretching_modifier_is_backed_by_the_tvc_record`; two new tests
   were added —
   `test_transport_velocity_record_is_reproducible_from_raw_curves` (recompute
   from raw curves, `rel=1e-9`) and
   `test_every_catalogued_scheme_has_a_verification_record`
   (evidence-completeness gate: `METHODS ∪ MODIFIERS ⊆ methods_with_evidence`,
   `missing_sources == []`, every record sha256-pinned).

## ③ Results

Machine-generated (zero hand-copied numbers):
`docs/reports/method_selection_m20114_tables.md`, JSON summary in
`docs/reports/method_selection_m20114.json`.

# m20114 — transport-velocity evidence, machine-generated tables

All numbers below are recomputed from the committed TVC dam-break grid
against the in-repo Martin & Moyce digitisation
(the stored `score`
block of the artefact is ignored).

The experimental point at Z/a = 0 is used by the absolute error but
excluded from the relative error (division by zero), so the relative
error is averaged over 19 of the 20 gated points.

## Recompute vs banked registry record (dx = 0.03 m, tvc_a010_all)

| quantity | recomputed | banked in registry |
|---|---|---|
| mean relative error of Z/a | 10.8104 % | 10.8104 % |

## dx = 30 mm

| variant | points | MAE Z/a | mean rel. err | max abs err |
|---|---|---|---|---|
| baseline | 19 | 0.6449 | 50.98 % | 1.1074 |
| tvc_a010_all | 19 | 0.1408 | 10.81 % | 0.3067 |

## dx = 40 mm

| variant | points | MAE Z/a | mean rel. err | max abs err |
|---|---|---|---|---|
| baseline | 19 | 0.6168 | 46.74 % | 1.1508 |
| tvc_a010_all | 19 | 0.1492 | 16.96 % | 0.3228 |

## Registry coverage

- methods with evidence: 10
- methods without evidence: none
- catalogue size (METHODS + MODIFIERS): 10
- records: 12


Recomputed vs banked, `dx = 0.03 m`, `tvc_a010_all`: **10.8104 %** recomputed
from the raw curves == **10.8104 %** banked in the registry (agreement to the
printed precision; the earlier m18253 report printed 10.8 %).

Routing after the increment — profile "breaking wave over a slope":

```
primary : wcsph_delta_sph                  SELECTED (worst 2.97 %)
modifier: density_diffusion                SELECTED (1.02 %)
modifier: transport_velocity_correction    SELECTED (10.81 %)  ← was WITHHELD
modifier: particle_shifting                SELECTED (8.83 %)
modifier: compression_regularization       SELECTED (5.53 %)
```

## ④ Figures

* `docs/figures/method_selection_m20114/figure_tvc_front.png` — front
  trajectory `Z/a` vs `τ` for the experiment (points), baseline (dashed) and
  TVC α = 0.10 (solid), at `dx = 30 mm` and `40 mm`.
* `docs/figures/method_selection_m20114/figure_registry_coverage.png` — one bar
  per evidenced scheme (best record) with the worst banked record marked and the
  15 % owner standard drawn; the log abscissa shows how far the two failing
  records (consistent-CSF curvature 23.8 %, TVC coarse 16.96 %) sit past the line.

## ⑤ Notes / limitations

1. **The record inherits the m18253 evidence, it does not replace it.** The
   operator leg of that validation is an exact comparison against a literal
   transcription of upstream SPHinXsys C++ (the OSS library itself cannot be
   built here — no Boost/TBB), see
   `tests/test_transport_velocity_correction.py`.
2. **Resolution sensitivity is real**: the same arm is 10.81 % at `dx = 30 mm`
   and 16.96 % at `dx = 40 mm`. The selector therefore claims the *passing*
   bound (10.81 %) and surfaces the coarse record as a caveat; anyone quoting
   "TVC is within 15 %" must also quote `dx ≤ 0.03 m`.
3. **Single case family**: the banked evidence is the Martin & Moyce dam break
   only. Sloshing / wave-flume / 3D dam break are not banked, so the selector
   carries a `caveats` string for that; the registry format supports adding
   records per case without touching the selector.
4. **The α = 0.10 coefficient is solver-calibrated** (upstream default 0.2
   over-disperses here); the record is for that coefficient only, which is why
   the extractor pins the arm by name rather than taking "the best row".
5. **Reference uncertainty**: ±0.05 in `Z/a` digitisation is comparable to the
   MAE (0.141 at `dx = 30 mm`), so the dam-break gate is a smoke-level accuracy
   statement, not a 1 % claim.

## ⑥ Error assessment (vs the 15 % owner standard)

| Key quantity | Value | Reference | Error | < 15 %? |
|---|---|---|---|---|
| mean relative error of `Z/a`, TVC α=0.10, `dx = 30 mm` | 10.81 % | Martin & Moyce digitised points | 10.81 % | **yes** |
| same, `dx = 40 mm` (banked as caveat) | 16.96 % | same | 16.96 % | no (resolution-limited) |
| baseline (no correction) `dx = 30 mm` | 50.98 % | same | 50.98 % | no — this is the failure the scheme fixes |
| MAE of `Z/a`, TVC α=0.10, `dx = 30 mm` | 0.1408 | experimental gate 0.15 | inside gate | yes |
| recomputed vs banked registry value | 10.8104 % vs 10.8104 % | — | < 1e-9 relative | yes (self-consistency) |

Owner standard < 15 %: **satisfied at dx = 30 mm (10.81 %), not at dx = 40 mm
(16.96 %)** — stated explicitly rather than hidden by the passing rung.

## Reproduction

```bash
PYTHONPATH=src python benchmarks/method_selection/build_registry.py      # rebuild registry/docs
PYTHONPATH=src python benchmarks/method_selection/plot_method_selection_m20114.py
PYTHONPATH=src python -m pytest tests/test_method_selection.py -q        # 22 passed
```