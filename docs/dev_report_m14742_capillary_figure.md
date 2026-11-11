# m14742 — the open-free-surface audit record's figure, and why `main` was red

*TensorSPH, development record. Mission m14742 (re-open of #458, round 2). Scope: the
capillary-wave open-free-surface evidence record only — no solver, ledger or waiver
change.*

## 1. What was found

The record built by m13812 / m14576 (`benchmarks/capillary_wave/open_surface_audit.py`
+ `.json`, `docs/verification_capillary_wave_open_surface.md`,
`tests/test_capillary_wave_open_surface.py`) is on `main`, and its own gate **fails at
HEAD**:

```
$ python -m pytest tests/test_capillary_wave_open_surface.py -q
FAILED tests/test_capillary_wave_open_surface.py::test_artifact_page_and_figure_exist
1 failed, 8 passed
E   AssertionError: assert (False)
E    where False = PosixPath('.../benchmarks/capillary_wave/capillary_wave_open_surface.png').is_file()
```

A gate that is red at HEAD is not a gate: the page rendered a broken image and every
downstream reader of "the audit exists and is gated" was relying on a suite nobody ran.

## 2. Root cause (measured, not guessed)

The figure was written next to the JSON, under `benchmarks/`. Generated binaries do not
survive that location on the way to `main`:

| location | `*.png` tracked at HEAD |
|---|---|
| `benchmarks/**` | **0** |
| `docs/**` | **83** |

and the file was committed on four separate branches, never landing on `main`:

| commit | branch commit subject | carries the PNG |
|---|---|---|
| `9bd7148` | m13812: capillary-wave evidence debt E1 → quantified WITHHELD audit | yes |
| `04f6b24` | m14576: capillary-wave open free surface audit (re-cut of #453) | yes |
| `d9bb365` | m14639: capillary-wave open free-surface audit, round 3 | yes |
| `296dbee` | capillary_wave: audit the open free surface, retire waiver E1 | yes |

`git log --all --diff-filter=D -- benchmarks/capillary_wave/*.png` is empty: the file was
never *added* to `main`, so this is a location problem, not a deletion problem. Four
attempts at the same repair are evidence for changing the design, not for trying a fifth
time.

## 3. Decision and alternatives

**Decision:** the figure becomes a *rendering of the committed artifact*, produced by a
pure function of `open_surface_audit.json`, and it is committed under `docs/figures/`.

| alternative | why not |
|---|---|
| re-commit the PNG under `benchmarks/` (attempt 5) | measured to fail four times; the transport drops it |
| drop the figure from the record | the page's four panels *are* the argument (mode decay, spurious current, curvature noise, CSF excess); a figureless version is a weaker record |
| re-run the 278 s audit to draw the figure | unnecessary: every plotted series is already a field of the JSON; coupling the figure to a simulation makes it unverifiable without a solver |
| ship an SVG instead | avoids the binary question but adds a large XML diff, and it still would not prove the picture equals the measurement |

The design principle is the one the page already used for its numbers: **nothing in the
record is hand-typed, so nothing in it is trusted** — the page is re-rendered from the
JSON and compared, and now the figure is too.

## 4. What changed (5 files)

| file | change |
|---|---|
| `benchmarks/capillary_wave/make_doc.py` | `render_figure(payload, path)` draws the four panels from the JSON alone (no `torch`, no simulator); `FIGURE_DIR = docs/figures`; `--figures` flag; the page now carries the figure's **sha256** (tamper-evident binding) |
| `benchmarks/capillary_wave/open_surface_audit.py` | `make_figures` delegates to `make_doc.render_figure`: one figure, one code path (was ~80 duplicated lines) |
| `docs/figures/capillary_wave_open_surface.png` | the figure, in the location that survives |
| `docs/verification_capillary_wave_open_surface.md` | re-rendered by `make_doc.py`: **numbers untouched** (139 → 141 lines: image link + figure-provenance line + repro command) |
| `tests/test_capillary_wave_open_surface.py` | figure path moved; two new gates: *figure == fresh rendering of the JSON* and *every image link in the page resolves* |

## 5. Verification

```
$ PYTHONPATH=src python benchmarks/capillary_wave/make_doc.py --figures
wrote docs/verification_capillary_wave_open_surface.md (141 lines)
wrote docs/figures/capillary_wave_open_surface.png (210857 bytes)

$ python -m pytest tests/test_capillary_wave_open_surface.py tests/test_capillary_wave.py -q
21 passed

$ python -m pytest tests/test_capillary_wave_open_surface.py -q      # after the sha256 line
11 passed
```

The two new gates are the point:

1. `test_figure_is_a_fresh_rendering_of_the_artifact` — renders from the committed JSON
   into `tmp_path` and compares with the committed PNG (bytes, pixel fallback), so a
   hand-edited or stale figure cannot pass;
2. `test_every_image_link_in_the_page_resolves` — parses the page's markdown image links
   and requires each to exist, i.e. the exact defect above cannot return silently.

`make_doc.py --check` also fails when the figure is missing, so the CI-side stale check
covers the binary too.

## 6. Honest limits

- **Pre-existing, not fixed here.** `tests/test_verification_ledger.py` has 4 failures at
  HEAD *with and without* this change (verified by stashing): `unregistered_artifact` for
  `benchmarks/{aeration,contact_angle,fluctuating_hydro,rayleigh_plateau,
  surface_tension_dispersion}` reports, a waiver with an empty reason
  (`benchmarks/contact_angle/bc_ladder.json`), an unmarked superseded number in
  `docs/verified/fluctuating_hydrodynamics_verification.md`, and 6 uncovered benchmark
  harnesses. `benchmarks/capillary_wave` is **not** among the uncovered ones. Those are
  other areas' debt with other open work on them; touching them here would break the
  one-feature-per-PR discipline.
- **The verdict is unchanged.** This record still grades the capillary-standing-wave
  capability `WITHHELD` (CSF force on an open free surface ≈ 25× the wave's restoring
  acceleration; harness aborts at step 336). Repairing the evidence *transport* says
  nothing about the solver defect; `benchmarks/surface_tension_dispersion/` carries the
  live attempt at that.
- **Figure reproduction is environment-dependent at the byte level** if matplotlib
  changes; the gate compares pixels in that case rather than failing on metadata.