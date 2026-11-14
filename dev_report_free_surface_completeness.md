# Dev report — parameter-free free-surface closure of the SPHinXsys CSS flux

**Anchor** `XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION` · **Mission** m20292 (self-start, models
seat, TensorSPH) · **Verdict** the free-surface path's −30 % deficit is a *measured* support
truncation, not a model limitation: a support-completeness renormalisation with **zero free
parameters** brings it to +3.2 … +4.1 % (gate < 15 %, not < 3 %).

## 1. Injection point

The anchor's two-phase reading already passed the gate (`docs/xval_sphinxys_mixture_laplace`), but
the single-phase (vacuum-reduced) reading was −30 % and had been "closed" by a **fitted** coefficient
`alpha* = 0.4263` on a specular mirror (`benchmarks/surface_tension/out_single_phase_completion`).
A fitted coefficient is not evidence. This mission replaces it with a measurement.

## 2. Method

* `M_i = sum_j V_j (x_j − x_i) ⊗ ∇_i W_ij`, `s_i = trace(M_i)/2` is the *filled fraction of the
  kernel*: `1` complete support, `1/2` exact half-space, interpolating with depth.
* Closure `∇W_ij → c_ij ∇W_ij`, `c_ij = 1/[(s_i+s_j)/2]`, floor `s ≥ 0.25`. **One factor per pair**
  (not per particle) — the per-particle version destroys the pair cancellation and returns
  `Δp ~ 1e4 Pa`, a 5-order blow-up that is itself the evidence for why the factor must be
  pair-symmetric: only then are the two ends scaled identically and momentum stays exact.
* Implemented as `completion="completeness"` in `src/tensorsph/sphinxys_css.py`
  (`support_matrix`, `support_fraction`, `pair_renormalisation`); the literal upstream
  transcription (`completion="none"`) and the mirror closure are untouched, so every earlier
  artifact reproduces bit-for-bit.

## 3. Numbers (all from `benchmarks/surface_tension/out_free_surface_completeness/gate.json`)

| R/Δx | one-sided | mirror α=1 | mirror fitted α\* | completeness |
|---|---|---|---|---|
| 16 | −29.74 % | +41.74 % | +0.73 % | **+4.05 %** |
| 32 | −30.01 % | +40.38 % | +0.00 % | **+3.17 %** |
| 64 | −29.88 % | +39.78 % | −0.18 % | **+3.15 %** |

Gain 7.4×; momentum residual ≤ 4.4e−17; cross-parameter sweep (kernel × h/dx, nothing re-tuned):
+3.17 … +7.89 %, worst case inside the gate. Measured support: `s = 0.745` in the interface band,
`0.973` in the bulk (the bulk part is `h/dx = 1.3` lattice quadrature, removed by the same factor).

## 4. Is it < 3 %?

**No** — +3.15 … +4.05 %. The residual is a ~3 % over-restoration: `trace(M)/2` assumes the missing
solid angle is a half-space, while δ-band particles are graded in depth. The anisotropic 2×2
inverse was measured too and under-restores (−7.7 … −9.0 %), so the isotropic support fraction is
kept. Stated as-is; no coefficient was fitted to improve it.

## 5. Artifacts

| path | content |
|---|---|
| `src/tensorsph/sphinxys_css.py` | `support_matrix`, `support_fraction`, `pair_renormalisation`, `completion="completeness"` (additive) |
| `benchmarks/surface_tension/free_surface_completeness.py` | ladder × 4 closures + kernel/h sweep + support profile → `out_free_surface_completeness/{gate.json,gate.png}` |
| `tests/test_free_surface_completeness.py` | 5 tests: identity in the bulk, `1/2` on an exact half-space, unit factor / floor, momentum + gate at R/Δx = 16,32 |
| `docs/xval_sphinxys_free_surface_completeness/{README.md,tables.md}` | XVAL report + programmatic tables |
| `runs/m20292/{probe_complete.py,probe_pair_complete.py,emit_tables.py}` | the two diagnostic probes and the table emitter |

## 6. Checks run

```
PYTHONPATH=src python -m pytest tests/test_free_surface_completeness.py tests/test_sphinxys_css.py \
    tests/test_mixture_laplace.py tests/test_sphinxys_css_flux_audit.py -q     # 21 passed
PYTHONPATH=src python benchmarks/surface_tension/free_surface_completeness.py  # 3.4 s, gate pass
```

## 7. Next leads

1. The remaining +3 % is the isotropic-vs-anisotropic support question; a principled anisotropic
   form (support tensor projected on the interface tangent plane) is the next test.
2. The two-phase reading (−6.2 %) and the free-surface reading (+3.2 %) now bracket the exact
   answer from opposite sides: one closure that treats a free surface as the two-phase limit
   `ρ_other/ρ_self → 0` would unify them.
3. Nothing here was run on the cluster; every configuration is a 3 s CPU job.
