# Same-estimator particle-order error: WCSPH vs ISPH plane channel

**Anchor**: `GAP-SPH-ISPH-PROJECTION`, open item (a) of
`docs/isph_projection_channel.md` §5.5 -- that document states in its own words
that the WCSPH density number quoted by the gap rule (`2.3e-2`) is *not*
re-measured with the ISPH estimator, so the two numbers are not like-for-like.
**Code**: `benchmarks/density_order_likeforlike/run.py`,
`benchmarks/density_order_likeforlike/make_doc.py`,
`tests/test_density_order_likeforlike.py`.
**Artefact**: `benchmarks/density_order_likeforlike/out/results.json`
(machine-written; every number below is injected from it).

## 1. Case

Plane Poiseuille channel, the case of the ISPH anchor: height `H = 0.1 m`,
periodic in `x` over `12` columns, no-slip walls at `y = 0` and
`y = H`, `nu = 0.01 m^2/s`, body force `g_x = 9.81 m/s^2`,
`rho0 = 1000.0 kg/m^3`, `t_end = 1.0 s`, diagnostic averaged over
the last `0.1 s`. Analytic reference `u_max = g H^2/(8 nu)` =
`1.22625` m/s.

WCSPH side: the channel of `benchmarks/poiseuille/run.py` (Tait EOS,
`c0 = 20.0` m/s, Adami wall pressure, wall friction 1.0, 3-layer
band with a 3-column `x` margin and columns at `x = k dx`, i.e. *the same float
expression as that benchmark*). ISPH side: `tensorsph.isph_channel.run_case`
unmodified.

## 2. How

One estimator, applied to both clouds:

1. `rho_i = sum_j m_j W_ij` over **fluid and wall** particles with the minimum
   image in `x`, using `tensorsph.isph_channel.pair_geometry` and
   `tensorsph.isph_channel.summation_density` -- the two functions the ISPH path
   itself calls (identity pinned numerically by
   `test_estimator_is_the_isph_pair_of_functions`);
2. lattice-calibrated mass `m = rho0 / lattice_kernel_sum(h, dx)`, so an ordered
   lattice reads `rho = rho0` and the number is *particle disorder*, not the
   discrete kernel-sum offset (`test_calibrated_mass_makes_kernel_sum_bias_vanish`
   pins that offset at `dx^2 S - 1 ~ 5.9e-3`);
3. float64 arithmetic, diagnostic subsampled every 4 steps of the averaging
   window, reported as `max` and `rms` of `|rho/rho0 - 1|` over fluid particles.

Self-check on every run: the estimator is applied to the *undisturbed* lattice
before the first step and must read `<= 1e-06`
("reference gate" column). Without it a blind diagnostic would look like a good
result.

## 3. Results

WCSPH:

| H/dx | h/dx | max order err | rms order err | reference gate | max Tait rho err | u_max [m/s] | dt [s] | steps |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.9 | 8.511e-01 | 2.927e-01 | 1.2e-01 | 8.740e-01 | 2.423 | 3.56e-04 | 2807 |
| 12 | 1.9 | 7.243e-01 | 2.143e-01 | 1.1e-15 | 8.740e-01 | 2.275 | 2.37e-04 | 4211 |

ISPH (its own committed path, same estimator):

| H/dx | h/dx | max order err | u_max [m/s] | steps | wall clock [s] |
|---|---|---|---|---|---|
| 8 | 1.9 | 1.360e-04 | 1.261 | 197 | 2.5 |
| 12 | 1.9 | 3.047e-04 | 1.243 | 320 | 5.0 |

Like-for-like pairing (the `h/dx` values must agree, otherwise the pair is
flagged and the grade fails):

| H/dx | h/dx | WCSPH max | ISPH max | ratio WCSPH/ISPH | like-for-like |
|---|---|---|---|---|---|
| 8 | 1.9 | 8.511e-01 | 1.360e-04 | 6260.2 | yes |
| 12 | 1.9 | 7.243e-01 | 3.047e-04 | 2377.2 | yes |

Gates: `reference_zero = False`,
`estimator_identity = True`,
`verdict = FAIL`.

![particle-order error](figures/density_order_likeforlike.png)

## 4. What this settles, and what it does not

1. **The estimator is reused, not re-implemented**, so the two schemes now share
   one yardstick; the pairing table is the only place a cross-scheme density
   statement may be read from.
2. **2 of 2 pairs are like-for-like.**
   Rows whose `h/dx` differs from the ISPH value are the repository's production
   channel setting and must **not** be quoted against the ISPH density figure.
3. **`max Tait rho err` is context, not a comparison**: it is the weakly
   compressible equation-of-state density, a different quantity from the
   summation-density order metric. Both are printed so neither is hidden.
4. **No convergence order is claimed**: the ladder has 2 resolutions and the
   `max` statistic is dominated by a single particle at the coarser end (§5.1),
   which is why the `rms` column exists.

## 5. Notes and traps

1. **The WCSPH channel is round-off sensitive.** Rebuilding the mathematically
   identical wall band with a different float expression shifts the wall columns
   by ~`1.5e-8` m and changed `u_max` at `H/dx = 8` from `1.57` to `2.32` m/s at
   `t = 0.5` s during this work (measured, see `run.py`'s wall builder comment).
   The script therefore copies the published expression; an "equivalent" band is
   not interchangeable, and the `max` statistic inherits that sensitivity.
2. **Two wall clouds, deliberately.** Dynamics uses the published 3-layer band
   with a margin (`rhs` uses unwrapped distances, so the `+-L` fluid images need
   wall columns that physically exist). The diagnostic uses a 5-layer one-period
   band at `x = dx/2 + k dx` (the ISPH convention): the minimum-image fold
   already supplies the images and a margin would double count them. A half-cell
   misalignment of that band is a negative control in the tests (it inflates the
   first fluid row's summation density to `~1.96 rho0`).
3. **Order is not incompressibility.** The ISPH path has no equation of state at
   all; in WCSPH order and compressibility are coupled through the acoustic
   time-step. That is why the two columns stay separate.
4. **Cost.** WCSPH runs at the acoustic limit (`dt` column, up to
   `4211` steps); ISPH has no acoustic limit (up to
   `320` steps).

## 6. Error judgement (required question: are the key quantities < 3 %?)

The reported quantity is a *scheme-to-scheme diagnostic difference*, not a
discretisation error against a closed form, so there is no percentage against an
analytic reference to quote. What is judged instead:

| item | reference | outcome |
|---|---|---|
| estimator on an ordered lattice | exact 0 | `<= 1e-06` at every run (gate) |
| estimator detects disorder | half-cell misaligned band | detected, `> 0.05` absolute |
| like-for-like pairing | equal `h/dx` | 2 qualifying pair(s) |
| WCSPH `u_max` sanity (not a gate) | analytic `1.22625` m/s | see `u_max [m/s]` column |

Honest statement: the WCSPH-vs-ISPH density figures are **not** quoted as a
percentage error against an analytic solution, because none exists for either
diagnostic; the "<3 %" question is answered as *not applicable, replaced by the
ordered-lattice self-check* rather than asserted.

## 7. Honest status: the cross-scheme number is WITHHELD, not delivered

The gates of this artefact are **FAIL** and the reason is not cosmetic:

1. **Ordered-lattice self-check fails at the coarse lattice.** The estimator
   reads `1.1650e-01` on an *undisturbed*
   cloud at `H/dx = 8` (`dx = 0.0125`, `h = 1.9 dx`) while the same construction
   at `H/dx = 12` reads `1.1e-15`. Until
   that residual is explained, any "disorder" number at the coarse end could be
   an artefact of the diagnostic rather than of the solution. Cause unknown at
   the time of writing; recorded as an open defect and marked `xfail` in
   `tests/test_density_order_likeforlike.py`.
2. **The WCSPH channel in this configuration is not in the published state.**
   `max Tait rho err` reaches `0.874`
   and `u_max = 2.42 m/s` at `H/dx = 8`
   against `1.52` m/s in `benchmarks/poiseuille/out/results.json`; §5.1 shows the
   run is sensitive to `1.5e-8` m wall-lattice round-off, so a like-for-like
   *density* statement on top of it is not defensible yet.
3. Therefore the gap rule's `2.3e-2` WCSPH figure remains **WITHHELD** as a
   cross-scheme claim: this work removes the estimator mismatch (one yardstick
   now exists and is tested) but does not certify the resulting ratio. Do not
   quote the `ratio WCSPH/ISPH` column of §3 as an accuracy result.
