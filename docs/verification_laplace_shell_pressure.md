# Verification: free-surface shell pressure closure

**Mission** m3289 ｜ **Benchmark** `benchmarks/laplace_pressure_floor/` ｜
**Record** `benchmarks/laplace_pressure_floor/closure_report.json` ｜
**Figure** `benchmarks/laplace_pressure_floor/laplace_shell_closure.png`

## 1. The defect this closes

`docs/verification_laplace_pressure_floor.md` reduced the four-round Young-Laplace
miss to one remaining error source (C3): `tait_pressure` clamps at `p >= 0`, and
density summation returns `rho < rho0` over the outer `~2h` of any free surface,
so every particle in that shell sits at exactly `p = 0`. An interface built that
way cannot hold the jump that surface tension needs, and the interior settles at
a *fraction* of `sigma/R` -- measured 0.57 of the required compression at
`dx = 0.002`, independent of `c0` and `dx`, i.e. a closure defect and not a
resolution floor.

## 2. What was implemented

`tensorsph.droplet_impact.free_surface_pressure` implements the shell closure of
Adami, Hu & Adams (2012, J. Comput. Phys. 231, eq. 11) / Ihmsen et al. (2013,
sec. 3.3):

    p_i = max[0, sum_j (p_j + rho_j g.(r_i - r_j)) W_ij / sum_j W_ij]

applied to every fluid particle with `rho/rho0 < 0.99`. Two ingredients matter:
the **renormalisation** by the deficient kernel sum (so the average is exact for
a field that varies linearly over the shell) and the **body-force continuation**
term (so a flat free surface under gravity stays at `p ~ 0` instead of being
inflated by its own 2h of head). The same term was already present in the
static-wall extrapolation of `rhs`; the free surface was missing it.

`Config.free_surface_pressure` (default `True`), `.free_surface_density_ratio`
(0.99) and `.free_surface_pressure_sweeps` (2) control it; `rhs` and the dense
reference `rhs_all_pairs` both apply it, so the two implementations stay
bit-comparable (`tests/test_pair_based_rhs.py`, `tests/test_free_surface_pressure.py`).

## 3. Measured effect

`|Delta p - sigma/R| / (sigma/R)` on the absolute-pressure, tail-time-averaged
read-out; `compression_ratio` is the interior compression divided by the value
the Tait law would need for `sigma/R` (1.00 = the interface holds the full jump).

| dx [m] | c0 [m/s] | |err| closure off [%] | |err| closure on [%] | improvement [pt] | comp. ratio off | comp. ratio on | <15 % |
|---|---|---|---|---|---|---|---|
| 0.0015 | 20 | -38.1 | -15.1 | 23.0 | 0.56 | 0.80 | no |
| 0.0015 | 10 | -36.2 | -12.4 | 23.8 | 0.68 | 0.89 | yes |
| 0.0015 | 5 | -37.4 | -13.4 | 24.0 | 0.51 | 0.79 | yes |
| 0.002 | 20 | -37.9 | -6.5 | 31.4 | 0.57 | 0.70 | yes |
| 0.002 | 10 | -37.2 | -4.6 | 32.7 | 0.50 | 0.98 | yes |
| 0.002 | 5 | -34.0 | -7.0 | 27.0 | 0.57 | 0.83 | yes |

Verdict from `closure_report.json`: 6/6 points
improve by more than 20 percentage points, 5/6
land inside the 15 % bar and 0/6 inside 3 %.
At the reference resolution `dx = 0.002` all 3 sound speeds
are inside 15 % (3/3).

Sweep ladder at `dx=0.002, c0=20`:

| sweeps | |err| [%] |
|---|---|
| 1 | -20.1 |

## 4. Honest residuals

* **`dx = 0.0015, c0 = 20` sits at -15.1 %** (best point
  -4.6 %): the fine-grid, high-sound-speed corner misses the
  15 % bar. The closure is not wrong there -- the error still falls from -38 %
  to -15 % -- but the residual is no longer dominated by C3, and it is
  resolution-dependent in a way C3 was not, which points at the CSF curvature /
  `delta_s` interface operator rather than at the pressure closure. This is an
  open item, not a passed gate.
* **3 % is not reached anywhere.** The remaining gap is the same interface
  operator, not the EOS (C4 was falsified in the previous round and stays
  falsified: `tests/test_laplace_pressure_floor.py` still pins that on the
  closure-off record).
* The legacy `band` read-outs (`p_core - p(0.80..0.95R)`) are still measured in
  `closure_report.json`; they stay contaminated by the density deficit and must
  not be used as an accuracy statement (see P2 in
  `backlog_update_m2971.md`).

## 5. How to reproduce

```
PYTHONPATH=src python3 benchmarks/laplace_pressure_floor/run.py   # full record
PYTHONPATH=src python3 outputs/m3289_grid.py outputs/m3289_grid.json
PYTHONPATH=src python3 outputs/m3289_make_report.py              # this note
PYTHONPATH=src python3 -m pytest -q tests/test_free_surface_pressure.py
```
