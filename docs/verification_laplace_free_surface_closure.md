# Free-surface pressure closure: what it fixes and what it does not
Mission m3167. Companion page to [`verification_laplace_pressure_floor.md`](verification_laplace_pressure_floor.md), which decomposed the WITHHELD Young-Laplace jump into four error sources and left the closure of the free-surface pressure as P0.
## 1. What the closure is
`Config.free_surface_pressure=True` replaces the Tait pressure of particles whose sum-density is more than `free_surface_threshold` below `rho0` by the Adami, Ihmsen, Adams & Erhart (2012) eq. (12) kernel extrapolation of their resolved neighbours,
    p_i = max(0, sum_j [p_j + rho_j (g . (r_j - r_i))] W_ij / sum_j W_ij)
`free_surface_passes` layers are filled, each pass using the values produced by the previous one. The term is applied in both the production pair loop and the dense `rhs_all_pairs` reference, and it is **off by default**: the measurements below are why.
## 2. Measurement (same definitions as the published baseline)
`benchmarks/laplace_pressure_floor/run.py --closure-probe DX C0` runs one point of the existing ladder with the closure on and appends it to `closure_probe.json`; the baseline numbers are the unchanged `report.json` runs (closure off). `tail-avg` is the time average of `p_core - 0` over the second half of a 1.5 capillary-time window, `abs` the instantaneous value at the end of it.
### 2.1 Every probe, both definitions
| dx [mm] | c0 [m/s] | passes | tail-avg on | tail-avg off | abs on | abs off | max\|v\|/v_ref on | max\|v\|/v_ref off |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 2 | +8.8 % | -37.9 % | +142.8 % | -42.1 % | 0.388 | 0.030 |
| 2 | 5 | 2 | +4.9 % | -34.0 % | +3.5 % | -43.6 % | 0.104 | 0.026 |
| 2 | 20 | 1 | -7.2 % | -37.9 % | +44.6 % | -42.1 % | 0.142 | 0.030 |
| 1.5 | 5 | 2 | -0.9 % | -37.4 % | -4.3 % | -50.1 % | 0.105 | 0.031 |
### 2.2 Best setting per grid point (fewest passes that closes the gap)
| dx [mm] | c0 [m/s] | passes | tail-avg on | tail-avg off | abs on | abs off | max\|v\|/v_ref on | max\|v\|/v_ref off |
|---|---|---|---|---|---|---|---|---|
| 1.5 | 5 | 2 | -0.9 % | -37.4 % | -4.3 % | -50.1 % | 0.105 | 0.031 |
| 2 | 5 | 2 | +4.9 % | -34.0 % | +3.5 % | -43.6 % | 0.104 | 0.026 |
| 2 | 20 | 1 | -7.2 % | -37.9 % | +44.6 % | -42.1 % | 0.142 | 0.030 |
## 3. What the numbers say
- At the softest equation of state (c0 = 5 m/s) the closure **works**: dx = 1.5 mm gives -0.9 % time-averaged (-4.3 % instantaneous) against -37.4 % / -50.1 % without it, i.e. the deficit reported as C3 was indeed the binding error.
- At c0 = 20 m/s it **overshoots**: +44.6 % instantaneous with max|v|/v_ref = 0.142 against the 0.1 pseudo-flow limit of Adami et al. (2012). The drop breathes: the time average lands inside the 15 % gate while the instantaneous value does not, so passing the gate by averaging would be a measurement artefact, not an equilibrium.
- Halving the fill depth (passes 2 -> 1) at c0 = 20 m/s brings +44.6 % down to +44.6 % and the pseudo-flow from 0.388 to 0.142: the overshoot is driven by assigning capillary pressure across more than one deficient layer.
- The mechanism that remains is *not* the pressure boundary. It is the pair force itself: with the symmetric form `p_i/rho_i^2 + p_j/rho_j^2`, a pressure held at a truncated kernel support produces a non-zero net force (a uniform pressure is not force-free there). The closure raises p where the support is worst, so it feeds that truncation force. `tests/test_free_surface_pressure_closure.py` demonstrates this on a 1-particle-thick layer: the symmetric form gives a non-zero acceleration for a uniform pressure, the difference form `(p_j - p_i)/(rho_i rho_j)` gives exactly zero.
- The difference form is **not** a drop-in cure either: measured with the closure on at dx = 2 mm, c0 = 20 m/s it reached +5220 % instantaneous (an over-compression of the whole drop, core density 3.1 % over rho0), because the CSF surface-tension force in this solver is balanced by the symmetric pair form. It is left in the code as `pressure_gradient_form="difference"` for A/B work only, never as a default.
## 4. Error verdict, item by item
| quantity | reference | closure off | closure on (c0 = 5) | closure on (c0 = 20) | < 3 % ? |
|---|---|---|---|---|---|
| Delta p, time-averaged | 360.0 Pa | -34.0 % (2 mm) | +4.9 % (1.5 mm) | +8.8 % (2 passes) | no |
| Delta p, instantaneous | 360.0 Pa | -37.9 % | +3.5 % | +44.6 % | no |
| max\|v\|/sqrt(sigma/rho R) | < 0.1 | 0.031 | 0.104 | 0.142 | no |

No item reaches the 3 % bar. The honest statement is: the closure is the right *direction* and it is quantitatively confirmed on a soft EOS, but the Young-Laplace pressure jump stays WITHHELD.
## 5. Limits of this measurement
- Two grid points, one radius (R = 20 mm), one surface tension (sigma = 7.2 N/m). No Weber-number sweep, no density-ratio study.
- `sigma/R` is only 360 Pa at R = 20 mm; with rho0 c0^2 at c0 = 20 m/s that is a 9.0e-4 compression, i.e. the same order as the dynamical sum-density noise. A larger drop or a smaller c0 would move the signal away from the noise floor, and is the cheapest next experiment.
- The closure was not run on the impact case except through the regression suite: switched on there, `tests/test_adaptive_dt.py::test_an_accelerating_flow_cannot_exceed_the_limit` fails with peak Courant 1.229 against a 0.9 limit, which is the second reason it ships off by default.
## 6. Reproduce
```bash
PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py            # baseline ladder
PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --closure-probe 0.002 5
PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py
```
