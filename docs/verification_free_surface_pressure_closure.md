# Free-surface pressure closure: the Young–Laplace gate is met (P0 of m2971)

**Mission**: m3246 ｜ **Repo**: TensorSPH ｜ **Date**: 2026-11-04
**Type**: solver change (`src/`) + verification benchmark + tests ｜ **Status**: P0 acceptance criterion **met**; < 3 % target met at 2 of 4 points

---

## 1. Why this change

`docs/verification_laplace_pressure_floor.md` (m2971) decomposed the WITHHELD
Young–Laplace jump into four error sources and found exactly one that mattered:

| source | measurement | verdict |
|---|---|---|
| C1 measurement definition | −44…−55 pt | artefact of the old `p_core − p(shell)` read-out |
| C2 lattice kernel-sum bias | ±9 pt steady | real, but P2-sized |
| C3 **free-surface pressure closure** | compression **0.57 ± 0.06** of what $\sigma/R$ needs, c0- and dx-independent | **the gate** |
| C4 EOS noise floor | residual does not follow the predicted noise | falsified |

C3 is structural, not numerical: `tait_pressure` is normalised at $\rho_0$ and
clamped at $p\ge 0$, while the whole outer shell of a lattice drop has a
*summation-density deficit* (−11.6 % at 0.85–0.95 R, −34 % at 0.95–1.05 R). That
shell therefore carries $p = 0$ exactly, so it cannot hold any of the jump the
interface must support; the interior can only carry the fraction the shell does
not take — 0.57.

## 2. What was implemented

`Config.free_surface_pressure` (default **off**, backward compatible) activates
`extrapolate_free_surface_pressure()` in `src/tensorsph/droplet_impact.py`, the
shell-pressure boundary condition of **Adami, Ihmsen & Adams (2012) §3.3**:

$$p_i = \max\Big(0,\; \frac{\sum_j \big(p_j + \rho_j\, \mathbf{g}\cdot(\mathbf{r}_i-\mathbf{r}_j)\big) W_{ij}}{\sum_j W_{ij}}\Big)
\qquad \text{for } \rho_i < 0.99\,\rho_0 .$$

Design decisions, each pinned by a test:

* **only the flux pressure is closed.** `rho` and hence the Tait law are
  untouched, so the next step sees the same density field; the closure is a
  boundary condition on $p$, not a fudge on $\rho$. `rhs` returns the pressure
  the momentum flux actually used.
* **the body term uses $\rho_j$, not $\rho_0$** — Adami's form. On a linear
  hydrostatic field this is an exact Taylor identity; the residual from the
  shell's own density deficit is bounded below.
* **wall pairs do not contribute.** The wall pressure is itself extrapolated
  from the fluid; feeding it back would close the loop
  (`test_closure_ignores_wall_particles`).
* **`free_surface_pressure_passes`** sweeps the closure per step. One sweep only
  reaches one layer, because the sweep reads the *thermodynamic* pressure
  ($p=0$ across the whole band). Three sweeps fill the band; six overshoot (§4).

## 3. Results

Reference: $\sigma/R = 360$ Pa ($\sigma$ = 7.2 N/m, $R$ = 20 mm,
$\rho_0$ = 1000 kg/m³, $t_\sigma$ = 0.0333 s; 1.5 $t_\sigma$ of relaxation,
mass-calibrated lattice, gravity = 0). Errors are the **time average over the
last 40 %** of the window — the drop breathes at $t_\sigma$, so a final-snapshot
reading is shown only as a stability indicator.

### Acceptance criterion (m2971 §7: four points, all inside 15 %)

| sweeps | all four points within 15 % | worst abs error | per point (dx\_c0) |
|---|---|---|---|
| 0 | no | 38.10 % | 0.002_20.0: -37.9 %, 0.002_5.0: -34.0 %, 0.0015_20.0: -38.1 %, 0.0015_5.0: -37.4 % |
| 1 | no | 25.26 % | 0.002_20.0: -20.1 %, 0.002_5.0: -19.9 %, 0.0015_20.0: -25.3 %, 0.0015_5.0: -24.7 % |
| 2 | no | 15.10 % | 0.002_20.0: -6.5 %, 0.002_5.0: -7.0 %, 0.0015_20.0: -15.1 %, 0.0015_5.0: -13.4 % |
| 3 | yes | 6.77 % | 0.002_20.0: -1.8 %, 0.002_5.0: -0.6 %, 0.0015_20.0: -6.8 %, 0.0015_5.0: -6.4 % |
| 6 | yes | 12.09 % | 0.002_20.0: +12.1 %, 0.002_5.0: +5.6 %, 0.0015_20.0: -3.8 %, 0.0015_5.0: -2.8 % |

**Verdict: passes = 3 meets the P0 criterion at all four points, worst 6.77 %**
(open loop: worst 38.1 %). One sweep already removes ~18 pt, two ~28 pt.

### Full ladder

**Closure off (baseline, the previous published configuration)**

| dx [mm] | c0 [m/s] | N | steps | $\Delta p$ [Pa] | time-avg err vs $\sigma/R$ | final-snapshot err | compression | $v_{max}/v_\sigma$ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 5 | 642 | 1052 | 225.3 | -37.4 % | -50.1 % | 0.63 | 0.031 | outside 15 % |
| 1.5 | 20 | 642 | 1710 | 222.9 | -38.1 % | -43.6 % | 0.62 | 0.025 | outside 15 % |
| 2.0 | 5 | 362 | 592 | 237.6 | -34.0 % | -43.6 % | 0.66 | 0.026 | outside 15 % |
| 2.0 | 20 | 362 | 1283 | 223.6 | -37.9 % | -42.1 % | 0.62 | 0.030 | outside 15 % |

**1 sweep**

| dx [mm] | c0 [m/s] | N | steps | $\Delta p$ [Pa] | time-avg err vs $\sigma/R$ | final-snapshot err | compression | $v_{max}/v_\sigma$ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 5 | 642 | 1052 | 271.1 | -24.7 % | -36.8 % | 0.75 | 0.021 | outside 15 % |
| 1.5 | 20 | 642 | 1710 | 269.1 | -25.3 % | -30.5 % | 0.75 | 0.019 | outside 15 % |
| 2.0 | 5 | 362 | 592 | 288.5 | -19.9 % | -34.0 % | 0.80 | 0.023 | outside 15 % |
| 2.0 | 20 | 362 | 1283 | 287.5 | -20.1 % | -21.3 % | 0.80 | 0.020 | outside 15 % |

**2 sweeps**

| dx [mm] | c0 [m/s] | N | steps | $\Delta p$ [Pa] | time-avg err vs $\sigma/R$ | final-snapshot err | compression | $v_{max}/v_\sigma$ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 5 | 642 | 1052 | 311.7 | -13.4 % | -22.1 % | 0.87 | 0.032 | pass |
| 1.5 | 20 | 642 | 1710 | 305.6 | -15.1 % | -20.5 % | 0.85 | 0.032 | outside 15 % |
| 2.0 | 5 | 362 | 592 | 334.9 | -7.0 % | -18.0 % | 0.93 | 0.039 | pass |
| 2.0 | 20 | 362 | 1283 | 336.6 | -6.5 % | -30.1 % | 0.93 | 0.033 | pass |

**3 sweeps — recommended setting**

| dx [mm] | c0 [m/s] | N | steps | $\Delta p$ [Pa] | time-avg err vs $\sigma/R$ | final-snapshot err | compression | $v_{max}/v_\sigma$ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 5 | 642 | 1052 | 337.0 | -6.4 % | -13.5 % | 0.94 | 0.038 | pass |
| 1.5 | 20 | 642 | 1710 | 335.6 | -6.8 % | +0.2 % | 0.93 | 0.032 | pass |
| 2.0 | 5 | 362 | 592 | 358.0 | -0.6 % ( <3 % ) | -7.5 % | 0.99 | 0.029 | pass |
| 2.0 | 20 | 362 | 1283 | 353.4 | -1.8 % ( <3 % ) | -25.4 % | 0.98 | 0.031 | pass |

**6 sweeps — documented failure mode**

| dx [mm] | c0 [m/s] | N | steps | $\Delta p$ [Pa] | time-avg err vs $\sigma/R$ | final-snapshot err | compression | $v_{max}/v_\sigma$ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 5 | 642 | 1052 | 349.9 | -2.8 % ( <3 % ) | -2.5 % | 0.97 | 0.517 | **unstable** |
| 1.5 | 20 | 642 | 1710 | 346.4 | -3.8 % | +25.3 % | 0.96 | 0.203 | **unstable** |
| 2.0 | 5 | 362 | 592 | 380.1 | +5.6 % | +13.6 % | 1.06 | 0.306 | **unstable** |
| 2.0 | 20 | 362 | 1283 | 403.5 | +12.1 % | +114.4 % | 1.12 | 0.267 | **unstable** |

Sum of the wall clock of all 20 points: 1604 s
(189 s wall at 10 processes).

![closure ladder](benchmarks/laplace_pressure_floor/closure_ladder.png)

Panel (a): error vs sweeps, with the ±15 % band. Panel (b): the radial pressure
profile that explains the mechanism — with the closure off, the profile dies to
$p=0$ inside the shell and never reaches $\sigma/R$; three sweeps continue the
interior value through the shell and the plateau lands on $\sigma/R$.

## 4. Notes, limitations, applicability

1. **More sweeps is not better.** At 6 sweeps the tail error is *smaller* (−2.8…+5.6 %)
   but the spurious-current criterion fails ($v_{max}/v_\sigma$ = 0.20–0.52 vs the
   0.1 bound of Adami et al.) and the final snapshot drifts (+114 % at 2 mm/20 m/s).
   Over-closing makes the shell pressure a fixed point of an inconsistent operator:
   the band carries $\sigma/R$ in $p$ while its density says $p=0$, and the mismatch
   feeds back as noise. Three sweeps is the largest setting that stays in the
   low-current regime on all four points.
2. **Time average, not snapshot.** The 2 mm / 20 m/s / 3-sweep final snapshot reads
   −25.4 % while its time average reads −1.8 %: the drop's fundamental mode has not
   decayed after 1.5 $t_\sigma$ (viscosity 0.01 m²/s). Snapshots must not be quoted.
3. **Gravity is zero in this benchmark** ($\sigma/R \gg \rho_0 g R$ for a 2 cm drop is false —
   it is not; the benchmark sets g = 0 to isolate the capillary balance). The
   hydrostatic branch of the closure is therefore verified by *unit* algebra
   (`test_hydrostatic_shift_recovers_the_linear_field`), not end-to-end. A
   hydrostatic column case is the natural next check.
4. **The density field is still inconsistent.** P1 of m2971 (kernel-sum mass
   calibration inside the solver) is untouched: the benchmark calibrates the
   particle mass externally. With $c_0 = 20$ an uncalibrated run still starts with
   a ~2.4 kPa spurious pressure.
5. **Scope**: 2-D, static drop, lattice initialisation, one curvature scheme
   (`fit_quadratic`). The closure is generic but these numbers are not.
6. Flag is **off by default** because every number in `docs/benchmarks/` was
   produced without it; turning it on changes a boundary condition, so it must be
   A/B'd per case (dam break, droplet impact).

## 5. Error judgement against the benchmark

Analytic reference $\sigma/R = 360$ Pa, worst-case time-averaged error over
the four acceptance points at the recommended setting (3 sweeps): **6.77 %**.

Per point: 1.5mm/c0=5: -6.4 %, 1.5mm/c0=20: -6.8 %, 2.0mm/c0=5: -0.6 %, 2.0mm/c0=20: -1.8 %.

* Is the **P0 criterion (< 15 %)** met? **Yes** — worst 6.77 %.
* Is every point **< 3 %**? **No.** 2 of 4 are inside 3 % (−0.6 %, −1.8 %); the
  1.5 mm points are −6.4 % and −6.8 %. The residual is not a tuning artefact of
  the closure (it is monotone in the sweep count and still shrinking at 3
  sweeps) — it is the remaining surface-tension discretisation error: the
  consistent CSF curvature carries a +22 % bias on a lattice circle
  (`.scratch/` attribution from m2971's audit), and a 15 % error in $\kappa$
  propagates 1:1 into $\Delta p$. Addressing it is a curvature-operator task,
  not a pressure-closure one.

## 6. Reproduce

```bash
PYTHONPATH=src python benchmarks/laplace_pressure_floor/closure_ladder.py --jobs 10   # 20 points, ~3 min wall
PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py           # figure + tables
PYTHONPATH=src python -m pytest tests/test_free_surface_pressure.py -q
```

Artifacts: `benchmarks/laplace_pressure_floor/closure_ladder.json` (all 20 runs,
including per-shell profiles and traces), `closure_ladder.png`, this document.
