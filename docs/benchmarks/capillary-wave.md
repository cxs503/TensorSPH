# Capillary-gravity wave: dynamic verification of surface tension (WITHHELD)

**Status: verification debt opened, not closed.** This page documents what the
benchmark does, what it already proved about the shipped code, and why no
accuracy number is quoted yet. Numbers below are from the trials recorded in
the JSON/logs of `benchmarks/capillary_wave/`; regenerate the page with
`python benchmarks/capillary_wave/make_doc.py` once a full run exists (the
script refuses to invent numbers - it reads `validation_report.json`).

## 1. Why this case exists

The Young-Laplace gate (`docs/validation/young_laplace_csf.md`,
`docs/verification_surface_tension_laplace.md`) verifies the *static* side of
surface tension. The *dynamic* side is a different operator chain: curvature of
a moving interface, momentum coupling, the free-surface treatment. A model can
pass Laplace and still miss the dispersion relation. The classical reference is
the capillary-gravity standing wave

    omega^2 = (g k + sigma k^3 / rho) tanh(k H)        (Lamb 1932, Art. 267)

`benchmarks/capillary_wave/run.py` sets that wave up and measures its period.

## 2. What was found about the shipped state (before any new physics)

The `benchmarks/capillary_wave/` directory existed but had never produced a
result:

| finding | evidence |
|---|---|
| the runner could not start | `Config.dt_from_courant()` no longer exists -> `AttributeError` on line 203 |
| the observable was unsigned | `track_amplitude` returned `max abs deviation >= 0`; `find_period` looks for sign changes, so it could never return a period and the script always exited 1 |
| the physics setup had no equilibrium | with `g = 0` a finite pool is not flat: measured mean-level drift **45 mm on a 190 mm pool** (6 amplitudes), the mode is destroyed |
| the initial condition was in free fall | masses from the hydrostatic density of a square lattice leave the SPH density **0.8 % low**; that is larger than the hydrostatic compression `p_bottom/B = 2.1 %`, so the pressure collapsed to the `max(p,0)` clamp and the interior started at **-33 m/s^2** (wave scale is `omega^2 A = 2.2 m/s^2`) |

So the dynamic dispersion relation of the CSF surface-tension model is
**WITHHELD**: implementation present, no valid dynamic verification exists.

## 3. Setup of the rebuilt benchmark

| parameter | value |
|---|---|
| `sigma` | 0.72 N/m (10x water: compresses the period 3.2x, keeps `ak` fixed; the dispersion relation is unchanged) |
| `g` | 9.81 m/s^2 |
| `lambda` | 17.0 mm (2 pi capillary length of water) |
| `k` | 369.6 1/m |
| `H` | 2.72 mm, `kH = 1.005` (intermediate depth; `tanh(kH)=0.762` is in the reference) |
| amplitude | 0.54 mm (`ak = 0.20`, `A = 3 dx`) |
| `dx` | 0.18 mm, 710 particles, 8.5 x 4.35 mm box |
| `c0` | 2.0 m/s (Mach of the wave ~0.05) |
| CSF scheme | `consistent` (kernel-sum colour, MLS operators), curvature `mls` |
| gravity share of `omega^2` | 9.1 % (so >90 % of the signal is the capillary term) |
| analytical period | 0.03596 s |

Modelling choices: free-slip walls with the mode having antinodes on both
(`pool_width = lambda/2`); no viscosity, no density diffusion; initial
condition = displaced interface plus the *local* hydrostatic density
(`p = rho0 g (eta(x) - y)`) and one global mass rescale so the lattice
kernel-sum density matches that target (the "free fall" entry above).
Observable = mass-weighted `cos(kx)` mode of the vertical velocity,
`a_v = <m v_y cos kx>/<m cos^2 kx> = -A omega sin(omega t)`: smooth, averaged
over every particle, exact zeros every `T/2`. The interface height is still
recorded for the physics picture but it is quantised to `dx` and noise
dominated at `A/dx = 3`, which is why it is not the period observable.

## 4. Results

No completed production run in this session: the 1573-step run (1.3 periods,
1 case) did not finish on the host, which was running at load average
**230-340 on 128 cores** (2.6x oversubscribed) while it was started.
Measured per-step cost: **21-25 ms at 710 particles**, single threaded, i.e.
~35 s of CPU per period per resolution level when the host is quiet.

Consequently:

* measured period: **n/a**
* relative error vs `omega^2 = (gk + sigma k^3/rho) tanh(kH)`: **n/a**
* "is it < 3 %?": **cannot be answered - no measurement**
* "is it < 15 %?": **cannot be answered - no measurement**

The evidence level therefore stays **WITHHELD** and no claim about dynamic
surface tension may be derived from this page.

## 5. What is now in place, and how to close the debt

Delivered and pinned by tests (24 passed, `tests/test_capillary_wave.py`):

* analytical reference including gravity and the `tanh(kH)` factor, checked
  against its own decomposition (gravity share = 50 % at the capillary length by
  construction);
* both modal observables on synthetic fields (a `cos(kx)` velocity field is
  recovered to 1e-6; a `sin(kx)` field projects to zero; horizontal motion is
  ignored);
* the period estimator on a *synthetic damped sinusoid* with a known period
  (2e-3 relative) and a monotone signal that must return `None`;
* the hydrostatic initial condition: kernel-sum density equals the analytical
  hydrostatic density to 2e-3 in the interior, and the initial interior
  acceleration is bounded away from `g`;
* an in-process smoke run of the case driver and of the report schema.

Next steps, in order of expected payoff:

1. Re-run the wired ladder on a quiet host:
   `python benchmarks/capillary_wave/run.py --dx-list 0.00018,0.000135 --json-out benchmarks/capillary_wave/validation_report.json`
   then `python benchmarks/capillary_wave/make_doc.py`.
2. If the period error is large, A/B the two operators that are most likely
   responsible: `--csf-curvature fit_circle` (the `mls` curvature is +25 % on a
   circle, per `docs/verification_surface_tension_curvature.md`) and a
   density-diffusion run (`cfg.density_diffusion`) to suppress the
   particle-scale pressure noise, which at 1 % density jitter is of the same
   order as `p_bottom` itself.
3. Report the numerical damping (`envelope decay` in the JSON) alongside the
   period: the inviscid theory predicts 1.0, so any decay is numerical and
   bounds the credibility of the crossing fit.
