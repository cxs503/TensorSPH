# Ice-ocean thermal-saline-mechanical multi-physics: verification record

*Campaign node*: `IN13W2-N07` -- *repository*: `TensorSPH` -- *module*: `tensorsph.ice_thermal`

Generated 2026-11-10T19:36:44 by `benchmarks/ice_thermal_multiphysics/run.py` (python 3.11.6, torch 2.7.1, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38). Every number below is read out of `results.json` by `make_doc.py`; nothing is transcribed.

## 1. What is verified, against what

| anchor | reference | computed | tolerance | verdict |
|---|---|---|---|---|
| 1-D Stefan problem (SPH enthalpy) | Neumann similarity solution, dx = 0.0200 m | front 0.992 % | 3 % | PASS |
| 1-D Stefan problem (SPH enthalpy) | Neumann similarity solution, dx = 0.0125 m | front 0.072 % | 3 % | PASS |
| 1-D Stefan problem (SPH enthalpy) | Neumann similarity solution, dx = 0.0080 m | front 0.197 % | 3 % | PASS |
| Darcy-Benard onset | exact $Ra_c = 4\pi^2$ | 0.0157 % | 1 % | PASS |
| Timoshenko restrained plate | $E\alpha\Delta T/2(1-\nu)$ | 0.0000 % | 1 % | PASS |
| Free plate, linear gradient | stress-free identity | 4.66e-08 Pa | machine | PASS |
| Free plate, non-linear gradient | net force / moment vanish | 2.26e-10 Pa | 1e-6 $\sigma_{max}$ | PASS |
| Desalination equilibrium | per-level $v_b = v_{irr}$ floor (exact) | min margin 0.614 psu | >= 0 | PASS |
| Desalination vs observation | first-year ice band 3-8 psu | 5.40 psu | 3-8 psu | PASS |
| Salt conservation (spring_two_way) | salt mass balance | 1.5845e-13 % | 0.1 % | PASS |
| Salt conservation (winter) | salt mass balance | 1.7606e-14 % | 0.1 % | PASS |

## 2. Stefan problem (phase change)

| dx [m] | N | dt [s] | steps | Ste | $\lambda$ | front [mm] | Neumann [mm] | err final [%] | err max [%] | profile RMS [K] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0200 | 40 | 61.27 | 653 | 0.0629 | 0.17549 | 74.30 | 75.04 | 0.992 | 3.737 | 0.3523 |
| 0.0125 | 64 | 23.93 | 1672 | 0.0629 | 0.17549 | 74.99 | 75.05 | 0.072 | 3.737 | 0.4333 |
| 0.0080 | 100 | 9.80 | 4081 | 0.0629 | 0.17549 | 74.89 | 75.04 | 0.197 | 3.737 | 0.1383 |

SPH conduction operator lattice calibration: `alpha_eff/alpha = 1.02182` (scatter over the three longest modes 7.13840e-04, h = 1.3 dx). Enthalpy inversion round-trip: fresh 0.00 K, sea ice 2.84e-04 K.

## 3. Brine drainage / desalination

| $k_{ref}$ [m$^2$] | final bulk salinity [psu] | days to 90 % desalination |
|---|---|---|
| 3.00e-10 | 5.85 | 25.00 |
| 3.00e-09 | 5.40 | 5.00 |
| 3.00e-08 | 5.36 | 1.00 |

Equilibrium floor, checked per level: the minimum of `S(z) - S_eq(T(z))` over the 25 levels is 0.614 psu (mean 2.154 psu); gravity drainage never crosses the irreducible-brine floor.

Salt conservation of the drainage solve: 2.48e-16 relative. Final salinity 5.40 psu against the observed first-year band [3.0, 8.0] psu (ratio to ocean 0.169).

## 4. Buoyancy-driven convection onset

Numerical eigenvalue solve with 60 finite-difference nodes: $Ra_c = 39.4722$ vs exact $4\pi^2 = 39.4784$ (error 0.0157 %), at wavenumber $a_c = 3.167$; the whole marginal curve agrees with $(a^2+\pi^2)^2/a^2$ to 0.043 %.

## 5. Thermo-elastic stress and cracking

| quantity | reference | computed | error [%] |
|---|---|---|---|
| Timoshenko restrained surface stress [Pa] | 3.36e+06 | 3.36e+06 | 0.0000 |
| free plate, linear gradient [Pa] | 0 | 4.66e-08 | - |
| free plate, non-linear gradient [Pa] | - | 3.39e+06 | - |
| net force of the free solution [Pa] | 0 | -2.259e-10 | - |
| net moment of the free solution [Pa m] | 0 | -2.288e-10 | - |
| critical temperature drop [K] | 0.7 MPa strength | 1.042 | - |
| $K_I$ of a 10 mm edge crack [MPa m$^{1/2}$] | $K_{IC}=0.1$ | 0.800 | - |

## 6. Three-field coupled column

| case | S_bulk [psu] | S_ocean [psu] | $Q_{oc}$ [W/m$^2$] | melt [mm/d] | melt total [m] | $\sigma_{max}$ restrained [MPa] | $\sigma_{max}$ free [MPa] | salt err |
|---|---|---|---|---|---|---|---|---|
| spring_two_way | 0.35 | 32.0480 | 5.41 | 3.75 | 0.0763 | 0.77 | 3.982e-04 | 1.58e-15 |
| spring_one_way | 0.35 | 32.1559 | 9.57 | 4.33 | 0.0883 | 0.77 | 2.332e-04 | 0.00 |
| winter | 7.46 | 32.0608 | 6.05 | 0.00 | 0.0000 | 5.46 | 0.006 | 1.76e-16 |

Two-way coupling strength (spring case): melt ratio 0.864, ocean heat-flux ratio 0.565, mixed-layer salinity shift -0.1079 psu, bulk-salinity shift 0.0000 psu.

Mush-layer latent buffering: the sea-ice column needs n/a s to reach half of a cold-snap step against 67800.0 s for fresh ice (conduction time $h^2/\alpha$ = 78778.6 s).

## 7. Traps found and fixed while building this face

1. **Kelvin vs freezing-point depression.** The mushy latent term must be written in `theta = 273.15 - T`, not `T`: using `T` makes the latent heat of a 12 psu ice 273x too small (2.9 J/kg/K instead of 5e5 J/kg/K near Tf).
2. **Enthalpy is a function of (T, S).** Draining brine is an isothermal mass change; keeping `H` fixed while `S` drops adds the S-dependent latent term to the remaining ice and heats it spuriously by ~L per kg of salt removed (observed: the column jumped to +15 degC).
3. **No density in the specific-enthalpy capacity.** `dH/dt = c dT/dt`; carrying `rho` in the Picard diagonal suppresses the temperature response by 917 and the column never cools.
4. **CFL on the particle, not on the face average.** The face flux below particle i is carried by particle i, so a face-averaged velocity under-estimates the limit by up to 2x and destabilises the upwind scheme (observed: bulk salinity 5.5e4 psu in one step).
5. **A free plate with a linear gradient is stress free.** Evaluating the bending moment with a continuum `12/h^3` on a discrete grid leaves an O(dz^2/h^2) phantom stress (~66 Pa); the moment balance must use the same discrete measure in numerator and denominator.

## 8. Limitations

* the ice column does not ablate: melt is diagnosed and enters the ocean salt/heat budget, but the base does not move (no SPH moving boundary);
* the ocean is a single well-mixed layer, and `gamma_0`, `a` are declared bulk exchange coefficients (not fitted to a dataset);
* the permeability scale `k_ref` is uncertain by about a decade in the literature; the scan in section 3 quantifies its effect;
* the Greeley & Wilson (1987) brine-drainage table is not reachable from this sandbox, so that anchor is WITHHELD; the observed first-year salinity band (Cox & Weeks 1983) is used instead and labelled as a band;
* a sub-kelvin non-monotone layer can appear at the ice-ocean interface because the fixed-thickness column re-projects its enthalpy while the base drains.

