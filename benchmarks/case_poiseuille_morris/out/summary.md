# CASE-SPH-POISEUILLE-MORRIS -- ref vs got

| H/dx | u_max [m/s] | err % | RMS % | tau_w err % | Q err % | slip % | pass |
|---|---|---|---|---|---|---|---|
| 8 | ref 1.363979e-01 / got 1.391480e-01 | +2.02 | 2.72 | 0.62 | +2.88 | 3.06 | PASS |
| 12 | ref 1.375776e-01 / got 1.388324e-01 | +0.91 | 1.44 | 0.23 | +1.34 | 1.58 | PASS |
| 16 | ref 1.379904e-01 / got 1.386999e-01 | +0.51 | 0.72 | 0.12 | +0.77 | 1.07 | PASS |

- observed order (interior RMS): 2.4250193107876648
- repro: `PYTHONPATH=src python -m tensorsph.case_poiseuille_morris --out runs/case_poiseuille_morris --resolutions 8 12 16 24`
