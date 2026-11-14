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
