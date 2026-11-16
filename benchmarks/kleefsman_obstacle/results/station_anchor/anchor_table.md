| trace | arrival [s] | x implied by Ritter [m] | required gamma @x=2.426 | required gamma @x=2.228 |
|---|---|---|---|---|
| experiment_interp | 0.3481 | 2.3826 | 1.038 | 0.866 |
| experiment_sample | 0.3591 | 2.4191 | 1.006 | 0.840 |
| comflow_vof | 0.3707 | 2.4575 | 0.974 | 0.813 |

| station | x [m] | Ritter arrival [s] |
|---|---|---|
| primary (0.05 m up of box face) | 2.426 | 0.3612 |
| upstream (0.248 m up of box face) | 2.228 | 0.3015 |

| grid | front arrival @2.228 [s] | front arrival @2.426 [s] | gamma near | gamma far | d gamma / d d [1/m] |
|---|---|---|---|---|---|
| dx0.02 | 0.3895 | 0.4397 | 0.774 | 0.821 | 0.239 |
| dx0.04 | 0.4147 | 0.4667 | 0.727 | 0.774 | 0.237 |

| trace | d gamma / d d required [1/m] | solver [1/m] | ratio |
|---|---|---|---|
| experiment_interp | 0.866 | 0.238 | 3.6x |
| experiment_sample | 0.840 | 0.238 | 3.5x |
| comflow_vof | 0.813 | 0.238 | 3.4x |
