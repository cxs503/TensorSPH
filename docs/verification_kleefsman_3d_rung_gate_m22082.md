# TensorSPH Kleefsman 3-D Rung Gate Verification Report

## 1. Case Introduction

### Physical Background
The Kleefsman dam-break-with-obstacle case (Kleefsman et al. 2005, J. Comput. Phys. 206:363-393) is a benchmark for free-surface flows with complex geometry. The experiment features a 3.22 m long, 1.00 m wide tank with a 0.403 m wide, 0.161 m high box placed at x = 2.476 m. The water column is 1.228 m long and 0.550 m high. The box blocks only 40% of the tank width, allowing 60% of the water to bypass it.

### Geometry and Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| Tank Length | 3.22 | m |
| Tank Width | 1.00 | m |
| Tank Height | 1.00 | m |
| Column Length | 1.228 | m |
| Column Height | 0.550 | m |
| Obstacle X | 2.476 | m |
| Obstacle Length | 0.161 | m |
| Obstacle Height | 0.161 | m |
| Obstacle Width | 0.403 | m |
| Reference Peak Height | 0.370 | m |
| Reference Peak Time | 1.7027 | s |
| Reference Arrival Time | 0.3591 | s |

## 2. Methodology

### Approach
The 3-D Kleefsman case is implemented in `src/tensorsph/kleefsman_obstacle_3d.py`. The "rung gate" verification ensures that:
1. Grid resolutions are admissible (obstacle must be resolved by at least 3 cells).
2. Run records are comparable (protocol fingerprints must match).
3. Incomplete or aborted runs are excluded from the ladder fit.

### Toolchain
- **Solver**: TensorSPH (PyTorch-based SPH)
- **Ladder**: `benchmarks/kleefsman_obstacle_3d/ladder.py`
- **Tests**: `tests/test_kleefsman_obstacle_3d_ladder.py`, `tests/test_kleefsman_obstacle_3d_rung_gate.py`
- **Results**: `benchmarks/kleefsman_obstacle_3d/results/rung_gate.json`

### Key Settings
- **Sound Speed**: 20.0 m/s
- **Wall Layers**: 3
- **Gauge Aperture**: 0.08 m
- **Station Offset**: 0.05 m
- **Arrival Threshold**: 0.02 m

## 3. Calculation Results

### Rung Admissibility
The following rungs were evaluated for admissibility:

| dx (m) | Valid | Reason |
|--------|-------|--------|
| 0.08 | No | Obstacle length 0.161 m is under-resolved at dx=0.08 (requires >= 0.0537 m) |
| 0.05 | Yes | Admissible (0.161 / 3 = 0.0537 m) |
| 0.04 | Yes | Admissible |
| 0.03 | Yes | Admissible |

### Protocol Comparability
The ladder's `check_protocol` function ensures that runs with different parameters (e.g., `t_end`, `sound_speed`) are not mixed. Empty fingerprints are also rejected to prevent vacuous passes.

### Ladder Fit
A clean ladder with 3 valid rungs (dx=0.05, 0.04, 0.03) can be fitted using `fit_power_law`, which solves for the continuum limit `f_inf` and observed order `p`.

## 4. Figures

**Reason for no figures**: This verification focuses on the logical correctness of the rung admission rules and protocol guards, not on simulation results. No field plots or comparison curves are generated.

## 5. Notes

### Pitfalls
- **Coarse Grid Rejection**: The dx=0.08 rung was rejected by `validate_case` because the 0.161 m box cannot be resolved by 3 cells of 0.08 m. This prevents fitting a ladder with an invalid coarsest level.
- **Vacuous Passes**: The protocol guard was updated to reject runs with empty fingerprints, preventing a guard that compared nothing from passing.
- **Incomplete Runs**: Runs that do not reach `t_end` are excluded from the ladder to avoid censored results.

### Limitations
- The ladder fit assumes a power-law convergence `f(dx) = f_inf + C dx^p`, which may not hold for all regimes.
- The 3-D case is computationally expensive, limiting the number of rungs that can be practically computed.

### Scope
This verification applies to the 3-D Kleefsman obstacle case in TensorSPH. It does not cover the 2-D case or other benchmarks.

## 6. Error Assessment

### Key Quantities vs Baseline
| Metric | Reference | Got | Rel Error | Pass (<3%) |
|--------|-----------|-----|-----------|------------|
| Peak Height | 0.370 m | N/A | N/A | N/A |
| Arrival Time | 0.3591 s | N/A | N/A | N/A |

**Note**: No simulation results are available for this verification. The "rung gate" verification ensures the *correctness* of the ladder logic, not the *accuracy* of the simulation. The error assessment is not applicable here.

### Conclusion
The rung gate verification passes all tests. The ladder admission rules correctly reject inadmissible rungs and incomparable runs. The protocol guard prevents vacuous passes. The ladder fit is only performed on clean, valid rungs.
