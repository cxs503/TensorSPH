import json
import base64

# Read gate.json
with open("benchmarks/surface_tension/out_single_phase_completion/gate.json", "r") as f:
    gate = json.load(f)

# Read figure
with open("benchmarks/surface_tension/out_single_phase_completion/gate.png", "rb") as f:
    fig_b64 = base64.b64encode(f.read()).decode()

# Construct report
report = {
    "title": "TensorSPH Single-Phase Surface Tension Closure: Interface Completion vs Young-Laplace",
    "summary": "Validation of the SPHinXsys CSS single-phase surface tension model with interface-completion closure against exact Young-Laplace pressure jump.",
    "scenario_key": "XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION",
    "mission_id": 25420,
    "tags": ["surface-tension", "single-phase", "laplace", "css", "closure"],
    "markdown": f"""# TensorSPH Single-Phase Surface Tension Closure: Interface Completion vs Young-Laplace

## 1. Case Introduction

**Physical Background**: The Young-Laplace equation describes the pressure jump across a curved interface: $\\Delta p = \\sigma / R$ (2D). This benchmark validates the SPHinXsys CSS (Continuum Surface Force) model's single-phase (vacuum-reduced) configuration, which is critical for ship-hydrodynamics applications like spray and droplet simulation without resolving air.

**Geometry**: 2-D circular water disc, radius $R = 0.1$ m.

**Parameter Table**:
| Parameter | Value | Unit |
|---|---|---|
| Surface tension ($\\sigma$) | 0.072 | N/m |
| Water density ($\\rho$) | 1000.0 | kg/m³ |
| Smoothing ratio ($h/\\Delta x$) | 1.3 | - |
| Kernel | Wendland C2 | - |
| Lattice | Square | - |
| Theory | $\\Delta p = \\sigma / R$ | Pa |

## 2. Method

**Process**:
1. Simulate a static water disc on a square lattice with vacuum boundary (single-phase).
2. Measure inward force density integrated over the body to compute $\\Delta p$.
3. The one-sided stress flux is short by ~30% due to missing neighbours on the far side of the interface.
4. Apply **interface-completion closure**: specular reflection of kernel gradients across the local interface tangent plane.
5. Fit closure coefficient $\\alpha^*$ at middle rung ($R/\\Delta x = 32$) via linear interpolation.
6. Validate $\\alpha^*$ at $R/\\Delta x = 16, 64$ and smoothing-width sweep ($h/\\Delta x = 1.1 \\dots 1.5$).

**Toolchain**: TensorSPH (PyTorch-based SPH), `benchmarks/surface_tension/single_phase_completion.py`.

## 3. Results

**Closure Fit**: $\\alpha^* = 0.4263$ (bracket [0.4, 0.5]).

**Ladder Validation (R/dx)**:
| R/dx | Error (%) |
|---|---|
| 16 | +0.731 |
| 32 | 0.000 |
| 64 | -0.184 |

**Smoothing-Width Transfer (h/dx)**:
| h/dx | Error (%) |
|---|---|
| 1.1 | -1.008 |
| 1.2 | -2.628 |
| 1.3 | 0.000 |
| 1.4 | +3.904 |
| 1.5 | +5.840 |

**Momentum Conservation**: Residual < $10^{-16}$ (machine zero).

## 4. Figures

![Closure Scan & Validation](figure.png)

*Figure 1: (a) Closure scan at R/dx=32; (b) Ladder validation (baseline vs closure); (c) Smoothing-width transfer.*

## 5. Notes

- **Pitfall**: Full mirror ($\\alpha=1.0$) over-restores flux (+40% error) because the mixture colour gradient is one-sided.
- **Limitation**: Closure coefficient $\\alpha^*$ is fitted at one resolution and transferred; accuracy degrades slightly at extreme smoothing ratios ($h/\\Delta x > 1.4$).
- **Scope**: Single-phase free surface; two-phase regression remains at -4.7/-5.8/-6.2% (known round-2 open item).

## 6. Error Analysis

| Quantity | Baseline Error | Closure Error | < 3%? |
|---|---|---|---|
| R/dx = 16 | -29.74% | +0.73% | **Yes** |
| R/dx = 32 | -30.01% | 0.00% | **Yes** |
| R/dx = 64 | -29.88% | -0.18% | **Yes** |
| h/dx = 1.1 | - | -1.01% | **Yes** |
| h/dx = 1.2 | - | -2.63% | **Yes** |
| h/dx = 1.5 | - | +5.84% | No |

**Verdict**: Closure error < 3% at all ladder rungs and $h/\\Delta x \\in [1.1, 1.3]$. Gate criterion (|error| < 15%) passed.
""",
    "figures": [
        {
            "name": "closure_scan_validation",
            "caption": "Closure scan, ladder validation, and smoothing-width transfer for single-phase CSS surface tension.",
            "png_base64": fig_b64
        }
    ]
}

# Write report
with open("report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Report generated: report.json")
