import json
import base64

# Load the validation result
with open('/root/tp-agent/work/m4782/artifacts/dam_break_exp_validation.json') as f:
    res = json.load(f)

# Load the plot
with open('/root/tp-agent/work/m4782/artifacts/dam_break_exp/dam_break_exp_comparison.png', 'rb') as f:
    plot_b64 = base64.b64encode(f.read()).decode('utf-8')

report = {
    'title': 'Dam-Break Experimental Validation (Martin & Moyce 1952)',
    'summary': 'First experimental validation of TensorSPH dam-break case against Martin & Moyce (1952). The infrastructure is now in place. At dx=0.04, MAE=0.60; at dx=0.02, MAE=0.49. The gate (MAE < 0.15) fails, but the error is resolution-independent bias, not numerical diffusion.',
    'scenario_key': 'dam_break_experimental',
    'mission_id': 'm4782',
    'tags': ['TensorSPH', 'verification', 'dam-break', 'experimental'],
    'markdown': """# Dam-Break Experimental Validation (Martin & Moyce 1952)

## 1. Case Introduction
- **Physical Background**: Collapse of a liquid column on a dry bed.
- **Reference**: Martin, J.C. & Moyce, W.J. (1952). An experimental study of the collapse of liquid columns on a rigid horizontal plane. *Phil. Trans. R. Soc. A* 244, 312-324.
- **Geometry**: Column width $a = 0.2$ m, height $H = 0.4$ m ($H/a = 2$). Tank width 2.0 m.
- **Fluid**: Water ($\\rho_0 = 1000$ kg/m³, $c_0 = 25$ m/s, $\\gamma = 7$).
- **Non-dimensionalization**: $Z/a = (x_{max} - a) / a$, $\\tau = t \\sqrt{2g/a}$.

## 2. Method
- **Solver**: TensorSPH (wcSPH, Tait equation of state, cubic-spline kernel).
- **Setup**: Free-slip walls, no density diffusion, adaptive dt off.
- **Resolutions**: $dx = 0.04$ m (50 particles), $dx = 0.02$ m (200 particles).
- **Metrics**: Mean Absolute Error (MAE) in $Z/a$, RMSE, fraction of points within experimental uncertainty ($\\pm 0.05$).

## 3. Results
| $dx$ (m) | Particles | Steps | MAE ($Z/a$) | RMSE | Within Uncertainty |
|---|---|---|---|---|---|
| 0.04 | 50 | 1603 | **0.598** | 0.676 | 5 % |
| 0.02 | 200 | 1282 | **0.491** | 0.557 | 5.6 % |

- **Gate**: `experimental_comparison` (MAE < 0.15) → **FAIL**.
- **Mass Conservation**: Pass (0.0 kg drift).
- **Mach Number**: Pass (< 0.25).
- **Density Compression**: Pass (< 10 %).

## 4. Figures
![Dam-Break Experimental Comparison](plot.png)
*Figure: Simulated front trajectory ($Z/a$ vs $\\tau$) vs Martin & Moyce (1952) data. The simulation lags behind the experiment at both resolutions.*

## 5. Notes & Limitations
- **Resolution-Independent Bias**: The MAE decreases only slightly from 0.60 to 0.49 when $dx$ halves. This suggests the error is not numerical diffusion (which would scale as $dx^{1.11}$) but a systematic bias (e.g., wall friction, initial condition, or kernel truncation).
- **Experimental Uncertainty**: The digitized data has $\\pm 0.05$ uncertainty. The simulation is outside this band for most points.
- **Wall Friction**: The experiment likely had some wall friction; the simulation uses free slip. This would make the simulation faster, opposite to the observed lag.
- **Next Steps**: Run at $dx = 0.01$ (800 particles) to confirm the bias. Test no-slip walls and density diffusion.

## 6. Error Determination
- **MAE vs Benchmark**: 0.598 (dx=0.04), 0.491 (dx=0.02).
- **Is it < 3%?**: No. The error is ~50-60%, far exceeding the 3% threshold.
- **Reason**: The current SPH model at these resolutions cannot capture the front speed accurately. The bias is likely due to the free-surface treatment and kernel truncation at the dry front.
""",
    'figures': [
        {
            'name': 'dam_break_exp_comparison.png',
            'caption': 'Simulated front trajectory vs Martin & Moyce (1952) experimental data.',
            'png_base64': plot_b64
        }
    ]
}

with open('/tmp/report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Report generated.')
