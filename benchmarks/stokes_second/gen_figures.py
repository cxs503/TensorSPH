#!/usr/bin/env python3
"""Generate figures for the Stokes second problem benchmark report."""
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'out'
ladder = json.loads((OUT / 'results_ladder.json').read_text())
finest = max(ladder['runs'], key=lambda r: r['resolution_delta_over_dx'])
dod = finest['resolution_delta_over_dx']

# Figure 1: Amplitude decay comparison
fig, ax = plt.subplots(figsize=(8, 5))
y_over_delta = np.array([r['y_over_delta'] for r in finest['rows']])
amp_sph = np.array([r['amplitude_over_U0'] for r in finest['rows']])
amp_half = np.array([r['amplitude_half_space'] / finest['u0'] for r in finest['rows']])
amp_fd = np.array([r['amplitude_finite_depth'] / finest['u0'] for r in finest['rows']])

mask = y_over_delta <= 4.0
ax.plot(y_over_delta[mask], amp_half[mask], 'k--', linewidth=2, label=r'Half-space exact: $e^{-y/\delta}$')
ax.plot(y_over_delta[mask], amp_fd[mask], 'b-.', linewidth=1.5, label='Finite-depth exact')
ax.plot(y_over_delta[mask], amp_sph[mask], 'ro', markersize=4, label=f'SPH (d/dx={dod})')

ax.set_xlabel(r'$y/\delta$', fontsize=12)
ax.set_ylabel(r'$A/U_0$', fontsize=12)
ax.set_title(f'Stokes Second Problem: Amplitude Decay (d/dx={dod})', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 4)
plt.tight_layout()
fig.savefig(str(OUT / 'amplitude_decay.png'), dpi=150)
plt.close(fig)
print("Wrote amplitude_decay.png")

# Figure 2: Convergence across resolutions
fig, ax = plt.subplots(figsize=(7, 4.5))
resolutions = [r['resolution_delta_over_dx'] for r in ladder['runs']]
l2_half = [r['metrics']['amplitude_l2_half_space_pct'] for r in ladder['runs']]
l2_fd = [r['metrics']['amplitude_l2_finite_depth_pct'] for r in ladder['runs']]

ax.semilogy(resolutions, l2_half, 'ro-', linewidth=2, markersize=8, label='vs Half-space')
ax.semilogy(resolutions, l2_fd, 'bs-', linewidth=2, markersize=8, label='vs Finite-depth')
ax.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, label='G1/G4 gate (5%)')
ax.set_xlabel(r'$\delta/dx$', fontsize=12)
ax.set_ylabel('L2 error (%)', fontsize=12)
ax.set_title('Convergence: Amplitude L2 Error vs Resolution', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
ax.set_xticks(resolutions)
plt.tight_layout()
fig.savefig(str(OUT / 'convergence.png'), dpi=150)
plt.close(fig)
print("Wrote convergence.png")

# Figure 3: Space-time field (heatmap)
field = finest.get('field', {})
if field and 'u' in field and 't' in field and 'y' in field:
    fig, ax = plt.subplots(figsize=(10, 5))
    # u is stored as (n_y, n_t)
    u_data = np.array(field['u'], dtype=float)
    t_vals = np.array(field['t']) / finest['period']  # in periods
    y_vals = np.array(field['y']) / finest['delta']    # in delta units
    
    # Downsample for readability
    t_step = max(1, len(t_vals) // 80)
    y_step = max(1, len(y_vals) // 40)
    u_ds = u_data[::y_step, ::t_step]
    t_ds = t_vals[::t_step]
    y_ds = y_vals[::y_step]
    
    T, Y = np.meshgrid(t_ds, y_ds)
    im = ax.pcolormesh(T, Y, u_ds, shading='auto', cmap='RdBu_r', vmin=-0.012, vmax=0.012)
    plt.colorbar(im, ax=ax, label=r'$u_x$ [m/s]')
    ax.set_xlabel('Time (periods)', fontsize=12)
    ax.set_ylabel(r'$y/\delta$', fontsize=12)
    ax.set_title(f'Space-time velocity field (d/dx={dod})', fontsize=13)
    plt.tight_layout()
    fig.savefig(str(OUT / 'field_snapshot.png'), dpi=150)
    plt.close(fig)
    print("Wrote field_snapshot.png")
else:
    print("No field data available for heatmap")

# Figure 4: Error per row
fig, ax = plt.subplots(figsize=(8, 4.5))
rel_err_half = [100 * r['rel_err_half_space'] for r in finest['rows'] if r['y_over_delta'] <= 4.0]
y_vals = [r['y_over_delta'] for r in finest['rows'] if r['y_over_delta'] <= 4.0]
ax.bar(y_vals, rel_err_half, width=0.15, color='steelblue', alpha=0.7, edgecolor='navy')
ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='G1 gate (5%)')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel(r'$y/\delta$', fontsize=12)
ax.set_ylabel('Relative error (%)', fontsize=12)
ax.set_title(f'Row-wise Amplitude Error (d/dx={dod})', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(str(OUT / 'error_per_row.png'), dpi=150)
plt.close(fig)
print("Wrote error_per_row.png")

print("All figures generated.")
