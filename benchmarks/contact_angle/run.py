"""Analytic benchmark for the wall contact-angle boundary condition.

Young's relation ``n . w = cos(theta)`` is an exact statement, so the only error
budget of a correct implementation is floating-point round-off.  This script
measures it over a grid of contact angles and wall orientations, writes the
numbers to JSON, and renders the figure the report uses.  It runs in about a
second: no time integration, no resolution study -- the quantity is geometric.

Usage::

    PYTHONPATH=src python benchmarks/contact_angle/run.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.surface_tension import apply_contact_angle_boundary
from tensorsph.wetting import measure_contact_angle, wall_normals

DX = 0.01
H = 1.3 * DX
ANGLES_DEG = [10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 160.0, 170.0]
ROTATIONS_DEG = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 170.0]


def rotated(angle_deg):
    a = math.radians(angle_deg)
    return torch.tensor([[math.cos(a), -math.sin(a)],
                         [math.sin(a), math.cos(a)]], dtype=torch.float64)


def scenario(theta_deg, rotation_deg, incoming=(0.8, 0.6)):
    """One fluid particle half a smoothing length above a planar wall lattice."""
    rot = rotated(rotation_deg)
    x = torch.arange(-0.1, 0.1 + 1e-12, DX, dtype=torch.float64)
    plane = torch.stack((x, torch.zeros_like(x)), dim=1) @ rot.T
    pos = (rot @ torch.tensor([0.0, 0.5 * H], dtype=torch.float64)).reshape(1, 2)
    w = (rot @ torch.tensor([0.0, 1.0], dtype=torch.float64))
    grad_c = torch.tensor([incoming], dtype=torch.float64) * 2.5
    out = apply_contact_angle_boundary(grad_c, pos, plane, H,
                                       math.radians(theta_deg))
    measured = float(measure_contact_angle(out, w.reshape(1, 2)))
    magnitude_error = abs(float(torch.linalg.vector_norm(out))
                          - float(torch.linalg.vector_norm(grad_c)))
    return {
        "theta_deg": theta_deg,
        "wall_rotation_deg": rotation_deg,
        "incoming_gradient": list(incoming),
        "theta_measured_deg": math.degrees(measured),
        "abs_error_deg": abs(math.degrees(measured) - theta_deg),
        "magnitude_abs_error": magnitude_error,
    }


def main():
    out = Path(__file__).resolve().parent
    rows = [scenario(theta, rot) for theta in ANGLES_DEG for rot in ROTATIONS_DEG]
    worst = max(rows, key=lambda r: r["abs_error_deg"])
    worst_mag = max(rows, key=lambda r: r["magnitude_abs_error"])

    # A second, independent check: the wall normal itself must come out of the
    # particle geometry, not from an assumed +y axis.
    normal_rows = []
    for rot in ROTATIONS_DEG:
        r = rotated(rot)
        x = torch.arange(-0.1, 0.1 + 1e-12, DX, dtype=torch.float64)
        plane = torch.stack((x, torch.zeros_like(x)), dim=1) @ r.T
        pos = (r @ torch.tensor([0.0, 0.5 * H], dtype=torch.float64)).reshape(1, 2)
        normal, distance, near = wall_normals(pos, plane, H)
        expected = r @ torch.tensor([0.0, 1.0], dtype=torch.float64)
        normal_rows.append({
            "wall_rotation_deg": rot,
            "normal_recovered": [float(v) for v in normal[0]],
            "max_abs_component_error": float(
                torch.max(torch.abs(normal[0] - expected))),
            "distance_over_h": float(distance[0]) / H,
            "detected": bool(near.all()),
        })

    report = {
        "what": "Young's relation n.w = cos(theta) for the wall contact-angle "
                "boundary condition of the CSF model",
        "reference": {
            "relation": "n . w = cos(theta), theta measured inside the liquid "
                        "between the interface and the wall",
            "sources": ["Young (1805)", "Brackbill, Kothe & Zemach (1992)",
                        "Tartakovsky & Meakin (2005)"],
            "close_form": True,
        },
        "setup": {
            "dx_m": DX,
            "h_m": H,
            "particle_distance_to_wall_over_h": 0.5,
            "wall_lattice": "planar, spacing dx, half-length 0.1 m",
            "dtype": "float64",
            "n_cases": len(rows),
        },
        "cases": rows,
        "worst_case": worst,
        "magnitude": {
            "worst_abs_error": worst_mag["magnitude_abs_error"],
            "tolerance": 1e-12,
            "within_1e_12": worst_mag["magnitude_abs_error"] < 1e-12,
        },
        "verdict": {
            "max_abs_angle_error_deg": worst["abs_error_deg"],
            "tolerance_deg": 1e-6,
            "within_3pct_of_all_angles": bool(
                all(r["abs_error_deg"] / r["theta_deg"] < 0.03 for r in rows)),
            "passes": worst["abs_error_deg"] < 1e-6,
        },
        "wall_normal_recovery": normal_rows,
        "not_verified_here": [
            "the equilibrium contact angle of a sessile drop under the force "
            "field: the shipped CSF curvature/colour field does not reproduce "
            "the Laplace law (see docs/verification_surface_tension_laplace.md), "
            "so an end-to-end wetting simulation cannot yet be scored",
            "dynamic contact angle and contact-line pinning: not implemented",
        ],
    }
    (out / "validation_report.json").write_text(json.dumps(report, indent=2))
    make_figure(rows, out / "contact_angle.png")
    print(json.dumps(report["verdict"], indent=2))
    print("worst magnitude error:", report["magnitude"])


def make_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for rot in ROTATIONS_DEG:
        sel = [r for r in rows if r["wall_rotation_deg"] == rot]
        ax.semilogy([r["theta_deg"] for r in sel],
                    [max(r["abs_error_deg"], 1e-16) for r in sel],
                    marker="o", ms=3, lw=1,
                    label=f"wall rotated {rot:g} deg")
    ax.axhline(1e-6, color="k", ls="--", lw=1, label="1e-6 deg tolerance")
    ax.set_xlabel("prescribed contact angle theta (deg)")
    ax.set_ylabel("|theta_measured - theta| (deg)")
    ax.set_title("Young's relation reproduced to round-off")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    theta = torch.linspace(0.0, math.pi, 181, dtype=torch.float64)
    rot = rotated(0.0)
    x = torch.arange(-0.1, 0.1 + 1e-12, DX, dtype=torch.float64)
    plane = torch.stack((x, torch.zeros_like(x)), dim=1)
    pos = torch.tensor([[0.0, 0.5 * H]], dtype=torch.float64)
    w = torch.tensor([0.0, 1.0], dtype=torch.float64)
    grad_c = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    out = torch.stack([apply_contact_angle_boundary(grad_c, pos, plane, H,
                                                    float(t))[0]
                       for t in theta])
    outward = -out / torch.linalg.vector_norm(out, dim=1, keepdim=True)
    ax.plot(torch.rad2deg(theta), outward[:, 0], label="n_x (tangential)")
    ax.plot(torch.rad2deg(theta), outward[:, 1], label="n_y (wall normal)")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("prescribed contact angle theta (deg)")
    ax.set_ylabel("component of the outward interface normal")
    ax.set_title("constructed normal: n_y = cos(theta)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()