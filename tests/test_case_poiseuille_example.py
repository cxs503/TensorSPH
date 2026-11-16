#!/usr/bin/env python3
"""Tests for the enhanced Poiseuille channel example.

Verifies that the example produces correct artifacts and passes all gates.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


# --------------------------------------------------------------------------- #
# Analytic reference
# --------------------------------------------------------------------------- #

HEIGHT = 0.1
BODY_FORCE = 9.81
VISCOSITY = 0.01
RHO0 = 1000.0


def analytic_u_max():
    return BODY_FORCE * HEIGHT**2 / (8.0 * VISCOSITY)


def analytic_wall_shear():
    return RHO0 * BODY_FORCE * HEIGHT / 2.0


# --------------------------------------------------------------------------- #
# Artifact tests
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def artifact():
    """Load the verification artifact."""
    path = REPO / "artifacts" / "poiseuille_verification" / "result.json"
    if not path.exists():
        pytest.skip("Artifact not yet generated; run examples/poiseuille_channel.py --json first")
    with open(path) as f:
        return json.load(f)


def test_artifact_exists():
    """The JSON artifact must exist."""
    path = REPO / "artifacts" / "poiseuille_verification" / "result.json"
    assert path.exists(), "Run examples/poiseuille_channel.py --json to generate"


def test_artifact_structure(artifact):
    """The artifact must have the required top-level keys."""
    for key in ("case", "title", "solver", "physics", "reference",
                "acceptance_threshold_pct", "results", "overall_verdict"):
        assert key in artifact, f"Missing key: {key}"


def test_physics_constants(artifact):
    """Physics constants must match the analytic reference."""
    p = artifact["physics"]
    assert p["height_m"] == pytest.approx(HEIGHT)
    assert p["viscosity_m2s"] == pytest.approx(VISCOSITY)
    assert p["density_kgm3"] == pytest.approx(RHO0)
    assert p["body_force_m2s"] == pytest.approx(BODY_FORCE)
    assert p["analytic_u_max"] == pytest.approx(analytic_u_max())
    assert p["analytic_wall_shear_pa"] == pytest.approx(analytic_wall_shear())


def test_all_resolutions_pass(artifact):
    """Every resolution must pass all gates."""
    threshold = artifact["acceptance_threshold_pct"]
    for r in artifact["results"]:
        assert r["centreline_error_pct"] <= threshold, \
            f"H/dx={r['resolution']}: centreline error {r['centreline_error_pct']}% > {threshold}%"
        assert r["profile_rms_interior_pct"] <= threshold, \
            f"H/dx={r['resolution']}: profile RMS {r['profile_rms_interior_pct']}% > {threshold}%"
        assert abs(r["wall_shear_error_pct"]) <= threshold, \
            f"H/dx={r['resolution']}: wall shear error {r['wall_shear_error_pct']}% > {threshold}%"
        assert r["verdict"] == "PASS"


def test_overall_verdict(artifact):
    """Overall verdict must be PASS."""
    assert artifact["overall_verdict"] == "PASS"


def test_centreline_error_within_5pct(artifact):
    """Centreline error must be < 5% at all resolutions."""
    for r in artifact["results"]:
        assert r["centreline_error_pct"] < 5.0, \
            f"H/dx={r['resolution']}: {r['centreline_error_pct']}%"


def test_profile_rms_within_5pct(artifact):
    """Profile RMS must be < 5% at all resolutions."""
    for r in artifact["results"]:
        assert r["profile_rms_interior_pct"] < 5.0, \
            f"H/dx={r['resolution']}: {r['profile_rms_interior_pct']}%"


def test_wall_shear_within_5pct(artifact):
    """Wall shear error must be < 5% at all resolutions."""
    for r in artifact["results"]:
        assert abs(r["wall_shear_error_pct"]) < 5.0, \
            f"H/dx={r['resolution']}: {r['wall_shear_error_pct']}%"


def test_max_mach_below_01(artifact):
    """Max Mach number must be below 0.1 (WCSPH weakly-compressible regime)."""
    for r in artifact["results"]:
        assert r["max_mach"] < 0.1, \
            f"H/dx={r['resolution']}: Mach {r['max_mach']} >= 0.1"


def test_resolution_ladder_has_3_levels(artifact):
    """The resolution ladder must have at least 3 levels."""
    assert len(artifact["results"]) >= 3


def test_convergence_trend(artifact):
    """Profile RMS should generally decrease with resolution."""
    rms_values = [r["profile_rms_interior_pct"] for r in artifact["results"]]
    # RMS at finest resolution should be lower than at coarsest
    assert rms_values[-1] < rms_values[0], \
        f"RMS not converging: {rms_values[0]} -> {rms_values[-1]}"


def test_figures_exist():
    """Both figures must exist."""
    out_dir = REPO / "artifacts" / "poiseuille_verification"
    assert (out_dir / "poiseuille_profile.png").exists()
    assert (out_dir / "poiseuille_convergence.png").exists()


# --------------------------------------------------------------------------- #
# Live run test (lightweight)
# --------------------------------------------------------------------------- #

def test_example_runs_hdx8():
    """H/dx=8 must complete and pass within a reasonable time."""
    from tensorsph.droplet_impact import Config, resolve_device
    from tensorsph.wall_mirror import plane_walls
    import torch

    # Import the example's run function
    sys.path.insert(0, str(REPO / "examples"))
    from poiseuille_channel import run_single_resolution, analytic_u_max

    device = resolve_device("cpu")
    torch.set_num_threads(2)

    r = run_single_resolution(8, device, t_end=0.5, t_average=0.1)

    assert r["centreline_error_pct"] < 5.0
    assert r["profile_rms_interior_pct"] < 5.0
    assert abs(r["wall_shear_error_pct"]) < 5.0
    assert r["max_mach"] < 0.1
