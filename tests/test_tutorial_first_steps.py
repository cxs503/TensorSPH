"""Pin the tutorial's numbers.

``docs/tutorials/01-first-steps.md`` is a runnable walk-through: every number in
it is printed by ``examples/tutorial/first_steps.py``. These tests keep that
honest in three ways:

1. they re-run each tutorial step and check the physics it claims;
2. they run the script as a user would (subprocess) and require the printed
   ``MEASURED`` lines and the JSON dump to agree;
3. they re-parse the ``MEASURED`` block quoted in the tutorial document and
   compare it with a fresh run, so a hand-edited number in the prose fails the
   suite instead of drifting.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import tensorsph as sph

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "tutorial" / "first_steps.py"
DOCUMENT = ROOT / "docs" / "tutorials" / "01-first-steps.md"
MEASURED = re.compile(r"^MEASURED ([A-Za-z0-9_]+\.[A-Za-z0-9_]+)=(.*)$")


def load_tutorial():
    """Import the tutorial script as a module (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("tutorial_first_steps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_suite_imports_this_worktree():
    """A sibling worktree's editable install must not shadow the tree under test."""
    assert Path(sph.__file__).resolve().parents[1] == (ROOT / "src").resolve()


@pytest.fixture(scope="module")
def tutorial():
    return load_tutorial()


@pytest.fixture(scope="module")
def results(tutorial):
    """One run of all six steps, shared by the tests in this module."""
    return tutorial.run_all()


# --------------------------------------------------------------------------- #
# Step 1
# --------------------------------------------------------------------------- #

def test_kernel_is_normalised_in_two_dimensions(results):
    measured = results["step1_kernel_normalisation"]
    assert measured["kernel_integral"] == pytest.approx(1.0, abs=1e-4)
    assert measured["kernel_at_2h"] == 0.0
    # 10/(7 pi h^2) at h = 0.026 m, i.e. what the quadrature integrates over.
    expected_peak = 10.0 / (7.0 * math.pi * measured["h"] ** 2)
    assert measured["kernel_peak"] == pytest.approx(expected_peak, rel=1e-12)


# --------------------------------------------------------------------------- #
# Step 2
# --------------------------------------------------------------------------- #

def test_lattice_bias_is_uniform_and_calibration_removes_it(results):
    measured = results["step2_density_calibration"]
    # Uniform: the bias carries no scatter, it is a lattice quadrature error.
    assert measured["lattice_bias_spread"] < 1e-12
    assert abs(measured["lattice_bias_square"]) < 1e-3
    # Calibration leaves the interior at rho0 to machine precision.
    assert measured["calibrated_error"] < 1e-12
    # ... but only for the packing it was measured on.
    assert measured["wrong_packing_error"] > 1e-3
    # Hex bias is the one the initial-density calibration docs quote.
    assert measured["lattice_bias_hex"] == pytest.approx(5.8835e-3, rel=1e-3)


def test_density_calibration_matches_the_library_bias(results):
    measured = results["step2_density_calibration"]
    dx, h_factor = 0.02, 1.3
    library_bias = sph.lattice_bias(dx, h_factor * dx)
    assert measured["lattice_bias_square"] == pytest.approx(library_bias, abs=1e-12)
    assert measured["lattice_kernel_sum"] == pytest.approx(
        sph.lattice_kernel_sum(dx, h_factor * dx), rel=1e-12)


# --------------------------------------------------------------------------- #
# Step 3
# --------------------------------------------------------------------------- #

def test_equation_of_state_carries_the_configured_sound_speed(results):
    measured = results["step3_equation_of_state"]
    assert measured["p_at_rho0"] == 0.0
    assert abs(measured["sound_speed_error"]) < 1e-4
    # Away from rho0 a central difference is valid: c = c0 (rho/rho0)^3.
    assert measured["compressed_sound_speed"] == pytest.approx(
        measured["compressed_sound_speed_expected"], rel=1e-4)
    assert measured["compressed_sound_speed"] > 20.0


def test_bias_becomes_a_startup_pressure_and_the_clamp_bites(results):
    measured = results["step3_equation_of_state"]
    # Negative bias clamps to zero pressure: compressive-only EOS.
    assert measured["bias_pressure_square"] == 0.0
    assert measured["tensile_pressure"] == 0.0
    # The hex bias survives the clamp and is thousands of pascals.
    assert 2.0e3 < measured["bias_pressure_hex"] < 2.8e3


# --------------------------------------------------------------------------- #
# Step 4
# --------------------------------------------------------------------------- #

def test_cell_linked_search_is_identical_to_brute_force(results):
    measured = results["step4_neighbour_search"]
    assert measured["pairs_identical"] == 1.0
    assert measured["pair_count_cell_linked"] == measured["pair_count_brute_force"]
    assert measured["pair_count_brute_force"] > 1000
    # Reusing a Verlet list is exact here, not merely close.
    assert measured["skin_reuse_density_error"] < 1e-15


# --------------------------------------------------------------------------- #
# Step 5
# --------------------------------------------------------------------------- #

def test_the_acoustic_limit_binds_and_is_reported(results):
    measured = results["step5_time_step_policy"]
    cfg = sph.Config(dx=0.02, h_factor=1.3)
    limits = [sph.acoustic_dt_limit(cfg), sph.viscous_dt_limit(cfg),
              sph.gravity_dt_limit(cfg), sph.diffusion_dt_limit(cfg),
              sph.capillary_dt_limit(cfg)]
    assert measured["stable_dt"] == pytest.approx(min(limits), rel=1e-12)
    assert measured["stable_dt"] == pytest.approx(measured["acoustic_dt_limit"], rel=1e-12)
    assert measured["moving_to_stable_ratio"] == pytest.approx(20.0 / 30.0, rel=1e-12)
    assert "acoustic Courant" in measured["refused_message"]
    assert "use dt <=" in measured["refused_message"]


def test_configuration_that_breaks_the_courant_limit_raises():
    cfg = sph.Config(dx=0.02, dt=1.0e-3)
    with pytest.raises(ValueError, match="acoustic Courant"):
        sph.validate_config(cfg)


# --------------------------------------------------------------------------- #
# Step 6
# --------------------------------------------------------------------------- #

def test_short_run_records_its_own_evidence(results):
    measured = results["step6_run_case"]
    assert measured["numerical_verification"] == "pass"
    assert measured["particle_count"] == 152.0
    assert measured["wall_particle_count"] == 159.0
    # Mass is a constant of this motion; summation density is not in the list.
    assert measured["mass_drift_relative"] < 1e-12
    assert measured["peak_courant"] < 0.3
    assert measured["peak_density_compression"] <= 0.05
    # 40 steps of free fall: the drop speed must be g t, not an arbitrary number.
    assert measured["max_speed_end"] == pytest.approx(9.81 * measured["time_end"], rel=0.05)


def test_manifest_criteria_agree_with_the_diagnostics_csv(tmp_path):
    cfg = sph.Config(dx=0.05, steps=30, output_every=5, device="cpu")
    sph.run(cfg, tmp_path, plot=False)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    criteria = {item["id"]: item for item in manifest["acceptance"]["criteria"]}
    peak_courant = criteria["acoustic_courant"]["measured"]
    # The all-steps extremum cannot be below the sampled maximum, and must
    # respect the limit it claims to pass.
    sampled = manifest["sampling"]["extrema"]["peak_courant"]
    assert peak_courant == pytest.approx(sampled, rel=1e-12)
    assert peak_courant <= cfg.max_courant
    assert criteria["finite_state"]["result"] == "pass"
    assert criteria["boundary_escape"]["result"] == "pass"


# --------------------------------------------------------------------------- #
# The script must run the way a reader runs it, and the prose must match
# --------------------------------------------------------------------------- #

def test_script_runs_standalone_and_reports_measured_values(tmp_path):
    dump = tmp_path / "measured.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", str(dump)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
             "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert "all tutorial assertions passed" in completed.stdout
    printed = [line for line in completed.stdout.splitlines() if line.startswith("MEASURED ")]
    assert len(printed) >= 30, f"only {len(printed)} measured lines"
    dumped = json.loads(dump.read_text())
    assert set(dumped) == {"step1_kernel_normalisation", "step2_density_calibration",
                           "step3_equation_of_state", "step4_neighbour_search",
                           "step5_time_step_policy", "step6_run_case"}


def test_tutorial_document_quotes_a_run_of_this_script(results):
    """Every ``MEASURED`` value in the doc must match a fresh run."""
    assert DOCUMENT.exists(), f"missing tutorial document {DOCUMENT}"
    quoted = [MEASURED.match(line.strip()) for line in DOCUMENT.read_text().splitlines()]
    quoted = [match for match in quoted if match is not None]
    assert len(quoted) >= 30, f"the tutorial quotes only {len(quoted)} measured values"

    fresh = {f"{name}.{key}": value
             for name, measurements in results.items()
             for key, value in measurements.items()}

    for match in quoted:
        key, text = match.group(1), match.group(2)
        assert key in fresh, f"the document quotes unknown measurement {key}"
        value = fresh[key]
        if isinstance(value, str):
            assert text == value, f"{key} drifted: doc {text!r} vs run {value!r}"
        else:
            assert float(text) == pytest.approx(value, rel=1e-9, abs=1e-12), (
                f"{key} drifted: doc {text} vs run {value!r}")


def test_tutorial_artifacts_are_committed(results):
    """The captured run the doc quotes is kept next to it as evidence."""
    captured = ROOT / "docs" / "tutorials" / "captured_run" / "first_steps.json"
    assert captured.exists(), "the captured run log is missing"
    recorded = json.loads(captured.read_text())
    for name, measurements in results.items():
        assert name in recorded
        for key, value in measurements.items():
            if isinstance(value, str):
                assert recorded[name][key] == value
            else:
                assert recorded[name][key] == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_steps_are_reusable_with_their_own_parameters(tutorial):
    """Each step is a function of its arguments, not a one-shot script."""
    small = tutorial.step4_neighbour_search(count=300)
    full = tutorial.step4_neighbour_search(count=900)
    assert small["pairs_identical"] == full["pairs_identical"] == 1.0
    assert small["pair_count_brute_force"] < full["pair_count_brute_force"]
    assert torch.get_default_dtype() in (torch.float32, torch.float64)
