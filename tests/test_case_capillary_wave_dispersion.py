"""Case-bank gate: CASE-CAPILLARY-WAVE (capillary-wave dispersion, TensorSPH).

Machine-checkable form of the case contract:

* the reference is a *closed form* evaluated at 50 digits, and its two textbook
  reductions (Lamb 1932 Art. 267; finite-depth gravity-capillary at g = 0) hold,
* the measured dispersion of the shipped CSF operator on the two-layer cloud is
  within the case tolerance (10 %) at the design point,
* the ``delta_s`` of the two-layer construction integrates to the Brackbill unit
  mass (1.0 +- 5 %),
* the measurement is deterministic (same numbers on a second run).

The full campaign (resolution / amplitude / sigma ladders + figures) lives in
``benchmarks/case_capillary_wave_dispersion/run.py``; this test runs the design
point only, so it stays a few seconds.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = ROOT / "benchmarks" / "case_capillary_wave_dispersion"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CASE_DIR))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load("case_cw_ref", CASE_DIR / "ref.py")
run = _load("case_cw_run", CASE_DIR / "run.py")


def test_reference_reductions():
    """The two-layer formula collapses onto the textbook limits to 1e-12."""
    checks = ref.reduction_checks(314.159, 0.072, 1000.0, 1.2, 0.016, 0.010)
    assert checks["all_below_1e-12"], checks


def test_reference_value_is_lamb_in_the_deep_limit():
    """For rho2 = 0 and k h1 >> 1 the reference equals sqrt(sigma k^3/rho1)."""
    k, sigma, rho = 2.0 * math.pi / 0.020, 0.072, 1000.0
    two = float(ref.omega_two_layer(k, sigma, rho, 0.0, 1.0e4 / k, 1.0e4 / k))
    lamb = float(ref.omega_deep_lamb(sigma, k, rho))
    assert abs(two - lamb) / lamb < 1e-9


@pytest.fixture(scope="module")
def design_point():
    return run.case_point(ppw=run.DESIGN["ppw"],
                          curvature_scheme=run.DESIGN["curvature_scheme"],
                          amplitude_fraction=run.DESIGN["amplitude_fraction"],
                          sigma=run.PARAMS["sigma_N_m"], shifts=run.SHIFTS)


def test_design_point_within_tolerance(design_point):
    assert design_point["abs_error_pct"] < run.TOL_PCT, design_point
    assert design_point["pass_case"]


def test_delta_s_unit_mass(design_point):
    assert abs(design_point["delta_s_integral"] - 1.0) < 0.05, design_point


def test_sigma_linearity():
    """omega ~ sqrt(sigma): the stiffness ratio must not move with sigma."""
    rows = [run.case_point(ppw=run.DESIGN["ppw"], curvature_scheme="mls",
                           amplitude_fraction=run.DESIGN["amplitude_fraction"],
                           sigma=f * run.PARAMS["sigma_N_m"], shifts=1)
            for f in (0.5, 2.0)]
    ratios = [r["stiffness_ratio"] for r in rows]
    assert abs(ratios[0] - ratios[1]) / max(abs(ratios[0]), 1e-30) < 0.05, ratios


def test_deterministic():
    kw = dict(ppw=24, curvature_scheme="mls", amplitude_fraction=0.02,
              sigma=run.PARAMS["sigma_N_m"], shifts=1)
    a = run.case_point(**kw)
    b = run.case_point(**kw)
    assert a["stiffness_ratio"] == b["stiffness_ratio"]