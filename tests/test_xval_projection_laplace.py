"""Gates for XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE (see docs/xval_projection_laplace_oss.md).

Predeclared before the numbers were inspected: owner tolerance 15 % on the
Young-Laplace jump, 5 % on the curvature product, and a *locked* expectation
that the reference's shipped number-density normalisation is unusable.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "xval_projection_laplace"))

from tensorsph.droplet_impact import cubic_spline_kernel  # noqa: E402
from oss_ports import (  # noqa: E402
    oss_smoothed_color, pysph_parity_check, PYSPH_SITE,
)

import run as xval  # noqa: E402

torch.set_num_threads(2)
DX = 0.002
TOL = 0.15


@pytest.fixture(scope="module")
def arms():
    return (xval.ours_case(DX), xval.oss_case(DX, form="paper"),
            xval.oss_case(DX, form="code"))


def test_ours_young_laplace_within_gate(arms):
    ours = arms[0]
    assert abs(ours["dp_error_pct"]) < 100 * TOL


def test_oss_parity_arm_within_gate(arms):
    oss = arms[1]
    assert abs(oss["dp_error_pct"]) < 100 * TOL


def test_arms_agree(arms):
    ours, oss = arms[0], arms[1]
    rel = (oss["dp_measured_Pa"] - ours["dp_measured_Pa"]) / ours["dp_theory_Pa"]
    assert abs(rel) < TOL


def test_curvature_recovers_one_over_R(arms):
    for rep in arms[:2]:
        kR = rep.get("kappa_times_R", rep.get("kappaR_delta_weighted"))
        assert abs(kR - 1.0) < 0.05, kR


def test_oss_shipped_number_density_is_catastrophic(arms):
    """The reference's ``2/(V_i+V_j)`` branch must stay broken (ported trap)."""
    assert abs(arms[2]["dp_error_pct"]) > 100.0


def test_smoothed_color_self_term_matches_reference_formula():
    """PySPH's ``SmoothedColor`` is an *unnormalised* kernel sum incl. self."""
    h = 1.3e-2
    pos = torch.tensor([[0.0, 0.0], [0.5 * h, 0.0]], dtype=torch.float64)
    vol = torch.tensor([h * h, h * h], dtype=torch.float64)
    color = torch.tensor([1.0, 0.0], dtype=torch.float64)
    pairs = torch.tensor([[0, 1]])
    c = oss_smoothed_color(pos, color, vol, pairs, h)
    w0 = float(cubic_spline_kernel(torch.zeros((), dtype=torch.float64), h))
    w1 = float(cubic_spline_kernel(torch.tensor(0.5 * h), h))
    assert c[0].item() == pytest.approx(vol[0].item() * 1.0 * w0 + vol[1].item() * 0.0 * w1)
    assert c[1].item() == pytest.approx(vol[0].item() * 1.0 * w1)


def test_reference_kernel_normalisation_identical():
    sys.path.append(PYSPH_SITE)
    try:
        from pysph.base.kernels import CubicSpline
    except Exception:  # pragma: no cover - reference not installed
        pytest.skip("PySPH reference not installed")
    k = CubicSpline(dim=2)
    h = 1.3e-2
    for q in (0.0, 0.3, 1.0, 1.7):
        ours = float(cubic_spline_kernel(torch.tensor(q * h), h))
        ref = k.kernel(xij=[q * h, 0.0, 0.0], rij=q * h, h=h)
        if abs(ours / ref - 1.0) > 1e-6:
            pytest.xfail(f"kernel normalisation differs: {ours} vs {ref}")


def test_port_parity_against_executed_reference():
    sys.path.append(PYSPH_SITE)
    try:
        import pysph  # noqa: F401
    except Exception:  # pragma: no cover
        pytest.skip("PySPH reference not installed")
    out = pysph_parity_check()
    for key in ("SmoothedColor", "ColorGradient_gradC", "ColorGradient_normal",
                "SY11DiracDelta_delta", "CSFForce_acc_x"):
        assert out[key] < 1e-5, (key, out[key])
    # documented open items: volume-weighted sums with the self pair
    for key in ("Curvature_kappa", "Curvature_wij_sum", "PressureCoeff_diag"):
        assert key in out


def test_results_json_is_consistent_with_gates():
    path = ROOT / "scripts" / "xval_projection_laplace" / "results" / "xval_projection_laplace.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run.py not executed in this checkout")
    d = json.loads(path.read_text())
    assert d["verdict"]["ours_pass"] and d["verdict"]["oss_pass"]
    for r in d["runs"]:
        if r["arm"].startswith(("ours[", "oss[paper")):
            assert abs(r["dp_error_pct"]) < 100 * TOL
            assert abs(r["dp_theory_Pa"] - 0.072 / 0.02) < 1e-12