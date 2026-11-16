"""Tests for the C6 reversed-peak level/shape attribution (CASE-POISEUILLE-SPH).

The study's claim is *analytic*, so the tests are built to break it rather than
to re-run it:

* the bin-average bias must equal an independently computed midpoint-rule
  integral of the closed form (not the formula quoting itself);
* the level/shape split must be an identity on synthetic rows with a *known*
  injected offset and shape;
* a injected **slope** (what a wrong wall speed would look like) must show up in
  the shear channel and must *not* be absorbed by ``dc`` -- otherwise the "it is
  a level, not a shear" attribution would be unfalsifiable;
* the published C6 numbers must be reproduced exactly, and the artifact must be
  bound to the source artifact's SHA.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import backflow_level as BL  # noqa: E402

H = 0.1
NU = 0.01
G = -4.905
UW = 1.47


def make_rows(y, u):
    return [{"y": float(a), "u": float(b)} for a, b in zip(y, u)]


def row_grid(resolution: int):
    dx = H / resolution
    y = dx * 0.5 + dx * np.arange(resolution)
    return y, dx


# --------------------------------------------------------------------------- #
# closed-form pieces
# --------------------------------------------------------------------------- #

def test_curvature_is_constant_minus_g_over_nu():
    assert BL.curvature(H, G, NU, UW) == pytest.approx(-G / NU)
    # Couette term is linear: switching the wall speed must not move u''
    assert BL.curvature(H, G, NU, 3.0 * UW) == pytest.approx(BL.curvature(H, G, NU, UW))


@pytest.mark.parametrize("resolution", [8, 12, 16, 24])
def test_bin_average_bias_equals_midpoint_rule_integral(resolution):
    """``(dx^2/24) u''`` vs a numerically integrated bin average of the closed form."""
    y, dx = row_grid(resolution)
    quad_pts = np.linspace(-0.5, 0.5, 20001)
    for yc in y[2:-2]:
        ys = yc + quad_pts * dx
        avg = np.trapezoid(BL.profile_u(ys, H, G, NU, UW), ys) / dx
        centre = BL.profile_u([yc], H, G, NU, UW)[0]
        assert avg - centre == pytest.approx(
            BL.bin_average_bias(dx, H, G, NU, UW), rel=1e-6, abs=1e-12)


def test_quadrature_bias_sign_follows_curvature():
    """Convex profile -> bin mean above the centre value; concave -> below."""
    dx = H / 8
    assert BL.curvature(H, G, NU, UW) > 0.0
    assert BL.bin_average_bias(dx, H, G, NU, UW) > 0.0
    assert BL.bin_average_bias(dx, H, -G, NU, UW) < 0.0     # adverse gradient
    # scales as dx^2
    assert BL.bin_average_bias(2 * dx, H, G, NU, UW) == pytest.approx(
        4.0 * BL.bin_average_bias(dx, H, G, NU, UW))


# --------------------------------------------------------------------------- #
# decomposition identity and its falsifiers
# --------------------------------------------------------------------------- #

def _synthetic(resolution=12, dc=0.0, slope=0.0, shape_amp=0.0):
    y, dx = row_grid(resolution)
    u_a = BL.profile_u(y, H, G, NU, UW)
    shape = shape_amp * np.sin(3.0 * np.pi * y / H)
    u = u_a + dc + slope * (y / H) + shape
    return y, dx, u, u_a, shape


def test_decomposition_is_an_identity_with_known_offset_and_shape():
    y, dx, u, u_a, shape = _synthetic(dc=-0.012, shape_amp=1e-4)
    d = BL.decompose(make_rows(y, u), H, G, NU, UW, dx)
    # ``dc`` is the row mean of the residual: the injected offset *plus* whatever
    # mean the shape itself carries on this row grid (it is not zero at 12 rows)
    assert d["dc"] == pytest.approx(-0.012 + shape.mean(), rel=0, abs=1e-12)
    # ``shape`` here is the *level-removed* residual, so it is the centred shape
    assert d["shape_rms"] == pytest.approx(
        float(np.sqrt(((shape - shape.mean()) ** 2).mean())), rel=1e-9)
    assert d["c6_recomposition_rel"] < 1e-9
    # the level term, converted to the C6 scale, is the whole offset
    assert d["c6_level_term_pct"] == pytest.approx(
        100.0 * d["dc"] / abs(d["u_ref_at_min_row"]), rel=1e-12)


def test_amplification_factor_is_uw_over_uref():
    """The 15x magnifier: ``U_w / |u_ref|`` -- the study's central quantity."""
    y, dx, u, u_a, _ = _synthetic()
    d = BL.decompose(make_rows(y, u), H, G, NU, UW, dx)
    assert d["amplification_Uw_over_uref"] == pytest.approx(
        UW / abs(d["u_ref_at_min_row"]), rel=1e-12)
    assert d["amplification_Uw_over_uref"] == pytest.approx(14.9, abs=0.5)


def test_shape_is_invariant_to_a_constant_offset():
    y, dx, u, _, _ = _synthetic(dc=0.0, shape_amp=1e-4)
    y2, dx2, u2, _, _ = _synthetic(dc=5e-3, shape_amp=1e-4)
    a = BL.decompose(make_rows(y, u), H, G, NU, UW, dx)
    b = BL.decompose(make_rows(y2, u2), H, G, NU, UW, dx2)
    assert a["shape_rms"] == pytest.approx(b["shape_rms"], rel=1e-12)
    assert b["dc"] - a["dc"] == pytest.approx(5e-3, rel=0, abs=1e-12)


def test_a_shear_residual_is_not_absorbed_by_the_level():
    """The L6 falsifier: a wall-speed error must land in the slope channel.

    A moving-wall coupling error of ``eps`` in ``U_w`` produces the residual
    ``eps y/H``: zero mean, so ``dc`` cannot see it, and its *peak* signature is
    small.  If the attribution had only a level channel, this error would be
    invisible -- which is exactly why the study grades the slope separately.
    """
    eps = 0.02                                    # 1.4 % of U_w on the wall
    dc0 = -0.004
    y, dx, u, _, _ = _synthetic(dc=dc0, slope=eps)
    d = BL.decompose(make_rows(y, u), H, G, NU, UW, dx)

    # the two-term projection recovers both coefficients exactly
    assert d["slope_pct_Uw"] == pytest.approx(100.0 * eps / UW, rel=1e-9)
    assert d["dc"] == pytest.approx(dc0 + eps * float(np.mean(y / H)), rel=0,
                                   abs=1e-12)

    # ... whereas the raw row mean cannot tell them apart: half of a shear reads
    # as level, so a level-only attribution would misreport a wall-speed error.
    # This is the reason L6 grades the slope separately instead of trusting L2.
    assert d["dc"] != pytest.approx(dc0, abs=1e-9)
    assert abs(d["slope_pct_Uw"]) > BL.GATES["L6_slope_pct_Uw"]
    assert d["c6_recomposition_rel"] < 1e-9


def test_order_fit_recovers_a_known_second_order_series():
    dxs = np.array([H / 8, H / 12, H / 16, H / 24])
    values = 3.7e-3 * dxs ** 2
    fit = BL.fit_order(values, dxs)
    assert fit["order"] == pytest.approx(2.0, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_order_fit_handles_a_non_monotone_series_without_pretending():
    fit = BL.fit_order([1e-3, -2e-3, 1e-5], [H / 8, H / 12, H / 16])
    assert np.isfinite(fit["order"])
    assert fit["n"] == 3


# --------------------------------------------------------------------------- #
# the published record
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def artifact():
    path = BL.ARTIFACT
    if not path.exists():
        pytest.skip("run benchmarks/case_poiseuille/backflow_level.py first")
    return json.loads(path.read_text())


def test_artifact_reproduces_the_published_c6_values(artifact):
    """L0: the attribution uses the gate's own estimator, not a lookalike."""
    src = json.loads(BL.SOURCE.read_text())
    stored = {r["resolution_H_over_dx"]: r["backflow"]["peak_pct"]
              for r in src["runs"]}
    for r in artifact["rungs"]:
        assert r["binding_rel"] <= BL.GATES["L0_binding_rel"]
        assert r["c6_peak_pct"] == pytest.approx(
            stored[r["resolution_H_over_dx"]], rel=1e-9)


def test_artifact_records_the_honest_fail_it_does_not_regrade(artifact):
    """The anchor FAIL is carried, not explained away."""
    failed = [r for r in artifact["rungs"]
              if abs(r["c6_peak_pct"]) > artifact["tolerance_percent"]]
    assert [r["resolution_H_over_dx"] for r in failed] == [8, 12]
    assert "FAIL" in artifact["note"].upper() or "carried" in artifact["note"]


def test_artifact_gates_all_pass_and_are_recomputed(artifact):
    assert artifact["all_gates_pass"] is True
    assert set(artifact["gates"]) == set(BL.GATES)
    for name, verdict in artifact["gates"].items():
        assert verdict is True, name


def test_artifact_source_sha_is_current(artifact):
    assert artifact["source_artifact_sha256"] == BL.sha256(BL.SOURCE)


def test_artifact_ladder_bounds_hold(artifact):
    for r in artifact["rungs"]:
        assert r["shape_rms_pct_Uw"] <= BL.GATES["L1_shape_rms_pct_Uw"]
        assert abs(r["dc_pct_Uw"]) <= BL.GATES["L2_dc_pct_Uw"]
        assert abs(r["shape_only_peak_pct"]) <= BL.GATES["L4_shape_only_peak_pct"]
        assert abs(r["slope_pct_Uw"]) <= BL.GATES["L6_slope_pct_Uw"]
        assert r["quadrature_bias"] > 0.0          # convex profile, see above
        assert r["solver_dc"] < 0.0               # the solver undershoots the level
    assert artifact["orders"]["solver_dc"]["order"] > BL.GATES["L3_order_solver_dc"]


def test_check_mode_agrees_with_the_written_artifact():
    proc = subprocess.run([sys.executable, str(Path(BL.__file__)), "--check"],
                          cwd=str(REPO), capture_output=True, text=True,
                          env={"PYTHONPATH": f".:{REPO / 'src'}",
                               "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "2"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout