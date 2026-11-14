"""Gates for round 5 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE (3-D error budget).

Round 1 (``docs/xval_projection_laplace_oss.md``) cross-validated the 2-D jump
against an OSS projection-SPH implementation, round 2
(``docs/xval_projection_laplace_robustness.md``) measured the error budget of
that claim over every knob the physics cannot see, round 3
(``docs/xval_projection_laplace_wellbalanced.md``) made the time loop
well-balanced, round 4 (``docs/verification_laplace_sph_isph_dynamic.md``) ran
the dynamic equilibrium.

Round 5 asks the round-2 question in **3-D**, where a circle's second principal
curvature is no longer identically zero: ``Delta p = 2 sigma / R`` on a sphere
and ``2 sigma (1/R_in + 1/R_out)`` on a shell.  The gates below were declared
before the sweep numbers were inspected and are written as *spreads* of a knob
sweep, so a change that moves the fitted point value while making it more
knob-dependent cannot pass.

The tests re-run the sweep themselves instead of trusting the stored artifact;
``test_artifact_agrees_with_the_doc`` additionally cross-checks the JSON against
the markdown so a hand-edited number fails the gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "hollow_sphere_3d"))

from tensorsph.threed.hollow_sphere import (  # noqa: E402
    hollow_sphere_case,
    shell_laplace_analytic,
    sphere_shell_lattice,
)

torch.set_num_threads(2)

SIGMA = 0.072
RHO0 = 1000.0
BASE = dict(radius_in=0.0, radius_out=8.0, dx=1.0, pad=4.0, h_factor=1.3,
            lattice="cubic", curvature="analytic")

RESULTS = ROOT / "benchmarks" / "hollow_sphere_3d" / "results_knobs"
FULL_JSON = RESULTS / "results_knobs.json"
DOC = ROOT / "docs" / "xval_projection_laplace_3d_budget.md"

TOL = 15.0          # owner acceptance on the jump, percent (all dp_* are %)
TIGHT = 5.0         # spread gate on a knob sweep, percent
FCC = 5.0           # fcc vs cubic spread, percent


def _run(**over):
    kw = dict(BASE)
    kw.update(over)
    kw.setdefault("sigma", SIGMA)
    return hollow_sphere_case(rho0=RHO0, **kw)


# --------------------------------------------------------------------------- #
# defect D1: the fcc option must book one *site* volume, not one cube
# --------------------------------------------------------------------------- #
def test_fcc_carries_the_same_geometric_volume_as_cubic():
    """The doubled fcc lattice must not double the carried mass."""
    p_c, m_c, _, _, half_c = sphere_shell_lattice(0.0, 8.0, 1.0, 4.0, RHO0, "cubic")
    p_f, m_f, _, _, half_f = sphere_shell_lattice(0.0, 8.0, 1.0, 4.0, RHO0, "fcc")
    # nearest-neighbour spacing: dx for cubic, dx/sqrt(2) for the doubled copy
    d_c = torch.cdist(p_c[:1], p_c).flatten()
    d_f = torch.cdist(p_f[:1], p_f).flatten()
    assert abs(float(d_c[d_c > 1e-9].min()) - 1.0) < 1e-9
    assert abs(float(d_f[d_f > 1e-9].min()) - 2.0 ** -0.5) < 1e-9
    # ... and the sampled *volume* is the same, i.e. mass per site is halved
    vol_c = float((m_c / RHO0).sum())
    vol_f = float((m_f / RHO0).sum())
    assert vol_f / vol_c == pytest.approx(1.0, rel=0.06)
    assert float(m_f[0] / RHO0) == pytest.approx(0.5, rel=1e-12)


def test_fcc_young_laplace_jump_is_not_halved():
    """Regression for the shipped -50.7 % fcc defect: both packings agree."""
    cubic = _run(lattice="cubic")
    fcc = _run(lattice="fcc")
    # the delta_s shell normalisation is what broke with the doubled mass
    assert cubic["delta_s_integral_dn"] == pytest.approx(1.0, abs=0.06)
    assert fcc["delta_s_integral_dn"] == pytest.approx(1.0, abs=0.06)
    assert abs(fcc["dp_error_pct"]) < FCC
    assert abs(fcc["dp_error_pct"] - cubic["dp_error_pct"]) < FCC
    assert abs(fcc["dp_error_pct"]) < TOL


# --------------------------------------------------------------------------- #
# physical invariance: where the interface sits inside a lattice cell
# --------------------------------------------------------------------------- #
def test_translation_invariance_of_the_3d_jump():
    offsets = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]]
    errs = [_run(center=o)["dp_error_pct"] for o in offsets]
    assert max(errs) - min(errs) < TIGHT
    assert all(abs(e) < TOL for e in errs)


def test_center_must_be_a_3_sequence():
    with pytest.raises(ValueError):
        sphere_shell_lattice(0.0, 4.0, 1.0, 4.0, RHO0, "cubic", [0.5, 0.5])


# --------------------------------------------------------------------------- #
# numerical knobs: gauge pad and kernel support
# --------------------------------------------------------------------------- #
def test_gauge_pad_and_kernel_support_spreads():
    pads = [_run(pad=p)["dp_error_pct"] for p in (3.0, 4.0, 6.0)]
    hs = [_run(h_factor=h)["dp_error_pct"] for h in (1.2, 1.4)]
    assert max(pads) - min(pads) < TIGHT
    assert max(hs) - min(hs) < TIGHT


# --------------------------------------------------------------------------- #
# the claim: 2 sigma / R on the sphere, both curvature routes
# --------------------------------------------------------------------------- #
def test_spherical_jump_is_twice_the_circle_value():
    """A solver that returns sigma/R instead of 2 sigma/R must fail here."""
    rep = _run(radius_out=8.0)
    assert rep["dp_theory_Pa"] == pytest.approx(2.0 * SIGMA / 8.0, rel=1e-12)
    assert abs(rep["dp_error_pct"]) < 3.0
    # the 2-D value would be a -50 % error; make that explicit
    assert abs(rep["dp_measured_Pa"] - SIGMA / 8.0) > 0.4 * rep["dp_theory_Pa"]


def test_resolution_and_self_consistent_curvature():
    fine = _run(radius_out=10.0)
    mls = _run(curvature="mls")
    assert abs(fine["dp_error_pct"]) < 3.0
    assert abs(mls["dp_error_pct"]) < TOL
    # the imposed-curvature route isolates the projection, the div n route the CSF
    assert abs(mls["dp_flux_error_pct"]) < TOL


def test_hollow_sphere_two_interfaces():
    rep = _run(radius_in=6.0, radius_out=14.0)
    ref = shell_laplace_analytic(SIGMA, 6.0, 14.0)
    assert rep["dp_theory_Pa"] == pytest.approx(ref, rel=1e-12)
    assert abs(rep["dp_error_pct"]) < TIGHT
    thin = _run(radius_in=6.0, radius_out=7.0)
    # thin film limit: 4 sigma / R is the soap-bubble value, not 2 sigma / R
    assert thin["dp_theory_Pa"] == pytest.approx(2.0 * SIGMA * (1 / 6.0 + 1 / 7.0),
                                                 rel=1e-12)


def test_negative_control_sigma_zero():
    rep = _run(sigma=0.0)
    assert abs(rep["dp_measured_Pa"]) < 1e-6


# --------------------------------------------------------------------------- #
# artifact honesty
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not FULL_JSON.exists(), reason="run audit_knobs.py first")
def test_artifact_agrees_with_the_doc():
    res = json.loads(FULL_JSON.read_text())
    assert res["checks"]["all_pass"] is True
    checks = res["checks"]
    assert checks["worst_inside_15pct"] is True
    assert checks["fcc_inside_3pct"] is True
    assert checks["resolution_12_inside_3pct"] is True
    assert checks["shell_inside_5pct"] is True
    assert checks["all_pass"] is True
    # every axis must carry rows, and no axis may be quoted from a single point
    for name, axis in res["axes"].items():
        if name == "control":
            continue
        assert len(axis["rows"]) >= 2, name
    if DOC.exists():
        text = DOC.read_text()
        worst = checks["worst_single_config_pct"]
        assert f"{worst:+.2f}" in text or f"{worst:.2f}" in text, (
            "the doc must quote the stored worst-case error")
        fcc = checks["fcc_pct"]
        assert f"{fcc:+.2f}" in text or f"{fcc:.2f}" in text