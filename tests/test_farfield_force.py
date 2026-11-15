"""Guards for the far-field (control-volume) force-from-fluid channel.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.  Every assertion below
is an *invariant* with a pre-known value, not a re-run of the driver:

* the closed-surface quadrature must close (``oint n dS = 0``),
* it must be linear-exact (``oint (g . x) n dS = A g``),
* Kutta-Joukowski lift and d'Alembert drag must come out of the flux form,
* the wake momentum deficit must reproduce its closed form,
* and at the coupled ``fsi2`` spacing the far-field arm must stay inside the
  owner gate (3 %) while the near-field kernel arm must *fail* it -- that
  contrast is the whole point of the channel, so it is asserted too.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

from tensorsph.farfield_force import (  # noqa: E402
    circular_surface_bins,
    control_volume_force,
    potential_flow_cylinder,
    wake_plane_drag,
)

import farfield_force_audit as audit  # noqa: E402

DT = torch.float64


def test_closed_surface_normals_sum_to_zero():
    centre = torch.zeros(2, dtype=DT)
    surf = circular_surface_bins(torch.tensor([[0.5, 0.0], [-0.5, 0.0]], dtype=DT),
                                 centre, 0.5, 0.02, n_bins=64)
    total = (surf["normals"] * surf["ds"]).sum(dim=0)
    assert float(total.norm()) < 1e-12


def test_null_and_buoyancy_identities():
    out = audit.gate_g1()
    assert out["worst_null_rel"] < 1e-9, out["worst_null_rel"]
    assert out["worst_buoy_rel"] < 1e-6, out["worst_buoy_rel"]
    # the plain bin average is *not* linear-exact; the fit is what buys the identity
    assert out["worst_buoy_mean_rel"] > 1e-4


def test_kutta_joukowski_and_dalembert():
    rows = audit.gate_g2(a=0.1, u_inf=1.0)
    worst = max(abs(r["lift_rel_err"]) for r in rows)
    assert worst < 2e-3, worst
    # d'Alembert: the drag of a steady potential flow about a cylinder is zero
    for r in rows:
        assert abs(r["drag_rel_to_lift"]) < 2e-3, r
    # sign convention: lift = -rho U Gamma for the counter-clockwise Gamma
    for r in rows:
        assert r["lift"] < 0.0


def test_wake_momentum_deficit_closed_form():
    out = audit.gate_g3()
    worst = max(abs(r["rel_err"]) for r in out["rows"])
    assert worst < 2e-3, worst


def test_far_field_beats_near_field_at_fsi2_scale():
    rows = audit.gate_g4(dx_list=(0.02,))
    row = rows[0]
    assert abs(row["h_over_R"] - 0.52) < 1e-12
    # owner gate, 3 % target: the far-field channel holds it
    assert abs(row["farfield_lift_rel_err"]) < 0.03, row
    # a control volume that actually fits inside the 0.41 m channel (3R) too
    assert abs(row["farfield_lift_rel_err_cv3R"]) < 0.03, row
    # and the near-field arms at this spacing do not -- the documented defect
    assert abs(row["kernel_lift_rel_err"]) > 0.15, row
    assert abs(row["fit_lift_rel_err"]) > 0.15, row
    # ... and that is not an artefact of one annulus thickness
    assert abs(row["kernel_lift_rel_err_5h"]) > 0.15, row
    assert abs(row["fit_lift_rel_err_5h"]) > 0.15, row
    assert row["nearfield_coverage"] == 1.0
    assert row["cv_coverage"] == 1.0 and row["cv3R_coverage"] == 1.0


def test_circular_control_volume_rejects_empty_band():
    pos = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)
    with pytest.raises(ValueError):
        control_volume_force(pos, torch.zeros_like(pos),
                             torch.zeros(2, dtype=DT), 1.0,
                             torch.zeros(2, dtype=DT), 0.5, 0.01)
    with pytest.raises(ValueError):
        wake_plane_drag(pos, torch.zeros_like(pos), torch.zeros(2, dtype=DT),
                        1.0, 1.0, 0.0, 5.0, 0.5, 0.01)


def test_reconstruction_rejects_unknown_mode():
    pos = torch.tensor([[0.5, 0.0], [0.4, 0.1]], dtype=DT)
    with pytest.raises(ValueError):
        control_volume_force(pos, torch.zeros_like(pos), torch.zeros(2, dtype=DT),
                             1.0, torch.zeros(2, dtype=DT), 0.5, 0.05,
                             reconstruct="magic")