"""Regression guards for the CSF curvature-bias fix (mission m2592).

The audit behind these tests lives in
``benchmarks/csf_consistent/curvature_audit.py`` and the write-up in
``docs/verification_csf_curvature_bias.md``.  Everything here is instant-only
(no time stepping) and uses the two coarsest audit resolutions to stay fast.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import (
    csf_fields,
    mls_divergence,
)
from tensorsph.droplet_impact import density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs

ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = ROOT / "benchmarks" / "csf_consistent" / "curvature_audit.json"

RADIUS = 0.02
SIGMA = 7.2
RHO0 = 1000.0
H_FACTOR = 1.3
KAPPA_EXACT = 1.0 / RADIUS


def _droplet(dx):
    spec = importlib.util.spec_from_file_location(
        "laplace_law_run", ROOT / "benchmarks" / "laplace_law" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pos, mass = module.hexagonal_droplet(RADIUS, dx, RHO0)
    pos, mass = pos.double(), mass.double()
    h = H_FACTOR * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    return pos, mass, rho, h, pairs


@pytest.fixture(scope="module")
def fields():
    """Per-resolution diagnostics, computed once for the whole module."""
    out = {}
    torch.set_num_threads(2)
    for dx in (0.002, 0.003):
        pos, mass, rho, h, pairs = _droplet(dx)
        vol = mass / RHO0
        base = csf_fields(pos, mass, pairs, h, RHO0, curvature_color_smoothing=0)
        improved = csf_fields(pos, mass, pairs, h, RHO0,
                               curvature_color_smoothing=1)  # fix path (opt-in)
        r = torch.linalg.vector_norm(pos, dim=1)
        band = base.delta_s > 0.05 * float(base.delta_s.max())

        def kappa_err(kappa):
            w = base.delta_s[band]
            kappa_w = float((kappa[band] * w).sum() / w.sum())
            return (kappa_w - KAPPA_EXACT) / KAPPA_EXACT * 100.0

        control = mls_divergence(pos, pos / r[:, None], vol, pairs, h)
        acc = (-SIGMA * improved.kappa * improved.delta_s)[:, None] \
            * improved.normal / rho[:, None]
        cut = float((mass[pos[:, 0] > 0, None] * acc[pos[:, 0] > 0]).sum(dim=0)[0])
        sat = band & (base.color >= 0.99)
        out[dx] = {
            "base_err": kappa_err(base.kappa),
            "default_err": kappa_err(improved.kappa),
            "control_err": kappa_err(control),
            "cut_ratio": cut / (-2.0 * SIGMA),
            "band": band,
            "sat": sat,
            "ds_share_sat": float(base.delta_s[sat].sum() / base.delta_s[band].sum()),
            "kappa_sat_mean": float(base.kappa[sat].mean()) if int(sat.sum()) else float("nan"),
            "n_sat": int(sat.sum()),
        }
    return out


def test_operator_itself_is_not_biased(fields):
    """MLS div n fed with the exact radial normal must be much better than with
    the estimated normal -- otherwise the fix would be in the wrong place."""
    for dx, f in fields.items():
        # strictly better at both resolutions, decisively so at dx = 2 mm
        margin = 5.0 if dx <= 0.002 else 0.0
        assert abs(f["control_err"]) < abs(f["base_err"]) - margin, \
            (dx, f["control_err"], f["base_err"])


def test_legacy_curvature_error_does_not_converge(fields):
    """The m2447 failure was not caused by a too-coarse mesh (both > 15 %)."""
    for dx, f in fields.items():
        assert abs(f["base_err"]) > 15.0, (dx, f["base_err"])


def test_fix_path_curvature_passes_the_gate(fields):
    """The audited fix path (explicit curvature_color_smoothing=1; the shipped
    default is 0 = m2447) is inside the 15 % platform gate for dx <= 2 mm,
    and the cut force stays inside it too."""
    f = fields[0.002]
    assert abs(f["default_err"]) < 15.0, f["default_err"]
    assert abs(f["cut_ratio"] - 1.0) < 0.15, f["cut_ratio"]


def test_smoothing_switch_reproduces_the_legacy_path(fields):
    """curvature_color_smoothing=0 must stay available (regression control)."""
    for dx, f in fields.items():
        assert abs(f["default_err"]) < abs(f["base_err"]), (dx, f)


def test_bias_is_carried_by_saturated_inner_shell(fields):
    """The saturated (c >= 0.99) particles are a minority of the delta_s weight
    but are where the curvature blows up -- the attribution claimed in the docs."""
    for dx, f in fields.items():
        assert f["n_sat"] > 0, dx
        assert f["kappa_sat_mean"] / KAPPA_EXACT > 1.5, (dx, f["kappa_sat_mean"])
        assert f["ds_share_sat"] < 0.25, (dx, f["ds_share_sat"])
        # ... and they still carry enough weight to move the weighted mean
        assert f["ds_share_sat"] > 0.05, (dx, f["ds_share_sat"])


def test_committed_audit_artifact_matches_its_verdicts():
    """The committed JSON must be internally consistent (numbers not hand-edited)."""
    payload = json.loads(AUDIT_JSON.read_text())
    rows = payload["rows"]
    assert len(rows) >= 4
    shipped = [v for r in rows for v in r["variants"]
               if v["label"].startswith("V0")]
    assert len(shipped) == len(rows)
    for row, entry in zip(rows, [r["variants"][0] for r in rows]):
        assert entry["kappa_rel_error_pct"] == pytest.approx(
            (entry["kappa_delta_s_weighted_1_per_m"] / KAPPA_EXACT - 1.0) * 100.0, abs=1e-6)
        assert row["dx_m"] in payload["setup"]["resolutions_m"]
