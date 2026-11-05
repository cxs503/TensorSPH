"""Tests for the surface delta function: exact-gradient scheme + force density.

The delta function ``delta_s`` is the amplitude carrier of the CSF force
(``a = -sigma kappa delta_s n / rho``), so it is scored against two analytic
requirements of the interface: ``sum_{x>0} delta_s n_x dV = 2 R`` (the vector
form of ``int delta_s dn = 1``) and the cut-force identity ``F_cut = 2 sigma``.
See ``docs/verification_delta_s_normalization.md``.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import (
    kernel_sum_color,
    kernel_sum_gradient,
    surface_tension_acceleration,
)
from tensorsph.droplet_impact import Config, validate_config
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "delta_s_normalization"
TOL15 = 0.15


def _load_run_module():
    spec = importlib.util.spec_from_file_location("ds_run", BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _droplet(radius=0.02, dx=0.004, rho0=1000.0):
    dy = dx * math.sqrt(3.0) / 2.0
    mass = rho0 * dx * dy
    pts = []
    j = 0
    y = -radius
    while y <= radius:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -radius + offset
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), mass, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# the estimator itself
# --------------------------------------------------------------------------- #
def test_kernel_sum_gradient_is_the_exact_derivative_of_the_colour():
    """dc_i/dx_i = sum_j V_j grad_i W_ij, checked against finite differences."""
    rho0, dx = 1000.0, 0.005
    h = 1.3 * dx
    pos, mass = _droplet(dx=dx, rho0=rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    grad = kernel_sum_gradient(pos, mass, pairs, h, rho0)

    eps = 1.0e-7
    for idx in (0, pos.shape[0] // 3, pos.shape[0] // 2, pos.shape[0] - 2):
        num = torch.zeros(2, dtype=pos.dtype)
        for axis in range(2):
            p_plus = pos.clone()
            p_plus[idx, axis] += eps
            p_minus = pos.clone()
            p_minus[idx, axis] -= eps
            c_plus = kernel_sum_color(p_plus, mass, pairs, h, rho0)[idx]
            c_minus = kernel_sum_color(p_minus, mass, pairs, h, rho0)[idx]
            num[axis] = (c_plus - c_minus) / (2.0 * eps)
        assert torch.allclose(grad[idx], num, rtol=1e-4, atol=1e-4), (
            f"particle {idx}: analytic {grad[idx].tolist()} vs FD {num.tolist()}")


def test_kernel_sum_gradient_of_a_symmetric_stencil_vanishes():
    """Interior particles sit in a symmetric stencil: no spurious gradient."""
    rho0, dx = 1000.0, 0.005
    h = 1.3 * dx
    pos, mass = _droplet(dx=dx, rho0=rho0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    grad = kernel_sum_gradient(pos, mass, pairs, h, rho0)
    r = torch.linalg.vector_norm(pos, dim=1)
    interior = r < 0.5 * 0.02
    assert bool(interior.any())
    assert float(torch.linalg.vector_norm(grad[interior], dim=1).max()) < 1e-6


def test_bulk_traction_is_still_exactly_zero_for_both_schemes():
    """Any scheme must leave the momentum of the bulk untouched, bit for bit."""
    rho0, dx = 1000.0, 0.0025
    h = 1.3 * dx
    pos, mass = _droplet(dx=dx, rho0=rho0)
    pairs = cell_linked_pairs(pos, 4.0 * h)
    rho = torch.full_like(mass, rho0)
    acc0, _ = surface_tension_acceleration(pos, rho, mass, pairs, h, 7.2,
                                           rho0=rho0, delta_s_scheme="two_sided")
    r = torch.linalg.vector_norm(pos, dim=1)
    interior = r < 0.5 * 0.02
    for scheme in ("two_sided", "kernel_sum"):
        for fd in ("local", "reference"):
            acc, _ = surface_tension_acceleration(
                pos, rho, mass, pairs, h, 7.2, rho0=rho0, delta_s_scheme=scheme,
                force_density=fd)
            assert torch.equal(acc[interior], torch.zeros_like(acc[interior]))
    # and no surface tension at all is a no-op
    acc_off, _ = surface_tension_acceleration(pos, rho, mass, pairs, h, 0.0,
                                              rho0=rho0)
    assert torch.equal(acc_off, torch.zeros_like(acc_off))


def test_config_rejects_unknown_scheme_names():
    with pytest.raises(ValueError, match="csf_delta_s"):
        validate_config(Config(dx=0.005, dt=1.0e-5, csf_delta_s="bogus"))
    with pytest.raises(ValueError, match="csf_force_density"):
        validate_config(Config(dx=0.005, dt=1.0e-5, csf_force_density="bogus"))
    # a valid pair of names must pass (dt chosen inside the acoustic limit)
    validate_config(Config(dx=0.005, dt=1.0e-5, csf_delta_s="kernel_sum",
                           csf_force_density="reference"))


# --------------------------------------------------------------------------- #
# the measurement (small live case, a few seconds)
# --------------------------------------------------------------------------- #
def test_live_kernel_sum_scheme_normalises_the_delta_function():
    run = _load_run_module()
    base = run.instant(dx=0.002, delta_s="two_sided", force_density="local")
    new = run.instant(dx=0.002, delta_s="kernel_sum", force_density="reference")
    # int delta_s dn: the historical estimator loses ~25 %, the exact-gradient
    # one lands inside the platform gate
    assert base["delta_s"]["vector_identity"] < 0.80
    assert abs(new["delta_s"]["vector_identity"] - 1.0) < TOL15
    assert new["delta_s"]["vector_identity"] > base["delta_s"]["vector_identity"]
    # cut-force identity with the corrected delta_s *and* the reference density
    assert abs(new["cut_force"]["ratio_measured_over_theory"] - 1.0) < TOL15
    # the historical combination only passed through error cancellation
    base_ref = run.instant(dx=0.002, delta_s="two_sided", force_density="reference")
    assert abs(base_ref["cut_force"]["ratio_measured_over_theory"] - 1.0) > TOL15
    # direction is untouched by the delta_s choice
    assert new["direction"]["cos_band_median"] > 0.9


# --------------------------------------------------------------------------- #
# committed record (no compute): the honest verdicts must stay in the JSON
# --------------------------------------------------------------------------- #
def test_committed_record_keeps_the_honest_verdicts():
    rep = json.loads((BENCH / "validation_report.json").read_text())
    rows = {(r["delta_s_scheme"], r["force_density"], round(r["dx_m"], 5)): r
            for r in rep["instant"]}

    # 1. the delta function normalisation is fixed and the old deficit recorded
    for dx in (0.003, 0.002, 0.0015):
        old = rows[("two_sided", "local", round(dx, 5))]
        new = rows[("kernel_sum", "reference", round(dx, 5))]
        assert old["delta_s"]["vector_rel_error_pct"] < -20.0, dx
        assert abs(new["delta_s"]["vector_rel_error_pct"]) < 15.0, dx

    # 2. intended resolution window of the cut-force identity (R/h >= 7.7)
    for dx in (0.002, 0.0015):
        new = rows[("kernel_sum", "reference", round(dx, 5))]
        assert abs(new["cut_force"]["ratio_measured_over_theory"] - 1.0) < TOL15
        assert abs(new["kappa"]["rel_error_pct"]) < 15.0
        # the historical pass was a cancellation of -32 % delta_s and +35 % 1/rho
        old = rows[("two_sided", "local", round(dx, 5))]
        assert abs(old["cut_force"]["ratio_measured_over_theory"] - 1.0) < TOL15
        assert abs(old["interface_density_ratio"] - 1.0) > 0.10

    # 3. the coarse resolution (R/h ~ 5) is *not* claimed: it stays a failure
    coarse = rows[("kernel_sum", "reference", 0.003)]
    assert abs(coarse["cut_force"]["ratio_measured_over_theory"] - 1.0) > 0.15
    assert abs(coarse["kappa"]["rel_error_pct"]) > 15.0

    # 4. spurious currents: reported, not tuned -- one combo is above Adami
    passes = [r["spurious_current"]["passes_adami_0p1"] for r in rep["relax"]]
    assert len(passes) == 4
    assert not all(passes), "the measured Adami violation must stay in the record"