"""The controls behind the three PR #309 relaxations (mission m4817, round 3).

Background: PR #309 was closed, its substance landed through m4659/m4793 as three
relaxed assertions -- ``|bias| < 1e-4`` in ``test_kernel_sum_consistency.py``,
``rtol=2e-4, atol=2e-4`` in ``test_wall_pressure.py`` and an explicit
``density_calibration`` flag in ``test_initial_density_calibration.py``.  A relaxed
assertion is a claim that the *strict* one was measuring the wrong thing.  Those
claims lived in prose in ``docs/mission_m4659_fix_summary.md``; this file makes
them machine-checkable, one control per file, each with a **failing side** so the
gate cannot be quietly widened later:

* A reduction control -- the 1e-4 kernel-sum gate must be a property of the
  lattice, not of the summation tree (float32 vs float64, natural vs permuted
  order).  Measured spread: 1.13e-7, i.e. 0.11 % of the gate.
* A negative control on the wall-pressure tolerance -- the pre-#309 gate
  (``rtol=1e-5, atol=1e-6``) must be *red* on the two-term state that motivated
  the relaxation and *green* on the body-force-only state, so the relaxation is a
  correction rather than a blanket weakening.
* A positive control -- the shipped tolerance must still be red on a 5 % error.
* The calibration default must be load bearing where the tension clamp cannot
  help (a positive-bias lattice keeps its spurious pressure).

Numbers quoted here come from ``docs/reports/pr309_round3_probe.json``
(``pr309_round3_probe.py``); ``test_probe_json_agrees_with_this_tree`` re-derives
them live, so the document cannot drift away from the code it describes.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config, make_particles, pool_lattice, rhs
from tensorsph.initial_density import (
    calibration_factor,
    lattice_bias,
    spurious_pressure_ratio,
)
from tensorsph.lattice_consistency import interior_mask, kernel_sum_factor

from test_wall_pressure import SMALL, closed_form, two_particle_state

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "docs" / "reports" / "pr309_round3_probe.json"

GATE = 1.0e-4                 # m2971 acceptance gate, the merged assertion
OLD_PIN = -1.19e-7            # the digit PR #309 tried to restore
STRICT = dict(rel_tol=1.0e-5, abs_tol=1.0e-6)     # gate before PR #309
SHIPPED = dict(rel_tol=2.0e-4, abs_tol=2.0e-4)    # gate on main since m4635
DX, H_FACTOR = 0.01, 1.3


def _shipped_pool():
    """The pool ``make_particles`` ships: square lattice, h = 1.3 dx."""
    cfg = Config(dx=DX, h_factor=H_FACTOR, density_calibration=False)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    n_pool = int(pool_lattice(cfg, torch.device("cpu")).shape[0])
    return cfg, pos[:n_pool], mass[:n_pool], H_FACTOR * DX


@lru_cache(maxsize=1)
def _kernel_sum_variants() -> dict:
    """Mean kernel-sum bias of one lattice over four reductions."""
    cfg, pos, mass, h = _shipped_pool()
    # Permute *inside the pool*: mixing drop particles into the prefix would put
    # free-surface particles under the interior mask -- the -1.09e-2 "masked
    # trap" of docs/mission_m4659_fix_summary.md, not a bias.
    perm = torch.randperm(pos.shape[0], generator=torch.Generator().manual_seed(4659))

    def bias(dtype, order=None):
        p, m = pos.to(dtype), mass.to(dtype)
        c = kernel_sum_factor(p, m / cfg.rho0, h)
        if order is not None:
            c, p = c[order], p[order]
        return float(c[interior_mask(p, h)].mean()) - 1.0

    return {
        "fp32": bias(torch.float32),
        "fp32_permuted": bias(torch.float32, perm),
        "fp64": bias(torch.float64),
        "fp64_permuted": bias(torch.float64, perm),
    }


# --------------------------------------------------------------------------- #
# A. the kernel-sum gate is a lattice property
# --------------------------------------------------------------------------- #
def test_kernel_sum_gate_is_a_lattice_property_not_a_reduction_artefact():
    """Changing precision *and* summation order moves the bias by <1 % of the gate.

    This is what licenses replacing PR #309's digit pin by ``abs(bias) < 1e-4``:
    the fifth digit is a property of the reduction, the magnitude is a property
    of the lattice.  Measured spread 1.13e-7 (0.11 % of the gate).
    """
    v = _kernel_sum_variants()
    spread = max(v.values()) - min(v.values())
    assert spread < 0.01 * GATE, v                 # measured 1.1e-3 * GATE
    for name, b in v.items():
        assert abs(b) < GATE, (name, b)            # so all four pass the gate
        assert b < 0.0, (name, b)                  # tension side, hence 0 Pa


def test_the_restored_digit_pin_would_be_red_on_this_tree():
    """``-1.19e-7`` is a property of the db0c3265 tree, not of the lattice.

    Measured here at 446x the pinned value, i.e. restoring PR #309's pin verbatim
    would have reddened main -- which is why m4659 replaced it by gate + sign.
    """
    b = _kernel_sum_variants()["fp32"]
    assert abs(b) > 100.0 * abs(OLD_PIN), abs(b)


# --------------------------------------------------------------------------- #
# B. the wall-pressure tolerance: red where it must be, green where it must be
# --------------------------------------------------------------------------- #
def _wall_acceleration(mass_scale: float) -> tuple[float, float]:
    cfg = Config(**SMALL)
    pos, wall, mass = two_particle_state(cfg, mass_scale)
    _, _, acc = rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall)
    expected = closed_form(cfg, mass_scale)[1]
    return float(acc[0, 1]), float(expected[1])


def test_the_pre_309_wall_tolerance_is_red_only_where_it_should_be():
    """Negative control: the old gate must fail the two-term state.

    Body-force-only (mass_scale 0.8, 5.32e-6) passes ``rtol=1e-5``; adding the
    extrapolation term (mass_scale 3.0, 1.39e-5) does not.  A relaxation that
    failed *both* would be a weakening; this one is a correction.
    """
    a0, e0 = _wall_acceleration(0.8)
    a3, e3 = _wall_acceleration(3.0)
    assert math.isclose(a0, e0, **STRICT)             # green before #309
    assert not math.isclose(a3, e3, **STRICT)         # red before #309
    assert math.isclose(a0, e0, **SHIPPED)
    assert math.isclose(a3, e3, **SHIPPED)
    # and the shipped gate still leaves a double-digit margin, i.e. it is not
    # sitting on the measured error (measured 1.39e-5 -> 14.4x margin)
    assert SHIPPED["rel_tol"] > 10.0 * abs(a3 - e3) / abs(e3)


def test_the_shipped_wall_tolerance_still_detects_a_real_regression():
    """Positive control: a 5 % error must be red, so the gate has teeth."""
    a, e = _wall_acceleration(3.0)
    assert not math.isclose(1.05 * a, e, **SHIPPED)
    assert not math.isclose(0.95 * a, e, **SHIPPED)


# --------------------------------------------------------------------------- #
# C. the calibration default: load bearing where the clamp cannot help
# --------------------------------------------------------------------------- #
def test_calibration_default_is_load_bearing_on_a_positive_bias_lattice():
    """The square pool is rescued by the clamp, a hex pool is not.

    ``Config().density_calibration`` defaults to True.  On the shipped square
    lattice that default is free -- the raw bias is negative, so the tension
    clamp turns it into 0 Pa either way.  On a hex lattice at the same h/dx the
    raw bias is +5.88e-3, the clamp cannot hide it and the spurious pressure is
    ~2.4 kPa at c0 = 20 m/s: there the default is the difference between a quiet
    and a ringing start.
    """
    cfg, _, _, h = _shipped_pool()
    assert Config().density_calibration is True

    def ratio(bias: float) -> float:
        rho = torch.full((8,), cfg.rho0 * (1.0 + bias), dtype=torch.float64)
        return spurious_pressure_ratio(rho, cfg.rho0, 20.0)

    square_raw = lattice_bias(cfg.dx, h, kind="square")
    hex_raw = lattice_bias(cfg.dx, h, kind="hex")
    factor = calibration_factor(cfg.dx, h, kind="hex")

    assert square_raw < 0.0 and ratio(square_raw) == 0.0      # clamped away
    assert hex_raw > 0.0 and ratio(hex_raw) > GATE            # not clamped away
    assert ratio(hex_raw) > 10.0 * GATE
    calibrated = (1.0 + hex_raw) * factor - 1.0
    assert abs(calibrated) < 1e-6                             # gate met
    assert ratio(calibrated) < GATE


# --------------------------------------------------------------------------- #
# the document's numbers are re-derived, not trusted
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not PROBE.exists(), reason="probe JSON not committed")
def test_probe_json_agrees_with_this_tree():
    """The committed probe must still describe the tree that reads it."""
    probe = json.loads(PROBE.read_text())
    a = probe["control_a_kernel_sum"]
    live = _kernel_sum_variants()["fp32"]
    assert abs(live - a["bias_by_variant"]["fp32_as_shipped"]) < 1e-8
    assert a["all_inside_gate"] is True and a["all_same_sign"] is True

    b = probe["control_b_wall_pressure"]
    a3, e3 = _wall_acceleration(3.0)
    assert abs(a3 - b["rows"][1]["acc_rhs"]) < 1e-5
    assert b["old_gate_red_on_two_term"] is True
    assert b["old_gate_green_on_body_only"] is True

    c = probe["control_c_initial_density"]
    cfg, _, _, h = _shipped_pool()
    assert abs(lattice_bias(cfg.dx, h, kind="hex")
               - c["hex_same_h"]["bias_raw"]) < 1e-12
    assert c["density_calibration_default"] is True
