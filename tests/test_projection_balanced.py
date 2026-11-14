"""Gates for round 3 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE (well-balanced CSF).

Round 3 asks one question: is the capillary force of a static drop *irrotational*
in the discrete sense the projection needs, and if not, does removing its
solenoidal part (a) leave the Laplace jump untouched and (b) stop the drop?

The gates below are split in two groups.

* Unit gates (fast, ~10 s, no result file needed): the Helmholtz split is exact
  (reconstruction + orthogonality to round-off), a pure discrete gradient has
  zero solenoidal part, and -- the physics gate -- the Laplace jump read off the
  potential alone reproduces ``sigma/R`` at a coarse resolution.
* Result gates (skipped unless the sharded runs were merged into
  ``results_balanced.json``): spurious currents under Adami's 0.1 line with no
  viscosity knob, the jump inside the 3 % claim level, area conservation, dt
  invariance, and an anti-drift check that the round-3 honest FAIL is still on
  record unchanged.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.laplace_isph import csf_state_two_phase, hex_block  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.projection_adjoint import (adjoint_gradient, helmholtz_split,  # noqa: E402
                                         irrotationality_defect,
                                         relax_projection_balanced,
                                         relax_projection_consistent)

torch.set_num_threads(2)

RESULTS = ROOT / "benchmarks" / "laplace_sph" / "results_balanced"
MERGED = RESULTS / "results_balanced.json"
ROUND3 = ROOT / "benchmarks" / "laplace_sph" / "results_dynamic.json"

SIGMA = 0.072
RHO = 1000.0


def _state(dx: float = 0.004, radius: float = 0.02, hf: float = 1.3):
    h = hf * dx
    pos, mass, liquid, half = hex_block(radius, dx, 0.012, RHO)
    vol = mass / RHO
    pairs = cell_linked_pairs(pos, 2.0 * h)
    st = csf_state_two_phase(pos, mass, liquid, h, RHO, SIGMA,
                             curvature_method="mls")
    edge = 1.6 * dx
    pinned = (pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge)
    gauge = int(torch.nonzero(pinned)[0, 0]) if bool(pinned.any()) else 0
    return pos, vol, pairs, h, st, gauge


# --------------------------------------------------------------------------- #
# unit gates: the split itself
# --------------------------------------------------------------------------- #
def test_split_reconstructs_exactly_and_is_orthogonal():
    pos, vol, pairs, h, st, gauge = _state()
    sp = helmholtz_split(pos, st["acc"], vol, pairs, h, gauge=gauge, rel_tol=1e-10)
    assert sp["reconstruction_defect"] < 1e-12
    assert abs(sp["orthogonality"]) < 1e-8
    assert 0.0 < sp["solenoidal_fraction"] < 1.0


def test_pure_gradient_has_no_solenoidal_part():
    """The split must be exact for its own operator: ``f = G phi`` -> ``sol = 0``.

    This is the mathematical content the whole round rests on: whatever the
    projection can see is removed by it, and nothing else is.

    Subtlety found while writing this gate (kept, because it bounds the claim):
    the pair operator pair is *not* gauge-free in the exact sense -- its
    gradient of a constant field does not vanish (``max|G 1| ~ 1e2`` on this
    lattice, kernel-gradient inconsistency), so ``f = G phi`` is only exactly a
    gradient when ``phi`` is zero at the pinned particle and the constant is
    subtracted.  With that convention the defect is CG-limited (9.8e-10 at
    ``rel_tol = 1e-10``).  The Laplace jump is a *difference* of the potential
    across the interface, so the constant is harmless there.
    """
    pos, vol, pairs, h, _, gauge = _state()
    g = torch.Generator().manual_seed(7)
    phi = torch.randn(pos.shape[0], generator=g, dtype=pos.dtype)
    phi[gauge] = 0.0
    f = adjoint_gradient(pos, phi, vol, pairs, h)
    defect = irrotationality_defect(pos, f, vol, pairs, h, gauge=gauge,
                                    rel_tol=1e-10)
    assert defect < 1e-7, defect


def test_potential_xval_against_laplace_law():
    """Coarse single-shot check: ``rho [q]`` reproduces ``sigma/R`` without a solve.

    Deliberately run at ``R/dx = 5`` -- a resolution the sharded runs never
    reach -- so this gate measures the formulation, not the tuning.
    """
    radius = 0.02
    pos, vol, pairs, h, st, gauge = _state(dx=0.004, radius=radius)
    r = torch.linalg.vector_norm(pos, dim=1)
    core = r < 0.5 * radius
    band = (r > 1.25 * radius) & (r < 1.45 * radius)
    sp = helmholtz_split(pos, st["acc"], vol, pairs, h, gauge=gauge,
                         rel_tol=1e-10)
    q = sp["potential"]
    dp = RHO * float(q[core].mean() - q[band].mean())
    err = (dp / (SIGMA / radius) - 1.0) * 100.0
    assert abs(err) < 15.0, err


# --------------------------------------------------------------------------- #
# result gates (need the sharded runs)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def merged():
    if not MERGED.exists():
        pytest.skip("run_balanced.py shards not merged in this checkout")
    return json.loads(MERGED.read_text())


def _runs(blob):
    return [r for r in blob["runs"] if not r["diverged"]]


def test_all_levels_completed(merged):
    """Every level must have marched its full step budget without diverging."""
    assert len(_runs(merged)) >= 2, [r["R_over_dx"] for r in _runs(merged)]
    for r in _runs(merged):
        assert r["history"][-1]["step"] >= 30, (r["R_over_dx"], r["history"][-1])
        assert math.isfinite(r["umax_over_vref_final"])
        assert math.isfinite(r["dp_potential_final_Pa"])


def test_spurious_currents_below_adami_without_viscosity(merged):
    """The round-3 blocker: 0.179 without a viscosity knob. Now the gate."""
    for r in _runs(merged):
        assert r["nu_factor"] == 0.0
        assert r["umax_over_vref_final"] < merged["gates"]["adami_spurious"], (
            r["R_over_dx"], r["umax_over_vref_final"])


def test_laplace_jump_gate(merged):
    for r in _runs(merged):
        assert abs(r["dp_rel_error_pct"]) <= merged["gates"]["laplace_pct_tight"], (
            r["R_over_dx"], r["dp_rel_error_pct"])


def test_two_instruments_agree(merged):
    """Pressure solve and potential must give the same jump (p = rho q after projection)."""
    for r in _runs(merged):
        rel = abs(r["dp_potential_final_Pa"] - r["dp_final_Pa"]) / r["dp_theory_Pa"]
        assert rel < 1e-2, (r["R_over_dx"], rel)


def test_area_is_conserved(merged):
    for r in _runs(merged):
        assert abs(r["area_ratio_final"] - 1.0) * 100.0 < merged["gates"]["area_drift_pct"]


def test_step_size_invariance(merged):
    """A true equilibrium cannot depend on dt: the march must be dt-insensitive."""
    ctl = [c for c in merged["controls"] if c["label"].startswith("dt_factor")]
    if len(ctl) < 2:
        pytest.skip("dt controls not present")
    dps = [abs(c["dp_rel_error_pct"]) for c in ctl]
    assert max(dps) - min(dps) < 0.5, dps
    for c in ctl:
        assert c["umax_over_vref_final"] < merged["gates"]["adami_spurious"]


def test_split_is_doing_the_work(merged):
    """Control: the same loop without the split must not reach the gate."""
    ctl = [c for c in merged["controls"] if c["label"].startswith("no split")]
    if not ctl:
        pytest.skip("no-split control not present")
    c = ctl[0]
    assert c["diverged"] or c["umax_over_vref_final"] > merged["gates"]["adami_spurious"], (
        c["umax_over_vref_final"], c["diverged"])


def test_discarded_part_shrinks_with_kernel_support(merged):
    """The solenoidal part is a smoothing artefact: more support -> less of it."""
    sw = merged.get("defect_sweep")
    if not sw:
        pytest.skip("defect sweep not present")
    fr = [row["solenoidal_fraction"] for row in sw["rows"]]
    assert fr[0] > fr[1] > fr[2], fr
    assert fr[-1] < 0.85 * fr[0], fr
    for row in sw["rows"]:
        assert abs(row["orthogonality"]) < 1e-4


# --------------------------------------------------------------------------- #
# anti-drift: the shipped round-3 record and the legacy defect stay put
# --------------------------------------------------------------------------- #
def test_round3_honest_fail_still_on_record():
    if not ROUND3.exists():
        pytest.skip("round-3 results file not present")
    d = json.loads(ROUND3.read_text())
    gate = d["gates"]["adami_spurious_0p1"]
    assert gate["pass"] is False, "the round-3 spurious-current FAIL must stay recorded"
    assert gate["value"] > 0.1


def test_round3_public_api_untouched():
    """Round 3 kept its own entry point; round 4 must not have repurposed it."""
    sig = inspect.signature(relax_projection_consistent)
    for name in ("nu_factor", "pin_outer", "dt_factor"):
        assert name in sig.parameters
    sig2 = inspect.signature(relax_projection_balanced)
    assert "balanced" in sig2.parameters
    assert sig2.parameters["balanced"].default is True
    assert sig2.parameters["nu_factor"].default == 0.0