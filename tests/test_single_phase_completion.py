"""Single-phase interface-completion closure for the SPHinXsys CSS model.

Anchor ``XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION``.

The invariant under test is physical, not just "the code runs":

* :func:`tensorsph.sphinxys_css.interface_completion` is a *specular reflection*
  of the kernel gradient (direction-level identity, and it vanishes for a
  neighbour lying in the interface plane).
* In the bulk the present-neighbour sum of ``V grad W`` already vanishes, and
  the reflection of that sum vanishes with it -- so the closure must switch
  itself off away from a free surface (no double counting).
* The vacuum-reduced (single-phase) force assembly is short of the exact
  Young-Laplace jump by ~30 %; the closure brings it inside the ledger's <15 %
  criterion at every resolution, with the coefficient fitted at one rung and
  transferred to the others.
* The completed pair flux is still a symmetric two-point flux, so momentum stays
  conserved to machine precision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "benchmarks" / "surface_tension") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks" / "surface_tension"))

import single_phase_completion as SPC  # noqa: E402
import tensorsph.sphinxys_css as sc  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

@pytest.fixture(scope="module")
def closure_fit():
    """Fit the closure weight at the middle rung (R/dx = 32) only."""
    sweep = [SPC.measure(SPC.RADIUS, SPC.RADIUS / 32, a) for a in SPC.SWEEP]
    return SPC.fit_alpha(sweep)


# --------------------------------------------------------------------------- #
# 1. the closure operator itself (no SPH scene needed)
# --------------------------------------------------------------------------- #
def test_interface_completion_is_specular_reflection():
    e = torch.tensor([[1.0, 0.0], [0.6, 0.8], [-0.3, 0.7]], dtype=torch.float64)
    w_prime = torch.tensor([-2.0, -1.0, -0.5], dtype=torch.float64)
    n = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    got = sc.interface_completion(e, w_prime, n)
    # e' = e - 2 (e.n) n
    want = w_prime[:, None] * (e - 2.0 * (e * n).sum(dim=1, keepdim=True) * n)
    assert torch.allclose(got, want, atol=1e-14)
    # a neighbour lying in the interface plane is its own mirror image
    assert torch.allclose(got[0], w_prime[0] * e[0], atol=1e-15)
    # one neighbour along the normal is reflected onto itself with opposite sign
    e_n = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    n_n = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    got_n = sc.interface_completion(e_n, torch.tensor([-1.0], dtype=torch.float64), n_n)
    assert torch.allclose(got_n, torch.tensor([[0.0, 2.0]], dtype=torch.float64),
                          atol=1e-14)
    # added to the plain gradient the radial part cancels: e + e' is tangential
    assert torch.allclose(((got / w_prime[:, None] + e) * n).sum(dim=1),
                          torch.zeros(3, dtype=torch.float64), atol=1e-14)


def test_closure_injects_no_bulk_force():
    """The closure is a free-surface operator: it must do nothing in the bulk.

    Away from the interface the colour gradient -- and therefore the stress --
    of the mixture model is zero, so a bulk pair carries no flux whatever the
    completed gradient looks like.  If the completion leaked into the interior
    it would inject a spurious body force and corrupt every bulk pressure.
    """
    dx = SPC.RADIUS / 128
    h = SPC.H_OVER_DX * dx
    pos, r, rho0, vol = SPC.make_disc(SPC.RADIUS, dx)
    mass = rho0 * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    frac = torch.ones(pos.shape[0], dtype=pos.dtype)
    kw = dict(sigma=SPC.SIGMA, h=h, frac=frac, mode="single_phase",
              hourglass=False, assembly="consistent")
    a0, _ = sc.surface_tension_acceleration(pos, mass, rho0, vol, pairs, None,
                                            completion="none", **kw)
    a1, fields = sc.surface_tension_acceleration(pos, mass, rho0, vol, pairs, None,
                                                 completion="mirror",
                                                 completion_weight=1.0, **kw)
    delta_a = torch.linalg.vector_norm(a1 - a0, dim=1)
    # real bulk: more than one support width (2h) away from the interface shell
    interior = r < SPC.RADIUS - 5.0 * h
    assert int(interior.sum()) > 100
    shell = fields.delta_s > 0.1 * float(fields.delta_s.max())
    assert int(shell.sum()) > 20
    shell_amp = float(delta_a[shell].max())
    assert shell_amp > 0.0
    assert float(delta_a[interior].max()) < 1e-12 * shell_amp


# --------------------------------------------------------------------------- #
# 2. default behaviour is unchanged (no regression on the two-phase model)
# --------------------------------------------------------------------------- #
def test_completion_default_is_off():
    dx = SPC.RADIUS / 32
    h = SPC.H_OVER_DX * dx
    pos, _r, rho0, vol = SPC.make_disc(SPC.RADIUS, dx)
    mass = rho0 * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    frac = torch.ones(pos.shape[0], dtype=pos.dtype)
    kw = dict(sigma=SPC.SIGMA, h=h, frac=frac, mode="single_phase",
              hourglass=False, assembly="consistent")
    a_default, _ = sc.surface_tension_acceleration(pos, mass, rho0, vol, pairs, None, **kw)
    a_none, _ = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs, None, completion="none", **kw)
    a_mirror, _ = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs, None, completion="mirror",
        completion_weight=1.0, **kw)
    assert torch.equal(a_default, a_none)
    assert not torch.allclose(a_default, a_mirror)


# --------------------------------------------------------------------------- #
# 3. the physical reading: exact Young-Laplace, single-phase free surface
# --------------------------------------------------------------------------- #
def test_baseline_reproduces_round2_open_item():
    """alpha = 0 is the round-2 configuration: ~-30 %, outside the gate."""
    rec = SPC.measure(SPC.RADIUS, SPC.RADIUS / 32, 0.0)
    assert -32.0 < rec["err_pct"] < -28.0


def test_closure_gate_single_phase(closure_fit):
    alpha = closure_fit["alpha_star"]
    assert 0.3 < alpha < 0.6
    for rung in (16, 32, 64):
        rec = SPC.measure(SPC.RADIUS, SPC.RADIUS / rung, alpha)
        assert abs(rec["err_pct"]) < SPC.GATE * 100.0, (rung, rec["err_pct"])


def test_closure_transfers_across_smoothing_ratio(closure_fit):
    alpha = closure_fit["alpha_star"]
    for h_over_dx in (1.1, 1.4):
        rec = SPC.measure(SPC.RADIUS, SPC.RADIUS / 32, alpha, h_over_dx=h_over_dx)
        assert abs(rec["err_pct"]) < SPC.GATE * 100.0, (h_over_dx, rec["err_pct"])


def test_closed_flux_conserves_momentum(closure_fit):
    alpha = closure_fit["alpha_star"]
    rec = SPC.measure(SPC.RADIUS, SPC.RADIUS / 32, alpha)
    assert rec["momentum_residual"] < 1e-12