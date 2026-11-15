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

# --------------------------------------------------------------------------- #
# 4. the document is derived from gate.json, and it points at files that exist
# --------------------------------------------------------------------------- #
def test_document_is_rendered_from_gate_json():
    """The XVAL document must be a function of gate.json, not a hand copy.

    ``emit_single_phase_tables.py`` renders the whole document from
    ``out_single_phase_completion/gate.json``.  If the CSS model, the ladder or
    the closure changes and ``gate.json`` is regenerated, the document has to be
    re-emitted; a hand-edited (or stale) document is a documentation defect that
    silently survives review.  This test makes that drift a failure: it
    re-renders in memory and demands a byte-identical match.
    """
    import emit_single_phase_tables as EMIT

    rendered, verdict = EMIT.render_doc()
    committed = EMIT.DOC.read_text()
    assert verdict == "pass"
    if rendered != committed:
        import difflib
        diff = "".join(difflib.unified_diff(
            committed.splitlines(True), rendered.splitlines(True),
            fromfile="committed", tofile="render(gate.json)", n=2))
        pytest.fail("document is stale vs gate.json; run "
                    "`emit_single_phase_tables.py --write`:\n" + diff[:4000])


def test_document_artifact_paths_exist():
    """Every repository path the document advertises must be present.

    The task that produced this document shipped its tables generator under
    ``runs/``; ``runs/`` was not carried by the merge gate, so the published
    document ended up pointing at a generator and a figure that do not exist in
    the repository.  A dangling artifact table is exactly the kind of "done"
    that is not done, so it is locked here.
    """
    import re

    doc = (ROOT / "docs" / "xval_sphinxys_mixture_laplace"
           / "single_phase_completion.md").read_text()
    roots = ("src/", "benchmarks/", "tests/", "docs/")
    figures = (".png", ".svg", ".pdf", ".jpg")
    refs = {m for m in re.findall(r"`([^`\n]+)`", doc)
            if m.startswith(roots) and "*" not in m and "<" not in m
            and not m.endswith(figures)}
    assert refs, "no artifact paths found in the document"
    # a reference may pin a pytest node id (``file.py::test_name``)
    missing = sorted(m for m in refs if not (ROOT / m.split("::")[0]).exists())
    assert not missing, missing
    # and the document must not resurrect the un-merged runs/ generator
    assert "runs/m20035" not in doc


# --------------------------------------------------------------------------- #
# 5. gate.json is still the output of *this* benchmark, and the figure draws
# --------------------------------------------------------------------------- #
def test_gate_json_is_self_consistent_with_the_benchmark():
    """``gate.json`` must come from the current benchmark, not from an older run.

    The document is only as trustworthy as the JSON it is rendered from, so the
    JSON is pinned to the source: the case constants, the gate, the ladder and
    the fitted closure coefficient must all be reproducible from the module.  A
    changed ``SIGMA`` / ``H_OVER_DX`` / ladder without a re-run is the classic
    way a "zero hand-copy" pipeline still ends up publishing stale numbers.
    """
    import json

    P = json.loads((SPC.OUT / "gate.json").read_text())
    case = P["case"]
    assert case["sigma_nm"] == SPC.SIGMA
    assert case["radius_m"] == SPC.RADIUS
    assert case["rho_water"] == SPC.RHO_WATER
    assert case["h_over_dx"] == SPC.H_OVER_DX
    assert P["gate_pct"] == SPC.GATE * 100.0
    assert P["fit_rung"] == SPC.FIT_RUNG
    assert P["ladder"] == list(SPC.LADDER)
    # the fitted coefficient is a function of the stored sweep, nothing else
    refit = SPC.fit_alpha(P["sweep"])
    assert refit["alpha_star"] == pytest.approx(P["fit"]["alpha_star"], abs=1e-12)
    assert refit["bracket"] == P["fit"]["bracket"]


def test_figure_renders(tmp_path):
    """The 3-panel figure is committed as code, so it must still draw.

    Images are not carried by the merge gate, therefore the deliverable is the
    drawing code plus the numbers, and this test is what keeps the figure
    reproducible: it renders the panels from the *committed* ``gate.json`` and
    checks the three panel titles are present.
    """
    import json

    P = json.loads((SPC.OUT / "gate.json").read_text())
    target = tmp_path / "gate.svg"
    SPC.plot(P, target)
    svg = target.read_text()
    assert len(svg) > 10_000
    for title in ("closure scan", "baseline vs closure", "smoothing-width transfer"):
        assert title in svg, title
