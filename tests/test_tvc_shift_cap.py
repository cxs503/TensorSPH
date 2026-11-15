"""Checks for the ``tvc_max_shift`` leg (``SHIFT_CAP_LEG.md``).

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

Two kinds of assertions:

* **operator-level** (fast, no time integration) -- the cap is a radial rescale
  and the free-surface displacement scale really is ``30 alpha / (7 pi) h``,
  which is the number the whole leg is derived from;
* **payload-level** -- the hypotheses of ``benchmarks/xval_sphinx_tvc/shift_cap.py``
  are graded against the committed payload, so a re-run with new resolutions
  cannot silently contradict the document (the document is generated from the
  same payload and re-grades itself).
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

from tensorsph.transport_velocity_correction import (  # noqa: E402
    transport_velocity_correction)

DATA = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "shift_cap.json"
SURFACE_FACTOR = 30.0 / (7.0 * math.pi)


def slab(nx: int = 60, ny: int = 10, dx: float = 0.05, h_over_dx: float = 1.3):
    """A half-plane of particles (open top, wall-free sides)."""
    xs = torch.arange(nx, dtype=torch.float64) * dx
    ys = torch.arange(ny, dtype=torch.float64) * dx
    pos = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
    vol = torch.full((pos.shape[0],), dx ** 2, dtype=torch.float64)
    return pos, vol, h_over_dx * dx


# ---------------------------------------------------------------- operator leg


def test_surface_shift_scale_is_thirty_alpha_over_seven_pi():
    """``|delta r|/h = 30 alpha/(7 pi)`` on a half-plane (the leg's equation 1)."""
    # ``h/dx = 4``: the companion ``SURFACE_ANCHOR.md`` leg measures the
    # production ``h/dx = 1.3`` lattice as 14.15 % *below* the continuum value,
    # converging at local order 2, so the closed-form factor is tested where the
    # truncation bias is inside the 5 % band (measured: -1.5 %).
    pos, vol, h = slab(h_over_dx=4.0)
    dx = 0.05
    alpha = 0.10
    res = transport_velocity_correction(pos, vol, h, alpha=alpha)
    top = pos[:, 1].max()
    # the middle of the top row: far from both lateral ends
    mid = (pos[:, 1] == top) & (pos[:, 0] > 15 * dx) & (pos[:, 0] < 44 * dx)
    assert bool(mid.any())
    scale = float(torch.linalg.vector_norm(res.shift[mid], dim=1).mean()) / h
    predicted = SURFACE_FACTOR * alpha
    assert scale == pytest.approx(predicted, rel=0.05), (scale, predicted)
    # and it points outwards (up), the de-clumping direction
    assert float(res.shift[mid, 1].mean()) > 0.0


def test_cap_above_peak_is_bit_identical():
    pos, vol, h = slab()
    base = transport_velocity_correction(pos, vol, h, alpha=0.10)
    peak = float(torch.linalg.vector_norm(base.shift, dim=1).max()) / h
    capped = transport_velocity_correction(pos, vol, h, alpha=0.10,
                                           max_shift=1.05 * peak * h)
    assert capped.clipped == 0.0
    assert torch.equal(capped.shift, base.shift)


def test_cap_below_peak_rescales_only_the_over_cap_population():
    pos, vol, h = slab()
    alpha = 0.10
    base = transport_velocity_correction(pos, vol, h, alpha=alpha)
    cap = 0.5 * float(torch.linalg.vector_norm(base.shift, dim=1).max()) / h
    capped = transport_velocity_correction(pos, vol, h, alpha=alpha,
                                           max_shift=cap * h)
    norm = torch.linalg.vector_norm(capped.shift, dim=1) / h
    # never exceeds the cap ...
    assert float(norm.max()) <= cap * (1.0 + 1e-12)
    # ... the population actually touched is the one that was over the cap
    over = torch.linalg.vector_norm(base.shift, dim=1) / h > cap
    assert capped.clipped == pytest.approx(float(over.to(torch.float64).mean()))
    assert capped.clipped > 0.0
    # a capped shift keeps its direction (radial rescale, not a clip per axis)
    cos = torch.nn.functional.cosine_similarity(capped.shift[over],
                                                base.shift[over], dim=1)
    assert float(cos.min()) > 1.0 - 1e-9


def test_effective_coefficient_law_holds_on_the_operator_bulk_is_not_scaled():
    """Why H1 fails at solver level: the cap leaves sub-cap particles alone."""
    pos, vol, h = slab()
    alpha, alpha_eff = 0.30, 0.15
    cap = SURFACE_FACTOR * alpha_eff
    base_eff = transport_velocity_correction(pos, vol, h, alpha=alpha_eff)
    capped = transport_velocity_correction(pos, vol, h, alpha=alpha,
                                           max_shift=cap * h)
    n_base = torch.linalg.vector_norm(base_eff.shift, dim=1)
    n_cap = torch.linalg.vector_norm(capped.shift, dim=1)
    # keep the population that stays *below* the cap even at alpha, so the
    # comparison isolates the untouched bulk from the clipped surface tail
    under = ((n_base / h > 0.10 * cap) & (n_base / h < 0.45 * cap))
    assert bool(under.any()) and bool((~under).any())
    # particles the cap never touches keep alpha = 0.30, i.e. *twice* the
    # displacement of the alpha_eff operator -> the capped run is NOT an
    # alpha_eff run (this is the mechanism behind the H1 falsification)
    ratio = float((n_cap[under] / n_base[under]).median())
    assert ratio == pytest.approx(alpha / alpha_eff, rel=1e-6)


def test_dam_break_records_the_clipped_diagnostic():
    from tensorsph.dam_break import DamBreakConfig, run_case

    dx = 0.05
    dt = DamBreakConfig(dx=dx).dt_from_courant()
    common = dict(dx=dx, dt=dt, steps=30, output_every=30, device="cpu",
                  tvc_every=1, tvc_alpha=0.10)
    uncapped = run_case(DamBreakConfig(**common))
    capped = run_case(DamBreakConfig(**common, tvc_max_shift=0.05))
    assert uncapped.max_tvc_clipped == 0.0
    assert 0.0 < capped.max_tvc_clipped <= 1.0
    assert capped.max_tvc_shift <= 0.05 * (1.3 * dx) * (1.0 + 1e-6)


# ----------------------------------------------------------------- payload leg


@pytest.fixture(scope="module")
def payload():
    if not DATA.exists():
        pytest.skip(f"missing payload {DATA}")
    return json.loads(DATA.read_text())


def index(payload):
    return {(r["dx"], r["variant"]): r for r in payload["records"]}


def test_payload_grid_is_complete_per_level(payload):
    idx = index(payload)
    dxs = sorted({r["dx"] for r in payload["records"]})
    assert dxs, "no records"
    for dx in dxs:
        names = {v for (d, v) in idx if d == dx}
        # the H1/H2/H3 controls must be present at every measured level
        for required in ("nocap_a010", "cap020_a010", "nocap_a015"):
            assert required in names, f"{required} missing at dx={dx}"
    for rec in payload["records"]:
        for key in ("dx", "variant", "alpha", "cap_over_h",
                    "alpha_eff_predicted", "max_tvc_shift_over_h",
                    "max_tvc_clipped", "score", "clumping"):
            assert key in rec, key


def test_payload_gate_is_the_owner_standard(payload):
    assert payload["gates"]["owner_rel"] == 0.15
    assert payload["surface_factor"] == pytest.approx(SURFACE_FACTOR)


def test_h2_cap_above_peak_is_inert_on_every_level(payload):
    idx = index(payload)
    for dx in sorted({r["dx"] for r in payload["records"]}):
        capped, base = idx.get((dx, "cap020_a010")), idx.get((dx, "nocap_a010"))
        if capped is None or base is None:
            continue
        assert capped["max_tvc_clipped"] == 0.0
        assert capped["max_tvc_shift_over_h"] == pytest.approx(
            base["max_tvc_shift_over_h"], rel=1e-12)
        assert capped["score"]["mae"] == pytest.approx(base["score"]["mae"],
                                                      rel=1e-12)


def test_capped_runs_never_exceed_the_cap(payload):
    for rec in payload["records"]:
        cap = rec["cap_over_h"]
        if cap is None:
            continue
        assert rec["max_tvc_shift_over_h"] <= cap * (1.0 + 1e-6), rec["variant"]


def test_h1_is_falsified_by_the_payload(payload):
    """cap(0.20 h) at alpha=0.20/0.30 is not the uncapped alpha_eff=0.147 run."""
    idx = index(payload)
    checked = 0
    for dx in sorted({r["dx"] for r in payload["records"]}):
        ref = idx.get((dx, "nocap_a015"))
        if ref is None:
            continue
        for name in ("cap020_a020", "cap020_a030"):
            rec = idx.get((dx, name))
            if rec is None:
                continue
            assert rec["alpha_eff_predicted"] == pytest.approx(0.20 / SURFACE_FACTOR)
            assert rec["score"]["mean_rel_error"] > ref["score"]["mean_rel_error"]
            checked += 1
    assert checked > 0


def test_cap_below_surface_scale_is_best_at_dx003(payload):
    """H3 falsified in the useful direction: the clipped tail was hurting."""
    idx = index(payload)
    capped, base = idx.get((0.03, "cap010_a010")), idx.get((0.03, "nocap_a010"))
    if capped is None or base is None:
        pytest.skip("dx = 0.03 rows missing")
    assert capped["score"]["mean_rel_error"] < base["score"]["mean_rel_error"]
    assert capped["score"]["mae"] < base["score"]["mae"]
    assert capped["score"]["mean_rel_error"] <= 0.15
    assert capped["max_tvc_clipped"] > 0.0


def test_leg_has_no_row_claiming_the_three_percent_standard(payload):
    """Guard against an overstated document: nothing here is a 3 % result."""
    for rec in payload["records"]:
        rel = rec["score"].get("mean_rel_error")
        if rel is not None:
            assert rel > 0.03, rec["variant"]