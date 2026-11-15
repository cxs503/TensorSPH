"""Gates for the consistent-operator study of the closed-tank rest gap.

What is asserted, and why each assertion can fail
-------------------------------------------------
The anchor ``CASE-SPH-LAKE-AT-REST`` asks for rest to machine precision in a
closed tank.  The tank does not reach it, and ``docs/verification_lake_at_rest_wall_gap.md``
left the cure as an unmeasured promise ("a well-balanced / consistent momentum
operator is required").  These gates decide the two halves of that sentence:

* **the operator claim is true and exact** -- on a strictly linear pressure
  field the first-order-consistent operator returns the exact gradient at every
  particle of the truncated closed-tank lattice, wall band included, at every
  shipped resolution.  That is measured against the shipped pair form on the
  *same* lattice and the *same* field, so it is a comparison, not an identity
  with a hard-coded zero: the pair form gives ``1e-2 … 1e-1`` on the same input.
* **the cure does not close the case** -- on the equilibrium the shipped solver
  actually uses (Tait pressure of a mass field fitted for the vertical balance)
  the consistent operator is *worse* than the pair form, because the field it
  has to differentiate is not linear at the clamped free surface.  The gate that
  says so is what stops the next candidate cure from re-proposing the operator.

The mass-field statement is tested too: a fit against both acceleration
components -- the strongest thing the reconstruction admits -- still cannot make
the closed tank rest, so "the defect is in the pressure field" is not just a
reading of one operator's numbers.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch

from tensorsph import wellbalanced as wb
from tensorsph.lake_at_rest import LakeConfig, build_state, reconstruct, residual_report

ARTEFACT = pathlib.Path(__file__).parent / "data" / "lake_at_rest" / "wellbalanced.json"
WALL_GAP = pathlib.Path(__file__).parent / "data" / "lake_at_rest" / "wall_gap.json"


@pytest.fixture(scope="module")
def record():
    assert ARTEFACT.is_file(), "run benchmarks/lake_at_rest/wellbalanced_gap.py"
    return json.loads(ARTEFACT.read_text())


# --------------------------------------------------------------------------- #
# 1. the estimator's own invariant: exact on a linear field, any point set
# --------------------------------------------------------------------------- #
def test_consistent_gradient_is_exact_on_a_linear_field():
    """``A^-1 S == grad p`` exactly for a linear ``p``, truncated or not.

    A synthetic cloud with a deliberately one-sided support: the moment matrix
    is badly conditioned there and *still* the identity holds, because both
    sides are the same truncated sum.  This is the mathematical content the
    whole study rests on; if it fails, every other number here is meaningless.
    """
    torch.manual_seed(0)
    points = torch.rand(220, 2, dtype=torch.float64) * torch.tensor([1.0, 0.4],
                                                                   dtype=torch.float64)
    points[110:, 1] *= 0.35                        # one-sided / truncated half
    h = 0.12
    delta = points[:, None, :] - points[None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=2)
    close = (distance < 2.0 * h) & (distance > 0)
    i, j = torch.nonzero(close, as_tuple=True)
    from tensorsph.droplet_impact import pair_kernels
    _, grad = pair_kernels(points[i] - points[j], h)

    slope = torch.tensor([2.5, -7.0], dtype=torch.float64)
    offset = 3.0
    p = offset + points @ slope
    volume = torch.full((points.shape[0],), 0.01, dtype=torch.float64)

    dxj = points[j] - points[i]
    outer = grad.unsqueeze(2) * dxj.unsqueeze(1)
    A = torch.zeros(points.shape[0], 2, 2, dtype=torch.float64)
    A = A.index_add(0, i, volume[j].view(-1, 1, 1) * outer)
    A = A.index_add(0, j, volume[i].view(-1, 1, 1) * outer)
    S = torch.zeros(points.shape[0], 2, dtype=torch.float64)
    S = S.index_add(0, i, (volume[j] * (p[j] - p[i])).view(-1, 1) * grad)
    S = S.index_add(0, j, (volume[i] * (p[j] - p[i])).view(-1, 1) * grad)

    usable = torch.linalg.det(A).abs() > 1e-10 * torch.linalg.det(A).abs().max()
    gradient = torch.linalg.solve(A[usable], S[usable].unsqueeze(2)).squeeze(2)
    error = (gradient - slope).abs().max()
    assert error < 1e-12, f"linear-field exactness broken: {error:.3e}"


# --------------------------------------------------------------------------- #
# 2. the re-implementation is the shipped operator (both components)
# --------------------------------------------------------------------------- #
def test_symmetric_form_reproduces_the_shipped_operator():
    """``wall_gap.py`` validated only the horizontal sum; both must match."""
    torch.set_default_dtype(torch.float64)
    cfg = LakeConfig(dx=0.050, solver_max_iter=40)
    state = build_state(cfg)
    solved = reconstruct(state, cfg)
    shipped = residual_report(state, cfg, solved["mass"])
    ours = wb.pressure_acceleration(
        state, cfg, solved["mass"], wb.WellBalancedOptions("symmetric"))
    reference = shipped["acceleration"]
    assert float((ours["acceleration"][:, 1] - reference[:, 1]).abs().max()) < 1e-10
    assert float((ours["acceleration"][:, 0] - reference[:, 0]).abs().max()) < 1e-10
    # and it must reproduce the *failure*, not a fixed-up version of it
    assert shipped["a_max_over_g_horizontal"] > 0.05


# --------------------------------------------------------------------------- #
# 3. the decisive gate: exact on a linear field, on the real truncated lattice
# --------------------------------------------------------------------------- #
def test_linear_field_wall_band_is_roundoff_for_every_resolution(record):
    rows = record["linear_field"]
    assert len(rows) >= 3
    for row in rows:
        moment = row["moment_tight"]
        shipped = row["symmetric"]
        assert moment["a_x_wall_band"] < 1e-11, (
            f"dx={row['dx']}: consistent operator is not exact at the wall band "
            f"({moment['a_x_wall_band']:.3e})")
        assert moment["a_y_max_over_g"] < 1e-11, (
            f"dx={row['dx']}: vertical balance not exact on a linear field")
        # the shipped default regularisation floors exactness at ~1e-7 g, which
        # must still be orders below the failure it replaces
        assert row["moment"]["a_x_wall_band"] < 1e-6
        assert row["moment"]["a_x_wall_band"] * 1e4 < shipped["a_x_wall_band"]
        assert shipped["a_x_wall_band"] > 1e-3, (
            f"dx={row['dx']}: the pair form was supposed to fail here; the "
            f"comparison is only decisive if it does")
        # the uncorrected difference form is not the cure either
        assert row["difference"]["a_x_wall_band"] > 1e-3


def test_clamping_the_field_breaks_the_exactness(record):
    """The residual needs the free-surface pressure cut, not the operator."""
    for row in record["linear_field"]:
        unclamped = row["moment_tight"]["a_x_wall_band"]
        clamped = row["moment_tight_clamped"]["a_x_wall_band"]
        assert unclamped < 1e-11
        assert clamped > 100.0 * max(unclamped, 1e-16), (
            "clamping p at the free surface should be what re-introduces the "
            "wall-band force")


# --------------------------------------------------------------------------- #
# 4. the honest half: the cure does not close the shipped case
# --------------------------------------------------------------------------- #
def test_shipped_equilibrium_is_not_closed_by_the_operator(record):
    for row in record["reconstructed"]:
        symmetric = row["symmetric"]["a_x_wall_band"]
        moment = row["moment"]["a_x_wall_band"]
        assert symmetric > 0.05, "the gap this study explains has disappeared"
        assert moment > 0.5 * symmetric, (
            f"dx={row['dx']}: the consistent operator improved the reconstructed "
            f"case ({moment:.3e} vs {symmetric:.3e}); the finding that the "
            f"defect is in the field rather than the operator would be wrong")
        assert row["planarity"]["wall_band"] > 1e-3, (
            "the pressure field across the wall-band supports must be measurably "
            "non-planar for the explanation to hold")


def test_the_gap_is_reproduced_from_the_wall_gap_channel(record):
    """Cross-artefact: same case, same resolutions, same failure."""
    other = json.loads(WALL_GAP.read_text())
    by_dx = {round(row["dx"], 6): row for row in other["ladder"]}
    compared = 0
    for row in record["reconstructed"]:
        key = round(row["dx"], 6)
        if key not in by_dx:
            continue
        reference = None
        for holder in (by_dx[key].get("adami") or {}, by_dx[key]):
            if not isinstance(holder, dict):
                continue
            for name in ("a_x_wall_band", "a_x_wall_band_over_g",
                         "a_x_max_over_g_horizontal", "a_x_max_over_g"):
                if name in holder:
                    reference = holder[name]
                    break
            if reference is not None:
                break
        if reference is None:
            continue
        mine = row["symmetric"]["a_x_wall_band"]
        assert abs(mine - reference) <= 1e-9 + 1e-6 * abs(reference), (
            f"dx={row['dx']}: {mine:.6e} vs the shipped wall_gap record "
            f"{reference:.6e}")
        compared += 1
    assert compared >= 1, ("no shared field with wall_gap.json: the two evidence "
                           "channels for this case are no longer cross-checkable")


def test_no_mass_field_closes_the_gap(record):
    """Fitting both components is the strongest statement the reconstruction has."""
    for row in record["two_component"]:
        both = row["both_components"]
        assert both["a_x_wall_band"] > 0.02, (
            f"dx={row['dx']}: a two-component mass fit reached "
            f"{both['a_x_wall_band']:.3e}; that would refute the audit's "
            f"conclusion and needs its own review")
        assert both["a_y_max_over_g"] > 1e-6, (
            f"dx={row['dx']}: a_x improved by giving up the vertical balance")
        assert both["mass_scatter_vs_analytic"] >= 0.0


# --------------------------------------------------------------------------- #
# 5. the mechanism: truncation moment deficiency, per column
# --------------------------------------------------------------------------- #
def test_moment_deficiency_lives_at_the_wall_and_the_free_surface(record):
    for row in record["moments"]:
        assert row["A_yy_min_interior"] is None or row["A_yy_min_interior"] > 0.8, (
            f"dx={row['dx']}: an interior particle should have a complete "
            f"support (A_yy ~ 1), got {row['A_yy_min_interior']}")
        assert row["A_yy_over_A_xx_worst"] < 0.8, (
            f"dx={row['dx']}: no truncation signature at the corner")
        # Measured: the *worst* A_yy is on the free-surface row of an interior
        # column (~0.41, the surface truncates the support for every column),
        # while the wall columns sit at ~0.55 -- the side wall actually
        # *reinforces* the moment the free surface took away.  The wall's own
        # signature is therefore the xy coupling, not a smaller A_yy.
        assert row["A_yy_min_wall_band"] < 0.8
        assert row["A_xy_max_over_A_xx"] > 0.05, (
            "no xy coupling: the side wall is supposed to skew the moment")


def test_recorded_numbers_come_from_the_shipped_operator():
    """A live re-derivation at the cheapest resolution, from scratch."""
    torch.set_default_dtype(torch.float64)
    cfg = LakeConfig(dx=0.050, depth=0.30, solver_max_iter=40)
    state = build_state(cfg)
    mass = state["m_unit"] * torch.ones(state["pos"].shape[0],
                                       dtype=state["pos"].dtype)
    H = cfg.depth
    p_full = torch.cat((cfg.rho0 * cfg.gravity * (H - state["pos"][:, 1]),
                        cfg.rho0 * cfg.gravity * (H - state["wall"][:, 1])))
    field = {"p_full": p_full,
             "rho_full": torch.full_like(p_full, cfg.rho0)}
    moment = wb.band_report(state, cfg, mass, wb.WellBalancedOptions("moment"), field)
    symmetric = wb.band_report(state, cfg, mass,
                               wb.WellBalancedOptions("symmetric"), field)
    tight = wb.band_report(state, cfg, mass,
                           wb.WellBalancedOptions("moment", regularization=1e-12), field)
    assert tight["a_x_wall_band"] < 1e-11
    assert symmetric["a_x_wall_band"] > 1e-3

    record = json.loads(ARTEFACT.read_text())
    row = min(record["linear_field"], key=lambda item: abs(item["dx"] - 0.050))
    assert abs(row["moment"]["a_x_wall_band"] - moment["a_x_wall_band"]) < 1e-12
    assert abs(row["moment_tight"]["a_x_wall_band"]
               - tight["a_x_wall_band"]) < 1e-12
    assert abs(row["symmetric"]["a_x_wall_band"]
               - symmetric["a_x_wall_band"]) < 1e-12