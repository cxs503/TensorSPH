"""Is the closed-tank rest gap a *gradient operator* defect or a *field* defect?

``docs/verification_lake_at_rest_wall_gap.md`` ended with a verdict and a
promise: the closed tank's horizontal residual ``max|a_x|/g = 0.10 … 0.12`` is
not removable by refinement, wall mass, wall-face pressure or any shipped
operator option, and closes only with "a well-balanced / consistent momentum
operator (kernel gradient correction or a boundary-integral wall treatment);
that is a solver change, not a case change".  That sentence left the cure
unbuilt and unmeasured.

This runner builds it and measures it -- and the answer is more interesting than
"it works": the cure is *exact* on the geometry of this very tank, and it still
does not close the shipped case, because the defect the audit located is in the
**pressure field**, not in the gradient operator.

Four measurements:

``validation``      The re-implementation reproduces the shipped operator
                    component by component (not just horizontally, which is all
                    ``wall_gap.py`` validated).
``linear_field``    The decisive one.  Put the *analytic, strictly linear*
                    hydrostatic pressure ``p = rho0 g (H - y)`` on the real
                    closed-tank lattice (wall particles included), then ask each
                    operator for the acceleration.  A consistent estimator must
                    return zero at every particle, however truncated the
                    support.  Measured: roundoff at the wall band, where the
                    shipped form gives ``1e-1``.
``moments``         Why the shipped form cannot: the truncated support halves
                    the ``yy`` moment at the free surface and adds an ``xy``
                    coupling at the wall, per column, per resolution.
``reconstructed``   The same operators on the *shipped* equilibrium, whose
                    pressure field is the Tait pressure of a mass field fitted
                    for the vertical balance -- i.e. non-linear at the clamped
                    free surface.  Here the consistent operator is worse, and
                    the planarity diagnostic says why: the very non-linearity
                    the consistent fit must extrapolate is what it amplifies.
``two_component``   Independent confirmation that no mass field closes the gap:
                    a mass fit against *both* acceleration components, which is
                    the strongest statement the reconstruction admits.

Produce the evidence + figures:

    PYTHONPATH=src python3 benchmarks/lake_at_rest/wellbalanced_gap.py \\
        --json tests/data/lake_at_rest/wellbalanced.json --figures docs/figures
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorsph import wellbalanced as wb
from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    reconstruct, residual_report)

RESOLUTIONS = (0.050, 0.040, 0.025, 0.020)
# The two studies that need a Gauss-Newton mass reconstruction are the expensive
# ones; the operator comparison itself is cheap at every resolution, and the
# resolution ladder of the *failure* is already shipped in ``wall_gap.json``.
FIT_RESOLUTIONS = (0.050, 0.040)
FORMS = ("symmetric", "difference", "moment")
# ``moment`` keeps the module's robust default (eps = 1e-6 relative to the moment
# trace), which floors the estimator's exactness at ~1e-7 g.  ``moment_tight``
# drops the regularisation to 1e-12 to show that the residual is the
# regularisation and not the estimator.
FORM_OPTIONS = {
    "symmetric": lambda: wb.WellBalancedOptions("symmetric"),
    "difference": lambda: wb.WellBalancedOptions("difference"),
    "moment": lambda: wb.WellBalancedOptions("moment"),
    "moment_tight": lambda: wb.WellBalancedOptions("moment", regularization=1e-12),
}


# --------------------------------------------------------------------------- #
# synthetic fields
# --------------------------------------------------------------------------- #
def linear_field(state, cfg, clamp: bool = False) -> dict:
    """Strictly linear hydrostatic pressure ``rho0 g (H - y)`` on the lattice.

    ``clamp=True`` reproduces the free-surface cut the Tait law applies
    (``p = max(p, 0)``), which is exactly what makes the field non-linear.
    """
    H = cfg.depth
    p_fluid = cfg.rho0 * cfg.gravity * (H - state["pos"][:, 1])
    p_wall = cfg.rho0 * cfg.gravity * (H - state["wall"][:, 1])
    if clamp:
        p_fluid = p_fluid.clamp_min(0.0)
        p_wall = p_wall.clamp_min(0.0)
    p_full = torch.cat((p_fluid, p_wall))
    return {"p_full": p_full,
            "rho_full": torch.full_like(p_full, cfg.rho0)}


def band_masks(state, cfg) -> dict:
    pos = state["pos"]
    band = 2.0 * cfg.h()
    wall_band = (pos[:, 0] < band) | (pos[:, 0] > cfg.width - band)
    interior = (pos[:, 1] > band) & (pos[:, 1] < cfg.depth - band)
    return {"wall_band": wall_band, "interior": interior}


def residual_row(state, cfg, mass, options, fields=None, label=None) -> dict:
    report = wb.band_report(state, cfg, mass, options, fields)
    row = {key: report[key] for key in
           ("a_y_max_over_g", "a_x_max_over_g", "a_x_wall_band",
            "a_x_interior", "net_horizontal_force_over_weight")}
    row["condition_max"] = report["condition_max"]
    row["fallbacks"] = report["fallbacks"]
    if label:
        row["formulation"] = label
    return row


def planarity(state, cfg, mass, options=None) -> dict:
    """How far the pressure field departs from a plane across each support.

    ``res_j = (p_j - p_i) - grad p_i . (x_j - x_i)`` with ``grad p_i`` from the
    consistent estimator.  For a strictly linear field this is roundoff by
    construction (it is the same identity the operator relies on); for the
    reconstructed Tait field it is the misfit the estimator has to extrapolate.
    """
    options = options or wb.WellBalancedOptions()
    f = wb.build_fields(state, cfg, mass, options)
    n = f["n"]
    moments = wb.moment_operator(f)
    A, S = moments["A"][:n], moments["S"][:n]
    gradient = torch.linalg.solve(A, S.unsqueeze(2)).squeeze(2)
    masks = band_masks(state, cfg)
    i, j, grad = f["i"], f["j"], f["grad_w"]
    keep = wb._pair_selection(f)
    i, j, grad = i[keep], j[keep], grad[keep]
    volume = f["all_mass"] / f["rho_full"]
    delta = (f["all_pos"][j] - f["all_pos"][i])
    dp = f["p_full"][j] - f["p_full"][i]
    predicted = (gradient[i.clamp_max(n - 1)] * delta).sum(dim=1)
    # only pairs whose *first* endpoint is a fluid particle are attributable
    own = i < n
    mismatch = (dp - predicted).abs()
    scale = float(f["p_full"][:n].max())
    row = {}
    for name, mask in masks.items():
        idx = own & mask[i.clamp_max(n - 1)]
        row[name] = float(mismatch[idx].max() / scale) if bool(idx.any()) else None
        row[name + "_rms"] = (float((mismatch[idx].square().mean().sqrt() / scale))
                             if bool(idx.any()) else None)
    row["p_max"] = scale
    return row


# --------------------------------------------------------------------------- #
# the five studies
# --------------------------------------------------------------------------- #
def study_validation(depth: float) -> list:
    rows = []
    for dx in FIT_RESOLUTIONS:
        cfg = LakeConfig(dx=dx, depth=depth, solver_max_iter=40)
        state = build_state(cfg)
        solved = reconstruct(state, cfg)
        shipped = residual_report(state, cfg, solved["mass"])
        ours = wb.pressure_acceleration(
            state, cfg, solved["mass"], wb.WellBalancedOptions("symmetric"))
        acc = ours["acceleration"]
        ref = shipped["acceleration"]
        rows.append({
            "dx": dx, "n_fluid": int(state["pos"].shape[0]),
            "a_y_over_g": shipped["a_max_over_g"],
            "a_x_over_g": shipped["a_max_over_g_horizontal"],
            "max_abs_diff_y": float((acc[:, 1] - ref[:, 1]).abs().max()),
            "max_abs_diff_x": float((acc[:, 0] - ref[:, 0]).abs().max()),
            "relative": float((acc - ref).abs().max()
                              / max(float(ref.abs().max()), 1e-30)),
        })
        print(f"  validation dx={dx:.3f} re-implementation matches shipped to "
              f"{rows[-1]['max_abs_diff_y']:.2e} (y), "
              f"{rows[-1]['max_abs_diff_x']:.2e} (x)", flush=True)
    return rows


def study_linear_field(depth: float) -> list:
    rows = []
    for dx in RESOLUTIONS:
        cfg = LakeConfig(dx=dx, depth=depth, solver_max_iter=40)
        state = build_state(cfg)
        mass = state["m_unit"] * torch.ones(
            state["pos"].shape[0], dtype=state["pos"].dtype)
        row = {"dx": dx}
        for clamp in (False, True):
            field = linear_field(state, cfg, clamp=clamp)
            for form, factory in FORM_OPTIONS.items():
                key = f"{form}{'_clamped' if clamp else ''}"
                row[key] = residual_row(state, cfg, mass, factory(), field)
        rows.append(row)
        unclamped = row["moment"]
        print(f"  linear dx={dx:.3f}  moment a_x/g = "
              f"{unclamped['a_x_max_over_g']:.2e} (shipped form "
              f"{row['symmetric']['a_x_max_over_g']:.2e})", flush=True)
    return rows


def study_moments(depth: float) -> list:
    rows = []
    for dx in RESOLUTIONS:
        cfg = LakeConfig(dx=dx, depth=depth, solver_max_iter=40)
        state = build_state(cfg)
        # The moment deficiency is a property of the *support*, not of the mass
        # field, so the analytic profile is enough and no solve is needed.
        row = wb.moment_anisotropy(state, cfg, analytic_profile(state, cfg)["mass"])
        row["dx"] = dx
        rows.append(row)
        print(f"  moments dx={dx:.3f}  A_yy/A_xx worst = "
              f"{row['A_yy_over_A_xx_worst']:.3f}  A_xy/A_xx = "
              f"{row['A_xy_max_over_A_xx']:.3f}", flush=True)
    return rows


def study_reconstructed(depth: float) -> list:
    rows = []
    for dx in FIT_RESOLUTIONS:
        cfg = LakeConfig(dx=dx, depth=depth, solver_max_iter=40)
        state = build_state(cfg)
        solved = reconstruct(state, cfg)
        mass = solved["mass"]
        row = {"dx": dx, "reconstruct_iterations": solved["iterations"],
               "shipped_a_y_over_g": solved["a_max_over_g"],
               "shipped_a_x_over_g": solved["a_max_over_g_horizontal"]}
        for form in FORMS:
            row[form] = residual_row(state, cfg, mass, wb.WellBalancedOptions(form))
        row["planarity"] = planarity(state, cfg, mass)
        rows.append(row)
        print(f"  reconstructed dx={dx:.3f}  shipped {row['symmetric']['a_x_max_over_g']:.3e}"
              f"  moment {row['moment']['a_x_max_over_g']:.3e}"
              f"  planarity {row['planarity']['wall_band']:.2e}", flush=True)
    return rows


def study_two_component(depth: float) -> list:
    """Best acceleration the mass field can reach when *both* components count."""
    rows = []
    for dx in (0.050,):
        cfg = LakeConfig(dx=dx, depth=depth, solver_max_iter=40)
        state = build_state(cfg)
        prior = analytic_profile(state, cfg)
        groups = wb._tie_groups(state).long()
        count = int(groups.max()) + 1
        unit = float(state["m_unit"])
        prior_mass = prior["mass"]
        prior_u = torch.zeros(count, dtype=prior_mass.dtype)
        prior_u = prior_u.index_add(0, groups, torch.log(prior_mass / unit)) / \
            torch.zeros(count, dtype=prior_mass.dtype).index_add(
                0, groups, torch.ones_like(prior_mass))
        options = wb.WellBalancedOptions("symmetric")

        def masses(u):
            return unit * torch.exp(u[groups])

        def residual(u, both: bool):
            acc = wb.pressure_acceleration(state, cfg, masses(u), options)["acceleration"]
            return torch.cat((acc[:, 1], acc[:, 0])) if both else acc[:, 1]

        def solve(both: bool, iterations: int = 12, step: float = 1.0e-6,
                  prior_weight: float = 1.0e-8):
            u, history = prior_u.clone(), []
            eye = torch.eye(count, dtype=u.dtype)
            for _ in range(iterations):
                r = residual(u, both)
                metric = float(r.abs().max() / cfg.gravity)
                history.append(metric)
                if metric < 1.0e-11:
                    break
                columns = [(residual(u + eye[k] * step, both)
                            - residual(u - eye[k] * step, both)) / (2 * step)
                           for k in range(count)]
                jac = torch.stack(columns, dim=1)
                lam = prior_weight ** 0.5
                block = torch.cat((jac, lam * eye), dim=0)
                rhs = torch.cat((-r, lam * (prior_u - u)), dim=0)
                delta = torch.linalg.lstsq(block, rhs.unsqueeze(1)).solution.squeeze(1)
                alpha, best, accepted = 1.0, metric, None
                for _ in range(20):
                    trial = u + alpha * delta
                    value = float(residual(trial, both).abs().max() / cfg.gravity)
                    if value < best:
                        accepted, best = trial, value
                        break
                    alpha *= 0.5
                if accepted is None:
                    break
                u = accepted
            return u, history

        row = {"dx": dx}
        for name, both in (("vertical_only", False), ("both_components", True)):
            u, history = solve(both)
            report = wb.band_report(state, cfg, masses(u), options)
            row[name] = {
                "a_y_max_over_g": report["a_y_max_over_g"],
                "a_x_max_over_g": report["a_x_max_over_g"],
                "a_x_wall_band": report["a_x_wall_band"],
                "iterations": len(history),
                "mass_scatter_vs_analytic": float(
                    (masses(u) / prior_mass - 1.0).abs().max()),
            }
            print(f"  two-component dx={dx:.3f} {name:>15s}: "
                  f"a_y={report['a_y_max_over_g']:.2e} "
                  f"a_x={report['a_x_max_over_g']:.2e} "
                  f"scatter={row[name]['mass_scatter_vs_analytic']:.2%}", flush=True)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(payload: dict, directory: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written = []

    linear = payload["linear_field"]
    dxs = [row["dx"] for row in linear]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for key, style in (("symmetric", "o-"), ("difference", "s--"),
                       ("moment", "^-"), ("moment_tight", "v:")):
        axes[0].plot(dxs, [row[key]["a_x_wall_band"] for row in linear], style,
                     label=key)
        axes[1].plot(dxs, [row[key]["a_x_interior"] for row in linear], style,
                     label=key)
    for ax, title in zip(axes, ("wall band", "interior")):
        ax.set_yscale("log")
        ax.set_xlabel("dx [m]")
        ax.set_ylabel(r"$\max|a_x|/g$")
        ax.set_title(f"linear hydrostatic p, {title}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Consistent operator is exact on a linear field; the pair form is not")
    fig.tight_layout()
    path = directory / "lake_wellbalanced_linear_field.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    moments = payload["moments"]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for row in moments:
        xs = [float(k) for k in row["per_column_A_yy_min"]]
        ys = [row["per_column_A_yy_min"][k] for k in row["per_column_A_yy_min"]]
        ax.plot(xs, ys, "o-", label=f"dx={row['dx']:.3f}")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\min A_{yy}/A_{xx}$ per column")
    ax.set_title("Truncation signature: free-surface / wall moment deficiency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "lake_wellbalanced_moments.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    reconstructed = payload["reconstructed"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    dxr = [row["dx"] for row in reconstructed]
    for key, style in (("symmetric", "o-"), ("difference", "s--"), ("moment", "^-")):
        ax.plot(dxr, [row[key]["a_x_wall_band"] for row in reconstructed], style,
                label=f"{key} (reconstructed p)")
    ax.plot(dxs, [row["moment"]["a_x_wall_band"] for row in linear], "k:",
            label="moment (linear p)")
    ax.set_yscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"wall-band $\max|a_x|/g$")
    ax.set_title("On the shipped equilibrium the cure needs a linear p, not an operator")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "lake_wellbalanced_reconstructed.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return [str(p) for p in written]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path,
                        default=Path("tests/data/lake_at_rest/wellbalanced.json"))
    parser.add_argument("--figures", type=Path, default=Path("docs/figures"))
    parser.add_argument("--depth", type=float, default=0.30)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)

    t0 = time.time()
    payload = {
        "case": "CASE-SPH-LAKE-AT-REST",
        "anchor_tolerance": "machine precision (analytic identity)",
        "geometry": {"depth": args.depth, "width": 0.40, "rho0": 1000.0,
                     "sound_speed": 20.0, "gamma": 7.0, "gravity": 9.81,
                     "h_factor": 1.3, "wall_layers": 3},
        "resolutions": list(RESOLUTIONS),
        "fit_resolutions": list(FIT_RESOLUTIONS),
    }
    print("validation:", flush=True)
    payload["validation"] = study_validation(args.depth)
    print("linear field:", flush=True)
    payload["linear_field"] = study_linear_field(args.depth)
    print("moment deficiency:", flush=True)
    payload["moments"] = study_moments(args.depth)
    print("reconstructed equilibrium:", flush=True)
    payload["reconstructed"] = study_reconstructed(args.depth)
    print("two-component mass fit:", flush=True)
    payload["two_component"] = study_two_component(args.depth)
    payload["seconds"] = round(time.time() - t0, 1)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.json}", flush=True)
    if args.figures:
        for path in make_figures(payload, args.figures):
            print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())