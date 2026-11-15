"""Does a *consistent momentum operator* close the closed-tank rest gap?

`docs/verification_lake_at_rest_wall_gap.md` ends with the open item

    "A well-balanced/consistent momentum operator (kernel gradient correction or
    a boundary-integral wall treatment) is required; that is a solver change, not
    a case change."

and `tests/test_lake_at_rest_wall_gap.py` pins the gap itself.  This study takes
that open item and *decides* it: four standard consistent/well-balanced pressure
gradients are implemented (module :mod:`tensorsph.lake_at_rest_wb`), each vertical
balance is closed by the case's own Gauss-Newton mass solve, and the horizontal
residual is measured against the shipped baseline on the same lattice.

Writes `tests/data/lake_at_rest/wb_operator.json` (numbers only, no figures
inside) and, with ``--figures``, the comparison plot.

    PYTHONPATH=src python3 benchmarks/lake_at_rest_wb/run.py \
        --json tests/data/lake_at_rest/wb_operator.json --figures docs/figures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from tensorsph.lake_at_rest import LakeConfig, build_state, reconstruct, residual_report
from tensorsph.lake_at_rest_wb import (ANCHOR_TOLERANCE, cancellation_pressure,
                                       wb_accel, wb_fields, wb_reconstruct)

RESOLUTIONS = (0.050, 0.040, 0.025)

# (name, form, own_pressure, corrected_gradient, description)
VARIANTS = (
    ("shipped_reimplementation", "symmetrised", "eos", False,
     "the shipped form through this code path (implementation control: must "
     "reproduce residual_report)"),
    ("own_pressure_reconstructed", "symmetrised", "reconstructed", False,
     "shipped form, but the free-surface particle's clamped p=0 replaced by its "
     "support's least-squares hydrostatic value"),
    ("corrected_gradient", "symmetrised", "eos", True,
     "shipped form with the Randles-Libersky first-order moment correction M_i^-1"),
    ("corrected_gradient_own_reconstructed", "symmetrised", "reconstructed", True,
     "both of the above"),
    ("difference_consistent", "difference", "reconstructed", True,
     "well-balanced difference form -M^-1 sum V (p_j - p_i) grad W"),
    ("lsq_plane_gradient", "lsq", "reconstructed", True,
     "MLS plane-fit gradient (exact for any linear field on any support)"),
)


def one_resolution(dx: float, max_iter: int = 25) -> dict:
    cfg = LakeConfig(periodic_x=False, dx=dx)
    state = build_state(cfg)
    g = cfg.gravity
    started = time.time()
    base = reconstruct(state, cfg)
    mass0 = base["mass"]
    report = residual_report(state, cfg, mass0)
    fields = wb_fields(cfg, mass0)
    rows = []
    for name, form, own, corr, why in VARIANTS:
        at_shipped = wb_accel(fields, own_pressure=own, form=form,
                              corrected_gradient=corr)
        record = {
            "name": name, "form": form, "own_pressure": own,
            "corrected_gradient": corr, "description": why,
            "at_shipped_mass": {
                "max_ax_over_g": float(at_shipped[:, 0].abs().max() / g),
                "max_ay_over_g": float(at_shipped[:, 1].abs().max() / g),
                "max_a_over_g": float(torch.linalg.vector_norm(at_shipped, dim=1).max() / g),
            },
        }
        t0 = time.time()
        rec = wb_reconstruct(cfg, own_pressure=own, form=form,
                             corrected_gradient=corr, prior=mass0,
                             max_iter=max_iter, target="vertical")
        record["retuned"] = {
            "converged": bool(rec["converged"]),
            "iterations": int(rec["iterations"]),
            "max_ax_over_g": rec["max_ax_over_g"],
            "max_ay_over_g": rec["max_ay_over_g"],
            "max_a_over_g": rec["max_a_over_g"],
            "residual_history": [float(v) for v in rec["residual_history"]],
            "seconds": round(time.time() - t0, 2),
        }
        rows.append(record)
    # controls on the *baseline* operator: the measurement must be sensitive to
    # the pressure data it is given, otherwise "rest" is manufactured.
    zero_pressure = dict(fields)
    zero_pressure["p_all"] = torch.zeros_like(fields["p_all"])
    controls = {
        "zero_pressure_max_a_over_g": float(torch.linalg.vector_norm(
            wb_accel(zero_pressure), dim=1).max() / g),
        "pressure_scaled_1pct_max_ax_over_g": float(wb_accel(
            fields, pressure_scale=1.01)[:, 0].abs().max() / g),
        "gravity_off_max_a_over_g": float(torch.linalg.vector_norm(
            wb_accel(fields, gravity_on=False), dim=1).max() / g),
    }
    return {
        "dx": dx,
        "h": cfg.h(),
        "fluid_particles": int(state["pos"].shape[0]),
        "wall_particles": int(state["wall"].shape[0]),
        "unknowns": int(state["columns"].max().item()) + 1,
        "shipped": {
            "converged": bool(base["converged"]),
            "iterations": int(base["iterations"]),
            "max_a_over_g": report["a_max_over_g"],
            "max_ax_over_g": report["a_max_over_g_horizontal"],
            "max_ax_over_g_wall_band": report["a_max_over_g_horizontal_wall_band"],
            "max_ax_over_g_interior": report["a_max_over_g_horizontal_interior"],
        },
        "variants": rows,
        "controls": controls,
        "cancellation": {
            key: (float(value) if isinstance(value, float) else value)
            for key, value in cancellation_pressure(fields).items()
            if key in ("worst_position", "worst_ax_over_g", "worst_deficiency",
                       "worst_p_required", "worst_p_physical", "worst_p_eos",
                       "worst_required_over_physical",
                       "wall_band_residual_ax_max_over_g",
                       "interior_residual_ax_max_over_g")
        },
        "seconds": round(time.time() - started, 1),
    }


def figures(payload: dict, directory: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(directory, exist_ok=True)
    ladder = payload["ladder"]
    names = [v["name"] for v in ladder[0]["variants"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    colors = plt.cm.tab10.colors
    for k, name in enumerate(names):
        xs = [row["dx"] for row in ladder]
        shipped = [next(v for v in row["variants"] if v["name"] == name)["at_shipped_mass"]
                   ["max_ax_over_g"] for row in ladder]
        retuned = [next(v for v in row["variants"] if v["name"] == name)["retuned"]
                   ["max_ax_over_g"] for row in ladder]
        axes[0].plot(xs, shipped, "o--", color=colors[k % 10], label=name, alpha=0.9)
        axes[1].plot(xs, retuned, "s-", color=colors[k % 10], label=name, alpha=0.9)
    for ax, title in zip(axes, ("at the shipped mass field",
                                "vertical balance re-closed per operator")):
        ax.axhline(ANCHOR_TOLERANCE, color="k", lw=1.2, ls=":",
                   label="anchor tolerance 1e-9")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("dx [m]")
        ax.set_ylabel(r"max $|a_x|/g$")
        ax.set_title(f"closed tank: horizontal rest residual, {title}")
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "lake_wb_operator_ax.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    row = ladder[len(ladder) // 2]
    labels, values = [], []
    for v in row["variants"]:
        labels.append(v["name"])
        values.append(v["retuned"]["max_a_over_g"])
    values.append(row["shipped"]["max_ax_over_g"] * 0 + row["shipped"]["max_a_over_g"])
    labels.append("shipped (whole vector)")
    ax.barh(range(len(values)), values, color=colors[:len(values)])
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xscale("log")
    ax.axvline(ANCHOR_TOLERANCE, color="k", ls=":", lw=1.2)
    ax.set_xlabel(r"max $|a|/g$ (both components), $dx$ = %g m" % row["dx"])
    ax.set_title("no consistent-gradient variant reaches the anchor tolerance")
    ax.grid(alpha=0.3, axis="x", which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "lake_wb_operator_bar.png"), dpi=160)
    plt.close(fig)


def summarise(payload: dict) -> dict:
    out = {"anchor_tolerance": ANCHOR_TOLERANCE, "rows": []}
    for row in payload["ladder"]:
        entry = {"dx": row["dx"], "shipped_ax": row["shipped"]["max_ax_over_g"],
                 "shipped_ay": row["shipped"]["max_a_over_g"]}
        for v in row["variants"]:
            entry[v["name"] + "_ax"] = v["retuned"]["max_ax_over_g"]
            entry[v["name"] + "_ay"] = v["retuned"]["max_ay_over_g"]
        out["rows"].append(entry)
    best = min(v["retuned"]["max_a_over_g"] for row in payload["ladder"]
               for v in row["variants"])
    out["best_variant_max_a_over_g"] = best
    out["best_over_tolerance"] = best / ANCHOR_TOLERANCE
    out["verdict"] = ("no consistent-gradient variant reaches the anchor tolerance"
                      if best > ANCHOR_TOLERANCE else "a variant reaches tolerance")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="tests/data/lake_at_rest/wb_operator.json")
    parser.add_argument("--figures", default="")
    parser.add_argument("--resolutions", default=",".join(str(v) for v in RESOLUTIONS))
    parser.add_argument("--max-iter", type=int, default=25)
    args = parser.parse_args()

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        ladder = [one_resolution(float(v), args.max_iter)
                  for v in args.resolutions.split(",") if v.strip()]
    finally:
        torch.set_default_dtype(previous)
    payload = {
        "case": "CASE-SPH-LAKE-AT-REST",
        "geometry": {"periodic_x": False, "depth": 0.30, "width": 0.40,
                     "c0": 20.0, "rho0": 1000.0, "gravity": 9.81,
                     "wall_layers": 3},
        "anchor_tolerance": ANCHOR_TOLERANCE,
        "question": ("does a consistent/well-balanced pressure gradient close the "
                     "closed-tank rest gap demanded by verification_lake_at_rest_wall_gap.md"),
        "ladder": ladder,
        "summary": summarise({"ladder": ladder}),
    }
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=1)
    if args.figures:
        figures(payload, args.figures)
    print(json.dumps(payload["summary"], indent=1))
    for row in ladder:
        print(f"dx={row['dx']:.3f} shipped ax={row['shipped']['max_ax_over_g']:.3e}"
              f"  " + "  ".join(f"{v['name'][:18]}={v['retuned']['max_ax_over_g']:.3e}"
                                f"/ay={v['retuned']['max_ay_over_g']:.1e}"
                                for v in row["variants"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())