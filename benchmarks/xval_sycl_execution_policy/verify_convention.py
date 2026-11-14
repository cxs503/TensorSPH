"""Verification of the xval_sycl_execution_policy deliverable (m19962).

Re-measures the shipped deliverable's claims with an independently authored
harness (this file, no import of the original benchmark) and adds the check the
original deliverable did not perform: the two arms' kernel-evaluation
convention under a **spatially varying** smoothing length.

Stages
------
S1  uniform h: sweep vs the shipped symmetric reference, 10 work-group sizes
    (bit-exactness + integer-ULP distance) -- re-runs the original claim.
S2  variable h: h = h0 (1 + amp * sin(2 pi x / L)), over an amp x dx grid.
      (a) sweep vs symmetric reference  -> *convention* difference
      (b) sweep vs per-row-h reference  -> *launch-geometry* difference
S3  negative control: tail-mask removed, plus the degeneracy case where the
    last swept row has no neighbours (control becomes vacuous).
S4  edge cases: bound = 0, bound > n, empty pair list, n < local_size.

Outputs (artifacts/xval_execution_policy/verify_m19962/): results.json,
summary.md (programmatic markdown tables), fig_*.png.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.wcsph import kernels                            # noqa: E402
from tensorsph.execution_policy import work_item_counts          # noqa: E402
from tensorsph.neighbor_search import PairListCache              # noqa: E402
from tensorsph.wcsph import swept                                # noqa: E402
from tensorsph.wcsph.cases import lattice_box                    # noqa: E402

DEFAULT_OUT = ROOT / "artifacts" / "xval_execution_policy" / "verify_m19962"
D = torch.float64
DX = 0.02
H0 = 1.3 * DX
RHO0 = 1000.0
KERNEL = "quintic"
GATE_OWNER = 0.15          # owner standard: >15 % is not acceptable
GATE_TARGET = 1.0e-12      # predeclared engineering target for a neutral rewrite


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float(((got - ref).abs() / ref.abs().clamp_min(1.0e-300)).max())


def ulp_gap(got: torch.Tensor, ref: torch.Tensor) -> int:
    return int((got.view(torch.int64) - ref.view(torch.int64)).abs().max())


def build_particles(h_modulation: float = 0.0, dx: float = DX) -> dict:
    """Fluid box [0,0.4]x[0,0.2] + 3 layers of wall particles, static."""
    h0 = 1.3 * dx
    pos = lattice_box([0.0, 0.0], [0.4, 0.2], dx, 2)
    n = int(pos.shape[0])
    x = pos[:, 0]
    length = float(x.max()) or 1.0
    h = h0 * (1.0 + h_modulation * torch.sin(2.0 * math.pi * x / length))
    cutoff = kernels.SUPPORT_FACTOR[KERNEL] * float(h.max())
    cache = PairListCache(cutoff, 0.0, fluid_count=n)
    pairs, _ = cache.get(pos)
    mass = torch.full((n,), RHO0 * dx ** 2, dtype=D)
    sym = swept.summation_density_vectorised(pos, mass, h, 2, pairs, kernel=KERNEL)
    return {
        "pos": pos,
        "mass": mass,
        "h": h,
        "pairs": pairs,
        "n": n,
        "dx": dx,
        "h_modulation": h_modulation,
        "cutoff": cutoff,
        "pairs_per_particle": 2.0 * int(pairs.shape[0]) / n,
        "h_over_h0": (float(h.min()) / h0, float(h.max()) / h0),
        "h_over_dx": (float(h.min()) / dx, float(h.max()) / dx),
        "rho_over_rho0": (float(sym.min()) / RHO0, float(sym.max()) / RHO0),
    }


def s1_uniform(case: dict) -> list[dict]:
    rows = []
    ref = swept.summation_density_vectorised(case["pos"], case["mass"], case["h"], 2,
                                             case["pairs"], kernel=KERNEL)
    rowh = swept.summation_density_vectorised_row_h(case["pos"], case["mass"], case["h"], 2,
                                                    case["pairs"], kernel=KERNEL)
    for local in (1, 2, 3, 7, 8, 16, 31, 64, 100, 128):
        got = swept.summation_density_swept(case["pos"], case["mass"], case["h"], 2,
                                            case["pairs"], kernel=KERNEL, local_size=local)
        rows.append({
            "local_size": local,
            "launch": work_item_counts(case["n"], local),
            "rel_err_vs_shipped": rel_err(got, ref),
            "ulp_vs_shipped": ulp_gap(got, ref),
            "bit_exact_vs_shipped": bool(torch.equal(got, ref)),
            "bit_exact_vs_row_h": bool(torch.equal(got, rowh)),
        })
    return rows


def s2_variable_h(amplitudes, dx_axis=(0.04, 0.02, 0.01), locals_=(1, 8, 64, 100)) -> list[dict]:
    rows = []
    for dx in dx_axis:
        for amp in amplitudes:
            case = build_particles(amp, dx)
            pos, mass, h, pairs = case["pos"], case["mass"], case["h"], case["pairs"]
            sym = swept.summation_density_vectorised(pos, mass, h, 2, pairs, kernel=KERNEL)
            rowh = swept.summation_density_vectorised_row_h(pos, mass, h, 2, pairs, kernel=KERNEL)
            per_local = []
            for local in locals_:
                got = swept.summation_density_swept(pos, mass, h, 2, pairs, kernel=KERNEL,
                                                   local_size=local)
                per_local.append({
                    "local_size": local,
                    "rel_err_vs_symmetric": rel_err(got, sym),
                    "rel_err_vs_row_h": rel_err(got, rowh),
                    "bit_exact_vs_row_h": bool(torch.equal(got, rowh)),
                })
            gap = max(r["rel_err_vs_symmetric"] for r in per_local)
            rows.append({
                "dx": dx,
                "particles": case["n"],
                "pairs_per_particle": case["pairs_per_particle"],
                "h_over_h0_min": case["h_over_h0"][0],
                "h_over_h0_max": case["h_over_h0"][1],
                "h_over_dx_max": case["h_over_dx"][1],
                "rho_over_rho0_min": case["rho_over_rho0"][0],
                "rho_over_rho0_max": case["rho_over_rho0"][1],
                "h_modulation": amp,
                "convention_gap_max": gap,
                "convention_gap_symmetric_only": rel_err(rowh, sym),
                "launch_invariance_max": max(r["rel_err_vs_row_h"] for r in per_local),
                "launch_invariance_all_bitexact": all(r["bit_exact_vs_row_h"] for r in per_local),
                "per_local": per_local,
                "exceeds_owner_gate": bool(gap > GATE_OWNER),
            })
    return rows


def s3_control(case: dict) -> dict:
    pos, mass, h, pairs = case["pos"], case["mass"], case["h"], case["pairs"]
    ref = swept.summation_density_vectorised(pos, mass, h, 2, pairs, kernel=KERNEL)
    standard = {}
    for local in (3, 64, 256):
        bad = swept.summation_density_swept(pos, mass, h, 2, pairs, kernel=KERNEL,
                                            local_size=local, mask_tail=False)
        standard[local] = {"rel_err": rel_err(bad, ref),
                           "above_owner_gate": bool(rel_err(bad, ref) > GATE_OWNER)}
    # degeneracy: the last row to be clamped onto has no neighbours at all
    p = torch.tensor([[0.0, 0.0], [50.0, 0.0], [0.2, 0.0]], dtype=D)
    m = torch.full((3,), 1.0, dtype=D)
    hh = torch.full((3,), 0.5, dtype=D)
    pr = torch.tensor([[0, 2]], dtype=torch.long)
    ref3 = swept.summation_density_vectorised(p, m, hh, 2, pr)
    bad3 = swept.summation_density_swept(p, m, hh, 2, pr, local_size=4,
                                         mask_tail=False, bound=2)
    return {
        "standard_case": {str(k): v for k, v in standard.items()},
        "degenerate_isolated_last_row": {
            "rel_err": rel_err(bad3[:2], ref3[:2]),
            "control_is_vacuous": bool(rel_err(bad3[:2], ref3[:2]) < GATE_TARGET),
            "note": "last swept row has no neighbours -> the block's pair total is "
                    "0 and the unmasked contribution silently vanishes",
        },
    }


def s4_edges() -> list[dict]:
    out = []

    def run(name, **kw):
        pos = torch.tensor([[0.0, 0.0], [0.4, 0.0]], dtype=D)
        mass = torch.tensor([1.0, 1.0], dtype=D)
        h = torch.tensor([0.5, 0.5], dtype=D)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        try:
            got = swept.summation_density_swept(pos, mass, h, 2, pairs, **kw).tolist()
            out.append({"case": name, "ok": True, "value": [round(v, 9) for v in got]})
        except Exception as exc:                                   # noqa: BLE001
            out.append({"case": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    run("bound=0", local_size=4, bound=0)
    run("bound>n", local_size=4, bound=5)
    run("local=64 n=2 (62 idle)", local_size=64)
    empty = torch.empty((0, 2), dtype=torch.long)
    got = swept.summation_density_swept(torch.zeros((3, 2), dtype=D),
                                        torch.ones(3, dtype=D), 0.5, 2, empty, local_size=8)
    out.append({"case": "empty pair list", "ok": True,
                "value": [round(v, 9) for v in got.tolist()]})
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def figures(results: dict, out: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    import numpy as np
    amps = sorted({r["h_modulation"] for r in results["s2_variable_h"]})
    dxs = sorted({r["dx"] for r in results["s2_variable_h"]}, reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax = axes[0]
    for dx, colour in zip(dxs, ("#c0392b", "#e67e22", "#1f6feb")):
        rows = [next(r for r in results["s2_variable_h"]
                     if r["dx"] == dx and r["h_modulation"] == x) for x in amps]
        ax.semilogy(amps, [max(r["convention_gap_max"] * 100, 1e-13) for r in rows],
                    "o-", color=colour,
                    label="convention gap, dx=%g m (n=%d)" % (dx, rows[0]["particles"]))
        ax.semilogy(amps, [max(r["launch_invariance_max"] * 100, 1e-13) for r in rows],
                    "s--", color=colour, lw=0.9, alpha=0.6,
                    label="launch invariance, dx=%g m" % dx)
    ax.axhline(GATE_OWNER * 100, ls="--", color="k", lw=1.2, label="owner gate 15 %")
    ax.axhline(3.0, ls=":", color="0.4", lw=1.0, label="3 % line")
    ax.set_xlabel("smoothing-length modulation amp   (h = 1.3 dx (1 + amp sin 2 pi x / L))")
    ax.set_ylabel("max relative density difference  [%]")
    ax.set_title("convention gap (solid) vs launch invariance (dashed, 1e-13 floor)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.0, loc="lower right", ncol=2)

    ax = axes[1]
    grid = np.full((len(dxs), len(amps)), np.nan)
    for i, dx in enumerate(dxs):
        for j, x in enumerate(amps):
            grid[i, j] = next(r for r in results["s2_variable_h"]
                              if r["dx"] == dx and r["h_modulation"] == x)["convention_gap_max"] * 100
    im = ax.imshow(grid, cmap="inferno", aspect="auto", origin="upper")
    ax.set_xticks(range(len(amps)), ["%.2f" % x for x in amps])
    ax.set_yticks(range(len(dxs)), ["%g" % dx for dx in dxs])
    ax.set_xlabel("h modulation amplitude")
    ax.set_ylabel("dx [m]")
    ax.set_title("convention gap [%]  (contour = 15 % gate)")
    try:
        ax.contour(grid, levels=[GATE_OWNER * 100], colors="w", linewidths=1.4)
    except Exception:                                    # pragma: no cover
        pass
    for i in range(len(dxs)):
        for j in range(len(amps)):
            ax.text(j, i, "%.1f" % grid[i, j], ha="center", va="center",
                    color="w" if grid[i, j] < 20 else "k", fontsize=7)
    fig.colorbar(im, ax=ax, label="max rel. difference vs shipped reference [%]")
    fig.tight_layout()
    path = out / "fig_convention_gap_grid.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": path.name,
                 "caption": "Kernel-evaluation convention vs launch geometry. Solid lines / heat map: "
                            "max relative density difference between the padded-masked sweep and the "
                            "shipped symmetric-h reference as the smoothing length is modulated "
                            "(h = 1.3 dx (1 + amp sin 2 pi x / L)); the gap grows with the modulation "
                            "and with coarseness and crosses the owner's 15 % gate at dx=0.04 m, "
                            "amp=0.9. Dashed lines: the same sweep against a reference that shares its "
                            "per-row h convention -- bit-exact at every work-group size, so the launch "
                            "geometry itself contributes nothing. The solid/dashed split is the finding: "
                            "the equivalence claim holds for a uniform h only."})

    # field map of the convention difference at the largest amplitude
    case = build_particles(0.9)
    sym = swept.summation_density_vectorised(case["pos"], case["mass"], case["h"], 2,
                                             case["pairs"], kernel=KERNEL)
    rowh = swept.summation_density_vectorised_row_h(case["pos"], case["mass"], case["h"], 2,
                                                    case["pairs"], kernel=KERNEL)
    diff = ((rowh - sym).abs() / sym.abs()).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    sc = axes[0].scatter(case["pos"][:, 0], case["pos"][:, 1], c=(rowh / RHO0).numpy(),
                         s=26, cmap="viridis", marker="s")
    axes[0].set_aspect("equal")
    axes[0].set_title("$\\rho/\\rho_0$, per-row-h operator (amp=0.9)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.colorbar(sc, ax=axes[0], label="$\\rho/\\rho_0$")
    sc2 = axes[1].scatter(case["pos"][:, 0], case["pos"][:, 1], c=diff * 100.0, s=26,
                          cmap="inferno", marker="s", vmin=0.0)
    axes[1].set_aspect("equal")
    axes[1].set_title("|row-h − symmetric| / symmetric  [%]  (same particles)")
    axes[1].set_xlabel("x [m]")
    fig.colorbar(sc2, ax=axes[1], label="relative difference [%]")
    fig.tight_layout()
    path = out / "fig_convention_field_map.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": path.name,
                 "caption": "Spatial distribution of the h-convention difference at amp=0.9. "
                            "Left: the per-row-h (one-sided) operator the sweep implements. "
                            "Right: its relative difference from the shipped symmetric operator; "
                            "the error is largest where the smoothing length has its steepest "
                            "gradient, i.e. it is a convention artefact and not a launch defect."})
    return figs


# --------------------------------------------------------------------------- #
# markdown + main
# --------------------------------------------------------------------------- #


def markdown(results: dict) -> str:
    lines = []
    a = lines.append
    a("### S1  uniform h: sweep vs shipped reference (replays the original claim)")
    a("")
    a("| work-group | padded | idle | max rel err | ULP | bit-exact vs shipped | bit-exact vs row-h |")
    a("|---|---|---|---|---|---|---|")
    for r in results["s1_uniform"]:
        a(f"| {r['local_size']} | {r['launch']['padded_global_size']} | "
          f"{r['launch']['idle_work_items']} | {r['rel_err_vs_shipped']:.3e} | "
          f"{r['ulp_vs_shipped']} | {r['bit_exact_vs_shipped']} | {r['bit_exact_vs_row_h']} |")
    a("")
    a("### S2  spatially varying smoothing length (convention gap, %)")
    a("")
    amps = sorted({r["h_modulation"] for r in results["s2_variable_h"]})
    dxs = sorted({r["dx"] for r in results["s2_variable_h"]}, reverse=True)
    a("| dx [m] | n | " + " | ".join(f"amp={x:.2f}" for x in amps) + " |")
    a("|---|---|" + "---|" * len(amps))
    for dx in dxs:
        row = [r for r in results["s2_variable_h"] if r["dx"] == dx]
        cells = []
        for x in amps:
            m = next(r for r in row if r["h_modulation"] == x)
            flag = " **>15%**" if m["exceeds_owner_gate"] else ""
            cells.append(f"{m['convention_gap_max'] * 100:.2f}{flag}")
        a(f"| {dx} | {row[0]['particles']} | " + " | ".join(cells) + " |")
    a("")
    a("Launch-geometry invariance against the convention-matched reference "
      "(max rel. err over local_size = 1, 8, 64, 100):")
    a("")
    a("| dx | amp | max rel err vs row-h ref | all bit-exact | rho/rho0 range | h/dx max |")
    a("|---|---|---|---|---|---|")
    for r in results["s2_variable_h"]:
        a(f"| {r['dx']} | {r['h_modulation']:.2f} | {r['launch_invariance_max']:.3e} | "
          f"{r['launch_invariance_all_bitexact']} | "
          f"{r['rho_over_rho0_min']:.3f}–{r['rho_over_rho0_max']:.3f} | {r['h_over_dx_max']:.2f} |")
    a("")
    a("### S3  negative control (tail mask removed)")
    a("")
    a("| case | max rel err | above 15 % |")
    a("|---|---|---|")
    for local, v in results["s3_control"]["standard_case"].items():
        a(f"| uniform h, local_size={local} | {v['rel_err']:.3f} | {v['above_owner_gate']} |")
    d = results["s3_control"]["degenerate_isolated_last_row"]
    a(f"| degenerate: isolated last swept row | {d['rel_err']:.3e} | {d['control_is_vacuous']} (vacuous) |")
    a("")
    a("### S4  edge cases")
    a("")
    a("| case | ok | value / error |")
    a("|---|---|---|")
    for r in results["s4_edges"]:
        a(f"| {r['case']} | {r['ok']} | {r.get('value', r.get('error'))} |")
    return "\n".join(lines)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(D)

    uniform = build_particles(0.0)
    results = {
        "anchor": "XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE",
        "artefact": "verification of the shipped deliverable (m19962)",
        "criteria": {"owner_gate": GATE_OWNER, "engineering_target": GATE_TARGET,
                     "note": "thresholds same as the deliverable under test"},
        "geometry": {"dx": DX, "h0": H0, "h_over_dx": 1.3, "kernel": KERNEL,
                     "particles": uniform["n"],
                     "pairs": int(uniform["pairs"].shape[0]),
                     "pairs_per_particle": uniform["pairs_per_particle"],
                     "domain": "[0,0.4] x [0,0.2] m lattice, 3 wall layers",
                     "core_tensor_sph": "does not apply: one summation density pass, no time integration"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "machine": platform.machine(), "threads": torch.get_num_threads()},
        "s1_uniform": s1_uniform(uniform),
        "s2_variable_h": s2_variable_h([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]),
        "s3_control": s3_control(uniform),
        "s4_edges": s4_edges(),
    }
    results["figures"] = figures(results, out)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "summary.md").write_text(markdown(results) + "\n")
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, list)}, indent=1))
    print(markdown(results))
    return results


if __name__ == "__main__":
    main()