"""Viscous-time convergence of the plane-Couette wall-shear reading.

Why this exists
---------------
``benchmarks/couette/out/results.json`` (the artefact the method-selection
registry cites for ``wcsph_wall_no_slip``) was produced by ``run.py --quick``,
which overrides the ``DESIGN.md`` terminal time ``t_end = 1.2 H^2/nu`` down to
``0.15 s = 0.3 H^2/nu``.  At that time the flow is still in its startup
transient, so the *reading* ``|1 - tau_SPH/tau_exact|`` is 32.6 % and the
artefact's own verdict is FAIL.  An independent audit
(``docs/audit_couette_wall_shear_reading.md``, m16438) showed the deficit is the
transient, not the wall model, but the registry never absorbed that finding and
still banks the 32.6 % as the scheme's worst evidence.

This driver banks the missing evidence properly: the same solver, lattice and
estimator, swept over viscous times ``t_end/(H^2/nu)`` in
``(0.6, 1.2, 2.4, 3.6)`` at ``H/dx = 8`` and ``16``.  Each run records the
wall-shear reading together with the run's own steadiness diagnostics
(``steady_drift``), so a reading can be marked *admissible* only when the flow
is actually steady -- that is the precondition the registry records need.

Run::

    PYTHONPATH=src python benchmarks/couette/viscous_time_convergence.py
    PYTHONPATH=src python benchmarks/couette/viscous_time_convergence.py --cached
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# matplotlib is imported lazily, inside ``_pyplot``: the compute nodes that run
# this sweep do not ship matplotlib, and a module-level import there killed the
# whole scan before a single run happened (cluster job cj2857, rc=1,
# ``ModuleNotFoundError: No module named 'matplotlib'``) while the queue still
# reported the job as succeeded.  Plotting is a *reporting* step; it must never
# be a precondition for producing the numbers.

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import run as couette  # noqa: E402  (benchmarks/couette/run.py)

OUT_DIR = HERE / "out_design"
RESULT_PATH = OUT_DIR / "viscous_time_convergence.json"
FIGURE_PATH = OUT_DIR / "viscous_time_convergence.png"

#: Viscous times swept, and the two resolutions of the committed ladder.
VISCOUS_TIMES_SWEPT = (0.6, 1.2, 2.4, 3.6)
RESOLUTIONS = (8, 16)
AVERAGE_FRACTION = couette.AVERAGE_FRACTION


def run_point(resolution: int, viscous_times: float, device) -> dict:
    nu = couette.VISCOSITY
    height = couette.HEIGHT
    t_end = viscous_times * height ** 2 / nu
    t_average = AVERAGE_FRACTION * t_end
    label = f"r{resolution}-vt{viscous_times:g}"
    print(f"[couette-conv] {label}: t_end={t_end:.3f}s "
          f"({viscous_times:g} H^2/nu)", flush=True)
    started = time.time()
    res = couette.run_one(resolution, nu, couette.WALL_SPEED, couette.FRICTION,
                          height, t_end, t_average, device, label=label)
    m = res["metrics"]
    row = {
        "label": label,
        "resolution_H_over_dx": resolution,
        "viscous_times": float(viscous_times),
        "t_end": t_end,
        "steps": int(res["steps"]),
        "dx": height / resolution,
        "wall_shear_ratio_bottom": float(m["wall_shear_ratio"]),
        "wall_shear_ratio_top": float(m["wall_shear_ratio_top"]),
        "wall_shear_error_pct_bottom": abs(1.0 - float(m["wall_shear_ratio"])) * 100.0,
        "wall_shear_error_pct_top": abs(1.0 - float(m["wall_shear_ratio_top"])) * 100.0,
        "line_fit_slope_ratio": float(m["line_fit_slope_ratio"]),
        "interior_l2_pct": float(m["interior_l2"]) * 100.0,
        "flux_ratio": float(m["flux_ratio"]),
        "steady_drift": float(res["steady_drift"]),
        "steady_jitter": float(res["steady_jitter"]),
        "wall_clock_s": float(time.time() - started),
    }
    row["admissible"] = bool(row["steady_drift"] <= couette.GATES["steady_drift"])
    print(f"[couette-conv]   tau_bot={row['wall_shear_ratio_bottom']:.4f} "
          f"({row['wall_shear_error_pct_bottom']:.2f} %) "
          f"tau_top={row['wall_shear_ratio_top']:.4f} "
          f"L2={row['interior_l2_pct']:.2f} % drift={row['steady_drift']:.2e} "
          f"admissible={row['admissible']}", flush=True)
    return row


def summarise(points: list[dict],
              resolutions: tuple[int, ...] | None = None) -> dict:
    """Per resolution: does the reading stop moving once the drift gate passes?"""
    per_res: dict[str, dict] = {}
    for resolution in (resolutions or RESOLUTIONS):
        rows = sorted((p for p in points if p["resolution_H_over_dx"] == resolution),
                      key=lambda p: p["viscous_times"])
        admissible = [r for r in rows if r["admissible"]]
        rest = admissible or rows
        spread = (max(r["wall_shear_error_pct_bottom"] for r in rest)
                  - min(r["wall_shear_error_pct_bottom"] for r in rest))
        per_res[str(resolution)] = {
            "n_points": len(rows),
            "converged_error_pct": rest[-1]["wall_shear_error_pct_bottom"],
            "spread_over_admissible_pct": spread,
            "admissible_viscous_times": [r["viscous_times"] for r in admissible],
            "startup_reading": rows[0]["wall_shear_error_pct_bottom"] if rows else None,
            "startup_viscous_times": rows[0]["viscous_times"] if rows else None,
        }
    return per_res


def _pyplot():
    """Import matplotlib lazily; raise a readable error when it is absent."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised on compute nodes
        raise RuntimeError(
            "matplotlib is not installed; the numbers are still computed and "
            "written to the JSON artefact -- pass --no-figure on headless nodes"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def make_figure(points: list[dict], summary: dict, out: Path) -> Path:
    plt = _pyplot()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {8: "tab:blue", 16: "tab:red"}
    for resolution in RESOLUTIONS:
        rows = sorted((p for p in points if p["resolution_H_over_dx"] == resolution),
                      key=lambda p: p["viscous_times"])
        vt = [r["viscous_times"] for r in rows]
        ax.plot(vt, [r["wall_shear_error_pct_bottom"] for r in rows], "o-",
                color=colors[resolution], label=f"H/dx = {resolution} (bottom wall)")
        ax.plot(vt, [r["wall_shear_error_pct_top"] for r in rows], "s--",
                color=colors[resolution], alpha=0.55,
                label=f"H/dx = {resolution} (moving wall)")
        ax2.semilogy(vt, [max(r["steady_drift"], 1e-6) for r in rows], "o-",
                     color=colors[resolution], label=f"H/dx = {resolution}")
    ax.axhline(couette.GATES["wall_shear_rel"] * 100.0, color="k", ls=":", lw=1.2,
               label="artefact gate 5 %")
    ax.axhline(15.0, color="grey", ls="-.", lw=1.2, label="owner standard 15 %")
    ax.axvline(couette.VISCOUS_TIMES, color="green", ls=":", lw=1.0,
               label="DESIGN.md $t_{end}$ = 1.2 $H^2/\\nu$")
    ax.set_xlabel("viscous time  $t_{end}/(H^2/\\nu)$")
    ax.set_ylabel("wall-shear error  $|1-\\tau_{SPH}/\\tau_{exact}|$  [%]")
    ax.set_title("Couette wall-shear reading vs run length")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    ax2.axhline(couette.GATES["steady_drift"], color="k", ls=":", lw=1.2,
                label="drift gate 1 %")
    ax2.set_xlabel("viscous time  $t_{end}/(H^2/\\nu)$")
    ax2.set_ylabel("steady-drift diagnostic")
    ax2.set_title("Steadiness of the same runs")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cached", action="store_true",
                        help="reuse points already present in the result file")
    parser.add_argument("--no-figure", action="store_true",
                        help="skip the PNG (headless compute nodes have no matplotlib)")
    parser.add_argument("--figure-only", action="store_true",
                        help="re-plot an existing result file without running the solver")
    parser.add_argument("--resolutions", default=",".join(str(r) for r in RESOLUTIONS),
                        help="comma-separated H/dx rungs (default: the committed ladder "
                             f"{','.join(str(r) for r in RESOLUTIONS)})")
    parser.add_argument("--viscous-times", default=",".join(f"{v:g}" for v in VISCOUS_TIMES_SWEPT),
                        help="comma-separated t_end/(H^2/nu) tiers "
                             f"(default: the committed sweep {VISCOUS_TIMES_SWEPT})")
    parser.add_argument("--out", default=str(RESULT_PATH),
                        help="result artefact path (default: the committed sweep; "
                             "an extension ladder must use its own file)")
    args = parser.parse_args(argv)
    resolutions = tuple(int(r) for r in str(args.resolutions).split(",") if r.strip())
    viscous_times = tuple(float(v) for v in str(args.viscous_times).split(",") if v.strip())
    result_path = Path(args.out)
    figure_path = result_path.with_suffix(".png")

    if args.figure_only:
        doc = json.loads(result_path.read_text())
        make_figure(doc["points"], doc["summary"], figure_path)
        print(f"[couette-conv] re-plotted {figure_path}", flush=True)
        return 0

    import torch
    torch.set_num_threads(args.threads)
    device = couette.resolve_device(args.device)

    cached: dict[str, dict] = {}
    if args.cached and result_path.exists():
        doc = json.loads(result_path.read_text())
        cached = {p["label"]: p for p in doc.get("points", [])}

    points: list[dict] = []
    for tier in viscous_times:
        for resolution in resolutions:
            label = f"r{resolution}-vt{tier:g}"
            if label in cached:
                print(f"[couette-conv] {label}: cached", flush=True)
                points.append(cached[label])
                continue
            points.append(run_point(resolution, tier, device))

    summary = summarise(points, resolutions)
    payload = {
        "case": "plane_couette_channel",
        "quantity": "wall-shear reading |1 - tau_SPH/tau_exact| vs run length",
        "reference": couette.REFERENCE_NOTE,
        "height": couette.HEIGHT,
        "viscosity": couette.VISCOSITY,
        "wall_speed": couette.WALL_SPEED,
        "sound_speed": couette.SOUND_SPEED,
        "rho0": couette.RHO0,
        "friction": couette.FRICTION,
        "gates": dict(couette.GATES),
        "design_viscous_times": couette.VISCOUS_TIMES,
        "average_fraction": AVERAGE_FRACTION,
        "viscous_times_swept": list(viscous_times),
        "resolutions": list(resolutions),
        "generated_by": "benchmarks/couette/viscous_time_convergence.py",
        "points": points,
        "summary": summary,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # Numbers first: the JSON artefact is the evidence, the PNG is reporting.
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        print(f"[couette-conv] wrote {result_path.relative_to(REPO)}", flush=True)
    except ValueError:  # --out outside the repo root (tests, scratch dirs)
        print(f"[couette-conv] wrote {result_path}", flush=True)
    if args.no_figure:
        print("[couette-conv] --no-figure: PNG skipped", flush=True)
    else:
        try:
            make_figure(points, summary, figure_path)
        except RuntimeError as exc:
            # Headless node: keep the numbers, report the missing plot.
            print(f"[couette-conv] figure skipped: {exc}", flush=True)
        else:
            try:
                print(f"[couette-conv] wrote {figure_path.relative_to(REPO)}", flush=True)
            except ValueError:
                print(f"[couette-conv] wrote {figure_path}", flush=True)
    for resolution, block in summary.items():
        print(f"[couette-conv] H/dx={resolution}: converged "
              f"{block['converged_error_pct']:.2f} % "
              f"(spread over admissible points {block['spread_over_admissible_pct']:.2f} pp, "
              f"startup {block['startup_reading']:.2f} % at "
              f"{block['startup_viscous_times']:g} H^2/nu)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())