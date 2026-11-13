"""Force-from-fluid audit of the ``fsi2`` (elastic-gate) geometry.

Runs the rigid-gate dam break in five configurations and reports, for each one,

* the three load routes on the gate and on the floor (pair / panel / analytic),
* the closed-form reference (:func:`tensorsph.force_from_fluid.hydrostatic_reference`),
* the control-volume momentum budget (residual, impulse, momentum change),
* the diagnostic that arbitrates between them: the fluid's own acceleration,
  ``sum_i m_i a_i - sum_i m_i g``, which *must* equal the reaction the walls
  report if the load route is consistent.

Outputs ``artifacts/force_from_fluid/budget_cases.json`` (numbers) and
``fig_budget.png`` (four panels).  No external reference is needed: every
comparison is against a closed-form hydrostatic integral or against the solver's
own momentum balance.

Usage::

    python benchmarks/force_from_fluid/run_budget.py [--dx 0.02] [--steps 1200]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.force_from_fluid import (  # noqa: E402
    GateCaseConfig,
    hydrostatic_reference,
    run_gate_case,
)

CASES = {
    # name: (config overrides, description)
    "conservative": ({}, "default solver settings (viscosity 0.01, no diffusion, "
                         "free-slip wall)"),
    "inviscid": ({"viscosity": 0.0}, "viscosity off: isolates the momentum "
                                     "leak of the asymmetric viscous Laplacian"),
    "diffusion": ({"density_diffusion": 0.1}, "Molteni density diffusion on"),
    "friction": ({"wall_friction": 1.0}, "no-slip wall friction on"),
    "no_gate": ({"_gate": False}, "plain dam break, no gate particles"),
}


def static_case(cfg: GateCaseConfig) -> dict:
    """Forces of the initial (hydrostatic) state: step 0 only.

    Also repeated on a two-level resolution ladder, because the size of the
    deficit (and whether it shrinks with ``dx``) is what separates a
    boundary-support effect from a mis-normalised force formula.
    """
    out = run_gate_case(cfg)
    ref = hydrostatic_reference(cfg)
    thrust, weight = ref["vertical_wall_thrust"], ref["fluid_weight"]
    reaction = {k: v[-1] for k, v in out["reaction_forces"].items()}
    panel = {k: v[-1] for k, v in out["panel_forces"].items()}
    return {
        "reference": ref,
        "reaction": reaction,
        "panel": panel,
        "pair_gate_over_thrust": reaction["gate"][0] / thrust,
        "panel_gate_over_thrust": panel["gate"][0] / thrust,
        "pair_gate_vs_panel": (reaction["gate"][0] / panel["gate"][0]
                               if panel["gate"][0] else float("nan")),
        "pair_floor_over_weight": reaction["floor"][1] / weight,
        "panel_floor_over_weight": panel["floor"][1] / weight,
        "vertical_total_over_weight": (
            (reaction["floor"][1] + reaction["gate"][1]
             + reaction["left"][1]) / weight),
        "resolution": resolution_ladder(cfg),
    }


def resolution_ladder(cfg: GateCaseConfig,
                      levels=(0.02, 0.01, 0.005)) -> list:
    """Static gate/floor loads at several spacings (pair and panel routes)."""
    rows = []
    for dx in levels:
        fine = GateCaseConfig(**{**cfg.__dict__, "dx": dx,
                                 "dt": min(cfg.dt, 0.25 * dx / cfg.sound_speed)})
        coarse_ref = hydrostatic_reference(fine)
        o = run_gate_case(fine)
        r = {k: v[-1] for k, v in o["reaction_forces"].items()}
        pl = {k: v[-1] for k, v in o["panel_forces"].items()}
        rows.append({
            "dx": dx,
            "particles": o["summary"]["particle_count"],
            "pair_gate_over_thrust": r["gate"][0] / coarse_ref["vertical_wall_thrust"],
            "panel_gate_over_thrust": pl["gate"][0] / coarse_ref["vertical_wall_thrust"],
            "pair_floor_over_weight": r["floor"][1] / coarse_ref["fluid_weight"],
            "panel_floor_over_weight": pl["floor"][1] / coarse_ref["fluid_weight"],
            "vertical_total_over_weight": (
                (r["floor"][1] + r["gate"][1] + r["left"][1])
                / coarse_ref["fluid_weight"]),
        })
        print("  dx=%.3f  pair gate %.4f panel gate %.4f vertical %.4f"
              % (dx, rows[-1]["pair_gate_over_thrust"],
                 rows[-1]["panel_gate_over_thrust"],
                 rows[-1]["vertical_total_over_weight"]))
    return rows


def transient_case(cfg: GateCaseConfig, gate: bool = True) -> dict:
    out = run_gate_case(cfg, gate=gate, record_every=max(1, cfg.steps // 200))
    ref = hydrostatic_reference(cfg)
    thrust, weight = ref["vertical_wall_thrust"], ref["fluid_weight"]
    times = out["times"]
    reaction = out["reaction_forces"]
    panel = out["panel_forces"]
    series = {
        "time": times,
        "pair_gate_over_thrust": [v[0] / thrust for v in reaction["gate"]],
        "panel_gate_over_thrust": [v[0] / thrust for v in panel["gate"]],
        "vertical_total_over_weight": [
            (reaction["floor"][k][1] + reaction["gate"][k][1]
             + reaction["left"][k][1]) / weight for k in range(len(times))],
        "front_x": out["front"],
    }
    tail = slice(len(times) // 3, None)   # skip the startup transient
    summary = dict(out["summary"])
    summary["gate_over_thrust_mean_tail"] = float(
        sum(series["pair_gate_over_thrust"][tail])
        / max(1, len(series["pair_gate_over_thrust"][tail])))
    summary["vertical_total_over_weight_mean_tail"] = float(
        sum(series["vertical_total_over_weight"][tail])
        / max(1, len(series["vertical_total_over_weight"][tail])))
    # time mean of the fluid's own net acceleration, the momentum-route arbiter
    mom = torch.tensor(out["momentum"])
    if mom.shape[0] > 2:
        dt = cfg.dt
        acc_net = (mom[-1] - mom[0]) / (dt * (mom.shape[0] - 1))
        summary["net_acceleration_x"] = float(acc_net[0])
        summary["net_acceleration_y_over_g"] = float(
            acc_net[1] / (-cfg.gravity * out["budget"].total_mass))
    return {"summary": summary, "series": series,
            "reference": ref, "config": cfg}


def figure(results: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib unavailable, skipping figure: {exc}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))

    ax = axes[0][0]
    tr = results["transient"]["conservative"]["series"]
    ax.plot(tr["time"], tr["pair_gate_over_thrust"], label="pair route",
            color="C0", lw=1.6)
    ax.plot(tr["time"], tr["panel_gate_over_thrust"], label="panel route",
            color="C1", lw=1.2, ls="--")
    ax.axhline(1.0, color="k", lw=0.8, ls=":",
               label=r"analytic $0.5\rho g H^2$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$F_{x,\mathrm{gate}} / (0.5\rho g H^2)$")
    ax.set_title("gate thrust, two routes (dx = %.3g m, %d steps)"
                 % (results["dx"], results["steps"]))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(tr["time"], tr["vertical_total_over_weight"], color="C2", lw=1.6,
            label="reaction / fluid weight")
    ax.axhline(1.0, color="k", lw=0.8, ls=":", label="fluid weight")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$F_{y,\mathrm{total}} / (Mg)$")
    ax.set_title("vertical support (momentum route arbiter)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    names = [n for n in CASES if n in results["transient"]]
    vals = [results["transient"][n]["summary"]["residual_relative"]
            for n in names]
    ax.bar(range(len(names)), vals, color="C3")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("momentum budget residual / accounted impulse")
    ax.set_title("who leaks momentum (log scale)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    st = results["static"]
    labels, pair, panel = [], [], []
    for face in ("gate", "floor"):
        labels.append(face)
        norm = (st["reference"]["vertical_wall_thrust"] if face == "gate"
                else st["reference"]["fluid_weight"])
        pair.append(st["reaction"][face][0 if face == "gate" else 1] / norm)
        panel.append(st["panel"][face][0 if face == "gate" else 1] / norm)
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], pair, width=0.4, label="pair route", color="C0")
    ax.bar([i + 0.2 for i in x], panel, width=0.4, label="panel route",
           color="C1")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("load / analytic")
    ax.set_title("initial (hydrostatic) state, step 0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("TensorSPH force-from-fluid audit (rigid gate, fsi2 geometry)")
    fig.tight_layout()
    fig.savefig(out / "fig_budget.png", dpi=130)
    plt.close(fig)


def merge(out: Path, dx: float, steps: int) -> dict:
    """Assemble the per-case partial files into one result and draw the figure."""
    results: dict = {"transient": {}, "cases": {}, "dx": dx, "steps": steps,
                     "static": {}}
    for path in sorted(out.glob("partial_*.json")):
        blob = json.loads(path.read_text())
        name = path.stem.replace("partial_", "")
        if name == "static":
            results["static"] = blob["static"]
            continue
        results["transient"][name] = blob["transient"]
        results["cases"][name] = blob["case"]
    ordered = {n: results["transient"][n] for n in CASES
               if n in results["transient"]}
    results["transient"] = ordered
    (out / "budget_cases.json").write_text(json.dumps(results, indent=2,
                                                      default=float))
    figure(results, out)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cases", default=None,
                    help="comma-separated subset of %s" % ",".join(CASES))
    ap.add_argument("--merge", action="store_true",
                    help="only assemble the partial files and draw the figure")
    args = ap.parse_args()
    out = (args.out or Path(__file__).resolve().parents[2]
           / "artifacts" / "force_from_fluid")
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        res = merge(out, args.dx, args.steps)
        print("merged cases:", list(res["transient"]))
        return 0

    base = dict(dx=args.dx, steps=args.steps, pool_width=1.0, tank_height=0.6,
                output_every=max(1, args.steps // 10))
    wanted = list(CASES) if args.cases is None else args.cases.split(",")
    if "static" in wanted:
        st = static_case(GateCaseConfig(**{**base, "steps": 0}))
        (out / "partial_static.json").write_text(
            json.dumps({"static": st}, indent=2, default=float))
        print("static: pair/analytic=%.4f panel/analytic=%.4f pair/panel=%.4f "
              "vertical/weight=%.4f"
              % (st["pair_gate_over_thrust"], st["panel_gate_over_thrust"],
                 st["pair_gate_vs_panel"], st["vertical_total_over_weight"]))
        wanted = [w for w in wanted if w != "static"]

    for name in wanted:
        overrides, description = CASES[name]
        gate = overrides.get("_gate", True)
        kw = {k: v for k, v in overrides.items() if not k.startswith("_")}
        cfg = GateCaseConfig(**base, **kw)
        print(f"[{name}] {description}")
        res = transient_case(cfg, gate=gate)
        s = res["summary"]
        payload = {k: v for k, v in res.items()
                   if k in ("series", "reference")}
        payload["summary"] = res["summary"]
        (out / f"partial_{name}.json").write_text(json.dumps(
            {"transient": payload,
             "case": {"description": description,
                      "summary": {k: v for k, v in s.items()
                                  if isinstance(v, (int, float, bool, str))}}},
            indent=2, default=float))
        print("  residual_relative=%.3e  gate/thrust(tail)=%.4f  "
              "vertical/weight(tail)=%.4f  clamp=%d"
              % (s["residual_relative"], s["gate_over_thrust_mean_tail"],
                 s["vertical_total_over_weight_mean_tail"],
                 s["boundary_clamp_events"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())