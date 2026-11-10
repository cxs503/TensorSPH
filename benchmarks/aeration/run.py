"""Driver for the air-entrainment benchmark suite (campaign node P165W2-N04).

Runs the three case modules, the air-volume conservation checks and the test
suite, then writes ``report.json`` (all criterion numbers) and the figures used
by the report.  Every number in ``docs/benchmarks/aeration.md`` is injected from
this JSON by ``make_doc.py`` -- nothing is copied by hand.

Usage::

    python benchmarks/aeration/run.py            # numbers + figures
    python benchmarks/aeration/run.py --no-figs  # numbers only
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

import bubble_rise_velocity  # noqa: E402
import drag_reduction_aerated  # noqa: E402
import hydraulic_jump_aeration  # noqa: E402
from tensorsph.aeration import (  # noqa: E402
    C_CRIT,
    CAVITY_FRACTION,
    derived_crit_constant,
    entrainment_flux,
    inception_velocity,
    onset_velocity_by_bisection,
    rise_velocity,
    transport_bubbles,
    void_fraction_from_bubbles,
)

CRITERIA = {
    "C1": ("inception Froude / criterion constant vs Chanson", 0.10),
    "C2": ("steady void fraction profile vs erf anchor (relative L2)", 0.15),
    "C3": ("friction reduction vs Chisholm multiplier (relative)", 0.20),
    "C4": ("bubble rise velocity vs Mendelson, R = 0.5-5 mm (relative)", 0.10),
    "C5": ("air volume drift per characteristic time (relative)", 0.05),
    "C6": ("pytest suite for tensorsph.aeration (count, minimum)", 15.0),
}

CHARACTERISTIC_TIME = 2.0
PRESCRIBED_VOID_FRACTION = 0.05


def inception_case() -> dict:
    """C1: energy-balance constant vs the empirical constant, plus the onset check."""
    depth, angle = 0.05, math.radians(15.0)
    derived = derived_crit_constant(CAVITY_FRACTION)
    algebraic = inception_velocity(depth, angle)
    numerical = onset_velocity_by_bisection(depth, angle)
    slopes = [5.0, 10.0, 15.0, 30.0, 45.0]
    table = []
    for degrees in slopes:
        theta = math.radians(degrees)
        v_i = inception_velocity(depth, theta)
        table.append(
            {
                "slope_deg": degrees,
                "inception_velocity": v_i,
                "inception_froude": v_i / math.sqrt(9.81 * depth),
                "parameter_at_onset": float(
                    entrainment_flux(v_i * 1.001, depth, theta)
                ),
            }
        )
    return {
        "cavity_fraction": CAVITY_FRACTION,
        "empirical_constant": C_CRIT,
        "derived_constant": derived,
        "constant_rel_error": abs(derived - C_CRIT) / C_CRIT,
        "onset_rel_error": abs(numerical - algebraic) / algebraic,
        "reference_depth": depth,
        "reference_angle_deg": 15.0,
        "inception_velocity_at_reference": algebraic,
        "slope_table": table,
    }


def conservation_case() -> dict:
    """C5: air volume bookkeeping of the Lagrangian cloud and the SPH mapping."""
    generator = torch.Generator().manual_seed(9315)
    count = 4000
    depth_scale = 0.2
    steps = 1500
    dt = CHARACTERISTIC_TIME / steps
    drift = float(rise_velocity(1.5e-4))
    depths = torch.rand(count, dtype=torch.float64, generator=generator) * depth_scale

    chunks = []
    y = depths.clone()
    for _ in range(4):  # four characteristic times
        y, _ = transport_bubbles(
            y, steps=steps, dt=dt, drift=drift, diffusivity=0.05, generator=generator
        )
        chunks.append({"time": CHARACTERISTIC_TIME * (len(chunks) + 1), "count": int(y.numel())})
    drift_rel = abs(chunks[-1]["count"] - count) / count

    # Escape bookkeeping: in-domain + escaped must close the initial volume.
    escaped_y, escaped = transport_bubbles(
        depths,
        steps=steps,
        dt=dt,
        drift=drift,
        diffusivity=0.05,
        generator=generator,
        surface="escape",
    )
    in_domain = float((~escaped).float().mean())
    escaped_fraction = float(escaped.float().mean())
    closure_error = abs(in_domain + escaped_fraction - 1.0)

    # SPH map: kernel summation of the same cloud must return the prescribed
    # void fraction (particle average; the scatter is a resolution property).
    box = 0.5
    cloud = 20000
    bubble_vol = PRESCRIBED_VOID_FRACTION * box * box / cloud
    bubbles = torch.rand(cloud, 2, dtype=torch.float64, generator=generator) * box
    axis = torch.linspace(0.07, box - 0.07, 12, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    fluid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    alpha = void_fraction_from_bubbles(
        bubbles, torch.full((cloud,), bubble_vol), fluid, 0.02
    )
    sph_rel_error = abs(float(alpha.mean()) - PRESCRIBED_VOID_FRACTION) / PRESCRIBED_VOID_FRACTION

    return {
        "characteristic_time": CHARACTERISTIC_TIME,
        "steps_per_characteristic_time": steps,
        "bubble_count": count,
        "drift_velocity": drift,
        "diffusivity": 0.05,
        "chunks": chunks,
        "lagrangian_volume_drift": drift_rel,
        "escape_closure_error": closure_error,
        "escape_fraction": escaped_fraction,
        "sph_prescribed_void_fraction": PRESCRIBED_VOID_FRACTION,
        "sph_measured_void_fraction": float(alpha.mean()),
        "sph_rel_error": sph_rel_error,
        "max_volume_drift": max(drift_rel, closure_error, sph_rel_error),
    }


def run_tests() -> dict:
    """C6: run the aeration test suite and record the pass count."""
    command = [sys.executable, "-m", "pytest", "tests/test_aeration.py", "-q"]
    try:
        completed = subprocess.run(
            command, cwd=str(REPO), capture_output=True, text=True, timeout=1800
        )
    except (OSError, subprocess.SubprocessError) as error:  # pragma: no cover
        return {"passed": 0, "failed": 0, "error": str(error), "command": " ".join(command)}
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    summary = tail[-1] if tail else completed.stdout.strip()
    passed = failed = 0
    for token in summary.replace(",", " ").split():
        if token.isdigit():
            value = int(token)
            if "passed" in summary and "failed" not in summary:
                passed = value
                break
            if failed == 0 and passed == 0:
                passed = value
    if "failed" in summary:
        for token in summary.split():
            if token.rstrip(",").isdigit():
                passed = int(token.rstrip(","))
                break
    return {
        "command": " ".join(command),
        "summary": summary,
        "passed": passed,
        "returncode": completed.returncode,
    }


def figures(out_dir: Path, jump_profiles, rise_curves, drag_result) -> list:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []

    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(jump_profiles["anchor"], jump_profiles["y"], "k--", lw=2.0,
                 label=r"erf anchor $1-\mathrm{erf}(y/(Y_{90}/\sqrt{2}))$")
    axes[0].plot(jump_profiles["with_drift"], jump_profiles["y"], "-", lw=1.8,
                 label="FV, with buoyant drift")
    axes[0].plot(jump_profiles["drift_free"], jump_profiles["y"], ":", lw=2.0,
                 label="FV, drift free")
    axes[0].axvline(0.0, color="0.7", lw=0.5)
    axes[0].set_xlabel("air concentration C [-]")
    axes[0].set_ylabel("depth below free surface y [m]")
    axes[0].set_ylim(2.0 * jump_profiles["y90"], 0.0)
    axes[0].set_title(r"aeration layer, $Fr_1$ = %.1f" % jump_profiles["froude"])
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].loglog(rise_curves["radius"], rise_curves["rigid_sphere"], "-",
                   label="rigid-sphere drag balance")
    axes[1].loglog(rise_curves["radius"], rise_curves["mendelson"], "--",
                   label="Mendelson wave theory")
    axes[1].loglog(rise_curves["radius"], rise_curves["model"], "k.", ms=3,
                   label="rise_velocity (regime switch)")
    axes[1].axvspan(5.0e-4, 5.0e-3, color="0.85", alpha=0.6, label="campaign band")
    axes[1].set_xlabel("bubble radius R [m]")
    axes[1].set_ylabel("terminal rise velocity [m/s]")
    axes[1].set_title("rise velocity: three closures")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "aeration_profile_and_rise_velocity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path.name)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    alphas = [row["alpha"] for row in drag_result["rows"]]
    ax.plot(alphas, [row["reduction_chisholm"] for row in drag_result["rows"]], "o-",
            label="Chisholm two-phase multiplier")
    ax.plot(alphas, [row["reduction_loglaw"] for row in drag_result["rows"]], "s--",
            label="mixture log-law (Wood form)")
    ax.axvspan(*drag_result["closure"]["band"], color="0.85", alpha=0.6,
               label="predeclared band")
    ax.set_xlabel(r"void fraction $\alpha$ [-]")
    ax.set_ylabel(r"friction reduction rate $R$ [-]")
    ax.set_title("aerated friction reduction at fixed mixture flux")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "drag_reduction.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path.name)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    slopes = [row["slope_deg"] for row in INCEPTION["slope_table"]]
    froude = [row["inception_froude"] for row in INCEPTION["slope_table"]]
    ax.semilogy(slopes, froude, "o-")
    ax.set_xlabel("chute slope [deg]")
    ax.set_ylabel(r"inception Froude number $Fr_i = V_i/\sqrt{gd}$ [-]")
    ax.set_title("self-aeration inception vs chute slope (Chanson criterion)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "inception_froude.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path.name)
    return paths


INCEPTION: dict = {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figs", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    global INCEPTION
    INCEPTION = inception_case()
    jump = hydraulic_jump_aeration.evaluate()
    rise = bubble_rise_velocity.evaluate()
    drag = drag_reduction_aerated.evaluate()
    conservation = conservation_case()
    tests = {"passed": 0, "summary": "skipped"} if args.skip_tests else run_tests()

    measured = {
        "C1": INCEPTION["constant_rel_error"],
        "C2": jump["worst_l2_vs_erf_anchor"],
        "C3": drag["worst_rel_deviation_in_band"],
        "C4": rise["worst_rel_error"],
        "C5": conservation["max_volume_drift"],
        "C6": float(tests.get("passed", 0)),
    }
    criteria = {}
    for key, (description, threshold) in CRITERIA.items():
        value = measured[key]
        passed = value <= threshold if key != "C6" else value >= threshold
        criteria[key] = {
            "description": description,
            "measured": value,
            "threshold": threshold,
            "comparison": "<=" if key != "C6" else ">=",
            "passed": bool(passed),
        }

    figure_names = []
    if not args.no_figs:
        figure_names = figures(
            HERE, hydraulic_jump_aeration.profiles(), bubble_rise_velocity.curves(), drag
        )

    report = {
        "campaign_node": "P165W2-N04",
        "repository": "TensorSPH",
        "module": "src/tensorsph/aeration",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "criteria": criteria,
        "all_criteria_passed": all(item["passed"] for item in criteria.values()),
        "inception": INCEPTION,
        "hydraulic_jump": jump,
        "bubble_rise": rise,
        "drag_reduction": drag,
        "conservation": conservation,
        "tests": tests,
        "figures": figure_names,
    }
    out = HERE / "report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{'id':>3} {'measured':>12} {'threshold':>10}  pass  description")
    for key, item in criteria.items():
        print(
            f"{key:>3} {item['measured']:>12.5f} {item['threshold']:>10.5f}"
            f"  {str(item['passed']):>5}  {item['description']}"
        )
    print(f"\nall criteria passed: {report['all_criteria_passed']}")
    print(f"written: {out}")
    return 0 if report["all_criteria_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
