#!/usr/bin/env python3
"""XVAL: our port vs SPHinXsys' ``sycl-execution-policy-not-transferable``.

Anchor ``XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE``.  This harness
measures the two things the OSS practice is responsible for, on a real SPH
geometry (2-D dam-break column with dynamic boundary particles):

* **equivalence** -- the policy-driven, padded-and-masked particle sweep
  reproduces the repository's vectorised summation density (and the Tait
  pressure it implies) for *any* work-group size.  Gate: < 15 % (owner's
  cross-validation standard); the predeclared engineering target is < 1e-12.
* **load-bearing mask** -- the same sweep with the tail mask removed
  (the antipattern the OSS kernel body guards against) fails the same gate, so
  the agreement above is not vacuously true.

Two sensitivity axes, both predeclared before the run:
``local_size`` in {1, 3, 7, 8, 16, 64, 128, 256, 1024} and particle count via
``dx`` in {0.03, 0.02, 0.0125}.  Every number in ``results.json`` and in the
report is written by this script; nothing is typed by hand.

Usage::

    python benchmarks/xval_sycl_execution_policy/run.py
    python benchmarks/xval_sycl_execution_policy/run.py --quick \
        --oss-root /tmp/sxs2/sphinxsim/sphinxsys
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.execution_policy import (  # noqa: E402
    MAIN_EXECUTION_POLICY,
    USE_DEVICE_BACKEND,
    execution_instance,
    work_item_counts,
)
from tensorsph.neighbor_search import PairListCache  # noqa: E402
from tensorsph.wcsph import kernels, swept  # noqa: E402
from tensorsph.wcsph.cases import box_walls, lattice_box  # noqa: E402

OUT = ROOT / "artifacts" / "xval_execution_policy"

# ---- predeclared criteria (fixed before the run, see docs/) --------------- #
GATE_OWNER = 0.15          # owner's cross-validation gate: >15 % unacceptable
GATE_TARGET = 1.0e-12      # engineering target for a numerics-neutral refactor
GATE_CONTROL = 0.15        # the negative control must exceed the owner gate

#: Physical / discretisation parameters of the comparison case (dimensionless
#: group at the bottom of the table).
CASE = {
    "tank": [0.8, 0.4],        # m
    "column": [0.2, 0.4],      # m, water column against the left wall
    "kernel": "quintic",
    "h_over_dx": 1.3,
    "wall_layers": 3,
    "rho0": 1000.0,            # kg/m^3
    "c0": 25.0,                # m/s
    "gamma": 7.0,
}

LOCAL_SIZES = [1, 3, 7, 8, 16, 64, 128, 256, 1024]
DX_AXIS = [0.03, 0.02, 0.0125]

OSS_FILES = [
    "src/shared/particle_dynamics/execution/execution_policy.h",
    "src/src_sycl/shared/particle_dynamics/implementation_sycl.h",
    "src/src_sycl/shared/common/algorithm_primitive_sycl.h",
    "src/src_sycl/shared/common/device_copyable_variable.h",
]

#: Facts the port relies on, each pinned by a regex run over the OSS source at
#: run time.  Only counts / line numbers / sha256 of the matched text are
#: stored: no third-party code is transcribed into this repository.
OSS_FACTS = [
    ("compile_time_policy_switch", r"#if\s+SPHINXSYS_USE_SYCL"),
    ("main_execution_policy_alias", r"using\s+MainExecutionPolicy\s*="),
    ("device_policy_template", r"using\s+ParallelDevicePolicy\s*=\s*DeviceExecution<"),
    ("execution_instance_copy_deleted",
     r"ExecutionInstance\(ExecutionInstance const &\)\s*=\s*delete"),
    ("execution_instance_assignment_deleted",
     r"void\s+operator=\(ExecutionInstance const &\)\s*=\s*delete"),
    ("singleton_accessor", r"static\s+ExecutionInstance\s*&\s*getInstance\(\)"),
    ("default_device_selector", r"sycl::default_selector_v"),
    ("work_group_clamp", r"SMIN\(max_workgroup_size,\s*64UL\)"),
    ("uniform_nd_range_round_up", r"global_size\s*%\s*local_size\s*\?"),
    ("tail_mask_in_kernel", r"index\.get_global_id\(0\)\s*<\s*loop_bound"),
    ("device_memory_bound_to_queue",
     r"sycl::malloc_device<T>\(size,\s*execution::execution_instance\.getQueue\(\)\)"),
]


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oss_commit(oss_root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(oss_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def collect_provenance(oss_root: Path | None) -> dict:
    """Machine-extracted provenance: hashes, line numbers, no transcription."""
    entry = {
        "oss_project": "Xiangyu-Hu/SPHinXsys",
        "oss_url": "https://github.com/Xiangyu-Hu/SPHinXsys",
        "oss_root": str(oss_root) if oss_root else None,
        "checkout_available": bool(oss_root and oss_root.is_dir()),
        "commit": oss_commit(oss_root) if oss_root and oss_root.is_dir() else None,
        "files": [],
        "facts": [],
        "note": (
            "Facts are located by regex over the OSS checkout at run time; only "
            "line numbers, hit counts and sha256 of the matched text are stored. "
            "When checkouts are unavailable the entry degrades to "
            "checkout_available=false instead of falling back to hand notes."
        ),
    }
    if not oss_root or not oss_root.is_dir():
        return entry

    texts: dict[str, str] = {}
    for rel in OSS_FILES:
        path = oss_root / rel
        digest = sha256_file(path)
        lines = None
        if digest is not None:
            texts[rel] = path.read_text(encoding="utf-8", errors="replace")
            lines = len(texts[rel].splitlines())
        entry["files"].append({"path": rel, "sha256": digest, "lines": lines})

    for label, pattern in OSS_FACTS:
        hits = []
        rx = re.compile(pattern)
        for rel, text in texts.items():
            for number, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append({"file": rel, "line": number,
                                 "matched_sha256": sha256_text(line.strip())})
        entry["facts"].append({"fact": label, "pattern": pattern, "hits": hits,
                               "found": bool(hits)})
    return entry


# --------------------------------------------------------------------------- #
# comparison case
# --------------------------------------------------------------------------- #


def build_case(dx: float) -> dict:
    """2-D dam-break column + dynamic boundary particles (static positions)."""
    dim = 2
    h = CASE["h_over_dx"] * dx
    fluid = lattice_box([0.0, 0.0], list(CASE["column"]), dx, dim)
    wall = box_walls([0.0, 0.0], list(CASE["tank"]), dx, dim, layers=CASE["wall_layers"])
    pos = torch.cat((fluid, wall), dim=0)
    n_fluid = int(fluid.shape[0])
    n = int(pos.shape[0])
    mass = torch.full((n,), CASE["rho0"] * dx ** dim, dtype=torch.float64)
    h_t = torch.full((n,), h, dtype=torch.float64)
    cutoff = kernels.SUPPORT_FACTOR[CASE["kernel"]] * h
    cache = PairListCache(cutoff, 0.0, fluid_count=n_fluid)
    pairs, _ = cache.get(pos)
    return {
        "dx": dx, "h": h, "dim": dim, "pos": pos, "mass": mass, "h_t": h_t,
        "pairs": pairs, "n_fluid": n_fluid, "n": n, "cutoff": cutoff,
        "support_over_dx": cutoff / dx,
        "particles_per_support": (cutoff / dx) ** dim,
        "pairs_per_particle": (2.0 * int(pairs.shape[0])) / max(n, 1),
    }


def tait(rho: torch.Tensor) -> torch.Tensor:
    bulk = CASE["rho0"] * CASE["c0"] ** 2 / CASE["gamma"]
    p = bulk * ((rho / CASE["rho0"]).pow(CASE["gamma"]) - 1.0)
    return p.clamp_min(0.0)


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> dict:
    denom = ref.abs().clamp_min(1.0e-30)
    rel = (got - ref).abs() / denom
    return {
        "max": float(rel.max()),
        "median": float(rel.median()),
        "mean": float(rel.mean()),
        "argmax": int(torch.argmax(rel)),
        "above_owner_gate": int((rel > GATE_OWNER).sum()),
        "count": int(rel.numel()),
    }


def timed(fn, repeats: int = 3) -> tuple[float, object]:
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1.0e3, result   # milliseconds


def run_arm(case: dict, local_size: int) -> dict:
    """Compare both arms on the fluid rows.

    The launch bound is ``n_fluid``: that is the OSS main loop (the boundary
    particles are advanced by their own dynamics, so the sweep is not launched
    over them), and the pair list still contains the fluid/wall pairs.
    """
    pos, mass, h_t = case["pos"], case["mass"], case["h_t"]
    dim, pairs = case["dim"], case["pairs"]
    nf = case["n_fluid"]

    t_ref, rho_ref = timed(lambda: swept.summation_density_vectorised(
        pos, mass, h_t, dim, pairs, kernel=CASE["kernel"]))
    t_port, rho_port = timed(lambda: swept.summation_density_swept(
        pos, mass, h_t, dim, pairs, kernel=CASE["kernel"], local_size=local_size,
        bound=nf))
    t_bad, rho_bad = timed(lambda: swept.summation_density_swept(
        pos, mass, h_t, dim, pairs, kernel=CASE["kernel"], local_size=local_size,
        mask_tail=False, bound=nf))

    rho_ref_f, rho_port_f, rho_bad_f = rho_ref[:nf], rho_port[:nf], rho_bad[:nf]
    density = rel_err(rho_port_f, rho_ref_f)
    p_ref, p_port = tait(rho_ref_f), tait(rho_port_f)
    pressure = rel_err(p_port, p_ref)
    control = rel_err(rho_bad_f, rho_ref_f)
    launch = work_item_counts(nf, local_size)

    return {
        "local_size": local_size,
        "launch": launch,
        "density": density,
        "pressure": pressure,
        "control_unmasked_density": control,
        "gates": {
            "owner_lt_15pct": bool(density["max"] < GATE_OWNER),
            "target_lt_1e_12": bool(density["max"] < GATE_TARGET),
            "control_fails_owner_gate": bool(control["max"] > GATE_CONTROL),
        },
        "wall_clock_ms": {"vectorised": t_ref, "swept": t_port, "unmasked": t_bad},
        "port_overhead": (t_port / t_ref) if t_ref > 0 else None,
        "rho_over_rho0_median": float((rho_ref_f / CASE["rho0"]).median()),
        "rho_over_rho0_min": float((rho_ref_f / CASE["rho0"]).min()),
        "rho_over_rho0_max": float((rho_ref_f / CASE["rho0"]).max()),
    }


def naive_crosscheck(case: dict, local_size: int) -> dict:
    """Third opinion: scalar per-particle loop (small cases only)."""
    rho_naive = swept.summation_density_naive(
        case["pos"], case["mass"], case["h_t"], case["dim"], case["pairs"],
        kernel=CASE["kernel"])
    rho_ref = swept.summation_density_vectorised(
        case["pos"], case["mass"], case["h_t"], case["dim"], case["pairs"],
        kernel=CASE["kernel"])
    rho_port = swept.summation_density_swept(
        case["pos"], case["mass"], case["h_t"], case["dim"], case["pairs"],
        kernel=CASE["kernel"], local_size=local_size)
    return {
        "particles": case["n"],
        "vectorised_vs_naive": rel_err(rho_ref, rho_naive),
        "swept_vs_naive": rel_err(rho_port, rho_naive),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def make_figures(results: dict, case: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    figures = []

    # 1. error vs work-group size, masked vs unmasked, against the 15 % gate
    rows = results["local_size_sweep"]
    local = [r["local_size"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(local, [max(r["density"]["max"], 1e-18) for r in rows], "o-",
            label="ported sweep (tail masked)")
    ax.plot(local, [max(r["control_unmasked_density"]["max"], 1e-18) for r in rows],
            "s--", label="negative control (no tail mask)")
    ax.axhline(GATE_OWNER, color="tab:red", ls=":", label="owner gate 15 %")
    ax.axhline(GATE_TARGET, color="tab:green", ls=":", label="target 1e-12")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("work-group size (particles)")
    ax.set_ylabel("max relative density error vs vectorised reference")
    ax.set_title(f"Launch-geometry invariance (n={case['n']}, dx={case['dx']} m)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    path = OUT / "fig_error_vs_workgroup_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Max relative density error against the vectorised "
                               "reference as a function of the padded launch "
                               "geometry; the ported sweep is flat at round-off, "
                               "the unmasked control sits above the 15 % gate "
                               "whenever the launch has idle work items."})

    # 2. padding cost: idle work-item fraction vs particle count
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for wg in (8, 64, 256):
        xs = [row["particle_count"] for row in results["particle_sweep"]]
        ys = [100.0 * work_item_counts(row["particle_count"], wg)["idle_fraction"]
              for row in results["particle_sweep"]]
        ax.plot(xs, ys, "o-", label=f"work-group {wg}")
    ax.set_xscale("log")
    ax.set_xlabel("particle count n")
    ax.set_ylabel("idle (padded) work items / n  [%]")
    ax.set_title("Padding cost of the OSS uniform nd-range")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    path = OUT / "fig_padding_cost.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Idle fraction of the padded launch "
                               "(ceil(n/l)*l - n)/n for three work-group sizes: "
                               "the cost the OSS design accepts in exchange for a "
                               "uniform nd-range, bounded by (l-1)/n."})

    # 3. density field + error field
    pos = case["pos"][: case["n_fluid"]]
    rho_ref = swept.summation_density_vectorised(
        case["pos"], case["mass"], case["h_t"], case["dim"], case["pairs"],
        kernel=CASE["kernel"])[: case["n_fluid"]]
    worst_local = max(r["local_size"] for r in rows)
    rho_port = swept.summation_density_swept(
        case["pos"], case["mass"], case["h_t"], case["dim"], case["pairs"],
        kernel=CASE["kernel"], local_size=worst_local)[: case["n_fluid"]]
    err = (rho_port - rho_ref).abs() / rho_ref.abs()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    sc = axes[0].scatter(pos[:, 0].numpy(), pos[:, 1].numpy(),
                         c=(rho_ref / CASE["rho0"]).numpy(), s=12, cmap="viridis")
    fig.colorbar(sc, ax=axes[0], label=r"$\rho/\rho_0$")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("summation density (vectorised arm)")
    axes[0].set_aspect("equal")
    sc2 = axes[1].scatter(pos[:, 0].numpy(), pos[:, 1].numpy(),
                          c=torch.log10(err.clamp_min(1e-18)).numpy(), s=12,
                          cmap="magma")
    fig.colorbar(sc2, ax=axes[1], label=r"$\log_{10}$ relative error")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_title(f"ported sweep, worst-padded work-group ({worst_local})")
    axes[1].set_aspect("equal")
    path = OUT / "fig_density_field.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Left: summation density field of the dam-break "
                               "column with dynamic boundary particles. Right: "
                               "relative deviation of the padded masked sweep, on "
                               "a log scale, for the worst-padded work-group; the "
                               "deviation stays at round-off everywhere including "
                               "the clamped last particle."})

    # 4. wall-clock of the two arms vs work-group size
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(local, [r["wall_clock_ms"]["vectorised"] for r in rows], "k--",
            label="vectorised reference")
    ax.plot(local, [r["wall_clock_ms"]["swept"] for r in rows], "o-",
            label="ported sweep")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("work-group size (particles)")
    ax.set_ylabel("wall clock [ms] (best of 3)")
    ax.set_title("Cost of the ported launch (torch CPU)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    path = OUT / "fig_wallclock.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Wall clock of the two arms as a function of "
                               "work-group size (CPU backend): the padded sweep pays "
                               "per-work-group overhead, which is the honest price of "
                               "the OSS launch discipline on a host backend."})
    return figures


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "USE_DEVICE_BACKEND": USE_DEVICE_BACKEND,
        "MAIN_EXECUTION_POLICY": type(MAIN_EXECUTION_POLICY).__name__,
        "execution_instance_device": str(execution_instance.get_device()),
        "execution_instance_work_group_size": execution_instance.work_group_size,
        "torch_dtype": "float64",
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip() or None,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oss-root", type=Path,
                        default=Path(os.environ.get("SPHINXSYS_ROOT",
                                                    "/tmp/sxs2/sphinxsim/sphinxsys")))
    parser.add_argument("--quick", action="store_true",
                        help="skip the dx axis apart from the base resolution")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    base = build_case(0.02)                     # base resolution
    dx_axis = [0.02] if args.quick else DX_AXIS
    local_axis = [1, 64, 256] if args.quick else LOCAL_SIZES

    results = {
        "anchor": "XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "criteria": {
            "owner_gate": GATE_OWNER,
            "engineering_target": GATE_TARGET,
            "negative_control_must_exceed": GATE_CONTROL,
            "unit_of_analysis": "single density evaluation on a static particle "
                                "snapshot (no time integration)",
            "note": "thresholds fixed before the run; no tuning afterwards",
        },
        "case": {**{k: v for k, v in CASE.items()},
                 "n_fluid": base["n_fluid"], "n_total": base["n"],
                 "n_pairs": int(base["pairs"].shape[0]),
                 "pairs_per_particle": base["pairs_per_particle"],
                 "support_over_dx": base["support_over_dx"],
                 "particles_per_support": base["particles_per_support"],
                 "dimensionless": {
                     "h_over_dx": CASE["h_over_dx"],
                     "support_over_dx": base["support_over_dx"],
                     "particles_per_support": base["particles_per_support"],
                     "Mach_ref": None,
                     "note": "static snapshot: no velocity scale enters, so Mach "
                             "and CFL are not defined for this measurement",
                 }},
        "environment": environment(),
        "provenance": collect_provenance(args.oss_root),
    }

    # (1) work-group-size sweep at the base resolution
    results["local_size_sweep"] = [run_arm(base, local) for local in local_axis]

    # (2) particle-count sweep at a fixed work-group size
    particle_rows = []
    for dx in dx_axis:
        case = base if dx == 0.02 else build_case(dx)
        row = {
            "dx": dx,
            "particle_count": case["n"],
            "n_fluid": case["n_fluid"],
            "pairs": int(case["pairs"].shape[0]),
            "pairs_per_particle": case["pairs_per_particle"],
            "rho_over_rho0_median": float(
                (swept.summation_density_vectorised(
                    case["pos"], case["mass"], case["h_t"], case["dim"],
                    case["pairs"], kernel=CASE["kernel"])[: case["n_fluid"]]
                 / CASE["rho0"]).median()),
        }
        for local in (8, 64, 256):
            arm = run_arm(case, local)
            row[f"local_{local}"] = {
                "density_max_rel_err": arm["density"]["max"],
                "launch": arm["launch"],
                "wall_clock_ms": arm["wall_clock_ms"],
                "gates": arm["gates"],
            }
        particle_rows.append(row)
    results["particle_sweep"] = particle_rows

    # (3) third opinion on the two smallest cases
    results["naive_crosscheck"] = [
        naive_crosscheck(build_case(dx), 64) for dx in sorted(dx_axis)[-2:]
    ] if len(dx_axis) > 1 else [naive_crosscheck(base, 64)]

    # (4) the gate table, one row per measured quantity
    def judge(err: float, baseline: str) -> dict:
        return {"ref": baseline, "got": "ported sweep", "max_rel_err": err,
                "owner_15pct": "PASS" if err < GATE_OWNER else "FAIL",
                "target_1e_12": "PASS" if err < GATE_TARGET else "FAIL"}

    worst = max(results["local_size_sweep"], key=lambda r: r["density"]["max"])
    worst_control = max(results["local_size_sweep"],
                        key=lambda r: r["control_unmasked_density"]["max"])
    results["xval_table"] = [
        {"quantity": "summation density, worst work-group size",
         **judge(worst["density"]["max"],
                 "vectorised index_add (TensorSPH shipped path)"),
         "detail": f"local_size={worst['local_size']}"},
        {"quantity": "Tait pressure, worst work-group size",
         **judge(worst["pressure"]["max"],
                 "Tait law applied to the vectorised density"),
         "detail": f"local_size={worst['local_size']}"},
        {"quantity": "summation density vs scalar per-particle loop",
         **judge(max(r["swept_vs_naive"]["max"] for r in results["naive_crosscheck"]),
                 "independent scalar loop (no tensor reduction)"),
         "detail": "small cases only"},
        {"quantity": "negative control (tail mask removed)",
         **judge(worst_control["control_unmasked_density"]["max"],
                 "vectorised index_add"),
         "detail": f"local_size={worst_control['local_size']}, "
                   f"{worst_control['launch']['idle_work_items']} idle work items"},
    ]

    results["figures"] = make_figures(results, base)

    results_path = out / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    provenance_path = out / "provenance.json"
    provenance_path.write_text(json.dumps(results["provenance"], indent=2) + "\n")

    summary = {
        "results": str(results_path.relative_to(ROOT)),
        "provenance": str(provenance_path.relative_to(ROOT)),
        "worst_density_rel_err": worst["density"]["max"],
        "worst_control_rel_err": worst_control["control_unmasked_density"]["max"],
        "all_gates_pass": all(row["gates"]["owner_lt_15pct"] and row["gates"]["target_lt_1e_12"]
                              for row in results["local_size_sweep"]),
        "checkout_available": results["provenance"]["checkout_available"],
        "facts_found": sum(1 for f in results["provenance"]["facts"] if f["found"]),
        "facts_total": len(OSS_FACTS),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return results


if __name__ == "__main__":
    main()
