"""M4 benchmark: particle-spacing convergence and the kernel-support control.

    python benchmarks/verified/sph_convergence/run.py
    python benchmarks/verified/sph_convergence/run.py --figure

Three things are measured and written to the evidence store:

``ladder``
    relative ``L2`` error of the SPH interpolation of an analytic field on
    lattices of spacing ``0.02 ... 0.0025 m`` (interior particles only), for the
    Shepard-normalised interpolant (gated), the raw summation (reported) and a
    kernel truncated at ``R = 1.5 h`` (reported -- with normalisation it
    re-converges, which is stated rather than hidden).
``support_defect``
    the partition-of-unity defect ``|sum_j W dx^2 - 1|`` for ``R = 2h`` and
    ``R = 1.5h``; the truncated kernel breaks it by an ``O(1)`` amount.
``control``
    the hydrostatic case integrated with the truncated kernel: the balance must
    break (measured ``max band error`` vs the ``2 %`` profile gate).

Exit status is non-zero if the convergence gate or the control fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorsph.verification import convergence as m4, hydrostatic as m2, report
from tensorsph.verification.evidence import EvidenceStore

HERE = Path(__file__).resolve().parent
STORE = HERE / "evidence.sqlite"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def regate(store: EvidenceStore) -> int:
    """Recompute the J4 verdict from stored measurements (write-once, additive).

    The gate *rule* is code and can be reviewed; the measurements are data and
    must not be recomputed by hand.  This mode therefore reads the raw metrics of
    the latest ladder run and the latest truncated-support run, applies the
    current :func:`tensonesph_gate` rule and writes one new run of each case with
    a note pointing at the run whose measurements it re-uses.
    """
    ladder_rows = store.query(
        "SELECT run_id FROM runs WHERE case_name='kernel_interpolation_convergence'"
        " AND notes NOT LIKE '%regate%' ORDER BY run_id DESC LIMIT 1")
    ctrl_rows = store.query(
        "SELECT run_id FROM runs WHERE case_name='hydrostatic_truncated_support'"
        " AND notes NOT LIKE '%regate%' ORDER BY run_id DESC LIMIT 1")
    if not ladder_rows or not ctrl_rows:
        raise SystemExit("no original runs to re-gate")
    ladder_id = int(ladder_rows[0]["run_id"])
    ctrl_id = int(ctrl_rows[0]["run_id"])

    def value(run_id, name):
        row = store.query("SELECT value FROM metrics WHERE run_id=? AND name=?",
                          (run_id, name))
        if not row:
            raise SystemExit(f"metric {name!r} missing from run {run_id}")
        return row[0]["value"]

    order = value(ladder_id, "min_order_normalized")
    mean_order = value(ladder_id, "mean_order_normalized")
    raw_order = value(ladder_id, "order_raw_summation")
    ratio = value(ladder_id, "defect_ratio")
    baseline = value(ladder_id, "hydrostatic_baseline_pct")
    control_pct = value(ctrl_id, "max_band_error_pct")

    result = {"orders": {"normalized": {"min_order": order}},
              "support_defect": {"defect_ratio": ratio},
              "hydrostatic_baseline_pct": baseline}
    control = {"max_band_error_pct": control_pct, "diverged": False}
    gate = m4.gate(result, min_order=args_min_order(), control=control)

    new_id = store.add_run(
        "kernel_interpolation_convergence",
        {"regate_of": ladder_id, "control_of": ctrl_id},
        notes=f"regate: verdicts recomputed from the measurements of run "
              f"{ladder_id} with the current gate rule; no new integration")
    store.add_metrics(new_id, {
        "min_order_normalized": dict(value=order, unit="1", gate_limit=1.0,
                                     verdict=_verdict(order >= 1.0),
                                     provenance=f"copied from run {ladder_id}"),
        "mean_order_normalized": dict(value=mean_order, unit="1"),
        "order_raw_summation": dict(value=raw_order, unit="1"),
        "defect_ratio": dict(value=ratio, unit="1", gate_limit=10.0,
                             verdict=_verdict(ratio >= 10.0)),
        "defect_2h": dict(value=value(ladder_id, "defect_2h"), unit="1"),
        "defect_1_5h": dict(value=value(ladder_id, "defect_1_5h"), unit="1"),
        "hydrostatic_baseline_pct": dict(value=baseline, unit="%"),
        "gate_pass": dict(value=float(gate["pass"]), unit="bool", gate_limit=1.0,
                          verdict=_verdict(gate["pass"])),
    })
    new_ctrl = store.add_run(
        "hydrostatic_truncated_support",
        {"regate_of": ctrl_id},
        notes=f"regate: verdict recomputed from the measurement of run {ctrl_id}")
    ctrl_pass = bool(gate["control_pass"])
    store.add_metrics(new_ctrl, {
        "max_band_error_pct": dict(value=control_pct, unit="%",
                                   gate_limit=5.0 * baseline,
                                   verdict=_verdict(ctrl_pass)),
        "diverged": dict(value=0.0, unit="bool"),
        "gate_pass": dict(value=float(ctrl_pass), unit="bool", gate_limit=1.0,
                          verdict=_verdict(ctrl_pass)),
    })
    print(f"regate: order {order:.3f}, defect ratio {ratio:.1f}x, baseline "
          f"{baseline:.3f} %, control {control_pct:.2f} % "
          f"({control_pct / baseline:.1f}x) -> gate "
          f"{'PASS' if gate['pass'] else 'FAIL'}", flush=True)
    return 0 if gate["pass"] else 1


def args_min_order() -> float:
    return 1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--min-order", type=float, default=1.0)
    ap.add_argument("--control-dx", type=float, default=0.02)
    ap.add_argument("--control-seconds", type=float, default=None)
    ap.add_argument("--baseline-seconds", type=float, default=4.0)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--regate", action="store_true",
                    help="recompute the J4 verdicts from the stored raw "
                         "measurements with the current gate rule (the raw runs "
                         "are kept; this writes one new run)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    if args.control_seconds is None:
        # The control and its baseline must be integrated for the same time,
        # otherwise the comparison measures the averaging window rather than the
        # kernel.
        args.control_seconds = args.baseline_seconds
    torch.set_num_threads(args.threads)

    store = EvidenceStore(STORE)
    if args.figure:
        rows = store.query("SELECT run_id FROM runs WHERE case_name="
                           "'kernel_interpolation_convergence' ORDER BY run_id DESC LIMIT 1")
        if not rows:
            raise SystemExit("no convergence run recorded")
        png = report.figure_convergence(store, int(rows[0]["run_id"]),
                                        HERE / "figures" / "convergence.png")
        print("figure:", png.relative_to(HERE))
        store.close()
        return 0

    if args.regate:
        return regate(store)

    result = m4.run()
    defect = m4.kernel_sum_defect(dx=args.dx)
    result["support_defect"] = defect

    # baseline hydrostatic error of the same case (the control's reference)
    # The baseline must be a *settled* hydrostatic state.  The verification
    # ladder needs 4 s and a time mean of the last half to reach sub-1 %; a 0.5 s
    # run is still sloshing and reads several per cent, which would make the
    # control look successful for the wrong reason (a big ratio against a big
    # baseline).  The control is therefore always compared against the settled
    # value, and the same 4 s is used for the truncated-kernel run.
    baseline = m2.run_hydrostatic_case(m2.HydrostaticCase(
        dx=args.control_dx, tank_width=0.4, depth=0.4,
        seconds=args.baseline_seconds, min_particles=1))
    result["hydrostatic_baseline_pct"] = baseline["measured"]["max_band_error_pct"]

    control = m4.truncated_support_hydrostatic(support_factor=1.5,
                                               dx=args.control_dx,
                                               seconds=args.control_seconds)
    gate = m4.gate(result, min_order=args.min_order, control=control)

    run_id = store.add_run("kernel_interpolation_convergence",
                           {"domain": result["config"]["domain"],
                            "h_factor": result["config"]["h_factor"],
                            "spacings": result["config"]["spacings"],
                            "control_support": 1.5},
                           notes="static interpolation ladder + truncated-support control")
    orders = result["orders"]
    store.add_metrics(run_id, {
        "min_order_normalized": dict(value=orders["normalized"]["min_order"],
                                     unit="1", gate_limit=args.min_order,
                                     verdict=_verdict(orders["normalized"]["min_order"]
                                                      >= args.min_order),
                                     provenance="min pairwise order of the Shepard-"
                                                "normalised L2 error over the ladder"),
        "mean_order_normalized": dict(value=orders["normalized"]["mean_order"], unit="1"),
        "order_raw_summation": dict(value=orders["raw"]["mean_order"], unit="1"),
        "defect_ratio": dict(value=defect["defect_ratio"], unit="1",
                             gate_limit=10.0,
                             verdict=_verdict(defect["defect_ratio"] >= 10.0),
                             provenance="partition-of-unity defect R=1.5h over R=2h"),
        "defect_2h": dict(value=defect["support_2h"]["mean_defect"], unit="1"),
        "defect_1_5h": dict(value=defect["support_1.5h"]["mean_defect"], unit="1"),
        "hydrostatic_baseline_pct": dict(value=result["hydrostatic_baseline_pct"],
                                         unit="%"),
        "gate_pass": dict(value=float(gate["pass"]), unit="bool", gate_limit=1.0,
                          verdict=_verdict(gate["pass"])),
    })
    store.add_series(run_id, "ladder", result["ladder"])

    ctrl_id = store.add_run("hydrostatic_truncated_support",
                            {"support_factor": 1.5, "dx": args.control_dx,
                             "seconds": args.control_seconds},
                            notes="negative control: kernel support R < 2h")
    ctrl_pass = bool(control["diverged"]
                     or control["max_band_error_pct"] >= 10.0
                     * result["hydrostatic_baseline_pct"])
    store.add_metrics(ctrl_id, {
        "max_band_error_pct": dict(value=control["max_band_error_pct"], unit="%",
                                   gate_limit=10.0 * result["hydrostatic_baseline_pct"],
                                   verdict=_verdict(ctrl_pass)),
        "diverged": dict(value=float(control["diverged"]), unit="bool"),
        "gate_pass": dict(value=float(ctrl_pass), unit="bool", gate_limit=1.0,
                          verdict=_verdict(ctrl_pass)),
    })

    out = {"ladder": result["ladder"], "orders": orders,
           "support_defect": defect, "baseline_pct": result["hydrostatic_baseline_pct"],
           "control": control, "gate": gate, "run_id": run_id,
           "control_run_id": ctrl_id}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print("orders normalized", [round(o, 2) for o in orders["normalized"]["pair_orders"]],
          "| raw", [round(o, 2) for o in orders["raw"]["pair_orders"]],
          "| truncated", [round(o, 2) for o in orders["truncated_support"]["pair_orders"]],
          flush=True)
    print(f"defect ratio {defect['defect_ratio']:.1f}x, control band error "
          f"{control['max_band_error_pct']:.2f} % vs baseline "
          f"{result['hydrostatic_baseline_pct']:.3f} %, gate "
          f"{'PASS' if gate['pass'] else 'FAIL'}", flush=True)
    store.close()
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
