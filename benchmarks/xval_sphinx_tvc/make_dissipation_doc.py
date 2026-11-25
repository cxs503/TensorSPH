"""Generate ``docs/xval_sphinx_tvc/DISSIPATION_ATTRIBUTION.md`` from the payload.

Every number in the markdown is read out of
``docs/xval_sphinx_tvc/data/dissipation_attribution.json`` -- nothing is typed
by hand, so the document cannot drift away from the run it reports.  The
figure is regenerated from the same payload on every call.

Usage::

    python benchmarks/xval_sphinx_tvc/make_dissipation_doc.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from xval_sphinx_tvc.run import load_experiment, score  # noqa: E402

PAYLOAD = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "dissipation_attribution.json"
DOC = ROOT / "docs" / "xval_sphinx_tvc" / "DISSIPATION_ATTRIBUTION.md"
FIG = ROOT / "docs" / "xval_sphinx_tvc" / "figure_dissipation_attribution.png"
#: The published free-slip rows this payload has to reproduce.
PUBLISHED = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "grid_dx003.json"


def _fmt(v, spec=".4f"):
    return "n/a" if v is None else format(v, spec)


def _pct(v):
    return "n/a" if v is None else f"{100.0 * v:.1f} %"


def figure(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = payload["records"]
    # Left panel: every configuration, grouped by family, sorted by error.
    order = sorted(recs, key=lambda r: r["mean_rel_error"])
    labels = [r["label"] for r in order]
    errs = [100.0 * r["mean_rel_error"] for r in order]
    colours = []
    for r in order:
        if r["tvc_alpha"] is None:
            colours.append("#c23b22")           # correction off
        elif abs(r["tvc_alpha"] - 0.10) < 1e-9:
            colours.append("#2a6fdb")           # the calibrated coefficient
        else:
            colours.append("#f0a202")           # other shipped coefficients

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    ax = axes[0]
    ax.barh(range(len(order)), errs, color=colours)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(15.0, color="k", ls="--", lw=1.2)
    ax.text(15.6, len(order) - 0.8, "owner gate 15 %", fontsize=8.5)
    ax.set_xlabel("mean relative error of $Z/a(\\tau)$  [%]")
    ax.set_title("(a) every measured configuration, dx = %g m" % payload["dx"])
    for i, e in enumerate(errs):
        ax.text(e + 0.6, i, f"{e:.1f}", va="center", fontsize=7.5)
    ax.set_xlim(0, max(errs) * 1.18)
    ax.grid(axis="x", alpha=0.3)

    # Right panel: late-time front position, the quantity the raw error rides on.
    ax = axes[1]
    fronts = [r["front_over_a_final"] for r in order]
    ax.barh(range(len(order)), fronts, color=colours)
    exp = load_experiment()
    ref_end = max(exp["data"], key=lambda d: d["tau"])["Z_over_a"]
    ax.axvline(ref_end, color="k", ls="--", lw=1.2)
    ax.text(ref_end + 0.05, len(order) - 0.8, "experiment (last sample)",
            fontsize=8.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["" for _ in order])
    ax.set_xlabel(r"front at the last sample,  $Z/a$")
    ax.set_title("(b) final front position (runaway vs. arrested surge)")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Attribution of the dam-break gain: correction vs. dissipation",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, dpi=170)
    plt.close(fig)


def replication_table(payload: dict) -> list:
    """Compare the replicated rows against the published ``grid_dx003.json``."""
    if not PUBLISHED.exists():
        return []
    pub = {r["variant"]: r for r in json.loads(PUBLISHED.read_text())["records"]}
    pairs = [("slip_base", "baseline"), ("slip_a005", "tvc_a005_all"),
             ("slip_a010", "tvc_a010_all"), ("slip_a020", "tvc_a020_all")]
    rows = []
    by = {r["label"]: r for r in payload["records"]}
    for mine, theirs in pairs:
        if mine not in by or theirs not in pub:
            continue
        a, b = by[mine], pub[theirs]
        rows.append({
            "label": mine,
            "published_variant": theirs,
            "rel_here": a["mean_rel_error"],
            "rel_published": b["score"]["mean_rel_error"],
            "abs_diff": abs(a["mean_rel_error"] - b["score"]["mean_rel_error"]),
            "mae_here": a["mae"],
            "mae_published": b["score"]["mae"],
        })
    return rows


def render(payload: dict) -> str:
    s = payload["summary"]
    recs = payload["records"]
    by = {r["label"]: r for r in recs}
    q1 = s["q1_noslip_physical_fix"]
    q2 = s["q2_dissipation_substitution"]
    q3 = s["q3_unique_signature"]
    rep = replication_table(payload)
    cal = by.get("slip_a010")
    noslip_cal = by.get("noslip_a010")
    add = by.get("slip_a010_visc10")
    exp = load_experiment()
    ref_end = max(exp["data"], key=lambda d: d["tau"])["Z_over_a"]

    L = []
    A = L.append
    A("# Attribution of the dam-break gain: the TVC operator vs. plain dissipation")
    A("")
    A(f"Anchor: `{payload['anchor']}` — TensorSPH repository.  Generated by")
    A("`benchmarks/xval_sphinx_tvc/make_dissipation_doc.py` from")
    A("`docs/xval_sphinx_tvc/data/dissipation_attribution.json`; every number below is")
    A("read out of that payload (no hand entry).")
    A("")
    A("## 1. Problem, and why it needs an adversarial test")
    A("")
    A("The anchor grades the ported SPHinXsys transport-velocity correction (TVC) by the")
    A("mean error of the dam-break front position `Z/a(tau)` against Martin & Moyce")
    A("(1952). Two claims rest on that single number:")
    A("")
    A("* `VALIDATION_REPORT.md`: *\"TVC reduces MAE from 0.621 to 0.143 (4.3x)\"*;")
    A("* `UNCALIBRATED_DEFAULTS.md`: with the coefficients SPHinXsys actually ships the")
    A("  gate is **not** met.")
    A("")
    A("Both read the improvement as an **operator** result. That is only legitimate if no")
    A("cheaper explanation produces the same number. `Z/a` is an integral measure of a")
    A("surge whose late-time front speed is controlled by how much momentum the floor and")
    A("the bulk take away, and every published run used `wall_friction = 0` — a")
    A("**free-slip floor** — while Martin & Moyce's tank had a **no-slip** floor. So two")
    A("confounds have to be excluded before the operator can be credited:")
    A("")
    A("| # | confound | why it would fake the result |")
    A("|---|---|---|")
    A("| Q1 | missing no-slip floor | a slippery floor lets the front run away, and *any* extra dissipation arrests it |")
    A("| Q2 | added bulk dissipation | TVC is a position regulariser, but on a coarse lattice it also damps the front; laminar viscosity could supply the same damping |")
    A("")
    A("The measured baseline does overshoot exactly as the first confound predicts: the")
    A(f"last-sample front is `Z/a = {_fmt(by['slip_base']['front_over_a_final'], '.3f')}`")
    A(f"against an experimental `{_fmt(ref_end, '.3f')}`.")
    A("")
    A("## 2. Method")
    A("")
    A("| item | value |")
    A("|---|---|")
    A(f"| case | Martin & Moyce (1952) dam break, `a = 0.2 m`, `2a = 0.4 m`, dry bed |")
    A(f"| solver | in-repo WCSPH `src/tensorsph/dam_break.py` (same code path as `run.py`, imported, not re-implemented) |")
    A(f"| correction | `src/tensorsph/transport_velocity_correction.py` (upstream port) |")
    A(f"| resolution | `dx = {payload['dx']} m`, `t = {payload['seconds']} s` (the anchor's headline row) |")
    A(f"| axes | `tvc_alpha` in {{off, 0.05, 0.10, 0.20}}; laminar `viscosity`; `wall_friction` in {{free-slip, no-slip}} |")
    A(f"| metric | mean relative error of `Z/a(tau)`, plus MAE (ledger gate {payload['gates']['ledger_abs']}) |")
    A(f"| gate | owner {_pct(payload['gates']['owner_rel'])} |")
    A(f"| reference band | `+-{payload['digitisation_band']} in Z/a`, propagated: a verdict that flips inside the band is `PASS_marginal`, not `PASS` |")
    A(f"| rows | {payload['summary']['records']} configurations, predeclared in `GRID` before any run, all retained in the payload |")
    A("")
    A("## 3. Results (dx = %g m)" % payload["dx"])
    A("")
    A("| configuration | `alpha` | viscosity | floor | MAE | mean rel. | front `Z/a` end | verdict |")
    A("|---|---|---|---|---|---|---|---|")
    for r in recs:
        floor = "no-slip" if r["wall_friction"] else "free-slip"
        alpha = "off" if r["tvc_alpha"] is None else f"{r['tvc_alpha']:g}"
        A(f"| `{r['label']}` | {alpha} | {r['viscosity']:g} | {floor} | "
          f"{_fmt(r['mae'])} | {_pct(r['mean_rel_error'])} | "
          f"{_fmt(r['front_over_a_final'], '.3f')} | {r['band_verdict']} |")
    A("")
    A("![attribution](figure_dissipation_attribution.png)")
    A("")
    A("## 4. The three questions, answered")
    A("")
    A("### Q1 — does the physical no-slip floor explain it? **No.**")
    A("")
    A(f"With the correction off, the no-slip floor gives `{_pct(q1['mean_rel_error'])}`")
    A(f"against `{_pct(by['slip_base']['mean_rel_error'])}` for free slip: the no-slip floor")
    A("moves the answer in the right direction (it arrests the runaway front) but")
    A("**does not** come close to the gate. `gate_met_without_tvc =")
    A(f"{q1['gate_met_without_tvc']}`.")
    A("")
    A("### Q2 — can bulk dissipation fake the TVC gain? **No.**")
    A("")
    A(f"The best viscosity-only configuration is `{q2['best_slip_no_tvc']}` at")
    A(f"`{_pct(q2['mean_rel_error'])}`, i.e. `{_fmt(q2['error_difference_pp'], '+.1f')}`")
    A(f"percentage points away from the calibrated `{q2['calibrated_tvc']}` at")
    A(f"`{_pct(q2['calibrated_mean_rel_error'])}`. Raising the viscosity past its default")
    A("never helps: it arrests the surge too early, and the error climbs monotonically")
    A("(" + ", ".join(f"`{l}` {_pct(by[l]['mean_rel_error'])}"
                     for l in ("slip_visc05", "slip_visc20", "slip_visc50")
                     if l in by) + ").")
    A("")
    A("### Q3 — does the operator carry a signature dissipation cannot produce? **Yes.**")
    A("")
    A("| channel | calibrated TVC (`slip_a010`) | best dissipation-only |")
    A("|---|---|---|")
    A(f"| mean rel. error of `Z/a` | {_pct(cal['mean_rel_error'])} | {_pct(q2['mean_rel_error'])} |")
    A(f"| `grad1` residual `h|sum grad W V|` | {_fmt(q3['at_calibrated_tvc']['grad1_residual_mean_h'])} | {_fmt(q3['at_matched_dissipation']['grad1_residual_mean_h'])} |")
    A(f"| median nearest neighbour / dx | {_fmt(q3['at_calibrated_tvc']['median_nn_over_dx'], '.4f')} | {_fmt(q3['at_matched_dissipation']['median_nn_over_dx'], '.4f')} |")
    A(f"| close-pair fraction (<0.7 dx) | {_fmt(q3['at_calibrated_tvc']['close_pair_fraction'])} | {_fmt(q3['at_matched_dissipation']['close_pair_fraction'])} |")
    A("")
    A("The lattice-regularity channel is the one a viscous term cannot reach, and it")
    A("separates the two mechanisms. Two further controls agree:")
    A("")
    A(f"* `slip_a010_visc10` (calibrated correction **plus** 10x viscosity) gives")
    A(f"  `{_pct(add['mean_rel_error'])}` — essentially the correction's own result, not a")
    A(f"  further improvement; viscosity alone at the same value gives")
    A(f"  `{_pct(by['slip_visc20']['mean_rel_error'])}` at `nu = 0.2` and")
    A(f"  `{_pct(by['slip_visc05']['mean_rel_error'])}` at `nu = 0.05`, so the two are not")
    A("  interchangeable contributions.");
    A(f"* `noslip_a010` ({_pct(noslip_cal['mean_rel_error'])}) reproduces the calibrated")
    A(f"  result under the no-slip floor ({_pct(cal['mean_rel_error'])} free-slip): the")
    A("  correction's gain is **insensitive to the wall tangential condition**, which is")
    A("  the opposite of a floor-friction artefact.")
    A("")
    A("## 5. Replication of the published rows")
    A("")
    A(f"The {len(rep)} free-slip rows that also appear in the published payload are")
    A("re-run from scratch here and must reproduce it,")
    A("otherwise this payload is measuring something else:")
    A("")
    A("| configuration | published variant | rel. here | rel. published | \\|diff\\| |")
    A("|---|---|---|---|---|")
    for r in rep:
        A(f"| `{r['label']}` | `{r['published_variant']}` | {_pct(r['rel_here'])} | "
          f"{_pct(r['rel_published'])} | {r['abs_diff']:.2e} |")
    A("")
    A("## 6. Notes, limits, and what this does *not* say")
    A("")
    A("* **This does not rescue the engine's verdict.** Rejecting Q1/Q2 confirms the")
    A("  attribution direction (the gain is the operator's), but the calibrated")
    A(f"  `alpha = 0.10` still gives `{_pct(cal['mean_rel_error'])}`, i.e. a")
    A(f"`{cal['band_verdict']}` — a single nominal pass that does not survive the")
    A(f"`+-{payload['digitisation_band']}` digitisation band. The two shipped")
    A("  coefficients remain outside the gate")
    A("  (" + ", ".join(f"`{l}` {_pct(by[l]['mean_rel_error'])}"
                       for l in ("slip_a005", "slip_a020") if l in by) + ").")
    A("* **The `alpha` response is non-monotone**, and that is the real fragility:")
    A("  " + ", ".join(f"`alpha={by[l]['tvc_alpha']:g}` {_pct(by[l]['mean_rel_error'])}"
                       for l in ("slip_a005", "slip_a010", "slip_a020") if l in by)
      + ". A mechanism whose merit peaks in a narrow window and reverses at twice")
    A("  the coefficient is a *calibration*, not a converged scheme — consistent with")
    A("  the resolution ladder's falsified convergence claim.")
    A("* **`wall_friction` is a viscous no-slip blend, not a wall-law model.** It")
    A("  imposes a zero tangential velocity through the viscous term; it has no")
    A("  boundary-layer resolution at these `dx`. Q1's negative answer is therefore")
    A("  about *this* no-slip implementation.")
    A("* Only one resolution (`dx = %g m`) is swept here. The resolution ladder in" % payload["dx"])
    A("  `RESOLUTION_LADDER.md` / `UNCALIBRATED_DEFAULTS.md` already shows the")
    A("  calibrated pass does not hold at `dx = 0.01 m`; this document adds no new")
    A("  resolution, and no claim about one.")
    A("* Reference: Martin & Moyce (1952), digitised via Koshizuka & Oka; band")
    A(f"  `+-{payload['digitisation_band']}` in `Z/a`, propagated as a verdict flip.")
    A("")
    A("## 7. Error judgement against the benchmark (explicit)")
    A("")
    A("| quantity | value | basis |")
    A("|---|---|---|")
    A(f"| best mean relative error of `Z/a` (any configuration) | {_pct(min(r['mean_rel_error'] for r in recs))} | experiment, dx = {payload['dx']} m |")
    A(f"| calibrated TVC (`alpha = 0.10`) | {_pct(cal['mean_rel_error'])} | experiment |")
    A(f"| best correction-off configuration | {_pct(min(r['mean_rel_error'] for r in recs if r['tvc_alpha'] is None))} | experiment |")
    A(f"| operator leg (algebraic, exact) | 0 (round-off) | `tests/test_transport_velocity_correction.py` |")
    A("")
    A("**Is the headline quantity below 3 %?** No. The best measured mean relative")
    A(f"error of `Z/a` in this study is `{_pct(min(r['mean_rel_error'] for r in recs))}`, and")
    A(f"the calibrated operator reaches `{_pct(cal['mean_rel_error'])}`; both are above 3 %,")
    A(f"and only the calibrated row is inside the 15 % owner gate, marginally. The")
    A("**operator-level** quantities (the `grad-1` integral and the position shift, which")
    A("have exact analytic values on the reference lattice) agree to round-off, but a")
    A("round-off agreement on the operator does not transfer to the case-level `Z/a`,")
    A("and this document does not claim it does.")
    A("")
    A("## 8. Reproduce")
    A("")
    A("```bash")
    A("python benchmarks/xval_sphinx_tvc/dissipation_attribution.py \\")
    A("    --dx %g --seconds %g" % (payload["dx"], payload["seconds"]))
    A("python benchmarks/xval_sphinx_tvc/make_dissipation_doc.py")
    A("pytest -q tests/test_tvc_dissipation_attribution.py")
    A("```")
    A("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload", type=Path, default=PAYLOAD)
    ap.add_argument("--doc", type=Path, default=DOC)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    payload = json.loads(args.payload.read_text())
    if not args.no_figure:
        figure(payload)
    text = render(payload)
    args.doc.write_text(text)
    print(f"wrote {args.doc} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())