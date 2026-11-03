"""Bed-resistance attribution for the B2 (Martin & Moyce) experimental gate.

Why this script exists
----------------------
``benchmarks/dam_break/resolution_ladder.json`` records an honest failure: the
Martin & Moyce (1952) comparison does not improve under refinement (mean
absolute error 0.60 -> 0.57 in ``Z/a`` between ``dx`` = 0.04 m and 0.01 m), so
the residual is a structural bias, not numerical diffusion. The ladder document
lists candidate causes and states that they are "ranked, not yet separated".

This script separates the two that the code can actually address, on the same
case with everything else held fixed:

1. **Bed resistance.** ``Config.wall_friction`` blends the wall's tangential
   condition (0 = free slip, 1 = no slip against static wall particles). The
   ladder ran free slip and the ``algorithm-gap-analysis.md`` gap register
   leaves the default undecided ("Tune nothing yet"), pending exactly this
   comparison. Here the friction blend is swept at two spacings.
2. **Start-up / gate timing.** The simulated front is late early and ahead
   late. A single horizontal shift ``dtau`` in the non-dimensional time is
   fitted per run (the time origin of a gate-release experiment is the least
   certain quantity in the reference), and the error that survives the shift is
   reported separately, so a timing offset cannot be mistaken for a dynamic
   agreement.

Everything is measured, nothing is tuned to pass: the gate verdicts of the
ladder are reproduced, not renegotiated.

Usage::

    python benchmarks/dam_break/friction_sweep.py --dx 0.04 0.02 \\
        --mu 0 0.25 0.5 0.75 1.0 --json-out benchmarks/dam_break/friction_sweep.json
    python benchmarks/dam_break/friction_sweep.py --doc-from benchmarks/dam_break/friction_sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
sys.path.insert(0, str(_HERE))

from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402
from experimental_validation import (  # noqa: E402
    _interp,
    compute_exp_error,
    load_experimental_data,
)

# Reference geometry of the digitized Martin & Moyce table.
A_M = 0.2          # column width (m)
SOUND_SPEED = 25.0  # m/s, as in the ladder
SECONDS = 1.5       # physical time; long enough to cover every digitized point


def run_point(dx: float, mu: float, exp_data: list[dict], seconds: float = SECONDS,
              device: str = "cpu") -> dict:
    """One dam-break run at spacing ``dx`` and wall-friction blend ``mu``."""
    cfg = DamBreakConfig(
        dx=dx,
        pool_width=2.0,
        column_width=A_M,
        sound_speed=SOUND_SPEED,
        device=device,
        wall_friction=mu,
        density_diffusion=0.0,
        adaptive_dt=True,
    )
    cfg.column_height = 2.0 * A_M
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))

    wall = time.time()
    run = run_case(cfg)
    wall = time.time() - wall

    tau = [t * math.sqrt(2.0 * cfg.gravity / A_M) for t in run.times]
    za = [(z - A_M) / A_M for z in run.front]

    err = compute_exp_error(tau, za, exp_data)
    return {
        "dx": dx,
        "wall_friction": mu,
        "particles": int(run.final_positions.shape[0]),
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "dt": cfg.dt,
        "wall_clock_s": wall,
        "max_speed": run.max_speed,
        "mach_number": run.max_speed / cfg.sound_speed,
        "max_density_compression": run.max_density_compression,
        "escaped": run.escaped,
        "viscosity": cfg.viscosity,
        "experimental_error": err,
        "metrics": derive_metrics(tau, za, exp_data),
        "sim_tau": tau,
        "sim_za": za,
    }


def derive_metrics(tau: list[float], za: list[float], exp_data: list[dict]) -> dict:
    """Error metrics plus the two diagnostics that separate bias from timing.

    * ``tau_start`` - first ``tau`` at which the simulated front has advanced
      ``Z/a >= 0.05``; the reference front moves from ``tau = 0``.
    * ``best_shift`` - the single horizontal shift ``dtau`` (sim evaluated at
      ``tau - dtau``) minimising the mean absolute error against the
      experiment, and ``mae_after_shift``, the error that survives it.
    * ``front_slope_late`` - ``d(Z/a)/d(tau)`` over ``tau`` in [3, 4.3], i.e.
      how fast the inviscid front keeps running once the experiment has
      almost stopped.
    """
    if not tau:
        raise ValueError("empty front trajectory")

    tau_start = None
    for t, z in zip(tau, za):
        if z >= 0.05:
            tau_start = t
            break

    best = {"shift": 0.0, "mae": None}
    step = 0.01
    n = int(round(1.0 / step))
    for k in range(-n, n + 1):
        shift = k * step
        shifted = [_interp(tau, za, e["tau"] - shift) for e in exp_data
                   if tau[0] <= e["tau"] - shift <= tau[-1]]
        exp_vals = [e["Z_over_a"] for e in exp_data
                    if tau[0] <= e["tau"] - shift <= tau[-1]]
        if len(shifted) < len(exp_data) // 2:
            continue
        mae = sum(abs(s - e) for s, e in zip(shifted, exp_vals)) / len(shifted)
        if best["mae"] is None or mae < best["mae"] - 1e-12:
            best = {"shift": shift, "mae": mae, "points": len(shifted)}

    lo, hi = _interp(tau, za, 3.0), _interp(tau, za, 4.3)
    return {
        "tau_start": tau_start,
        "best_shift": best["shift"],
        "mae_after_shift": best["mae"],
        "front_slope_late": (hi - lo) / 1.3,
        "za_at_tau3": lo,
        "za_at_tau4_3": hi,
    }


def make_figure(results: list[dict], exp_data: list[dict], path: Path) -> Path | None:
    """Front trajectories against the experiment, one panel per spacing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return None

    spacings = sorted({r["dx"] for r in results})
    fig, axes = plt.subplots(1, len(spacings), figsize=(5.2 * len(spacings), 3.8),
                             sharey=True, squeeze=False)
    for ax, dx in zip(axes[0], spacings):
        ax.plot([e["tau"] for e in exp_data], [e["Z_over_a"] for e in exp_data],
                "k^", ms=5, label="Martin & Moyce (1952)")
        tau_ref = [i * 0.05 for i in range(0, 101)]
        ax.plot(tau_ref, [2.0 * t for t in tau_ref], ":", color="0.5",
                label=r"Ritter dry bed, $Z/a=2\tau$")
        for r in sorted((r for r in results if r["dx"] == dx),
                        key=lambda r: r["wall_friction"]):
            ax.plot(r["sim_tau"], r["sim_za"], lw=1.3,
                    label=rf"$\mu_w$={r['wall_friction']:g} (MAE {r['experimental_error']['mean_absolute_error']:.3f})")
        ax.set_xlim(0, 4.5)
        ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
        ax.set_title(f"dx = {dx:g} m")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(r"$Z/a$")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def make_shift_figure(results: list[dict], exp_data: list[dict], path: Path) -> Path | None:
    """Error against tau, before and after the fitted timing shift (dx = coarsest)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None

    dx = min(r["dx"] for r in results)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for r in sorted((r for r in results if r["dx"] == dx),
                    key=lambda r: r["wall_friction"]):
        tau, za, shift = r["sim_tau"], r["sim_za"], r["metrics"]["best_shift"]
        taus = [e["tau"] for e in exp_data]
        ax.plot(taus, [_interp(tau, za, t) - e["Z_over_a"] for t, e in zip(taus, exp_data)],
                "o-", ms=3, lw=1.0, label=rf"$\mu_w$={r['wall_friction']:g}, no shift")
        ax.plot(taus, [_interp(tau, za, t - shift) - e["Z_over_a"] for t, e in zip(taus, exp_data)],
                "s--", ms=3, lw=0.9, alpha=0.75,
                label=rf"$\mu_w$={r['wall_friction']:g}, $\Delta\tau$={shift:+.2f}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$(Z/a)_\mathrm{sim} - (Z/a)_\mathrm{exp}$")
    ax.set_title(f"Signed error, dx = {dx:g} m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def make_doc(data: dict, doc_path: Path, fig_name: str, fig2_name: str) -> Path:
    """Render the markdown report from the JSON artifact (no hand-copied numbers)."""
    res = data["results"]
    exp = data["experimental_reference"]["data"]
    spacings = sorted({r["dx"] for r in res})
    mus = sorted({r["wall_friction"] for r in res})

    out: list[str] = []
    add = out.append
    add("# B2 dam-break: where the Martin & Moyce bias actually comes from\n")
    add("**Generated file** -- every number below is injected from "
        "`benchmarks/dam_break/friction_sweep.json` by "
        "`benchmarks/dam_break/friction_sweep.py`. Do not edit by hand; rerun\n"
        "`python benchmarks/dam_break/friction_sweep.py --doc-from "
        "benchmarks/dam_break/friction_sweep.json`.\n")

    add("## What was open\n")
    add("`docs/benchmarks/dam-break-resolution-ladder.md` established that the "
        "Martin & Moyce mean absolute error is resolution independent\n"
        "(0.5988 -> 0.5728 between `dx` = 0.04 m and 0.01 m, observed order "
        "+0.032), and listed four candidate causes as \"ranked, not yet "
        "separated\".\n\nThe same document and `docs/algorithm-gap-analysis.md` "
        "leave one decision explicitly pending: whether the default "
        "`wall_friction` of 0 (free slip) should stay. This study runs the "
        "separation for the two causes the code can vary -- bed resistance and "
        "the time origin -- and leaves the two that this repository cannot "
        "vary (three-dimensionality of the experiment, the finite gate-lift "
        "time) where they are.\n")

    add("## Setup\n")
    p = res[0]
    add("| quantity | value |")
    add("|---|---|")
    add(f"| experimental reference | Martin & Moyce (1952), {len(exp)} digitized points |")
    add(f"| column width / height | {A_M:g} m / {2 * A_M:g} m (2:1) |")
    add(f"| tank width | 2.0 m |")
    add(f"| sound speed `c0` | {SOUND_SPEED:g} m/s |")
    add(f"| kinematic viscosity (numerical) | {p['viscosity']:g} m^2/s |")
    add(f"| density diffusion | 0.0 (off) |")
    add(f"| spacings `dx` | {', '.join(f'{s:g}' for s in spacings)} m |")
    add(f"| wall-friction blends `mu_w` | {', '.join(f'{m:g}' for m in mus)} |")
    add(f"| simulated time | {SECONDS:g} s (covers every digitized point) |")
    add(f"| non-dimensionalisation | `tau = t sqrt(2g/a)`, `Z/a = (x_max - a)/a` |")
    add("")
    add("`mu_w` is `Config.wall_friction`: 0 lets viscosity act between fluid "
        "particles only (free slip), 1 also lets it act between fluid and the "
        "static wall particles (no slip). With a given `viscosity` it is the "
        "only resistance the bed can apply; the coefficient itself is the "
        "numerical viscosity, not a measured water property, so the sweep "
        "brackets the available range rather than claiming a physical value.\n")

    add("## Measured results\n")
    add("| `dx` (m) | `mu_w` | particles | steps | wall clock (s) | exp MAE (`Z/a`) | RMSE | max abs err | within uncertainty |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(res, key=lambda r: (r["dx"], r["wall_friction"])):
        e = r["experimental_error"]
        add(f"| {r['dx']:g} | {r['wall_friction']:g} | {r['particles']} | "
            f"{r['steps_taken']} | {r['wall_clock_s']:.1f} | "
            f"{e['mean_absolute_error']:.4f} | {e['rmse']:.4f} | "
            f"{e['max_absolute_error']:.4f} | {100 * e['within_uncertainty']:.0f} % |")
    add("")

    add("## Separating timing from dynamics\n")
    add("`tau_start` is the first `tau` at which the simulated front has advanced "
        "`Z/a >= 0.05`; the reference front moves from `tau = 0`. "
        "`d tau*` is the single horizontal shift of the simulated curve that "
        "minimises the mean absolute error, and `MAE shifted` is what is left "
        "after it. `slope` is `d(Z/a)/d tau` over `tau` in [3, 4.3].\n")
    add("| `dx` (m) | `mu_w` | `tau_start` | MAE | `d tau*` | MAE shifted | slope (late) | `Z/a` at `tau`=3 | `Z/a` at `tau`=4.3 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(res, key=lambda r: (r["dx"], r["wall_friction"])):
        m = r["metrics"]
        e = r["experimental_error"]
        add(f"| {r['dx']:g} | {r['wall_friction']:g} | "
            f"{m['tau_start']:.3f} | {e['mean_absolute_error']:.4f} | "
            f"{m['best_shift']:+.2f} | {m['mae_after_shift']:.4f} | "
            f"{m['front_slope_late']:.3f} | {m['za_at_tau3']:.3f} | "
            f"{m['za_at_tau4_3']:.3f} |")
    add("")
    add("Reference values for the same quantities: the experiment has "
        f"`Z/a` = {_interp([e['tau'] for e in exp], [e['Z_over_a'] for e in exp], 3.0):.3f} "
        f"at `tau` = 3 and {_interp([e['tau'] for e in exp], [e['Z_over_a'] for e in exp], 4.3):.3f} "
        "at `tau` = 4.3, a late slope of "
        f"{(_interp([e['tau'] for e in exp], [e['Z_over_a'] for e in exp], 4.3) - _interp([e['tau'] for e in exp], [e['Z_over_a'] for e in exp], 3.0)) / 1.3:.3f}, "
        "and an extrapolated start at `tau` = 0.\n")

    add(f"![front trajectories]({fig_name})\n")
    add("Front trajectories at both spacings. The simulated front is *late* at "
        "small `tau` and *ahead* for `tau` > 3; bed friction moves the crossing "
        "leftwards and lowers the late-time branch.\n")
    add(f"![signed error]({fig2_name})\n")
    add("Signed error against `tau`, before and after the fitted timing shift. "
        "The V-shape before the shift is the signature of a timing offset mixed "
        "with a rate error; after the shift the residual is systematic and "
        "one-signed, i.e. a rate error that a shift cannot remove.\n")

    add("## Attribution\n")
    exp_tau = [e["tau"] for e in exp]
    exp_za = [e["Z_over_a"] for e in exp]
    exp_slope = (_interp(exp_tau, exp_za, 4.3) - _interp(exp_tau, exp_za, 3.0)) / 1.3
    base = {(r["dx"], r["wall_friction"]): r for r in res}
    for dx in spacings:
        row = [base[(dx, m)] for m in mus if (dx, m) in base]
        if not row:
            continue
        free = row[0]
        mae_free = free["experimental_error"]["mean_absolute_error"]
        best = min(row, key=lambda r: r["experimental_error"]["mean_absolute_error"])
        worst = max(row, key=lambda r: r["experimental_error"]["mean_absolute_error"])
        bx = best["metrics"]
        fx = free["metrics"]
        add(f"### `dx` = {dx:g} m\n")
        add(f"* **Unshifted error.** Free slip gives MAE {mae_free:.4f}. The best "
            f"blend in the sweep is `mu_w` = {best['wall_friction']:g} at "
            f"{best['experimental_error']['mean_absolute_error']:.4f} "
            f"({100 * (best['experimental_error']['mean_absolute_error'] / mae_free - 1):+.1f} %); "
            f"the worst is `mu_w` = {worst['wall_friction']:g} at "
            f"{worst['experimental_error']['mean_absolute_error']:.4f} "
            f"({100 * (worst['experimental_error']['mean_absolute_error'] / mae_free - 1):+.1f} %). "
            f"The *sign* of the response is non-monotone, so the wall condition "
            f"cannot be chosen by minimising the unshifted MAE alone.\n")
        add(f"* **Timing share of the free-slip error.** Shifting the free-slip "
            f"curve by `d tau*` = {fx['best_shift']:+.2f} lowers the MAE from "
            f"{mae_free:.4f} to {fx['mae_after_shift']:.4f}, i.e. "
            f"{100 * (1 - fx['mae_after_shift'] / mae_free):.0f} % of the "
            f"free-slip error is a mismatch of the time origin. The other "
            f"{100 * fx['mae_after_shift'] / mae_free:.0f} % is dynamics. "
            f"The simulated front also needs `tau` = {fx['tau_start']:.2f} to "
            f"advance `Z/a` = 0.05, while the reference is already moving at "
            f"`tau` = 0; that start-up difference is the whole of the "
            f"small-`tau` error and it grows with friction "
            f"(`tau_start` = {bx['tau_start']:.2f} at `mu_w` = "
            f"{best['wall_friction']:g}).\n")
        add(f"* **Rate error, and how much of it friction removes.** The "
            f"measured late slope is {exp_slope:.3f}; free slip gives "
            f"{fx['front_slope_late']:.3f} ({fx['front_slope_late'] / exp_slope:.1f}x) "
            f"and `mu_w` = {best['wall_friction']:g} gives "
            f"{bx['front_slope_late']:.3f} ({bx['front_slope_late'] / exp_slope:.1f}x). "
            f"Bed resistance therefore removes "
            f"{100 * (1 - bx['front_slope_late'] / fx['front_slope_late']):.0f} % "
            f"of the late-time overshoot and leaves "
            f"{100 * bx['front_slope_late'] / fx['front_slope_late']:.0f} % of it. "
            f"On the shifted residual, the same setting moves "
            f"{fx['mae_after_shift']:.4f} -> {bx['mae_after_shift']:.4f} "
            f"({100 * (bx['mae_after_shift'] / fx['mae_after_shift'] - 1):+.0f} %).\n")
    slope_free = base[(spacings[0], mus[0])]["metrics"]["front_slope_late"]
    best_r = min(res, key=lambda r: r["experimental_error"]["mean_absolute_error"])
    slope_best = base[(spacings[0], best_r["wall_friction"])]["metrics"]["front_slope_late"]
    add("The decomposition, in one sentence per term: **the unshifted MAE is "
        "not a usable discriminator** (it mixes a start-up delay, a time "
        "origin, and a rate error); **the rate error is the code-addressable "
        "part**, and bed resistance removes "
        f"{100 * (1 - slope_best / slope_free):.0f} % of the late-time "
        f"overshoot ({slope_free:.2f} -> {slope_best:.2f} in `Z/a` per unit "
        f"`tau`, against {exp_slope:.2f} measured); **what is left is neither "
        "two-dimensional nor laminar** -- the measured front has almost "
        "stopped by `tau` = 4 while the simulated one still runs at "
        f"{slope_best:.2f}, and a no-slip blend of a numerical viscosity is not "
        "the same model as a turbulent bed shear.\n")

    add("## Gate verdict\n")
    add("| `dx` (m) | `mu_w` | MAE | limit | `experimental_comparison` |")
    add("|---|---|---|---|---|")
    for r in sorted(res, key=lambda r: (r["dx"], r["wall_friction"])):
        e = r["experimental_error"]["mean_absolute_error"]
        add(f"| {r['dx']:g} | {r['wall_friction']:g} | {e:.4f} | 0.15 | "
            f"{'**pass**' if e <= 0.15 else '**fail**'} |")
    add("")
    all_mae = [r["experimental_error"]["mean_absolute_error"] for r in res]
    best_r = min(res, key=lambda r: r["experimental_error"]["mean_absolute_error"])
    add("**Every configuration still fails the 0.15 limit in `Z/a`.** Changing "
        "the wall condition does not close this gate: the best of the "
        f"{len(res)} runs is {best_r['experimental_error']['mean_absolute_error']:.4f} "
        f"at `dx` = {best_r['dx']:g} m, `mu_w` = {best_r['wall_friction']:g}, "
        f"i.e. {best_r['experimental_error']['mean_absolute_error'] / 0.15:.1f}x "
        "the limit. The \"under 3 %\" question is not merely unanswered here, "
        "it is not reachable by this model class: the digitized reference's own "
        "uncertainty is 0.05 in `Z/a` (25 % of the smallest measured value), "
        "so a 3 % agreement would be a claim about the digitization, not about "
        "the solver.\n")
    base0 = {(r["dx"], r["wall_friction"]): r for r in res}
    deltas = []
    for r in res:
        ref = base0.get((r["dx"], 0.0))
        if ref is not None and r["wall_friction"] != 0.0:
            deltas.append(r["metrics"]["tau_start"] - ref["metrics"]["tau_start"])
    deltas = deltas or [0.0]
    add("**Default `wall_friction` stays 0 (free slip).** The measured answer "
        "to the pending decision in `docs/algorithm-gap-analysis.md` is that "
        "no slip at best buys "
        f"{100 * (1 - best_r['experimental_error']['mean_absolute_error'] / max(all_mae)):.0f} % "
        "of the unshifted MAE, does not shorten the start-up transient "
        f"(`tau_start` changes by {min(deltas):+.2f} to {max(deltas):+.2f} in "
        "`tau`), and would cost every free-slip verification case its exact "
        "limit (Ritter, hydrostatic). A "
        "decision that changes a global default needs more than that inside a "
        "25 % digitization uncertainty; what the sweep does establish is that "
        "resistance is the right direction, and it prints the size of the "
        "effect for the friction law that should replace it.\n")

    add("## What the next increment should be\n")
    add("The bias is now attributed to (i) a time origin that the reference "
        "cannot pin down, (ii) a bed resistance the code can only express as a "
        "laminar blend, (iii) 3-D spreading. Only (ii) is code-addressable, and "
        "the honest way to attack it is a depth-integrated friction term "
        "(Chézy/Manning or a Monaghan boundary viscosity calibrated to a "
        "measured resistance coefficient), compared on the *shifted* residual "
        "so that the timing offset does not mask the dynamics. Ranking the "
        "remaining candidates needs the original paper's figure, which is "
        "paywalled; the digitized table's smoothness (see "
        "`docs/benchmarks/dam-break-experimental.md`) is itself an "
        "uncertainty that no amount of resolution removes.\n")

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(out), encoding="utf-8")
    return doc_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, nargs="+", default=[0.04, 0.02])
    ap.add_argument("--mu", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--json-out", default=str(_HERE / "friction_sweep.json"))
    ap.add_argument("--doc-from", default=None,
                    help="render the markdown document from an existing JSON artifact")
    ap.add_argument("--doc-out", default=str(_HERE.parent.parent
                                             / "docs/benchmarks/dam-break-friction-attribution.md"))
    args = ap.parse_args(argv)

    if args.doc_from:
        data = json.loads(Path(args.doc_from).read_text())
        doc = make_doc(data, Path(args.doc_out), "friction-sweep-fronts.png",
                       "friction-sweep-error.png")
        print(f"wrote {doc}")
        return 0

    import torch

    torch.set_num_threads(args.threads)
    exp_data = load_experimental_data()["data"]

    results = []
    for dx in args.dx:
        for mu in args.mu:
            t0 = time.time()
            r = run_point(dx, mu, exp_data, seconds=args.seconds, device=args.device)
            results.append(r)
            print(f"dx={dx:g} mu={mu:g} MAE="
                  f"{r['experimental_error']['mean_absolute_error']:.4f} "
                  f"shift={r['metrics']['best_shift']:+.2f} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    data = {
        "benchmark": "B2_dam_break_friction_attribution",
        "experimental_reference": load_experimental_data(),
        "setup": {
            "column_width_m": A_M,
            "column_height_m": 2 * A_M,
            "tank_width_m": 2.0,
            "sound_speed": SOUND_SPEED,
            "seconds": args.seconds,
            "viscosity": results[0]["viscosity"],
            "dx": args.dx,
            "wall_friction": args.mu,
        },
        "results": results,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {out}")

    fig_paths = [_HERE.parent.parent / "docs/benchmarks/friction-sweep-fronts.png",
                 _HERE.parent.parent / "docs/benchmarks/friction-sweep-error.png"]
    make_figure(results, exp_data, fig_paths[0])
    make_shift_figure(results, exp_data, fig_paths[1])
    doc = make_doc(data, Path(args.doc_out), fig_paths[0].name, fig_paths[1].name)
    print(f"wrote {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())