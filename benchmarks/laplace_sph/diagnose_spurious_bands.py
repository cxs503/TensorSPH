"""Attribute the spurious-current reading of the dynamic Young-Laplace march.

The dynamic ladder (`run_dynamic_ladder.py`) reports one number per level,
``max|u|/v_ref`` over all particles, and gates it against Adami et al. (2012)'s
``0.10`` line.  That number decides nothing about the *formulation*: a residual
confined to the interface shell (where the curvature operator lives) and a bulk
current that never decays both read the same.  This driver runs the **same**
march with `band_report=True` and reports the split

    liquid bulk      r/R < 0.8
    interface shell  0.8 <= r/R <= 1.2
    vapour carrier   r/R > 1.2

plus the radial profile at the final step and the radial/tangential nature of
the worst particle.  Physics parameters are imported from `run_dynamic_ladder`
so the diagnostic cannot silently use a different case than the ladder.

    PYTHONPATH=src python benchmarks/laplace_sph/diagnose_spurious_bands.py \
        --levels 8,12 --threads 6 --render
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_dynamic_ladder import (  # noqa: E402  (path juggling above)
    HF, LEVELS, NU_PRIMARY, OUTER, R, RHO, SAMPLE, SIGMA, STEPS_PRIMARY,
)

from tensorsph.projection_adjoint import relax_projection_consistent  # noqa: E402

OUTDIR = HERE / "results_dynamic_ladder" / "spurious_bands"
DOC = HERE.parents[1] / "docs" / "xval_projection_laplace_spurious_bands.md"
FIG = OUTDIR / "laplace_sph_spurious_bands.png"

BANDS = ("liquid_core", "interface_shell", "vapour_carrier")


def run_level(level: int, steps: int, threads: int, nu_factor: float):
    dx = LEVELS[level]
    torch.set_num_threads(threads)
    t0 = time.time()
    rec = relax_projection_consistent(
        radius=R, dx=dx, sigma=SIGMA, rho0=RHO, outer=OUTER, h_factor=HF,
        steps=steps, sample_every=SAMPLE, nu_factor=nu_factor,
        pin_outer=True, rel_tol=1e-6, cg_max_iter=4000, band_report=True)
    rec["wall_s"] = time.time() - t0
    rec["job"] = "bands_Rdx%d" % level
    rec["threads"] = threads
    rec["steps_requested"] = steps
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / ("Rdx%d.json" % level)
    path.write_text(json.dumps(rec, indent=1))
    last = rec["history"][-1]
    print("[Rdx%d] n=%d dp_err=%.3f%% umax=%.4g bulk=%.4g shell=%.4g "
          "umax_r/R=%.3f radial=%.3f wall=%.0fs"
          % (level, rec["n_particles"], rec["dp_rel_error_pct"],
             last["umax_over_vref"], last["bands"]["liquid_core"]["max_over_vref"],
             last["bands"]["interface_shell"]["max_over_vref"],
             last["bands"]["umax_r_over_R"],
             last["bands"]["umax_radial_fraction"], rec["wall_s"]), flush=True)
    return rec


def load_levels():
    out = {}
    if OUTDIR.exists():
        for p in sorted(OUTDIR.glob("Rdx*.json")):
            blob = json.loads(p.read_text())
            out[int(blob["R_over_dx"])] = blob
    return out


def make_figure(levels: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorsph.projection_adjoint import BAND_EDGES

    lo, hi = BAND_EDGES
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) radial profile of |u|/v_ref at the final step, finest level available
    finest = max(levels)
    rec = levels[finest]
    prof = rec["history"][-1]["bands"]["profile"]
    r = [row["r_over_R"] for row in prof if row["n"]]
    mx = [row["max_over_vref"] for row in prof if row["n"]]
    mn = [row["mean_over_vref"] for row in prof if row["n"]]
    a = ax[0]
    a.axvspan(lo, hi, color="0.85", label="interface shell")
    a.semilogy(r, mx, "o-", label="max |u|/v_ref")
    a.semilogy(r, mn, "s-", label="mean |u|/v_ref")
    a.axhline(0.10, color="crimson", ls="--", lw=1.0,
              label="Adami gate 0.10")
    a.set_xlabel("r / R")
    a.set_ylabel("|u| / v_ref")
    a.set_title("(a) radial profile, R/dx = %d" % finest)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # (b) time history of the band maxima
    b = ax[1]
    hist = [s for s in rec["history"] if "bands" in s]
    t = [s["t_over_t_sigma"] for s in hist]
    for band, style in zip(BANDS, ("o-", "s-", "^-")):
        b.plot(t, [s["bands"][band]["max_over_vref"] for s in hist], style,
               label=band)
    b.plot(t, [s["umax_over_vref"] for s in hist], "k--", lw=1.0,
           label="global max (gated)")
    b.axhline(0.10, color="crimson", ls="--", lw=1.0)
    b.set_yscale("log")
    b.set_xlabel("t / t_sigma")
    b.set_ylabel("|u| / v_ref")
    b.set_title("(b) growth of each band, R/dx = %d" % finest)
    b.legend(fontsize=8)
    b.grid(alpha=0.3)

    # (c) attribution across levels: global vs bulk
    c = ax[2]
    xs = sorted(levels)
    glob = [levels[k]["history"][-1]["umax_over_vref"] for k in xs]
    bulk = [levels[k]["history"][-1]["bands"]["liquid_core"]["max_over_vref"]
            for k in xs]
    c.semilogy(xs, glob, "o-", label="global max (gated)")
    c.semilogy(xs, bulk, "s-", label="liquid bulk max")
    c.axhline(0.10, color="crimson", ls="--", lw=1.0, label="Adami gate 0.10")
    c.set_xlabel("R / dx")
    c.set_ylabel("|u| / v_ref")
    c.set_title("(c) what the gate is reading")
    c.legend(fontsize=8)
    c.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print("wrote", path)
    return path


def make_doc(levels: dict, fig: Path):
    finest = max(levels)
    rows = []
    for k in sorted(levels):
        rec = levels[k]
        last = rec["history"][-1]
        b = last["bands"]
        rows.append(
            "| %d | %d | %.3f | %.4f | %.4f | %.4f | %.3f | %.2f | %.3f |"
            % (k, rec["n_particles"], rec["dp_rel_error_pct"],
               last["umax_over_vref"], b["liquid_core"]["max_over_vref"],
               b["interface_shell"]["max_over_vref"],
               b["vapour_carrier"]["max_over_vref"], b["umax_r_over_R"],
               b["umax_radial_fraction"]))

    def verdict(v):
        return "PASS (<0.10)" if (isinstance(v, float) and math.isfinite(v)
                                  and v <= 0.10) else "FAIL (>=0.10)"

    fin = levels[finest]["history"][-1]
    global_v = fin["umax_over_vref"]
    bulk_v = fin["bands"]["liquid_core"]["max_over_vref"]
    shell_v = fin["bands"]["interface_shell"]["max_over_vref"]

    text = f"""# Where the projection-SPH spurious current lives (band attribution)

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · mission **m22813**
**Type** verification instrument (capability extension of the existing march, no new physics)

## ① Why the single gated number is not enough

The dynamic ladder gates `max|u|/v_ref` over **all** particles against Adami et al. (2012)'s
`0.10`. That gate cannot say *what* is being measured, and the two possibilities have opposite
meaning for the formulation:

* the residual lives in the **interface shell** — the curvature is a second derivative of the
  smoothed colour field, so its lattice noise is one order higher than the velocity it drives;
  such a residual is expected to scale with the colour-field stencil, not with the flow;
* the residual lives in the **liquid bulk** — then the projection is not holding the drop at rest
  and the "static equilibrium" claim is not a property of the formulation.

`relax_projection_consistent(..., band_report=True)` now reports the split
(`liquid_core` `r/R < 0.8`, `interface_shell` `0.8 ≤ r/R ≤ 1.2`, `vapour_carrier` `r/R > 1.2`),
the radial profile at the final step, and whether the worst particle moves radially (a breathing
mode, which changes the area) or tangentially (an area-neutral lattice current). The switch is
**off by default** so the shipped ladder records keep their schema and stay byte-reproducible.

## ② Case and settings

Same physics as the ladder (`run_dynamic_ladder` constants are imported, not re-typed):
`R = {R * 1e3:.1f} mm`, `sigma = {SIGMA} N/m`, `rho0 = {RHO} kg/m3`, `h/dx = {HF}`,
carrier padding `{OUTER * 1e3:.0f} mm`, `nu = {NU_PRIMARY}·sqrt(sigma R/rho)`,
`dt = 0.1 t_sigma`, outer layer velocity-pinned, PCG `rel_tol = 1e-6`.

## ③ Measured

| R/dx | particles | `dp` err % | global `max|u|/v_ref` | bulk max | shell max | vapour max | r(umax)/R | radial fraction of umax |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

At the finest level measured here (`R/dx = {finest}`): the gated global reading is
**{global_v:.4f}** → {verdict(global_v)}; the liquid-bulk reading is **{bulk_v:.4f}** →
{verdict(bulk_v)}; the interface-shell reading is **{shell_v:.4f}** → {verdict(shell_v)}.

![spurious-current band attribution]({fig.name})

## ④ What this does and does not settle

* It **does** make the ladder's gated number interpretable: the same number now has an owner.
* It does **not** by itself decide the ladder's open question (does the residual refine away).
  That needs several `R/dx` levels of the *same* march; the ladder jobs are submitted to the
  queued compute path and their verdicts are adjudicated on
  `docs/xval_projection_laplace_dynamic_ladder.md`, not here.
* The band edges are geometric (`0.8 R`, `1.2 R`) and therefore fixed with respect to the drop,
  not to `dx`: at `R/dx = 8` the shell is ~3 cells wide, at `R/dx = 24` it is ~10, so a shell
  fraction that is flat in `R/dx` is a physical statement only for `R/dx ≳ 12`.

## ⑤ Reproduce

```bash
PYTHONPATH=src python benchmarks/laplace_sph/diagnose_spurious_bands.py --levels 8 --threads 6 --render
PYTHONPATH=src python -m pytest tests/test_projection_spurious_bands.py -q
```
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print("wrote", DOC)
    return DOC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="8",
                    help="comma separated R/dx levels actually present in LEVELS")
    ap.add_argument("--steps", type=int, default=STEPS_PRIMARY)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--nu-factor", type=float, default=NU_PRIMARY)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--no-run", action="store_true",
                    help="render from the records already on disk")
    args = ap.parse_args()

    if not args.no_run:
        for lvl in [int(x) for x in args.levels.split(",") if x.strip()]:
            if lvl not in LEVELS:
                raise SystemExit("unknown level %d; LEVELS has %s"
                                 % (lvl, sorted(LEVELS)))
            run_level(lvl, args.steps, args.threads, args.nu_factor)

    levels = load_levels()
    if args.render:
        if not levels:
            raise SystemExit("no band records to render")
        make_figure(levels, FIG)
        make_doc(levels, FIG)


if __name__ == "__main__":
    main()