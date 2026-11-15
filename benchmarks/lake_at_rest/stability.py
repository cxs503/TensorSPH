"""How long does the still lake actually stay still?

Case-pool anchor ``CASE-SPH-LAKE-AT-REST`` (reference: the analytic hydrostatic
identity, ``u = 0`` and a flat free surface; tolerance: machine precision,
``1e-9``).  The published PASS in ``docs/verification_lake_at_rest_periodic.md``
is measured over ``t_end = 0.50 s`` (1282 steps at ``dt = 3.9e-4 s``).  A
tolerance without a horizon is not a verification statement, so this runner
measures the horizon:

* **L1 certified horizon.**  Largest simulated time for which the rest state
  still satisfies ``max|v|/c0 < 1e-9``, ``max|dy|/H < 1e-9``,
  ``max|drho|/rho0 < 1e-9`` -- reported as the running maximum, i.e. it is the
  time at which the *worst* deviation so far crosses the tolerance.
* **L2 growth law.**  Slope of ``ln max|v|/c0`` against ``t`` over the window
  where the motion is neither roundoff nor saturated (a straight line in this
  window is an exponential departure; ``R^2`` is reported next to it).
* **L3 step-size discrimination.**  The same case at half the acoustic Courant
  number.  If the growth rate (per *second*) is unchanged the departing mode is
  a property of the discrete equilibrium; if it scales with ``dt`` the departure
  is manufactured by the explicit integrator.  This is the control that decides
  which of the two honest verdicts applies.
* **L4 seed linearity.**  The same case seeded with an explicit displacement
  ``eps * dx`` in ``y`` at ``eps = 1e-8`` and ``1e-6``: a linear instability has
  the same slope and a vertical offset of ``ln(eps1/eps2) / lambda``.
* **L5 mode identification.**  Dominant ``x``-wavenumber (periodic direction) of
  the growing vertical-displacement field, from the column-mean displacement
  spectrum.  Grid-scale ``k`` points at a kernel/pairing mode, low ``k`` at a
  bulk (sloshing-like) mode.
* **L6 pair-list staleness.**  The neighbour list is frozen at ``t = 0`` by the
  case (``build_state`` docstring).  Rebuilding it every 20 steps tells whether
  the departure is an artefact of the frozen list.
* **L7 resolution.**  ``dx = 0.025 m`` ladder point, same protocol.
* **L8 saturation.**  A long run (8 s) measures what the exponential departure
  saturates to: whether the lake ends up sloshing at finite amplitude or settles.

Nothing here changes shipped physics: every arm drives
:func:`tensorsph.droplet_impact.rhs` through :mod:`tensorsph.lake_at_rest`.

Usage (the run is minutes long, so it is submitted, not run in the foreground)::

    PYTHONPATH=src python3 benchmarks/lake_at_rest/stability.py \
        --json tests/data/lake_at_rest/stability.json --figures docs/figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

import band_fit
from tensorsph.droplet_impact import rhs
from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    periodic_pairs, reconstruct)

CASE_ID = "CASE-SPH-LAKE-AT-REST"
TOLERANCE = 1.0e-9          # anchor tolerance: machine precision
FIT_LOW = 1.0e-10           # fit window: above roundoff ...
FIT_HIGH = 1.0e-5           # ... and below saturation


# ---------------------------------------------------------------------------
# one arm of the study
# ---------------------------------------------------------------------------
def _column_index(pos: torch.Tensor) -> torch.Tensor:
    """Map every fluid particle to its lattice column (``x``), order-independent."""
    xs, inverse = torch.unique(pos[:, 0], return_inverse=True)
    return inverse, int(xs.numel())


def integrate(cfg: LakeConfig, arm: dict) -> dict:
    """Advance the reconstructed lake and record the departure from rest.

    ``arm`` keys: ``seed`` (displacement amplitude in units of ``dx``),
    ``rebuild_pairs`` (period in steps, 0 = frozen list), ``sample`` (steps
    between samples).
    """
    torch.set_default_dtype(cfg.torch_dtype)
    state = build_state(cfg)
    prior = analytic_profile(state, cfg)["mass"]
    solution = reconstruct(state, cfg, prior=prior)
    mass = solution["mass"]

    pos = state["pos"].clone()
    if arm.get("seed", 0.0):
        pos = pos.clone()
        pos[:, 1] = pos[:, 1] + float(arm["seed"]) * cfg.dx
    vel = torch.zeros_like(pos)
    dt = cfg.dt()
    steps = int(round(cfg.t_end / dt))
    y0 = state["pos"][:, 1].clone()
    rho_ref = rhs(state["pos"], torch.zeros_like(pos), mass, state["config"],
                  pairs=state["pairs"], wall=state["wall"],
                  wrap_width=state["wrap_width"])[0]
    column, n_columns = _column_index(state["pos"])
    total_mass = float(mass.sum())
    sample = int(arm.get("sample", 25))
    rebuild = int(arm.get("rebuild_pairs", 0))

    samples = []
    worst = {"v": 0.0, "dy": 0.0, "rho": 0.0}
    first_cross = {"v": None, "dy": None, "rho": None}   # tolerance horizon
    pairs = state["pairs"]
    wall = state["wall"]
    for step in range(steps):
        if rebuild and step and step % rebuild == 0:
            pairs = periodic_pairs(pos, wall, cfg.h(), cfg.width)
        rho, _p, acc = rhs(pos, vel, mass, state["config"], pairs=pairs,
                           wall=wall, wrap_width=state["wrap_width"])
        vel = vel + dt * acc
        pos = pos + dt * vel
        v = float(vel.abs().max()) / cfg.sound_speed
        dy = float((pos[:, 1] - y0).abs().max()) / cfg.depth
        drho = float(((rho - rho_ref) / cfg.rho0).abs().max())
        t = (step + 1) * dt
        for key, value in (("v", v), ("dy", dy), ("rho", drho)):
            worst[key] = max(worst[key], value)
            if first_cross[key] is None and worst[key] > TOLERANCE:
                first_cross[key] = t
        if (step + 1) % sample == 0 or step == steps - 1:
            ke = float(0.5 * (mass[:, None] * vel ** 2).sum())
            column_mean = torch.zeros(n_columns, dtype=pos.dtype)
            column_mean = column_mean.index_add(
                0, column, pos[:, 1] - y0) / torch.bincount(
                    column, minlength=n_columns).to(pos.dtype)
            spectrum = torch.fft.rfft(column_mean).abs()
            dominant = int(spectrum[1:].argmax().item()) + 1 if n_columns > 2 else 0
            samples.append({
                "t": t, "v_over_c0": v, "dy_over_H": dy,
                "drho_over_rho0": drho,
                "ke_over_mc02": ke / (total_mass * cfg.sound_speed ** 2),
                "column_mode": dominant,
                "column_amplitude": float(spectrum[dominant].item()
                                          / max(n_columns, 1)),
            })
    certified = {"v": first_cross["v"], "dy": first_cross["dy"],
                 "rho": first_cross["rho"]}
    return {
        "seed_in_dx": float(arm.get("seed", 0.0)),
        "rebuild_pairs_every": rebuild,
        "dt": dt, "steps": steps, "t_end": steps * dt, "sample_every": sample,
        "equilibrium": {
            "a_y_over_g": solution["a_max_over_g"],
            "a_x_over_g": solution["a_max_over_g_horizontal"],
            "iterations": solution["iterations"],
            "converged": solution["converged"],
        },
        "certified_horizon_s": certified,
        "max_v_over_c0": worst["v"], "max_dy_over_H": worst["dy"],
        "max_drho_over_rho0": worst["rho"],
        "samples": samples,
        "columns": n_columns,
    }


# ---------------------------------------------------------------------------
# reduction of one arm to the verdict numbers
# ---------------------------------------------------------------------------
def growth_fit(samples: list[dict], field: str = "v_over_c0") -> dict:
    """Exponential slope of ``field`` in the ``FIT_LOW .. FIT_HIGH`` window.

    Fitted on ``ln`` against ``t`` by least squares *and* on ``ln`` against
    ``ln t`` so the answer says which law the data prefer; a straight line in
    the first is exponential growth, in the second a power law.
    """
    window = [(s["t"], s[field]) for s in samples
              if FIT_LOW <= s[field] <= FIT_HIGH and s[field] > 0.0]
    if len(window) < 4:
        return {"points": len(window), "lambda_per_s": None, "r2_exp": None,
                "power_exponent": None, "r2_power": None}
    t = [w[0] for w in window]
    ln_v = [math.log(w[1]) for w in window]
    n = len(t)
    mean_t = sum(t) / n
    mean_y = sum(ln_v) / n
    sxx = sum((ti - mean_t) ** 2 for ti in t)
    sxy = sum((ti - mean_t) * (yi - mean_y) for ti, yi in zip(t, ln_v))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_t
    ss_res = sum((yi - (intercept + slope * ti)) ** 2 for ti, yi in zip(t, ln_v))
    ss_tot = sum((yi - mean_y) ** 2 for yi in ln_v)
    ln_t = [math.log(ti) for ti in t]
    mean_lt = sum(ln_t) / n
    sxx_p = sum((li - mean_lt) ** 2 for li in ln_t)
    sxy_p = sum((li - mean_lt) * (yi - mean_y) for li, yi in zip(ln_t, ln_v))
    slope_p = sxy_p / sxx_p if sxx_p else float("nan")
    intercept_p = mean_y - slope_p * mean_lt
    ss_res_p = sum((yi - (intercept_p + slope_p * li)) ** 2
                   for li, yi in zip(ln_t, ln_v))
    return {
        "points": n,
        "t_from": min(t), "t_to": max(t),
        "lambda_per_s": slope,
        "r2_exp": 1.0 - ss_res / ss_tot if ss_tot else None,
        "power_exponent": slope_p,
        "r2_power": 1.0 - ss_res_p / ss_tot if ss_tot else None,
        "fit_form": "ln(v) = a + lambda t",
    }


def arm_report(name: str, cfg: LakeConfig, run: dict) -> dict:
    fit = growth_fit(run["samples"])
    # The band estimator (``benchmarks/lake_at_rest/band_fit.py``) is recorded
    # next to the legacy single-window fit because the legacy window is not an
    # exponential-rate estimator for an arm that starts moving (see that
    # module's docstring): on the seeded arms it fits the transient and
    # returns R^2 < 0.6.  Both numbers ship; the gates use the band one.
    band = band_fit.fit_lambda(run["samples"], band_fit.BAND_LOW,
                              band_fit.BAND_HIGH)
    peak = max(run["samples"], key=lambda s: s["dy_over_H"])
    return {
        "arm": name,
        "config": {
            "dx": cfg.dx, "width": cfg.width, "depth": cfg.depth,
            "periodic_x": cfg.periodic_x, "max_courant": cfg.max_courant,
            "viscosity": cfg.viscosity, "h": cfg.h(),
            "sound_speed": cfg.sound_speed, "gravity": cfg.gravity,
        },
        "run": {k: v for k, v in run.items() if k != "samples"},
        "samples": run["samples"],       # raw series: the gates re-derive from it
        "growth": fit,
        "growth_band": band,
        "peak_sample": peak,
    }


def build_arms(args) -> list[tuple[str, LakeConfig, dict]]:
    """The study, as (name, config, protocol) triples.  Declared, not tuned."""
    base = dict(periodic_x=True, dx=args.dx, t_end=args.t_end, width=0.40,
                depth=0.30)
    arms = [
        ("base", dict(base), {"sample": 25}),
        ("courant_half", dict(base, max_courant=0.15), {"sample": 50}),
        ("resolution_dx025", dict(base, dx=0.025), {"sample": 25}),
        ("pairs_rebuilt", dict(base), {"sample": 25, "rebuild_pairs": 20}),
        ("seed_1e-6", dict(base), {"sample": 25, "seed": 1.0e-6}),
        ("seed_1e-8", dict(base), {"sample": 25, "seed": 1.0e-8}),
        ("saturation_8s", dict(base, t_end=8.0), {"sample": 200}),
    ]
    if args.viscosity:
        arms.append(("viscosity_1e-6", dict(base, viscosity=1.0e-6), {"sample": 25}))
    return [(name, LakeConfig(**cfg), proto) for name, cfg, proto in arms]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="tests/data/lake_at_rest/stability.json")
    ap.add_argument("--figures", default="docs/figures")
    ap.add_argument("--dx", type=float, default=0.020)
    ap.add_argument("--t-end", type=float, default=2.5)
    ap.add_argument("--viscosity", type=float, default=0.0)
    ap.add_argument("--arms", default="")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    started = time.time()
    arms = build_arms(args)
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",") if a.strip()}
        arms = [a for a in arms if a[0] in keep]
    reports, raw = [], {}
    for name, cfg, proto in arms:
        t0 = time.time()
        run = integrate(cfg, proto)
        reports.append(arm_report(name, cfg, run))
        raw[name] = run["samples"]
        print(f"[{name}] steps={run['steps']} "
              f"t*_v={run['certified_horizon_s']['v']} "
              f"lambda={reports[-1]['growth']['lambda_per_s']} "
              f"({time.time() - t0:.0f} s)", flush=True)

    by_name = {r["arm"]: r for r in reports}

    def lam(name):
        return by_name.get(name, {}).get("growth", {}).get("lambda_per_s")

    base_lam = lam("base")
    half_lam = lam("courant_half")
    verdict = {
        "tolerance": TOLERANCE,
        "claimed_horizon_s": 0.50,
        "certified_horizon_s": by_name.get("base", {}).get(
            "run", {}).get("certified_horizon_s", {}).get("v"),
        "lambda_base_per_s": base_lam,
        "lambda_courant_half_per_s": half_lam,
        "lambda_ratio_dt_half": (half_lam / base_lam
                                 if base_lam and half_lam else None),
        "persists_to_claimed_horizon": (
            None if "base" not in by_name else
            (by_name["base"]["run"]["certified_horizon_s"]["v"] is None
             or by_name["base"]["run"]["certified_horizon_s"]["v"] >= 0.50)),
    }
    payload = {
        "case": CASE_ID,
        "kind": "rest-horizon and instability study (addendum to the PASS)",
        "reference": "analytic hydrostatic identity (u = 0, eta = const)",
        "tolerance": TOLERANCE,
        "fit_window": [FIT_LOW, FIT_HIGH],
        "runtime_s": time.time() - started,
        "torch": torch.__version__,
        "verdict": verdict,
        "arms": reports,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))

    if not args.no_figures:
        try:
            make_figures(payload, raw, Path(args.figures))
        except Exception as exc:                       # figures are not the gate
            print(f"figures skipped: {exc}")
    print(json.dumps(verdict, indent=1))
    return 0


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(payload: dict, raw: dict, outdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    # (1) departure from rest against time, every arm, on a log scale
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, samples in raw.items():
        t = [s["t"] for s in samples]
        axes[0].semilogy(t, [max(s["v_over_c0"], 1e-20) for s in samples],
                         label=name, lw=1.2)
        axes[1].semilogy(t, [max(s["dy_over_H"], 1e-20) for s in samples],
                         label=name, lw=1.2)
    for ax, title in zip(axes, (r"$\max|v|/c_0$", r"$\max|\Delta y|/H$")):
        ax.axhline(payload["tolerance"], color="k", ls=":", lw=1.0,
                   label="anchor tolerance 1e-9")
        ax.axvline(payload["verdict"]["claimed_horizon_s"], color="r", ls="--",
                   lw=1.0, label="claimed horizon 0.50 s")
        ax.set_xlabel("t [s]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("departure from rest")
    axes[0].legend(fontsize=7)
    fig.suptitle("CASE-SPH-LAKE-AT-REST: the rest state is not persistent")
    fig.tight_layout()
    path = outdir / "lake_stability_departure.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    # (2) growth rate per arm versus dt, and the mode spectrum
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names, lams, dts = [], [], []
    for arm in payload["arms"]:
        if arm["growth"]["lambda_per_s"] is None:
            continue
        names.append(arm["arm"])
        lams.append(arm["growth"]["lambda_per_s"])
        dts.append(arm["run"]["dt"])
    axes[0].plot(dts, lams, "o")
    for n, d, lam in zip(names, dts, lams):
        axes[0].annotate(n, (d, lam), fontsize=7,
                         textcoords="offset points", xytext=(4, 4))
    axes[0].set_xlabel("dt [s]")
    axes[0].set_ylabel(r"growth rate $\lambda$ [1/s]")
    axes[0].set_title("is the departure dt-driven?")
    axes[0].grid(alpha=0.3)

    base = raw.get("base", [])
    if base:
        modes = [s["column_mode"] for s in base]
        amps = [s["column_amplitude"] for s in base]
        axes[1].plot([s["t"] for s in base], modes, label="dominant column mode")
        twin = axes[1].twinx()
        twin.semilogy([s["t"] for s in base], [max(a, 1e-20) for a in amps],
                      color="tab:orange", label="mode amplitude / H")
        axes[1].set_xlabel("t [s]")
        axes[1].set_ylabel("dominant wavenumber index")
        twin.set_ylabel(r"amplitude / $H$")
        axes[1].set_title("which mode grows (base arm)")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "lake_stability_growth.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))
    return written


if __name__ == "__main__":
    raise SystemExit(main())