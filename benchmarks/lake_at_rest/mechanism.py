"""What makes the still lake leave rest? -- mechanism identification.

Case-pool anchor ``CASE-SPH-LAKE-AT-REST`` asks for the analytic hydrostatic
identity at machine precision.  ``docs/verification_lake_at_rest.md`` records
the honest **FAIL**, and ``docs/verification_lake_at_rest_horizon.md`` turns the
verdict into a measured lifetime: the rest state holds ``max|v|/c0 < 1e-9`` up
to ``t* ~ 1.2 s`` and afterwards departs **exponentially** with
``lambda ~ 16.3 1/s`` (``R^2 = 0.99999``), into finite-amplitude sloshing.  That
study established *that* the state is linearly unstable and excluded the
integrator (``dt``), the seed, the frozen neighbour list, one lattice and water
viscosity as causes.  Its own disposition records what it did **not** do:

    "No attempt was made to identify the *mechanism* of the growing mode beyond
    the recorded dominant column wavenumber."

This runner is that attempt.  It is a *discrimination* study, not a re-run: the
question is which term of the discrete equilibrium owns the negative stiffness,
and the answer has to be a scaling law or a switch, not an opinion.

Measurements
------------
For every arm the departure series is recorded (``v_over_c0``, ``dy_over_H``,
``drho_over_rho0``), together with three things the horizon study did not keep:

* **localisation** -- ``|v|``-weighted height fraction of the fastest particle
  and the fraction of the kinetic energy held by the top three rows (the free
  surface), the bulk and the bottom three rows (the wall band);
* **mode content** -- the complex ``x``-Fourier coefficient of the column-mean
  vertical displacement at every sample, so both the spatial wavenumber *and*
  the temporal behaviour of one mode can be read off the same series;
* **temporal behaviour of that mode** -- its oscillation frequency ``omega``
  (counted from sign changes of the coefficient inside the exponential band)
  next to its growth rate ``lambda``.  A monotone clumping mode has
  ``omega = 0``; a gravity-wave branch has ``omega ~ sqrt(g k tanh(kH))`` and a
  small growth rate.

Arms (each changes *one* thing against ``base``):

======= ============================================================= ==========
family  arm                                                           decides
======= ============================================================= ==========
scale   ``g_quarter`` ``g_quad`` ``g_zero`` (gravity ``g/4``, ``4g``, 0) is the rate set by the
                                                                       gravity load (``lambda ~ g^p``)
scale   ``c0_half`` ``c0_double``                                     is the rate an acoustic/kernel
                                                                       stiffness (``lambda ~ c0^p``)
scale   ``h_lo`` (``h=1.15 dx``) ``h_hi`` (``h=1.60 dx``)             is it the classic kernel-pairing
                                                                       regime (strong ``h/dx`` dependence)
scale   ``depth_shallow`` (``H/3``)                                    is the rate set by the tank's
                                                                       gravity-wave branch
switch  ``sw_renorm`` ``sw_surfclos`` ``sw_bcd`` ``sw_delta``          does a shipped closure remove it
======= ============================================================= ==========

Verdict rules, declared before the numbers
------------------------------------------
Let ``p_x = d ln(lambda) / d ln(x)`` from the two-point arms of that family,
and ``sigma`` the ratio of a switch arm's rate to ``base``.

* ``|p_g| < 0.15`` and ``|p_c| < 0.15``  -> the rate is a property of the
  *kernel/lattice*, not of gravity or of the sound speed; the instability is a
  discretisation mode of the equilibrium.
* ``p_g ~ 0.5`` -> a gravity-wave branch (``omega ~ sqrt(g)``) is unstable.
* ``p_c ~ 1`` -> the rate is an acoustic scale (``c0/h``): numerical stiffness.
* ``sigma < 0.1`` for some switch -> that closure owns the mechanism (and the
  fix is a shipped knob, not new physics).

Usage (a sweep of ~15 arms; submit it, do not run it in the foreground)::

    PYTHONPATH=src python3 benchmarks/lake_at_rest/mechanism.py \\
        --json tests/data/lake_at_rest/mechanism.json --figures docs/figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

import band_fit
from tensorsph.droplet_impact import rhs
from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    periodic_pairs, reconstruct)

CASE_ID = "CASE-SPH-LAKE-AT-REST"
BAND_LOW = band_fit.BAND_LOW          # 1e-6: the declared exponential band
BAND_HIGH = band_fit.BAND_HIGH        # 1e-3
SURFACE_ROWS = 3                      # rows counted as "the free surface"
BOTTOM_ROWS = 3                       # rows counted as "the wall band"

#: one arm = one change against the reference protocol
ARMS = {
    # -- reference -------------------------------------------------------
    "base":            {},
    # -- gravity scaling -------------------------------------------------
    "g_quarter":       {"gravity": 9.81 / 4.0},
    "g_quad":          {"gravity": 9.81 * 4.0},
    "g_zero":          {"gravity": 0.0},
    # -- sound-speed scaling (dt follows 1/c0) ---------------------------
    "c0_half":         {"sound_speed": 10.0},
    "c0_double":       {"sound_speed": 40.0},
    # -- kernel support --------------------------------------------------
    "h_lo":            {"h_factor": 1.15},
    "h_hi":            {"h_factor": 1.60},
    # -- tank depth (gravity-wave branch) --------------------------------
    "depth_shallow":   {"depth": 0.10, "width": 0.40},
    # -- shipped closures as switches ------------------------------------
    "sw_renorm":       {"density_renormalization": True},
    "sw_surfclos":     {"surface_closure": True},
    "sw_bcd":          {"boundary_consistent_density": True},
    "sw_delta":        {"density_diffusion": 0.1},
}

FAMILY = {
    "base": "reference", "g_quarter": "gravity", "g_quad": "gravity",
    "g_zero": "gravity", "c0_half": "sound_speed", "c0_double": "sound_speed",
    "h_lo": "kernel", "h_hi": "kernel", "depth_shallow": "geometry",
    "sw_renorm": "switch", "sw_surfclos": "switch", "sw_bcd": "switch",
    "sw_delta": "switch",
}


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------
def _rows(pos: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Map every particle to its lattice row (``y`` level)."""
    ys, inverse = torch.unique(pos[:, 1], return_inverse=True)
    return inverse, int(ys.numel())


def _columns(pos: torch.Tensor) -> tuple[torch.Tensor, int]:
    xs, inverse = torch.unique(pos[:, 0], return_inverse=True)
    return inverse, int(xs.numel())


def integrate(cfg: LakeConfig, arm: dict) -> dict:
    """Advance the reconstructed lake and record the departure from rest.

    Stops early once the departure is three decades into the exponential band
    (nothing above it is new physics) or at ``cfg.t_end``, whichever comes
    first.  The protocol is the shipped one: frozen pair list, analytic prior
    for the reconstruction, leapfrog on ``tensorsph.droplet_impact.rhs``.
    """
    torch.set_default_dtype(cfg.torch_dtype)
    state = build_state(cfg)
    prior = analytic_profile(state, cfg)["mass"]
    solution = reconstruct(state, cfg, prior=prior)
    mass = solution["mass"]

    pos = state["pos"].clone()
    vel = torch.zeros_like(pos)
    dt = cfg.dt()
    steps = int(round(cfg.t_end / dt))
    y0 = state["pos"][:, 1].clone()
    rho_ref = rhs(state["pos"], torch.zeros_like(pos), mass, state["config"],
                  pairs=state["pairs"], wall=state["wall"],
                  wrap_width=state["wrap_width"])[0]
    rows, n_rows = _rows(state["pos"])
    columns, n_columns = _columns(state["pos"])
    count_row = torch.bincount(rows, minlength=n_rows).to(pos.dtype)
    count_column = torch.bincount(columns, minlength=n_columns).to(pos.dtype)
    total_mass = float(mass.sum())
    top = rows >= n_rows - SURFACE_ROWS
    bottom = rows < BOTTOM_ROWS
    height = float(y0.max() - y0.min())
    mode_orders = torch.arange(n_columns, dtype=pos.dtype)
    sample = int(arm.get("sample", 20))
    stop_above = 1.0e3 * BAND_HIGH          # 3 decades past the band top

    samples = []
    for step in range(steps):
        rho, _p, acc = rhs(pos, vel, mass, state["config"], pairs=state["pairs"],
                           wall=state["wall"], wrap_width=state["wrap_width"])
        vel = vel + dt * acc
        pos = pos + dt * vel
        v = float(vel.abs().max()) / cfg.sound_speed
        if (step + 1) % sample == 0 or step == steps - 1:
            speed = torch.linalg.vector_norm(vel, dim=1)
            ke_particle = 0.5 * mass * speed.square()
            ke_total = float(ke_particle.sum())
            dy = (pos[:, 1] - y0)
            column_mean = torch.zeros(n_columns, dtype=pos.dtype)
            column_mean = column_mean.index_add(0, columns, dy) / count_column
            # complex x-Fourier coefficient of the column-mean displacement
            coefficient = (torch.cos(-2.0 * math.pi * mode_orders
                                     / n_columns).to(pos.dtype) * column_mean).sum()
            row_mean = torch.zeros(n_rows, dtype=pos.dtype)
            row_mean = row_mean.index_add(0, rows, speed) / count_row
            fastest = int(rows[int(speed.argmax())].item())
            samples.append({
                "t": (step + 1) * dt,
                "v_over_c0": v,
                "dy_over_H": float(dy.abs().max()) / cfg.depth,
                "drho_over_rho0": float(((rho - rho_ref) / cfg.rho0).abs().max()),
                "ke_over_mc02": ke_total / (total_mass * cfg.sound_speed ** 2),
                "ke_surface": float(ke_particle[top].sum()) / max(ke_total, 1e-300),
                "ke_bottom": float(ke_particle[bottom].sum()) / max(ke_total, 1e-300),
                "fastest_row_height": (float(y0[rows == fastest].mean()) - float(y0.min()))
                                      / max(height, 1e-300),
                "fastest_row_top3": bool(fastest >= n_rows - SURFACE_ROWS),
                "column_mean": [float(x) for x in column_mean],
                "mode1_real": float(coefficient),
                "row_speed_mean": [float(x) for x in row_mean],
            })
            if v > stop_above:
                break

    return {
        "dt": dt, "steps": len(samples) * sample, "t_end": samples[-1]["t"],
        "rows": n_rows, "columns": n_columns,
        "particles": int(state["n_fluid"]),
        "equilibrium": {
            "a_max_over_g": solution["a_max_over_g"],
            "a_max_over_g_horizontal": solution["a_max_over_g_horizontal"],
            "iterations": solution["iterations"],
            "converged": solution["converged"],
        },
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------
def _fit(series: list[dict], field: str) -> dict:
    return band_fit.fit_lambda(series, BAND_LOW, BAND_HIGH, field=field)


def dispersion(cfg: LakeConfig, mode: int) -> float | None:
    """Inviscid gravity-wave frequency ``sqrt(g k tanh(kH))`` of one x-mode."""
    if mode <= 0 or cfg.gravity <= 0.0:
        return None
    k = 2.0 * math.pi * mode / cfg.width
    return math.sqrt(cfg.gravity * k * math.tanh(k * cfg.depth))


def mode_content(sample: dict, n_columns: int) -> dict:
    """x-spectrum of the column-mean displacement of one sample."""
    values = torch.tensor(sample["column_mean"], dtype=torch.float64)
    spectrum = torch.fft.rfft(values).abs() / max(n_columns, 1)
    modes = list(range(1, n_columns // 2 + 1))
    amplitudes = [float(spectrum[m]) for m in modes]
    best = max(modes, key=lambda m: amplitudes[m - 1])
    return {"dominant_mode": best,
            "dominant_amplitude": amplitudes[best - 1],
            "amplitudes": dict(zip(modes, amplitudes))}


def mode_frequency(series: list[dict], n_columns: int, mode: int) -> dict:
    """Oscillation frequency of one x-mode inside the exponential band.

    The coefficient stored per sample is the mode-1 real projection of the
    column-mean displacement (a fixed linear functional of the field), so sign
    changes of it inside the band count half periods of the *dominant*
    motion.  Reported as ``omega = pi * sign_changes / duration``; ``0`` means
    the departure is monotone (a clumping mode), not a wave.
    """
    window = [s for s in series
              if BAND_LOW <= s["v_over_c0"] <= BAND_HIGH]
    if len(window) < 6:
        return {"omega_per_s": None, "sign_changes": 0, "points": len(window)}
    values = [s["mode1_real"] for s in window]
    changes = sum(1 for a, b in zip(values, values[1:]) if a * b < 0.0)
    duration = window[-1]["t"] - window[0]["t"]
    if duration <= 0.0:
        return {"omega_per_s": None, "sign_changes": changes,
                "points": len(window)}
    return {"omega_per_s": math.pi * changes / duration,
            "sign_changes": changes, "points": len(window),
            "t_from": window[0]["t"], "t_to": window[-1]["t"]}


def arm_report(name: str, cfg: LakeConfig, run: dict) -> dict:
    series = run["samples"]
    growth = _fit(series, "v_over_c0")
    dy_growth = _fit(series, "dy_over_H")
    rho_growth = _fit(series, "drho_over_rho0")
    peak = max(series, key=lambda s: s["v_over_c0"])
    content = mode_content(peak, run["columns"])
    frequency = mode_frequency(series, run["columns"], content["dominant_mode"])
    # where the motion lives, measured at the last sample in the band
    in_band = [s for s in series if s["v_over_c0"] <= BAND_HIGH]
    late = in_band[-1] if in_band else series[-1]
    equilibrium = run["equilibrium"]
    # A switch that changes the density/pressure operator changes the
    # equilibrium too.  The reconstruction solves the mass field for the *same*
    # operator, so a converged arm starts at rest -- but where no such mass
    # field exists the arm starts far from rest and says nothing about
    # stability.  Such an arm is recorded and excluded, never averaged in.
    valid = bool(equilibrium["converged"]
                 and equilibrium["a_max_over_g"] < 1.0e-6
                 and equilibrium["a_max_over_g_horizontal"] < 1.0e-6)
    return {
        "arm": name,
        "family": FAMILY.get(name, "other"),
        "equilibrium_valid": valid,
        "config": {
            "dx": cfg.dx, "width": cfg.width, "depth": cfg.depth,
            "h": cfg.h(), "h_factor": cfg.h_factor,
            "sound_speed": cfg.sound_speed, "gravity": cfg.gravity,
            "max_courant": cfg.max_courant, "periodic_x": cfg.periodic_x,
            "density_renormalization": cfg.density_renormalization,
            "surface_pressure_closure": cfg.surface_closure,
            "boundary_consistent_density": cfg.boundary_consistent_density,
            "density_diffusion": cfg.density_diffusion,
        },
        "run": {k: v for k, v in run.items() if k != "samples"},
        "growth": growth,
        "growth_dy": dy_growth,
        "growth_drho": rho_growth,
        "entered_band": growth["points"] > 0,
        "max_v_over_c0": max(s["v_over_c0"] for s in series),
        "max_dy_over_H": max(s["dy_over_H"] for s in series),
        "peak_sample": {k: v for k, v in peak.items()
                        if k not in ("column_mean", "row_speed_mean")},
        "mode": content,
        "oscillation": frequency,
        "dispersion": {
            "omega_pred_per_s": dispersion(cfg, content["dominant_mode"]),
            "mode": content["dominant_mode"],
        },
        "localisation": {
            "ke_surface_fraction": late["ke_surface"],
            "ke_bottom_fraction": late["ke_bottom"],
            "fastest_row_height": late["fastest_row_height"],
            "fastest_row_top3_share": sum(
                1 for s in in_band if s["fastest_row_top3"]) / max(len(in_band), 1),
            "row_speed_mean": late["row_speed_mean"],
        },
        "samples": series,          # raw series: the gates re-derive from it
    }


def _loglog_slope(a: float, b: float, xa: float, xb: float) -> float | None:
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return None
    if xa <= 0.0 or xb <= 0.0 or xa == xb:
        return None
    return (math.log(b) - math.log(a)) / (math.log(xb) - math.log(xa))


def reductions(arms: dict) -> dict:
    """Scaling exponents and switch ratios, from the arms' own rates."""
    def lam(name):
        if not arms[name].get("equilibrium_valid", False):
            return None
        record = arms[name]["growth"]["lambda_per_s"]
        return record if record and record > 0.0 else None

    base = lam("base")
    out = {"lambda_base": base,
           "invalid_arms": {name: arm["run"]["equilibrium"]["a_max_over_g"]
                            for name, arm in arms.items()
                            if not arm.get("equilibrium_valid", False)},
           "switch_ratio": {}, "exponent": {}}
    for name, record in arms.items():
        if record["family"] == "switch":
            value = lam(name)
            out["switch_ratio"][name] = (None if value is None or base is None
                                         else value / base)
    if base is not None:
        # Two arms of a family give the exponent directly; when the sweep is a
        # reduced one, each family point is scored against the reference arm.
        families = {"gravity": ("g_quarter", "g_quad", "gravity"),
                    "sound_speed": ("c0_half", "c0_double", "sound_speed"),
                    "h": ("h_lo", "h_hi", "h")}
        for key, (lo, hi, field) in families.items():
            present = [name for name in (lo, hi) if name in arms]
            if len(present) == 2:
                out["exponent"][key] = _loglog_slope(
                    lam(lo), lam(hi), arms[lo]["config"][field],
                    arms[hi]["config"][field])
            elif present:
                other = present[0]
                out["exponent"][key] = _loglog_slope(
                    base, lam(other), arms["base"]["config"][field],
                    arms[other]["config"][field])
            else:
                out["exponent"][key] = None
        out["g_zero_rate"] = lam("g_zero") if "g_zero" in arms else None
        # which arm's rate is closest to the inviscid branch at its own mode
        out["dispersion_match"] = {
            name: (None if lam(name) is None or
                   record["dispersion"]["omega_pred_per_s"] in (None, 0.0)
                   else lam(name) / record["dispersion"]["omega_pred_per_s"])
            for name, record in arms.items()}
    return out


def verdict(red: dict) -> dict:
    """Apply the declared rules to the measured numbers (no free parameters)."""
    exponent = red.get("exponent", {})
    p_g, p_c, p_h = (exponent.get("gravity"), exponent.get("sound_speed"),
                     exponent.get("h"))
    ratios = red.get("switch_ratio", {})
    best_switch = None
    if ratios and all(v is not None for v in ratios.values()):
        best_switch = min(ratios, key=lambda name: ratios[name])
    cleared = [name for name, value in ratios.items()
               if value is not None and value < 0.1]
    governed_by_gravity = p_g is not None and abs(p_g - 0.5) <= 0.15
    governed_by_acoustics = p_c is not None and abs(p_c - 1.0) <= 0.25
    lattice_mode = (p_g is not None and abs(p_g) < 0.15
                    and p_c is not None and abs(p_c) < 0.15)
    return {
        "p_gravity": p_g, "p_sound_speed": p_c, "p_h": p_h,
        "gravity_wave_branch": governed_by_gravity,
        "acoustic_stiffness": governed_by_acoustics,
        "kernel_lattice_mode": lattice_mode,
        "switch_that_clears": cleared,
        "most_effective_switch": best_switch,
        "switch_ratio_best": (None if best_switch is None
                              else ratios[best_switch]),
        "statement": _statement(governed_by_gravity, governed_by_acoustics,
                                lattice_mode, cleared, ratios, p_g, p_c, p_h,
                                red.get("invalid_arms")),
        "excluded_arms": sorted(red.get("invalid_arms") or {}),
    }


def _statement(gravity_branch, acoustic, lattice, cleared, ratios, p_g, p_c, p_h,
               invalid=None):
    def f(x):
        return "n/a" if x is None else f"{x:+.2f}"

    parts = [f"d ln(lambda)/d ln(g) = {f(p_g)}",
             f"d ln(lambda)/d ln(c0) = {f(p_c)}",
             f"d ln(lambda)/d ln(h) = {f(p_h)}"]
    if gravity_branch:
        parts.append("the unstable branch scales like a gravity wave")
    if acoustic:
        parts.append("the rate is an acoustic scale (lambda ~ c0/h)")
    if lattice:
        parts.append("the rate is a property of the kernel/lattice, not of g or c0")
    if cleared:
        parts.append("cleared by the shipped switch(es) " + ", ".join(sorted(cleared)))
    elif ratios:
        ranked = [(value, name) for name, value in ratios.items()
                  if value is not None]
        if ranked:
            best = min(ranked)
            parts.append(f"no switch clears the growth (best: {best[1]} at "
                         f"{best[0]:.2f} x base)")
        else:
            parts.append("no switch arm produced a rate inside the band, so no "
                         "switch ratio is claimed")
    if invalid:
        parts.append("excluded (the reconstruction did not reach an equilibrium, so the "
                     "arm does not test stability): "
                     + ", ".join(sorted(invalid)))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_config(overrides: dict, base: dict | None = None) -> LakeConfig:
    settings = dict(base or {})
    settings.update(overrides)
    return LakeConfig(**settings)


def run_arm(name: str, overrides: dict, base: dict, verbose: bool = True) -> dict:
    cfg = build_config(overrides, base)
    started = time.time()
    run = integrate(cfg, overrides)
    report = arm_report(name, cfg, run)
    report["runtime_s"] = time.time() - started
    if verbose:
        growth = report["growth"]
        rate = growth["lambda_per_s"]
        r2 = growth["r2_exp"]
        shown_rate = "n/a" if rate is None else format(rate, "8.3f")
        shown_r2 = "n/a" if r2 is None else format(r2, ".4f")
        print(f"  {name:16s} lambda={shown_rate}  R2={shown_r2}"
              f"  max|v|/c0={report['max_v_over_c0']:.3e}"
              f"  mode={report['mode']['dominant_mode']:2d}"
              f"  {report['runtime_s']:.0f}s", flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", default=None,
                        help="subset of arms (default: every arm)")
    parser.add_argument("--t-end", type=float, default=2.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dx", type=float, default=0.020)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    base = {"t_end": args.t_end, "periodic_x": True, "dx": args.dx}
    names = args.arms or list(ARMS)
    started = time.time()
    records = {}
    for name in names:
        records[name] = run_arm(name, ARMS[name], base)
    payload = {
        "case": CASE_ID,
        "kind": "mechanism identification of the lake-at-rest departure",
        "band": [BAND_LOW, BAND_HIGH],
        "protocol": {"dx": args.dx, "t_end_cap": args.t_end,
                     "periodic_x": True, "threads": args.threads},
        "torch": torch.__version__,
        "runtime_s": time.time() - started,
        "arms": records,
    }
    if set(names) == set(ARMS):
        payload["reductions"] = reductions(records)
        payload["verdict"] = verdict(payload["reductions"])
        print("\n" + payload["verdict"]["statement"])
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {args.json}  ({payload['runtime_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())