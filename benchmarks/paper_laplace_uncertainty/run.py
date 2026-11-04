"""Results-grade material for ``paper-draft-tensorsph-kernel-verification``.

Two kinds of material are produced, both machine-generated from one program:

(a) **Independent baseline / comparison table.**  Every quantity is scored
    point-by-point against a closed-form baseline, and the numerator and the
    denominator of every error are written out explicitly:

      * Young-Laplace pressure jump        Delta p_meas  vs  sigma/R
      * CSF curvature, three estimators    kappa_i       vs  1/R
        (MLS ``div n``, quadratic interface fit, algebraic circle fit -- the
        last two are *independent implementations* of the same geometric
        quantity, so the columns also serve as an implementation A/B)
      * cut-force identity                 F_cut         vs  -2 sigma
      * free-surface pressure closure      p_shell       vs  0

    Each row additionally carries a 95 % confidence interval over an ensemble of
    statistically equivalent lattice realisations (the hexagonal lattice is
    rotated/shifted, which is the dominant source of anisotropy noise on a
    static droplet) and an explicit three-state verdict PASS / WITHHELD / FAIL.

(b) **Uncertainty and field contours.**  The verdict rows above are the error
    bars; the figures written alongside are the field contours (pressure and
    curvature-error clouds at two resolutions) plus the radial profile and the
    time history of the measured jump.

Everything is written to ``benchmarks/paper_laplace_uncertainty/out/``:

    runs/*.json      one file per (resolution, realisation)  [raw, resumable]
    results.json     aggregated table + verdicts + provenance
    tables.md        the same table rendered from results.json (no hand copy)
    fig1_pressure_field.png / fig2_curvature_field.png
    fig3_uncertainty.png / fig4_baseline_compare.png

Usage::

    PYTHONPATH=src python benchmarks/paper_laplace_uncertainty/run.py \
        --dx-list 0.003 0.0025 0.002 0.0015 0.001 --realisations 3
    PYTHONPATH=src python benchmarks/paper_laplace_uncertainty/run.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.csf_curvature import fit_curvature
from tensorsph.droplet_impact import (
    Config,
    density_from_neighbor_pairs,
    rhs,
    stable_dt,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    compute_color_gradient,
    compute_curvature,
    compute_normal,
)

torch.set_num_threads(2)

BENCH = Path(__file__).resolve().parent
OUT = BENCH / "out"
RUNS = OUT / "runs"

# Platform accuracy gates (AGENTS.md / docs/PROJECT_CHARTER.md).
GATE_3 = 3.0     # "is it < 3 % ?" -- the question the paper must answer
GATE_15 = 15.0   # hard platform tolerance

# Student-t 0.975 quantile, used for the 95 % CI of the ensemble mean.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

# t / t_sigma at which the pressure jump is sampled (dynamics, not statics).
SAMPLE_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


# --------------------------------------------------------------------------- #
# geometry: hexagonal droplet + ensemble realisation
# --------------------------------------------------------------------------- #
def hexagonal_droplet(radius, dx, rho0, shift=(0.0, 0.0), jitter=0.0, seed=0):
    """Hexagonally packed circular drop, translated by ``shift`` before clipping.

    ``mass * number_density = rho0`` is enforced exactly, so no density bias is
    introduced by the lattice.  The circle stays centred on the origin while the
    lattice is translated, therefore different ``shift`` values are genuinely
    different discretisations of the *same* continuum problem (rotating the
    lattice instead would be an exact symmetry of an axisymmetric drop and
    reproduces the same numbers bit for bit -- checked, and deliberately not
    used as an ensemble axis).  ``jitter`` adds a seeded Gaussian perturbation of
    the particle positions, which is the other standard way of asking "how much
    of this error is lattice placement?".
    """
    dy = dx * math.sqrt(3.0) / 2.0
    mass = rho0 * dx * dy
    pts = []
    j = 0
    y = -radius - dy
    while y <= radius + dy:
        offset = 0.5 * dx if (j % 2) else 0.0
        x = -radius - dx + offset
        while x <= radius + dx:
            pts.append((x + offset + shift[0], y + shift[1]))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    pos = pos[torch.linalg.vector_norm(pos, dim=1) <= radius]
    if jitter > 0.0:
        g = torch.Generator().manual_seed(seed)
        pos = pos + torch.randn(pos.shape, generator=g, dtype=torch.float64) * jitter
    pos = pos.contiguous()
    return pos, torch.full((pos.shape[0],), float(mass), dtype=torch.float64)


def realisation(index, dx, radius, rho0):
    """Deterministic, reproducible ensemble member.

    Lattice phase offsets are the primary axis (they change which particles sit
    on the rim, i.e. the actual discretisation of the interface); one member
    adds a 2 %-of-dx positional jitter as an independent realisation.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    phases = [(0.0, 0.0), (0.5 * dx, 0.0), (0.0, 0.5 * dy),
              (0.5 * dx, 0.5 * dy), (0.25 * dx, 0.75 * dy)]
    shift = phases[index % len(phases)]
    jitter = 0.0
    seed = 0
    if index >= len(phases):
        jitter, seed = 0.02 * dx, 1234 + index
    return dict(shift=shift, jitter=jitter, seed=seed,
                label=("shift%.2f_%.2f" % (shift[0] / dx, shift[1] / dy)).replace(".", "p")
                + ("_jit" if jitter else ""))


# --------------------------------------------------------------------------- #
# CSF ingredient fields (same production path as benchmarks/laplace_law)
# --------------------------------------------------------------------------- #
def csf_fields(pos, rho, mass, pairs, cfg):
    h = cfg.h_factor * cfg.dx
    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    count = torch.zeros(n, dtype=pos.dtype)
    one = torch.ones_like(i, dtype=pos.dtype)
    count = count.index_add(0, i, one).index_add(0, j, one)
    avg = count.mean()
    is_surface = count < 0.7 * avg
    color = torch.ones(n, dtype=pos.dtype)
    color[is_surface] = count[is_surface] / avg
    grad_c = compute_color_gradient(pos, color, mass, rho, pairs, h)
    normal = compute_normal(-grad_c)
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)
    return normal, delta_s, is_surface


def curvature_estimators(pos, rho, mass, pairs, cfg, radius):
    """Three curvatures for the same interface, all scored against 1/R.

    ``mls_div n``      -- production CSF definition (surface_tension.py)
    ``fit_quadratic``  -- independent implementation (csf_curvature.py)
    ``fit_circle``     -- second independent implementation (Pratt/Kasa fit)
    """
    h = cfg.h_factor * cfg.dx
    normal, delta_s, _ = csf_fields(pos, rho, mass, pairs, cfg)
    kappa_mls = compute_curvature(pos, normal, mass, rho, pairs, h)
    # the fit family needs a wider stencil than the kernel support
    pairs_fit = cell_linked_pairs(pos, 4.0 * h)
    kappa_quad = fit_curvature(pos, normal, delta_s, pairs_fit, h,
                               fallback=kappa_mls, method="fit_quadratic")
    kappa_circ = fit_curvature(pos, normal, delta_s, pairs_fit, h,
                               fallback=kappa_mls, method="fit_circle")

    r_norm = torch.linalg.vector_norm(pos, dim=1) / radius
    band = (r_norm >= 0.6) & (r_norm <= 0.98)
    w = delta_s.abs()[band]
    w = w / w.sum().clamp_min(1e-30)
    exact = 1.0 / radius
    out = {}
    for name, kap in (("mls_divn", kappa_mls), ("fit_quadratic", kappa_quad),
                      ("fit_circle", kappa_circ)):
        val = float((kap.abs()[band] * w).sum())
        out[name] = {
            "kappa_measured_1_per_m": val,
            "kappa_exact_1_per_m": exact,
            "numerator_1_per_m": val - exact,
            "denominator_1_per_m": exact,
            "rel_error_pct": (val - exact) / exact * 100.0,
        }
    return out


# --------------------------------------------------------------------------- #
# dynamic relaxation to t = 1 t_sigma, sampled on the way
# --------------------------------------------------------------------------- #
def relax_run(radius, dx, sigma, rho0, sound_speed, viscosity, rea):
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=viscosity,
                 surface_tension_coefficient=sigma)
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    t_sigma = math.sqrt(rho0 * radius ** 3 / sigma)
    t_end = SAMPLE_FRACTIONS[-1] * t_sigma
    n_steps = max(8, int(math.ceil(t_end / dt)))

    pos, mass = hexagonal_droplet(radius, dx, rho0, rea["shift"],
                                  rea["jitter"], rea["seed"])
    n = pos.shape[0]
    vel = torch.zeros_like(pos)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)

    targets = {round(f, 4): f * t_sigma for f in SAMPLE_FRACTIONS}
    samples = {}
    snap = None
    dp_theory = sigma / radius
    for step in range(n_steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None,
                                 dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        t = (step + 1) * dt
        due = [f for f, tt in targets.items() if f not in samples and t >= tt]
        if due or step == n_steps - 1:
            r = torch.linalg.vector_norm(pos, dim=1)
            p_core = float(pressure[r < 0.5 * radius].mean())
            band = (r > 0.8 * radius) & (r < 0.95 * radius)
            p_band = float(pressure[band].mean())
            shell = (r > 0.95 * radius) & (r < 1.05 * radius)
            p_shell = float(pressure[shell].mean()) if bool(shell.any()) else float("nan")
            dp = p_core - p_band
            for f in due:
                samples[f] = {
                    "t_over_t_sigma": t / t_sigma,
                    "p_core_Pa": p_core,
                    "p_band_Pa": p_band,
                    "p_shell_Pa": p_shell,
                    "delta_p_measured_Pa": dp,
                    "delta_p_theory_Pa": dp_theory,
                    "numerator_Pa": dp - dp_theory,
                    "denominator_Pa": dp_theory,
                    "rel_error_pct": (dp - dp_theory) / dp_theory * 100.0,
                    "vmax_m_per_s": float(torch.linalg.vector_norm(vel, dim=1).max()),
                }
            if step == n_steps - 1:
                normal, delta_s, is_surface = csf_fields(pos, rho, mass, pairs, cfg)
                snap = {
                    "pos": pos.clone(), "pressure": pressure.clone(),
                    "rho": rho.clone(), "normal": normal.clone(),
                    "delta_s": delta_s.clone(),
                }

    # ---- static scores on the relaxed state ------------------------------- #
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho, pressure, _ = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=dt)
    normal, delta_s, _ = csf_fields(pos, rho, mass, pairs, cfg)
    from tensorsph.surface_tension import compute_surface_tension_force
    acc_st = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    right = pos[:, 0] > 0.0
    f_cut = float((mass[right, None] * acc_st[right]).sum(dim=0)[0])
    curv = curvature_estimators(pos, rho, mass, pairs, cfg, radius)

    r_norm = torch.linalg.vector_norm(pos, dim=1) / radius
    prof_edges = torch.linspace(0.0, 1.1, 23, dtype=pos.dtype)
    profile = []
    for k in range(len(prof_edges) - 1):
        m = (r_norm >= prof_edges[k]) & (r_norm < prof_edges[k + 1])
        if int(m.sum()) > 0:
            profile.append({
                "r_over_R": float(0.5 * (prof_edges[k] + prof_edges[k + 1])),
                "p_mean_Pa": float(pressure[m].mean()),
                "p_std_Pa": float(pressure[m].std()) if int(m.sum()) > 1 else 0.0,
                "n": int(m.sum()),
            })

    # curvature-error field, for the contour figure (delta-s weighted fit)
    pairs_fit = cell_linked_pairs(pos, 4.0 * h)
    kappa_fit = fit_curvature(pos, normal, delta_s, pairs_fit, h,
                              fallback=compute_curvature(pos, normal, mass, rho,
                                                         pairs, h),
                              method="fit_quadratic")
    shell = (r_norm > 0.95) & (r_norm < 1.05)
    report = {
        "realisation": rea["label"],
        "shift_m": list(rea["shift"]),
        "jitter_m": rea["jitter"],
        "setup": {
            "radius_m": radius, "dx_m": dx, "h_m": h, "dt_s": dt,
            "n_particles": n, "sigma_N_per_m": sigma, "rho0": rho0,
            "sound_speed_m_per_s": sound_speed, "viscosity_m2_per_s": viscosity,
            "steps": n_steps, "physical_time_s": n_steps * dt,
            "t_over_t_sigma": n_steps * dt / t_sigma,
            "R_over_dx": radius / dx, "lattice": "hexagonal", "gravity": 0.0,
        },
        "reference": {
            "delta_p_Pa": dp_theory,
            "curvature_1_per_m": 1.0 / radius,
            "cut_force_N_per_m": -2.0 * sigma,
            "capillary_time_s": t_sigma,
            "capillary_velocity_m_per_s": math.sqrt(sigma / (rho0 * radius)),
        },
        "samples": {str(k): v for k, v in sorted(samples.items())},
        "final": {
            "p_shell_Pa": float(pressure[shell].mean()) if bool(shell.any())
            else float("nan"),
            "free_surface_closure_rel_pct": (
                float(pressure[shell].mean()) / dp_theory * 100.0
                if bool(shell.any()) else float("nan")),
            "cut_force_measured_N_per_m": f_cut,
            "cut_force_theory_N_per_m": -2.0 * sigma,
            "cut_force_numerator_N_per_m": f_cut + 2.0 * sigma,
            "cut_force_denominator_N_per_m": -2.0 * sigma,
            "cut_force_rel_error_pct": (f_cut + 2.0 * sigma) / (-2.0 * sigma) * 100.0,
            "curvature": curv,
            "rho_core_over_rho0": float(rho[r_norm < 0.5].mean() / rho0),
        },
        "pressure_profile": profile,
    }
    field = {
        "pos": snap["pos"].numpy().tolist(),
        "pressure": snap["pressure"].numpy().tolist(),
        "kappa_error": (kappa_fit.abs() - 1.0 / radius).numpy().tolist(),
    }
    return report, field


# --------------------------------------------------------------------------- #
# aggregation: 95 % CI + three-state verdict
# --------------------------------------------------------------------------- #
def ci95(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    half = T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    return mean, sd, half


def verdict(err_pct, ci_half):
    """Three-state decision, stated once so the paper cannot fudge it.

    PASS      |err| + ci < 3 %        -- inside the paper's 3 % question
    WITHHELD  3 % <= |err| < 15 %, or the CI straddles a gate
    FAIL      |err| >= 15 %           -- hard platform tolerance breached
    """
    a = abs(err_pct)
    if not math.isfinite(a):
        return "WITHHELD"
    if a < GATE_3 and (not math.isfinite(ci_half) or a + ci_half < GATE_3):
        return "PASS"
    if a >= GATE_15:
        return "FAIL"
    return "WITHHELD"


METRIC_KEYS = ("band", "core_to_shell", "force_implied")
METRIC_DOC = {
    "band": "Delta p = mean(p, r<0.5R) - mean(p, 0.8R<r<0.95R)",
    "core_to_shell": "Delta p = mean(p, r<0.5R) - mean(p, 0.95R<r<1.05R) "
                     "(analytic p_out = 0)",
    "force_implied": "Delta p = |F_cut| / (2R), F_cut = sum_{x>0} m a_st,x "
                     "(analytic |F_cut| = 2 sigma)",
}


def per_realisation_metric(run, key):
    """One number per realisation, all three from the same final state."""
    radius = run["setup"]["radius_m"]
    sfin = run["samples"][str(SAMPLE_FRACTIONS[-1])]
    if key == "band":
        return sfin["delta_p_measured_Pa"]
    if key == "core_to_shell":
        return sfin["p_core_Pa"] - sfin["p_shell_Pa"]
    if key == "force_implied":
        return abs(run["final"]["cut_force_measured_N_per_m"]) / (2.0 * radius)
    raise KeyError(key)


def metric_block(values, theory):
    m, sd, ci = ci95(values)
    errs = [(v - theory) / theory * 100.0 for v in values]
    _, _, err_ci = ci95(errs)
    num = m - theory
    return {
        "delta_p_theory_Pa": theory,
        "delta_p_measured_mean_Pa": m,
        "delta_p_measured_sd_Pa": sd,
        "delta_p_ci95_Pa": ci,
        "numerator_Pa": num,
        "denominator_Pa": theory,
        "rel_error_pct": num / theory * 100.0,
        "rel_error_ci95_pct": err_ci,
        "rel_error_min_pct": min(errs), "rel_error_max_pct": max(errs),
        "verdict_3pct": verdict(num / theory * 100.0, err_ci),
        "within_3pct": bool(abs(num / theory * 100.0) < GATE_3
                            and abs(num / theory * 100.0) + err_ci < GATE_3),
        "within_15pct": bool(abs(num / theory * 100.0) < GATE_15),
    }


def aggregate():
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("run_*.json"))]
    if not runs:
        raise SystemExit("no runs found -- run the sweep first")
    fields = {p.stem.replace("field_", ""): json.loads(p.read_text())
              for p in sorted((OUT / "fields").glob("field_*.json"))}
    by_dx = {}
    for r in runs:
        by_dx.setdefault(r["setup"]["dx_m"], []).append(r)

    rows = []
    for dx in sorted(by_dx, reverse=True):
        group = by_dx[dx]
        tkey = str(SAMPLE_FRACTIONS[-1])
        pth = group[0]["reference"]["delta_p_Pa"]
        metrics = {k: metric_block([per_realisation_metric(r, k) for r in group], pth)
                   for k in METRIC_KEYS}
        band = metrics["band"]
        row = {
            "dx_m": dx,
            "R_over_dx": group[0]["setup"]["R_over_dx"],
            "n_particles": group[0]["setup"]["n_particles"],
            "n_realisations": len(group),
            "n_particles_range": [min(r["setup"]["n_particles"] for r in group),
                                  max(r["setup"]["n_particles"] for r in group)],
            "delta_p_theory_Pa": pth,
            "metrics": metrics,
            # primary (band) metric mirrored at top level for the tables/tests
            "delta_p_measured_mean_Pa": band["delta_p_measured_mean_Pa"],
            "delta_p_measured_sd_Pa": band["delta_p_measured_sd_Pa"],
            "delta_p_ci95_Pa": band["delta_p_ci95_Pa"],
            "numerator_Pa": band["numerator_Pa"],
            "denominator_Pa": band["denominator_Pa"],
            "rel_error_pct": band["rel_error_pct"],
            "rel_error_ci95_pct": band["rel_error_ci95_pct"],
            "rel_error_min_pct": band["rel_error_min_pct"],
            "rel_error_max_pct": band["rel_error_max_pct"],
            "verdict_3pct": band["verdict_3pct"],
            "within_3pct": band["within_3pct"],
            "within_15pct": band["within_15pct"],
            "spurious_current_ratio": max(
                r["samples"][tkey]["vmax_m_per_s"]
                / r["reference"]["capillary_velocity_m_per_s"] for r in group),
            "p_shell_over_dp_theory_pct": sum(
                r["final"]["free_surface_closure_rel_pct"] for r in group) / len(group),
            "curvature": {},
            "cut_force_rel_error_pct": sum(
                r["final"]["cut_force_rel_error_pct"] for r in group) / len(group),
            "rho_core_over_rho0": sum(
                r["final"]["rho_core_over_rho0"] for r in group) / len(group),
            "time_history": {},
            "wall_time_s": sum(r.get("wall_time_s", 0.0) for r in group),
        }
        for est in ("mls_divn", "fit_quadratic", "fit_circle"):
            vals = [r["final"]["curvature"][est]["rel_error_pct"] for r in group]
            m, s, c = ci95(vals)
            row["curvature"][est] = {
                "kappa_measured_mean_1_per_m": sum(
                    r["final"]["curvature"][est]["kappa_measured_1_per_m"]
                    for r in group) / len(group),
                "kappa_exact_1_per_m": 1.0 / group[0]["setup"]["radius_m"],
                "rel_error_pct": m, "sd_pct": s, "ci95_pct": c,
                "verdict_3pct": verdict(m, c),
            }
        for f in SAMPLE_FRACTIONS:
            vals = [r["samples"][str(f)]["rel_error_pct"] for r in group]
            m, s, c = ci95(vals)
            row["time_history"][str(f)] = {
                "rel_error_pct": m, "sd_pct": s, "ci95_pct": c,
            }
        rows.append(row)

    results = {
        "provenance": {
            "repo": "TensorSPH",
            "script": "benchmarks/paper_laplace_uncertainty/run.py",
            "runs": len(runs),
            "gate_3pct": GATE_3, "gate_15pct": GATE_15,
            "verdict_rule": "PASS: |e|+CI<3%; FAIL: |e|>=15%; else WITHHELD",
            "metrics": METRIC_DOC,
            "primary_metric": "band",
            "theory": "Delta p = sigma/R (2-D, one principal curvature)",
            "ensemble": ("hexagonal lattice phase offsets (0,0)/(0.5dx,0)/"
                         "(0,0.5dy)/(0.5dx,0.5dy)/(0.25dx,0.75dy) with the drop "
                         "centre fixed; one member adds 2%dx seeded jitter"),
            "sampling_time": "t = 1.0 t_sigma",
        },
        "rows": rows,
        "fields": fields,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    (OUT / "tables.md").write_text(render_tables(results))
    make_figures(results)
    return results


def _fmt(x, nd=2):
    """NaN is not a number in a paper table -- print it as n/a."""
    return "n/a" if not (isinstance(x, float) and math.isfinite(x)) else f"{x:.{nd}f}"


def _row_metric_table(res, key, title, note):
    out = [f"## {title}", "", note, "",
           "| dx [mm] | R/dx | N | n | measured [Pa] | 95% CI [Pa] | theory [Pa] "
           "| numerator [Pa] | denominator [Pa] | rel. err [%] | 95% CI [%] "
           "| verdict | <3%? |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        m = r["metrics"][key]
        out.append(
            f"| {r['dx_m']*1e3:.2f} | {r['R_over_dx']:.1f} | {r['n_particles']} | "
            f"{r['n_realisations']} | {_fmt(m['delta_p_measured_mean_Pa'])} | "
            f"{_fmt(m['delta_p_ci95_Pa'])} | {m['delta_p_theory_Pa']:.2f} | "
            f"{_fmt(m['numerator_Pa'])} | {m['denominator_Pa']:.2f} | "
            f"{_fmt(m['rel_error_pct'])} | {_fmt(m['rel_error_ci95_pct'])} | "
            f"{m['verdict_3pct']} | {'yes' if m['within_3pct'] else 'no'} |")
    out.append("")
    return out


def render_tables(res):
    L = []
    A = L.append
    A("# Laplace-law verification: machine-rendered tables")
    A("")
    A(f"Runs aggregated: {res['provenance']['runs']}. Theory: "
      f"{res['provenance']['theory']}. Ensemble: {res['provenance']['ensemble']}. "
      f"Sampling: {res['provenance']['sampling_time']}. "
      f"Verdict rule: {res['provenance']['verdict_rule']}.")
    A("")
    for key, title, note in (
        ("band", "Table 1a -- pressure jump, metric A (core vs interface band)",
         "Metric A: " + METRIC_DOC["band"] + ". Used as the primary metric because "
         "it samples the pressure *at the interface*, which is where Young-Laplace "
         "is asserted."),
        ("core_to_shell", "Table 1b -- pressure jump, metric B (core vs free surface)",
         "Metric B: " + METRIC_DOC["core_to_shell"] + ". This is the metric a "
         "surface-tension test would like to use, but it is only valid if the "
         "interior pressure is uniform -- see Table 3 for the gradient that "
         "invalidates it."),
        ("force_implied", "Table 1c -- pressure jump, metric C (force-implied)",
         "Metric C: " + METRIC_DOC["force_implied"] + ". Independent of any "
         "pressure measurement, so it isolates the force amplitude from the "
         "pressure transport."),
    ):
        L.extend(_row_metric_table(res, key, title, note))

    A("## Table 2 -- CSF curvature, three estimators vs analytic 1/R")
    A("")
    A("| dx [mm] | R/dx | kappa exact [1/m] | MLS div n mean | rel. err [%] (95% CI) | verdict "
      "| quadratic fit mean | rel. err [%] (95% CI) | verdict "
      "| circle fit mean | rel. err [%] (95% CI) | verdict |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        c = r["curvature"]
        def cell(e):
            return (f"{_fmt(c[e]['kappa_measured_mean_1_per_m'])} | "
                    f"{_fmt(c[e]['rel_error_pct'])} ({_fmt(c[e]['ci95_pct'])}) | "
                    f"{c[e]['verdict_3pct']}")
        A(f"| {r['dx_m']*1e3:.2f} | {r['R_over_dx']:.1f} | "
          f"{c['mls_divn']['kappa_exact_1_per_m']:.2f} | {cell('mls_divn')} | "
          f"{cell('fit_quadratic')} | {cell('fit_circle')} |")
    A("")
    A("## Table 3 -- independent checks (force magnitude, free surface, spurious current)")
    A("")
    A("| dx [mm] | R/dx | F_cut vs -2 sigma [%] | p_shell / (sigma/R) [%] "
      "| max|v| / sqrt(sigma/(rho R)) | rho_core/rho0 |")
    A("|---|---|---|---|---|---|")
    for r in res["rows"]:
        A(f"| {r['dx_m']*1e3:.2f} | {r['R_over_dx']:.1f} | {_fmt(r['cut_force_rel_error_pct'])} "
          f"| {_fmt(r['p_shell_over_dp_theory_pct'])} | {r['spurious_current_ratio']:.3f} "
          f"| {r['rho_core_over_rho0']:.4f} |")
    A("")
    A("## Table 4 -- time dependence of the measured jump (metric A, t/t_sigma)")
    A("")
    A("| dx [mm] | " + " | ".join(f"t={f}" for f in SAMPLE_FRACTIONS) + " |")
    A("|---|" + "---|" * len(SAMPLE_FRACTIONS))
    for r in res["rows"]:
        cells = " | ".join(f"{_fmt(r['time_history'][str(f)]['rel_error_pct'])} "
                           f"± {_fmt(r['time_history'][str(f)]['ci95_pct'])}"
                           for f in SAMPLE_FRACTIONS)
        A(f"| {r['dx_m']*1e3:.2f} | {cells} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.tri as mtri
    import matplotlib.pyplot as plt
    import numpy as np

    OUT.mkdir(parents=True, exist_ok=True)
    rows = res["rows"]
    fields = res["fields"]
    keys = sorted(fields, key=float)

    # ---- fig1/fig2: field contours --------------------------------------- #
    def contour(kind, fname, title, cmap, unit):
        pick = keys[:1] + keys[-1:] if len(keys) > 1 else keys
        fig, axes = plt.subplots(1, len(pick), figsize=(5.4 * len(pick), 4.6),
                                 squeeze=False)
        for ax, key in zip(axes[0], pick):
            f = fields[key]
            x = np.asarray(f["pos"])[:, 0] * 1e3
            y = np.asarray(f["pos"])[:, 1] * 1e3
            v = np.asarray(f[kind])
            tri = mtri.Triangulation(x, y)
            cf = ax.tricontourf(tri, v, levels=18, cmap=cmap)
            ax.tricontour(tri, v, levels=8, colors="k", linewidths=0.25, alpha=0.5)
            fig.colorbar(cf, ax=ax, label=unit)
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_title(f"{title}\ndx = {key} mm, t = 1.0 t_sigma", fontsize=9)
        fig.tight_layout()
        path = OUT / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("figure ->", path)

    contour("pressure", "fig1_pressure_field.png",
            "SPH pressure field in a static drop", "viridis", "p [Pa]")
    contour("kappa_error", "fig2_curvature_field.png",
            "Curvature error field  |kappa| - 1/R (quadratic fit)", "coolwarm",
            "|kappa| - 1/R [1/m]")

    # ---- fig3: uncertainty vs resolution + gates -------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    x = np.array([r["R_over_dx"] for r in rows])
    styles = {"band": ("o-", "A: core vs interface band"),
              "core_to_shell": ("s-", "B: core vs free surface"),
              "force_implied": ("^-", "C: force-implied")}
    for key, (mk, lab) in styles.items():
        ee = np.array([r["metrics"][key]["rel_error_pct"] for r in rows])
        cc = np.array([r["metrics"][key]["rel_error_ci95_pct"] for r in rows])
        axes[0].errorbar(x, ee, yerr=np.nan_to_num(cc), fmt=mk, capsize=3, label=lab)
    axes[0].axhline(0, color="k", ls="--")
    for g, s in ((GATE_3, ":"), (GATE_15, "-.")):
        axes[0].axhline(g, color="r", ls=s, label=f"gate {g:.0f}%")
        axes[0].axhline(-g, color="r", ls=s)
    axes[0].set_xlabel("R / dx")
    axes[0].set_ylabel("relative error of Delta p [%]")
    axes[0].set_title("(a) Young-Laplace error vs resolution (95% CI)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for est, mk in (("mls_divn", "o-"), ("fit_quadratic", "s-"), ("fit_circle", "^-")):
        ke = np.array([r["curvature"][est]["rel_error_pct"] for r in rows])
        kc = np.array([r["curvature"][est]["ci95_pct"] for r in rows])
        axes[1].errorbar(x, ke, yerr=np.nan_to_num(kc), fmt=mk, capsize=3, label=est)
    axes[1].axhline(0, color="k", ls="--")
    axes[1].axhline(GATE_15, color="r", ls="-.")
    axes[1].axhline(-GATE_15, color="r", ls="-.")
    axes[1].axhline(GATE_3, color="r", ls=":")
    axes[1].axhline(-GATE_3, color="r", ls=":")
    axes[1].set_xlabel("R / dx")
    axes[1].set_ylabel("relative error of kappa [%]")
    axes[1].set_title("(b) curvature estimators vs 1/R")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    v = np.array([r["spurious_current_ratio"] for r in rows])
    fc = np.array([r["cut_force_rel_error_pct"] for r in rows])
    axes[2].semilogy(x, np.maximum(v, 1e-12), "o-", label="max|v|/sqrt(sigma/rhoR)")
    axes[2].axhline(0.1, color="r", ls="--", label="Adami 2012 gate 0.1")
    axes[2].set_xlabel("R / dx")
    axes[2].set_ylabel("spurious current")
    ax2 = axes[2].twinx()
    ax2.plot(x, fc, "s--", color="tab:orange", label="cut-force rel. err [%]")
    ax2.set_ylabel("F_cut vs -2sigma [%]")
    axes[2].set_title("(c) spurious current and force magnitude")
    axes[2].grid(alpha=0.3, which="both")
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout()
    path = OUT / "fig3_uncertainty.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("figure ->", path)

    # ---- fig4: radial profile + time history ----------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    run_files = sorted(RUNS.glob("run_*.json"))
    fine = None
    for p in run_files:
        d = json.loads(p.read_text())
        if fine is None or d["setup"]["R_over_dx"] > fine["setup"]["R_over_dx"]:
            fine = d
    pr = fine["pressure_profile"]
    axes[0].errorbar([q["r_over_R"] for q in pr], [q["p_mean_Pa"] for q in pr],
                     yerr=[q["p_std_Pa"] for q in pr], fmt="o-", ms=4,
                     label="SPH pressure (mean +- sd in shell)")
    axes[0].axhline(fine["reference"]["delta_p_Pa"], color="k", ls="--",
                    label=r"analytic $\Delta p=\sigma/R$")
    axes[0].axhline(0.0, color="g", ls=":", label="analytic p_outside = 0")
    axes[0].set_xlabel("r / R")
    axes[0].set_ylabel("p [Pa]")
    axes[0].set_title(f"(a) radial pressure profile, R/dx = "
                      f"{fine['setup']['R_over_dx']:.1f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    xr = np.array([r["R_over_dx"] for r in rows])
    for f, mk in zip(SAMPLE_FRACTIONS, ("o-", "s-", "^-", "d-")):
        vals = np.array([r["time_history"][str(f)]["rel_error_pct"] for r in rows])
        cis = np.array([r["time_history"][str(f)]["ci95_pct"] for r in rows])
        axes[1].errorbar(xr, vals, yerr=np.nan_to_num(cis), fmt=mk, capsize=3,
                         label=f"t = {f} $t_\\sigma$")
    axes[1].axhline(0, color="k", ls="--")
    axes[1].axhline(-GATE_15, color="r", ls="-.")
    axes[1].set_xlabel("R / dx")
    axes[1].set_ylabel("relative error of Delta p [%]")
    axes[1].set_title("(b) time development of the measured jump")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig4_baseline_compare.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("figure ->", path)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx-list", type=float, nargs="+",
                    default=[0.003, 0.0025, 0.002, 0.0015, 0.001])
    ap.add_argument("--realisations", type=int, default=3)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=7.2)
    ap.add_argument("--steps-cap", type=int, default=4000)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    (OUT / "fields").mkdir(parents=True, exist_ok=True)

    if not a.aggregate:
        dx_list = a.dx_list
        n_rea = a.realisations
        if a.quick:
            dx_list, n_rea, a.radius = [0.004], 2, 0.008
        for dx in dx_list:
            for k in range(n_rea):
                rea = realisation(k, dx, a.radius, 1000.0)
                tag = f"dx{dx*1e3:.2f}_{rea['label']}"
                path = RUNS / f"run_{tag}.json"
                if path.exists():
                    print("skip (cached)", tag)
                    continue
                t0 = time.time()
                rep, field = relax_run(a.radius, dx, a.sigma, 1000.0, 20.0, 0.01, rea)
                rep["wall_time_s"] = time.time() - t0
                path.write_text(json.dumps(rep, indent=1))
                (OUT / "fields" / f"field_{dx*1e3:.2f}.json").write_text(
                    json.dumps(field))
                print(f"{tag}: R/dx={rep['setup']['R_over_dx']:.1f} "
                      f"N={rep['setup']['n_particles']} "
                      f"dp err={rep['samples']['1.0']['rel_error_pct']:.1f}% "
                      f"({rep['wall_time_s']:.1f}s)")
        return aggregate()
    return aggregate()


if __name__ == "__main__":
    main()
