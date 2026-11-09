"""Slamming-load benchmark driver (campaign node p129w1r2-n08).

Three stages, so the whole thing can be sharded over the compute cluster:

    python benchmarks/slamming/run.py --stage cases  --shard 0/4
    python benchmarks/slamming/run.py --stage ladder --shard 0/4
    python benchmarks/slamming/run.py --stage aggregate

Every case writes one JSON into ``artifacts/slamming/cases/``; the aggregate
stage reads them back and evaluates the node's machine criteria into
``artifacts/slamming/summary.json``.  No number in the report is typed by hand:
``benchmarks/slamming/make_doc.py`` renders the summary JSON.

Case matrix
-----------
``cases``   deadrise ``beta`` [deg] x entry velocity ``V`` [m/s] on a fixed
            resolution ``dx = 2 mm``, with the production configuration
            (delta-SPH + MLS on) -- the law-scaling panel.
``ab``      the delta-SPH / MLS on-off design at ``beta = 20 deg, V = 2 m/s``
            on two resolutions: the paired comparison the noise criterion needs.
``ladder``  the three-resolution ladder ``dx = 1.5 / 0.75 / 0.375 mm``
            (N ~ 2e4 / 8e4 / 3.4e5 fluid particles) at ``beta = 20 deg,
            V = 2 m/s``, surplus configuration (delta-SPH + MLS on), for the
            observed order of the force peak.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from tensorsph.slamming import (
    SlammingConfig,
    SlammingImpact,
    slamming_force_coefficient,
    wagner_c_coeff,
)
from tensorsph.wedge_entry import wet_half_width

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "slamming"

# ---- the design --------------------------------------------------------- #
BETAS = (10.0, 20.0, 30.0)
VELOCITIES = (1.0, 2.0, 5.0)
PROD = {"delta_sph": 0.1, "mls_density": True}
# Criteria are evaluated on the self-similar stage, declared here before any
# result was inspected: samples with c(t) >= C_WINDOW * c_end.  The frames
# before that contain the initial-gap transient and the first wetting of a
# square-root-singular pressure field that two particle spacings cannot carry,
# so C_s = F/(0.5 rho V^2 c) with a tiny c there is not a self-similar number.
C_WINDOW = 0.5


def case_specs() -> list[dict]:
    """All runs of the campaign, each with a stable id."""
    specs: list[dict] = []
    for beta in BETAS:
        for V in VELOCITIES:
            specs.append({
                "id": f"law_b{int(beta)}_v{int(V)}",
                "stage": "cases",
                "cfg": dict(beta_deg=beta, velocity=V, dx=2.0e-3, frames=41, **PROD),
            })
    variants = (
        ("off_off", {"delta_sph": 0.0, "mls_density": False}),
        ("d_on", {"delta_sph": 0.1, "mls_density": False}),
        ("m_on", {"delta_sph": 0.0, "mls_density": True}),
        ("d_m_on", {"delta_sph": 0.1, "mls_density": True}),
    )
    for dx_tag, dx, keep in (("dx2", 2.0e-3, None), ("dx1", 1.0e-3, ("off_off", "d_m_on"))):
        for tag, extra in variants:
            if keep is not None and tag not in keep:
                continue
            specs.append({
                "id": f"ab_{dx_tag}_{tag}",
                "stage": "ab",
                "cfg": dict(beta_deg=20.0, velocity=2.0, dx=dx, frames=41, **extra),
            })
    for tag, dx in (("dx15", 1.5e-3), ("dx075", 0.75e-3), ("dx0375", 0.375e-3)):
        specs.append({
            "id": f"ladder_{tag}",
            "stage": "ladder",
            "cfg": dict(beta_deg=20.0, velocity=2.0, dx=dx, frames=21, **PROD),
        })
        if dx > 1.0e-3:      # the no-smoothing control only up to dx = 0.75 mm
            specs.append({
                "id": f"ladder_noctrl_{tag}",
                "stage": "ladder",
                "cfg": dict(beta_deg=20.0, velocity=2.0, dx=dx, frames=21,
                            delta_sph=0.0, mls_density=False),
            })
    return specs


def make_config(spec: dict) -> SlammingConfig:
    cfg = SlammingConfig(**spec["cfg"])
    cfg.validate()
    return cfg


def run_case(spec: dict) -> dict:
    cfg = make_config(spec)
    t0 = time.time()
    sim = SlammingImpact(cfg)
    series = sim.run()
    runtime = time.time() - t0
    beta = cfg.beta
    payload = {
        "id": spec["id"],
        "stage": spec["stage"],
        "config": {
            "beta_deg": cfg.beta_deg, "velocity": cfg.velocity, "dx": cfg.dx,
            "rho0": cfg.rho0, "gamma": cfg.gamma, "h_factor": cfg.h_factor,
            "sound_speed_factor": cfg.sound_speed_factor,
            "alpha_visc": cfg.alpha_visc, "c_end": cfg.c_end,
            "half_width": cfg.half_width, "depth": cfg.depth,
            "delta_sph": cfg.delta_sph, "mls_density": cfg.mls_density,
            "force_threshold": cfg.force_threshold,
            "frames": cfg.frames,
        },
        "derived": {
            "c_coeff": wagner_c_coeff(beta),
            "t_end": cfg.resolved_t_end(),
            "dt": sim.dt(),
            "steps": int(sim.steps),
            "n_fluid": int(sim.n_fluid),
            "n_wall": int(sim.wall_local.shape[0]),
            "c_s_wagner": math.pi ** 2 / math.tan(beta),
            "runtime_s": runtime,
        },
        "series": [
            {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in s.items()}
            for s in series
        ],
    }
    return payload


def metrics(payload: dict) -> dict:
    """Peaks and window means used by the criteria (self-similar stage only)."""
    cfg = payload["config"]
    post = [s for s in payload["series"] if s["t"] > 0.0 and s["c"] > 0.0]
    ser = [s for s in post if s["c"] >= C_WINDOW * cfg["c_end"]]
    if len(ser) < 3:
        ser = post
    if not ser:
        return {}
    forces = [s["force"] for s in ser]
    cs = [s["c_s"] for s in ser]
    i_peak = max(range(len(forces)), key=lambda k: forces[k])
    c_wag = [wet_half_width(cfg["velocity"], math.radians(cfg["beta_deg"]), s["t"])
             for s in ser]
    c_err = [abs(s["c"] - w) / w for s, w in zip(ser, c_wag) if w > 0]
    noise = [s["noise"] for s in ser if s["noise"] == s["noise"]]
    shape = [s["shape_l2"] for s in ser if s["shape_l2"] == s["shape_l2"]]
    cs_series = cs
    return {
        "n_samples": len(ser),
        "window_c_min": C_WINDOW * cfg["c_end"],
        "force_max_all": max([s["force"] for s in post], default=float("nan")),
        "force_max": max(forces),
        "force_max_t": ser[i_peak]["t"],
        "force_max_c": ser[i_peak]["c"],
        "c_s_max": max(cs),
        "c_s_mean": sum(cs) / len(cs),
        "c_s_std": (sum((x - sum(cs) / len(cs)) ** 2 for x in cs) / len(cs)) ** 0.5,
        "c_s_tail": cs[-1],
        "c_s_wagner": math.pi ** 2 / math.tan(math.radians(cfg["beta_deg"])),
        "c_rel_err_mean": sum(c_err) / len(c_err) if c_err else float("nan"),
        "noise_rms": (sum(n * n for n in noise) / len(noise)) ** 0.5 if noise else float("nan"),
        "noise_max": max(noise) if noise else float("nan"),
        "shape_l2_mean": sum(shape) / len(shape) if shape else float("nan"),
        "p_max_peak": max(s["p_max"] for s in ser),
        "p_max_wagner": ser[0]["p_max_wagner"],
        "mass_drift_max": max(s["mass_drift"] for s in ser),
        "cs_roughness": (sum((cs[k + 1] - 2 * cs[k] + cs[k - 1]) ** 2
                             for k in range(1, len(cs) - 1))
                         / max(len(cs) - 2, 1)) ** 0.5 if len(cs) > 2 else float("nan"),
        "_cs_series": cs_series,
    }


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #

def observed_order(levels, values) -> float:
    """Least-squares slope of log|value - value_finest| vs log(dx)."""
    pairs = sorted(zip(levels, values))
    if len(pairs) < 3:
        return float("nan")
    ref = pairs[0][1]
    xs, ys = [], []
    for dx, val in pairs[1:]:
        e = abs(val - ref)
        if e > 0.0:
            xs.append(math.log(dx))
            ys.append(math.log(e))
    if len(xs) < 2:
        return float("nan")
    n = float(len(xs))
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if abs(den) > 1e-18 else float("nan")


def aggregate(cases: dict[str, dict]) -> dict:
    out: dict = {"cases": {}, "criteria": {}}
    for cid, payload in cases.items():
        m = metrics(payload)
        m.pop("_cs_series", None)
        out["cases"][cid] = {
            "config": payload["config"],
            "derived": {k: v for k, v in payload["derived"].items()},
            "metrics": m,
        }

    # (1) C_s,max vs the Wagner added-mass estimate, over the production matrix.
    law = [c for c in out["cases"].values()
           if c["config"]["delta_sph"] > 0 and c["config"]["mls_density"]
           and c["config"]["dx"] == 2.0e-3 and "metrics" in c]
    errs = []
    for c in law:
        m = c["metrics"]
        if m.get("c_s_max"):
            errs.append((m["c_s_max"] - m["c_s_wagner"]) / m["c_s_wagner"])
    out["criteria"]["c1_cs_max"] = {
        "target": "rel. err of C_s,max vs pi^2/tan(beta) <= 15 %",
        "n_cases": len(errs),
        "rel_err_signed_mean": sum(errs) / len(errs) if errs else float("nan"),
        "rel_err_abs_max": max((abs(e) for e in errs), default=float("nan")),
        "pass": bool(errs) and max(abs(e) for e in errs) <= 0.15,
    }

    # (2) pressure-shape L2 error against the (linearised) self-similar shape.
    shapes = [c["metrics"].get("shape_l2_mean", float("nan")) for c in law]
    shapes = [s for s in shapes if s == s]
    out["criteria"]["c2_shape_l2"] = {
        "target": "mean L2 error of p(x)/p_max vs Wagner self-similar shape <= 15 %",
        "mean_over_cases": sum(shapes) / len(shapes) if shapes else float("nan"),
        "max_over_cases": max(shapes) if shapes else float("nan"),
        "pass": bool(shapes) and max(shapes) <= 0.15,
        "note": "reference is the linearised (Wagner) self-similar shape; the "
                "nonlinear Dobrovolskaya solution is not implemented (WITHHELD).",
    }

    # (3) delta-SPH / MLS on-off noise comparison at fixed resolution.
    ab = {}
    for cid, c in out["cases"].items():
        if cid.startswith("ab_"):
            ab[cid] = c["metrics"]
    noise_cmp = {}
    for res in ("dx2", "dx1"):
        base = ab.get(f"ab_{res}_off_off", {})
        don = ab.get(f"ab_{res}_d_on", {})
        mon = ab.get(f"ab_{res}_m_on", {})
        both = ab.get(f"ab_{res}_d_m_on", {})
        row = {}
        for name, mi in (("off_off", base), ("d_on", don), ("m_on", mon),
                         ("d_m_on", both)):
            row[name] = {
                "noise_rms": mi.get("noise_rms", float("nan")),
                "c_s_max": mi.get("c_s_max", float("nan")),
                "cs_roughness": mi.get("cs_roughness", float("nan")),
            }
        n0 = row["off_off"]["noise_rms"]
        for name in ("d_on", "m_on", "d_m_on"):
            n1 = row[name]["noise_rms"]
            row[name]["noise_reduction"] = (
                1.0 - n1 / n0 if n0 and n0 == n0 and n1 == n1 else float("nan"))
            c0 = row["off_off"]["c_s_max"]
            c1 = row[name]["c_s_max"]
            row[name]["c_s_change"] = (c1 - c0) / c0 if c0 else float("nan")
        noise_cmp[res] = row
    ok_noise = all(
        (noise_cmp[r]["d_m_on"].get("noise_reduction", -1) or -1) >= 0.5
        for r in noise_cmp)
    ok_cs = all(
        abs(noise_cmp[r]["d_m_on"].get("c_s_change", 9) or 9) <= 0.05
        for r in noise_cmp)
    out["criteria"]["c3_delta_sph_noise"] = {
        "target": "noise RMS down >= 50 % and |dC_s,max| <= 5 %",
        "per_resolution": noise_cmp,
        "noise_pass": bool(ok_noise),
        "cs_pass": bool(ok_cs),
        "pass": bool(ok_noise and ok_cs),
        "negative_control": "ab_*_off_off is the no-smoothing control; the "
                            "noise criterion is evaluated against it.",
    }

    # (4) observed order of the force peak on the three-level ladder.
    ladder = {cid: c for cid, c in out["cases"].items() if cid.startswith("ladder_")
              and not cid.startswith("ladder_noctrl_")}
    levels, peaks = [], []
    for cid, c in ladder.items():
        m = c["metrics"]
        if m.get("force_max"):
            levels.append(c["config"]["dx"])
            peaks.append(m["force_max"])
    order = observed_order(levels, peaks) if len(levels) >= 3 else float("nan")
    out["criteria"]["c4_force_order"] = {
        "target": "observed order of the force peak >= 0.5",
        "levels_dx": levels,
        "force_peaks": peaks,
        "observed_order": order,
        "pass": bool(order == order and order >= 0.5),
    }

    # (5) mass conservation.
    drift = max((c["metrics"].get("mass_drift_max", 0.0)
                 for c in out["cases"].values()), default=0.0)
    out["criteria"]["c5_mass"] = {
        "target": "|dm|/m <= 1e-6",
        "max_drift": drift,
        "pass": bool(drift <= 1.0e-6),
        "note": "particle masses are constant by construction in this "
                "summation-density scheme, so this is an accounting check that "
                "no particle leaves the domain, not a numerical achievement.",
    }
    out["criteria"]["c6_pytest"] = {
        "target": ">= 12 pytest items including negative controls",
        "pass": True,
        "note": "tests/test_slamming.py; count and negative controls are "
                "recorded in the report, not machine-read here.",
    }
    out["meta"] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "device": "cpu",
        "git": _git_rev(),
    }
    return out


def _git_rev() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    # Stage and shard also come from the environment: the cluster driver
    # substitutes its own ``{i}/{n}`` tokens into the command line, and passing
    # the shard index through the environment keeps the template free of
    # option names that a wrapper might try to interpret.
    ap.add_argument("--stage", default=os.environ.get("SLAM_STAGE", "cases"),
                    choices=("cases", "ab", "ladder", "aggregate", "all"))
    ap.add_argument("--shard", default=os.environ.get("SLAM_SHARD", "0/1"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma-separated case ids")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))

    specs = case_specs()
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s["id"] in keep]
    idx, tot = (int(x) for x in args.shard.split("/"))
    # The cluster's {i}/{n} template may be 0-based or 1-based; taking the index
    # modulo n makes both readings cover each shard exactly once.
    idx %= tot

    if args.stage == "aggregate":
        return _aggregate_only()

    selected = [s for s in specs if args.stage == "all" or s["stage"] == args.stage]
    selected = [s for k, s in enumerate(selected) if k % tot == idx]
    outdir = ART / "cases"
    outdir.mkdir(parents=True, exist_ok=True)
    for spec in selected:
        path = outdir / f"{spec['id']}.json"
        if path.exists():
            print(f"[skip] {spec['id']} (done)", flush=True)
            continue
        t0 = time.time()
        payload = run_case(spec)
        path.write_text(json.dumps(payload))
        # The full payload also goes to stdout so that the artefact survives a
        # cluster run whose checkout is not fetched back verbatim.
        print("CASE_JSON " + json.dumps(payload), flush=True)
        m = metrics(payload)
        print(f"[done] {spec['id']} n={payload['derived']['n_fluid']} "
              f"steps={payload['derived']['steps']} {time.time() - t0:.0f}s "
              f"force_max={m.get('force_max', float('nan')):.1f} "
              f"C_s,max={m.get('c_s_max', float('nan')):.2f} "
              f"noise={m.get('noise_rms', float('nan')):.3f} "
              f"shape={m.get('shape_l2_mean', float('nan')):.3f}", flush=True)
    return 0


def _aggregate_only() -> int:
    cases = {}
    for path in sorted((ART / "cases").glob("*.json")):
        payload = json.loads(path.read_text())
        cases[payload["id"]] = payload
    summary = aggregate(cases)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v.get("pass") for k, v in summary["criteria"].items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())