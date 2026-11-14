"""Audit of the one-sided pressure-transfer moment and its normalisation.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

What this driver measures (all numbers reproduced by ``--out``; nothing is
hand-copied into the report):

* **A** -- the raw OSS ``PressureForceFromFluid`` arm on the flat-wall
  hydrostatic cases X1 (wall normal transverse to gravity) and X2 (floor).
* **B** -- the same load with the per-particle half-space moment normalisation
  :func:`tensorsph.fsi_force_from_fluid.half_space_normalisation`.
* **C** -- the measured lattice moment ``g_i`` against the continuum closed form
  ``-pi int_0^R W dr`` as a function of ``h/dx`` (this decides whether the effect
  is a kernel-truncation law or a discretisation error).
* **D** -- the scope boundary: number of solid layers (interior layers must not
  be normalised) and the closed-body pressure-gradient case X4 (the
  normalisation is not applicable there).

Usage::

    python benchmarks/xval_fsi2/pressure_normalisation_audit.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

import cases as C  # noqa: E402
from tensorsph import fsi_force_from_fluid as ff  # noqa: E402

ART = ROOT / "artifacts" / "xval_fsi2"

KERNELS = ("wendland", "quintic")
LADDER_FLAT = (0.05, 0.025, 0.0125)
LADDER_INSERT = (0.1, 0.05, 0.025)

GATE = 0.15        # owner standard: |error| < 15 % required
TARGET = 0.03      # target: |error| < 3 %


def _cfg(case: C.LoadCase, kernel: str) -> ff.ForceFromFluidConfig:
    return ff.ForceFromFluidConfig(dim=2, h=case.h, kernel=kernel)


def _pressure(case: C.LoadCase, cfg: ff.ForceFromFluidConfig) -> torch.Tensor:
    return ff.pressure_force_from_fluid(
        case.pos_solid, case.vol_solid, case.normal_solid,
        torch.zeros_like(case.pos_solid), case.vel_solid,
        case.pos_fluid, case.vol_fluid, case.pressure_fluid, case.rho_fluid,
        case.vel_fluid, case.acc_ext_fluid, cfg)


def _pressure_normalised(case: C.LoadCase,
                         cfg: ff.ForceFromFluidConfig) -> torch.Tensor:
    return ff.pressure_force_from_fluid_normalised(
        case.pos_solid, case.vol_solid, case.normal_solid,
        torch.zeros_like(case.pos_solid), case.vel_solid,
        case.pos_fluid, case.vol_fluid, case.pressure_fluid, case.rho_fluid,
        case.vel_fluid, case.acc_ext_fluid, cfg)


def _score(force: torch.Tensor, ref: torch.Tensor) -> dict:
    direction = ref / ref.norm()
    got = float(force.sum(dim=0) @ direction)
    return {"got": got, "ref": float(ref.norm()),
            "error": got / float(ref.norm()) - 1.0,
            "leakage": float((force.sum(dim=0) - got * direction).norm())}


def flat_wall_table() -> list[dict]:
    rows = []
    for kernel in KERNELS:
        for name, make in (("X1_vertical_wall", C.vertical_wall),
                           ("X2_horizontal_gate", C.horizontal_gate)):
            for dx in LADDER_FLAT:
                case = make(dx)
                cfg = _cfg(case, kernel)
                raw = _score(_pressure(case, cfg), case.reference_force)
                norm = _score(_pressure_normalised(case, cfg), case.reference_force)
                g = ff.half_space_wall_moment(case.pos_solid, case.vol_solid,
                                              case.normal_solid, case.pos_fluid,
                                              case.vol_fluid, cfg)
                m = ff.half_space_normalisation(case.pos_solid, case.vol_solid,
                                                case.normal_solid, case.pos_fluid,
                                                case.vol_fluid, cfg)
                g_cont = ff.half_space_moment_continuum(case.h, 2, kernel)
                rows.append({
                    "case": name, "kernel": kernel, "dx": dx, "h": case.h,
                    "h_over_dx": case.h / dx,
                    "moment_continuum": g_cont,
                    "lattice_over_continuum": float(g.median()) / g_cont,
                    # independent continuum prediction of the *raw* branch:
                    # ratio to the exact traction = 2 dx g  ->  error = 2 dx g - 1
                    "raw_error_continuum_prediction": 2.0 * dx * g_cont - 1.0,
                    "raw_error": raw["error"], "normalised_error": norm["error"],
                    "raw_leakage": raw["leakage"], "normalised_leakage": norm["leakage"],
                    "moment_median": float(g.median()),
                    "multiplier_median": float(m.median()),
                })
    return rows


def moment_law_table(h_over_dx=(1.0, 1.3, 1.6, 2.0, 2.6, 3.0, 4.0)) -> list[dict]:
    """Lattice moment / continuum moment for a plane wall versus ``h/dx``."""
    rows = []
    dx = 0.01
    depth = 1.0
    for kernel in KERNELS:
        for ratio in h_over_dx:
            if ratio < kernels_support(kernel):
                continue
            case = C.vertical_wall(dx, depth=depth, h_factor=ratio)
            cfg = _cfg(case, kernel)
            g = ff.half_space_wall_moment(case.pos_solid, case.vol_solid,
                                          case.normal_solid, case.pos_fluid,
                                          case.vol_fluid, cfg)
            # only wall particles whose whole support is inside the column
            margin = cfg.support
            interior = ((case.pos_solid[:, 1] > margin)
                        & (case.pos_solid[:, 1] < depth - margin))
            lattice = float(g[interior].abs().median()) if bool(interior.any()) else float("nan")
            continuum = abs(ff.half_space_moment_continuum(case.h, 2, kernel))
            rows.append({"kernel": kernel, "h_over_dx": ratio, "lattice": lattice,
                         "continuum": continuum,
                         "ratio": lattice / continuum if continuum else float("nan")})
    return rows


def kernels_support(kernel: str) -> float:
    from tensorsph.wcsph import kernels as K
    return K.SUPPORT_FACTOR[kernel]


def thickness_table(layers=(1, 2, 3)) -> list[dict]:
    rows = []
    dx = 0.025
    for kernel in KERNELS:
        for n_layers in layers:
            case = C.vertical_wall(dx, layers=n_layers)
            cfg = _cfg(case, kernel)
            raw = _score(_pressure(case, cfg), case.reference_force)
            norm = _score(_pressure_normalised(case, cfg), case.reference_force)
            rows.append({"kernel": kernel, "layers": n_layers, "dx": dx,
                         "n_solid": int(case.pos_solid.shape[0]),
                         "raw_error": raw["error"],
                         "normalised_error": norm["error"]})
    return rows


def closed_body_table() -> list[dict]:
    rows = []
    for kernel in KERNELS:
        for dx in LADDER_INSERT:
            case = C.fsi2_insert_linear(dx)
            cfg = _cfg(case, kernel)
            raw = _score(_pressure(case, cfg), case.reference_force)
            norm = _score(_pressure_normalised(case, cfg), case.reference_force)
            panel = _score(ff.panel_pressure_force(
                case.pos_solid, case.vol_solid, case.normal_solid,
                case.pos_fluid, case.vol_fluid, case.pressure_fluid, cfg),
                case.reference_force)
            rows.append({"case": "X4_fsi2_insert_linear", "kernel": kernel,
                         "dx": dx, "raw_error": raw["error"],
                         "normalised_error": norm["error"],
                         "panel_error": panel["error"]})
    return rows


def main() -> dict:
    out = {
        "anchor": "XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID",
        "gate": GATE, "target": TARGET,
        "flat_wall": flat_wall_table(),
        "kernel_constant": {
            "wendland_2d_7_over_3pi_h": [7.0 / (3.0 * math.pi * r["h"])
                                         for r in []],
        },
        "moment_law": moment_law_table(),
        "thickness": thickness_table(),
        "closed_body": closed_body_table(),
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "pressure_normalisation.json").write_text(json.dumps(out, indent=1))

    print("== flat walls: raw vs half-space normalised ==")
    for r in out["flat_wall"]:
        print(f"{r['case']:20s} {r['kernel']:9s} dx={r['dx']:<7g} "
              f"raw={r['raw_error']*100:+7.2f}% "
              f"(continuum law {r['raw_error_continuum_prediction']*100:+7.2f}%) "
              f"norm={r['normalised_error']*100:+7.2f}% "
              f"lat/cont={r['lattice_over_continuum']:.3f}")
    print("== lattice moment / continuum moment ==")
    for r in out["moment_law"]:
        print(f"{r['kernel']:9s} h/dx={r['h_over_dx']:<5g} lattice={r['lattice']:8.4f} "
              f"continuum={r['continuum']:8.4f} ratio={r['ratio']:.4f}")
    print("== thickness (scope) ==")
    for r in out["thickness"]:
        print(f"{r['kernel']:9s} layers={r['layers']} n={r['n_solid']:4d} "
              f"raw={r['raw_error']*100:+7.2f}% norm={r['normalised_error']*100:+7.2f}%")
    print("== closed body X4 (scope) ==")
    for r in out["closed_body"]:
        print(f"{r['kernel']:9s} dx={r['dx']:<6g} raw={r['raw_error']*100:+7.2f}% "
              f"norm={r['normalised_error']*100:+7.2f}% panel={r['panel_error']*100:+7.2f}%")
    return out


if __name__ == "__main__":
    main()