"""X4 audit: the closed fsi2 insert body under a linear pressure field.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

The published ``GAP-SPH-FSI-BENCHMARK_xval.md`` table records case X4 (fsi2
insert body, analytic linear field, no gravity, analytic load ``-c A``) as a
FAIL for both the OSS arm (+67.7 % at ``dx = 0.025``) and the panel arm
(+32.3 %), with the error *growing* under refinement.  This driver decides
whether that is an operator defect or a harness defect, and re-measures the
arms on a fully wetted body.

What is measured (nothing hand-copied into the report):

* **census** -- number of solid surface particles and how many of them have *no*
  fluid neighbour inside the kernel support, for the legacy bounding box
  (``wetted=False``) and the corrected one (``wetted=True``).
* **arms** -- total load in the gradient direction for
    - ``quad``: ``-sum_i p(x_i) n_i A_i`` on the same particle set, i.e. the
      exact surface quadrature.  It isolates surface sampling from operator
      error: if ``quad`` is exact while an operator is not, the operator is at
      fault; if ``quad`` itself is off, the *surface* is at fault.
    - ``panel``: kernel-weighted WLS plane fit (``panel_pressure_force``)
    - ``oss``: raw one-sided sum (``pressure_force_from_fluid``)
    - ``oss_norm``: one-sided sum with the measured half-space moment
      normalisation (``pressure_force_from_fluid_normalised``)
* **segments** -- load per surface segment (cylinder arc / beam top / beam
  bottom / tip) against the analytic per-segment value, to localise any残留 error.

Usage::

    python benchmarks/xval_fsi2/closed_body_audit.py            # writes artifacts
    python benchmarks/xval_fsi2/closed_body_audit.py --quick    # dx = 0.1, 0.05
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
LADDER = (0.1, 0.05, 0.025, 0.0125)
GATE = 0.15
TARGET = 0.03


def _cfg(case: C.LoadCase) -> ff.ForceFromFluidConfig:
    return ff.ForceFromFluidConfig(dim=2, h=case.h, kernel="wendland")


def _arms(case: C.LoadCase) -> dict[str, torch.Tensor]:
    cfg = _cfg(case)
    ps, vs, ns = case.pos_solid, case.vol_solid, case.normal_solid
    pf, vf, p = case.pos_fluid, case.vol_fluid, case.pressure_fluid
    zero_s, zero_f = torch.zeros_like(ps), torch.zeros_like(pf)
    acc = torch.zeros(2, dtype=torch.float64)
    face = vs.pow(0.5)
    p_solid = case.pressure_at(ps) if case.pressure_at is not None else None
    arms = {
        "quad": -(p_solid[:, None] * ns * face[:, None]).sum(dim=0, keepdim=True)
        if p_solid is not None else torch.zeros_like(ns[:1]),
        "panel": ff.panel_pressure_force(ps, vs, ns, pf, vf, p, cfg).sum(
            dim=0, keepdim=True),
        "oss": ff.pressure_force_from_fluid(
            ps, vs, ns, zero_s, case.vel_solid, pf, vf, p, case.rho_fluid,
            case.vel_fluid, acc, cfg).sum(dim=0, keepdim=True),
        "oss_norm": ff.pressure_force_from_fluid_normalised(
            ps, vs, ns, zero_s, case.vel_solid, pf, vf, p, case.rho_fluid,
            case.vel_fluid, acc, cfg).sum(dim=0, keepdim=True),
    }
    return arms


def _census(case: C.LoadCase) -> dict:
    cfg = _cfg(case)
    i_s, _ = ff.cross_pairs(case.pos_solid, case.pos_fluid, cfg.support)
    if i_s.numel():
        cnt = torch.zeros(case.pos_solid.shape[0], dtype=torch.long)
        cnt.index_add_(0, i_s, torch.ones_like(i_s))
    else:
        cnt = torch.zeros(case.pos_solid.shape[0], dtype=torch.long)
    n_s = int(case.pos_solid.shape[0])
    dry = int((cnt == 0).sum())
    return {"n_solid": n_s, "n_fluid": int(case.pos_fluid.shape[0]),
            "dry_solid": dry, "dry_fraction": dry / max(n_s, 1),
            "support": float(cfg.support)}


def _segment_labels(dx: float, n_solid: int) -> torch.Tensor:
    """Rebuild the same surface with per-segment labels (0 cyl, 1 top, 2 bot, 3 tip)."""
    cx, cy = C.FSI2["insert_circle_center"]
    r, bh, bl = C.FSI2["insert_circle_radius"], C.FSI2["bh"], C.FSI2["bl"]
    pos, _ = C._fsi2_body(dx)
    labels = torch.full((pos.shape[0],), 3, dtype=torch.long)
    on_cyl = (torch.linalg.vector_norm(
        pos - torch.tensor([cx, cy], dtype=torch.float64), dim=1) - r).abs() < 1.0e-9
    top = (pos[:, 1] - cy).abs() > 0.5 * bh - 1.0e-9
    beam = pos[:, 0] > cx + r - 1.0e-9
    labels[on_cyl] = 0
    labels[beam & (pos[:, 1] > cy)] = 1
    labels[beam & (pos[:, 1] < cy)] = 2
    assert pos.shape[0] == n_solid, (pos.shape[0], n_solid)
    del top
    return labels


def _segments(case: C.LoadCase, force_panel: torch.Tensor,
              force_oss: torch.Tensor) -> list[dict]:
    """Per-segment load vs the analytic per-segment value ``-sum p n A``.

    Uses the case's own analytic pressure law on the *operator's* particle set,
    so the segment split cannot hide a surface-sampling error inside an
    operator error.
    """
    labels = _segment_labels(case.dx, case.pos_solid.shape[0])
    if case.pressure_at is None:
        return []
    p_solid = case.pressure_at(case.pos_solid)
    face = case.vol_solid.pow(0.5)
    exact = -(p_solid[:, None] * case.normal_solid * face[:, None])
    names = {0: "cylinder_arc", 1: "beam_top", 2: "beam_bottom", 3: "beam_tip"}
    rows = []
    for key, name in names.items():
        m = labels == key
        if not bool(m.any()):
            continue
        rows.append({
            "segment": name, "n": int(m.sum()),
            "exact_y": float(exact[m, 1].sum()),
            "panel_y": float(force_panel[m, 1].sum()),
            "oss_y": float(force_oss[m, 1].sum()),
        })
    return rows


def run(quick: bool = False) -> dict:
    ladder = LADDER[:2] if quick else LADDER
    out: dict = {
        "anchor": "XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID",
        "driver": "benchmarks/xval_fsi2/closed_body_audit.py",
        "case": "fsi2_insert_linear (fsi2 insert body, analytic linear field)",
        "reference": "-c A, A = pi r^2 + bh*bl (divergence theorem, closed body)",
        "gate": GATE, "target": TARGET, "kernel": "wendland", "h_factor": 1.3,
        "ladder": list(ladder), "census": [], "loads": [], "segments": [],
    }
    for variant, kw in (("legacy_dry", {"wetted": False, "include_buried": True}),
                        ("wetted", {"wetted": True, "include_buried": False})):
        for dx in ladder:
            case = C.fsi2_insert_linear(dx, **kw)
            ref_y = float(case.reference_force[1])
            cen = _census(case)
            cen.update({"dx": dx, "variant": variant})
            out["census"].append(cen)
            cfg = _cfg(case)
            arms = _arms(case)
            for name, f in arms.items():
                fy = float(f[0, 1])
                out["loads"].append({
                    "dx": dx, "variant": variant, "arm": name,
                    "F_x": float(f[0, 0]), "F_y": fy, "ref_y": ref_y,
                    "err_y": (fy - ref_y) / abs(ref_y),
                    "pass_gate": abs(fy - ref_y) / abs(ref_y) < GATE,
                    "pass_target": abs(fy - ref_y) / abs(ref_y) < TARGET,
                })
            # per-segment localisation on the wetted build only
            if variant == "wetted":
                ps, vs, ns = case.pos_solid, case.vol_solid, case.normal_solid
                pf, vf, p = case.pos_fluid, case.vol_fluid, case.pressure_fluid
                panel = ff.panel_pressure_force(ps, vs, ns, pf, vf, p, cfg)
                oss = ff.pressure_force_from_fluid(
                    ps, vs, ns, torch.zeros_like(ps), case.vel_solid, pf, vf, p,
                    case.rho_fluid, case.vel_fluid, torch.zeros(2,
                    dtype=torch.float64), cfg)
                for row in _segments(case, panel, oss):
                    row["dx"] = dx
                    out["segments"].append(row)
            print(f"[{variant}] dx={dx}: " + ", ".join(
                f"{r['arm']} {r['err_y'] * 100:+.2f}%" for r in out["loads"]
                if r["dx"] == dx and r["variant"] == variant), flush=True)
    out["verdict"] = _verdict(out)
    return out


def _verdict(data: dict) -> dict:
    """Programmatic statement of what the numbers support (no prose claims)."""
    wet = [r for r in data["loads"] if r["variant"] == "wetted"]
    dry = [r for r in data["loads"] if r["variant"] == "legacy_dry"]
    wet_cen = [c for c in data["census"] if c["variant"] == "wetted"]
    dry_cen = [c for c in data["census"] if c["variant"] == "legacy_dry"]
    return {
        "legacy_dry_fraction_max": max(c["dry_fraction"] for c in dry_cen),
        "wetted_dry_fraction_max": max(c["dry_fraction"] for c in wet_cen),
        "quad_max_abs_err_wetted": max(abs(r["err_y"]) for r in wet
                                       if r["arm"] == "quad"),
        "panel_max_abs_err_wetted": max(abs(r["err_y"]) for r in wet
                                        if r["arm"] == "panel"),
        "oss_max_abs_err_wetted": max(abs(r["err_y"]) for r in wet
                                      if r["arm"] == "oss"),
        "oss_norm_max_abs_err_wetted": max(abs(r["err_y"]) for r in wet
                                           if r["arm"] == "oss_norm"),
        "panel_all_pass_gate_wetted": all(r["pass_gate"] for r in wet
                                          if r["arm"] == "panel"),
        "panel_all_pass_target_wetted": all(r["pass_target"] for r in wet
                                            if r["arm"] == "panel"),
        "oss_max_abs_err_legacy": max(abs(r["err_y"]) for r in dry
                                      if r["arm"] == "oss"),
        "panel_max_abs_err_legacy": max(abs(r["err_y"]) for r in dry
                                        if r["arm"] == "panel"),
    }


def main() -> int:
    quick = "--quick" in sys.argv
    data = run(quick=quick)
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "closed_body_x4.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True))
    print(json.dumps(data["verdict"], indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())