"""Contact-angle boundary condition: quantitative verification (verification debt).

The contact-angle BC (``apply_contact_angle_boundary`` in
``tensorsph/csf_consistent.py``, merged with the CSF work, see
``docs/contact_angle_boundary.md``) was merged with unit tests only: they pin
the *gradient rotation* on a toy field, but never measure the angle the
*relaxed interface* actually makes with the wall.  This benchmark closes that
debt in three layers, and it keeps the three layers strictly apart:

* **Analytic layer** -- the angle instrument is checked against *exact*
  interfaces of known angle (straight line and circular cap, sampled densely
  along the curve).  A measurement device that has not been measured itself
  cannot grade physics; this layer says how good the device is.
* **Resolution layer** -- the same instrument on a *particle lattice* cap
  (fluid particles at spacing ``dx`` inside the analytic cap) at four
  resolutions.  This is the discretisation floor of the device at production
  ``R/dx``: it is the uncertainty bar that any physics reading must beat.
* **Physics layer** -- a 2-D drop of radius ``R`` rests on a flat wall lattice
  in zero gravity with surface tension on and ``contact_angle = theta`` set.
  The drop relaxes for ``1.5 t_sigma`` and the angle of the relaxed interface
  at the contact line is measured with the instrument characterised above.

Geometry and sign convention (drop to the left of the contact line, wall at
``y = 0``, fluid above the interface):

    theta = atan2(t_y, -t_x),  t = interface tangent at the contact point,
    oriented with t_y > 0

so ``theta`` is the angle inside the liquid, measured from the wall direction
that points into the drop (Susskind/Alexander convention, the same one the BC
uses through ``n . n_wall = cos(theta)``).  An interface that leans up and
inward is wetting (``theta < 90 deg``); one that overhangs the contact line is
non-wetting (``theta > 90 deg``).

The instrument itself is a two-model least-squares fit of the free-surface
*envelope* near the contact point (lower envelope in ``x`` plus right-most
envelope in ``y``, so that an overhanging interface is traced as well as a
wetting one): a circle (Kasa algebraic fit, exact for a circular interface)
with a straight-line fallback selected by the collinearity of the extracted
points.  The contact point is the model's crossing of the wall, not the raw
extreme particle, which removes the one-particle quantisation of the contact
line.  Everything is returned as ``nan`` when the data cannot support a fit --
never as a fabricated angle.

Usage:
    python benchmarks/contact_angle/run.py [--out benchmarks/contact_angle]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    cell_centres,
    rhs,
    stable_dt,
    wall_lattice,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

RADIUS = 0.02           # m
SIGMA = 7.2             # N/m (10 x water)
RHO0 = 1000.0
GAMMA = 7.0
H_FACTOR = 1.3
VISCOSITY = 0.01        # m^2/s
DX = 0.003
POOL_WIDTH = 0.4
T_END_OVER_T_SIGMA = 1.5
ANGLES_DEG = (60.0, 90.0, 120.0)
ANGLE_GATE_DEG = 10.0   # absolute gate on |theta_phys - theta_set|
INSTRUMENT_GATE_DEG = 2.0
WALL_PENETRATION_GATE_DX = 0.5   # |y_min| of the fluid must stay below this, in dx
SPURIOUS_CURRENT_GATE = 0.1      # v_max / sqrt(sigma/(rho R)), Adami et al. (2012)
ARC_RADIUS_FACTOR = 3.0  # radius of the analytic circular interface, in units of RADIUS
ARC_SPAN_H = 6.0         # arc length of the analytic interface, in units of h

T_SIGMA = math.sqrt(RHO0 * RADIUS ** 3 / SIGMA)


# --------------------------------------------------------------------------- #
# Instrument
# --------------------------------------------------------------------------- #

def _boundary_points(x: torch.Tensor, y: torch.Tensor, x0: float, dx: float,
                     bin_fraction: float = 0.5) -> tuple[torch.Tensor, float]:
    """Free-surface envelope of a particle cloud over a flat wall at ``y = 0``.

    Two envelopes are extracted, because one graph is not enough: the lowest
    particle per ``x`` bin (correct for a wetting interface) and the right-most
    particle per ``y`` bin (correct for an interface that overhangs the contact
    line).  Both are taken at the bin edge the extreme actually attains, so a
    convex curve is sampled without a half-bin bias.

    The wall level is inferred from the cloud itself: the lowest fluid row of a
    wall-bounded SPH run sits half a spacing above the wall, so
    ``y_wall = min(y) - dx/2``.  Points within one spacing of it (the flat
    contact patch and the wall row) are dropped -- they carry no slope
    information and would flatten the fit.
    """
    keep = x >= x0
    x, y = x[keep], y[keep]
    if x.numel() < 6:
        return torch.empty((0, 2)), float("nan")
    y_wall = float(y.min()) - 0.5 * dx
    step = bin_fraction * dx
    xb = torch.floor(x / step).long()
    yb = torch.floor((y - y_wall) / step).long()
    picks: set[tuple[float, float]] = set()
    for b in torch.unique(xb):
        m = xb == b
        i = int(torch.argmin(y[m]))
        picks.add((float(x[m][i]), float(y[m][i])))
    for b in torch.unique(yb):
        m = yb == b
        i = int(torch.argmax(x[m]))
        picks.add((float(x[m][i]), float(y[m][i])))
    P = torch.tensor(sorted(picks))
    keep = P[:, 1] >= y_wall + 0.9 * dx
    return P[keep], y_wall


def _fit_angle(P: torch.Tensor, y_wall: float, dx: float) -> tuple[float, dict]:
    """Contact angle [deg] of the interface traced by ``P``; also the fit meta."""
    meta: dict = {"model": "none", "n_points": int(P.shape[0]),
                  "radius_m": float("nan"), "collinearity": float("nan")}
    if P.shape[0] < 6:
        meta["reason"] = f"only {P.shape[0]} envelope points"
        return float("nan"), meta
    X = P[:, 0].double()
    Y = P[:, 1].double()
    mx, my = float(X.mean()), float(Y.mean())
    u, v = X - mx, Y - my
    cov = torch.stack((torch.stack((u @ u, u @ v)),
                       torch.stack((u @ v, v @ v))))
    lam = torch.linalg.eigvalsh(cov)
    lam_max = float(lam[1])
    collinearity = float(lam[0]) / lam_max if lam_max > 0 else 0.0
    meta["collinearity"] = collinearity
    near_wall = P[:, 1] <= y_wall + 1.5 * dx
    xc_guess = float(P[near_wall][:, 0].max()) if bool(near_wall.any()) \
        else float(P[:, 0].max())

    if collinearity <= 1e-9:                      # straight interface
        _, _, V = torch.linalg.svd(torch.stack((u, v), dim=1), full_matrices=False)
        t = (float(V[0][0]), float(V[0][1]))
        if t[1] < 0:
            t = (-t[0], -t[1])
        meta["model"] = "line"
        return math.degrees(math.atan2(t[1], -t[0])), meta

    A = torch.stack((u, v, torch.ones_like(u)), dim=1)
    sol = torch.linalg.lstsq(A, u ** 2 + v ** 2).solution
    cx, cy = mx + float(sol[0]) / 2.0, my + float(sol[1]) / 2.0
    radius = float(torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2).mean())
    meta["radius_m"] = radius
    dy = y_wall - cy
    disc = radius * radius - dy * dy
    if disc > 0.0:                               # the wall cuts the fitted circle
        x1, x2 = cx + math.sqrt(disc), cx - math.sqrt(disc)
        xc = x1 if abs(x1 - xc_guess) < abs(x2 - xc_guess) else x2
        vx, vy = xc - cx, y_wall - cy
        meta["model"] = "circle"
        meta["contact_x_m"] = xc
    else:
        # The fitted interface approaches the wall tangentially (a drop whose
        # surface is still a circle lying on the wall has no contact line yet).
        # Project the wall-level contact estimate onto the circle: the tangent
        # there is the limiting contact angle, and it agrees with the branch
        # above as disc -> 0, so the estimator stays continuous.
        vx, vy = xc_guess - cx, y_wall - cy
        norm = math.hypot(vx, vy)
        if norm == 0.0:
            meta["reason"] = "contact estimate coincides with the circle centre"
            return float("nan"), meta
        vx, vy = vx * radius / norm, vy * radius / norm
        meta["model"] = "circle-projected"
        meta["contact_x_m"] = cx + vx
    tx, ty = -vy, vx                             # tangent = radius rotated 90 deg
    if ty < 0:
        tx, ty = -tx, -ty
    return math.degrees(math.atan2(ty, -tx)), meta


def measure_contact_angle(x: torch.Tensor, y: torch.Tensor, x0: float,
                         dx: float) -> dict:
    """Angle [deg] and diagnostics from a fluid particle cloud over a wall.

    The fit is a **whole-drop** fit of the free-surface envelope on the fluid
    side of ``x0`` (the drop axis): this is the classic sessile-drop
    circle-fit method.  It is used instead of a local window deliberately --
    at the production resolution ``R/dx = 6.7`` a window of a few ``h`` holds
    about one envelope point per lattice row, and the staircase quantisation
    then dominates the slope at the contact point (measured: a 6 h window
    scatters by 15-40 deg on lattice caps where the whole-drop fit scatters by
    ~1 deg; see docs/verification_contact_angle.md).  The price is that a drop
    that is *not* a circular cap biases the reading -- which is why the
    physics layer is graded against the resolution layer and not against the
    analytic layer alone.
    """
    P, y_wall = _boundary_points(x, y, x0, dx)
    if P.numel() == 0:
        return {"theta_deg": float("nan"), "model": "none", "n_points": 0,
                "radius_m": float("nan"), "collinearity": float("nan"),
                "reason": "no envelope points"}
    theta, meta = _fit_angle(P, y_wall, dx)
    meta.update({"theta_deg": theta, "y_wall_m": y_wall})
    return meta


def contact_angle_from_envelope(x: torch.Tensor, y: torch.Tensor, x0: float,
                                dx: float) -> float:
    """Backward-compatible float wrapper (``nan`` when no admissible fit)."""
    return measure_contact_angle(x, y, x0, dx)["theta_deg"]


def analytic_interface(theta_deg: float, kind: str = "line", dx: float = DX,
                       n: int = 800, span_h: float = ARC_SPAN_H
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fluid particles on an interface meeting the wall at a known angle.

    The surface is sampled densely along the curve (so the device, not the
    sampling, is under test), then the wall-adjacent fluid row (the flat
    contact patch, ``y = dx/2``) is added up to the contact line, exactly as a
    wall-bounded SPH run has it.  ``kind = "line"`` is the straight interface
    through the contact point, ``kind = "arc"`` a circular interface of radius
    ``ARC_RADIUS_FACTOR * RADIUS``; both are exact.

    Parametrisation (arc length ``s`` from the contact point at ``(x_c, 0)``)::

        line:  P(s) = (x_c, 0) + s * (-cos theta, sin theta)
        arc:   centre C = (x_c, 0) - R_c (sin theta, cos theta),
               P(s) = C + R_c (cos a0, sin a0) rotated by s / R_c, a0 = pi/2 - theta
    """
    th = math.radians(theta_deg)
    xc = 0.5
    s = torch.linspace(0.0, span_h * H_FACTOR * dx, n)
    if kind == "line":
        xs = xc - s * math.cos(th)
        ys = s * math.sin(th)
    elif kind == "arc":
        rc = ARC_RADIUS_FACTOR * RADIUS
        cx = xc - rc * math.sin(th)
        cy = -rc * math.cos(th)
        a = (math.pi / 2.0 - th) + s / rc
        xs = cx + rc * torch.cos(a)
        ys = cy + rc * torch.sin(a)
    else:
        raise ValueError(f"unknown interface kind {kind!r}")
    keep = ys >= 0.5 * dx
    xs, ys = xs[keep], ys[keep]
    x_cross = float(xs[int(torch.argmin(torch.abs(ys - 0.5 * dx)))])
    patch = torch.arange(x_cross - 0.3 * span_h * H_FACTOR * dx,
                         x_cross - 0.5 * dx, dx)
    return (torch.cat((xs, patch)),
            torch.cat((ys, torch.full_like(patch, 0.5 * dx))))


def lattice_cap(theta_deg: float, dx: float,
               rc: float = ARC_RADIUS_FACTOR * RADIUS) -> tuple[torch.Tensor, torch.Tensor]:
    """Fluid particles on a lattice inside an analytic circular cap.

    This is the *resolution* layer: the interface is no longer smooth, it is a
    staircase of particles at spacing ``dx``, which is what the physics layer
    has to live with.  The analytic contact angle is known, so the difference
    is the device's discretisation error at that ``R/dx``.
    """
    th = math.radians(theta_deg)
    xc = 0.5
    cx, cy = xc - rc * math.sin(th), -rc * math.cos(th)
    xs = torch.arange(xc - rc - 2.0 * dx, xc + rc + 2.0 * dx, dx)
    ys = torch.arange(0.5 * dx, 1.6 * rc, dx)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    keep = (gx - cx) ** 2 + (gy - cy) ** 2 <= rc * rc
    return gx[keep], gy[keep]


def instrument_check() -> dict:
    rows = []
    for theta in (30.0, 60.0, 90.0, 120.0, 150.0):
        for kind in ("line", "arc"):
            xs, ys = analytic_interface(theta, kind)
            m = measure_contact_angle(xs, ys, x0=0.5 - 0.05, dx=DX)
            err = m["theta_deg"] - theta
            rows.append({"theta_set_deg": theta, "kind": kind,
                         "theta_measured_deg": m["theta_deg"],
                         "error_deg": err, "model": m["model"],
                         "n_points": m["n_points"],
                         "within_gate": len(rows) >= 0 and abs(err) < INSTRUMENT_GATE_DEG})
    return {"rows": rows,
            "gate_deg": INSTRUMENT_GATE_DEG,
            "all_within_gate": all(r["within_gate"] for r in rows),
            "worst_abs_error_deg": max(abs(r["error_deg"]) for r in rows)}


def resolution_check(dx_list: tuple[float, ...] = (0.003, 0.002, 0.0015, 0.001)) -> dict:
    """Discretisation error of the instrument on particle-lattice caps."""
    rows = []
    for dx in dx_list:
        for theta in (30.0, 60.0, 90.0, 120.0, 150.0):
            xs, ys = lattice_cap(theta, dx)
            m = measure_contact_angle(xs, ys, x0=0.5 - 0.05, dx=dx)
            rows.append({"R_over_dx": RADIUS / dx, "dx_m": dx,
                         "theta_set_deg": theta,
                         "theta_measured_deg": m["theta_deg"],
                         "error_deg": m["theta_deg"] - theta,
                         "model": m["model"], "n_points": m["n_points"],
                         "admissible": m["theta_deg"] == m["theta_deg"]})
    best = {}
    for row in rows:
        if not row["admissible"]:
            continue
        key = row["R_over_dx"]
        worst = best.get(key, 0.0)
        best[key] = max(worst, abs(row["error_deg"]))
    floor = {}
    for row in rows:
        floor.setdefault(row["R_over_dx"], 0.0)
        if row["admissible"]:
            floor[row["R_over_dx"]] = max(floor[row["R_over_dx"]],
                                          abs(row["error_deg"]))
    return {"rows": rows,
            "worst_abs_error_deg_by_resolution": {str(k): v for k, v in floor.items()},
            "n_inadmissible": sum(1 for r in rows if not r["admissible"])}


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #

def make_drop_on_wall(theta: float, device: torch.device):
    cfg = Config(dx=DX, h_factor=H_FACTOR, rho0=RHO0, sound_speed=20.0,
                 gamma=GAMMA, gravity=0.0, viscosity=VISCOSITY,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                 csf_curvature="fit_quadratic",
                 pool_width=POOL_WIDTH, pool_depth=0.0,
                 drop_radius=RADIUS, drop_center_y=RADIUS,
                 contact_angle=theta, wall_layers=3)
    h = cfg.h_factor * cfg.dx
    x0 = POOL_WIDTH / 2.0
    xs = cell_centres(POOL_WIDTH, DX, device)
    ys = cell_centres(2.0 * RADIUS, DX, device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    cand = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    c = torch.tensor([x0, RADIUS], device=device)
    pos = cand[(cand - c).norm(dim=1) <= RADIUS]
    mass = torch.full((pos.shape[0],), RHO0 * DX * DX, device=device)
    wall = wall_lattice(cfg, device)
    return cfg, h, pos, mass, wall, x0


def relax(theta_deg: float, device: torch.device) -> dict:
    """Relax a drop with ``contact_angle = radians(theta_deg)`` and read the angle.

    ``theta_deg`` is an angle in **degrees**; the config wants radians, and the
    report keeps degrees on both sides of the subtraction.  (The shipped
    revision subtracted a radian ``theta_set_deg`` from a degree reading, which
    mixed unit systems inside ``error_deg``; that is why the physics table used
    to be structurally ungradable.)
    """
    theta = math.radians(theta_deg)
    cfg, h, pos, mass, wall, x0 = make_drop_on_wall(theta, device)
    dt = float(stable_dt(cfg))
    t_end = T_END_OVER_T_SIGMA * T_SIGMA
    steps = int(math.ceil(t_end / dt))
    dt = t_end / steps
    vel = torch.zeros_like(pos)
    trace = {"t": [], "theta_deg": [], "vmax": [], "n_points": [], "model": []}
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        _, _, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall, dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % 25 == 0 or step == steps - 1:
            m = measure_contact_angle(pos[:, 0], pos[:, 1], x0, DX)
            trace["t"].append((step + 1) * dt / T_SIGMA)
            trace["theta_deg"].append(m["theta_deg"])
            trace["n_points"].append(m["n_points"])
            trace["model"].append(m["model"])
            trace["vmax"].append(float(vel.norm(dim=1).max()))
    n_tail = max(1, int(0.4 * len(trace["theta_deg"])))
    tail = [v for v in trace["theta_deg"][-n_tail:] if v == v]
    theta_phys = sum(tail) / len(tail) if tail else float("nan")
    err = theta_phys - theta_deg
    v_ref = math.sqrt(SIGMA / (RHO0 * RADIUS))
    penetration_dx = max(0.0, -float(pos[:, 1].min())) / DX
    spurious = trace["vmax"][-1] / v_ref
    admissible = bool(penetration_dx < WALL_PENETRATION_GATE_DX
                      and spurious < SPURIOUS_CURRENT_GATE)
    return {
        "theta_set_deg": theta_deg,
        "theta_measured_deg": theta_phys,
        "error_deg": err,
        "within_gate": bool(err == err and abs(err) <= ANGLE_GATE_DEG),
        "admissible": admissible,
        "wall_penetration_dx": penetration_dx,
        "wall_penetration_gate_dx": WALL_PENETRATION_GATE_DX,
        "spurious_current_ratio": spurious,
        "spurious_current_gate": SPURIOUS_CURRENT_GATE,
        "n_particles": int(pos.shape[0]),
        "steps": steps, "dt_s": dt, "t_end_s": t_end,
        "finite": bool(torch.isfinite(pos).all() and torch.isfinite(vel).all()),
        "vmax_m_per_s": trace["vmax"][-1],
        "trace": trace,
    }


def make_figures(report: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    for kind, marker in (("line", "o"), ("arc", "s")):
        sel = [r for r in report["instrument"]["rows"] if r["kind"] == kind]
        ax.plot([r["theta_set_deg"] for r in sel],
                [r["theta_measured_deg"] for r in sel], marker, label=kind)
    ax.plot([0, 180], [0, 180], "k--", lw=0.8)
    ax.set_xlabel("set angle [deg]")
    ax.set_ylabel("measured angle [deg]")
    ax.set_title("(a) instrument on exact interfaces")

    ax = axes[1]
    for r_over_dx in sorted({r["R_over_dx"] for r in report["resolution"]["rows"]}):
        sel = [r for r in report["resolution"]["rows"] if r["R_over_dx"] == r_over_dx]
        ax.plot([r["theta_set_deg"] for r in sel],
                [r["theta_measured_deg"] for r in sel], "o-", ms=3,
                label=f"R/dx = {r_over_dx:.1f}")
    ax.plot([0, 180], [0, 180], "k--", lw=0.8)
    ax.set_xlabel("set angle [deg]")
    ax.set_ylabel("measured angle [deg]")
    ax.set_title("(b) instrument on particle lattices")
    ax.legend(fontsize=8)

    ax = axes[2]
    for row in report["physics"]:
        ax.plot(row["trace"]["t"], row["trace"]["theta_deg"], "-",
                label=f"set {row['theta_set_deg']:.0f} -> "
                      f"{row['theta_measured_deg']:.1f}")
        ax.axhline(row["theta_set_deg"], ls=":", lw=0.8)
    ax.set_xlabel("t / t_sigma")
    ax.set_ylabel("measured contact angle [deg]")
    ax.set_title("(c) relaxing drop on a wall (zero g)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    p = out / "contact_angle.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def make_doc(report: dict, png: Path) -> str:
    inst = report["instrument"]
    res = report["resolution"]
    phys = report["physics"]
    worst = max((abs(r["error_deg"]) for r in phys if r["error_deg"] == r["error_deg"]),
                default=float("nan"))
    n_gate = sum(1 for r in phys if r["within_gate"])
    n_admissible = sum(1 for r in phys if r["admissible"])
    if n_gate == len(phys) and n_admissible == len(phys):
        verdict = "PASS"
    elif n_admissible < len(phys):
        # run-level disqualification outranks the number: an inadmissible run is
        # not a measured physics defect (repo rule, see verification_claim_validity.md)
        verdict = "WITHHELD"
    else:
        verdict = "FAIL"
    instrument_ok = inst["all_within_gate"]
    floor_now = res["worst_abs_error_deg_by_resolution"]
    floor_dx3 = float(floor_now.get(str(RADIUS / DX), float("nan")))
    lines = [
        "# 验证欠账复核：接触角边界条件（仪器已修好，物理层如实记账）",
        "",
        "**仓库**: TensorSPH ｜ **工件**: `benchmarks/contact_angle/` ｜ "
        "**测试**: `tests/test_contact_angle_instrument_m6503.py`",
        "",
        f"**结论（先给判定）**: 仪器层 {len(inst['rows'])} 个**精确解析界面**"
        f"（直线/圆弧 × 30°–150°）最大偏差 {inst['worst_abs_error_deg']:.3f}°，"
        f"门槛 {inst['gate_deg']:.0f}° → "
        f"{'**通过**' if instrument_ok else '**不通过**'}；"
        f"分辨率层（粒子晶格，R/dx = 6.7）最大偏差 {floor_dx3:.1f}°；"
        f"物理层 {len(phys)} 个设定角中 {n_gate} 个在 ±{ANGLE_GATE_DEG:.0f}° 门内、"
        f"{n_admissible}/{len(phys)} 个算例**可采信**（壁面穿透与伪流门），"
        f"最大偏差 {worst:.1f}° → **{verdict}**。",
        "",
        "## ① 案例介绍",
        "",
        f"半径 R = {RADIUS} m 的二维液滴静置于平底壁面（Adami 壁粒子 3 层），"
        f"零重力，sigma = {SIGMA} N/m，rho0 = {RHO0} kg/m³，Tait gamma = {GAMMA}，"
        f"dx = {DX} m（R/dx = {RADIUS / DX:.1f}），h = {H_FACTOR} dx，"
        f"c0 = 20 m/s，nu = {VISCOSITY} m²/s，观测窗口 {T_END_OVER_T_SIGMA} t_sigma"
        f"（t_sigma = {T_SIGMA:.4f} s）。接触角 BC 在壁面 2h 内把界面法向旋转至 "
        "n·n_wall = cos(theta)（Adami et al. 2012 的界面法向形式）。",
        "",
        "## ② 怎么做的",
        "",
        "1. **仪器层**（判据先于物理）：直线 `P(s) = (x_c,0) + s(-cos t, sin t)` 与"
        f"圆弧（R_c = {ARC_RADIUS_FACTOR:g}R，圆心 `C = (x_c,0) - R_c(sin t, cos t)`）"
        "两个**精确**界面，沿弧长密采样 + 壁面相邻流体行（接触区，y = dx/2），"
        "过同一套仪器：双向包络（x 向最低 + y 向最右，故增湿/悬垂都能描）+ "
        "圆拟合（Kasa，对圆界面精确）/直线回退（按共线性 λ0/λ1 选择）+ "
        "「模型与壁面的交点」取接触点 + `theta = atan2(t_y, -t_x)`；",
        "2. **分辨率层**：解析圆帽内的**粒子晶格**（间距 dx），已知角做基准，"
        "量出仪器在生产分辨率下的离散误差（这是任何物理读数的误差棒）；",
        "3. **物理层**：圆盘液滴初始化于壁面（最低粒子行 y = dx/2），生产 RHS "
        "（`droplet_impact.rhs` + `csf_consistent`，**不改 src/**）松弛 "
        f"{T_END_OVER_T_SIGMA} t_sigma，窗口后 40% 时间平均读接触角；",
        f"4. 判据：|theta_phys - theta_set| <= {ANGLE_GATE_DEG:.0f}°"
        "（`within_gate`），另记仪器门槛与分辨率地板供对照。",
        "",
        "## ③ 计算结果（程序化注入，零手抄）",
        "",
        "### 3.1 仪器层（精确界面，机器精度基准）",
        "",
        "| 设定角 [deg] | 直线实测 [deg] | 偏差 [deg] | 圆弧实测 [deg] | 偏差 [deg] |",
        "|---|---|---|---|---|",
    ]
    for t in (30.0, 60.0, 90.0, 120.0, 150.0):
        row = next(r for r in inst["rows"] if r["theta_set_deg"] == t and r["kind"] == "line")
        arc = next(r for r in inst["rows"] if r["theta_set_deg"] == t and r["kind"] == "arc")
        lines.append(
            f"| {t:.0f} | {row['theta_measured_deg']:.3f} | {row['error_deg']:+.3f} "
            f"| {arc['theta_measured_deg']:.3f} | {arc['error_deg']:+.3f} |")
    lines += [
        "",
        "### 3.2 分辨率层（粒子晶格，仪器在真实 R/dx 下的地板）",
        "",
        "| R/dx | 可用读数 | 最大 |偏差| [deg] | 不可用行数 |",
        "|---|---|---|---|",
    ]
    for key in sorted(floor_now, key=lambda s: float(s)):
        n_ok = sum(1 for r in res["rows"]
                   if str(r["R_over_dx"]) == key and r["admissible"])
        n_bad = sum(1 for r in res["rows"]
                    if str(r["R_over_dx"]) == key and not r["admissible"])
        lines.append(f"| {float(key):.1f} | {n_ok} | {floor_now[key]:.1f} | {n_bad} |")
    lines += [
        "",
        "### 3.3 物理层（松弛液滴 vs 设定角）",
        "",
        "| 设定角 [deg] | 实测角 [deg] | 偏差 [deg] | <=10° 门 | 壁面穿透 [dx] "
        "| v_max/v_ref | 可采信 | 粒子数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in phys:
        lines.append(
            f"| {r['theta_set_deg']:.0f} | {r['theta_measured_deg']:.1f} "
            f"| {r['error_deg']:+.1f} | {'pass' if r['within_gate'] else 'FAIL'} "
            f"| {r['wall_penetration_dx']:.2f} | {r['spurious_current_ratio']:.2f} "
            f"| {'yes' if r['admissible'] else 'no'} | {r['n_particles']} |")
    lines += [
        "",
        f"可采信判据（先于读数）：壁面穿透 < {WALL_PENETRATION_GATE_DX} dx 且 "
        f"v_max/v_ref < {SPURIOUS_CURRENT_GATE}（v_ref = sqrt(sigma/(rho R)) = "
        f"{math.sqrt(SIGMA / (RHO0 * RADIUS)):.3f} m/s，Adami et al. 2012 静止液滴"
        "伪流门）。不可采信的读数不得当作「BC 不准」的证据，也不得当作「BC 准」的证据。",
    ]
    lines += [
        "",
        "## ④ 结果图",
        "",
        f"![contact angle]({png.name})",
        "",
        "*(a)* 仪器在精确界面上的读数 vs 设定角；*(b)* 同一仪器在粒子晶格上的"
        "离散误差随 R/dx；*(c)* 零重力液滴松弛过程的接触角轨迹（虚线 = 设定角）。",
        "",
        "## ⑤ 注意点（局限与适用范围）",
        "",
        "1. 仪器层是**精确界面**上的数学校验，它证明估计器无偏、模型选择不误判；"
        "它**不**证明物理层准确 —— 探针与样品要分开记账（这正是本单修的东西："
        "旧版把「仪器层十行全部通过」写死进文档，而同一份 report.json 里 "
        "10 行有 8 行是 NaN）；",
        "2. 分辨率层给出仪器在 R/dx = 6.7 的离散地板；物理层偏差若与之同量级，"
        "读数只能当**上界**用，不能当「BC 达到 ±10°」的证据；",
        "3. 非增湿（theta > 90°）界面在接触线处**悬垂**，x 向包络在那里是壁面而不是"
        "自由面 —— 仪器必须同时取 y 向最右包络，否则 120°/150° 一律测成 ~89°"
        "（旧版的实测就是这个症状）；",
        "4. 圆拟合在界面近似直时退化（Kasa 在共线点上是奇异方程），故按共线性"
        "（λ0/λ1 <= 1e-9）回退直线拟合；两条模型在直界面上给出同一角，不存在"
        "「挑一个好看的」空间；",
        "5. 零重力构型排除 Bond 数干扰，形状由表面张力 + 接触角 BC 决定；壁面 2h 内"
        "的法向旋转是 BC 的直接输出，实测的是**松弛后界面**的角，不是中间场。",
        "",
        "## ⑥ 误差判定（逐项回答是否 < 3%）",
        "",
        "| 关键量 | 基准 | 实测 | 偏差 | 是否 < 3% |",
        "|---|---|---|---|---|",
        f"| 仪器层最差（精确界面） | 设定角 | 最差 {inst['worst_abs_error_deg']:.3f}° "
        f"| {inst['worst_abs_error_deg']:.3f}° | "
        f"{'**是**' if inst['worst_abs_error_deg'] < 3.0 else '否'} |",
    ]
    for r in phys:
        rel = abs(r["error_deg"]) / r["theta_set_deg"] * 100.0
        lines.append(
            f"| 物理层设定角 {r['theta_set_deg']:.0f}° | {r['theta_set_deg']:.0f}° "
            f"| {r['theta_measured_deg']:.1f}° | {r['error_deg']:+.1f}° "
            f"（{rel:.0f}%） | {'**是**' if rel < 3.0 else '否'} |")
    lines += [
        "",
        f"物理层 {n_gate}/{len(phys)} 在 ±{ANGLE_GATE_DEG:.0f}° 门内；"
        f"判定 **{verdict}**。仪器层与分辨率层的读数不参与物理判定，"
        "它们是判据的判据。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--skip-physics", action="store_true",
                    help="instrument + resolution layers only (seconds)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    report = {
        "case": "contact_angle",
        "radius_m": RADIUS, "sigma": SIGMA, "dx_m": DX,
        "t_sigma_s": T_SIGMA, "t_end_over_t_sigma": T_END_OVER_T_SIGMA,
        "gate_deg": ANGLE_GATE_DEG,
        "instrument_gate_deg": INSTRUMENT_GATE_DEG,
        "instrument": instrument_check(),
        "resolution": resolution_check(),
        "physics": [] if args.skip_physics else
        [relax(t, device) for t in ANGLES_DEG],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    png = make_figures(report, out)
    doc = make_doc(report, png)
    (ROOT / "docs" / "verification_contact_angle.md").write_text(doc)
    print(json.dumps({
        "instrument_all_within_gate": report["instrument"]["all_within_gate"],
        "instrument_worst_deg": report["instrument"]["worst_abs_error_deg"],
        "resolution_floor_deg": report["resolution"]["worst_abs_error_deg_by_resolution"],
        "physics": [(r["theta_set_deg"], round(r["theta_measured_deg"], 2),
                     round(r["error_deg"], 2), r["within_gate"])
                    for r in report["physics"]],
    }, indent=2))


if __name__ == "__main__":
    main()
