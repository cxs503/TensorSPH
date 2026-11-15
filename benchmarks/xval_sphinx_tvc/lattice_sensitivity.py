"""Is the free-surface TVC anchor *lattice independent*?

Why this file exists
--------------------
``docs/xval_tvc_surface_anchor.md`` graded the free-surface branch of the
transport-velocity-correction operator against the exact continuum limit

    g_i -> 2 I(h, d) n,     I(h, d) = int W_2D(s, d) ds

and reported, for the production lattice (``h/dx = 1.3``), a **14.15 %**
deficit, i.e. the ``< 15 %`` gate *passes with 0.85 points of margin*.  Every
number in that leg came from **one** particle arrangement: a square lattice
with the surface row exactly on the plane.  A production SPH run does not look
like that after the first few hundred steps -- advection shears and jitters the
distribution -- so "the anchor is met" is only as broad as the lattice family
it was measured on.

This script widens the family and re-measures the same anchor:

* ``square``   -- the arrangement of the existing anchor (regression leg);
* ``hex``      -- the densest 2-D packing, rows ``sqrt(3)/2 s`` apart, equal
                  volume per particle ``sqrt(3)/2 s^2`` (so ``h/dx_eff`` is
                  comparable across lattices, ``dx_eff = sqrt(V)``);
* ``rot``      -- the square lattice rotated against the surface normal
                  (``15/30/45`` deg): the truncation of the kernel support by
                  the free-surface plane is *anisotropic*, so a rotation probes
                  the worst orientation rather than the friendliest one;
* ``jitter``   -- the square lattice displaced by uniform noise of amplitude
                  ``zeta dx`` (ensemble over seeds, ``zeta = 0.05 / 0.15 /
                  0.30``), which is what an irreproducible but *representative*
                  SPH distribution looks like.

Because a jittered/rotated lattice has no single "surface particle", the anchor
is evaluated **per particle** at that particle's own depth ``d_i``
(``I'(0) = 0``, so the anchor is smooth there and the per-particle comparison
is well conditioned), and the statistics reported are over all particles within
half a spacing of the free surface.

Reported quantities (all measured, none assumed)
------------------------------------------------
``mean_rel``   mean of ``(|g_i| - a_i) / a_i``  (signed -> bias direction)
``mae_rel``    mean of ``| |g_i| - a_i | / a_i``
``p95_rel``    95th percentile of ``| |g_i| - a_i | / a_i``
``cos``        mean cosine between ``g_i`` and ``+n`` (direction fidelity)
``bulk_rel``   ``|g| / a`` for particles deeper than ``2h`` (must be ~0)

Usage
-----
    python benchmarks/xval_sphinx_tvc/lattice_sensitivity.py \
        --out docs/xval_sphinx_tvc/data --figure docs/xval_sphinx_tvc/figure_lattice_sensitivity.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.transport_velocity_correction import (  # noqa: E402
    kernel_gradient_integral,
)
from surface_term import (  # noqa: E402
    HBAR,
    check_normalisation,
    closed_form_I0,
    kernel_2d,
    line_integral,
)

#: production h/dx of this repo (``surface_term.HBAR``); the x-ladder uses it
#: only as the coarsest rung because the whole point is its margin at 14.15 %.
LADDER = (HBAR, 2.0, 4.0, 8.0)


# --------------------------------------------------------------------------- #
# lattice construction: fluid half space x2 <= 0
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Lattice:
    positions: torch.Tensor   # (N, 2), all with x2 <= 0 (up to jitter)
    volume: float             # equal per-particle measure
    dx_eff: float             # sqrt(area per particle)
    surface: torch.Tensor     # (M,) bool mask of free-surface particles
    depth_box: float          # depth of the box floor below the free surface
    margin: float             # half-width of the fully-resolved interior strip

    @property
    def h_over_dx(self) -> float:  # filled by the builder
        return float(self._hbar)

    _hbar: float = 0.0


def _in_box(x1: float, x2: float, half_width: float, depth: float) -> bool:
    """Inside the box that resolves the support of every *reported* particle.

    ``half_width``/``depth`` are ``2h + margin``: the reported region is the
    ``margin``-wide strip around the origin, so each reported particle sees at
    least ``2h`` of lattice on every side and the kernel support is never
    clipped.  Clipping it would subtract real neighbours and masquerade as a
    surface deficit -- the exact artefact this probe must not create.
    """
    return abs(x1) <= half_width and -x2 <= depth


def build_lattice(
    kind: str,
    h: float,
    h_over_dx: float,
    *,
    margin_dx: float = 4.0,
    rotation_deg: float = 0.0,
    jitter: float = 0.0,
    seed: int = 0,
) -> Lattice:
    """Half-space lattice with ``sqrt(area per particle) = h / h_over_dx``.

    The reported (free-surface) region is ``|x1| <= margin_dx * dx_eff`` with
    ``margin_dx = 4``; the box extends ``2h`` further in every direction, which
    is the kernel support.
    """
    if kind not in ("square", "hex"):
        raise ValueError(f"unknown lattice kind {kind!r}")
    dx_eff = h / h_over_dx
    margin = margin_dx * dx_eff
    half_width = 2.0 * h + margin
    # bottom: the reported strip plus 2h of support *below the deepest bulk
    # particle*, so the bulk fixed-point check is not contaminated by the box
    # floor (a clipped floor would fake a gradient there).
    depth = 4.0 * h + 2.0 * margin

    if kind == "square":
        dx = dx_eff
        volume = dx * dx
        row_h = dx
        offset_odd = 0.0
    else:  # hexagonal: nearest spacing s, row height sqrt3/2 s, area sqrt3/2 s^2
        volume = dx_eff * dx_eff
        s = math.sqrt(2.0 * volume / math.sqrt(3.0))
        dx = s
        row_h = s * math.sqrt(3.0) / 2.0
        offset_odd = 0.5 * s

    n_col = int(math.ceil(half_width / dx)) + 1
    n_row = int(math.ceil(depth / row_h)) + 1

    coords = []
    for k in range(0, n_row + 1):
        x2 = -k * row_h
        shift = offset_odd if (k % 2) else 0.0
        for m in range(-n_col, n_col + 1):
            x1 = m * dx + shift
            if _in_box(x1, x2, half_width, depth):
                coords.append((x1, x2))

    pos = torch.tensor(coords, dtype=torch.float64)
    if rotation_deg:
        th = math.radians(rotation_deg)
        c, s_ = math.cos(th), math.sin(th)
        rot = torch.tensor([[c, -s_], [s_, c]], dtype=torch.float64)
        pos = pos @ rot.T
        keep = pos[:, 1] <= 1.0e-12           # keep the half space below the plane
        pos = pos[keep]
    if jitter:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise = (torch.rand(pos.shape, generator=gen, dtype=torch.float64) - 0.5)
        pos = pos + (jitter * dx_eff) * 2.0 * noise

    # a free-surface particle is one within half a spacing of the plane, and it
    # is reported only if its whole 2h support lies inside the box.
    surface = (pos[:, 1] > -0.5 * dx_eff) & (pos[:, 0].abs() <= margin)
    return Lattice(positions=pos.contiguous(), volume=volume, dx_eff=dx_eff,
                   surface=surface, _hbar=h_over_dx, depth_box=depth,
                   margin=margin)


def line_integrals(h: float, depths: torch.Tensor, n: int = 4001) -> torch.Tensor:
    """Vectorised ``I(h, d) = int W_2D(s, d) ds`` for a tensor of depths.

    Same trapezoid quadrature (and same node count) as
    ``surface_term.line_integral``, batched so that a few thousand surface
    particles cost one pass instead of a Python loop.
    """
    d = depths.clamp(min=0.0)
    ratio = (d / (2.0 * h)).clamp(max=1.0)
    span = 2.0 * h * torch.sqrt(torch.clamp(1.0 - ratio * ratio, min=0.0))
    t = torch.linspace(-1.0, 1.0, n, dtype=depths.dtype)
    s = span[:, None] * t[None, :]
    r = torch.sqrt(s * s + (d[:, None]) * (d[:, None]))
    w = kernel_2d(r, h)
    return (w[:, 1:] + w[:, :-1]).sum(dim=1) * 0.5 * (2.0 / (n - 1)) * span


def measure_lattice(lat: Lattice, h: float) -> dict:
    """Per-particle comparison of ``g_i`` with ``2 I(h, d_i) n`` on one lattice."""
    pos, vol = lat.positions, lat.volume
    pairs = cell_linked_pairs(pos, 2.0 * h)
    g = kernel_gradient_integral(pos, torch.full((pos.shape[0],), vol,
                                                dtype=pos.dtype), h, pairs=pairs)
    conv = 2.0 * torch.tensor([line_integral(h, 0.0)], dtype=pos.dtype)
    mag = torch.linalg.vector_norm(g, dim=1)
    surf = lat.surface
    if not bool(surf.any()):
        raise ValueError("no resolved free-surface particle in this box")
    d = torch.clamp(-pos[surf, 1], min=0.0)
    a = 2.0 * line_integrals(h, d)
    ms = mag[surf]
    good = a > 1.0e-2 * float(conv)      # particle inside 2h of the plane
    if not bool(good.any()):
        raise ValueError("anchor vanishes for every surface particle")
    rel = (ms[good] - a[good]) / a[good]
    cos = g[surf, 1][good] / torch.clamp(ms[good], min=1.0e-300)
    # a bulk particle is graded only if its *whole* 2h support is box-resolved:
    # deeper than 2h below the surface and 2h inside the floor and the side walls
    deep = ((pos[:, 1] < -2.0 * h)
            & (pos[:, 1] > -(lat.depth_box - 2.0 * h))
            & (pos[:, 0].abs() <= lat.margin))
    bulk = (mag[deep] / conv) if bool(deep.any()) else torch.zeros(1,
                                                                  dtype=pos.dtype)
    return {
        "n_particles": int(pos.shape[0]),
        "n_surface": int(good.sum()),
        "n_deep": int(deep.sum()),
        "dx_eff": lat.dx_eff,
        "mean_rel": float(rel.mean()),
        "mae_rel": float(rel.abs().mean()),
        "p95_rel": float(rel.abs().quantile(0.95)),
        "worst_rel": float(rel.max()),
        "best_rel": float(rel.min()),
        "cos": float(cos.mean()),
        "bulk_rel": float(bulk.mean()),
        "anchor_scale": float(a[good].mean()),
        "depth_min": float(d[good].min()),
        "depth_max": float(d[good].max()),
        # |dr| for alpha = 0.10 at the surface, in h units
        "shift_over_h_alpha_0p10": 0.10 * h * float(ms[good].mean()),
    }


# --------------------------------------------------------------------------- #
# the four probes
# --------------------------------------------------------------------------- #
def square_regression(h: float) -> list[dict]:
    rows = []
    for n in LADDER:
        lat = build_lattice("square", h, n)
        row = {"probe": "square", "h_over_dx": n, **measure_lattice(lat, h)}
        rows.append(row)
    return rows


def hex_ladder(h: float) -> list[dict]:
    rows = []
    for n in LADDER:
        lat = build_lattice("hex", h, n)
        row = {"probe": "hex", "h_over_dx": n, **measure_lattice(lat, h)}
        rows.append(row)
    return rows


def rotation_probe(h: float, h_over_dx: float = HBAR) -> list[dict]:
    rows = []
    for deg in (0.0, 15.0, 30.0, 45.0):
        lat = build_lattice("square", h, h_over_dx, rotation_deg=deg)
        row = {"probe": "rot", "rotation_deg": deg, "h_over_dx": h_over_dx,
               **measure_lattice(lat, h)}
        rows.append(row)
    return rows


def jitter_ensemble(h: float, h_over_dx: float = HBAR, seeds=range(24)) -> list[dict]:
    rows = []
    for zeta in (0.05, 0.15, 0.30):
        reps = []
        for seed in seeds:
            lat = build_lattice("square", h, h_over_dx, jitter=zeta, seed=int(seed))
            reps.append(measure_lattice(lat, h))
        mean_rel = [r["mean_rel"] for r in reps]
        mae = [r["mae_rel"] for r in reps]
        p95 = [r["p95_rel"] for r in reps]
        cm = [r["cos"] for r in reps]
        bm = [r["bulk_rel"] for r in reps]
        rows.append({
            "probe": "jitter",
            "h_over_dx": h_over_dx,
            "zeta": zeta,
            "n_seeds": len(reps),
            # ensemble statistics over independent realisations
            "mean_rel": sum(mean_rel) / len(mean_rel),
            "mean_rel_sd": (sum((x - sum(mean_rel) / len(mean_rel)) ** 2
                                for x in mean_rel) / max(len(mean_rel) - 1, 1)) ** 0.5,
            "mae_rel": sum(mae) / len(mae),
            "mae_rel_sd": (sum((x - sum(mae) / len(mae)) ** 2
                               for x in mae) / max(len(mae) - 1, 1)) ** 0.5,
            "p95_rel": sum(p95) / len(p95),
            "cos": sum(cm) / len(cm),
            "bulk_rel": sum(bm) / len(bm),
            "n_surface": int(sum(r["n_surface"] for r in reps) / len(reps)),
            "shift_over_h_alpha_0p10": sum(r["shift_over_h_alpha_0p10"]
                                           for r in reps) / len(reps),
        })
    return rows


# --------------------------------------------------------------------------- #
# wall branch on a hexagonal lattice: the exact mirror-image anchor
# --------------------------------------------------------------------------- #
def wall_hex_case(h: float, h_over_dx: float, wall_factor: float = 2.0) -> dict:
    """``|g| = 0`` to round-off iff ``wall_factor = 2`` -- on a hex lattice too.

    Construction (and why the naive mirror image is *wrong*): the exact anchor
    used for the square lattice mirrors the fluid rows ``(k+1/2)dx`` to
    ``-(k+1/2)dx``.  The union is then the full square lattice and the target
    is one of its sites, so the sum over the full lattice cancels pair by pair.
    Doing the same with a **triangular** lattice does *not* reproduce the
    lattice: mirroring a hex row reverses the ``s/2`` stagger, so the union
    has rows ``0.866 s`` apart with identical offsets and the target is no
    longer a centre of symmetry -- the sum then fails to cancel *by geometry*,
    not by any error in the port (measured: 0.92 of the surface scale even at
    ``wall_factor = 2``; that is a bug in the probe, and the first run of this
    script had it).

    The correct construction builds the **full** triangular lattice, with rows
    at ``(n + 1/2) row_h`` and the ``s/2`` stagger alternating in ``n``, and
    splits it by the sign of ``x2``: ``x2 > 0`` is fluid, ``x2 < 0`` is wall.
    No site lies on the plane (so no singular pair), the union is exactly the
    triangular lattice, and every lattice site is a centre of symmetry -- the
    full-space sum vanishes identically.  A fluid pair is weighted
    ``C_i + C_j = 2`` and a wall pair ``wall_factor * C_i``, so all weights are
    equal **iff** ``wall_factor = 2`` and

        |g| = 0  to round-off,  if and only if  wall_factor = 2.

    Only fluid particles whose complete ``2h`` support is inside the box are
    graded.
    """
    dx_eff = h / h_over_dx
    volume = dx_eff * dx_eff
    s_len = math.sqrt(2.0 * volume / math.sqrt(3.0))
    row_h = s_len * math.sqrt(3.0) / 2.0
    reach = 4.0 * h + 4.0 * dx_eff   # graded slab (2h deep) + 2h support margins
    n_col = int(math.ceil(reach / s_len)) + 1
    n_row = int(math.ceil(reach / row_h)) + 1
    fluid, wall = [], []
    for n in range(-n_row, n_row + 1):
        x2 = (n + 0.5) * row_h
        shift = 0.5 * s_len if (n % 2) else 0.0
        for m in range(-n_col, n_col + 1):
            x1 = m * s_len + shift
            if abs(x1) <= reach and abs(x2) <= reach:
                (fluid if x2 > 0.0 else wall).append((x1, x2))
    fluid = torch.tensor(fluid, dtype=torch.float64)
    wall = torch.tensor(wall, dtype=torch.float64)
    pos = torch.cat([fluid, wall])
    vol = torch.full((pos.shape[0],), volume, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    g = kernel_gradient_integral(pos, vol, h, pairs=pairs,
                                 wall_count=wall.shape[0],
                                 wall_factor=wall_factor)
    mag = torch.linalg.vector_norm(g, dim=1)
    # graded particles: those that actually *see* the wall (closer to the plane
    # than the 2h support) and whose tangential support is fully inside the box.
    # A particle deeper than 2h has no wall pair at all and would grade nothing.
    graded = ((fluid[:, 1] > 0.0)
              & (fluid[:, 1] < 2.0 * h)
              & (fluid[:, 0].abs() <= reach - 2.0 * h))
    scale = 2.0 * line_integral(h, 0.0)
    return {
        "probe": "wall_hex",
        "h_over_dx": h_over_dx,
        "wall_factor": wall_factor,
        "n_fluid": int(fluid.shape[0]),
        "n_wall": int(wall.shape[0]),
        "n_graded": int(graded.sum()),
        "g_max_over_scale": float(mag[graded].max()) / scale,
        "g_mean_over_scale": float(mag[graded].mean()) / scale,
    }


# --------------------------------------------------------------------------- #
# figure + driver
# --------------------------------------------------------------------------- #
def _report_ratio(kind: str, h: float, h_over_dx: float):
    """Per-particle ``|g| / 2I(h, d)`` on the reported surface region."""
    lat = build_lattice(kind, h, h_over_dx)
    pos, vol = lat.positions, lat.volume
    g = kernel_gradient_integral(pos, torch.full((pos.shape[0],), vol,
                                                dtype=pos.dtype), h,
                                 pairs=cell_linked_pairs(pos, 2.0 * h))
    mag = torch.linalg.vector_norm(g, dim=1)
    surf = lat.surface
    d = torch.clamp(-pos[surf, 1], min=0.0)
    a = 2.0 * line_integrals(h, d)
    keep = a > 1.0e-2 * float(2.0 * line_integral(h, 0.0))
    return (d[keep] / h).numpy(), (mag[surf][keep] / a[keep]).numpy()


def make_figure(rows: list[dict], out: Path, h: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for probe, style in (("square", "o-"), ("hex", "s--")):
        sel = [r for r in rows if r["probe"] == probe]
        ax[0].plot([r["h_over_dx"] for r in sel],
                   [100.0 * r["mae_rel"] for r in sel], style,
                   label=f"{probe} lattice")
    for r in [r for r in rows if r["probe"] == "rot"]:
        ax[0].plot([r["h_over_dx"]], [100.0 * r["mae_rel"]], "D",
                   color="tab:red", alpha=0.65, ms=6)
    for r in [r for r in rows if r["probe"] == "jitter"]:
        ax[0].errorbar([r["h_over_dx"]], [100.0 * r["mae_rel"]],
                       yerr=[100.0 * r["mae_rel_sd"]], fmt="v",
                       color="tab:green", ms=7, capsize=3)
    ax[0].plot([], [], "D", color="tab:red", label="square rotated 0/15/30/45 deg")
    ax[0].plot([], [], "v", color="tab:green",
               label="square + jitter zeta=0.05/0.15/0.30")
    ax[0].axhline(15.0, color="k", ls=":", lw=1.2, label="owner gate 15 %")
    ax[0].axhline(3.0, color="0.5", ls=":", lw=1.0, label="3 % target")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("h / dx_eff  (dx_eff = sqrt(area per particle))")
    ax[0].set_ylabel("MAE of |g| vs 2I(h,d)   [%]")
    ax[0].set_title("free-surface TVC anchor vs lattice family")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    for kind, colour, marker in (("hex", "tab:blue", "o"), ("square", "tab:orange", "s")):
        d, ratio = _report_ratio(kind, h, HBAR)
        ax[1].scatter(d, ratio, c=colour, s=24, marker=marker,
                      label=f"{kind}, h/dx={HBAR}")
    ax[1].axhline(1.0, color="k", lw=1.0, ls="--", label="continuum limit")
    ax[1].set_xlabel("depth of the surface particle  d / h")
    ax[1].set_ylabel("|g| / 2I(h,d)")
    ax[1].set_title("per-particle surface term at production resolution")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"figure -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.026)
    ap.add_argument("--out", type=Path, default=Path("docs/xval_sphinx_tvc/data"))
    ap.add_argument("--figure", type=Path,
                    default=Path("docs/xval_sphinx_tvc/figure_lattice_sensitivity.png"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    h = args.h

    norm = check_normalisation(h)
    i0 = line_integral(h, 0.0)
    print(f"kernel normalisation : {norm:.12f} (target 1)")
    print(f"I(h,0)               : {i0:.6e}   closed form {closed_form_I0(h):.6e}")

    rows: list[dict] = []
    rows += square_regression(h)
    rows += hex_ladder(h)
    rows += rotation_probe(h)
    rows += jitter_ensemble(h)
    walls = [wall_hex_case(h, n, wf) for n in LADDER for wf in (2.0, 1.0)]

    hdr = f"{'probe':7s} {'h/dx':>5s} {'n':>5s} {'mean_rel':>9s} {'mae_rel':>9s} {'p95':>8s} {'cos':>7s} {'bulk':>9s}"
    print(hdr)
    for r in rows:
        key = f"{r.get('rotation_deg', r.get('zeta', ''))}"
        print(f"{r['probe']:7s} {r['h_over_dx']:5.2f} {r['n_surface']:5d} "
              f"{100*r['mean_rel']:8.2f}% {100*r['mae_rel']:8.2f}% "
              f"{100*r['p95_rel']:7.2f}% {r['cos']:7.4f} {r['bulk_rel']:9.2e} {key}")
    for w in walls:
        print(f"wall_hex h/dx={w['h_over_dx']:g} wf={w['wall_factor']:g}: "
              f"max|g|/scale = {w['g_max_over_scale']:.3e} "
              f"mean = {w['g_mean_over_scale']:.3e} (n={w['n_graded']})")

    payload = {
        "h": h,
        "hb_ar": HBAR,
        "kernel_normalisation": norm,
        "I_h_0": i0,
        "I0_closed_form": closed_form_I0(h),
        "rows": rows,
        "wall_hex": walls,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "lattice_sensitivity.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"data -> {path}")
    if not args.no_figure:
        make_figure(rows, args.figure, h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())