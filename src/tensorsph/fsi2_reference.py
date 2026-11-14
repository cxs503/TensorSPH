"""Loader, unit-scaling and acceptance gate for the SPHinXsys FSI2 records.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

The archived reference bundle lives in
``benchmarks/xval_fsi2/reference/oss/`` (see the README there for provenance,
upstream blob shas and the licence).  This module turns those records into
physical units and gives the acceptance verdict against the Turek-Hron FSI2
values, so a coupled run of this repository can be judged against them (or
against the values quoted in :mod:`tensorsph.fsi2_turek_hron`) with one call.

Two unit facts about the upstream record, both derived (not assumed) here:

``length_scale = 10``
    ``fsi2.h`` sets ``insert_circle_radius = 0.5`` and ``DH = 4.1`` while the
    benchmark is D = 0.1 m and H = 0.41 m, and the stored beam-tip abscissa of a
    straight beam is 6.0 = (0.2 + 0.05 + 0.35) x 10.  Verified against the data
    in :func:`infer_length_scale`.

``force_scale = rho_f U_mean^2 / length_scale``
    In SPHinXsys' scaled system ``rho' = 1``, ``U' = 1`` and one length unit is
    ``0.1 m``, so a 2-D force per unit depth ``F'`` maps to
    ``F = F' * rho_f U_mean^2 / (rho' U'^2 * length_scale)`` N/m with the
    physical unit depth of 1 m (``length_scale`` scaled units).
    :func:`independent_friction_estimate` cross-checks the resulting drag
    coefficient against a laminar flat-plate estimate -- an independent order-of-
    magnitude check of the convention, not a benchmark value.

The point of the module is *not* to certify the OSS record: it is to make the
comparison mechanical and to let the audit state, with numbers, that the OSS
``test_2d_fsi2`` set-up is not the Turek-Hron FSI2 parameter point (see
:mod:`tensorsph.fsi2_similarity`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "REFERENCE_DIR",
    "LENGTH_SCALE",
    "BENCHMARK_TIP",
    "BENCHMARK_FORCES",
    "OSS_VISCOUS_FORCE_SCALED",
    "parse_tip_series",
    "load_tip_series",
    "parse_mean_variance",
    "load_mean_variance",
    "infer_length_scale",
    "summarise_tip",
    "viscous_force_physical",
    "drag_coefficient",
    "independent_friction_estimate",
    "compare_to_benchmark",
    "bundle",
]

#: Where the verbatim upstream records are archived in the repository.
REFERENCE_DIR = Path(
    os.environ.get(
        "TENSORSPH_XVAL_FSI2_REF",
        str(Path(__file__).resolve().parents[2]
            / "benchmarks" / "xval_fsi2" / "reference" / "oss"),
    )
)

#: Length scaling of the SPHinXsys ``test_2d_fsi2`` set-up (D' = 1.0 vs D = 0.1 m).
LENGTH_SCALE = 10.0

#: Turek-Hron FSI2 beam-tip displacement amplitudes [m].
#: Secondary provenance: quoted in ``fsi2_turek_hron.FSI2Case`` from
#: Turek & Hron (2006), LNCSE 53, 371-385; not re-verified against the paper here.
BENCHMARK_TIP: Dict[str, float] = {
    "tip_x_amplitude": 0.0028,
    "tip_y_amplitude": 0.0082,
}

#: Turek-Hron FSI2 force coefficients (same provenance as :data:`BENCHMARK_TIP`).
BENCHMARK_FORCES: Dict[str, float] = {
    "drag_mean": 3.2,
    "drag_amplitude": 0.96,
    "lift_amplitude": 0.22,
}

#: Time-averaged total viscous force from fluid on the insert body, scaled units
#: (``InsertedBody_TotalViscousForceFromFluid_time_averaged_mean_variance.xml``).
OSS_VISCOUS_FORCE_SCALED: Tuple[float, float] = (0.451157, 0.00157671)

#: The benchmark fluid properties, used only for unit conversions.
BENCHMARK_FLUID = {"rho_f": 1000.0, "u_mean": 1.0, "D": 0.1, "nu": 1.0e-3}

_PAIR_RE = re.compile(r"snapshot_\d+=\"~\[([-\d.eE+]+),([-\d.eE+]+)\]\"")
_VALUE_RE = re.compile(r"(TotalViscousForceFromFluid_\d+)=\"~\[([-\d.eE+]+),([-\d.eE+]+)\]\"")
_NK_RE = re.compile(r"number_of_snapshot_for_local_result_=\"(\d+)\"")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_tip_series(text: str) -> Tuple[int, np.ndarray]:
    """Parse a ``BeamObserver_Position_Run_*_result.xml`` string.

    Returns ``(declared_n_snapshots, positions)`` with ``positions`` of shape
    ``(N, 2)`` in the record's scaled length units.
    """
    m = _NK_RE.search(text)
    declared = int(m.group(1)) if m else -1
    pairs = _PAIR_RE.findall(text)
    if not pairs:
        raise ValueError("no snapshot pairs found; not a BeamObserver_Position record")
    arr = np.asarray(pairs, dtype=float)
    if declared >= 0 and arr.shape[0] != declared:
        raise ValueError(
            f"declared {declared} snapshots but parsed {arr.shape[0]}"
        )
    return declared, arr


def load_tip_series(run: str = "0", directory: Optional[Path] = None) -> Tuple[int, np.ndarray]:
    """Load ``BeamObserver_Position_Run_<run>_result.xml`` from the archive."""
    d = REFERENCE_DIR if directory is None else Path(directory)
    path = d / f"BeamObserver_Position_Run_{run}_result.xml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; the archival bundle must be present for this comparison"
        )
    return parse_tip_series(path.read_text())


def parse_mean_variance(text: str) -> Dict[str, Any]:
    """Parse the time-averaged mean/variance record (vector values)."""
    def _block(tag: str) -> str:
        start = text.find(f"<{tag}>")
        end = text.find(f"</{tag}>")
        if start < 0 or end < 0:
            return ""
        return text[start:end]

    def _values(block: str) -> Dict[str, np.ndarray]:
        return {
            m[0]: np.asarray((m[1], m[2]), dtype=float)
            for m in _VALUE_RE.findall(block)
        }

    mean = _values(_block("MeanValue_Element"))
    if not mean:
        raise ValueError("no TotalViscousForceFromFluid_* value found")
    return {"mean": mean, "variance": _values(_block("Variance_Element"))}


def load_mean_variance(directory: Optional[Path] = None) -> Dict[str, Any]:
    d = REFERENCE_DIR if directory is None else Path(directory)
    path = d / "InsertedBody_TotalViscousForceFromFluid_time_averaged_mean_variance.xml"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing from the archival bundle")
    return parse_mean_variance(path.read_text())


# --------------------------------------------------------------------------- #
# statistics and unit conversion
# --------------------------------------------------------------------------- #
def infer_length_scale(positions: np.ndarray, straight_tip_x: float = 0.6) -> float:
    """Recover the record's length scale from the data itself.

    The beam tip of an initially straight beam sits at
    ``x = cylinder_x + D/2 + L = 0.6`` m; the record stores it near ``6.0``.
    Only the abscissa is used (the beam bends downstream/upstream by a few
    percent, which is why the median, not the maximum, is compared).
    """
    return float(np.median(positions[:, 0])) / straight_tip_x


def summarise_tip(positions: np.ndarray, length_scale: float = LENGTH_SCALE) -> Dict[str, float]:
    """Mean and half-range amplitude of the tip position, scaled and physical."""
    x, y = positions[:, 0], positions[:, 1]
    return {
        "n_samples": int(positions.shape[0]),
        "x_mean_scaled": float(x.mean()),
        "y_mean_scaled": float(y.mean()),
        "x_amplitude_scaled": float(0.5 * (x.max() - x.min())),
        "y_amplitude_scaled": float(0.5 * (y.max() - y.min())),
        "x_mean": float(x.mean()) / length_scale,
        "y_mean": float(y.mean()) / length_scale,
        "tip_x_amplitude": float(0.5 * (x.max() - x.min())) / length_scale,
        "tip_y_amplitude": float(0.5 * (y.max() - y.min())) / length_scale,
        "length_scale": float(length_scale),
    }


def viscous_force_physical(
    mean_scaled: np.ndarray,
    rho_f: float = BENCHMARK_FLUID["rho_f"],
    u_mean: float = BENCHMARK_FLUID["u_mean"],
    length_scale: float = LENGTH_SCALE,
    rho_scaled: float = 1.0,
    u_scaled: float = 1.0,
) -> np.ndarray:
    """Scaled 2-D force per unit depth -> N/m (see the module docstring)."""
    factor = rho_f * u_mean ** 2 / (rho_scaled * u_scaled ** 2 * length_scale)
    return np.asarray(mean_scaled, dtype=float) * factor


def drag_coefficient(
    force: np.ndarray,
    rho_f: float = BENCHMARK_FLUID["rho_f"],
    u_mean: float = BENCHMARK_FLUID["u_mean"],
    D: float = BENCHMARK_FLUID["D"],
) -> np.ndarray:
    """``C = F / (0.5 rho U^2 D)`` per unit depth."""
    return np.asarray(force, dtype=float) / (0.5 * rho_f * u_mean ** 2 * D)


def independent_friction_estimate(
    L: float = 0.35, D: float = BENCHMARK_FLUID["D"],
    nu: float = BENCHMARK_FLUID["nu"], u_mean: float = BENCHMARK_FLUID["u_mean"],
) -> Dict[str, float]:
    """Laminar flat-plate friction estimate for the beam, both sides.

    ``C_f = 1.328 / sqrt(Re_L)`` (Blasius, average over the plate length), so the
    friction drag coefficient referred to ``0.5 rho U^2 D`` is
    ``Cd_f = 2 C_f L / D``.  This is an order-of-magnitude check of the force
    unit conversion only: the real body oscillates and has a cylinder at the
    leading edge, so a factor ~2 spread is expected and acceptable.
    """
    re_l = u_mean * L / nu
    cf = 1.328 / np.sqrt(re_l)
    return {"Re_L": float(re_l), "Cf": float(cf), "Cd_friction": float(2.0 * cf * L / D)}


# --------------------------------------------------------------------------- #
# acceptance gate
# --------------------------------------------------------------------------- #
def compare_to_benchmark(
    summary: Dict[str, float],
    reference: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Relative error of each amplitude against the benchmark, with the gates.

    ``pass_15`` is the owner's acceptance standard (error > 15 % is not
    acceptable); ``pass_3`` is the tighter reporting threshold used by the
    platform's report discipline.
    """
    reference = BENCHMARK_TIP if reference is None else reference
    rows = []
    for name, ref in reference.items():
        got = summary[name]
        rel = (got - ref) / ref
        rows.append({
            "quantity": name,
            "reference": ref,
            "computed": got,
            "rel_error": rel,
            "pass_15": abs(rel) <= 0.15,
            "pass_3": abs(rel) <= 0.03,
        })
    return {
        "rows": rows,
        "passed_15": all(r["pass_15"] for r in rows),
        "passed_3": all(r["pass_3"] for r in rows),
        "worst_rel_error": max(abs(r["rel_error"]) for r in rows),
    }


def bundle(directory: Optional[Path] = None) -> Dict[str, Any]:
    """Everything the audit needs, from the archived records alone."""
    runs = {}
    inferred = []
    for run in ("0", "4", "9"):
        try:
            _, pos = load_tip_series(run, directory)
        except FileNotFoundError:
            continue
        runs[run] = summarise_tip(pos)
        inferred.append(infer_length_scale(pos))
    if not runs:
        raise FileNotFoundError("no archived BeamObserver_Position records found")
    scale = float(np.median([r["length_scale"] for r in runs.values()]))
    mv = load_mean_variance(directory)
    f_scaled = next(iter(mv["mean"].values()))
    f_phys = viscous_force_physical(f_scaled)
    return {
        "runs": runs,
        "worst_run": max(runs, key=lambda r: runs[r]["tip_y_amplitude"]),
        "tip_gate": compare_to_benchmark(runs["0"]),
        "viscous_force_scaled": f_scaled.tolist(),
        "viscous_force_physical": f_phys.tolist(),
        "viscous_Cd": drag_coefficient(f_phys).tolist(),
        "flat_plate_estimate": independent_friction_estimate(),
        "length_scale": scale,
        "inferred_length_scale": float(np.median(inferred)),
        "amplitude_spread_tip_y": float(
            (max(r["tip_y_amplitude"] for r in runs.values())
             - min(r["tip_y_amplitude"] for r in runs.values()))
            / np.mean([r["tip_y_amplitude"] for r in runs.values()])
        ),
    }