"""Integrity gate for the Turek-Hron FSI2 *benchmark target* itself.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Why this module exists
----------------------
Every error percentage this repository reports for FSI2 (tip amplitudes, drag
and lift coefficients, similarity verdicts) is a ratio to a *benchmark target*
that is kept in four places:

======================  =====================================================
source                  what it declares
======================  =====================================================
``docs/fsi2_turek_hron/01_survey.md``       the physical (SI) parameter table
``fsi2_turek_hron.FSI2Case``                the shipped dataclass defaults
``fsi2_similarity.BENCHMARK_FSI2``          the dimensionless target groups
``benchmarks/xval_fsi2/reference/oss/README.md``  the archived upstream block
======================  =====================================================

A comparison is only meaningful if the target is *one self-consistent row of
numbers*.  Some of the identities that must hold inside such a row need no
paper at all:

* ``Re = rho_f U_mean D / mu`` with ``U_mean = (2/3) u_max`` for the benchmark's
  parabola ``u = 4 u_max y (H-y)/H^2``;
* ``Ae = E / (rho_f U_mean^2)`` -- the normalized stiffness that the tip
  amplitude scales with (``delta ~ 1/Ae``);
* ``L/t`` (slenderness), ``rho_s/rho_f``.

Freezing the *dimensional* numbers while rescaling the velocity silently moves
the physics (see :mod:`tensorsph.fsi2_similarity`); freezing the *dimensionless*
numbers while quoting an incoherent dimensional table makes the reported error
percentages unattributable.  This module makes both failures mechanical.

Evidence classes
----------------
Every number the gate handles carries one of

``VERBATIM``   copied from the archived upstream source block (README in
               ``benchmarks/xval_fsi2/reference/oss/``), which is itself a
               verbatim archive of ``fsi2.h``/``fsi2.cpp``.
``DERIVED``    produced by an identity in this module (not typed in).
``SECONDARY``  the repository's quotation of the paper (Turek & Hron 2006) that
               was *not* re-verified against the PDF -- the class this gate
               exists to police.
``DEFAULT``    a dataclass default of the shipped case.

The gate never upgrades a ``SECONDARY`` number to a verified one: when the
internal identities cannot single out a row, the field is reported as
``UNRESOLVED`` and the two coherent candidate rows are printed side by side
(:func:`candidate_points`), which is what a paper check would have to decide.

Time-coverage gate
------------------
The second half of the module is the complementary check that the earlier
coupled report was missing: a periodic benchmark response needs *time*.  The
reference period is measured from the archived record
(:func:`reference_period`) instead of quoted, and :func:`coverage` converts a
run's ``(dt, n_steps)`` into "cycles of the benchmark period covered", so a
40-step probe cannot be reported as an amplitude comparison.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "SURVEY_PATH",
    "OSS_README_PATH",
    "IDENTITIES",
    "TOL",
    "MIN_CYCLES_AMPLITUDE",
    "MIN_CYCLES_STATISTICS",
    "parse_survey_table",
    "parse_oss_block",
    "case_row",
    "target_row",
    "check_row",
    "provenance_matrix",
    "candidate_points",
    "reference_period",
    "coverage",
    "audit",
]

_ROOT = Path(__file__).resolve().parents[2]
SURVEY_PATH = _ROOT / "docs" / "fsi2_turek_hron" / "01_survey.md"
OSS_README_PATH = _ROOT / "benchmarks" / "xval_fsi2" / "reference" / "oss" / "README.md"

#: Relative tolerance for the internal identities (1 %).
TOL = 1.0e-2

#: Predeclared coverage thresholds: a half-range amplitude needs ~3 cycles of
#: the response period to be a half-range at all, and a mean/variance needs ~10
#: (the numbers are the *stopping rule*; they are not retuned after a run).
MIN_CYCLES_AMPLITUDE = 3.0
MIN_CYCLES_STATISTICS = 10.0

#: The identities that must hold inside any self-consistent FSI2 row.
IDENTITIES = (
    "reynolds",
    "mean_of_parabola",
    "normalised_stiffness",
    "slenderness",
)


# --------------------------------------------------------------------------- #
# parsing the declared sources
# --------------------------------------------------------------------------- #
#: label (as printed in the survey table) -> gate key
_SURVEY_LABELS: Dict[str, str] = {
    "槽长": "channel_length",
    "槽高": "channel_height",
    "圆柱直径": "cylinder_d",
    "梁长": "beam_length",
    "梁厚": "beam_thickness",
    "流体密度": "rho_f",
    "流体动力粘度": "mu_f",
    "入口最大速度": "u_max",
    "入口平均速度": "u_mean",
    "雷诺数 (D, U_mean)": "Re",
    "固体密度": "rho_s",
    "固体杨氏模量": "beam_E",
    "固体泊松比": "poisson_s",
}

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_survey_table(path: Optional[Path] = None) -> Dict[str, float]:
    """Parse the physical parameter table out of the survey document.

    Reading the document (instead of re-typing its numbers here) is the point:
    the gate and the document cannot drift apart.
    """
    p = SURVEY_PATH if path is None else Path(path)
    text = p.read_text(encoding="utf-8")
    row = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = _SURVEY_LABELS.get(cells[0])
        if key is None:
            continue
        m = _NUM.search(cells[1])
        if m is None:
            continue
        row[key] = float(m.group(0))
        # "(0.2, 0.2)" style cells are skipped by the dict lookup above
    missing = [k for k in _SURVEY_LABELS.values() if k not in row]
    if missing:
        raise ValueError(f"survey table incomplete, missing {missing}")
    return row


def parse_oss_block(path: Optional[Path] = None) -> Dict[str, float]:
    """Parse the archived upstream parameter block (``fsi2.h`` line list).

    Only the values that are *verbatim upstream source* are extracted; the
    parenthetical annotations in the same block ("(benchmark: 1)",
    "(benchmark FSI2: 5.6e3)") are the repository's own claims and are
    deliberately **not** parsed here -- they are ``SECONDARY`` and are audited
    by :func:`provenance_matrix` instead.
    """
    p = OSS_README_PATH if path is None else Path(path)
    text = p.read_text(encoding="utf-8")
    out: Dict[str, float] = {}
    patterns = {
        "insert_radius": r"insert_circle_radius\s*=\s*([0-9.eE+-]+)",
        "beam_thickness": r"\bbh\s*=\s*([0-9.]+)\s*\*\s*insert_circle_radius",
        "beam_length": r"\bbl\s*=\s*([0-9.]+)\s*\*\s*insert_circle_radius",
        "rho0_f": r"rho0_f\s*=\s*([0-9.eE+-]+)",
        "U_f": r"\bU_f\s*=\s*([0-9.eE+-]+)",
        "Re": r"\bRe\s*=\s*([0-9.eE+-]+)",
        "rho0_s": r"rho0_s\s*=\s*([0-9.eE+-]+)",
        "poisson_s": r"\bpoisson\s*=\s*([0-9.eE+-]+)",
        "Ae": r"\bAe\s*=\s*([0-9.eE+-]+)\s*;",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m is None:
            raise ValueError(f"archived upstream block missing {key}")
        out[key] = float(m.group(1))
    out["beam_thickness"] *= out["insert_radius"]
    out["beam_length"] *= out["insert_radius"]
    return out


def case_row(case: Any) -> Dict[str, float]:
    """Dimensional row of a shipped :class:`FSI2Case` (duck-typed, torch-free)."""
    u_mean = 2.0 / 3.0 * float(case.inlet_u_max)
    return {
        "rho_f": float(case.rho0),
        "mu_f": float(case.viscosity),
        "cylinder_d": float(case.cylinder_d),
        "u_max": float(case.inlet_u_max),
        "u_mean": u_mean,
        "Re": float(case.rho0) * u_mean * float(case.cylinder_d) / float(case.viscosity),
        "rho_s": float(case.beam_rho),
        "beam_E": float(case.beam_E),
        "beam_length": float(case.beam_length),
        "beam_thickness": float(case.beam_thickness),
        "poisson_s": float(case.beam_nu),
    }


def target_row() -> Dict[str, float]:
    """The declared dimensionless target (``fsi2_similarity.BENCHMARK_FSI2``)."""
    from .fsi2_similarity import BENCHMARK_FSI2

    return dict(BENCHMARK_FSI2)


def oss_row() -> Dict[str, float]:
    """The archived upstream case, expressed as the same dimensionless groups."""
    o = parse_oss_block()
    return {
        "Re": o["Re"],
        "Ae": o["Ae"],
        "rho_s_over_rho_f": o["rho0_s"] / o["rho0_f"],
        "poisson_s": o["poisson_s"],
        "slenderness": o["beam_length"] / o["beam_thickness"],
        # upstream fixes U' = 1 in its own scaled system; the physical velocity
        # is a free similarity choice, so it is not part of the group set.
        "U_f_scaled": o["U_f"],
    }


# --------------------------------------------------------------------------- #
# identities and provenance
# --------------------------------------------------------------------------- #
def _rel(got: float, want: float) -> float:
    if want == 0.0:
        return float("inf") if got != 0.0 else 0.0
    return (got - want) / want


def _groups_from_dimensional(row: Dict[str, float]) -> Dict[str, float]:
    u_mean = float(row["u_mean"])
    return {
        "Re": row["rho_f"] * u_mean * row["cylinder_d"] / row["mu_f"],
        "Ae": row["beam_E"] / (row["rho_f"] * u_mean ** 2),
        "rho_s_over_rho_f": row["rho_s"] / row["rho_f"],
        "poisson_s": row["poisson_s"],
        "slenderness": row["beam_length"] / row["beam_thickness"],
    }


def check_row(
    row: Dict[str, float],
    target: Optional[Dict[str, float]] = None,
    tol: float = TOL,
) -> Dict[str, Any]:
    """Run every internal identity on one declared row.

    The row may be either dimensional (``rho_f``, ``mu_f``, ``u_mean`` ...) or a
    group set (``Re``, ``Ae`` ...).  Checks whose inputs are absent are reported
    as ``not_run`` -- never as passed.
    """
    tgt = target_row() if target is None else dict(target)
    checks: List[Dict[str, Any]] = []

    def add(name: str, got: Optional[float], want: Optional[float],
            note: str = "") -> None:
        if got is None or want is None:
            checks.append({"name": name, "status": "not_run", "note": note})
            return
        rel = _rel(float(got), float(want))
        checks.append({
            "name": name,
            "status": "pass" if abs(rel) <= tol else "fail",
            "computed": float(got),
            "declared": float(want),
            "rel_error": float(rel),
            "note": note,
        })

    # Reynolds identity (needs the dimensional triple + the declared Re)
    if all(k in row for k in ("rho_f", "u_mean", "cylinder_d", "mu_f")):
        add("reynolds", row["rho_f"] * row["u_mean"] * row["cylinder_d"] / row["mu_f"],
            row.get("Re", tgt.get("Re")),
            "Re = rho_f U_mean D / mu_f")
    else:
        add("reynolds", None, None, "dimensional row incomplete")

    # benchmark parabola: U_mean = (2/3) u_max
    if "u_max" in row:
        add("mean_of_parabola", 2.0 / 3.0 * row["u_max"], row.get("u_mean"),
            "u_mean = (2/3) u_max for u = 4 u_max y (H-y) / H^2")

    # normalised stiffness, against the declared target
    if row.get("beam_E") is not None and row.get("rho_f") and row.get("u_mean"):
        add("normalised_stiffness",
            row["beam_E"] / (row["rho_f"] * row["u_mean"] ** 2),
            tgt.get("Ae"), "Ae = E / (rho_f U_mean^2)")
    elif "Ae" in row:
        add("normalised_stiffness", row["Ae"], tgt.get("Ae"),
            "Ae compared with the declared target")

    # geometry and density ratio
    if row.get("beam_length") and row.get("beam_thickness"):
        add("slenderness", row["beam_length"] / row["beam_thickness"],
            tgt.get("slenderness"), "L / t")
    if row.get("rho_s") is not None and row.get("rho_f"):
        add("density_ratio", row["rho_s"] / row["rho_f"],
            tgt.get("rho_s_over_rho_f"), "rho_s / rho_f")
    elif "rho_s_over_rho_f" in row:
        add("density_ratio", row["rho_s_over_rho_f"],
            tgt.get("rho_s_over_rho_f"), "group compared with the declared target")
    if row.get("poisson_s") is not None:
        add("poisson_s", row["poisson_s"], tgt.get("poisson_s"), "nu_s")

    failed = [c["name"] for c in checks if c["status"] == "fail"]
    not_run = [c["name"] for c in checks if c["status"] == "not_run"]
    return {
        "checks": checks,
        "passed": not failed and not not_run,
        "failed": failed,
        "not_run": not_run,
    }


def provenance_matrix() -> Dict[str, Dict[str, Any]]:
    """Every declared FSI2 number, with its evidence class and where it lives.

    ``VERBATIM`` numbers come from the archived upstream source block;
    ``DERIVED`` from the identities in this module; ``SECONDARY`` are the
    repository's unverified quotations of the paper.
    """
    survey = parse_survey_table()
    oss = parse_oss_block()
    from .fsi2_turek_hron import FSI2Case  # local import keeps the module torch-free

    default = case_row(FSI2Case())
    tgt = target_row()
    return {
        "survey": {
            "class": "SECONDARY",
            "source": str(SURVEY_PATH.relative_to(_ROOT)),
            "values": survey,
            "gate": check_row(survey, tgt),
        },
        "case_defaults": {
            "class": "DEFAULT",
            "source": "tensorsph.fsi2_turek_hron.FSI2Case()",
            "values": default,
            "gate": check_row(default, tgt),
        },
        "declared_target": {
            "class": "SECONDARY",
            "source": "tensorsph.fsi2_similarity.BENCHMARK_FSI2",
            "values": tgt,
            "gate": check_row(tgt, tgt),
        },
        "archived_upstream": {
            "class": "VERBATIM",
            "source": str(OSS_README_PATH.relative_to(_ROOT)),
            "values": oss,
            "groups": oss_row(),
            "gate": check_row(oss_row(), tgt),
        },
    }


def candidate_points(
    rho_f: float = 1000.0,
    cylinder_d: float = 0.1,
    Re: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """The coherent rows reachable from the repository's own numbers.

    Both candidates respect ``Re`` (``U_mean`` is *derived* from the survey's
    viscosity, not typed) and the benchmark geometry; they differ exactly on the
    two quantities that no internal identity can decide:

    * ``paper_target``      ``rho_s/rho_f = 1``, ``Ae = 5.6e3`` (the declared
      target), which forces ``E = Ae rho_f U_mean^2`` -- the survey's ``4.0e6``
      is then *another* mis-transcription;
    * ``upstream_consistent`` ``rho_s/rho_f = 10``, ``Ae = 1.4e3`` (the archived
      upstream values), which gives ``E = 1.4e6``.

    Paper access would decide between them; until then every comparison has to
    say which row it used.
    """
    survey = parse_survey_table()
    oss = parse_oss_block()
    re_target = float(Re if Re is not None else survey["Re"])
    u_mean = re_target * survey["mu_f"] / (rho_f * cylinder_d)
    out: Dict[str, Dict[str, Any]] = {}
    for name, ratio, ae, poisson, klass in (
        ("paper_target", target_row()["rho_s_over_rho_f"], target_row()["Ae"],
         target_row()["poisson_s"], "SECONDARY"),
        ("upstream_consistent", oss_row()["rho_s_over_rho_f"], oss_row()["Ae"],
         oss_row()["poisson_s"], "VERBATIM"),
    ):
        out[name] = {
            "class": klass,
            "u_mean": u_mean,
            "u_max": 1.5 * u_mean,
            "mu_f": survey["mu_f"],
            "rho_f": rho_f,
            "rho_s": ratio * rho_f,
            "beam_E": ae * rho_f * u_mean ** 2,
            "Ae": ae,
            "rho_s_over_rho_f": ratio,
            "poisson_s": poisson,
            "Re": rho_f * u_mean * cylinder_d / survey["mu_f"],
            "slenderness": survey["beam_length"] / survey["beam_thickness"],
            "note": ("declared target row" if name == "paper_target"
                     else "archived upstream row (fsi2.h:36/38)"),
        }
    return out


# --------------------------------------------------------------------------- #
# time-coverage gate
# --------------------------------------------------------------------------- #
def reference_period(run: str = "0") -> Dict[str, float]:
    """Measure the response period of the archived benchmark record.

    The record's sampling interval is fixed by the archived case
    (``fsi2.cpp:192``: output interval 1.0 scaled = 0.1 s physical).  The period
    is taken from the dominant peak of the detrended tip-``y`` spectrum and
    cross-checked by zero crossings of the detrended signal, so the number is
    measured from data, not quoted from the paper.
    """
    from .fsi2_reference import load_tip_series

    n, pos = load_tip_series(run)
    dt_out = 0.1  # s physical, from the archived fsi2.cpp output interval
    y = pos[:, 1] / 10.0  # archived record is in scaled length units (scale 10)
    y = y - y.mean()
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, d=dt_out)
    f_peak = float(freqs[1 + int(np.argmax(spec[1:]))])
    signs = np.signbit(y)
    crossings = np.flatnonzero(signs[1:] != signs[:-1])
    if crossings.size >= 3:
        period_zc = float(2.0 * np.mean(np.diff(crossings[::2])) * dt_out)
    else:  # pragma: no cover - the archived record always crosses many times
        period_zc = float("nan")
    period_fft = 1.0 / f_peak if f_peak > 0 else float("inf")
    # The archived record is sampled at 0.1 s (Nyquist 5 Hz), i.e. only ~5
    # samples per cycle at the Strouhal frequency, and it is not a single clean
    # sinusoid (the zero-crossing and spectral estimates differ by ~2x).  The
    # coverage gate therefore uses the *longest* candidate period, which is the
    # conservative choice for "how many cycles did a run cover".
    period_used = float(max(period_fft, period_zc))
    return {
        "n_samples": int(n),
        "duration": float(n * dt_out),
        "f_peak": f_peak,
        "period_fft": float(period_fft),
        "period_zero_crossing": float(period_zc),
        "period_used": period_used,
        "period_strouhal_expectation": 0.5,  # St = 0.2, D = 0.1 m, U = 1 m/s
        "y_amplitude": float(0.5 * (y.max() - y.min())),
    }


def coverage(dt: float, n_steps: int, period: float,
             min_cycles: float = MIN_CYCLES_AMPLITUDE) -> Dict[str, Any]:
    """Cycles of the benchmark period covered by a run of ``dt`` x ``n_steps``."""
    duration = float(dt) * int(n_steps)
    cycles = duration / float(period)
    return {
        "dt": float(dt),
        "n_steps": int(n_steps),
        "duration": duration,
        "period": float(period),
        "cycles": cycles,
        "min_cycles": float(min_cycles),
        "status": "pass" if cycles >= min_cycles else "fail",
    }


# --------------------------------------------------------------------------- #
# the assembled audit
# --------------------------------------------------------------------------- #
def audit(period: Optional[float] = None) -> Dict[str, Any]:
    """Full audit bundle (JSON-serializable) -- the artifact of this module."""
    matrix = provenance_matrix()
    cands = candidate_points()
    per = reference_period()
    p = float(per["period_used"] if period is None else period)

    # A coupled probe of the size the earlier report used, for the record.
    from .fsi2_similarity import similarity_corrected_case

    probe_case = similarity_corrected_case(U_mean=0.2, dx=0.02)
    probe = coverage(probe_case.dt(), 40, p)

    # The coverage verdict must not hinge on which period estimate is used.
    sensitivity = {
        key: coverage(probe_case.dt(), 40, float(per[key]))["cycles"]
        for key in ("period_fft", "period_zero_crossing", "period_used",
                    "period_strouhal_expectation")
    }

    return {
        "period": per,
        "coverage_probe_40_steps": probe,
        "coverage_sensitivity": sensitivity,
        "matrix": matrix,
        "candidates": cands,
        "identities": {
            "survey_failed": matrix["survey"]["gate"]["failed"],
            "case_defaults_failed": matrix["case_defaults"]["gate"]["failed"],
            "archived_upstream_failed": matrix["archived_upstream"]["gate"]["failed"],
        },
    }