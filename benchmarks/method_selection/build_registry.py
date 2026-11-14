"""Build the method-selection evidence registry, routing matrix, figures and report.

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12`` (mission 18056).

Everything the report prints is produced here from committed benchmark artefacts:
the registry stores, per method, the measured error, the gate it was judged
against, the source path and the sha256 of that source.  No number is typed in by
hand, so a hand-edited registry is detectable (``tests/test_method_selection.py``
rebuilds it and compares).

Usage::

    PYTHONPATH=src python benchmarks/method_selection/build_registry.py
    PYTHONPATH=src python benchmarks/method_selection/build_registry.py --check

``--check`` regenerates the documents in memory and fails (exit 1) if the files on
disk differ -- that is the anti-drift gate used by the test suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.method_selection import (  # noqa: E402
    METHODS,
    MODIFIERS,
    OWNER_TOLERANCE_PCT,
    CaseProfile,
    select_method,
)

REGISTRY_PATH = ROOT / "docs" / "reports" / "method_selection_registry.json"
MATRIX_PATH = ROOT / "docs" / "reports" / "method_selection_m18056.json"
TABLES_PATH = ROOT / "docs" / "reports" / "method_selection_m18056_tables.md"
FIG_DIR = ROOT / "docs" / "figures" / "method_selection_m18056"
REPORT_PATH = ROOT / "docs" / "report_method_selection_m18056.md"
NARRATIVE_PATH = ROOT / "docs" / "method_selection_m18056_narrative.md"
MARKER = "<!--INSERT-GENERATED-TABLES-->"


# --------------------------------------------------------------------------- #
# Extractors: (committed artefact) -> (measured error in percent)
# --------------------------------------------------------------------------- #
def x_wave_flume(doc: dict) -> float:
    return abs(float(doc["worst_error_percent"]))


def x_sod(doc: dict) -> float:
    return abs(float(doc["metrics"]["max_abs_rel_err"])) * 100.0


def x_conduction(doc: dict) -> float:
    return abs(float(doc["main_run"]["max_normalised_error"])) * 100.0


def x_laplace_pressure(doc: dict) -> float:
    return abs(float(doc["laplace"]["rel_error_pct"]))


def x_csf_curvature(doc: dict) -> float:
    entry = doc["instant"]["consistent"][0]
    return abs(float(entry["kappa"]["rel_error_pct"]))


def x_density_diffusion_decay(doc: dict) -> float:
    return abs(float(doc["max_rate_relative_error"])) * 100.0


def x_density_diffusion_finite_time(doc: dict) -> float:
    """Amplitude decay over a finite run (the artefact's second gate, 3.5 %)."""
    return abs(float(doc["finite_time"]["relative_error"])) * 100.0


def x_isph_channel(doc: dict) -> float:
    finest = max(doc["results"], key=lambda row: row["resolution_H_over_dx"])
    return abs(float(finest["gates"]["centre_velocity"]["rel_error_pct"]))


def x_compression_regularization(doc: dict) -> float:
    arm = next(r for r in doc["records"] if r["arm"] == "compression_refmean")
    return abs(float(arm["gate"]["max_band_error_pct"]))


def x_particle_shifting(doc: dict) -> float:
    return abs(float(doc["cases"]["dam_break"]["front_delta_over_column"])) * 100.0


def x_couette(doc: dict) -> float:
    finest = max(doc["ladder"], key=lambda row: row["resolution_H_over_dx"])
    return abs(float(finest["metrics"]["interior_l2"])) * 100.0

def _couette_finest(doc: dict) -> dict:
    return max(doc["ladder"], key=lambda row: row["resolution_H_over_dx"])


def x_couette_wall_shear(doc: dict) -> float:
    """Wall-shear error at the moving wall -- the quantity that *defines* no-slip.

    ``metrics.wall_shear_ratio = tau_wall_SPH / tau_wall_analytic``; the artefact
    gates it at ``gates.wall_shear_rel`` (5 %).  Banking only the interior L2
    profile error lets a scheme look accurate while its wall stress is off.
    """
    row = _couette_finest(doc)
    return abs(1.0 - float(row["metrics"]["wall_shear_ratio"])) * 100.0


def x_couette_flux(doc: dict) -> float:
    """Mass-flux error across the channel (``gates.flux_rel``, 5 %)."""
    row = _couette_finest(doc)
    return abs(1.0 - float(row["metrics"]["flux_ratio"])) * 100.0


def x_isph_wall_shear(doc: dict) -> float:
    """Momentum-budget wall shear on the finest rung (gated at 3 % in the artefact)."""
    finest = max(doc["results"], key=lambda row: row["resolution_H_over_dx"])
    return abs(float(finest["gates"]["wall_shear_budget"]["rel_error_pct"]))


def x_csf_cut_force(doc: dict) -> float:
    """Cut-force error of the consistent-CSF arm (the Laplace pressure companion)."""
    entry = doc["instant"]["consistent"][0]
    return abs(float(entry["cut_force"]["rel_error_pct"]))


#: Variant the transport-velocity record is scored on: the coefficient calibrated
#: in ``docs/xval_tvc_validation_report.md`` (upstream default 0.2 over-disperses
#: in this solver; 0.10 is the validated setting).  Pinned by name, not by index,
#: so re-ordering the grid cannot silently change which arm is read.
TVC_VARIANT = "tvc_a010_all"


def _tvc_record(doc: dict, dx: float) -> dict:
    matches = [r for r in doc["records"]
               if r["variant"] == TVC_VARIANT and abs(float(r["dx"]) - dx) < 1e-12]
    if not matches:
        raise KeyError(f"no {TVC_VARIANT!r} record at dx={dx} in the TVC artefact")
    return matches[0]


def x_transport_velocity(doc: dict) -> float:
    """Mean relative error of Z/a vs Martin & Moyce, dx = 0.03 m (finest committed)."""
    return abs(float(_tvc_record(doc, 0.03)["score"]["mean_rel_error"])) * 100.0


def x_transport_velocity_coarse(doc: dict) -> float:
    """Same quantity on the coarser rung -- kept in the registry on purpose."""
    return abs(float(_tvc_record(doc, 0.04)["score"]["mean_rel_error"])) * 100.0


@dataclass(frozen=True)
class Source:
    extractor: str
    method: str
    case: str
    quantity: str
    gate_pct: float
    path: str
    fn: object


SOURCES: tuple[Source, ...] = (
    Source(
        extractor="x_wave_flume",
        method="wcsph_delta_sph",
        case="CASE-WAVE-FLUME",
        quantity="max gauge wave-height error vs piston-transfer target (H/dx=3.2, delta=0.1)",
        gate_pct=10.0,
        path="docs/reports/wave_flume_numbers.json",
        fn=x_wave_flume,
    ),
    Source(
        extractor="x_sod",
        method="compressible_riemann",
        case="CASE-SPH-SOD-SHOCK-COMPRESSIBLE",
        quantity="max abs relative error over the gated shock quantities "
                 "(rho*, p*, u*, wave positions)",
        gate_pct=5.0,
        path="benchmarks/case_sod_shock_compressible/result.json",
        fn=x_sod,
    ),
    Source(
        extractor="x_conduction",
        method="heat_conduction_operator",
        case="CASE-SPH-CONDUCTION-1D",
        quantity="max normalised error vs the erfc similarity solution",
        gate_pct=3.0,
        path="benchmarks/case_conduction_1d/result.json",
        fn=x_conduction,
    ),
    Source(
        extractor="x_laplace_pressure",
        method="csf_surface_tension",
        case="CASE-LAPLACE-SPH",
        quantity="Laplace pressure jump rel. error (delta_p vs 2*sigma/R)",
        gate_pct=15.0,
        path="benchmarks/laplace_law/validation_report.json",
        fn=x_laplace_pressure,
    ),
    Source(
        extractor="x_csf_curvature",
        method="csf_surface_tension",
        case="CASE-LAPLACE-SPH (consistent CSF)",
        quantity="delta_s-weighted mean curvature rel. error",
        gate_pct=15.0,
        path="benchmarks/csf_consistent/validation_report.json",
        fn=x_csf_curvature,
    ),
    Source(
        extractor="x_density_diffusion_decay",
        method="density_diffusion",
        case="plane density wave decay",
        quantity="max relative error of the decay rate vs the analytic gamma = D k^2",
        gate_pct=3.0,
        path="benchmarks/density_diffusion_decay/decay_rate.json",
        fn=x_density_diffusion_decay,
    ),
    Source(
        extractor="x_density_diffusion_finite_time",
        method="density_diffusion",
        case="plane density wave decay (finite run)",
        quantity="final amplitude vs the analytic exp(-D k^2 t) after 100 steps, "
                 "artefact gate 3.5 %",
        gate_pct=3.5,
        path="benchmarks/density_diffusion_decay/decay_rate.json",
        fn=x_density_diffusion_finite_time,
    ),
    Source(
        extractor="x_isph_channel",
        method="isph_projection",
        case="CASE-ISPH-CHANNEL (GAP-SPH-ISPH-PROJECTION ladder)",
        quantity="centre-velocity error vs the analytic Poiseuille parabola "
                 "(H/dx=16, momentum-budget wall-shear record also banked)",
        gate_pct=3.0,
        path="benchmarks/isph_channel/results.json",
        fn=x_isph_channel,
    ),
    Source(
        extractor="x_compression_regularization",
        method="compression_regularization",
        case="XVAL-SPHINXSYS-COMPRESSION-DENSITY-REGULARIZATION",
        quantity="max interior-band hydrostatic pressure error vs analytic "
                 "p = rho0 g (H - y), SPHinXsys reference-mean arm",
        gate_pct=1.0,
        path="benchmarks/xval_sphinxsys_compression/records.json",
        fn=x_compression_regularization,
    ),
    Source(
        extractor="x_particle_shifting",
        method="particle_shifting",
        case="CASE-PARTICLE-SHIFTING-DAMBREAK (B2, shifting on vs off)",
        quantity="dam-break front-position deviation from the unshifted published "
                 "answer, relative to the 0.2 m column",
        gate_pct=10.0,
        path="benchmarks/particle_shifting/results.json",
        fn=x_particle_shifting,
    ),
    Source(
        extractor="x_couette",
        method="wcsph_wall_no_slip",
        case="plane Couette channel",
        quantity="interior L2 velocity error vs u=U*y/H (finest rung)",
        gate_pct=5.0,
        path="benchmarks/couette/out/results.json",
        fn=x_couette,
    ),
    Source(
        extractor="x_couette_wall_shear",
        method="wcsph_wall_no_slip",
        case="plane Couette channel",
        quantity="wall-shear error at the moving wall, |1 - tau_SPH/tau_exact| "
                 "(the no-slip quantity itself; artefact gate 5 %)",
        gate_pct=5.0,
        path="benchmarks/couette/out/results.json",
        fn=x_couette_wall_shear,
    ),
    Source(
        extractor="x_couette_flux",
        method="wcsph_wall_no_slip",
        case="plane Couette channel",
        quantity="cross-channel mass-flux error |1 - Q_SPH/Q_exact| (artefact gate 5 %)",
        gate_pct=5.0,
        path="benchmarks/couette/out/results.json",
        fn=x_couette_flux,
    ),
    Source(
        extractor="x_isph_wall_shear",
        method="isph_projection",
        case="CASE-ISPH-CHANNEL (GAP-SPH-ISPH-PROJECTION ladder)",
        quantity="momentum-budget wall shear vs rho g H/2 on the finest rung "
                 "(artefact gate 3 %)",
        gate_pct=3.0,
        path="benchmarks/isph_channel/results.json",
        fn=x_isph_wall_shear,
    ),
    Source(
        extractor="x_csf_cut_force",
        method="csf_surface_tension",
        case="CASE-LAPLACE-SPH (consistent CSF)",
        quantity="cut-force |F_SPH - sigma| error of the consistent-CSF arm",
        gate_pct=15.0,
        path="benchmarks/csf_consistent/validation_report.json",
        fn=x_csf_cut_force,
    ),
    Source(
        extractor="x_transport_velocity",
        method="transport_velocity_correction",
        case="XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION (dam break, dx=0.03 m)",
        quantity="mean relative error of the front position Z/a vs the Martin & "
                 "Moyce (1952) digitised points, alpha=0.10 over all particles",
        gate_pct=15.0,
        path="docs/xval_sphinx_tvc/data/grid_dx003.json",
        fn=x_transport_velocity,
    ),
    Source(
        extractor="x_transport_velocity_coarse",
        method="transport_velocity_correction",
        case="XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION (dam break, dx=0.04 m)",
        quantity="same quantity on the coarser rung: records the alpha=0.10 arm is "
                 "resolution-sensitive (kept in the registry, fails the owner standard)",
        gate_pct=15.0,
        path="docs/xval_sphinx_tvc/data/grid_dx004.json",
        fn=x_transport_velocity_coarse,
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_registry() -> dict:
    records = []
    missing = []
    for source in SOURCES:
        path = ROOT / source.path
        if not path.exists():
            missing.append(source.path)
            continue
        doc = json.loads(path.read_text())
        measured = float(source.fn(doc))
        records.append(
            {
                "method": source.method,
                "case": source.case,
                "quantity": source.quantity,
                "measured_pct": measured,
                "gate_pct": float(source.gate_pct),
                "within_owner_tolerance": measured <= OWNER_TOLERANCE_PCT,
                "own_gate_passed": measured <= float(source.gate_pct),
                "source": source.path,
                "source_sha256": sha256_of(path),
                "extractor": source.extractor,
            }
        )
    evidenced = sorted({r["method"] for r in records})
    registry = {
        "schema_version": "1.0",
        "generated_by": "benchmarks/method_selection/build_registry.py",
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "mission": 18056,
        "owner_tolerance_pct": OWNER_TOLERANCE_PCT,
        "records": records,
        "methods_with_evidence": evidenced,
        "methods_without_evidence": sorted((set(METHODS) | set(MODIFIERS)) - set(evidenced)),
        "missing_sources": missing,
    }
    return registry


# --------------------------------------------------------------------------- #
# Canonical profiles the router is exercised on
# --------------------------------------------------------------------------- #
PROFILES: tuple[tuple[str, dict, str], ...] = (
    (
        "regular wave in a flume",
        dict(acoustic_mach=0.03, has_free_surface=True, gravity_waves=True),
        "wcsph_delta_sph",
    ),
    (
        "Sod shock tube",
        dict(acoustic_mach=0.9, shock_advection=True, domain="closed"),
        "compressible_riemann",
    ),
    (
        "capillary droplet (Laplace law)",
        dict(acoustic_mach=0.05, has_free_surface=True, capillary_dominated=True),
        "csf_surface_tension",
    ),
    (
        "planar Couette channel",
        dict(acoustic_mach=0.01, reynolds=1.0, domain="closed"),
        "isph_projection",  # m18278: ISPH channel ladder supplies the record
    ),
    (
        "Gresho vortex (smooth inviscid, periodic)",
        dict(acoustic_mach=0.05, reynolds=1e4, domain="periodic",
             smooth_inviscid_vortex=True),
        None,  # projection only, and projection has no record
    ),
    (
        "1-D transient conduction",
        dict(thermal_transport=True, domain="closed"),
        "heat_conduction_operator",
    ),
    (
        "breaking wave over a slope",
        dict(acoustic_mach=0.04, has_free_surface=True, gravity_waves=True,
             tensile_stretching=True),
        "wcsph_delta_sph",
    ),
)


def build_matrix(registry: dict) -> dict:
    by_method: dict[str, list] = {}
    for raw in registry["records"]:
        from tensorsph.method_selection import Evidence

        by_method.setdefault(raw["method"], []).append(
            Evidence(
                method=raw["method"],
                case=raw["case"],
                quantity=raw["quantity"],
                measured_pct=raw["measured_pct"],
                gate_pct=raw["gate_pct"],
                source=raw["source"],
                source_sha256=raw["source_sha256"],
            )
        )
    rows = []
    for label, fields, expected in PROFILES:
        selection = select_method(CaseProfile.from_mapping(fields), registry=by_method)
        rows.append(
            {
                "label": label,
                "profile": fields,
                "expected_primary": expected,
                "primary_method": selection.primary.method if selection.primary else None,
                "primary_status": selection.primary.status if selection.primary else "NONE",
                "primary_worst_pct": (
                    selection.primary.worst_measured_pct if selection.primary else None
                ),
                "primary_worst_banked_pct": (
                    selection.primary.worst_banked_pct if selection.primary else None
                ),
                "modifiers": [
                    {"method": d.method, "status": d.status} for d in selection.modifiers
                ],
                "candidates": [
                    {"method": d.method, "status": d.status} for d in selection.candidates
                ],
                "notes": selection.notes,
                "verified": selection.is_verified,
            }
        )
    return {
        "anchor": registry["anchor"],
        "mission": registry["mission"],
        "owner_tolerance_pct": OWNER_TOLERANCE_PCT,
        "profiles": rows,
        "unexpected_routes": [
            row["label"]
            for row in rows
            if row["primary_method"] != row["expected_primary"]
        ],
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def figure_evidence(registry: dict, out: Path) -> Path:
    records = registry["records"]
    labels = [f"{r['method']}\n{r['case']}" for r in records]
    values = [max(r["measured_pct"], 1e-9) for r in records]
    colors = ["#2a7f62" if r["within_owner_tolerance"] else "#b23b3b" for r in records]
    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(records) + 1.9))
    y = range(len(records))
    ax.barh(list(y), values, color=colors, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xscale("log")
    ax.axvline(OWNER_TOLERANCE_PCT, color="black", ls="--", lw=1.2,
               label=f"owner standard {OWNER_TOLERANCE_PCT:g}%")
    for index, record in enumerate(records):
        ax.plot([record["gate_pct"]], [index], marker="|", color="#1f4e79", ms=14,
                mew=2.0)
        ax.annotate(f"{record['measured_pct']:.3g}%", (values[index], index),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=7)
    ax.plot([], [], marker="|", color="#1f4e79", ms=10, mew=2.0, ls="none",
            label="case-own gate")
    ax.invert_yaxis()
    ax.set_xlabel("measured error vs. reference [%] (log scale)")
    ax.set_title("Method-selection evidence registry: measured error per record")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_matrix(matrix: dict, out: Path) -> Path:
    rows = matrix["profiles"]
    columns = ["case profile", "primary scheme", "status", "worst err [%]", "modifiers"]
    cells = []
    for row in rows:
        modifiers = ", ".join(
            f"{m['method']}={m['status']}" for m in row["modifiers"]
        ) or "-"
        cells.append(
            [
                row["label"],
                row["primary_method"] or "(none)",
                row["primary_status"],
                "-" if row["primary_worst_pct"] is None else f"{row['primary_worst_pct']:.3g}",
                modifiers,
            ]
        )
    fig, ax = plt.subplots(figsize=(12.5, 0.55 * len(rows) + 1.6))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.35)
    for index, row in enumerate(rows, start=1):
        color = "#eaf5ee" if row["verified"] else "#fbeceb"
        for column in range(len(columns)):
            table[index, column].set_facecolor(color)
    ax.set_title("Evidence-gated routing: case profile -> scheme (green = evidence-backed)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Markdown tables / report composition
# --------------------------------------------------------------------------- #
def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_tables(registry: dict, matrix: dict, figures: list[Path]) -> str:
    parts: list[str] = []
    parts.append("### 3.1 证据登记表（程序化抽取，每行带源文件与 sha256）\n")
    rows = []
    for record in registry["records"]:
        rows.append([
            f"`{record['method']}`",
            record["case"],
            record["quantity"],
            f"{record['measured_pct']:.4g}",
            f"{record['gate_pct']:g}",
            "✔" if record["within_owner_tolerance"] else "✘",
            "✔" if record["own_gate_passed"] else "✘",
            f"`{record['source']}`",
        ])
    parts.append(_table(
        ["方法", "算例", "度量", "实测误差 [%]", "本算例门 [%]", "<15%", "过本门", "来源"],
        rows,
    ))
    worst = max(record["measured_pct"] for record in registry["records"])
    parts.append(
        f"\n登记记录 {len(registry['records'])} 条，覆盖 {len(registry['methods_with_evidence'])} "
        f"个方法；最坏误差 {worst:.4g}%。缺失源文件："
        f"{registry['missing_sources'] or '无'}。\n"
    )

    parts.append("\n### 3.2 路由矩阵（候选算例画像 → 选定方法）\n")
    rows = []
    for row in matrix["profiles"]:
        rows.append([
            row["label"],
            row["primary_method"] or "（无）",
            row["primary_status"],
            "-" if row["primary_worst_pct"] is None
            else f"{row['primary_worst_pct']:.4g}",
            ", ".join(f"{m['method']}={m['status']}" for m in row["modifiers"]) or "-",
        ])
    parts.append(_table(
        ["算例画像", "选定方法", "状态", "最坏实测误差 [%]", "附加项（modifier）"],
        rows,
    ))
    parts.append(
        f"\n{len(matrix['profiles'])} 个画像中 "
        f"{sum(1 for r in matrix['profiles'] if r['verified'])} 个给出证据背书的方案；"
        f"与预期不一致者：{matrix['unexpected_routes'] or '无'}。\n"
    )

    parts.append("\n### 3.3 无证据方法（evidence gate 拦下，附原因）\n")
    rows = []
    for key in registry["methods_without_evidence"]:
        spec = METHODS.get(key) or MODIFIERS[key]
        rows.append([f"`{key}`", spec.family, "无机器可读验证记录", spec.citation])
    parts.append(_table(["方法", "族", "状态", "文献出处"], rows))

    parts.append("\n### 3.4 图\n")
    captions = [
        "图 1 证据登记表：每个方法的实测误差（对数轴），虚线=owner 15% 精度标准，"
        "蓝色刻度=该算例自己的验收门。",
        "图 2 路由矩阵：算例画像 → 选定方法；绿底=证据背书（SELECTED），红底=无可选方法。",
    ]
    for figure, caption in zip(figures, captions):
        rel = figure.relative_to(ROOT)
        parts.append(f"![{figure.stem}]({rel.as_posix()})\n\n*{caption}*\n")
    return "\n".join(parts)


def compose_report(tables: str) -> str | None:
    if not NARRATIVE_PATH.exists():
        return None
    text = NARRATIVE_PATH.read_text()
    if MARKER not in text:
        raise SystemExit(f"narrative is missing the {MARKER} marker")
    return text.replace(MARKER, tables)


def dump(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                       help="compare regenerated documents to the committed ones")
    args = parser.parse_args(argv)

    registry = build_registry()
    matrix = build_matrix(registry)
    figures = [
        FIG_DIR / "evidence_registry.png",
        FIG_DIR / "routing_matrix.png",
    ]
    tables = build_tables(registry, matrix, figures)
    if not args.check:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        figure_evidence(registry, figures[0])
        figure_matrix(matrix, figures[1])

    registry_text = json.dumps(registry, indent=2, sort_keys=True) + "\n"
    matrix_text = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    report_text = compose_report(tables)

    outputs: list[tuple[Path, str]] = [
        (REGISTRY_PATH, registry_text),
        (MATRIX_PATH, matrix_text),
        (TABLES_PATH, tables + "\n"),
    ]
    if report_text is not None:
        outputs.append((REPORT_PATH, report_text))

    if args.check:
        drift = []
        for path, text in outputs:
            if not path.exists() or path.read_text() != text:
                drift.append(path.relative_to(ROOT).as_posix())
        if registry["missing_sources"]:
            drift.append(f"missing sources: {registry['missing_sources']}")
        if matrix["unexpected_routes"]:
            drift.append(f"unexpected routes: {matrix['unexpected_routes']}")
        if drift:
            print("DRIFT:", json.dumps(drift, ensure_ascii=False))
            return 1
        print("OK: registry, matrix, tables and report are up to date")
        return 0

    for path, text in outputs:
        dump(path, text)
        print("wrote", path.relative_to(ROOT).as_posix())
    print(f"records={len(registry['records'])} "
          f"evidenced_methods={registry['methods_with_evidence']} "
          f"unexpected_routes={matrix['unexpected_routes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
