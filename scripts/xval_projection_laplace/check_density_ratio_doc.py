"""Drift gate for ``docs/xval_projection_laplace_density_ratio.md``.

The round-3/4 record is a *measurement* (``results/density_ratio.json``) plus a
table of numbers that a human copied out of it.  Hand-transcribed tables are the
part of such a record that rots: the JSON is regenerated, the doc is not, and
the published table silently stops matching the artifact it cites (the same
defect was logged for the graded gate receipts in m22936 and for the
TensorDG advection ladder).  This checker closes that hole for this deliverable:

    $ python scripts/xval_projection_laplace/check_density_ratio_doc.py --check

It parses the markdown tables and the sentence-level claims and compares each
displayed number against the JSON at the precision *actually printed* (a cell
printed with two decimals may differ by at most half a unit in the last place).
It also checks that the figure the artifact table advertises is really in the
tree and is a PNG.

Runs in well under a second -- it never re-solves the Poisson equation.  The
``density_ratio.py`` sweep is the only thing that produces the JSON; this file
only verifies that the document still tells the truth about it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "scripts" / "xval_projection_laplace" / "results"
JSON_PATH = RESULTS / "density_ratio.json"
DOC_PATH = ROOT / "docs" / "xval_projection_laplace_density_ratio.md"
FIG_PATH = RESULTS / "density_ratio.png"

MINUS = "\u2212"          # the doc uses a typographic minus
NUM_RE = re.compile(r"[-+\u2212]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


# --------------------------------------------------------------------------- #
# markdown helpers
# --------------------------------------------------------------------------- #
def tables(md: str):
    """Return every pipe table as ``(header_cells, [row_cells, ...])``."""
    out, block = [], []
    for line in md.splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            block.append(stripped)
            continue
        if len(block) >= 2:
            rows = [[c.strip() for c in b.strip("|").split("|")] for b in block]
            out.append((rows[0], [r for r in rows[1:]
                                  if not set("".join(r)) <= set("-: ")]))
        block = []
    return out


def display_value(cell: str):
    """First number in a table cell, with the precision it was printed at.

    ``**-0.21 %**`` -> ``(-0.21, 2)``; ``1 360 %`` -> ``(1360.0, 0)`` (the doc
    uses a thin space as a thousands separator); ``3.3e-14 %`` -> ``(3.3e-14, -13)``
    where a negative "decimals" means "scientific, compare on one digit".
    """
    s = (cell.replace("**", "").replace("`", "").replace("$", "")
         .replace(MINUS, "-").replace("\u00a0", " ").replace("%", ""))
    m = NUM_RE.search(s.replace(" ", "") if re.search(r"\d \d\d\d", s) else s)
    if not m:
        return None
    token = m.group(0)
    value = float(token)
    if "e" in token.lower():
        return value, -13
    decimals = len(token.split(".")[1]) if "." in token else 0
    return value, decimals


def close(displayed, value, decimals: int) -> bool:
    if decimals < 0:                      # scientific notation, one digit
        if value == 0.0:
            return abs(displayed) < 1e-12
        return abs(displayed - value) <= 0.06 * abs(value)
    tol = 0.5 * 10.0 ** (-decimals) + 1e-12 * max(1.0, abs(value))
    return abs(displayed - value) <= tol


# --------------------------------------------------------------------------- #
# the individual checks
# --------------------------------------------------------------------------- #
def rows_by(rows, axis, arm):
    return {(r["ratio"], r["R_over_dx"]): r for r in rows
            if r["arm"] == arm and r["axis"] == axis}


def check_ratio_ladder(payload, md, problems):
    """Table A: the density-ratio ladder at R/dx = 20 (density + naive arms)."""
    dens = rows_by(payload["density_ratio"], "density_ratio", "density")
    naive = rows_by(payload["density_ratio"], "density_ratio", "uniform")
    table = next((t for t in tables(md) if "density-weighted err" in t[0]), None)
    if table is None:
        problems.append("table A (density-weighted err) not found")
        return
    header, body = table
    for cells in body:
        ratio = int(float(cells[0].strip("* ")))
        d, n = dens[(ratio, 20.0)], naive[(ratio, 20.0)]
        for col, (name, value, key) in enumerate([
                ("dp", d["dp_measured_Pa"], "dp_measured_Pa"),
                ("density err", d["err_pct"], "err_pct"),
                ("naive err", n["err_pct"], "err_pct"),
                ("delta_s integral", d["delta_s_integral_dn"], "density_ratio"),
                ("kappaR", d["kappaR"], "kappaR")]):
            parsed = display_value(cells[col + 1])
            if parsed is None:
                problems.append(f"table A ratio {ratio}: unparsable {name} cell "
                                f"{cells[col + 1]!r}")
                continue
            got, dec = parsed
            if not close(got, value, dec):
                problems.append(f"table A ratio {ratio}: {name} says {got} , "
                                f"json {value:.6g} ({key})")


def check_resolution_ladder(payload, md, problems):
    """Table B: the resolution ladder of the ``density`` arm (the failed gate)."""
    dens = rows_by(payload["resolution"], "resolution", "density")
    table = next((t for t in tables(md) if t[0][0].strip("` ") == "R/dx"
                  and len(t[0]) == 4), None)
    if table is None:
        problems.append("table B (R/dx x ratio) not found")
        return
    _, body = table
    for cells in body:
        level = float(cells[0].strip("* "))
        for col, ratio in enumerate((1, 10, 1000)):
            parsed = display_value(cells[col + 1])
            if parsed is None:
                problems.append(f"table B R/dx={level:g}: unparsable cell "
                                f"{cells[col + 1]!r}")
                continue
            got, dec = parsed
            value = dens[(ratio, level)]["err_pct"]
            if not close(got, value, dec):
                problems.append(f"table B R/dx={level:g} ratio={ratio}: doc {got} ,"
                                f" json {value:.6g}")


VERDICT_ROWS = {
    "`dp`, ratio 1000, `R/dx = 20`": 20.0,
    "`dp`, ratio 1000, `R/dx = 30/40`": (30.0, 40.0),
}


def check_verdict_tables(payload, md, problems):
    """The §④ (density arm) and §⑤.2 (force-density arm) verdict tables."""
    res = payload["resolution"]
    arms = {"\u2463": "density", "\u2464": "force_density"}
    seen = 0
    for section in re.split(r"\n## ", md):
        heading = section.splitlines()[0] if section.splitlines() else ""
        arm = next((a for mark, a in arms.items() if heading.startswith(mark)), None)
        if arm is None:
            continue
        table = next((t for t in tables(section) if "best measured" in "".join(t[0])),
                     None)
        if table is None:
            continue
        seen += 1
        _, body = table
        by_level = {(r["ratio"], r["R_over_dx"]): r["err_pct"] for r in res
                    if r["arm"] == arm}
        for cells in body:
            levels = VERDICT_ROWS.get(cells[0].strip())
            if levels is None:
                continue
            levels = levels if isinstance(levels, tuple) else (levels,)
            values = [float(c.replace(MINUS, "-"))
                      for c in re.findall(r"[-\u2212]?\d+(?:\.\d+)?", cells[2])]
            if len(values) != len(levels):
                problems.append(f"{arm} verdict row {cells[0].strip()}: "
                                f"{len(values)} numbers in {cells[2]!r}, "
                                f"expected {len(levels)}")
                continue
            for got, level in zip(values, levels):
                want = by_level[(1000, level)]
                if not close(got, want, 2):
                    problems.append(f"{arm} verdict R/dx={level:g}: doc {got} , "
                                    f"json {want:.4f}")
    if seen < 2:
        problems.append(f"expected 2 verdict tables (\u2463 density, \u2464 "
                        f"force-density), found {seen}")


def check_claims(payload, md, problems):
    """Sentence-level numbers, each tied to one JSON quantity."""
    s = payload["summary"]
    md = re.sub(r"\s+", " ", md)          # the doc wraps its lines
    claims = [
        # (regex with one numeric group, value, decimals)
        (r"ratio spread \*\*([-\d.]+) %\*\*", s["ratio_spread_pct"], 2),
        (r"spread over the whole ladder is \*\*([\d.eE+-]+) %\*\*",
         s["flux_spread_pct"], -13),
        (r"max \|err\| ([\d.]+) %\)", s["flux_err_pct_max_abs"], 2),
        (r"spread \*\*([\d.]+) %\*\*, max \|err\| ([\d.]+) %",
         s["translation_spread_pct"], 2),
        # the interface-invariance control (m23102): both numbers are measured
        (r"cell-volume spread \*\*([\d.eE+-]+)\*\*",
         s["cell_volume_rel_spread"], -13),
        (r"cell area \(\*\*([\d.eE+-]+)\*\* ppm", s["cell_volume_vs_analytic_ppm"], -13),
        (r"colour-field difference \*\*([\d.eE+-]+)\*\*",
         s["color_max_abs_diff_max"], -13),
    ]
    for pattern, value, dec in claims:
        m = re.search(pattern, md)
        if m is None:
            problems.append(f"claim not found: /{pattern}/")
            continue
        got = float(m.group(1).replace(MINUS, "-"))
        if not close(got, value, dec):
            problems.append(f"claim /{pattern}/ says {got} , json {value:.6g}")
    # second number of the translation claim
    m = re.search(r"spread \*\*[\d.]+ %\*\*, max \|err\| ([\d.]+) %", md)
    if m and not close(float(m.group(1)), s["translation_max_abs_err_pct"], 2):
        problems.append("translation max |err| disagrees with the json")


def check_arm_explicit_keys(payload, problems):
    """The summary must name the arm of every resolution map (m23102)."""
    s = payload["summary"]
    if "resolution_err_pct_by_ratio" in s:
        problems.append("ambiguous summary key resolution_err_pct_by_ratio is back")
    for arm in ("density", "force_density"):
        key = f"resolution_err_pct_by_ratio_{arm}"
        if key not in s:
            problems.append(f"summary key {key} missing")
    if "cell_volume_vs_analytic_ppm" not in s or "color_max_abs_diff_max" not in s:
        problems.append("interface-invariance control missing from the summary")


B64_PATH = FIG_PATH.with_name(FIG_PATH.name + ".b64")


def figure_bytes():
    """The shipped figure as PNG bytes, from the PNG or its ASCII copy."""
    if FIG_PATH.exists():
        return FIG_PATH.read_bytes()
    if B64_PATH.exists():
        import base64
        return base64.b64decode(B64_PATH.read_text().strip(), validate=True)
    return None


def check_figure(problems):
    """The artifact table advertises a figure; it has to exist and be a PNG.

    The PNG or its base64 copy is accepted: the delivery gate filters binary
    paths, so round 3 merged with the PNG stripped out (and this checker, and a
    pytest gate, merged expecting it -- main was red).  The text copy is the
    transport, so the gate follows it.
    """
    blob = figure_bytes()
    if blob is None:
        try:
            shown = FIG_PATH.relative_to(ROOT)
        except ValueError:          # a figure path outside the repo (tests)
            shown = FIG_PATH.name
        problems.append(
            f"figure {shown} and {B64_PATH.name} both missing "
            "(regenerate: python scripts/xval_projection_laplace/"
            "density_ratio.py --figure-only)")
        return
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        problems.append("density_ratio.png is not a PNG")
    if len(blob) < 20_000:
        problems.append(f"density_ratio.png suspiciously small ({len(blob)} B)")
    elif not blob.endswith(b"IEND\xaeB`\x82"):
        # a base64 copy can be interrupted mid-write and still start with the
        # PNG magic; the IEND chunk is what says the raster is complete.
        problems.append("density_ratio.png truncated (no IEND chunk)")


def check(json_path=JSON_PATH, doc_path=DOC_PATH):
    """Return the list of disagreements between the document and the JSON."""
    problems: list[str] = []
    payload = json.loads(json_path.read_text())
    md = doc_path.read_text()
    check_ratio_ladder(payload, md, problems)
    check_resolution_ladder(payload, md, problems)
    check_verdict_tables(payload, md, problems)
    check_claims(payload, md, problems)
    check_arm_explicit_keys(payload, problems)
    check_figure(problems)
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 on any doc/JSON disagreement (CI mode; "
                         "without it the findings are only printed)")
    args = ap.parse_args(argv)
    problems = check()
    if problems:
        print(f"FAIL  {len(problems)} disagreement(s) between "
              f"{DOC_PATH.name} and {JSON_PATH.name}")
        for problem in problems:
            print("  -", problem)
        return 1 if args.check else 0
    print("OK    doc tables, claims, summary keys and figure all match "
          f"{JSON_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
