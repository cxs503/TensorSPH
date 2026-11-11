#!/usr/bin/env python3
"""Probe TensorSPH's own tree for the features the OSS inventory surveys.

The output is the *local* half of the gap analysis: for every feature in the
inventory's vocabulary, where (if anywhere) does this repository implement it,
which tests guard it, and which claim/mission already owns it.  All of that is
computed by scanning files -- nothing is asserted from memory.

    python scripts/oss_sph_inventory/probe_tensorsph.py

Writes ``docs/oss_inventory_sph/tensorsph_capability.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "oss_inventory_sph" / "tensorsph_capability.json"

# feature key -> (description, [regexes searched in source files], [path hints])
PROBES = {
    "wc_sph": ("weakly compressible SPH core (Tait EOS / density summation)",
               [r"\bwcsph\b", r"tait", r"rho0\s*\*\*\s*7|\(rho\s*/\s*rho0\)\s*\*\*\s*7"], ["src/tensorsph/wcsph"]),
    "isph": ("incompressible / projection (ISPH, DFSPH, PCISPH)",
             [r"\bisph\b", r"\bdfsph\b", r"\bpcisph\b", r"pressure[-_ ]projection"],
             []),
    "density_diffusion": ("delta-SPH style density diffusion",
                          [r"density_diffusion", r"delta_sph", r"diffus(ive|ion)_?_?term"],
                          []),
    "riemann": ("Riemann-solver / Godunov flux inside the SPH solver",
                [r"riemann[_ ](solver|flux)", r"godunov\w*[_ ]sph", r"riemann[_ ]based"],
                []),
    "gpu": ("GPU backend reachable from the solver",
            [r"cuda", r"torch\.device", r"device\s*=\s*\"?cuda"], []),
    "mpi": ("MPI / multi-node distributed execution",
            [r"\bmpi4py\b", r"\bmpi\b"], []),
    "boundary_particles": ("wall / boundary particle treatment",
                           [r"wall_lattice", r"boundary_particle", r"ghost_particle"], []),
    "boundary_akinci": ("Akinci volume-weighted boundary",
                        [r"akinci"], []),
    "boundary_adami": ("Adami pressure extrapolation at walls",
                       [r"adami"], []),
    "surface_tension": ("surface tension (CSF / continuum surface force)",
                        [r"surface_tension", r"\bcsf\b"], []),
    "viscosity": ("physical viscosity models",
                  [r"viscosity"], []),
    "turbulence": ("turbulence closure (LES / SPS / Smagorinsky)",
                   [r"sps[_ ]?(stress|model|tensor)", r"les[_ ]?(model|coefficient|viscosit)",
                    r"smagorinsky", r"sub[- ]particle[- ]scale"], []),
    "multiphase": ("multiphase / two-phase",
                   [r"two_phase", r"multiphase", r"multi_phase"], []),
    "fsi": ("fluid-structure interaction coupling",
            [r"\bfsi\b", r"fluid_structure", r"added_mass", r"ice_floe", r"wave_ice"], []),
    "adaptive_h": ("variable / adaptive smoothing length h(t)",
                   [r"adaptive_h\b", r"variable_smoothing", r"h_adapt"], []),
    "coupling_tool": ("external coupling interface (preCICE/Chrono/MoorDyn)",
                      [r"precice", r"chrono", r"moordyn"], []),
    "validation": ("verification / validation assets",
                   [r"verification", r"benchmark"], ["benchmarks"]),
    "ci": ("automated test suite executed in CI",
           [r"pytest"], ["tests", ".github/workflows"]),
}

# Statements in the code that literally document the *absence* of a model must not
# be counted as an implementation: a docstring saying "no LES or SGS model" is
# evidence of a gap, not of a feature.
ABSENT_MARKERS = [
    r"\bno (les|sgs|turbulence|riemann|multiphase|two-phase|isph|mpi|akinci|coupling)\b",
    r"nothing here is",
    r"\bnot (a|an) [a-z-]* ?(model|solver)",
    r"is not modelled|is not modeled|are not modelled|are not modeled",
    r"\bno [a-z-]* ?(model|closure|term)\b",
]

SRC_DIRS = ["src"]
SPECIAL_DIRS = {"validation": ["benchmarks"], "ci": ["tests"]}


def _iter_files(roots: list[Path]):
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".yml", ".yaml", ".sh"}:
                yield path


def probe() -> dict:
    src_files = list(_iter_files([ROOT / "src"]))
    test_files = list(_iter_files([ROOT / "tests"])) if (ROOT / "tests").exists() else []
    bench_files = list(_iter_files([ROOT / "benchmarks"])) if (ROOT / "benchmarks").exists() else []
    doc_files = list(_iter_files([ROOT / "docs"])) if (ROOT / "docs").exists() else []

    out = {}
    for key, (desc, patterns, hints) in PROBES.items():
        rxs = [re.compile(p, re.I) for p in patterns]
        hits_src, hits_tests, hits_bench, hits_doc = [], [], [], []
        absent_rx = [re.compile(p, re.I) for p in ABSENT_MARKERS]
        filtered_out = []
        for path, bucket in ((p, hits_src) for p in src_files):
            text = path.read_text(errors="replace")
            for rx in rxs:
                m = rx.search(text)
                if m:
                    line = text[:m.start()].count("\n") + 1
                    snippet = text.splitlines()[line - 1].strip()
                    if any(a.search(snippet) for a in absent_rx):
                        filtered_out.append({
                            "file": str(path.relative_to(ROOT)), "line": line,
                            "pattern": rx.pattern, "reason": "line documents the model's absence",
                            "snippet": snippet[:120]})
                        break
                    bucket.append({
                        "file": str(path.relative_to(ROOT)),
                        "line": line,
                        "pattern": rx.pattern,
                        "snippet": snippet[:120],
                    })
                    break
        for path, bucket in ((p, hits_tests) for p in test_files):
            text = path.read_text(errors="replace")
            if any(rx.search(text) for rx in rxs):
                bucket.append({"file": str(path.relative_to(ROOT))})
        for path, bucket in ((p, hits_bench) for p in bench_files):
            text = path.read_text(errors="replace")
            if any(rx.search(text) for rx in rxs):
                bucket.append({"file": str(path.relative_to(ROOT))})
        for path, bucket in ((p, hits_doc) for p in doc_files):
            text = path.read_text(errors="replace")
            if any(rx.search(text) for rx in rxs):
                bucket.append({"file": str(path.relative_to(ROOT))})
        out[key] = {
            "description": desc,
            "patterns": patterns,
            "path_hints": hints,
            "src_hits": hits_src,
            "absent_marker_hits": filtered_out,
            "test_hits": hits_tests,
            "benchmark_hits": hits_bench,
            "doc_hits": hits_doc[:8],
            "in_source": bool(hits_src),
            "guarded_by_test": bool(hits_tests),
            "in_benchmarks": bool(hits_bench),
        }
    return {
        "generated_from": "src/, tests/, benchmarks/, docs/ of this worktree",
        "scan_counts": {"src": len(src_files), "tests": len(test_files),
                        "benchmarks": len(bench_files), "docs": len(doc_files)},
        "features": out,
    }


def main() -> int:
    data = probe()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")
    for key, row in data["features"].items():
        print(f"  {key:20s} src={len(row['src_hits']):3d} tests={len(row['test_hits']):2d} "
              f"bench={len(row['benchmark_hits']):2d} docs={len(row['doc_hits']):2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())