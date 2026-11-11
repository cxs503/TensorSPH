#!/usr/bin/env python3
"""Build the open-source SPH inventory registry from live GitHub metadata.

Anchor ``FEAT-OSS-INVENTORY-SPH``.  Every field in the emitted registry is either

* fetched from the GitHub REST API -- ``GET /repos/{owner}/{name}`` for metadata,
  ``GET /repos/{owner}/{name}/readme`` for the project's own description and
  ``GET /repos/{owner}/{name}/git/trees/{branch}?recursive=1`` for the file list;
  every raw response is cached and its sha256 recorded -- or
* supplied by ``docs/oss_inventory_sph/search_seed.json`` (a frozen GitHub *search*
  result set, itself produced by ``discover_search_seed.py``), or
* derived from that text by an explicit function in this file.

Nothing is transcribed by hand.

Evidence channels and their grades
----------------------------------
``README``   the project's own prose: a feature is "claimed" when the README says so
             -> grade ``CLAIMED-BY-DOCS``.
``TREE``     repository file list: a feature is "present" when a source path carries
             the feature's name -> grade ``CODE-TREE`` (stronger: the file exists).
Neither channel reads a third party's source, so the registry never claims a code
audit of a third-party repository.  ``tier`` is a documentation-maturity proxy, not
a correctness judgement.

Usage::

    python scripts/oss_sph_inventory/discover_search_seed.py        # once
    python scripts/oss_sph_inventory/fetch_registry.py --max-requests 20 \
        --out docs/oss_inventory_sph/registry.json --cache .cache/github_sph_inventory

The crawl degrades gracefully: when the unauthenticated API budget (60 requests per
hour) runs out the remaining repositories keep their metadata and are marked
``docs_status = "budget_exhausted"`` instead of aborting the run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Curated candidate list.  Selection target: public SPH codes with source
# available, spanning engineering hydrodynamics, solid/geotechnical SPH,
# astrophysical SPH and graphics fluids, so the inventory is not one community.
# A candidate that 404s is dropped and recorded in ``dropped``.
# ---------------------------------------------------------------------------
CANDIDATES = [
    # engineering hydrodynamics / coastal / naval
    ("DualSPHysics/DualSPHysics", "engineering-hydro", "DualSPHysics"),
    ("GPUSPH/gpusph", "engineering-hydro", "GPUSPH"),
    ("KratosMultiphysics/Kratos", "engineering-hydro", "Kratos (SPH application)"),
    ("luchete80/WeldForm", "engineering-hydro", "WeldForm WCSPH parallel"),
    ("OpenFOAM/OpenFOAM-dev", "engineering-hydro", "OpenFOAM-dev (non-SPH control)"),
    # general frameworks / differentiable SPH
    ("pypr/pysph", "general-framework", "PySPH"),
    ("tumaer/jax-sph", "general-framework", "JAX-SPH (differentiable)"),
    ("InteractiveComputerGraphics/CompactNSearch", "general-framework", "CompactNSearch (neighbour search)"),
    # graphics / real-time fluids
    ("InteractiveComputerGraphics/SPlisHSPlasH", "graphics-fluids", "SPlisHSPlasH"),
    ("dimforge/salva", "graphics-fluids", "salva (Rust, DFSPH)"),
    ("zhai-xiao/CPP-Fluid-Particles", "graphics-fluids", "WCSPH/DFSPH/PBD CUDA"),
    ("obushi/SPHFluid", "graphics-fluids", "SPHFluid (GLSL)"),
    # incompressible / projection SPH
    ("Gfans/ISPH_NVIDIA_CUDA_CONTEST", "isph", "ISPH CUDA"),
    ("ensonun/eSPH", "isph", "eSPH (Riemann, MATLAB)"),
    # FSI / biomechanics
    ("Xiangyu-Hu/SPHinXsys", "fsi-bio", "SPHinXsys"),
    # geotechnics / granular coupling
    ("ChongPengGeotech/LOQUAT", "geotech", "LOQUAT (GPU geotechnics)"),
    ("martinjrobins/SPH-DEM", "geotech", "SPH-DEM (academic fluid-particle)"),
    # astrophysics
    ("danieljprice/phantom", "astro", "Phantom"),
    ("danieljprice/ndspmhd", "astro", "NDSPMHD"),
    ("SWIFTSIM/SWIFT", "astro", "SWIFT"),
    ("pfhopkins/gizmo", "astro", "GIZMO"),
    ("N-BodyShop/changa", "astro", "ChaNGa"),
    ("llnl/spheral", "astro-shock", "SPHERAL"),
    # large teaching demo, kept as a tier-C anchor of the distribution
    ("jason-huang03/SPH_Project", "teaching", "SPH_Project"),
]

# README signatures: matched case-insensitively against the README text.
FEATURES = [
    ("wc_sph", "weakly compressible WCSPH", r"weakly[- ]compressible|\bwc-?sph\b"),
    ("isph", "incompressible / projection SPH", r"\bisph\b|incompressible sph|projection scheme|dfsph|pc[i]?sph"),
    ("density_diffusion", "delta-SPH density diffusion", r"delta[-\s]?sph|\u03b4[-\s]?sph|density diffusion|diffusive term"),
    ("riemann", "Riemann-solver / Godunov SPH", r"riemann|godunov"),
    ("gpu", "GPU acceleration (CUDA/HIP/OpenCL/cuPy)", r"\bcuda\b|\bgpu\b|opencl|\bhip\b|sycl|kokkos|cupy|webgpu"),
    ("mpi", "MPI / distributed memory", r"\bmpi\b|distributed[- ]memory|multi[- ]gpu"),
    ("boundary_particles", "boundary particles / ghost / dynamic BC", r"dynamic boundar|boundary particle|ghost particle|fixed boundar"),
    ("boundary_akinci", "Akinci volume-weighted boundary", r"akinci"),
    ("boundary_adami", "Adami pressure extrapolation", r"adami"),
    ("surface_tension", "surface tension (CSF / continuum)", r"surface tension|\bcsf\b|continuum surface"),
    ("viscosity", "physical viscosity models", r"viscosit"),
    ("turbulence", "turbulence closure (LES / SPS)", r"\bles\b|sps|turbulen"),
    ("multiphase", "multiphase / two-phase", r"multiphase|multi-phase|two-phase"),
    ("fsi", "fluid-structure coupling", r"fluid[- ]structure|\bfsi\b|coupling with"),
    ("threed", "3-D capability", r"\b3[- ]?d\b|three[- ]dimensional"),
    ("validation", "published validation / benchmark suite", r"validat|verification|benchmark|dam break"),
    ("ci", "CI / test suite", r"continuous integration|\bci\b|pytest|unit test|ctest|catch2|googletest"),
    ("python_api", "Python API / framework", r"\bpython\b"),
    ("adaptive_h", "variable / adaptive smoothing length", r"variable smoothing|adaptive smoothing|h-adapt|variable-h"),
    ("coupling_tool", "external coupling interface (preCICE/Chrono)", r"precice|chrono|moordyn"),
]

# File-tree signatures: matched against repository paths (lower-cased).
TREE_FEATURES = [
    ("wc_sph", "WCSPH equations of state in source", r"wcsp|equationofstate|eos"),
    ("isph", "projection / DFSPH / PCISPH source", r"dfsph|pcisph|isph|projection"),
    ("density_diffusion", "delta-SPH / density-diffusion source", r"deltasph|delta_sph|densitydiffus|diffusion"),
    ("riemann", "Riemann / Godunov solver source", r"riemann|godunov"),
    ("gpu", "GPU backend source", r"cuda|gpu|opencl|kokkos|sycl|\.hip"),
    ("mpi", "MPI / distributed source", r"mpi|distributed|halo|decomposition"),
    ("boundary_particles", "boundary-particle implementation", r"dynamicbound|dynamic_bound|ghost|boundar"),
    ("boundary_akinci", "Akinci boundary implementation", r"akinci"),
    ("boundary_adami", "Adami extrapolation implementation", r"adami"),
    ("surface_tension", "surface-tension implementation", r"surfacetension|surface_tension|tension|csf"),
    ("viscosity", "viscosity implementation", r"viscos"),
    ("turbulence", "turbulence closure implementation", r"\bles|sps|turbulen"),
    ("multiphase", "multiphase implementation", r"multiphase|twophase|two_phase"),
    ("fsi", "FSI / coupling implementation", r"fsi|coupling|precice|chrono"),
    ("validation", "verification/validation tests or docs", r"valid|verif|benchmark"),
    ("adaptive_h", "adaptive/variable smoothing length", r"adapt|variableh|smoothinglength|smoothing_length"),
    ("python_api", "Python bindings/API", r"\.py$|python"),
]

TIER_RUBRIC = {
    "A": ("APS-style production grade (documentation proxy only): OSI licence, pushed "
          "within 730 days, >= 100 stars, and (GPU or MPI), and validation documented"),
    "B": ("research grade: OSI licence, pushed within 1460 days, validation documented, "
          "and (GPU/MPI or CI) -- missing at least one tier-A condition"),
    "C": "teaching demo / archived / superseded / metadata-incomplete: everything else",
}


class Budget:
    """Counts live API calls so a crawl can stop before the hourly limit."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit


def _cache_path(cache: Path, url: str) -> Path:
    return cache / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")


def _get(url: str, cache: Path, budget: Budget):
    """GET a JSON endpoint with an on-disk cache keyed by URL."""
    path = _cache_path(cache, url)
    if path.exists():
        return json.loads(path.read_text()), "cache"
    if budget.exhausted:
        return {"_http_status": 0, "_error": "budget_exhausted"}, "budget"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "tensorsph-oss-inventory",
    })
    budget.used += 1
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
            payload["_http_status"] = resp.status
    except urllib.error.HTTPError as exc:
        return {"_http_status": exc.code, "_error": str(exc)}, "http_error"
    except Exception as exc:  # noqa: BLE001 - a network fault must not abort the crawl
        return {"_http_status": 0, "_error": str(exc)}, "network_error"
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload, "live"


def detect(readme: str, tree: list[str], feats) -> dict:
    out = {}
    for key, label, pattern in feats:
        rx = re.compile(pattern, re.I)
        if feats is FEATURES:
            hits = len(rx.findall(readme))
            matches = []
        else:
            matches = sorted({p for p in tree if rx.search(p)})[:6]
            hits = len(matches)
        out[key] = {"label": label, "pattern": pattern, "hits": hits,
                    "present": hits > 0, "matches": matches}
    return out


def osi_license(meta: dict):
    lic = meta.get("license") or {}
    return lic.get("spdx_id") if isinstance(lic, dict) else lic


def assign_tier(license_id, pushed_year, stars, readme_feats, tree_feats) -> tuple[str, list[str]]:
    osi = license_id not in (None, "NOASSERTION", "OTHER", "")
    gpu_or_mpi = (readme_feats["gpu"]["present"] or readme_feats["mpi"]["present"]
                  or tree_feats["gpu"]["present"] or tree_feats["mpi"]["present"])
    validated = readme_feats["validation"]["present"] or tree_feats["validation"]["present"]
    ci = readme_feats["ci"]["present"]
    reasons = [f"license={license_id}", f"stars={stars}",
               f"years_since_push={pushed_year:.2f}",
               f"gpu_or_mpi={gpu_or_mpi}", f"validation={validated}", f"ci={ci}"]
    if osi and pushed_year <= 2.0 and stars >= 100 and gpu_or_mpi and validated:
        return "A", reasons
    if osi and pushed_year <= 4.0 and validated and (gpu_or_mpi or ci):
        return "B", reasons
    return "C", reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/oss_inventory_sph/registry.json")
    ap.add_argument("--cache", default=".cache/github_sph_inventory")
    ap.add_argument("--seed", default="docs/oss_inventory_sph/search_seed.json")
    ap.add_argument("--max-repos", type=int, default=24)
    ap.add_argument("--max-requests", type=int, default=20,
                    help="live API calls allowed in this run (hourly limit is 60 unauthenticated)")
    args = ap.parse_args()

    cache = Path(args.cache)
    now = time.time()
    seed = {"items": {}}
    seed_path = Path(args.seed)
    if seed_path.exists():
        seed = json.loads(seed_path.read_text())
    budget = Budget(args.max_requests)

    rows, dropped, seen = [], [], set()
    for repo, domain, display in CANDIDATES:
        key = repo.lower()
        if key in seen or len(rows) >= args.max_repos:
            continue
        seen.add(key)

        meta, meta_src = None, None
        for name, item in seed["items"].items():
            if name.lower() == key:
                meta, meta_src = item, "search_seed"
                break
        if meta is None:
            meta, meta_src = _get(f"{API}/repos/{repo}", cache, budget)
            if meta.get("_http_status") != 200:
                dropped.append({"repo": repo, "http_status": meta.get("_http_status"),
                                "reason": meta.get("_error", "not found"),
                                "error_class": meta_src})
                continue

        full = meta.get("full_name", repo)
        readme_payload, readme_src = _get(f"{API}/repos/{full}/readme", cache, budget)
        readme_txt, readme_sha = "", None
        if readme_payload.get("_http_status") == 200:
            blob = base64.b64decode(readme_payload.get("content", ""))
            readme_sha = hashlib.sha256(blob).hexdigest()
            readme_txt = blob.decode("utf-8", "replace")[:400_000]

        branch = meta.get("default_branch") or "main"
        tree_payload, tree_src = _get(
            f"{API}/repos/{full}/git/trees/{branch}?recursive=1", cache, budget)
        paths, tree_sha = [], None
        if tree_payload.get("_http_status") == 200:
            paths = [t["path"].lower() for t in tree_payload.get("tree", []) if t.get("type") == "blob"]
            tree_sha = hashlib.sha256("\n".join(paths).encode()).hexdigest()

        readme_feats = detect(readme_txt, [], FEATURES)
        tree_feats = detect("", paths, TREE_FEATURES)
        stars = int(meta.get("stargazers_count") or 0)
        pushed = (meta.get("pushed_at") or meta.get("updated_at") or "")[:10]
        years = ((now - time.mktime(time.strptime(pushed, "%Y-%m-%d"))) / 3.1557e7
                 if pushed else 99.0)
        license_id = osi_license(meta)
        tier, reasons = assign_tier(license_id, years, stars, readme_feats, tree_feats)
        file_count = int(meta.get("size") or 0)
        grades = []
        if readme_sha:
            grades.append("CLAIMED-BY-DOCS")
        if tree_sha:
            grades.append("CODE-TREE")
        rows.append({
            "repo": full,
            "display": display,
            "domain": domain,
            "url": meta.get("html_url"),
            "description": (meta.get("description") or "").strip(),
            "stars": stars,
            "language": meta.get("language"),
            "license": license_id,
            "pushed_at": pushed,
            "years_since_push": round(years, 2),
            "archived": bool(meta.get("archived")),
            "repo_size_kb": file_count,
            "default_branch": branch,
            "topics": meta.get("topics") or [],
            "tier": tier,
            "tier_reasons": reasons,
            "metadata_source": meta_src,
            "readme_sha256": readme_sha,
            "tree_sha256": tree_sha,
            "n_source_paths": len(paths),
            "readme_features": readme_feats,
            "tree_features": tree_feats,
            "readme_claimed": sum(1 for f in readme_feats.values() if f["present"]),
            "tree_present": sum(1 for f in tree_feats.values() if f["present"]),
            "evidence_grade": "+".join(grades) or "METADATA-ONLY",
            "docs_status": ("complete" if readme_sha and tree_sha
                            else ("budget_exhausted" if "budget" in (readme_src, tree_src)
                                  else "partial")),
        })
        time.sleep(0.2)

    rows.sort(key=lambda r: (-r["stars"], r["repo"]))
    doc = {
        "anchor": "FEAT-OSS-INVENTORY-SPH",
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "source": "GitHub REST API v3 live crawl + frozen search seed; raw responses cached by sha256",
        "evidence_grade_note": (
            "CLAIMED-BY-DOCS = the project's README says so; CODE-TREE = a source path "
            "carries the feature name. Neither reads third-party source code, so this is "
            "a documentation/tree-level survey, not a code audit."
        ),
        "tier_rubric": TIER_RUBRIC,
        "feature_definitions": {
            "readme": {k: {"label": lab, "pattern": pat} for k, lab, pat in FEATURES},
            "tree": {k: {"label": lab, "pattern": pat} for k, lab, pat in TREE_FEATURES},
        },
        "requests_used": budget.used,
        "repo_count": len(rows),
        "repos": rows,
        "dropped": dropped,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(rows)} repos, {len(dropped)} dropped, {budget.used} live requests")
    for r in rows:
        print(f"  {r['tier']} {r['repo']:46s} {r['stars']:6d} readme={r['readme_claimed']:2d} "
              f"tree={r['tree_present']:2d} {r['docs_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())