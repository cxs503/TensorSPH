#!/usr/bin/env python3
"""Freeze the GitHub *search* result set used to discover extra SPH repositories.

The search API returns full repository objects, so caching them here lets
``fetch_registry.py`` add those repositories to the inventory without spending a
second API request per repository on ``GET /repos/{owner}/{name}``.  Only the
README still has to be fetched per repository.

    python scripts/oss_sph_inventory/discover_search_seed.py
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

QUERIES = [
    "smoothed particle hydrodynamics",
    "SPH fluid simulation",
    "SPH solver",
]
OUT = Path(__file__).resolve().parents[2] / "docs" / "oss_inventory_sph" / "search_seed.json"


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "tensorsph-oss-inventory",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    items: dict[str, dict] = {}
    counts = {}
    for q in QUERIES:
        url = ("https://api.github.com/search/repositories?q="
               + urllib.parse.quote(q) + "&sort=stars&per_page=25")
        payload = get(url)
        got = payload.get("items", [])
        counts[q] = len(got)
        for it in got:
            items.setdefault(it["full_name"], it)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"queries": QUERIES, "counts": counts, "items": items},
                              indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(items)} unique repositories from {len(QUERIES)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())