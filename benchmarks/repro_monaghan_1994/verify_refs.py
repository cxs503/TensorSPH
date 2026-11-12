"""Bibliographic verification for REPRO-SPH-MONAGHAN1994.

The anchor names the paper by title/journal/pages, not by DOI, so the DOI is
*resolved* here instead of being typed in from memory: Crossref is searched with
the bibliographic string, the candidates are recorded verbatim, and the record
that matches the anchor's citation (volume 110, pages 399-406, 1994) is selected
by rule.  OpenAlex is then queried at the resolved DOI, so the open-access status
of the paper that was actually identified is what gets recorded.

Nothing is filled in by hand: a failed query is recorded as failed, and a
non-matching candidate list is recorded as a mismatch.

Usage::

    python benchmarks/repro_monaghan_1994/verify_refs.py \
        --json-out benchmarks/repro_monaghan_1994/results/refs_verified.json
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The citation the feature-ledger anchor gives, verbatim.
CLAIM = {
    "title": "Simulating Free Surface Flows with SPH",
    "author_family": "Monaghan",
    "journal": "J. Comput. Phys.",
    "volume": "110",
    "pages": "399-406",
    "year": "1994",
}
CROSSREF_SEARCH = "https://api.crossref.org/works?" + urllib.parse.urlencode({
    "query.bibliographic": CLAIM["title"],
    "query.author": CLAIM["author_family"],
    "filter": "from-pub-date:1994-01-01,until-pub-date:1994-12-31",
    "rows": "8",
    "select": "DOI,title,container-title,volume,page,author,published-print,abstract",
})
TIMEOUT = 25


def _get(url: str) -> dict:
    """GET ``url`` and return ``{status, url, body, error}``; never raise."""
    req = urllib.request.Request(url, headers={"User-Agent": "tensorsph-repro/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            return {"status": int(response.status), "url": url,
                    "body": json.loads(response.read().decode())}
    except urllib.error.HTTPError as exc:
        return {"status": int(exc.code), "url": url, "body": None}
    except Exception as exc:
        return {"status": None, "url": url, "body": None, "error": repr(exc)}


def _first(value):
    return value[0] if isinstance(value, list) and value else value


def _families(message: dict) -> list[str]:
    return [(a.get("family") or "").lower() for a in message.get("author", [])]


def _score(message: dict) -> tuple[int, dict]:
    """How many of the anchor's citation fields this candidate reproduces."""
    hits = {}
    title = (_first(message.get("title")) or "").lower()
    hits["title_tokens"] = sum(
        1 for token in ("simulating", "free", "surface", "flows", "sph")
        if token in title)
    hits["author"] = CLAIM["author_family"].lower() in _families(message)
    hits["volume"] = str(message.get("volume")) == CLAIM["volume"]
    hits["pages"] = str(message.get("page")) == CLAIM["pages"]
    year = ((message.get("published-print") or {}).get("date-parts") or [[None]])[0][0]
    hits["year"] = str(year) == CLAIM["year"]
    # volume/pages/author identity dominate; title tokens break ties.
    score = (10 * hits["author"] + 10 * hits["volume"] + 10 * hits["pages"]
             + 5 * hits["year"] + hits["title_tokens"])
    return score, hits


def _summarise(message: dict) -> dict:
    score, hits = _score(message)
    return {
        "doi": message.get("DOI"),
        "title": _first(message.get("title")),
        "container": _first(message.get("container-title")),
        "volume": message.get("volume"),
        "page": message.get("page"),
        "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in message.get("author", [])],
        "published": ((message.get("published-print") or {}).get("date-parts")
                      or [[None]])[0],
        "has_abstract": bool(message.get("abstract")),
        "abstract": message.get("abstract"),
        "match_score": score,
        "match_fields": hits,
    }


def _openalex(doi: str) -> dict:
    url = f"https://api.openalex.org/works/doi:{doi}"
    raw = _get(url)
    fields: dict = {}
    if raw["body"]:
        body = raw["body"]
        index = body.get("abstract_inverted_index") or {}
        abstract = None
        if index:
            positions = sorted((pos, word) for word, pos_list in index.items()
                               for pos in pos_list)
            abstract = " ".join(word for _, word in positions)
        fields = {
            "title": body.get("title"),
            "publication_year": body.get("publication_year"),
            "open_access": (body.get("open_access") or {}).get("is_oa"),
            "oa_status": (body.get("open_access") or {}).get("oa_status"),
            "has_abstract": abstract is not None,
            "abstract": abstract,
        }
    return {"status": raw["status"], "url": url, "fields": fields,
            "error": raw.get("error")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=str(
        Path(__file__).resolve().parent / "results" / "refs_verified.json"))
    parser.add_argument("--min-score", type=int, default=40,
                        help="minimum match score for the resolution to count as a match")
    args = parser.parse_args()

    search = _get(CROSSREF_SEARCH)
    items = ((search["body"] or {}).get("message", {}).get("items") or [])
    candidates = [_summarise(item) for item in items]
    best = max(candidates, key=lambda c: c["match_score"]) if candidates else None
    resolved = bool(best and best["match_score"] >= args.min_score)

    payload = {
        "benchmark": "REPRO-SPH-MONAGHAN1994",
        "anchor_citation_claim": CLAIM,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crossref_search": {
            "status": search["status"],
            "url": CROSSREF_SEARCH,
            "error": search.get("error"),
            "n_candidates": len(candidates),
            "candidates": candidates,
            "selected": best,
            "resolved": resolved,
            "rule": ("highest score; score = 10*(author) + 10*(volume) + 10*(pages) "
                     "+ 5*(year) + (title token matches); resolved iff score >= "
                     f"{args.min_score}"),
        },
        "openalex": _openalex(best["doi"]) if resolved else {"status": None,
                                                             "fields": {},
                                                             "skipped": "DOI not resolved"},
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"crossref search HTTP {search['status']}: {len(candidates)} candidates")
    for cand in candidates[:4]:
        print(f"  [{cand['match_score']:2d}] {cand['doi']} {cand['title']!r} "
              f"{cand['container']} {cand['volume']}:{cand['page']} {cand['published']}")
    if resolved:
        print(f"selected DOI {best['doi']} (score {best['match_score']}, "
              f"fields {best['match_fields']})")
        print(f"openalex HTTP {payload['openalex']['status']}: "
              f"oa_status={payload['openalex']['fields'].get('oa_status')!r} "
              f"abstract={'present' if payload['openalex']['fields'].get('has_abstract') else 'absent'}")
    else:
        print("DOI NOT RESOLVED: no candidate reproduced the anchor's citation fields")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())