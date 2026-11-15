import json, os, pathlib, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        print("HTTP", exc.code, body[:1500])
        raise


tok = post(f"{BASE}/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token")
if not token:
    raise SystemExit("no token: " + json.dumps(tok)[:300])

doc = pathlib.Path("docs/verification_kleefsman_3d_blockage_m21728.md").read_text()

head = """# 3-D Kleefsman obstacle case: restoring the blockage the 2-D model cannot represent

**Summary.** The committed Kleefsman case is 2-D, so its 0.161 m box spans the full 1.00 m tank width
(100 % blockage) while the experiment blocks 40 % and lets 60 % of the water bypass along the box side.
Every 2-D verdict of this case failed the peak metric by 15-52 % (m21324, m21586), and m21586
*attributed* that residual to the blockage without ever measuring it. This shift delivers the
measurement instrument: `tensorsph.kleefsman_obstacle_3d`, the same solver with the tank's third
dimension restored, the box at its real 0.403 m width, the identical frozen gauge protocol, and an
explicit bypass count that is identically zero in 2-D.

## Case and method

Apparatus: tank 3.22 x 1.00 x 1.00 m, water column 1.228 x 0.550 m across the full width, box
0.161 x 0.161 x 0.403 m with its upstream face at x = 2.476 m (1.228 + 1.248 + 0.744 = 3.220 m,
residual 0). Gauge protocol identical to the 2-D case: slab, 0.08 m aperture, station 0.05 m upstream
of the box face (x = 2.426), 0.02 m arrival threshold. In 3-D the slab is a volume gauge,
`h = sum(m/rho0)/(aperture x blocked_width)`, which reduces exactly to the 2-D areal operator, so
"water height" means the same thing on both sides of the comparison. The box is a solid particle block
plus a hard non-penetration clamp that counts the particles it moves: a particle inside the box would
corrupt the gauge, so non-penetration is made an invariant, not an expectation.

## Results

Measured: a dx = 0.05 m self-check (5280 fluid / 10160 wall particles, dt = 8.775e-4 s) integrated to
t = 0.351 s with slab = 0.000 m, i.e. arrival > 0.351 s, mass drift 0, boundary escapes 0, box clamps
0. The 2-D family gives arrival 0.438-0.484 s against the experiment's 0.3591 s, so the 3-D self-check
sits in the same band: geometry and gauges are wired correctly.

**Not measured: the acceptance metrics.** The two scored runs were submitted and were still queued at
hand-off, so the verdict is recorded as WITHHELD.

| job | grid | t_end | state at hand-off |
|---|---|---|---|
| cj3661 | dx = 0.04 m, 8 cores | 2.6 s | queued 29 min (79 jobs in flight) |
| cj3662 | dx = 0.03 m, 8 cores | 2.6 s | queued |

## Figures: none, and why

Every figure depends on the time-series CSVs of the two queued runs.
`benchmarks/kleefsman_obstacle_3d/make_doc.py` is written and ready (H2 time histories 2-D/3-D/
experiment overlaid, per-run error bars with the 10 % bound, and the bypass-count history), but no
placeholder image is substituted for a result. A field contour of a half-integrated transient would be
a fabricated figure, not a measurement.

## Notes and limits

1. The production runs were not recovered - the shift's main unfinished item.
2. The box's lateral position is an assumption: the paper states only the 0.403 m width and the setup
   figure is a raster, so its spanwise coordinate is not machine-readable. It is a parameter
   (`obstacle_z`, default 0 = flush with a side wall, one 0.597 m bypass channel); the centred
   placement is one keyword away, and because the bypass topology differs, both should be run before
   any verdict.
3. Grid discretisation of the box width: 0.403 m is 10 cells at dx = 0.04 (0.40 m, 0.7 % narrow) and
   13 cells at dx = 0.03 (0.39 m, 3.2 % narrow). The gauge denominator keeps the experimental 0.403 m
   and the run JSON also records `obstacle_width_on_lattice_m`, so the discrepancy is visible.
4. Walls are free-slip (solver default); the experimental boundary layer would slow the 3-D front.
5. The "any metric outside 10 % is FAIL" discipline is unchanged; being closer to experiment does not
   buy a wider bound.
6. The dx = 0.05 self-check showed a transient 33.8 m/s excursion (2-D dx = 0.04: 5.98 m/s) during the
   initial collapse, with compression 0.163 vs the 2-D 0.172. Coarse-grid splash noise, not used as a
   result, to be re-checked when the production runs come back.

## Error judgement

| key quantity | reference (Kleefsman Fig. 19) | this shift | rel. error | < 3 %? |
|---|---|---|---|---|
| H2 peak water height | 0.370 m | not obtained | not obtained | cannot be claimed |
| H2 arrival (0.02 m) | 0.3591 s | not obtained (known > 0.351 s) | not obtained | cannot be claimed |

**Explicit answer to "is it < 3 %": no claim is possible.** The shift produced no comparable 3-D peak or
arrival time. Neither < 3 % nor < 10 % can be asserted. What can be said is only a consistency
statement: at t = 0.351 s the gauge is still dry, the same order as the 2-D family (0.438-0.484 s),
which rules out a unit or geometry mis-wiring but says nothing about accuracy.

## Verification run

`python -m pytest tests/test_kleefsman_obstacle_3d.py -q` -> 26 passed. Assertions are invariants: the
clamp empties the box and zeroes only the normal component of the face it exited through, and is
idempotent; the slab gauge reproduces 3*4*8*dx^3/(0.08*0.403) m to 1e-12; water that bypassed the box
reads zero; a run peaking on the last samples of a record that stops before 1.703 s is censored and
cannot pass. Existing case tests: `pytest tests/ -k kleefsman -m "not slow"` -> 264 passed, 5 failed,
all 5 in test_kleefsman_text_landmarks.py because the uncommitted artefact
outputs/kleefsman2005_text/quotes.json is missing - unrelated to these changes (new files only). Three
pre-existing collection errors between benchmarks/xval_sphinx_tvc/* and benchmarks/case_poiseuille/run.py
(bare-name `run` module collision) are untouched by this shift.

---

"""

body = {
    "title": "3-D Kleefsman obstacle case: measuring 40% vs 100% blockage (verdict WITHHELD)",
    "summary": (
        "Added tensorsph.kleefsman_obstacle_3d: the Kleefsman obstacle case in 3-D with the box at "
        "its real 0.403 m width (40% blockage vs the 2-D model's 100%), identical frozen gauge "
        "protocol and reference, plus a bypass count that is identically zero in 2-D. 26 tests pass "
        "on invariants. Both scored runs (cj3661, cj3662) were still queued at hand-off, so the "
        "verdict is WITHHELD and no peak/arrival error is claimed."
    ),
    "scenario_key": "sph_free_surface_impact",
    "mission_id": 21728,
    "tags": ["SPH", "Kleefsman", "dam-break", "obstacle", "3D", "blockage", "verification", "WITHHELD"],
    "markdown": head + doc,
    "content": head + doc,
    "figures": [],
}
res = post(f"{BASE}/tensor/reports", body, token)
pathlib.Path("out").mkdir(exist_ok=True)
pathlib.Path("out/report_post_m21728.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res)[:500])