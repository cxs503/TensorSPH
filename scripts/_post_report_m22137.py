import base64, json, urllib.request

TOKEN = open('/tmp/token.txt').read().strip()
BASE = 'http://127.0.0.1:8010/api/v1'
ROOT = '/root/tp-agent/work/m22137/'
png = base64.b64encode(open(ROOT + 'benchmarks/surface_tension/out_single_phase_completion_transfer/transfer.png', 'rb').read()).decode()
md = open(ROOT + 'docs/xval_sphinxys_mixture_laplace/single_phase_completion_transfer.md').read()
payload = {
    "title": "TensorSPH m22137: single-phase CSS interface-completion closure - lattice transfer vs calibration scope",
    "summary": "Round-2 of the single-phase surface-tension closure: the fitted weight transfers quantitatively to hexagonal and offset lattices (worst +6.6% vs the 15% gate) but shifts by -18.4% between lattices, so it is a discretisation calibration; a 0.35-period time-integration probe shows the static repair does not survive in a WCSPH time loop while the CSF control does.",
    "scenario_key": "XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION",
    "mission_id": 22137,
    "tags": ["TensorSPH", "surface-tension", "SPHinXsys", "CSS", "Young-Laplace", "verification", "m22137"],
    "markdown": md,
    "figures": [{
        "name": "single_phase_closure_transfer.png",
        "caption": "Left: round-2 closure weight on unseen lattices (S=square, H=hex, *=offset). Middle: closure scan per lattice; the null weight moves 0.426 (square) -> 0.348 (hex). Right: same force projected on the Rayleigh mode-2 shape (one-step indicator, no pressure feedback).",
        "png_base64": png,
    }],
}
body = json.dumps(payload).encode()
req = urllib.request.Request(BASE + '/tensor/reports', data=body,
                             headers={'Authorization': 'Bearer ' + TOKEN,
                                      'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        print(r.status, r.read()[:400].decode(errors='replace'))
except Exception as e:
    print('ERR', type(e).__name__, getattr(e, 'read', lambda: b'')()[:400])