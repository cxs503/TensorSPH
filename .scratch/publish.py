import base64, json, os, sys, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "agent")
PASS = os.environ.get("TENSOR_AGENT_PASS", "agent")


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + token} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def login():
    for path in ("/api/v1/auth/login", "/tensor/auth/login", "/auth/login"):
        try:
            out = post(path, {"username": USER, "password": PASS})
            tok = out.get("access_token") or out.get("token") or (out.get("data") or {}).get("access_token")
            if tok:
                return tok
        except Exception as exc:  # noqa: BLE001
            print("login try", path, "->", exc, file=sys.stderr)
    raise SystemExit("no token")


MARKDOWN = open("/root/tp-agent/work/m2737/docs/verification_delta_s_normalization.md").read()
extra = """

## 附：本单交付与验证

- 代码：`src/tensorsph/csf_consistent.py`（新增 `delta_s_normalization="fixed"|"jump"` + `DELTA_S_SCHEMES`）、
  `src/tensorsph/droplet_impact.py`（`Config.csf_delta_s` + 校验）、`src/tensorsph/surface_tension.py`（贯通）。
- 工件：`benchmarks/csf_delta_s/{run.py, run_relax.py, make_doc.py, results.json, results_relax.json, delta_s_normalization.png}`、
  `docs/verification_delta_s_normalization.md`。
- 测试：`tests/test_csf_delta_s_normalization.py`（8 条，全绿）；全套 `pytest -q -m "not slow"` =
  264 passed / 7 failed，7 条失败与本单无关（HEAD 上同样失败，见 baseline 记录，
  其中 4 条是 `test_laplace_droplet.py` 的过时 pinned-failure）。
- 复现：`PYTHONPATH=src python benchmarks/csf_delta_s/run.py --figure`（秒级）、
  `PYTHONPATH=src python benchmarks/csf_delta_s/run_relax.py`（~2 min）。
"""

payload = {
    "title": "CSF 表面张力 delta 函数归一化核验：旧「切割力达标」是相消误差，实测跳变归一使两条力判据同时进门",
    "summary": ("TensorSPH：定位 int delta_s dn = 0.74 的根因为离散色函数跨度实为 0.37（c_surf~0.63）而非硬编码假设的 0.5；"
                "改为按实测跳变归一后 int delta_s dn = 0.99-1.02，几何口径切割力 0.95-1.03（原 0.71-0.79）。"
                "证明旧「切割力 10.7% 在门内」系 delta_s 弱 26% 与界面密度亏损放大 25% 相消；默认方案保持不变（附实测理由：+32% 动量注入使伪流 0.048->0.147 超 Adami 阈值）。新增 8 条测试全绿，全套仅存 7 条与本单无关的既存失败。"),
    "scenario_key": "tensorsph-csf-delta-s-normalization",
    "mission_id": 2737,
    "tags": ["TensorSPH", "surface-tension", "CSF", "verification", "Young-Laplace", "delta-function", "honest-FAIL"],
    "markdown": MARKDOWN + extra,
    "figures": [{
        "name": "delta_s_normalization.png",
        "caption": "左：int delta_s dn vs 1（红线 15% 门）；中：几何口径 F_cut/(2 sigma)；右：曲率误差（控制变量，两方案重合）。dx = 3/2/1.5 mm。",
        "png_base64": base64.b64encode(open("/root/tp-agent/work/m2737/benchmarks/csf_delta_s/delta_s_normalization.png", "rb").read()).decode(),
    }],
}

tok = login()
for path in ("/api/v1/tensor/reports", "/tensor/reports"):
    try:
        out = post(path, payload, tok)
        print("published via", path, "->", json.dumps(out)[:300])
        break
    except Exception as exc:  # noqa: BLE001
        print("post try", path, "->", exc, file=sys.stderr)