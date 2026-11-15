import base64, json, os, urllib.request
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")

def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

cred = {"username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"]}
tok = post("/auth/login", cred)
TOKEN = tok.get("access_token") or tok.get("token")
assert TOKEN, list(tok)

root = "/root/tp-agent/work/m21080/"
md = open(root + "docs/xval_fsi2/TRANSIENT_VERDICT_AND_TOLERANCE.md").read()
def fig(p, cap):
    return {"name": p.split("/")[-1], "caption": cap,
            "png_base64": base64.b64encode(open(root + p, "rb").read()).decode()}

rep = {
 "title": "FSI2 瞬态体载荷：从检查点工件作出判定，并剥掉耦合容差的 2.9x 假成本",
 "summary": ("把 XVAL-SPH-FSI2 长期 WITHHELD 的 Cd/Cl 判定从「作业还在跑」推进到明确 FAIL"
  "（Cd 均值 -0.438 vs 3.2、Cd 幅值 17.48 vs 0.96、Cl 幅值 19.82 vs 0.22），"
  "成因诊断为未进入周期解（1500/4000 步、99% 的步未收敛、信号无 VIV 周期）；"
  "受控实验证明驱动默认 tol=1e-3 低于界面残差地板（实测 5.6e-2），"
  "纯耗 2.87x 子迭代而载荷系数仅差 0.4%——容差是成本缺陷而非精度缺陷。"
  "交付可在检查点工件上出判定的工具（窗口覆盖不足 1 个周期即判 WITHHELD）。"),
 "scenario_key": "xval_fsi2_transient_verdict", "mission_id": 21080,
 "tags": ["TensorSPH","FSI2","Turek-Hron","force-from-fluid","strong-coupling","verification"],
 "markdown": md,
 "figures": [
  fig("docs/xval_fsi2/assets/fig_transient_run_cj3135.png",
      "cj3135 原始序列：Cd（黑虚线=3.2，灰带 ±0.96）、Cl（灰带 ±0.22）、梁尖位移（虚线=基准幅值）；启动瞬态主导，无周期解"),
  fig("docs/xval_fsi2/assets/fig_tol_sensitivity.png",
      "容差受控对比（同初值同 dt，唯一变量 tol）：每步子迭代数、退出残差（红点线=请求 1e-3，黑虚线=实测地板 1e-2）、两臂整体系力时程"),
 ]}
print("POSTED:", json.dumps(post("/tensor/reports", rep, TOKEN))[:400])
