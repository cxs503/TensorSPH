"""Publish the mission report for TensorSPH#110 (m3912) to the platform API."""
import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
HERE = pathlib.Path(__file__).resolve().parent
DOC = HERE.parent / "merge_pr110_m3912.md"
PNG = HERE / "pr110_merge_evidence.png"


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return urllib.request.urlopen(req, json.dumps(payload).encode(), timeout=60)


def main():
    tok = json.load(
        post(
            "/auth/login",
            {
                "username": os.environ["TENSOR_AGENT_USER"],
                "password": os.environ["TENSOR_AGENT_PASS"],
            },
        )
    )
    token = tok.get("access_token") or tok.get("token")
    assert token, tok.keys()

    md = DOC.read_text()
    png = base64.b64encode(PNG.read_bytes()).decode()
    payload = {
        "title": "TensorSPH PR #110 治愈：三方合并冲突解法与验证证据（m3912）",
        "summary": (
            "PR #110（agent/m3265）对 main 的三个冲突全部是同一科学结论被两条分支独立修复的"
            "语义重叠：表面张力幅值与毛细松弛覆盖度。解法=取语义超集一侧（main），"
            "并机器核验 agent/m3265 无文件/无结论丢失；合并树相对 main 代码面逐字节相同，"
            "故基线不可能被拉低。核心子集 159 passed / 5 xfailed / 0 failed，ruff 全绿。"
            "推送受沙箱与出网双重阻塞，改为交接脚本 + 记录旧/新 SHA。"
        ),
        "scenario_key": "solver_merge_heal",
        "mission_id": 3912,
        "tags": ["TensorSPH", "PR合并", "表面张力", "验证证据", "m3912"],
        "markdown": md,
        "figures": [
            {
                "name": "pr110_merge_evidence.png",
                "caption": (
                    "(a) 三条已提交收敛记录的 t/t_sigma 与 0.5 自洽门禁（合并树实测）；"
                    "(b) R/h=6.15 方晶格上三种格式的 |F_cut|/(2σ)：legacy 0.2147、"
                    "二次拟合默认 0.7106、MLS 0.7907，虚线为标称 1.0，浅带为 15% 接受窗。"
                ),
                "png_base64": png,
            }
        ],
    }
    body = json.dumps(payload).encode()
    print("payload bytes:", len(body))
    r = post("/tensor/reports", payload, token)
    print(r.status, r.read()[:300].decode(errors="replace"))


if __name__ == "__main__":
    main()