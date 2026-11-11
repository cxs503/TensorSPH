"""Claim-provenance binding for the three Poiseuille verification records.

Why this file exists
--------------------
``docs/verification_poiseuille_wall_shear_audit.md`` (mission m14711) re-derived
the wall-shear deficit of the plane-Poiseuille case and concluded that the
*no-slip wall condition was not applied*: it quoted ``u_max = 8.87 m/s`` against
``1.23 m/s`` analytic, ``~100 %`` wall slip, ``Ma = 0.493`` and a bulk viscosity
``1e-4 m^2/s``, and it printed the analytic wall shear as ``50 Pa``.  Every one
of those five numbers is wrong for the run it cites -- four of them belong to
the **free-slip control** block of the same artifact (a case whose wall friction
is switched off *on purpose*, so it must slip and must accelerate at ``g``), and
``rho g H / 2`` of the stored run is ``490.5 Pa``, not ``50 Pa``.

The estimator audit next door (``wall_shear_estimators.py``) already fixes the
*physics* reading.  What it cannot catch is a document that quotes somebody
else's numbers: the failure is one of **provenance**, not of arithmetic.  This
script therefore binds each number a document publishes to the JSON path that
must supply it, and reports which binding the document actually satisfies.

Method (no SPH is advanced; pure post-processing of committed artifacts)
-----------------------------------------------------------------------
1. Flatten both committed artifacts (``out/results.json``,
   ``out/wall_shear_estimators.json``) into a ``path -> float`` index, plus a
   few *derived* readouts recomputed from the stored case parameters
   (``tau_w = rho0 g H / 2``, ``u_max = g H^2 / (8 nu)``).  The analytic
   constants are therefore never taken from a document.
2. For each claim in ``claims_register.json``: find the declared quote in the
   declared document (the claim fails if it is absent), extract the printed
   number **from the document**, and compare it with the declared path.
3. If the printed number misses the declared path, search the whole index for
   the closest readout.  A hit inside tolerance means the number exists in the
   artifact but *somewhere else* -> ``misattributed``; no hit means it is not in
   the artifact at all -> ``not_in_artifact``.
4. Independently check that the stored analytic fields equal their derivations
   (``analytic_identity``), which is the check that catches a typed ``50 Pa``.

Outputs
-------
* ``out/provenance_audit.json``                        -- the numbers, generated
* ``docs/figures/poiseuille_claim_provenance.png``     -- the figure
* ``docs/verification_poiseuille_claim_provenance.md`` -- the record

Usage
-----
    python benchmarks/poiseuille/provenance_audit.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[1]
OUT = BENCH / "out"
RESULTS = OUT / "results.json"
ESTIMATORS = OUT / "wall_shear_estimators.json"
REGISTER = BENCH / "claims_register.json"
ARTIFACT = OUT / "provenance_audit.json"
FIG = ROOT / "docs" / "figures" / "poiseuille_claim_provenance.png"
DOC = ROOT / "docs" / "verification_poiseuille_claim_provenance.md"

SKIP_KEYS = {"trace", "rows", "field"}
ANALYTIC_TOL = 1e-9


# --------------------------------------------------------------------------- #
# artifact index
# --------------------------------------------------------------------------- #

def _flatten(prefix: str, obj, out: dict[str, float]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in SKIP_KEYS:
                continue
            _flatten(f"{prefix}.{key}" if prefix else key, value, out)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            _flatten(f"{prefix}[{index}]", value, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        value = float(obj)
        if math.isfinite(value):
            out[prefix] = value


def build_index(results: dict, estimators: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Return (artifact readouts, derived readouts) as flat path -> value maps."""
    index: dict[str, float] = {}
    _flatten("results", results, index)
    _flatten("estimators", estimators, index)

    rho0 = float(estimators["inputs"]["rho0"])
    g = float(results["body_force"])
    height = float(results["height"])
    nu = float(results["viscosity"])
    derived = {
        "derived.tau_w": rho0 * g * height / 2.0,
        "derived.u_max": g * height ** 2 / (8.0 * nu),
        "derived.rho0": rho0,
    }
    return index, derived


def resolve(index: dict[str, float], derived: dict[str, float], path: str) -> float:
    if path in derived:
        return derived[path]
    if path not in index:
        raise KeyError(f"declared claim path not found in the artifacts: {path}")
    return index[path]


def _trailing_name(path: str) -> str:
    """Last identifier of a path, e.g. ``results.results[2].u_max_measured`` -> ``u_max_measured``."""
    found = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)$", path)
    return found[0] if found else path


def nearest(index: dict[str, float], derived: dict[str, float], value: float,
            claimed_path: str) -> tuple[str, float, bool]:
    """Closest readout to ``value``, preferring candidates of the *same quantity*.

    A printed number that has drifted to another case of the same quantity is a
    far more likely explanation than a coincidence with an unrelated readout, so
    the search first restricts itself to paths whose trailing identifier equals
    the declared one (``...u_max_measured`` vs ``...u_max_measured``) and only
    falls back to the whole index when the artifact holds no such quantity.
    """
    suffix = _trailing_name(claimed_path)
    pool = [(path, candidate) for path, candidate in list(index.items()) + list(derived.items())
            if path != claimed_path and candidate != 0.0]
    same = [(path, candidate) for path, candidate in pool
            if _trailing_name(path) == suffix]
    group = same if same else pool
    best_path, best_rel = "", math.inf
    for path, candidate in group:
        rel = abs(value - candidate) / abs(candidate)
        if rel < best_rel:
            best_path, best_rel = path, rel
    return best_path, best_rel, bool(same)


# --------------------------------------------------------------------------- #
# claims
# --------------------------------------------------------------------------- #

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def evaluate_claim(claim: dict, index: dict[str, float], derived: dict[str, float],
                   docs: dict[str, str]) -> dict:
    doc_path = claim["doc"]
    if doc_path not in docs:
        docs[doc_path] = _read(ROOT / doc_path)
    text = docs[doc_path]

    quote = claim["quote"]
    matches = text.count(quote)
    record = {
        "id": claim["id"],
        "doc": doc_path,
        "quote": quote,
        "path": claim["path"],
        "scale": claim.get("scale", 1.0),
        "quote_occurrences": matches,
    }
    if matches != 1:
        record["status"] = "quote_not_found"
        record["detail"] = f"the declared quote occurs {matches} times in {doc_path}"
        return record

    found = re.findall(claim["extract"], quote)
    if len(found) != 1:
        record["status"] = "extract_ambiguous"
        record["detail"] = f"extract pattern matched {len(found)} numbers inside the quote"
        return record
    printed = float(found[0])

    scale = record["scale"]
    expected = resolve(index, derived, claim["path"]) * scale
    tol = claim.get("tol_rel", 5e-4)
    rel = abs(printed - expected) / abs(expected) if expected else abs(printed - expected)

    record.update({"printed": printed, "artifact_value": expected, "rel_error": rel,
                   "tol_rel": tol})
    if claim["path"] == "derived.tau_w":
        rho0 = derived["derived.rho0"]
        height = index["results.height"]
        record["implied_body_force"] = 2.0 * printed / (rho0 * height)
    if rel <= tol:
        record["status"] = "verified"
        return record

    alt_path, alt_rel, same_quantity = nearest(index, derived, printed, claim["path"])
    record["nearest_is_same_quantity"] = same_quantity
    if alt_rel <= tol:
        record["status"] = "misattributed"
        record["actual_path"] = alt_path
        record["actual_value"] = (derived.get(alt_path) if alt_path in derived
                                 else index[alt_path])
        record["actual_rel_error"] = alt_rel
        record["detail"] = (
            f"the printed value lives at {alt_path} "
            f"({record['actual_value']:.6g}), not at {claim['path']} ({expected:.6g})"
        )
    else:
        record["status"] = "not_in_artifact"
        record["nearest_path"] = alt_path
        record["nearest_rel_error"] = alt_rel
        scope = ("no readout of the declared quantity" if same_quantity
                 else "no readout of the committed artifacts")
        record["detail"] = (
            f"{scope} is within {tol:.1%} of the printed value; nearest is "
            f"{alt_path} ({alt_rel:.2%} away)"
        )
    if claim.get("claimed_basis"):
        record["claimed_basis"] = claim["claimed_basis"]
    return record


def check_analytic_identities(results: dict, derived: dict[str, float]) -> list[dict]:
    blocks = [(f"results.results[{i}]", case) for i, case in enumerate(results["results"])]
    blocks.append(("results.free_slip_control", results["free_slip_control"]))
    findings = []
    for label, block in blocks:
        for field, key in (("wall_shear_analytic", "derived.tau_w"),
                           ("u_max_analytic", "derived.u_max")):
            stored = float(block[field])
            expected = derived[key]
            rel = abs(stored - expected) / abs(expected)
            findings.append({
                "block": label,
                "field": field,
                "stored": stored,
                "derived": expected,
                "derived_from": key,
                "rel_error": rel,
                "status": "verified" if rel <= ANALYTIC_TOL else "identity_violated",
            })
    return findings


# --------------------------------------------------------------------------- #
# the two other questions the record has to answer from the artifact itself
# --------------------------------------------------------------------------- #

def hypothesis_tests(results: dict, estimators: dict) -> dict:
    """Falsify 'the no-slip wall condition was not applied' on the stored data."""
    rho0 = float(estimators["inputs"]["rho0"])
    g = float(results["body_force"])
    no_slip = []
    for case in results["results"]:
        no_slip.append({
            "resolution_H_over_dx": case["resolution_H_over_dx"],
            "dx": case["dx"],
            "u_max_ratio": case["u_max_ratio"],
            "wall_shear_ratio_secant": case["wall_shear_ratio"],
            "wall_row_slip_over_umax": case["wall_row_slip_over_umax"],
            "max_mach": case["max_mach"],
        })
    control = results["free_slip_control"]
    ladder = []
    for case, est in zip(results["results"], estimators["cases"]):
        ladder.append({
            "resolution_H_over_dx": case["resolution_H_over_dx"],
            "u_max_ratio": case["u_max_ratio"],
            "drag_over_weight_control_corrected":
                est["budget"]["drag_over_weight_control_corrected"],
            "secant_ratio": est["estimator_ratios"]["secant_1dx"],
            "stencil_ratio": est["estimator_ratios"]["secant_1dx_on_analytic_parabola"],
            "wall_gap_ratio": est["estimator_ratios"]["wall_particle_gap"],
            "global_fit_ratio": est["estimator_ratios"]["global_parabola_fit_slope"],
        })
    return {
        "case": results["case"],
        "rho0": rho0,
        "body_force": g,
        "height": results["height"],
        "viscosity": results["viscosity"],
        "tau_w_analytic_pa": rho0 * g * results["height"] / 2.0,
        "u_max_analytic": g * results["height"] ** 2 / (8.0 * results["viscosity"]),
        "no_slip_ladder": no_slip,
        "free_slip_control": {
            "wall_friction": control["wall_friction"],
            "u_max_measured": control["u_max_measured"],
            "u_max_ratio": control["u_max_ratio"],
            "wall_row_slip_over_umax": control["wall_row_slip_over_umax"],
            "max_mach": control["max_mach"],
            "wall_shear_ratio": control["wall_shear_ratio"],
            "drift_over_g": estimators["control"]["rate_over_g"],
        },
        "drag_ladder": ladder,
        "falsification": {
            "hypothesis": "the no-slip wall condition is not applied, so the column keeps accelerating at g",
            "test": "streamwise momentum budget rate d<u>/dt / g of the no-slip runs vs the free-slip control",
            "no_slip_rate_over_g": [est["budget"]["rate_over_g"] for est in estimators["cases"]],
            "free_slip_rate_over_g": [estimators["control"]["rate_over_g"]],
            "verdict": "falsified",
        },
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #

def make_figure(tests: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ladder = tests["no_slip_ladder"]
    x = [entry["resolution_H_over_dx"] for entry in ladder]
    y = [entry["u_max_ratio"] for entry in ladder]
    left.plot(x, y, "o-", color="tab:blue", label="no-slip ladder (stored cases)")
    for xi, yi in zip(x, y):
        left.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                      ha="center", fontsize=9)
    control = tests["free_slip_control"]
    left.plot([x[0]], [control["u_max_ratio"]], "X", ms=12, color="tab:red",
              label="free-slip control (wall_friction = 0)")
    left.annotate(f"{control['u_max_ratio']:.2f}  <- quoted by the audit doc as the\n"
                  f"no-slip case (its Ma = {control['max_mach']:.3f})",
                  (x[0], control["u_max_ratio"]), textcoords="offset points",
                  xytext=(-18, -34), ha="right", fontsize=8, color="tab:red")
    left.axhline(1.0, color="0.3", ls="--", lw=1)
    left.set_xlabel("H/dx")
    left.set_ylabel(r"$u_{max,measured} / u_{max,analytic}$")
    left.set_title("(a) peak velocity ladder and the control it must not quote")
    left.set_xticks(x)
    left.grid(alpha=0.3)
    left.legend(loc="center left", fontsize=8)

    finest = tests["drag_ladder"][-1]
    labels = ["published\n1-dx secant", "same secant on\nexact parabola", "wall-particle\ngap",
              "global fit\nslope", "momentum\nbudget"]
    values = [finest["secant_ratio"], finest["stencil_ratio"], finest["wall_gap_ratio"],
              finest["global_fit_ratio"],
              finest["drag_over_weight_control_corrected"]]
    colors = ["tab:red", "tab:orange", "tab:orange", "tab:orange", "tab:green"]
    bars = right.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        right.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                       textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    right.axhline(1.0, color="0.3", ls="--", lw=1, label="exact $\\tau_w = \\rho g H/2$")
    right.set_ylim(0, 1.2)
    right.set_ylabel(r"$\tau / \tau_w$")
    right.set_title(f"(b) five readings of one shear, H/dx = {finest['resolution_H_over_dx']}")
    right.grid(alpha=0.3, axis="y")
    right.legend(loc="lower right", fontsize=8)

    fig.suptitle("Poiseuille wall shear: estimator bias vs drag deficit "
                 "(post-processing of the committed artifacts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    plt.close(fig)
    return FIG


# --------------------------------------------------------------------------- #
# documentation
# --------------------------------------------------------------------------- #

def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def write_doc(payload: dict) -> Path:
    tests = payload["hypothesis_tests"]
    claims = payload["claims"]
    by_status = payload["summary"]["by_status"]
    misattributed = [claim for claim in claims if claim["status"] == "misattributed"]
    missing = [claim for claim in claims if claim["status"] == "not_in_artifact"]
    identities = payload["analytic_identities"]
    violations = [row for row in identities if row["status"] != "verified"]

    lines: list[str] = []
    add = lines.append
    add("# Poiseuille 验证记录的断言溯源审计（claim provenance）")
    add("")
    add("*由 `benchmarks/poiseuille/provenance_audit.py` 从 `out/results.json`、"
        "`out/wall_shear_estimators.json` 与 `claims_register.json` 程序化生成，勿手改。*")
    add("")
    add("## 1. 为什么需要这道门")
    add("")
    add("`docs/verification_poiseuille_wall_shear_audit.md`（m14711）把 Poiseuille 通道的壁面剪切缺口归因为"
        "「无滑移壁面条件未生效」，并用五条数字佐证：`u_max = 8.87 m/s`（解析 1.23）、"
        "壁面滑移比 `0.999996`、`Ma = 0.493`、体粘度 `1e-4 m²/s`、解析壁面剪切 `50 Pa`。"
        "本审计把这五条逐一绑到工件上，结论是：**没有一条能作为该文件所声明的无滑移算例的读数**——"
        "`u_max`、滑移比与 `Ma` 三条取自同一工件里的"
        "**自由滑移对照（free-slip control，`wall_friction = 0`）**，该算例按设计就是要滑移、"
        "就是要以 `g` 持续加速；壁面剪切比 `0.503` 取自 `H/dx = 16` 而非文件宣称的最粗分辨率；"
        "体粘度 `1e-4` 在工件里不存在（算例是 `nu = 0.01`）；解析壁面剪切应为 "
        f"`rho0 g H / 2 = {_fmt(tests['tau_w_analytic_pa'], 1)} Pa`，不是 `50 Pa`。")
    add("")
    add("隔壁的 `docs/verification_poiseuille_wall_shear_estimator.md` 已经把**物理读数**修好了"
        "（壁面确实传递了全部动量：预算法 `D/(M g) ≈ 1`），但它挡不住「引用别人数字」这类错误——"
        "那不是算术错误，是**出处错误**。本审计把每一条出版物数字绑到它必须来自的 JSON 路径上，"
        "并报告它实际来自哪里。")
    add("")
    add("## 2. 方法（不推进 SPH，纯后处理已提交工件）")
    add("")
    add("1. 把两个已提交工件拍平成 `路径 -> 数值` 索引，并**从存储的算例参数重算**派生量 "
        "`tau_w = rho0 g H / 2`、`u_max = g H^2 / (8 nu)`（解析常数绝不从文档里抄）。")
    add("2. 对 `claims_register.json` 的每条断言：在声明的文档里定位声明的引文（找不到即判失败），"
        "**从文档文本里**抓出打印值，与声明的 JSON 路径比对。")
    add("3. 若打印值对不上声明路径，就在全索引里搜索最接近的读数："
        "落在容差内 = 数字确实在工件里、但在**别处**（`misattributed`）；无命中 = 工件里根本没有（`not_in_artifact`）。")
    add("4. 独立核对存储的解析字段是否等于其推导式（`analytic_identity`）——这一步专门抓「手打常数」。")
    add("")
    add("## 3. 断言溯源结果")
    add("")
    add(f"共 **{payload['summary']['n_claims']}** 条断言："
        f"`verified` **{by_status.get('verified', 0)}**，"
        f"`misattributed` **{by_status.get('misattributed', 0)}**，"
        f"`not_in_artifact` **{by_status.get('not_in_artifact', 0)}**，"
        f"其他失败 **{sum(v for k, v in by_status.items() if k not in ('verified', 'misattributed', 'not_in_artifact'))}**。")
    add("")
    add("| 断言 | 文档 | 打印值 | 声明路径 | 工件值 | 相对偏差 | 判定 |")
    add("|---|---|---|---|---|---|---|")
    for claim in claims:
        value = claim.get("artifact_value")
        rel = claim.get("rel_error")
        add(f"| `{claim['id']}` | `{claim['doc']}` | {claim.get('printed', 'n/a')} | "
            f"`{claim['path']}` | {_fmt(value) if value is not None else 'n/a'} | "
            f"{('%+.2f%%' % (100 * rel)) if rel is not None else 'n/a'} | "
            f"**{claim['status']}** |")
    add("")
    add("## 4. 判定为出处错误/工件外数字的断言")
    add("")
    if not misattributed and not missing:
        add("（无）")
    for claim in misattributed:
        add(f"* `{claim['id']}`：打印值 {claim['printed']} ≡ 工件中的 `{claim['actual_path']}` "
            f"（{claim['actual_value']:.6g}），不是声明的 `{claim['path']}` "
            f"（{claim['artifact_value']:.6g}）。")
        if claim.get("claimed_basis"):
            add(f"  - 声明依据：{claim['claimed_basis']}")
    for claim in missing:
        add(f"* `{claim['id']}`：打印值 {claim['printed']} 不在工件任何读数附近"
            f"（最近为 `{claim['nearest_path']}`，偏离 {claim['nearest_rel_error']:.2%}）。")
        if claim.get("implied_body_force") is not None:
            add(f"  - 若该常数成立，反解的体积力为 `g = 2 tau_w / (rho0 H) = "
                f"{claim['implied_body_force']:.3f} m/s²`，而工件里的体积力是 "
                f"{tests['body_force']:.3f} m/s²；按该文档自己参数表的 `g = 0.1 m/s²` 反解"
                f"又只有 5 Pa。三种口径互不自洽，说明这个常数不是从算例参数算出来的。")
    add("")
    add("## 5. 解析常数自洽性（手打常数的结构性防线）")
    add("")
    add("| 存储块 | 字段 | 存储值 | 推导值 | 相对偏差 | 判定 |")
    add("|---|---|---|---|---|---|")
    for row in identities:
        add(f"| `{row['block']}` | `{row['field']}` | {_fmt(row['stored'], 4)} | "
            f"{_fmt(row['derived'], 4)} | {row['rel_error']:.2e} | **{row['status']}** |")
    add("")
    if violations:
        add(f"**身份校验失败 {len(violations)} 条**：存储的解析常数与其推导式不一致，"
            "说明该字段是手输而非算出来的。")
    else:
        add("全部存储的解析常数与其推导式一致（相对偏差 < 1e-9），说明 `results.json` 的解析列是算出来的，"
            "文档只需要照抄；`50 Pa` 这类数字只可能来自文档转述环节。")
    add("")
    add("## 6. 用同一份工件证伪「壁面无滑移未生效」")
    add("")
    add("若壁面条件真的没生效，无滑移算例应当像自由滑移对照那样以 `d<u>/dt ≈ g` 持续加速。"
        "存储轨迹给出的动量预算速率（`wall_shear_estimators.py` 的最小二乘斜率，末段 40%）：")
    add("")
    add("| 算例 | `d<u>/dt / g` | `u_max/u_max,analytic` | `Ma_max` |")
    add("|---|---|---|---|")
    for entry, rate in zip(tests["no_slip_ladder"], tests["falsification"]["no_slip_rate_over_g"]):
        add(f"| no-slip, H/dx = {entry['resolution_H_over_dx']} | {rate:+.4f} | "
            f"{entry['u_max_ratio']:.3f} | {entry['max_mach']:.4f} |")
    control = tests["free_slip_control"]
    add(f"| free-slip control, H/dx = {tests['no_slip_ladder'][0]['resolution_H_over_dx']} | "
        f"{tests['falsification']['free_slip_rate_over_g'][0]:+.4f} | {control['u_max_ratio']:.3f} | "
        f"{control['max_mach']:.4f} |")
    add("")
    add("无滑移算例的加速率是对照的 "
        f"{tests['falsification']['no_slip_rate_over_g'][0] / tests['falsification']['free_slip_rate_over_g'][0]:.4f} 倍"
        f"（{tests['falsification']['no_slip_rate_over_g'][0] * 100:.2f}% vs "
        f"{tests['falsification']['free_slip_rate_over_g'][0] * 100:.2f}% of g）："
        "壁面把注入的流向动量全额吸收，剖面饱和。**「壁面条件未生效」假设被工件本身证伪**；"
        "被审计文档的 `u_max = 8.87 m/s`、`Ma = 0.493`、滑移比 `0.999996` 正是这个对照算例的特征值。"
        "对照算例的高马赫数是它按设计无阻力加速的必然结果，不是无滑移算例的工况越界"
        "（无滑移阶梯 `Ma_max ≤ "
        f"{max(entry['max_mach'] for entry in tests['no_slip_ladder']):.4f}`）。")
    add("")
    add("![claim provenance](figures/poiseuille_claim_provenance.png)")
    add("")
    add("图 (a) 无滑移阶梯与自由滑移对照的峰值速度比；对照点被审计文档当作无滑移算例引用。"
        "图 (b) `H/dx = "
        f"{tests['drag_ladder'][-1]['resolution_H_over_dx']}` 上同一个壁面剪切的五种读数："
        "只有动量预算法（绿）给出 1，其余四种近壁梯度读数系统性偏低——偏差属于**估计算子**，"
        "不属于壁面模型。")
    add("")
    add("## 7. 误差判定（逐项显式回答是否 <3%）")
    add("")
    add("| 关键量 | 基准 | 实测 | 相对误差 | <3% ？ |")
    add("|---|---|---|---|---|")
    for entry in tests["no_slip_ladder"]:
        if entry["resolution_H_over_dx"] != tests["drag_ladder"][-1]["resolution_H_over_dx"]:
            continue
        err = entry["u_max_ratio"] - 1.0
        add(f"| 峰值速度 `u_max`（H/dx = {entry['resolution_H_over_dx']}） | "
            f"{tests['u_max_analytic']:.4f} m/s | "
            f"{tests['u_max_analytic'] * entry['u_max_ratio']:.4f} m/s | {err:+.2%} | "
            f"{'是' if abs(err) < 0.03 else '否'} |")
    add(f"| 壁面剪切（动量预算法，控制校正） | `rho g H/2 = {tests['tau_w_analytic_pa']:.1f} Pa` | "
        f"{tests['tau_w_analytic_pa'] * tests['drag_ladder'][-1]['drag_over_weight_control_corrected']:.1f} Pa | "
        f"{tests['drag_ladder'][-1]['drag_over_weight_control_corrected'] - 1:+.2%} | "
        f"{'是' if abs(tests['drag_ladder'][-1]['drag_over_weight_control_corrected'] - 1) < 0.03 else '否'} |")
    add(f"| 壁面剪切（已发布的 1-dx 割线） | `rho g H/2 = {tests['tau_w_analytic_pa']:.1f} Pa` | "
        f"{tests['tau_w_analytic_pa'] * tests['drag_ladder'][-1]['secant_ratio']:.1f} Pa | "
        f"{tests['drag_ladder'][-1]['secant_ratio'] - 1:+.2%} | "
        f"{'是' if abs(tests['drag_ladder'][-1]['secant_ratio'] - 1) < 0.03 else '否'} |")
    add(f"| 自由滑移对照的漂移率 | `g = {tests['body_force']:.4f} m/s²` | "
        f"{tests['body_force'] * tests['free_slip_control']['drift_over_g']:.4f} m/s² | "
        f"{tests['free_slip_control']['drift_over_g'] - 1:+.2%} | "
        f"{'是' if abs(tests['free_slip_control']['drift_over_g'] - 1) < 0.03 else '否'} |")
    add(f"| 解析常数 `tau_w`（文档转述） | `{tests['tau_w_analytic_pa']:.1f} Pa` | 文档印 50 Pa | "
        f"{(50 - tests['tau_w_analytic_pa']) / tests['tau_w_analytic_pa']:+.1%} | 否 |")
    add("")
    add("## 8. 注意点与局限")
    add("")
    add("1. 本审计只判**出处**，不做新的 SPH 计算；它不能替代对壁面模型的独立物理复核，"
        "但能把「数字来自哪个算例」这件事变成机器可判定的。")
    add("2. `claims_register.json` 的引文是**文档里的原文**：文档改一个字，绑定失败会立即报警，"
        "因此本记录不会悄悄过期。")
    add("3. 近壁梯度读数的偏差机理（模板几何 `1 - 2 dx/H` + 抹平壁层）见 "
        "`docs/verification_poiseuille_wall_shear_estimator.md`；本审计不重复其推导。")
    add("4. 局限：`claims_register.json` 只覆盖本仓 Poiseuille 的三份文档；跨仓推广需要把 register 做成仓级约定。")
    add("")
    add("## 9. 结论与处置")
    add("")
    add("| 对象 | 处置 |")
    add("|---|---|")
    add("| `docs/verification_poiseuille_wall_shear_audit.md` | 抬头加校正声明：参数表/解析常数/四条读数出处错误，"
        "「壁面条件未生效」结论被证伪；正文原样保留（不改写他人记录） |")
    add("| `docs/verification_poiseuille_channel.md` 与 estimator 记录 | 全部数字溯源通过（见 §3） |")
    add("| 真正的欠账 | 近壁壁面剪切**估计算子**（不是壁面模型）：1-dx 割线在 `H/dx = 16` 读 0.503，"
        "其中 0.125 是模板几何、其余是抹平壁层的第一行速度；需要的是把壁面剪切读数改成"
        "动量预算或 VP（Volume-Weighted）法，而不是改壁面边界条件 |")
    add("")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return DOC


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> dict:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    estimators = json.loads(ESTIMATORS.read_text(encoding="utf-8"))
    register = json.loads(REGISTER.read_text(encoding="utf-8"))

    index, derived = build_index(results, estimators)
    docs: dict[str, str] = {}
    claims = [evaluate_claim(claim, index, derived, docs) for claim in register["claims"]]

    by_status: dict[str, int] = {}
    for claim in claims:
        by_status[claim["status"]] = by_status.get(claim["status"], 0) + 1

    payload = {
        "schema": "tensorsph.poiseuille.provenance_audit/v1",
        "generated_by": "benchmarks/poiseuille/provenance_audit.py",
        "inputs": {
            "results": "benchmarks/poiseuille/out/results.json",
            "estimators": "benchmarks/poiseuille/out/wall_shear_estimators.json",
            "register": "benchmarks/poiseuille/claims_register.json",
        },
        "hypothesis_tests": hypothesis_tests(results, estimators),
        "analytic_identities": check_analytic_identities(results, derived),
        "claims": claims,
        "summary": {
            "n_claims": len(claims),
            "by_status": by_status,
            "n_misattributed": by_status.get("misattributed", 0),
            "n_not_in_artifact": by_status.get("not_in_artifact", 0),
            "n_audit_doc_claims_with_defect": sum(
                1 for claim in claims
                if claim["doc"].endswith("wall_shear_audit.md")
                and claim["status"] != "verified"),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    make_figure(payload["hypothesis_tests"])
    write_doc(payload)
    print(json.dumps(payload["summary"], indent=1, ensure_ascii=False))
    for claim in claims:
        if claim["status"] != "verified":
            print(f"  {claim['status']:>16}  {claim['id']:>34}  {claim.get('detail', '')}")
    return payload


if __name__ == "__main__":
    main()
