"""m20099 交差核验的可复核不变量（不依赖 DB、不重跑 SPH，<1 s）。

断言的对象是 m19980 交付的**原始工件**（档案 json / 报告 / 队列日志）与本单独立复跑的产物，
用来防止"审计结论只是文字"。任何一条失败即说明本单的民族/判级证据链断了。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "audit/m19980_repro/benchmarks/koshizuka_dam_break"
REPRO = ROOT / "audit/m19980_repro/out"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def test_reference_matches_the_registered_tolerance_source():
    ref = _load(CASE / "reference.json")
    assert ref["reference_pressure_pa"] == 2700.0
    assert ref["tolerance_pct"] == 12
    assert "10.13182/NSE96-A24205" in ref["source"]["doi"]
    assert ref["source"]["figure"].startswith("Figure 15")


def test_archived_run_is_outside_tolerance_and_audit_rerun_is_bit_identical():
    arch = _load(CASE / "result_dx0014.json")
    rr = _load(REPRO / "rerun_dx0014.json")
    ref = arch["reference_pressure_pa"]
    assert abs(rr["peak_pressure_pa"] - arch["peak_pressure_pa"]) < 1e-6 * arch["peak_pressure_pa"]
    err = abs(rr["peak_pressure_pa"] - ref) / ref * 100.0
    assert err > 1000.0                      # 报告的 1161% 不是笔误
    assert arch["passes_tolerance"] is False


def test_archived_history_cannot_resolve_the_reported_peak():
    """峰值是逐步跟踪的单步尖峰：抽样时程的最大值系统性低于所报峰值。"""
    arch = _load(CASE / "result_dx0014.json")
    hist_max = max(p for _, p in arch["pressure_history"])
    assert hist_max < arch["peak_pressure_pa"]
    assert (arch["peak_pressure_pa"] - hist_max) / arch["peak_pressure_pa"] > 0.10


def test_dx0020_row_in_report_is_real_but_unarchived():
    rr = _load(REPRO / "rerun_dx0020.json")
    assert abs(rr["peak_pressure_pa"] - 24028.0) < 5.0     # 报告⑤ dx=0.02 行复现
    assert not (CASE / "result_dx0020.json").exists()      # 但原工作树没有该工件


def test_cluster_job_cj2420_claimed_success_contradicts_its_log():
    log = Path("/root/tp-agent/cluster/jobs/cj2420.log")
    if not log.exists():                                   # 日志不在共享盘时跳过（不掩盖结论）
        return
    txt = log.read_text()
    assert '"rc": 1' in txt
    assert "Permission denied" in txt
    assert not (CASE / "results.json").exists()            # 收敛性工件从未产生


def test_verify_output_is_derived_from_those_artifacts():
    v = _load(ROOT / "docs/board_audit/m20099/verify_output.json")
    assert v["benchmark"]["within_tolerance"] is False
    assert v["benchmark"]["registry_rows_tensor_benchmark_cases"] == 0
    assert v["artifact_self_consistency"]["history_undercounts_peak_pct"] > 10.0
    assert v["resolution_study"]["convergence_evidence"].startswith("absent")