from __future__ import annotations

import hashlib
import json

import pytest

from vlm_bench import runner
from vlm_bench.status import run_status


def _write_run(path):
    path.mkdir()
    manifest = {
        "schema_version": 2,
        "status": "paused",
        "created_at": "2026-10-05T00:00:00Z",
        "updated_at": "2026-10-05T00:01:00Z",
        "models": ["openai:example"],
        "samples": [
            {"id": "s1", "reference": "a", "hashes": {"crop": hashlib.sha256(b"a").hexdigest()}},
            {"id": "s2", "reference": "b", "hashes": {"crop": hashlib.sha256(b"b").hexdigest()}},
        ],
        "execution_controls": {
            "max_retries": 2,
            "concurrency": 2,
            "max_requests": 10,
            "max_spend_usd": 1.25,
            "cache_dir": None,
        },
    }
    manifest["integrity"] = runner._digest(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"integrity", "status", "updated_at"}
        }
    )
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {"model": "openai:example", "sample_id": "s1", "status": "success", "cache_hit": True},
        {"model": "openai:example", "sample_id": "s2", "status": "error", "attempt_count": 3},
    ]
    (path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    attempts = [
        {"sample_id": "s2", "attempt": 1, "kind": "request_started"},
        {"sample_id": "s2", "attempt": 1, "kind": "request_finished"},
        {"sample_id": "s2", "attempt": 2, "kind": "request_started"},
        {"sample_id": "s2", "attempt": 2, "kind": "request_finished"},
        {"sample_id": "s2", "attempt": 2, "kind": "retry_failed_archive"},
    ]
    (path / "attempts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8"
    )
    execution = {
        "schema_version": 1,
        "invocations": [
            {
                "started_at": "2026-10-05T00:00:00Z",
                "finished_at": "2026-10-05T00:01:00Z",
                "status": "paused",
                "generation_requests": 2,
                "estimated_spend_usd": 0.42,
                "unknown_spend_count": 1,
                "cache_hits": 2,
                "limit_reason": "max_requests",
                "execution_controls": manifest["execution_controls"],
            }
        ],
        "generation_requests_current": 2,
        "generation_requests_cumulative": 5,
        "estimated_spend_usd_current": 0.42,
        "estimated_spend_usd_cumulative": 0.99,
        "unknown_spend_count_current": 1,
        "unknown_spend_count_cumulative": 3,
        "cache_hits_current": 2,
        "cache_hits_cumulative": 7,
        "limit_reason": "max_requests",
        "execution_controls": {
            **manifest["execution_controls"],
            "max_requests": 3,
            "max_spend_usd": 0.8,
        },
    }
    (path / "execution.json").write_text(json.dumps(execution), encoding="utf-8")


def test_status_is_read_only_and_reports_coverage_attempts_limits_and_cache(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    result = run_status(run_dir)
    after = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}

    assert result["run_status"] == "paused"
    assert result["sample_count"] == 2
    assert result["expected_result_count"] == 2
    assert result["recorded_result_count"] == 2
    assert result["missing_result_count"] == 0
    assert result["failed_count"] == 1
    assert result["cache_hit_count_current"] == 2
    assert result["cache_hit_count_cumulative"] == 7
    assert result["generation_attempt_count_current"] == 2
    assert result["generation_attempt_count_cumulative"] == 5
    assert result["request_limit"] == 3
    assert result["spend_limit_usd"] == 0.8
    assert result["original_request_limit"] == 10
    assert result["original_spend_limit_usd"] == 1.25
    assert result["limit_reason"] == "max_requests"
    assert result["estimated_spend_usd_current"] == pytest.approx(0.42)
    assert result["estimated_spend_usd_cumulative"] == pytest.approx(0.99)
    assert before == after


def test_status_rejects_tampered_manifest(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"] = ["openai:tampered"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check failed"):
        run_status(run_dir)
