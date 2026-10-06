"""Read-only summaries of a benchmark run's durable execution state."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import runner


def _finite_nonnegative(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Execution summary {field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Execution summary {field} must be a finite non-negative number")
    return number


def run_status(run_dir: str | Path) -> dict[str, Any]:
    """Read summary counters without repairing or changing run artifacts."""
    path = Path(run_dir).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Run directory does not exist: {path}")
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read run manifest: {path / 'manifest.json'}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must contain a JSON object")
    if manifest.get("schema_version") == 2:
        immutable = {
            key: value
            for key, value in manifest.items()
            if key not in {"integrity", "status", "updated_at"}
        }
        if (
            not isinstance(manifest.get("integrity"), str)
            or runner._digest(immutable) != manifest["integrity"]
        ):
            raise ValueError(f"Run manifest integrity check failed: {path / 'manifest.json'}")

    samples = manifest.get("samples")
    models = manifest.get("models")
    if not isinstance(samples, list) or not isinstance(models, list):
        raise ValueError("Run manifest must include frozen samples and selected models")
    sample_ids = [
        str(sample.get("id"))
        for sample in samples
        if isinstance(sample, dict) and sample.get("id") is not None
    ]
    model_ids = [model for model in models if isinstance(model, str) and model]
    if len(sample_ids) != len(samples) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Run manifest contains missing or duplicate sample IDs")
    if len(model_ids) != len(models) or len(model_ids) != len(set(model_ids)):
        raise ValueError("Run manifest contains missing or duplicate model selectors")

    rows = runner.read_records(path / "results.jsonl")
    expected = {(model, sample_id) for model in model_ids for sample_id in sample_ids}
    pairs: set[tuple[str, str]] = set()
    statuses: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Result record {index} must be a JSON object")
        model = row.get("model")
        sample_id = row.get("sample_id")
        if not isinstance(model, str) or sample_id is None:
            raise ValueError(f"Result record {index} lacks a model/sample ID pair")
        pair = (model, str(sample_id))
        if pair not in expected or pair in pairs:
            raise ValueError(
                f"Result record {index} contains an unknown or duplicate pair {pair!r}"
            )
        pairs.add(pair)
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1

    execution_path = path / "execution.json"
    execution = None
    if execution_path.exists():
        try:
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read execution summary: {execution_path}") from exc
        if not isinstance(execution, dict) or execution.get("schema_version") != 1:
            raise ValueError(f"Unsupported execution summary: {execution_path}")
        if not isinstance(execution.get("invocations"), list):
            raise ValueError(f"Execution summary has no invocation history: {execution_path}")
        for index, invocation in enumerate(execution["invocations"], 1):
            if not isinstance(invocation, dict):
                raise ValueError(f"Execution invocation {index} must be a JSON object")
            for key in ("generation_requests", "unknown_spend_count", "cache_hits"):
                value = invocation.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"Execution invocation {index} {key} must be a non-negative integer"
                    )
            _finite_nonnegative(
                invocation.get("estimated_spend_usd"),
                f"invocations[{index - 1}].estimated_spend_usd",
            )
            invocation_controls = invocation.get("execution_controls")
            if not isinstance(invocation_controls, dict):
                raise ValueError(
                    f"Execution invocation {index} execution_controls must be a JSON object"
                )
            reason = invocation.get("limit_reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError(f"Execution invocation {index} limit_reason must be a string")
    controls = manifest.get("execution_controls")
    if controls is not None and not isinstance(controls, dict):
        raise ValueError("Run manifest execution_controls must be a JSON object")
    controls = dict(controls or {})
    for key in ("max_retries", "concurrency", "max_requests", "max_spend_usd", "cache_dir"):
        if key not in controls and key in manifest:
            controls[key] = manifest[key]
    current_controls = (
        execution.get("execution_controls", controls) if execution is not None else controls
    )
    if not isinstance(current_controls, dict):
        raise ValueError("Execution summary execution_controls must be a JSON object")

    failed_statuses = {
        status: count
        for status, count in statuses.items()
        if status not in {"success", "unsupported", "cached"}
    }

    def runtime_count(key: str) -> int | None:
        if execution is None:
            return None
        value = execution.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Execution summary {key} must be a non-negative integer")
        return value

    limit_reason = execution.get("limit_reason") if execution is not None else None
    return {
        "schema_version": 1,
        "run_directory": str(path),
        "run_id": path.name,
        "run_status": manifest.get("status", "unknown"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "models": model_ids,
        "sample_count": len(sample_ids),
        "expected_result_count": len(expected),
        "recorded_result_count": len(rows),
        "result_coverage": len(pairs) / len(expected) if expected else 1.0,
        "success_count": statuses.get("success", 0) + statuses.get("cached", 0),
        "unsupported_count": statuses.get("unsupported", 0),
        "failed_count": sum(failed_statuses.values()),
        "failed_status_counts": failed_statuses,
        "missing_result_count": len(expected - pairs),
        "cache_hit_count_current": runtime_count("cache_hits_current"),
        "cache_hit_count_cumulative": runtime_count("cache_hits_cumulative"),
        "generation_attempt_count_current": runtime_count("generation_requests_current"),
        "generation_attempt_count_cumulative": runtime_count("generation_requests_cumulative"),
        "request_limit": current_controls.get("max_requests"),
        "spend_limit_usd": current_controls.get("max_spend_usd"),
        "original_request_limit": controls.get("max_requests"),
        "original_spend_limit_usd": controls.get("max_spend_usd"),
        "limit_reason": limit_reason,
        "estimated_spend_usd_current": _finite_nonnegative(
            execution.get("estimated_spend_usd_current"), "estimated_spend_usd_current"
        )
        if execution is not None
        else None,
        "estimated_spend_usd_cumulative": _finite_nonnegative(
            execution.get("estimated_spend_usd_cumulative"), "estimated_spend_usd_cumulative"
        )
        if execution is not None
        else None,
        "unknown_spend_count_current": runtime_count("unknown_spend_count_current"),
        "unknown_spend_count_cumulative": runtime_count("unknown_spend_count_cumulative"),
        "execution_summary_available": execution is not None,
        "execution_invocations": execution["invocations"] if execution is not None else [],
        "execution_controls": controls,
        "current_execution_controls": current_controls,
        "original_execution_controls": controls,
    }
