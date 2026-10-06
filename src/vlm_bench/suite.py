"""Previewable and resumable execution of named experiment conditions."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import runner

_SUITE_FILE = "suite.json"
_REPORT_FILE = "suite-report.json"
_MUTABLE_FIELDS = {"integrity", "status", "updated_at", "runs", "summary", "comparisons"}
_FINISHED_RUN_STATUSES = {"complete", "complete_with_errors"}
_RESUMABLE_RUN_STATUSES = {"paused", "interrupted", "running", "incomplete"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    immutable = {key: value for key, value in manifest.items() if key not in _MUTABLE_FIELDS}
    manifest["integrity"] = runner._digest(immutable)
    manifest["updated_at"] = _now()
    runner._atomic_json(path, manifest)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read suite manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Suite manifest must contain a JSON object: {path}")
    integrity = manifest.get("integrity")
    immutable = {key: value for key, value in manifest.items() if key not in _MUTABLE_FIELDS}
    if not isinstance(integrity, str) or runner._digest(immutable) != integrity:
        raise ValueError(f"Suite manifest integrity check failed: {path}")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported suite manifest version: {manifest.get('schema_version')!r}")
    return manifest


def _config_hash(path: Path) -> str:
    try:
        data = path.expanduser().read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read suite config: {path}") from exc
    return hashlib.sha256(data).hexdigest()


def _condition_kwargs(
    condition: dict[str, Any], costs: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "prepared": Path(condition["prepared"]),
        "settings": condition.get("settings", {}),
        "strict_research": condition.get("strict_research", False),
        "protocol": condition.get("protocol", "document-disjoint"),
        "formula_rendering": condition.get("formula_rendering", False),
        "warmup": condition.get("warmup", True),
        "prose_prompt": condition.get("prose_prompt"),
        "math_prompt": condition.get("math_prompt"),
        "costs": costs or {},
    }


def _preview_kwargs(
    condition: dict[str, Any], costs: dict[str, Any] | None = None
) -> dict[str, Any]:
    values = _condition_kwargs(condition, costs)
    values.pop("warmup")
    return values


def _safe_name(value: str) -> str:
    readable = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    readable = "-".join(part for part in readable.split("-") if part)[:48] or "condition"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{digest}"


def _expected_repetitions(suite: dict[str, Any]) -> list[tuple[dict[str, Any], int]]:
    planned: list[tuple[dict[str, Any], int]] = []
    for condition in suite["conditions"]:
        for repetition in range(1, condition["repeats"] + 1):
            planned.append((condition, repetition))
    return planned


def _run_key(condition: str, repetition: int) -> str:
    return f"{condition}\0{repetition}"


def _load_run_state(run_dir: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read run manifest for suite run: {run_dir}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Run manifest must be a JSON object: {run_dir}")
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Suite runs require a schema-v2 run manifest: {run_dir}")
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"integrity", "status", "updated_at"}
    }
    if (
        not isinstance(manifest.get("integrity"), str)
        or runner._digest(immutable) != manifest["integrity"]
    ):
        raise ValueError(f"Run manifest integrity check failed: {run_dir}")
    samples = manifest.get("samples")
    preprocess = manifest.get("preprocess")
    if not isinstance(samples, list) or not isinstance(preprocess, str):
        raise ValueError(f"Run manifest has no frozen benchmark identity: {run_dir}")
    from .snapshot import fingerprint

    if fingerprint(samples, preprocess) != manifest.get("benchmark_fingerprint"):
        raise ValueError(f"Run benchmark fingerprint does not match its frozen samples: {run_dir}")
    return manifest


def _condition_fingerprint(preview: dict[str, Any]) -> str:
    value = preview.get("benchmark_fingerprint")
    if not isinstance(value, str) or not value:
        raise ValueError("Engine preview did not provide a benchmark fingerprint")
    return value


def _build_report(manifest: dict[str, Any], suite_root: Path) -> dict[str, Any]:
    from .research import research_reports

    conditions: dict[str, Any] = {}
    for condition in manifest["conditions"]:
        name = condition["name"]
        condition_runs = [
            entry
            for entry in manifest["runs"].values()
            if entry.get("condition") == name and entry.get("run_dir")
        ]
        condition_runs.sort(key=lambda entry: entry["repetition"])
        per_run: list[dict[str, Any]] = []
        model_values: dict[tuple[str, str], list[dict[str, Any]]] = {}
        track_counts: dict[str, dict[str, Any]] = {}
        for entry in condition_runs:
            run_dir = Path(entry["run_dir"])
            run_manifest = _load_run_state(run_dir)
            if run_manifest.get("benchmark_fingerprint") != condition["benchmark_fingerprint"]:
                raise ValueError(
                    f"Benchmark fingerprint changed within condition {name!r}: {run_dir}"
                )
            rows = runner.read_records(run_dir / "results.jsonl")
            report = research_reports(rows, run_manifest)
            summary = {
                "repetition": entry["repetition"],
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "status": run_manifest.get("status"),
                "benchmark_fingerprint": run_manifest.get("benchmark_fingerprint"),
                "independent_sample_counts": {},
                "independent_document_counts": {},
                "bootstrap_by_track": {},
            }
            sample_by_id = {
                str(sample.get("id")): sample
                for sample in run_manifest.get("samples", [])
                if isinstance(sample, dict) and sample.get("id") is not None
            }
            for track, track_report in report["tracks"].items():
                eligible_ids = track_report.get("eligible_sample_ids", [])
                known_documents = {
                    sample_id: _document_value(sample_by_id[sample_id])
                    for sample_id in eligible_ids
                    if sample_id in sample_by_id
                }
                documents = {value for value in known_documents.values() if value is not None}
                known_document_ids = {
                    sample_id for sample_id, value in known_documents.items() if value
                }
                unknown_document_count = len(eligible_ids) - len(known_document_ids)
                summary["independent_sample_counts"][track] = len(eligible_ids)
                summary["independent_document_counts"][track] = {
                    "known_document_count": len(documents),
                    "unknown_document_sample_count": unknown_document_count,
                    "independent_document_count": len(documents)
                    if not unknown_document_count
                    else None,
                    "bootstrap_unit_count": len(documents)
                    if not unknown_document_count
                    else len(eligible_ids),
                }
                summary["bootstrap_by_track"][track] = track_report.get("bootstrap")
                track_counts.setdefault(
                    track,
                    {
                        "eligible_sample_count": len(eligible_ids),
                        "known_document_count": len(documents),
                        "unknown_document_sample_count": unknown_document_count,
                        "independent_document_count": len(documents)
                        if not unknown_document_count
                        else None,
                        "bootstrap_unit_count": len(documents)
                        if not unknown_document_count
                        else len(eligible_ids),
                    },
                )
                for result in track_report.get("model_results", []):
                    key = (track, result["model"])
                    model_values.setdefault(key, []).append(
                        {
                            "repetition": entry["repetition"],
                            "run_id": run_dir.name,
                            "status": result.get("status"),
                            "cer": result.get("cer"),
                            "run_status": run_manifest.get("status"),
                            "eligible_sample_count": result.get("eligible_sample_count"),
                            "scored_sample_count": result.get("scored_sample_count"),
                            "sample_coverage": result.get("sample_coverage"),
                            "bootstrap": track_report.get("bootstrap"),
                        }
                    )
            per_run.append(summary)

        variability: dict[str, dict[str, Any]] = {}
        for (track, model), values in sorted(model_values.items()):
            cer_values = [
                value["cer"]
                for value in values
                if value["run_status"] == "complete"
                and value["status"] == "complete"
                and _is_finite(value["cer"])
            ]
            variability.setdefault(track, {})[model] = {
                "repetition_count": len(values),
                "scored_repetition_count": len(cer_values),
                "mean_cer": statistics.fmean(cer_values) if cer_values else None,
                "sample_stdev_cer": statistics.stdev(cer_values) if len(cer_values) > 1 else None,
                "min_cer": min(cer_values) if cer_values else None,
                "max_cer": max(cer_values) if cer_values else None,
                "independent_sample_count": track_counts.get(track, {}).get(
                    "eligible_sample_count", 0
                ),
                "independent_document_count": track_counts.get(track, {}).get(
                    "independent_document_count"
                ),
                "known_document_count": track_counts.get(track, {}).get("known_document_count", 0),
                "unknown_document_sample_count": track_counts.get(track, {}).get(
                    "unknown_document_sample_count", 0
                ),
                "bootstrap_unit_count": track_counts.get(track, {}).get("bootstrap_unit_count", 0),
                "per_repetition": values,
            }
        status_counts: dict[str, int] = {}
        for entry in condition_runs:
            status = str(entry.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        completed_repetitions = sum(
            run.get("status") in _FINISHED_RUN_STATUSES for run in condition_runs
        )
        planned_repetitions = condition["repeats"]
        condition_status = (
            "incomplete"
            if completed_repetitions < planned_repetitions
            else "complete_with_errors"
            if status_counts.get("complete_with_errors")
            else "complete"
        )
        conditions[name] = {
            "benchmark_fingerprint": condition["benchmark_fingerprint"],
            "planned_repetitions": planned_repetitions,
            "completed_repetitions": completed_repetitions,
            "status": condition_status,
            "run_status_counts": status_counts,
            "independent_counts": track_counts,
            "per_run": per_run,
            "variability": variability,
        }
    return {
        "schema_version": 1,
        "suite_id": manifest["suite_id"],
        "config_sha256": manifest["config_sha256"],
        "execution_controls": manifest["execution_controls"],
        "conditions": conditions,
        "interpretation": (
            "Repetition variability is summarized across runs. Sample and document counts "
            "remain counts of the shared benchmark, not multiplied by repetitions; each run "
            "retains its own document-bootstrap uncertainty."
        ),
        "comparisons": manifest.get("comparisons", []),
    }


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _document_value(sample: dict[str, Any]) -> str | None:
    nested = sample.get("metadata")
    metadata = nested if isinstance(nested, dict) else {}
    for key in ("document_id", "source_document", "document"):
        value = sample.get(key)
        if value in (None, ""):
            value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _persist_report(report: dict[str, Any], path: Path) -> None:
    runner._atomic_json(path, report)


def _write_suite_state(path: Path, manifest: dict[str, Any]) -> None:
    _write_manifest(path, manifest)


def _validate_run_records(
    manifest: dict[str, Any], suite: dict[str, Any], suite_root: Path
) -> None:
    records = manifest.get("runs")
    if not isinstance(records, dict):
        raise ValueError("Suite manifest runs must be a JSON object")
    expected = {
        _run_key(condition["name"], repetition): (condition["name"], repetition)
        for condition, repetition in _expected_repetitions(suite)
    }
    if set(records) - set(expected):
        raise ValueError("Suite manifest contains an unknown condition/repetition run")
    paths: set[Path] = set()
    runs_root = (suite_root / "runs").resolve()
    for key, record in records.items():
        if not isinstance(record, dict):
            raise ValueError(f"Suite run record {key!r} must be an object")
        name, repetition = expected[key]
        if record.get("condition") != name or record.get("repetition") != repetition:
            raise ValueError(f"Suite run record identity does not match {key!r}")
        run_value = record.get("run_dir")
        if run_value is None:
            continue
        if not isinstance(run_value, str) or not run_value:
            raise ValueError(f"Suite run record {key!r} has an invalid run directory")
        run_dir = Path(run_value).expanduser().resolve()
        if run_dir == runs_root or runs_root not in run_dir.parents:
            raise ValueError(f"Saved run directory escapes the suite output: {run_dir}")
        if run_dir in paths:
            raise ValueError(
                f"Multiple suite repetitions point to the same run directory: {run_dir}"
            )
        paths.add(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Saved suite run directory is missing: {run_dir}")


def _verify_run_identity(
    run_manifest: dict[str, Any],
    condition: dict[str, Any],
    repetition: int,
    preview: dict[str, Any],
    execution_controls: dict[str, Any],
) -> None:
    name = condition["name"]
    if run_manifest.get("suite_condition") != name or run_manifest.get("repetition") != repetition:
        raise ValueError(
            f"Saved run identity does not match suite condition {name!r}, repetition {repetition}"
        )
    if run_manifest.get("benchmark_fingerprint") != _condition_fingerprint(preview):
        raise ValueError(f"Saved run benchmark fingerprint does not match suite condition {name!r}")
    if run_manifest.get("models") != condition["models"]:
        raise ValueError(f"Saved run models do not match suite condition {name!r}")
    if run_manifest.get("settings", {}) != condition.get("settings", {}):
        raise ValueError(f"Saved run settings do not match suite condition {name!r}")
    if run_manifest.get("prompt_hashes") != preview.get("prompt_hashes"):
        raise ValueError(f"Saved run prompts do not match suite condition {name!r}")
    for field in ("strict_research", "protocol", "formula_rendering", "warmup"):
        if run_manifest.get(field) != condition.get(field):
            raise ValueError(f"Saved run {field} does not match suite condition {name!r}")
    if run_manifest.get("repeated_measurement") is not (condition["repeats"] > 1):
        raise ValueError(f"Saved run repetition mode does not match suite condition {name!r}")
    if run_manifest.get("base_url") != execution_controls["base_url"]:
        raise ValueError(f"Saved run base URL does not match suite execution controls for {name!r}")
    if run_manifest.get("timeout") != execution_controls["timeout"]:
        raise ValueError(f"Saved run timeout does not match suite execution controls for {name!r}")
    if (run_manifest.get("options") or {}).get("num_predict") != execution_controls["num_predict"]:
        raise ValueError(
            f"Saved run output limit does not match suite execution controls for {name!r}"
        )
    run_controls = run_manifest.get("execution_controls")
    if not isinstance(run_controls, dict):
        raise ValueError(f"Saved run has no execution controls for suite condition {name!r}")
    for field in (
        "max_retries",
        "concurrency",
        "max_requests",
        "max_spend_usd",
        "cache_dir",
    ):
        expected = execution_controls[field]
        actual = run_controls.get(field)
        if field == "cache_dir" and actual is not None:
            actual = str(Path(actual).expanduser().resolve())
        if actual != expected:
            raise ValueError(
                f"Saved run {field} does not match suite execution controls for {name!r}"
            )


def _compare_conditions(
    manifest: dict[str, Any],
    suite_root: Path,
    compare_fn: Callable,
    existing: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    entries_by_condition: dict[str, dict[int, dict[str, Any]]] = {}
    for entry in manifest["runs"].values():
        if entry.get("run_dir") and entry.get("status") in _FINISHED_RUN_STATUSES:
            entries_by_condition.setdefault(entry["condition"], {})[entry["repetition"]] = entry
    comparisons: list[dict[str, Any]] = []
    conditions = manifest["conditions"]
    for index, left in enumerate(conditions):
        for right in conditions[index + 1 :]:
            if left["benchmark_fingerprint"] != right["benchmark_fingerprint"]:
                continue
            repetitions = sorted(
                set(entries_by_condition.get(left["name"], {}))
                & set(entries_by_condition.get(right["name"], {}))
            )
            for repetition in repetitions:
                left_entry = entries_by_condition[left["name"]][repetition]
                right_entry = entries_by_condition[right["name"]][repetition]
                label = f"{_safe_name(left['name'])}-vs-{_safe_name(right['name'])}-r{repetition}"
                output = suite_root / "comparisons" / label
                existing_entry = next(
                    (item for item in (existing or []) if item.get("output") == str(output)), None
                )
                if output.exists():
                    report_path = output / "comparison.json"
                    try:
                        stored_report = json.loads(report_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        raise ValueError(
                            f"Could not verify saved suite comparison: {report_path}"
                        ) from exc
                    if stored_report.get("benchmark_fingerprint") != left["benchmark_fingerprint"]:
                        raise ValueError(
                            f"Saved suite comparison fingerprint changed: {report_path}"
                        )
                    comparisons.append(
                        existing_entry
                        or {
                            "left_condition": left["name"],
                            "right_condition": right["name"],
                            "repetition": repetition,
                            "output": str(output),
                            "benchmark_fingerprint": left["benchmark_fingerprint"],
                            "schema_version": stored_report.get("comparison_schema_version"),
                        }
                    )
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                result = compare_fn([left_entry["run_dir"], right_entry["run_dir"]], output)
                comparisons.append(
                    {
                        "left_condition": left["name"],
                        "right_condition": right["name"],
                        "repetition": repetition,
                        "output": str(output),
                        "benchmark_fingerprint": left["benchmark_fingerprint"],
                        "schema_version": result.get("comparison_schema_version"),
                    }
                )
    return comparisons


def _run_suite_impl(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    base_url: str = "http://localhost:11434",
    timeout: float = 300,
    num_predict: int = 4096,
    max_retries: int | None = None,
    concurrency: int | None = None,
    max_requests: int | None = None,
    max_spend_usd: float | None = None,
    cache_dir: str | Path | None = None,
    progress: Callable[[str], None] = print,
    preview_fn: Callable | None = None,
    run_fn: Callable | None = None,
    resume_fn: Callable | None = None,
    compare_fn: Callable | None = None,
    _lock_held: bool = False,
) -> dict[str, Any]:
    """Preview or execute a v2 suite; output creation is exclusive."""
    from .config import load_config
    from .engine import preview as engine_preview
    from .engine import resume as engine_resume
    from .engine import run as engine_run

    config_path = Path(config_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    config_digest = _config_hash(config_path)
    config = load_config(config_path)
    suite = config.get("suite")
    if config.get("version") != 2 or not isinstance(suite, dict) or suite.get("version") != 2:
        raise ValueError("suite requires an experiment config with version = 2")
    conditions = suite.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Version-2 suite config must define at least one condition")
    names = [condition.get("name") for condition in conditions]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Each suite condition requires a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("Suite condition names must be unique")
    experiment = config.get("experiment", {})
    max_retries = max_retries if max_retries is not None else experiment.get("max_retries", 2)
    concurrency = concurrency if concurrency is not None else experiment.get("concurrency", 1)
    max_requests = max_requests if max_requests is not None else experiment.get("max_requests")
    max_spend_usd = max_spend_usd if max_spend_usd is not None else experiment.get("max_spend_usd")
    cache_dir = cache_dir if cache_dir is not None else experiment.get("cache_dir")
    execution_controls = {
        "base_url": base_url,
        "timeout": timeout,
        "num_predict": num_predict,
        "max_retries": max_retries,
        "concurrency": concurrency,
        "max_requests": max_requests,
        "max_spend_usd": max_spend_usd,
        "cache_dir": str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else None,
    }
    preview_fn = preview_fn or engine_preview
    run_fn = run_fn or engine_run
    resume_fn = resume_fn or engine_resume
    if compare_fn is None:
        from .comparison import compare as comparison_fn

        compare_fn = comparison_fn

    existing_manifest: dict[str, Any] | None = None
    manifest_path = output / _SUITE_FILE
    if resume:
        if not output.is_dir() or not manifest_path.is_file():
            raise ValueError("--resume requires an existing suite output with suite.json")
        existing_manifest = _read_manifest(manifest_path)
        if existing_manifest.get("config_sha256") != config_digest:
            raise ValueError("Suite config changed since this output was created")
        if existing_manifest.get("execution_controls") != execution_controls:
            raise ValueError("Suite execution controls changed since this output was created")
        _validate_run_records(existing_manifest, suite, output)
        stored_plan = {
            (condition.get("name"), condition.get("repeats"))
            for condition in existing_manifest.get("conditions", [])
            if isinstance(condition, dict)
        }
        expected_plan = {(condition["name"], condition["repeats"]) for condition in conditions}
        if stored_plan != expected_plan:
            raise ValueError("Suite condition plan changed; cannot resume this output")
    elif output.exists():
        raise FileExistsError(f"Suite output already exists: {output}; use --resume to continue")

    previews: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        kwargs = _preview_kwargs(condition, config.get("costs"))
        kwargs.update(base_url=base_url, timeout=timeout, num_predict=num_predict)
        kwargs.update(
            max_retries=max_retries,
            concurrency=concurrency,
            max_requests=max_requests,
            max_spend_usd=max_spend_usd,
            cache_dir=cache_dir,
        )
        kwargs["suite_condition"] = condition["name"]
        kwargs["repetition"] = 1
        kwargs["repeated_measurement"] = condition["repeats"] > 1
        result = preview_fn(None, list(condition["models"]), **kwargs)
        if not isinstance(result, dict):
            raise ValueError(f"Engine preview returned an invalid result for {condition['name']!r}")
        previews[condition["name"]] = result

    preview_fingerprints = {
        name: _condition_fingerprint(result) for name, result in previews.items()
    }
    if existing_manifest is not None:
        stored = {c["name"]: c["benchmark_fingerprint"] for c in existing_manifest["conditions"]}
        if stored != preview_fingerprints:
            raise ValueError("Suite benchmark fingerprints changed; cannot resume this output")

    plan = {
        "schema_version": 1,
        "config_sha256": config_digest,
        "output": str(output),
        "execution_controls": execution_controls,
        "conditions": [
            {
                "name": condition["name"],
                "repeats": condition["repeats"],
                "benchmark_fingerprint": preview_fingerprints[condition["name"]],
                "runs": [
                    {"repetition": repetition, "status": "planned"}
                    for repetition in range(1, condition["repeats"] + 1)
                ],
                "preview": previews[condition["name"]],
            }
            for condition in conditions
        ],
        "inference": False,
    }
    if dry_run:
        return plan

    if existing_manifest is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        manifest = {
            "schema_version": 1,
            "suite_id": f"suite-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{config_digest[:8]}",
            "created_at": _now(),
            "config_path": str(config_path),
            "config_sha256": config_digest,
            "execution_controls": execution_controls,
            "normalized_suite": _jsonable(suite),
            "conditions": [
                {
                    "name": condition["name"],
                    "repeats": condition["repeats"],
                    "config": _jsonable(condition),
                    "benchmark_fingerprint": preview_fingerprints[condition["name"]],
                    "preview": _jsonable(previews[condition["name"]]),
                }
                for condition in conditions
            ],
            "runs": {},
            "comparisons": [],
            "summary": None,
            "status": "pending",
        }
    else:
        manifest = existing_manifest

    # Preview identity/fingerprints and config hash are immutable; all run records
    # are appended atomically as the engine creates their directories.
    suite_lock = None
    if not _lock_held:
        suite_lock = runner._lock(output)
        suite_lock.__enter__()
    try:
        _write_suite_state(manifest_path, manifest)
        for condition, repetition in _expected_repetitions(suite):
            name = condition["name"]
            key = _run_key(name, repetition)
            record = manifest["runs"].get(key)
            run_dir = Path(record["run_dir"]) if record and record.get("run_dir") else None
            if run_dir is not None:
                run_manifest = _load_run_state(run_dir)
                _verify_run_identity(
                    run_manifest, condition, repetition, previews[name], execution_controls
                )
                state = run_manifest.get("status")
                if state in _FINISHED_RUN_STATUSES:
                    record.update(status=state, run_id=run_dir.name)
                    _write_suite_state(manifest_path, manifest)
                    continue
                if state in _RESUMABLE_RUN_STATUSES:
                    record.update(status=state)
                    _write_suite_state(manifest_path, manifest)
                    progress(f"Resuming {name} repetition {repetition}: {run_dir}")
                    resumed = resume_fn(
                        run_dir,
                        progress=lambda message, n=name, r=repetition, d=run_dir: _progress(
                            message, n, r, d, manifest, manifest_path, key, progress
                        ),
                    )
                    run_dir = Path(resumed).resolve()
                elif state is None:
                    raise ValueError(f"Run manifest has no status: {run_dir}")
                elif state is not None:
                    raise ValueError(
                        f"Run {run_dir} has non-resumable status {state!r}; inspect it before continuing"
                    )

            if run_dir is None:
                record = {
                    "condition": name,
                    "repetition": repetition,
                    "status": "pending",
                    "run_dir": None,
                    "run_id": None,
                }
                manifest["runs"][key] = record
                _write_suite_state(manifest_path, manifest)
                kwargs = _condition_kwargs(condition, config.get("costs"))
                kwargs.update(
                    base_url=base_url,
                    timeout=timeout,
                    num_predict=num_predict,
                    max_retries=max_retries,
                    concurrency=concurrency,
                    max_requests=max_requests,
                    max_spend_usd=max_spend_usd,
                    cache_dir=cache_dir,
                    suite_condition=name,
                    repetition=repetition,
                    repeated_measurement=condition["repeats"] > 1,
                    progress=lambda message, n=name, r=repetition, d=None: _progress(
                        message, n, r, d, manifest, manifest_path, key, progress
                    ),
                )
                run_dir = Path(
                    run_fn(None, list(condition["models"]), output / "runs", **kwargs)
                ).resolve()

            state_manifest = _load_run_state(run_dir)
            actual_fingerprint = state_manifest.get("benchmark_fingerprint")
            expected_fingerprint = preview_fingerprints[name]
            if actual_fingerprint != expected_fingerprint:
                record = manifest["runs"].setdefault(
                    key,
                    {"condition": name, "repetition": repetition, "run_dir": str(run_dir)},
                )
                record.update(
                    run_dir=str(run_dir), run_id=run_dir.name, status="fingerprint_mismatch"
                )
                _write_suite_state(manifest_path, manifest)
                raise ValueError(
                    f"Run benchmark fingerprint differs from preview for condition {name!r}: "
                    f"expected {expected_fingerprint}, got {actual_fingerprint}"
                )
            _verify_run_identity(
                state_manifest, condition, repetition, previews[name], execution_controls
            )
            record = manifest["runs"].setdefault(key, {"condition": name, "repetition": repetition})
            record.update(
                condition=name,
                repetition=repetition,
                run_dir=str(run_dir),
                run_id=run_dir.name,
                status=state_manifest.get("status", "unknown"),
                benchmark_fingerprint=actual_fingerprint,
            )
            _write_suite_state(manifest_path, manifest)

        manifest["comparisons"] = _compare_conditions(
            manifest, output, compare_fn, manifest.get("comparisons", [])
        )
        report = _build_report(manifest, output)
        _persist_report(report, output / _REPORT_FILE)
        manifest["summary"] = str(output / _REPORT_FILE)
        if not all(
            entry.get("status") in _FINISHED_RUN_STATUSES for entry in manifest["runs"].values()
        ):
            manifest["status"] = "incomplete"
        elif any(
            entry.get("status") == "complete_with_errors" for entry in manifest["runs"].values()
        ):
            manifest["status"] = "complete_with_errors"
        else:
            manifest["status"] = "complete"
        _write_suite_state(manifest_path, manifest)
        report["comparisons"] = manifest["comparisons"]
        return report
    except BaseException:
        manifest["status"] = "interrupted"
        _write_suite_state(manifest_path, manifest)
        raise
    finally:
        if suite_lock is not None:
            suite_lock.__exit__(None, None, None)


def run_suite(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    base_url: str = "http://localhost:11434",
    timeout: float = 300,
    num_predict: int = 4096,
    max_retries: int | None = None,
    concurrency: int | None = None,
    max_requests: int | None = None,
    max_spend_usd: float | None = None,
    cache_dir: str | Path | None = None,
    progress: Callable[[str], None] = print,
    preview_fn: Callable | None = None,
    run_fn: Callable | None = None,
    resume_fn: Callable | None = None,
    compare_fn: Callable | None = None,
) -> dict[str, Any]:
    """Run a suite while holding its output lock across resume and execution."""
    output = Path(output_dir).expanduser().resolve()
    if resume and not dry_run:
        if not output.is_dir():
            raise ValueError("--resume requires an existing suite output with suite.json")
        with runner._lock(output):
            return _run_suite_impl(
                config_path,
                output,
                dry_run=dry_run,
                resume=resume,
                base_url=base_url,
                timeout=timeout,
                num_predict=num_predict,
                max_retries=max_retries,
                concurrency=concurrency,
                max_requests=max_requests,
                max_spend_usd=max_spend_usd,
                cache_dir=cache_dir,
                progress=progress,
                preview_fn=preview_fn,
                run_fn=run_fn,
                resume_fn=resume_fn,
                compare_fn=compare_fn,
                _lock_held=True,
            )
    return _run_suite_impl(
        config_path,
        output,
        dry_run=dry_run,
        resume=resume,
        base_url=base_url,
        timeout=timeout,
        num_predict=num_predict,
        max_retries=max_retries,
        concurrency=concurrency,
        max_requests=max_requests,
        max_spend_usd=max_spend_usd,
        cache_dir=cache_dir,
        progress=progress,
        preview_fn=preview_fn,
        run_fn=run_fn,
        resume_fn=resume_fn,
        compare_fn=compare_fn,
    )


def _progress(
    message: str,
    condition: str,
    repetition: int,
    run_dir: Path | None,
    manifest: dict[str, Any],
    manifest_path: Path,
    key: str,
    progress: Callable[[str], None],
) -> None:
    prefix = "Run directory: "
    if message.startswith(prefix):
        run_dir = Path(message[len(prefix) :]).expanduser().resolve()
        entry = manifest["runs"].setdefault(key, {})
        entry.update(
            condition=condition,
            repetition=repetition,
            status="running",
            run_dir=str(run_dir),
            run_id=run_dir.name,
        )
        _write_suite_state(manifest_path, manifest)
    progress(f"[{condition} repetition {repetition}] {message}")
