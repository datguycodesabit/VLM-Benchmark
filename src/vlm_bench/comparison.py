"""Compare saved benchmark runs that share one verified benchmark snapshot."""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import runner
from .backends import parse_model
from .export import _csv_value, _table_columns
from .metrics import score
from .research import _sample_kind, research_reports, score_equation

SCORING_VERSION = "1"

_MANIFEST_EXCLUSIONS = {"integrity", "status", "updated_at"}
_MODEL_SUMMARY_COLUMNS = (
    "track",
    "run",
    "original_model",
    "rank",
    "model",
    "status",
    "settings",
    "provider_controls",
    "effective_settings",
    "prompt_hashes",
    "prose_prompt_hash",
    "math_prompt_hash",
    "model_revision",
    "unsupported_reason",
    "eligible_sample_count",
    "scored_sample_count",
    "failed_sample_count",
    "failure_rate",
    "missing_sample_count",
    "sample_coverage",
    "cer",
    "wer",
    "exact_match_rate",
    "mean_latency_seconds",
    "median_latency_seconds",
    "p95_latency_seconds",
    "total_latency_seconds",
    "samples_per_minute",
    "truncated_count",
)
_PAIRED_COLUMNS = (
    "track",
    "left_model",
    "right_model",
    "status",
    "paired_sample_count",
    "left_corpus_cer",
    "right_corpus_cer",
    "corpus_cer_difference_right_minus_left",
    "bootstrap_unit",
    "bootstrap_groups",
    "bootstrap_replicates",
    "ci95_low",
    "ci95_high",
    "unsupported_models",
)
_COST_COLUMNS = (
    "run",
    "sample_volume",
    "local_total_cost_usd",
    "api_total_cost_usd",
    "local_cost_per_sample_usd",
    "api_cost_per_sample_usd",
    "local_minus_api_cost_usd",
    "inputs_complete",
    "break_even_status",
    "break_even_samples",
)


def _sample_fingerprint(samples: list[dict[str, Any]], preprocess: str) -> str:
    # Import lazily so inspection and legacy-run error handling do not depend on
    # the snapshot module until a fingerprint actually needs to be verified.
    try:
        from .snapshot import fingerprint
    except ImportError as exc:  # pragma: no cover - package always ships this module
        raise RuntimeError("Benchmark snapshot verification is unavailable") from exc
    return fingerprint(samples, preprocess)


def _preprocess_profile(manifest: dict[str, Any]) -> str:
    profile = manifest.get("preprocess", manifest.get("preprocessing"))
    if isinstance(profile, str) and profile:
        return profile
    sample_profiles = {
        sample.get("preprocess")
        for sample in manifest.get("samples", [])
        if isinstance(sample, dict) and isinstance(sample.get("preprocess"), str)
    }
    if len(sample_profiles) == 1:
        return next(iter(sample_profiles))
    raise ValueError(
        "Run manifest has no single preprocessing profile; comparison requires a frozen "
        "benchmark snapshot with its preprocessing setting."
    )


def _manifest_models(manifest: dict[str, Any], run_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    configured = manifest.get("models")
    if not isinstance(configured, list):
        raise ValueError(f"Run manifest has no model list: {run_dir}")
    result: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for item in configured:
        if isinstance(item, str) and item:
            selector, config = item, {}
        elif isinstance(item, dict):
            selector = next(
                (
                    item.get(key)
                    for key in ("selector", "name", "model", "id")
                    if isinstance(item.get(key), str) and item.get(key)
                ),
                None,
            )
            config = dict(item)
        else:
            selector, config = None, {}
        if not selector:
            raise ValueError(f"Run manifest has a model without a selector: {run_dir}")
        canonical = _canonical_model(selector)
        if canonical in seen:
            raise ValueError(
                f"Run manifest contains duplicate model selector {canonical!r}: {run_dir}"
            )
        seen.add(canonical)
        result.append((selector, config))
    return result


def _canonical_model(selector: str) -> str:
    provider, model_id = parse_model(selector)
    return f"{provider}:{model_id}"


def _validate_result_pairs(
    records: list[Any], manifest: dict[str, Any], run_dir: Path
) -> list[dict[str, Any]]:
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Run manifest has no frozen sample list: {run_dir}")
    sample_ids: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("id") is None:
            raise ValueError(f"Run manifest contains a sample without an ID: {run_dir}")
        sample_id = str(sample["id"])
        if sample_id in sample_ids:
            raise ValueError(f"Run manifest contains duplicate sample ID {sample_id!r}: {run_dir}")
        sample_ids.add(sample_id)
    model_names = {_canonical_model(name) for name, _ in _manifest_models(manifest, run_dir)}
    seen_pairs: set[tuple[str, str]] = set()
    valid: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Result record {index} is not an object: {run_dir}")
        model = record.get("model")
        if isinstance(model, dict):
            model = next(
                (
                    model.get(key)
                    for key in ("selector", "name", "model", "id")
                    if isinstance(model.get(key), str) and model.get(key)
                ),
                None,
            )
        sample_id = record.get("sample_id")
        if not isinstance(model, str) or not model or sample_id is None:
            raise ValueError(f"Result record {index} lacks a model/sample ID pair: {run_dir}")
        sample_id = str(sample_id)
        canonical_model = _canonical_model(model)
        pair = (canonical_model, sample_id)
        if canonical_model not in model_names or sample_id not in sample_ids:
            raise ValueError(
                f"Result record has unknown model/sample pair ({model!r}, {sample_id!r}): {run_dir}"
            )
        if pair in seen_pairs:
            raise ValueError(
                f"Run results contain duplicate model/sample pair ({canonical_model!r}, {sample_id!r}): {run_dir}"
            )
        seen_pairs.add(pair)
        valid.append(record)
    return valid


def _read_run(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Run manifest is missing: {manifest_path}")

    with runner._lock(run_dir):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read run manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"Run manifest must contain a JSON object: {manifest_path}")
        integrity = manifest.get("integrity")
        immutable = {
            key: value for key, value in manifest.items() if key not in _MANIFEST_EXCLUSIONS
        }
        if not isinstance(integrity, str) or runner._digest(immutable) != integrity:
            raise ValueError(f"Run manifest integrity check failed: {manifest_path}")

        stored_fingerprint = manifest.get("benchmark_fingerprint")
        if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
            raise ValueError(
                f"Cannot compare legacy run {run_dir}: it has no benchmark fingerprint. "
                "Create a new schema 2 run with the current benchmark version, then compare "
                "those runs. Existing runs can still be exported."
            )
        samples = manifest.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"Run manifest has no frozen sample list: {manifest_path}")
        preprocess = _preprocess_profile(manifest)
        computed_fingerprint = _sample_fingerprint(samples, preprocess)
        if computed_fingerprint != stored_fingerprint:
            raise ValueError(
                f"Benchmark fingerprint does not match the frozen snapshot in {run_dir}; "
                "sample IDs, crop images, references, preprocessing, or sample metadata changed."
            )
        try:
            records = runner.read_records(run_dir / "results.jsonl")
            warmups = runner.read_records(run_dir / "warmups.jsonl")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read saved results for {run_dir}: {exc}") from exc
        records = _validate_result_pairs(records, manifest, run_dir)
        warmups = [row for row in warmups if isinstance(row, dict)]

    return {
        "path": run_dir,
        "manifest": manifest,
        "records": records,
        "warmups": warmups,
        "fingerprint": stored_fingerprint,
        "preprocess": preprocess,
    }


def _unique_run_labels(runs: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    used: set[str] = set()
    for run in runs:
        base = run["path"].name or "run"
        label = base
        suffix = 2
        while label in used:
            label = f"{base}-{suffix}"
            suffix += 1
        used.add(label)
        labels.append(label)
    return labels


def _entry_selector(model: str, run_label: str) -> str:
    provider, model_id = parse_model(model)
    return f"{provider}:{model_id}@{run_label}"


def _effective_settings(
    manifest: dict[str, Any], preprocess: str, provider_controls: dict[str, Any]
) -> dict[str, Any]:
    return {
        "options": copy.deepcopy(manifest.get("options")),
        "provider_controls": copy.deepcopy(provider_controls.get("effective", {})),
        "prompt": manifest.get("prompt"),
        "timeout": manifest.get("timeout"),
        "warmup": manifest.get("warmup"),
        "preprocess": preprocess,
        "scoring_version": manifest.get("scoring_version"),
        "research": copy.deepcopy(manifest.get("research", {})),
    }


def _combine_runs(runs: list[dict[str, Any]], labels: list[str]):
    samples = copy.deepcopy(runs[0]["manifest"]["samples"])
    merged_models: list[str] = []
    merged_records: list[dict[str, Any]] = []
    merged_warmups: list[dict[str, Any]] = []
    model_info: dict[str, Any] = {}
    model_capabilities: dict[str, Any] = {}
    entries: dict[str, Any] = {}

    for run, run_label in zip(runs, labels, strict=True):
        manifest = run["manifest"]
        run_models = _manifest_models(manifest, run["path"])
        source_model_info = manifest.get("model_info", {})
        if not isinstance(source_model_info, dict):
            source_model_info = {}
        source_capabilities = manifest.get("model_capabilities", {})
        if not isinstance(source_capabilities, dict):
            source_capabilities = {}
        source_settings = manifest.get("settings", {})
        if not isinstance(source_settings, dict):
            source_settings = {}
        source_controls = manifest.get("provider_controls", {})
        if not isinstance(source_controls, dict):
            source_controls = {}
        prompt_hashes = copy.deepcopy(manifest.get("prompt_hashes", {}))
        if not isinstance(prompt_hashes, dict):
            prompt_hashes = {}
        model_mapping: dict[str, str] = {}

        for original_model, config in run_models:
            entry = _entry_selector(original_model, run_label)
            if entry in entries:
                raise ValueError(f"Comparison entry selector is not unique: {entry}")
            canonical_model = _canonical_model(original_model)
            model_mapping[canonical_model] = entry
            merged_models.append(entry)
            copied_info = copy.deepcopy(source_model_info.get(original_model))
            model_info[entry] = copied_info

            capability = source_capabilities.get(original_model)
            if not isinstance(capability, dict):
                capability = config
            if capability:
                model_capabilities[entry] = copy.deepcopy(capability)

            controls = copy.deepcopy(source_controls.get(original_model, {}))
            if not isinstance(controls, dict):
                controls = {}
            entries[entry] = {
                "run": run_label,
                "run_directory": str(run["path"]),
                "run_created_at": manifest.get("created_at"),
                "run_status": manifest.get("status"),
                "model": original_model,
                "canonical_model": _canonical_model(original_model),
                "settings": copy.deepcopy(source_settings.get(original_model, {})),
                "provider_controls": controls,
                "effective_settings": _effective_settings(manifest, run["preprocess"], controls),
                "prompt": manifest.get("prompt"),
                "math_prompt": manifest.get("math_prompt"),
                "prompt_hashes": prompt_hashes,
                "prose_prompt_hash": prompt_hashes.get("prose"),
                "math_prompt_hash": prompt_hashes.get("math", prompt_hashes.get("equation")),
                "model_revision": copied_info,
                "model_config": copy.deepcopy(capability),
                "benchmark_version": manifest.get("benchmark_version"),
                "scoring_version": manifest.get("scoring_version"),
            }

        frozen_samples = {str(sample["id"]): sample for sample in samples}
        for source_record in run["records"]:
            record = copy.deepcopy(source_record)
            original_model = record["model"]
            if isinstance(original_model, dict):
                original_model = next(
                    original_model[key]
                    for key in ("selector", "name", "model", "id")
                    if isinstance(original_model.get(key), str) and original_model.get(key)
                )
            sample_id = str(record["sample_id"])
            frozen_sample = frozen_samples[sample_id]
            reference = frozen_sample.get("reference")
            canonical_model = _canonical_model(original_model)
            record["model"] = model_mapping[canonical_model]
            record["original_model"] = original_model
            record["run"] = run_label
            record["reference"] = reference
            prediction = record.get("prediction")
            if (
                record.get("status") == "success"
                and isinstance(prediction, str)
                and isinstance(reference, str)
            ):
                try:
                    record["metrics"] = (
                        score_equation(prediction, reference)
                        if _sample_kind(frozen_sample) == "equation"
                        else score(prediction, reference)
                    )
                except ValueError:
                    record["metrics"] = None
            else:
                record["metrics"] = None
            merged_records.append(record)

        for source_warmup in run["warmups"]:
            warmup = copy.deepcopy(source_warmup)
            original_model = warmup.get("model")
            if (
                isinstance(original_model, str)
                and _canonical_model(original_model) in model_mapping
            ):
                warmup["model"] = model_mapping[_canonical_model(original_model)]
                warmup["run"] = run_label
                merged_warmups.append(warmup)

    base_manifest = copy.deepcopy(runs[0]["manifest"])
    base_manifest.update(
        {
            "samples": samples,
            "models": merged_models,
            "model_info": model_info,
            "model_capabilities": model_capabilities,
            "warmups": merged_warmups,
            "benchmark_fingerprint": runs[0]["fingerprint"],
            "scoring_version": SCORING_VERSION,
        }
    )
    return base_manifest, merged_records, entries


def _comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entries = report["entries"]
    for track_name, track in report["tracks"].items():
        for result in track["model_results"]:
            entry = entries[result["model"]]
            rows.append(
                {
                    "track": track_name,
                    "run": entry["run"],
                    "original_model": entry["model"],
                    "settings": entry["settings"],
                    "provider_controls": entry["provider_controls"],
                    "effective_settings": entry["effective_settings"],
                    "prompt_hashes": entry["prompt_hashes"],
                    "prose_prompt_hash": entry["prose_prompt_hash"],
                    "math_prompt_hash": entry["math_prompt_hash"],
                    "model_revision": entry["model_revision"],
                    **result,
                }
            )
    return rows


def _paired_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track_name, track in report["tracks"].items():
        for pair in track["paired_differences"]:
            bootstrap = pair.get("bootstrap", {})
            interval = bootstrap.get("ci95")
            rows.append(
                {
                    "track": track_name,
                    **pair,
                    "bootstrap_unit": bootstrap.get("unit"),
                    "bootstrap_groups": bootstrap.get("groups"),
                    "bootstrap_replicates": bootstrap.get("replicates"),
                    "ci95_low": interval[0]
                    if isinstance(interval, list) and len(interval) == 2
                    else None,
                    "ci95_high": interval[1]
                    if isinstance(interval, list) and len(interval) == 2
                    else None,
                }
            )
    return rows


def _cost_rows(costs_by_run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_label, report in costs_by_run.items():
        break_even = report["break_even"]
        scenarios = report["scenarios"]
        if not scenarios:
            rows.append(
                {
                    "run": run_label,
                    "break_even_status": break_even.get("status"),
                    "break_even_samples": break_even.get("samples"),
                }
            )
            continue
        for scenario in scenarios:
            rows.append(
                {
                    "run": run_label,
                    **scenario,
                    "break_even_status": break_even.get("status"),
                    "break_even_samples": break_even.get("samples"),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: tuple[str, ...]) -> None:
    columns = _table_columns(rows, preferred)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _write_report(stage: Path, report: dict[str, Any], costs_by_run: dict[str, Any]) -> None:
    report_path = stage / "comparison.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _write_csv(stage / "comparison.csv", _comparison_rows(report), _MODEL_SUMMARY_COLUMNS)
    _write_csv(stage / "paired.csv", _paired_rows(report), _PAIRED_COLUMNS)
    _write_csv(stage / "costs.csv", _cost_rows(costs_by_run), _COST_COLUMNS)


def compare(run_dirs, output_dir) -> dict[str, Any]:
    """Build a comparison from two or more saved runs on the same snapshot.

    Saved predictions and timing metadata are read under each run's OS lock.
    The returned metrics are recalculated against the frozen references stored
    in each run's manifest; no model inference is performed.
    """
    if isinstance(run_dirs, (str, os.PathLike)):
        requested_paths = [Path(run_dirs)]
    else:
        try:
            requested_paths = [Path(value) for value in run_dirs]
        except TypeError as exc:
            raise ValueError("Provide at least two run directories") from exc
    if len(requested_paths) < 2:
        raise ValueError("Comparison requires at least two distinct run directories")
    paths = [path.expanduser().resolve() for path in requested_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Comparison requires at least two distinct run directories")

    output_path = Path(output_dir).expanduser().resolve()
    if os.path.lexists(output_path):
        raise FileExistsError(f"Comparison output already exists: {output_path}")
    for source in paths:
        if output_path == source or source in output_path.parents:
            raise ValueError(f"Comparison output cannot be inside a source run: {source}")

    runs = [_read_run(path) for path in paths]
    fingerprints = {run["fingerprint"] for run in runs}
    if len(fingerprints) != 1:
        raise ValueError(
            "Runs do not share the same benchmark snapshot. Sample IDs, crop images, "
            "references, preprocessing, or sample metadata differ; compare runs created "
            "from the same frozen dataset snapshot."
        )

    labels = _unique_run_labels(runs)
    merged_manifest, merged_records, entries = _combine_runs(runs, labels)
    aggregate = research_reports(merged_records, merged_manifest)

    costs_by_run: dict[str, Any] = {}
    for run, run_label in zip(runs, labels, strict=True):
        per_run_report = research_reports(run["records"], run["manifest"])
        costs_by_run[run_label] = per_run_report["cost_scenarios"]
    aggregate["cost_scenarios"] = {
        "runs": [
            {"run": run_label, **cost_report} for run_label, cost_report in costs_by_run.items()
        ]
    }

    report = {
        **aggregate,
        "comparison_schema_version": 1,
        "scoring_version": SCORING_VERSION,
        "benchmark_fingerprint": runs[0]["fingerprint"],
        "runs": [
            {
                "label": label,
                "directory": str(run["path"]),
                "created_at": run["manifest"].get("created_at"),
                "status": run["manifest"].get("status"),
                "benchmark_version": run["manifest"].get("benchmark_version"),
                "scoring_version": run["manifest"].get("scoring_version"),
                "preprocess": run["preprocess"],
                "models": [
                    entry for entry, metadata in entries.items() if metadata["run"] == label
                ],
            }
            for run, label in zip(runs, labels, strict=True)
        ],
        "entries": entries,
        "model_info": copy.deepcopy(merged_manifest["model_info"]),
        "output_files": ["comparison.json", "comparison.csv", "paired.csv", "costs.csv"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reservation = output_path.with_name(f".{output_path.name}.comparison-lock")
    try:
        reservation_fd = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Another comparison is preparing this output (or left a lock): {reservation}"
        ) from exc
    os.close(reservation_fd)
    try:
        if os.path.lexists(output_path):
            raise FileExistsError(f"Comparison output already exists: {output_path}")
        stage = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
        try:
            _write_report(stage, report, costs_by_run)
            if os.path.lexists(output_path):
                raise FileExistsError(f"Comparison output already exists: {output_path}")
            stage.rename(output_path)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    finally:
        try:
            reservation.unlink()
        except FileNotFoundError:
            pass
    return report
