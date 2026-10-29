"""Offline weighted evaluation and its renderer-neutral report model.

This module consumes frozen samples and result records only. It never starts a
backend or calls a provider, and it leaves legacy CER summaries unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .equivalence import EQUIVALENCE_VERSION, compare_equations
from .metrics import normalize_text, score
from .ranking import rank_track
from .research import (
    _has_document_metadata,
    _model_name,
    _sample_document,
    _sample_exclusion,
    _sample_kind,
    _unsupported_reason,
    normalize_equation,
    score_equation,
)
from .scoring_policy import ScoringPolicy

REPORT_MODEL_VERSION = 1
EVALUATION_SCHEMA_VERSION = 2
_TERMINAL_RUN_STATUSES = {"complete", "complete_with_errors", "completed", "success"}


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _flatten_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    metadata = value.get("metadata")
    flattened = dict(metadata) if isinstance(metadata, Mapping) else {}
    flattened.update(value)
    return flattened


def _track_for_sample(sample: Mapping[str, Any]) -> str:
    return "equation" if _sample_kind(sample) == "equation" else "prose"


def _normalized_reference(reference: str, track: str) -> str:
    return normalize_equation(reference) if track == "equation" else normalize_text(reference)


def _speed_comparable(manifest: Mapping[str, Any]) -> bool:
    profile = manifest.get("measurement_profile")
    if manifest.get("external_predictions") and not isinstance(profile, Mapping):
        return False
    profile_name = profile.get("profile") if isinstance(profile, Mapping) else "latency"
    concurrency = (
        profile.get("concurrency", manifest.get("concurrency", 1))
        if isinstance(profile, Mapping)
        else manifest.get("concurrency", 1)
    )
    return profile_name == "latency" and concurrency == 1


def _record_pass(record: Mapping[str, Any]) -> int:
    value = next((record[key] for key in ("pass", "pass_index", "repetition") if key in record), 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("record pass identity must be a positive integer")
    return value


def _configured_models(
    models: Sequence[Any] | None,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    configured = models if models is not None else manifest.get("models", [])
    result: list[str] = []
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        for item in configured:
            name = _model_name(item)
            if not name:
                raise ValueError("model selectors must be non-empty strings or model mappings")
            if name not in result:
                result.append(name)
    elif configured:
        raise TypeError("models must be a sequence")
    for row in records:
        name = _model_name(row.get("model"))
        if name and name not in result:
            result.append(name)
    return result


def _key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    model = _model_name(record.get("model"))
    sample_id = record.get("sample_id")
    if not model or sample_id is None:
        raise ValueError("each result record must include a model and sample_id")
    return model, _record_pass(record), str(sample_id)


def _score_values(
    edits: int,
    reference_chars: int,
    word_edits: int,
    reference_words: int,
    exact_matches: int,
    sample_count: int,
    policy: ScoringPolicy,
) -> dict[str, Any]:
    cer = edits / reference_chars if reference_chars else None
    wer = word_edits / reference_words if reference_words else None
    exact_rate = exact_matches / sample_count if sample_count else None
    character_accuracy = max(0.0, 1.0 - cer) if cer is not None else None
    composite = (
        100.0
        * (
            policy.character_accuracy_weight * character_accuracy
            + policy.exact_match_weight * exact_rate
        )
        if character_accuracy is not None and exact_rate is not None
        else None
    )
    return {
        "score": composite,
        "corpus_cer": cer,
        "wer": wer,
        "exact_match_rate": exact_rate,
        "character_accuracy": character_accuracy,
        "character_edits": edits,
        "reference_chars": reference_chars,
        "word_edits": word_edits,
        "reference_words": reference_words,
        "exact_matches": exact_matches,
        "eligible_sample_count": sample_count,
    }


def _cell(
    record: Mapping[str, Any] | None,
    sample: Mapping[str, Any],
    track: str,
) -> tuple[dict[str, Any], str, str | None]:
    reference = sample.get("reference")
    if not isinstance(reference, str) or not normalize_text(reference):
        raise ValueError(f"eligible sample {sample.get('id')!r} has no non-empty reference")
    normalized_reference = _normalized_reference(reference, track)
    if record is None:
        return (
            {
                "char_edits": len(normalized_reference),
                "reference_chars": len(normalized_reference),
                "word_edits": len(normalized_reference.split()),
                "reference_words": len(normalized_reference.split()),
                "exact_match": False,
            },
            "missing",
            None,
        )
    if record.get("status") == "unsupported":
        return {}, "unsupported", None
    prediction = record.get("prediction")
    if record.get("status") == "success" and isinstance(prediction, str):
        try:
            metrics = (
                score_equation(prediction, reference)
                if track == "equation"
                else score(prediction, reference)
            )
        except (TypeError, ValueError):
            metrics = None
        if metrics is not None:
            return metrics, "success", prediction
    return (
        {
            "char_edits": len(normalized_reference),
            "reference_chars": len(normalized_reference),
            "word_edits": len(normalized_reference.split()),
            "reference_words": len(normalized_reference.split()),
            "exact_match": False,
        },
        "failed",
        None,
    )


def _aggregate_pass(
    sample_ids: Sequence[str],
    samples: Mapping[str, Mapping[str, Any]],
    records: Mapping[tuple[str, int, str], Mapping[str, Any]],
    model: str,
    pass_index: int,
    track: str,
    policy: ScoringPolicy,
    cells: Mapping[tuple[str, int, str], tuple[dict[str, Any], str, str | None]],
    speed_comparable: bool,
) -> tuple[dict[str, Any], list[float], list[str], list[str], int, int, int]:
    edits = reference_chars = word_edits = reference_words = exact_matches = 0
    missing_ids: list[str] = []
    failed_ids: list[str] = []
    success_count = cache_hits = empty_count = 0
    latencies: list[float] = []
    timing_observations: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        record = records.get((model, pass_index, sample_id))
        metrics, outcome, prediction = cells[(model, pass_index, sample_id)]
        if outcome == "unsupported":
            continue
        edits += int(metrics["char_edits"])
        reference_chars += int(metrics["reference_chars"])
        word_edits += int(metrics["word_edits"])
        reference_words += int(metrics["reference_words"])
        exact_matches += int(bool(metrics["exact_match"]))
        if outcome == "missing":
            missing_ids.append(sample_id)
        elif outcome == "failed":
            failed_ids.append(sample_id)
        else:
            success_count += 1
            if prediction is not None and not normalize_text(prediction):
                empty_count += 1
            if record and record.get("cache_hit") is True:
                cache_hits += 1
            elif record:
                latency = _finite_nonnegative(record.get("latency_seconds"))
                if latency is not None:
                    latencies.append(latency)
                observations = record.get("timing_observations")
                if not isinstance(observations, list):
                    details = record.get("provider_details")
                    observations = (
                        details.get("timing_observations") if isinstance(details, Mapping) else None
                    )
                if isinstance(observations, list):
                    timing_observations.extend(
                        {"sample_id": sample_id, "observation": observation}
                        for observation in observations
                        if isinstance(observation, Mapping)
                    )
    aggregate = _score_values(
        edits, reference_chars, word_edits, reference_words, exact_matches, len(sample_ids), policy
    )
    aggregate.update(
        {
            "pass": pass_index,
            "success_count": success_count,
            "failed_count": len(failed_ids),
            "missing_count": len(missing_ids),
            "failed_sample_ids": sorted(failed_ids),
            "missing_sample_ids": sorted(missing_ids),
            "empty_count": empty_count,
            "cache_hit_count": cache_hits,
            "measured_latency_sample_count": len(latencies),
            "measured_median_latency_seconds": statistics.median(latencies) if latencies else None,
            "measured_p95_latency_seconds": _percentile(latencies, 0.95),
            "median_latency_seconds": statistics.median(latencies)
            if latencies and speed_comparable
            else None,
            "p95_latency_seconds": _percentile(latencies, 0.95)
            if latencies and speed_comparable
            else None,
            "latency_rank_eligible": speed_comparable,
            "terminal_complete": not missing_ids,
            "timing_observations": timing_observations,
        }
    )
    return aggregate, latencies, missing_ids, failed_ids, success_count, cache_hits, empty_count


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _document_bootstrap(
    track: str,
    sample_ids: Sequence[str],
    samples: Mapping[str, Mapping[str, Any]],
    records: Mapping[tuple[str, int, str], Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    planned_passes: int,
    policy: ScoringPolicy,
    replicates: int,
    seed: int,
    cells: Mapping[tuple[str, int, str], tuple[dict[str, Any], str, str | None]],
) -> dict[str, Any]:
    if not sample_ids:
        return {"status": "unavailable", "reason": "no_eligible_samples", "unit": None}
    if not all(_has_document_metadata(samples[sample_id]) for sample_id in sample_ids):
        return {"status": "unavailable", "reason": "source_document_metadata_missing", "unit": None}
    by_document: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        document = _sample_document(dict(samples[sample_id]), sample_id)
        by_document.setdefault(document, []).append(sample_id)
    documents = sorted(by_document)
    if len(documents) < 2:
        return {
            "status": "unavailable",
            "reason": "fewer_than_two_source_documents",
            "unit": "source_document",
            "document_count": len(documents),
        }
    model_names = [str(row["model"]) for row in models]
    points = {str(row["model"]): row.get("score") for row in models}
    draws: dict[str, list[float]] = {model: [] for model in model_names}
    pairs = {
        (left, right): []
        for index, left in enumerate(model_names)
        for right in model_names[index + 1 :]
    }
    rng = random.Random(seed)
    for _ in range(replicates):
        picked = [rng.choice(documents) for _ in documents]
        selected_ids = [sample_id for document in picked for sample_id in by_document[document]]
        sample_scores: dict[str, float] = {}
        for model in model_names:
            pass_scores: list[float] = []
            for pass_index in range(1, planned_passes + 1):
                totals = [0, 0, 0, 0, 0]
                for sample_id in selected_ids:
                    metrics, outcome, _ = cells[(model, pass_index, sample_id)]
                    if outcome == "unsupported":
                        continue
                    totals[0] += int(metrics["char_edits"])
                    totals[1] += int(metrics["reference_chars"])
                    totals[2] += int(metrics["word_edits"])
                    totals[3] += int(metrics["reference_words"])
                    totals[4] += int(bool(metrics["exact_match"]))
                result = _score_values(*totals[:4], totals[4], len(selected_ids), policy)
                if result["score"] is not None:
                    pass_scores.append(float(result["score"]))
            if pass_scores:
                sample_scores[model] = statistics.fmean(pass_scores)
                draws[model].append(sample_scores[model])
        for left, right in pairs:
            if left in sample_scores and right in sample_scores:
                pairs[left, right].append(sample_scores[right] - sample_scores[left])
    model_intervals = {
        model: {
            "score": points.get(model),
            "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)] if values else None,
            "bootstrap_replicates_used": len(values),
        }
        for model, values in draws.items()
    }
    paired = [
        {
            "left_model": left,
            "right_model": right,
            "score_difference_right_minus_left": (
                float(points[right]) - float(points[left])
                if points.get(left) is not None and points.get(right) is not None
                else None
            ),
            "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)] if values else None,
            "bootstrap_replicates_used": len(values),
        }
        for (left, right), values in pairs.items()
    ]
    return {
        "status": "available",
        "unit": "source_document",
        "document_count": len(documents),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "pass_structure": "all passes remain grouped within the resampled document",
        "model_intervals": model_intervals,
        "paired_score_differences": paired,
    }


def _throughput_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    runs = manifest.get("execution_runs")
    if not isinstance(runs, list):
        state = manifest.get("execution_summary")
        invocations = state.get("invocations") if isinstance(state, Mapping) else None
        if isinstance(invocations, list):
            runs = [{"pass": 1, "run_id": None, "invocations": invocations}]
        else:
            runs = []
    if not runs:
        return {"status": "unavailable", "reason": "execution_invocations_not_recorded"}

    run_summaries: list[dict[str, Any]] = []
    all_complete = True
    total_wall = 0.0
    total_successful: int | None = 0
    total_terminal: int | None = 0
    numerator_labels: set[str] = set()
    denominator_labels: set[str] = set()
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        invocations = run.get("invocations", [])
        if not isinstance(invocations, list):
            invocations = []
        valid_invocations = [item for item in invocations if isinstance(item, Mapping)]
        durations = [
            duration
            for item in valid_invocations
            if (duration := _finite_nonnegative(item.get("workload_wall_seconds"))) is not None
        ]
        wall_seconds = sum(durations)
        total_wall += wall_seconds
        terminal_record = next(
            (
                item
                for item in reversed(valid_invocations)
                if item.get("status") in _TERMINAL_RUN_STATUSES
                and _nonnegative_count(item.get("successful_sample_count")) is not None
                and _nonnegative_count(item.get("terminal_sample_count")) is not None
            ),
            None,
        )
        successful = (
            _nonnegative_count(terminal_record.get("successful_sample_count"))
            if terminal_record is not None
            else None
        )
        terminal = (
            _nonnegative_count(terminal_record.get("terminal_sample_count"))
            if terminal_record is not None
            else None
        )
        complete = terminal_record is not None and wall_seconds > 0
        all_complete = all_complete and complete
        if successful is None or terminal is None:
            total_successful = None
            total_terminal = None
        elif total_successful is not None and total_terminal is not None:
            total_successful += successful
            total_terminal += terminal
        numerator = terminal_record.get("throughput_numerator") if terminal_record else None
        denominator = terminal_record.get("throughput_denominator") if terminal_record else None
        if isinstance(numerator, str):
            numerator_labels.add(numerator)
        if isinstance(denominator, str):
            denominator_labels.add(denominator)
        run_summaries.append(
            {
                "pass": run.get("pass"),
                "run_id": run.get("run_id"),
                "invocation_count": len(valid_invocations),
                "workload_wall_seconds": wall_seconds if durations else None,
                "successful_sample_count": successful,
                "terminal_sample_count": terminal,
                "successful_samples_per_second": (
                    successful / wall_seconds if complete and wall_seconds > 0 else None
                ),
                "terminal_samples_per_second": (
                    terminal / wall_seconds if complete and wall_seconds > 0 else None
                ),
                "status": "available" if complete else "partial",
                "throughput_numerator": numerator,
                "throughput_denominator": denominator,
                "invocations": [dict(item) for item in valid_invocations],
            }
        )
    return {
        "status": "available" if all_complete and run_summaries else "partial",
        "profile": (
            manifest.get("measurement_profile", {}).get("profile")
            if isinstance(manifest.get("measurement_profile"), Mapping)
            else None
        ),
        "concurrency": (
            manifest.get("measurement_profile", {}).get("concurrency", manifest.get("concurrency"))
            if isinstance(manifest.get("measurement_profile"), Mapping)
            else manifest.get("concurrency", 1)
        ),
        "run_count": len(run_summaries),
        "workload_wall_seconds": total_wall if total_wall > 0 else None,
        "successful_sample_count": total_successful,
        "terminal_sample_count": total_terminal,
        "successful_samples_per_second": (
            total_successful / total_wall
            if all_complete and total_successful is not None and total_wall > 0
            else None
        ),
        "terminal_samples_per_second": (
            total_terminal / total_wall
            if all_complete and total_terminal is not None and total_wall > 0
            else None
        ),
        "throughput_numerator": next(iter(numerator_labels))
        if len(numerator_labels) == 1
        else None,
        "throughput_denominator": next(iter(denominator_labels))
        if len(denominator_labels) == 1
        else None,
        "runs": run_summaries,
    }


def _nonnegative_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _equivalence_summary(
    model: str,
    sample_ids: Sequence[str],
    samples: Mapping[str, Mapping[str, Any]],
    records: Mapping[tuple[str, int, str], Mapping[str, Any]],
    planned_passes: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    attempted = 0
    witnesses: list[dict[str, Any]] = []
    for pass_index in range(1, planned_passes + 1):
        for sample_id in sample_ids:
            record = records.get((model, pass_index, sample_id))
            if not record or record.get("status") != "success":
                continue
            prediction = record.get("prediction")
            reference = samples[sample_id].get("reference")
            if not isinstance(prediction, str) or not isinstance(reference, str):
                continue
            domains = samples[sample_id].get("variable_domains")
            outcome = compare_equations(
                prediction, reference, domains if isinstance(domains, Mapping) else None
            )
            counts[outcome["status"]] += 1
            attempted += 1
            if outcome["status"] == "different" and outcome.get("witness") is not None:
                witnesses.append(
                    {"sample_id": sample_id, "pass": pass_index, "witness": outcome["witness"]}
                )
    determinate = counts["equivalent"] + counts["different"]
    return {
        "version": EQUIVALENCE_VERSION,
        "outcomes": {name: counts[name] for name in ("equivalent", "different", "undetermined")},
        "attempted_sample_pass_count": attempted,
        "applicable_sample_pass_count": len(sample_ids) * planned_passes,
        "attempted_coverage": attempted / (len(sample_ids) * planned_passes)
        if sample_ids
        else None,
        "determinate_sample_pass_count": determinate,
        "determinate_coverage": determinate / attempted if attempted else None,
        "witnesses": witnesses,
        "influence_on_primary_score": False,
    }


def evaluate_run(
    samples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    models: Sequence[Any] | None = None,
    policy: ScoringPolicy | Mapping[str, Any] | None = None,
    planned_passes: int = 1,
    run_status: str = "complete",
    manifest: Mapping[str, Any] | None = None,
    *,
    bootstrap_replicates: int = 250,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate offline evidence with failure penalties and separate track ranks."""
    if (
        isinstance(planned_passes, bool)
        or not isinstance(planned_passes, int)
        or planned_passes < 1
    ):
        raise ValueError("planned_passes must be a positive integer")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 1
    ):
        raise ValueError("bootstrap_replicates must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise TypeError("bootstrap_seed must be an integer")
    if not isinstance(run_status, str) or not run_status:
        raise ValueError("run_status must be a non-empty string")
    selected_policy = (
        policy if isinstance(policy, ScoringPolicy) else ScoringPolicy.from_dict(policy)
    )
    manifest_values = dict(manifest or {})
    version = manifest_values.get("evaluation_schema_version")
    if version not in (None, EVALUATION_SCHEMA_VERSION):
        raise ValueError(f"unsupported evaluation schema version: {version!r}")

    frozen: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(samples, start=1):
        if not isinstance(raw, Mapping):
            raise TypeError(f"sample {index} must be a mapping")
        sample = _flatten_metadata(raw)
        if sample.get("id") is None or not str(sample["id"]):
            raise ValueError(f"sample {index} must have a non-empty id")
        sample_id = str(sample["id"])
        if sample_id in frozen:
            raise ValueError(f"duplicate sample id: {sample_id!r}")
        sample["id"] = sample_id
        frozen[sample_id] = sample

    raw_records: list[dict[str, Any]] = []
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, Mapping):
            raise TypeError(f"result record {index} must be a mapping")
        raw_records.append(dict(raw))
    model_names = _configured_models(models, manifest_values, raw_records)
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in raw_records:
        key = _key(record)
        if key[1] > planned_passes:
            raise ValueError(f"record pass {key[1]} exceeds planned_passes={planned_passes}")
        if key[2] not in frozen:
            raise ValueError(f"result record references unknown sample id {key[2]!r}")
        if key in indexed:
            raise ValueError(f"duplicate result for model/pass/sample key {key!r}")
        indexed[key] = record
        if key[0] not in model_names:
            model_names.append(key[0])

    split_labels_exist = any(
        isinstance(sample.get("split"), str) and sample["split"].strip()
        for sample in frozen.values()
    )
    exclusions: list[dict[str, str]] = []
    track_samples: dict[str, dict[str, dict[str, Any]]] = {"prose": {}, "equation": {}}
    for sample_id, sample in frozen.items():
        track = _track_for_sample(sample)
        reason = _sample_exclusion(sample, split_labels_exist)
        reference = sample.get("reference")
        if not isinstance(reference, str) or not _normalized_reference(reference, track):
            reason = reason or "missing_reference"
        if reason:
            exclusions.append({"sample_id": sample_id, "track": track, "reason": reason})
        else:
            track_samples[track][sample_id] = sample

    capability_manifest = dict(manifest_values)
    capability_manifest.setdefault("models", list(models) if models is not None else model_names)
    track_reports: dict[str, Any] = {}
    run_is_complete = run_status.lower() in _TERMINAL_RUN_STATUSES
    speed_comparable = _speed_comparable(manifest_values)
    for track in ("prose", "equation"):
        eligible = track_samples[track]
        eligible_ids = sorted(eligible)
        model_rows: list[dict[str, Any]] = []
        track_cells: dict[tuple[str, int, str], tuple[dict[str, Any], str, str | None]] = {}
        for model in model_names:
            reason = _unsupported_reason(capability_manifest, model, track)
            unsupported_ids = {
                sample_id
                for sample_id in eligible_ids
                if any(
                    indexed.get((model, pass_index, sample_id), {}).get("status") == "unsupported"
                    for pass_index in range(1, planned_passes + 1)
                )
            }
            if reason:
                model_rows.append(
                    {
                        "model": model,
                        "status": "unsupported",
                        "unsupported_reason": reason,
                        "eligible_sample_count": 0,
                        "eligible_sample_ids": [],
                        "score": None,
                        "terminal_complete": True,
                        "per_pass": [],
                    }
                )
                continue
            model_ids = [
                sample_id for sample_id in eligible_ids if sample_id not in unsupported_ids
            ]
            if not model_ids:
                model_rows.append(
                    {
                        "model": model,
                        "status": "unsupported" if unsupported_ids else "no_eligible_samples",
                        "unsupported_reason": "provider_marked_task_unsupported"
                        if unsupported_ids
                        else None,
                        "unsupported_sample_ids": sorted(unsupported_ids),
                        "unsupported_sample_count": len(unsupported_ids),
                        "eligible_sample_count": 0,
                        "eligible_sample_ids": [],
                        "score": None,
                        "terminal_complete": True,
                        "per_pass": [],
                    }
                )
                continue
            cell_cache = {
                (model, pass_index, sample_id): _cell(
                    indexed.get((model, pass_index, sample_id)), eligible[sample_id], track
                )
                for pass_index in range(1, planned_passes + 1)
                for sample_id in model_ids
            }
            track_cells.update(cell_cache)
            per_pass: list[dict[str, Any]] = []
            latencies: list[float] = []
            successes = failed = missing = cache_hits = empty = 0
            all_terminal = True
            for pass_index in range(1, planned_passes + 1):
                (
                    result,
                    measured,
                    missing_ids,
                    failed_ids,
                    pass_successes,
                    pass_cache,
                    pass_empty,
                ) = _aggregate_pass(
                    model_ids,
                    eligible,
                    indexed,
                    model,
                    pass_index,
                    track,
                    selected_policy,
                    cell_cache,
                    speed_comparable,
                )
                per_pass.append(result)
                latencies.extend(measured)
                successes += pass_successes
                failed += len(failed_ids)
                missing += len(missing_ids)
                cache_hits += pass_cache
                empty += pass_empty
                all_terminal = all_terminal and result["terminal_complete"]
            pass_scores = [float(row["score"]) for row in per_pass if row.get("score") is not None]
            total_expected = len(model_ids) * planned_passes
            row: dict[str, Any] = {
                "model": model,
                "status": "complete" if all_terminal and run_is_complete else "incomplete",
                "unsupported_reason": None,
                "unsupported_sample_ids": sorted(unsupported_ids),
                "unsupported_sample_count": len(unsupported_ids),
                "eligible_sample_count": len(model_ids),
                "eligible_sample_ids": model_ids,
                "expected_sample_pass_count": total_expected,
                "completed_sample_pass_count": total_expected - missing,
                "success_count": successes,
                "failed_sample_count": failed,
                "missing_sample_count": missing,
                "failure_rate": (failed + missing) / total_expected if total_expected else None,
                "sample_coverage": successes / total_expected if total_expected else None,
                "score": statistics.fmean(pass_scores) if pass_scores else None,
                "accuracy_score": statistics.fmean(pass_scores) if pass_scores else None,
                "score_mean": statistics.fmean(pass_scores) if pass_scores else None,
                "score_stddev": statistics.stdev(pass_scores) if len(pass_scores) > 1 else None,
                "score_min": min(pass_scores) if pass_scores else None,
                "score_max": max(pass_scores) if pass_scores else None,
                "completed_pass_count": sum(bool(item["terminal_complete"]) for item in per_pass),
                "terminal_pass_count": sum(bool(item["terminal_complete"]) for item in per_pass),
                "corpus_cer": statistics.fmean(float(row["corpus_cer"]) for row in per_pass),
                "wer": statistics.fmean(float(row["wer"]) for row in per_pass),
                "exact_match_rate": statistics.fmean(
                    float(row["exact_match_rate"]) for row in per_pass
                ),
                "reference_chars": sum(int(row["reference_chars"]) for row in per_pass),
                "character_edits": sum(int(row["character_edits"]) for row in per_pass),
                "empty_count": empty,
                "cache_hit_count": cache_hits,
                "measured_latency_sample_count": len(latencies),
                "measured_median_latency_seconds": statistics.median(latencies)
                if latencies
                else None,
                "measured_p95_latency_seconds": _percentile(latencies, 0.95),
                "median_latency_seconds": (
                    statistics.median(latencies) if latencies and speed_comparable else None
                ),
                "p95_latency_seconds": (
                    _percentile(latencies, 0.95) if latencies and speed_comparable else None
                ),
                "latency_rank_eligible": speed_comparable,
                "latency_unavailable_reason": None
                if speed_comparable
                else "measurement_profile_not_sequential_latency",
                "terminal_complete": all_terminal and run_is_complete,
                "per_pass": per_pass,
            }
            if track == "equation":
                row["equivalence"] = _equivalence_summary(
                    model, model_ids, eligible, indexed, planned_passes
                )
            model_rows.append(row)

        rankable_rows = [row for row in model_rows if row.get("status") != "unsupported"]
        roster_signatures = {
            tuple(row.get("eligible_sample_ids", []))
            for row in rankable_rows
            if row.get("status") != "no_eligible_samples"
        }
        roster_compatible = len(roster_signatures) <= 1
        all_terminal = all(
            row.get("terminal_complete", False)
            for row in rankable_rows
            if row.get("status") != "no_eligible_samples"
        )
        finalizable = run_is_complete and all_terminal and roster_compatible
        ranking = rank_track(
            model_rows,
            accuracy_margin_points=selected_policy.accuracy_margin_points,
            finalizable=finalizable,
        )
        if not roster_compatible:
            ranking.update(
                {
                    "status": "incompatible_rosters",
                    "winner": None,
                    "winner_status": "withheld",
                    "winner_reason": "models_have_different_eligible_sample_rosters",
                    "bands": [],
                    "ranked_models": [],
                }
            )
        accuracy_order = [
            row["model"]
            for row in sorted(
                (row for row in model_rows if row.get("score") is not None),
                key=lambda row: (-float(row["score"]), row["model"]),
            )
        ]
        placements = {row["model"]: row for row in ranking["ranked_models"]}
        for row in model_rows:
            item = placements.get(row["model"], {})
            row["raw_accuracy_rank"] = (
                accuracy_order.index(row["model"]) + 1 if row["model"] in accuracy_order else None
            )
            row["rank"] = item.get("rank")
            row["accuracy_band"] = item.get("band")
            if not roster_compatible:
                row["rank"] = None
                row["accuracy_band"] = None

        uncertainty_rows = [
            row
            for row in model_rows
            if row.get("status") not in {"unsupported", "no_eligible_samples"}
            and row.get("score") is not None
        ]
        if roster_compatible:
            shared_ids = next(iter(roster_signatures), ())
            uncertainty = _document_bootstrap(
                track,
                shared_ids,
                eligible,
                indexed,
                uncertainty_rows,
                planned_passes,
                selected_policy,
                bootstrap_replicates,
                bootstrap_seed,
                track_cells,
            )
        else:
            uncertainty = {
                "status": "unavailable",
                "reason": "models_have_different_eligible_sample_rosters",
                "unit": None,
            }
        track_reports[track] = {
            "track": track,
            "eligible_sample_ids": eligible_ids,
            "eligible_sample_count": len(eligible_ids),
            "excluded_samples": [row for row in exclusions if row["track"] == track],
            "roster_compatible": roster_compatible,
            "measurement_profile": manifest_values.get(
                "measurement_profile", {"profile": "latency", "concurrency": 1}
            ),
            "speed_comparable": speed_comparable,
            "finalizable": finalizable,
            "models": model_rows,
            "model_results": model_rows,
            "raw_accuracy_order": accuracy_order,
            "ranking": ranking,
            "uncertainty": uncertainty,
        }

    identity = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "report_model_version": REPORT_MODEL_VERSION,
        "equivalence_version": EQUIVALENCE_VERSION,
        "samples": list(samples),
        "records": list(records),
        "models": model_names,
        "policy": selected_policy.to_dict(),
        "planned_passes": planned_passes,
        "run_status": run_status,
        "uncertainty_policy": {
            "method": "joint_document_percentile_bootstrap",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "missing_documents": "unavailable",
        },
        "execution_evidence": manifest_values.get(
            "execution_runs", manifest_values.get("execution_summary")
        ),
    }
    throughput = _throughput_summary(manifest_values)
    return {
        "schema_version": REPORT_MODEL_VERSION,
        "report_model_version": REPORT_MODEL_VERSION,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_id": _canonical_hash(identity),
        "policy": selected_policy.to_dict(),
        "planned_passes": planned_passes,
        "run_status": run_status,
        "throughput": throughput,
        "finalizable": run_is_complete
        and all(row["finalizable"] for row in track_reports.values()),
        "tracks": track_reports,
    }


def _canonical_hash(value: Any) -> str:
    value = _hash_evidence(value)
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_PATH_EVIDENCE_FIELDS = {
    "cache_dir",
    "crop_path",
    "image_path",
    "run_dir",
    "run_id",
    "source_path",
    "working_directory",
}


def _hash_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _hash_evidence(item)
            for key, item in value.items()
            if str(key).lower() not in _PATH_EVIDENCE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_hash_evidence(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return value


def _load_validated_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Keep run identity, result-pair, and fingerprint validation centralized in
    # the existing inspection loader.
    from .inspection import _load_run

    directory, manifest, samples, results, _evaluation, _eligible, _excluded = _load_run(run_dir)
    records = list(results.values())
    execution_path = directory / "execution.json"
    if execution_path.is_file():
        try:
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            execution = None
        if isinstance(execution, Mapping):
            manifest["execution_summary"] = dict(execution)
    return {**manifest, "samples": list(samples.values())}, records


def _validate_evaluation_schema(manifest: Mapping[str, Any]) -> None:
    version = manifest.get("evaluation_schema_version")
    if version is not None and (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != EVALUATION_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported evaluation schema version: {version!r}")
    profile = manifest.get("measurement_profile")
    for version in (
        manifest.get("measurement_schema_version"),
        profile.get("schema_version") if isinstance(profile, Mapping) else None,
    ):
        if version is not None and (type(version) is not int or version != 1):
            raise ValueError(f"unsupported measurement schema version: {version!r}")


def _policy_for_source(
    explicit: ScoringPolicy | Mapping[str, Any] | None,
    manifest: Mapping[str, Any],
    *,
    allow_missing_versioned_policy: bool = False,
) -> ScoringPolicy | None:
    value: ScoringPolicy | Mapping[str, Any] | None = explicit
    if value is None:
        stored = manifest.get("scoring_policy")
        value = stored if isinstance(stored, Mapping) else None
    if value is None:
        if (
            manifest.get("evaluation_schema_version") == EVALUATION_SCHEMA_VERSION
            and not allow_missing_versioned_policy
        ):
            raise ValueError("evaluation schema 2 requires a versioned scoring_policy")
        return None
    return value if isinstance(value, ScoringPolicy) else ScoringPolicy.from_dict(value)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read report source: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"report source must be a JSON object: {path}")
    return value


def _evaluate_runs(
    run_dirs: Sequence[Path],
    pass_indices: Sequence[int],
    planned_passes: int,
    policy: ScoringPolicy | None,
    *,
    declared_status: str | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    manifests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] | None = None
    model_names: list[Any] | None = None
    identity_fields: dict[str, Any] | None = None
    statuses: list[str] = []
    selected_policy = policy
    for run_dir, pass_index in zip(run_dirs, pass_indices, strict=True):
        manifest, run_records = _load_validated_run(run_dir)
        _validate_evaluation_schema(manifest)
        run_planned = manifest.get("planned_passes")
        if run_planned is not None and run_planned != planned_passes:
            raise ValueError(f"planned_passes changed between experiment and run {run_dir}")
        run_repetition = manifest.get("repetition")
        if (
            run_repetition is not None
            and not isinstance(run_repetition, bool)
            and isinstance(run_repetition, int)
            and run_repetition != pass_index
        ):
            raise ValueError(f"run repetition does not match its experiment pass: {run_dir}")
        if (
            expected_fingerprint is not None
            and manifest.get("benchmark_fingerprint") != expected_fingerprint
        ):
            raise ValueError(f"benchmark fingerprint mismatch in run {run_dir}")
        current_identity = {
            key: manifest.get(key)
            for key in (
                "benchmark_fingerprint",
                "preprocess",
                "prompt_hashes",
                "prose_prompt_hash",
                "math_prompt_hash",
                "scoring_policy",
                "models",
                "settings",
                "provider_controls",
                "model_info",
                "model_capabilities",
                "measurement_profile",
                "measurement_schema_version",
                "planned_passes",
                "evaluation_schema_version",
                "scoring_version",
                "layout",
                "split",
                "strict_research",
                "protocol",
                "formula_rendering",
                "task_metrics_version",
                "formula_renderer",
                "prompt_version",
                "repeated_measurement",
                "research",
                "validation_scope",
                "content_type",
                "limit",
                "seed",
                "base_url",
                "timeout",
                "prompt",
                "math_prompt",
                "options",
                "costs",
                "execution_controls",
                "max_retries",
                "concurrency",
                "max_requests",
                "max_spend_usd",
                "cache_dir",
                "warmup",
                "host",
            )
        }
        if selected_policy is None:
            raw_policy = manifest.get("scoring_policy")
            if isinstance(raw_policy, Mapping):
                selected_policy = ScoringPolicy.from_dict(raw_policy)
            elif manifest.get("evaluation_schema_version") == EVALUATION_SCHEMA_VERSION:
                raise ValueError(f"evaluation schema 2 run has no scoring_policy: {run_dir}")
        if identity_fields is None:
            identity_fields = current_identity
            samples = manifest["samples"]
            model_names = list(manifest.get("models", []))
        elif identity_fields != current_identity:
            raise ValueError(f"experiment identity changed between passes: {run_dir}")
        manifests.append(manifest)
        statuses.append(str(manifest.get("status", "incomplete")))
        records.extend({**row, "pass": pass_index} for row in run_records)
    if samples is None:
        return None
    if selected_policy is None:
        return {
            "status": "legacy_policy_unavailable",
            "evaluation": None,
            "legacy_scoring_version": manifests[0].get("scoring_version"),
            "reason": "run has no versioned weighted scoring policy; pass an explicit policy to re-evaluate",
        }
    runs_complete = len(run_dirs) == planned_passes and all(
        status.lower() in _TERMINAL_RUN_STATUSES for status in statuses
    )
    if declared_status is None:
        effective_status = "complete" if runs_complete else "incomplete"
    elif declared_status.lower() in _TERMINAL_RUN_STATUSES and runs_complete:
        effective_status = declared_status
    else:
        effective_status = (
            declared_status
            if declared_status.lower() not in _TERMINAL_RUN_STATUSES
            else "incomplete"
        )
    run_manifest = dict(manifests[0])
    run_manifest["models"] = model_names or []
    run_manifest["evaluation_schema_version"] = EVALUATION_SCHEMA_VERSION
    run_manifest.setdefault("scoring_policy", selected_policy.to_dict())
    run_manifest["execution_runs"] = [
        {
            "pass": pass_index,
            "run_id": str(run_dir),
            "invocations": (
                (manifest.get("execution_summary") or {}).get("invocations", [])
                if isinstance(manifest.get("execution_summary"), Mapping)
                else []
            ),
        }
        for run_dir, pass_index, manifest in zip(run_dirs, pass_indices, manifests, strict=True)
    ]
    return evaluate_run(
        samples,
        records,
        models=model_names,
        policy=selected_policy,
        planned_passes=planned_passes,
        run_status=effective_status,
        manifest=run_manifest,
    )


def _wrap_report(
    path: Path,
    source_type: str,
    conditions: Mapping[str, Any],
    policy: ScoringPolicy | None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "source_type": source_type,
        "policy": policy.to_dict() if policy is not None else None,
        "condition_evaluations": {
            name: value.get("evaluation_id") if isinstance(value, Mapping) else None
            for name, value in conditions.items()
        },
        "benchmark_fingerprint": (manifest or {}).get("benchmark_fingerprint"),
    }
    return {
        "schema_version": REPORT_MODEL_VERSION,
        "report_model_version": REPORT_MODEL_VERSION,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_id": _canonical_hash(identity),
        "source": {"type": source_type, "path": str(path)},
        "policy": policy.to_dict() if policy is not None else None,
        "conditions": dict(conditions),
    }


def _legacy_unavailable(manifest: Mapping[str, Any], planned_passes: int) -> dict[str, Any]:
    return {
        "status": "legacy_policy_unavailable",
        "evaluation": None,
        "legacy_scoring_version": manifest.get("scoring_version"),
        "planned_passes": planned_passes,
        "reason": "run has no versioned weighted scoring policy; pass an explicit policy to re-evaluate",
    }


def _single_run_records(
    records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    repetition = manifest.get("repetition")
    result: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        if not any(key in copied for key in ("pass", "pass_index", "repetition")):
            if isinstance(repetition, int) and not isinstance(repetition, bool) and repetition > 0:
                copied["pass"] = repetition
        result.append(copied)
    return result


def build_report(
    source: str | Path,
    policy: ScoringPolicy | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a renderer-neutral report from a validated run, experiment, or suite.

    Each named suite condition is evaluated independently. An
    ``experiment.json`` groups physical runs into passes for one condition.
    A direct run is treated as one pass unless its manifest says otherwise.
    """
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        if (path / "suite.json").is_file():
            path = path / "suite.json"
        elif (path / "experiment.json").is_file():
            path = path / "experiment.json"
        else:
            manifest, records = _load_validated_run(path)
            _validate_evaluation_schema(manifest)
            selected_policy = _policy_for_source(policy, manifest)
            planned = manifest.get("planned_passes", 1)
            if isinstance(planned, bool) or not isinstance(planned, int) or planned < 1:
                raise ValueError("run planned_passes must be a positive integer")
            if selected_policy is None:
                return _wrap_report(
                    path,
                    "run",
                    {"baseline": _legacy_unavailable(manifest, planned)},
                    None,
                    manifest,
                )
            evaluation = evaluate_run(
                manifest["samples"],
                _single_run_records(records, manifest),
                models=manifest.get("models"),
                policy=selected_policy,
                planned_passes=planned,
                run_status=str(manifest.get("status", "complete")),
                manifest=manifest,
            )
            return _wrap_report(path, "run", {"baseline": evaluation}, selected_policy, manifest)

    if path.name == "experiment.json":
        manifest = _load_json_object(path)
        _validate_evaluation_schema(manifest)
        selected_policy = _policy_for_source(policy, manifest, allow_missing_versioned_policy=True)
        planned = manifest.get("planned_passes", 1)
        if isinstance(planned, bool) or not isinstance(planned, int) or planned < 1:
            raise ValueError("experiment planned_passes must be a positive integer")
        entries = manifest.get("runs", [])
        if not isinstance(entries, list):
            raise ValueError("experiment runs must be a list")
        run_dirs: list[Path] = []
        pass_indices: list[int] = []
        for index, item in enumerate(entries, start=1):
            if isinstance(item, str):
                raw_path, pass_index = item, index
            elif isinstance(item, Mapping):
                raw_path = item.get("run_dir", item.get("path"))
                pass_index = item.get("pass", item.get("repetition", index))
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError("experiment run entries need a run_dir or path")
            else:
                raise ValueError("experiment run entries must be paths or mappings")
            run_dir = Path(raw_path)
            if not run_dir.is_absolute():
                run_dir = (path.parent / run_dir).resolve()
            if (
                isinstance(pass_index, bool)
                or not isinstance(pass_index, int)
                or not 1 <= pass_index <= planned
            ):
                raise ValueError("experiment pass identities must be in planned_passes")
            if pass_index in pass_indices:
                raise ValueError(f"duplicate experiment pass identity: {pass_index}")
            run_dirs.append(run_dir)
            pass_indices.append(pass_index)
        evaluation = _evaluate_runs(
            run_dirs,
            pass_indices,
            planned,
            selected_policy,
            declared_status=manifest.get("status"),
            expected_fingerprint=manifest.get("benchmark_fingerprint"),
        )
        condition = str(manifest.get("condition", "baseline"))
        if selected_policy is None and evaluation and isinstance(evaluation.get("policy"), Mapping):
            selected_policy = ScoringPolicy.from_dict(evaluation["policy"])
        conditions = (
            {condition: evaluation}
            if evaluation is not None
            else {
                condition: {"status": "not_started", "evaluation": None, "planned_passes": planned}
            }
        )
        return _wrap_report(path, "experiment", conditions, selected_policy, manifest)

    if path.name == "suite.json":
        from .suite import _read_manifest

        manifest = _read_manifest(path)
        _validate_evaluation_schema(manifest)
        selected_policy = _policy_for_source(policy, manifest, allow_missing_versioned_policy=True)
        report_policy = selected_policy
        inferred_policies: list[ScoringPolicy] = []
        run_entries = manifest.get("runs", {})
        if not isinstance(run_entries, Mapping):
            raise ValueError("suite runs must be an object")
        conditions: dict[str, Any] = {}
        for condition in manifest.get("conditions", []):
            if not isinstance(condition, Mapping) or not isinstance(condition.get("name"), str):
                continue
            name = condition["name"]
            condition_policy = report_policy
            if condition_policy is None:
                condition_policy = _policy_for_source(
                    None,
                    {"scoring_policy": condition.get("scoring_policy")},
                    allow_missing_versioned_policy=True,
                )
            planned = condition.get("repeats", 1)
            if isinstance(planned, bool) or not isinstance(planned, int) or planned < 1:
                raise ValueError(f"suite condition {name!r} has invalid repeats")
            entries = [
                item
                for item in run_entries.values()
                if isinstance(item, Mapping)
                and item.get("condition") == name
                and item.get("run_dir")
            ]
            entries.sort(key=lambda item: int(item.get("repetition", 0)))
            run_dirs: list[Path] = []
            pass_indices: list[int] = []
            for item in entries:
                run_dir = Path(item["run_dir"]).expanduser()
                if not run_dir.is_absolute():
                    run_dir = (path.parent / run_dir).resolve()
                run_dirs.append(run_dir)
                pass_indices.append(int(item["repetition"]))
            evaluation = _evaluate_runs(
                run_dirs,
                pass_indices,
                planned,
                condition_policy,
                expected_fingerprint=condition.get("benchmark_fingerprint"),
            )
            conditions[name] = (
                evaluation
                if evaluation is not None
                else {"status": "not_started", "evaluation": None, "planned_passes": planned}
            )
            if evaluation and isinstance(evaluation.get("policy"), Mapping):
                inferred_policies.append(ScoringPolicy.from_dict(evaluation["policy"]))
        if (
            report_policy is None
            and inferred_policies
            and all(candidate == inferred_policies[0] for candidate in inferred_policies)
        ):
            report_policy = inferred_policies[0]
        return _wrap_report(path, "suite", conditions, report_policy, manifest)

    manifest, records = _load_validated_run(path)
    _validate_evaluation_schema(manifest)
    selected_policy = _policy_for_source(policy, manifest)
    planned = manifest.get("planned_passes", 1)
    if isinstance(planned, bool) or not isinstance(planned, int) or planned < 1:
        raise ValueError("run planned_passes must be a positive integer")
    if selected_policy is None:
        return _wrap_report(
            path, "run", {"baseline": _legacy_unavailable(manifest, planned)}, None, manifest
        )
    evaluation = evaluate_run(
        manifest["samples"],
        _single_run_records(records, manifest),
        models=manifest.get("models"),
        policy=selected_policy,
        planned_passes=planned,
        run_status=str(manifest.get("status", "complete")),
        manifest=manifest,
    )
    return _wrap_report(path, "run", {"baseline": evaluation}, selected_policy, manifest)
