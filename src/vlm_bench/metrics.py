"""Text normalization, transcription scores, and per-model summaries.

The whitespace normalization and Levenshtein scoring approach is adapted from
``benchmark/metrics.py`` in the `vlm-ocr-research` project (Apache-2.0):
https://github.com/PyaesoneP/vlm-ocr-research/blob/fd4bd0ae44db0f57f7dcb0e301a0a718d3e6159f/benchmark/metrics.py

This standalone implementation additionally returns an edit-operation
breakdown and corpus-weighted summaries.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Sequence


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace to one space and trim the ends.

    Letter case, punctuation, and Unicode code points are kept unchanged.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(text.split())


def _distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Return Levenshtein distance using the upstream rolling-row method."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left):
        current = [i + 1]
        for j, right_item in enumerate(right):
            current.append(
                min(
                    current[-1] + 1,  # insert into left to reach right
                    previous[j + 1] + 1,  # delete from left
                    previous[j] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _edit_counts(reference: Sequence[Any], prediction: Sequence[Any]) -> tuple[int, int, int]:
    """Return substitutions, deletions, and insertions for a minimum edit path.

    Ties are resolved by preferring a diagonal operation, then a reference
    deletion, then a prediction insertion. This makes the breakdown stable for
    ambiguous repeated-character alignments while preserving minimum distance.
    """
    # Each state is (distance, substitutions, deletions, insertions). Keeping
    # only the previous/current rows bounds memory for long model responses.
    previous = [(j, 0, 0, j) for j in range(len(prediction) + 1)]
    for i, reference_item in enumerate(reference, start=1):
        current = [(i, 0, i, 0)]
        for j, prediction_item in enumerate(prediction, start=1):
            diagonal = previous[j - 1]
            if reference_item == prediction_item:
                diagonal_candidate = diagonal
            else:
                diagonal_candidate = (
                    diagonal[0] + 1,
                    diagonal[1] + 1,
                    diagonal[2],
                    diagonal[3],
                )
            deletion = previous[j]
            deletion_candidate = (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3])
            insertion = current[j - 1]
            insertion_candidate = (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1)
            # min() preserves candidate order for equal distances: diagonal,
            # then deletion, then insertion.
            current.append(
                min(
                    (diagonal_candidate, deletion_candidate, insertion_candidate),
                    key=lambda candidate: candidate[0],
                )
            )
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions


def score(prediction: str, reference: str) -> dict[str, Any]:
    """Score one transcription, preserving case and punctuation.

    CER and WER use whitespace-normalized text. ``raw_cer`` uses the strings as
    supplied, including their original whitespace. Empty normalized references
    are rejected because the error-rate denominator would be undefined.
    """
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise TypeError("prediction and reference must be strings")
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)
    if not normalized_reference:
        raise ValueError("reference must contain at least one non-whitespace character")

    substitutions, deletions, insertions = _edit_counts(normalized_reference, normalized_prediction)
    char_edits = substitutions + deletions + insertions
    reference_words = normalized_reference.split()
    prediction_words = normalized_prediction.split()
    word_edits = _distance(reference_words, prediction_words)

    raw_reference_chars = len(reference)
    return {
        "cer": char_edits / len(normalized_reference),
        "wer": word_edits / len(reference_words),
        "raw_cer": _distance(reference, prediction) / raw_reference_chars,
        "exact_match": normalized_prediction == normalized_reference,
        "char_substitutions": substitutions,
        "char_deletions": deletions,
        "char_insertions": insertions,
        "char_edits": char_edits,
        "word_edits": word_edits,
        "reference_chars": len(normalized_reference),
        "reference_words": len(reference_words),
    }


def _number(value: Any) -> float | None:
    """Accept finite non-negative timing values, ignoring malformed entries."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _metrics_for(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("status") != "success":
        return None
    metrics = record.get("metrics")
    required = {
        "cer",
        "wer",
        "char_edits",
        "word_edits",
        "reference_chars",
        "reference_words",
        "exact_match",
    }
    if isinstance(metrics, dict) and required.issubset(metrics):
        return metrics
    prediction = record.get("prediction", "")
    reference = record.get("reference", "")
    return score(prediction, reference)


def summarize(
    records: list[dict[str, Any]],
    expected_samples: int | Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    force_incomplete: bool = False,
    warmups: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate sample rows per model and rank complete runs by corpus CER.

    Corpus CER/WER divide total edit counts by total reference characters/words,
    so longer samples contribute in proportion to their reference length.
    Latency statistics use available non-negative per-sample timings; throughput
    is timed samples divided by their summed inference latency. Load time is
    totaled separately from inference time. Warmups do not contribute to
    accuracy or measured latency; their count and load time are reported apart.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each result record must be a dictionary")
        model = record.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("each result record must include a non-empty model name")
        grouped[model].append(record)

    warmups_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for warmup in warmups or []:
        if not isinstance(warmup, dict):
            raise TypeError("each warmup record must be a dictionary")
        model = warmup.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("each warmup record must include a non-empty model name")
        warmups_by_model[model].append(warmup)
        grouped.setdefault(model, [])

    if models is None:
        expected_models = None
    else:
        expected_models = set()
        for model in models:
            if not isinstance(model, str) or not model:
                raise ValueError("expected model names must be non-empty strings")
            expected_models.add(model)
    if expected_models is not None:
        for model in expected_models:
            grouped.setdefault(model, [])

    if isinstance(expected_samples, bool):
        raise TypeError("expected_samples must be a non-negative count or a sequence of IDs")
    if isinstance(expected_samples, int):
        if expected_samples < 0:
            raise ValueError("expected sample count cannot be negative")
        expected_sample_ids = None
        expected_sample_count: int | None = expected_samples
    elif expected_samples is None:
        expected_sample_ids = None
        expected_sample_count = None
    else:
        expected_sample_ids = {str(sample_id) for sample_id in expected_samples}
        expected_sample_count = len(expected_sample_ids)

    summaries: list[dict[str, Any]] = []
    for model, model_records in grouped.items():
        metric_rows: list[dict[str, Any]] = []
        success_count = failed_count = empty_count = truncated_count = 0
        latencies: list[float] = []
        load_durations: list[float] = []
        for record in model_records:
            if record.get("status") == "success":
                success_count += 1
                prediction = record.get("prediction", "")
                if not isinstance(prediction, str):
                    prediction = str(prediction)
                if not normalize_text(prediction):
                    empty_count += 1
                metrics = _metrics_for(record)
                if metrics is not None:
                    metric_rows.append(metrics)
            else:
                failed_count += 1

            if record.get("truncated"):
                truncated_count += 1
            latency = _number(record.get("latency_seconds"))
            if latency is not None:
                latencies.append(latency)
            load_duration = _number(record.get("load_duration_seconds"))
            if load_duration is not None:
                load_durations.append(load_duration)

        warmup_load_durations: list[float] = []
        for warmup in warmups_by_model.get(model, []):
            response = warmup.get("response")
            duration_ns = response.get("load_duration") if isinstance(response, dict) else None
            if duration_ns is None:
                duration_seconds = _number(warmup.get("load_duration_seconds"))
            else:
                parsed_ns = _number(duration_ns)
                duration_seconds = parsed_ns / 1_000_000_000 if parsed_ns is not None else None
            if duration_seconds is not None:
                warmup_load_durations.append(duration_seconds)

        reference_chars = sum(int(row["reference_chars"]) for row in metric_rows)
        reference_words = sum(int(row["reference_words"]) for row in metric_rows)
        char_edits = sum(int(row["char_edits"]) for row in metric_rows)
        word_edits = sum(int(row["word_edits"]) for row in metric_rows)
        scored_count = len(metric_rows)
        cer = char_edits / reference_chars if reference_chars else None
        wer = word_edits / reference_words if reference_words else None
        mean_cer = (
            statistics.fmean(float(row["cer"]) for row in metric_rows) if metric_rows else None
        )
        mean_wer = (
            statistics.fmean(float(row["wer"]) for row in metric_rows) if metric_rows else None
        )
        exact_match_rate = (
            sum(bool(row["exact_match"]) for row in metric_rows) / scored_count
            if scored_count
            else None
        )
        total_latency = sum(latencies)
        actual_sample_ids = {str(record.get("sample_id")) for record in model_records}
        if expected_sample_ids is not None:
            missing_sample_count = len(expected_sample_ids - actual_sample_ids)
            sample_coverage_complete = (
                missing_sample_count == 0 and len(model_records) == expected_sample_count
            )
        elif expected_sample_count is not None:
            missing_sample_count = max(0, expected_sample_count - len(model_records))
            sample_coverage_complete = len(model_records) == expected_sample_count
        else:
            missing_sample_count = None
            sample_coverage_complete = True
        unexpected_model = expected_models is not None and model not in expected_models
        complete = (
            failed_count == 0
            and sample_coverage_complete
            and not unexpected_model
            and not force_incomplete
        )

        summaries.append(
            {
                "rank": None,
                "model": model,
                "complete": complete,
                "sample_count": len(model_records),
                "expected_sample_count": expected_sample_count,
                "missing_sample_count": missing_sample_count,
                "sample_coverage_complete": sample_coverage_complete,
                "expected_model": not unexpected_model,
                "scored_sample_count": scored_count,
                "success_count": success_count,
                "failed_count": failed_count,
                "failure_rate": failed_count / len(model_records) if model_records else None,
                "empty_count": empty_count,
                "truncated_count": truncated_count,
                "cer": cer,
                "wer": wer,
                "mean_sample_cer": mean_cer,
                "mean_sample_wer": mean_wer,
                "exact_match_rate": exact_match_rate,
                "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
                "median_latency_seconds": statistics.median(latencies) if latencies else None,
                "p95_latency_seconds": _percentile(latencies, 0.95),
                "total_latency_seconds": total_latency if latencies else None,
                "samples_per_minute": (
                    len(latencies) * 60 / total_latency if total_latency > 0 else None
                ),
                "load_duration_seconds": sum(load_durations) if load_durations else None,
                "load_event_count": len(load_durations),
                "warmup_count": len(warmups_by_model.get(model, [])),
                "warmup_load_duration_seconds": (
                    sum(warmup_load_durations) if warmup_load_durations else None
                ),
            }
        )

    complete = [row for row in summaries if row["complete"]]
    incomplete = [row for row in summaries if not row["complete"]]
    complete.sort(
        key=lambda row: (
            row["cer"] if row["cer"] is not None else math.inf,
            row["wer"] if row["wer"] is not None else math.inf,
            row["model"],
        )
    )
    incomplete.sort(key=lambda row: row["model"])
    for rank, row in enumerate(complete, start=1):
        row["rank"] = rank
    return complete + incomplete
