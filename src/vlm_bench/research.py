"""Research-oriented comparisons for handwriting benchmark runs.

The regular export summarizes a model's available predictions. This module
adds task-aware summaries whose rankings only compare models that completed the
same verified test samples, plus paired comparisons and cost projections based
only on values the experimenter supplied.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import unicodedata
from collections import defaultdict
from typing import Any

from .metrics import normalize_text, score

_EQUATION_TYPES = {"equation", "math", "mathematics", "latex", "formula"}
_TRACKS = ("prose", "word", "equation")
_TEST_SPLITS = {"test", "testing", "eval", "evaluation", "heldout", "held-out"}
_NON_TEST_SPLITS = {"train", "training", "validation", "valid", "dev"}
_UNVERIFIED_STATES = {
    "unverified",
    "not verified",
    "not_verified",
    "uncertain",
    "unresolved",
    "needs review",
    "needs_review",
    "review needed",
    "review_needed",
    "excluded",
    "pending",
}


def normalize_equation(text: str) -> str:
    """Normalize equation transcription for literal string comparison.

    Unicode is normalized to NFC and whitespace is collapsed. Symbols,
    operators, grouping, and LaTeX commands are otherwise preserved. This is
    transcription scoring; it does not test mathematical equivalence.
    """
    if not isinstance(text, str):
        raise TypeError("equation text must be a string")
    return normalize_text(unicodedata.normalize("NFC", text))


def score_equation(prediction: str, reference: str) -> dict[str, Any]:
    """Score literal equation transcription after conservative normalization."""
    normalized_prediction = normalize_equation(prediction)
    normalized_reference = normalize_equation(reference)
    if not normalized_reference:
        raise ValueError("equation reference must contain at least one non-whitespace character")
    metrics = score(normalized_prediction, normalized_reference)
    return {
        **metrics,
        "equation_cer": metrics["cer"],
        "normalized_exact_match": metrics["exact_match"],
        "normalization": (
            "Unicode NFC; trim and collapse whitespace; preserve symbols and syntax; "
            "literal transcription score, not mathematical equivalence"
        ),
    }


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "verified", "approved"}:
            return True
        if lowered in {"false", "no", "unverified", "uncertain", "unresolved", "excluded"}:
            return False
    return None


def _sample_kind(sample: dict[str, Any]) -> str:
    value = next(
        (
            sample.get(key)
            for key in ("content_type", "task", "track")
            if sample.get(key) not in (None, "")
        ),
        "prose",
    )
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in _EQUATION_TYPES or "equation" in normalized:
        return "equation"
    sample_type = str(sample.get("sample_type", "")).strip().lower()
    if normalized == "word" or (normalized in {"", "none", "prose"} and sample_type == "word"):
        return "word"
    return "prose"


def _sample_document(sample: dict[str, Any], sample_id: str) -> str:
    value = sample.get("document_id", sample.get("source_document", sample.get("document")))
    return str(value) if value not in (None, "") else sample_id


def _has_document_metadata(sample: dict[str, Any]) -> bool:
    return any(
        sample.get(key) not in (None, "") for key in ("document_id", "source_document", "document")
    )


def _sample_exclusion(sample: dict[str, Any], has_split_labels: bool) -> str | None:
    if sample.get("excluded") is True or _as_bool(sample.get("eligible")) is False:
        return "explicitly_excluded"
    verified = _as_bool(sample.get("verified"))
    if verified is False:
        return "reference_unverified"
    state = sample.get("verification_status", sample.get("reference_status"))
    if isinstance(state, str) and state.strip().lower() in _UNVERIFIED_STATES:
        return "reference_unverified"
    split = sample.get("split")
    if has_split_labels and not (isinstance(split, str) and split.strip()):
        return "missing_split"
    if isinstance(split, str) and split.strip():
        normalized_split = split.strip().lower().replace("_", "-")
        if normalized_split in _NON_TEST_SPLITS:
            return "not_test_split"
        if has_split_labels and normalized_split not in _TEST_SPLITS:
            return "not_test_split"
    return None


def _model_name(model: Any) -> str | None:
    if isinstance(model, str) and model:
        return model
    if isinstance(model, dict):
        for key in ("selector", "name", "model", "id"):
            value = model.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _configured_models(manifest: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    configured = manifest.get("models", [])
    if isinstance(configured, list):
        for item in configured:
            name = _model_name(item)
            if name and name not in result:
                result.append(name)
    for row in records:
        name = _model_name(row.get("model"))
        if name and name not in result:
            result.append(name)
    return result


def _model_config(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    configured = manifest.get("models", [])
    if isinstance(configured, list):
        for item in configured:
            if isinstance(item, dict) and _model_name(item) == name:
                return item
    capabilities = manifest.get("model_capabilities", {})
    if isinstance(capabilities, dict) and isinstance(capabilities.get(name), dict):
        return capabilities[name]
    return {}


def _unsupported_reason(manifest: dict[str, Any], name: str, track: str) -> str | None:
    config = _model_config(manifest, name)
    supported = config.get("supported_content_types", config.get("supported_tasks"))
    unsupported = config.get("unsupported_content_types", config.get("unsupported_tasks", []))
    if isinstance(supported, (list, tuple, set)):
        allowed = {str(value).strip().lower() for value in supported}
        if track not in allowed and not (
            track == "equation" and allowed.intersection(_EQUATION_TYPES)
        ):
            return "model_does_not_support_task"
    if isinstance(unsupported, (list, tuple, set)):
        denied = {str(value).strip().lower() for value in unsupported}
        if track in denied or (track == "equation" and denied.intersection(_EQUATION_TYPES)):
            return "model_does_not_support_task"
    supports_equations = config.get("supports_equations")
    if track == "equation" and supports_equations is False:
        return "model_does_not_support_task"
    # The standard TrOCR handwritten checkpoint recognizes prose lines; it is
    # intentionally shown as unsupported on the equation track.
    lowered = name.lower()
    has_declared_equation_support = (
        (isinstance(supported, (list, tuple, set)) and bool(supported))
        or (isinstance(unsupported, (list, tuple, set)) and bool(unsupported))
        or isinstance(supports_equations, bool)
    )
    if (
        track == "equation"
        and (lowered.startswith("trocr:") or "trocr" in lowered)
        and not has_declared_equation_support
    ):
        return "standard_trocr_is_prose_only"
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _warmup_load_info(
    model: str,
    track: str,
    samples: dict[str, dict[str, Any]],
    warmups: list[dict[str, Any]],
) -> tuple[int, list[float]]:
    relevant: list[float] = []
    count = 0
    run_tracks = {_sample_kind(sample) for sample in samples.values()}
    for warmup in warmups:
        if warmup.get("model") != model:
            continue
        sample_id = warmup.get("sample_id")
        if sample_id is None:
            if run_tracks != {track}:
                continue
        elif str(sample_id) not in samples or _sample_kind(samples[str(sample_id)]) != track:
            continue
        count += 1
        duration = _finite_number(warmup.get("load_duration_seconds"))
        if duration is None:
            response = warmup.get("response")
            duration_ns = _finite_number(
                response.get("load_duration") if isinstance(response, dict) else None
            )
            if duration_ns is not None:
                duration = duration_ns / 1_000_000_000
        if duration is not None:
            relevant.append(duration)
    return count, relevant


def _percentile(values: list[float], quantile: float) -> float | None:
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


def _bootstrap_seed(seed: int, track: str, left: str, right: str) -> int:
    material = f"{seed}\0{track}\0{left}\0{right}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _ratio(rows: list[dict[str, Any]], key: str) -> float | None:
    denominator = sum(int(row["metrics"]["reference_chars"]) for row in rows)
    if denominator <= 0:
        return None
    edits = sum(int(row["metrics"]["char_edits"]) for row in rows)
    return edits / denominator


def _paired_bootstrap(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    seed: int,
    track: str,
    left_model: str,
    right_model: str,
    replicates: int,
) -> dict[str, Any]:
    left_by_id = {row["sample_id"]: row for row in left_rows}
    right_by_id = {row["sample_id"]: row for row in right_rows}
    paired_ids = sorted(left_by_id.keys() & right_by_id.keys())
    if not paired_ids:
        return {"unit": None, "groups": 0, "replicates": replicates, "ci95": None}

    paired_left = [left_by_id[sample_id] for sample_id in paired_ids]
    document_ids = {row["sample_id"]: row["document_id"] for row in paired_left}
    by_document: dict[str, list[str]] = defaultdict(list)
    for sample_id in paired_ids:
        by_document[document_ids[sample_id]].append(sample_id)
    groups = sorted(by_document)
    rng = random.Random(_bootstrap_seed(seed, track, left_model, right_model))
    deltas: list[float] = []
    for _ in range(replicates):
        selected_groups = [rng.choice(groups) for _ in groups]
        selected_ids = [sample_id for group in selected_groups for sample_id in by_document[group]]
        left_sample = [left_by_id[sample_id] for sample_id in selected_ids]
        right_sample = [right_by_id[sample_id] for sample_id in selected_ids]
        left_cer = _ratio(left_sample, "cer")
        right_cer = _ratio(right_sample, "cer")
        if left_cer is not None and right_cer is not None:
            deltas.append(right_cer - left_cer)
    return {
        "unit": (
            "document"
            if any(row.get("document_metadata_present") for row in paired_left)
            else "sample"
        ),
        "groups": len(groups),
        "replicates": replicates,
        "ci95": [
            _percentile(deltas, 0.025),
            _percentile(deltas, 0.975),
        ]
        if deltas
        else None,
    }


def _track_report(
    track: str,
    samples: dict[str, dict[str, Any]],
    excluded: list[dict[str, str]],
    models: list[str],
    manifest: dict[str, Any],
    records_by_model: dict[str, dict[str, dict[str, Any]]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    eligible = sorted(
        sample_id for sample_id, sample in samples.items() if _sample_kind(sample) == track
    )
    model_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    for model in models:
        reason = _unsupported_reason(manifest, model, track)
        model_records = records_by_model.get(model, {})
        scored: list[dict[str, Any]] = []
        failed_ids: list[str] = []
        unsupported_ids: list[str] = []
        latencies: list[float] = []
        inference_latencies: list[float] = []
        load_durations: list[float] = []
        truncated_count = 0
        empty_count = 0
        for sample_id in eligible:
            record = model_records.get(sample_id)
            if record is None:
                continue
            if record.get("truncated") is True:
                truncated_count += 1
            latency = _finite_number(record.get("latency_seconds"))
            if latency is not None:
                latencies.append(latency)
            inference_latency = _finite_number(record.get("inference_latency_seconds"))
            if inference_latency is not None:
                inference_latencies.append(inference_latency)
            load_duration = _finite_number(record.get("load_duration_seconds"))
            if load_duration is not None:
                load_durations.append(load_duration)
            if record.get("status") == "unsupported":
                unsupported_ids.append(sample_id)
                continue
            if record.get("status") != "success":
                failed_ids.append(sample_id)
                continue
            prediction = record.get("prediction")
            reference = record.get("reference", samples[sample_id].get("reference", ""))
            if (
                not isinstance(prediction, str)
                or not isinstance(reference, str)
                or not reference.strip()
            ):
                failed_ids.append(sample_id)
                continue
            if not normalize_text(prediction):
                empty_count += 1
            metrics = (
                score_equation(prediction, reference)
                if track == "equation"
                else score(prediction, reference)
            )
            scored.append(
                {
                    "sample_id": sample_id,
                    "document_id": _sample_document(samples[sample_id], sample_id),
                    "document_metadata_present": _has_document_metadata(samples[sample_id]),
                    "metrics": metrics,
                }
            )
        model_rows[model] = scored
        scored_count = len(scored)
        char_edits = sum(row["metrics"]["char_edits"] for row in scored)
        reference_chars = sum(row["metrics"]["reference_chars"] for row in scored)
        reference_words = sum(row["metrics"]["reference_words"] for row in scored)
        word_edits = sum(row["metrics"]["word_edits"] for row in scored)
        cer = char_edits / reference_chars if reference_chars else None
        wer = word_edits / reference_words if reference_words else None
        exact = (
            sum(bool(row["metrics"]["exact_match"]) for row in scored) / scored_count
            if scored_count
            else None
        )
        missing_ids = sorted(
            set(eligible)
            - {row["sample_id"] for row in scored}
            - set(failed_ids)
            - set(unsupported_ids)
        )
        if reason is None and eligible and len(set(unsupported_ids)) == len(eligible):
            reason = "provider_marked_task_unsupported"
        if reason:
            status = "unsupported"
        elif not model_records:
            status = "not_run"
        elif not eligible:
            status = "no_eligible_samples"
        elif not missing_ids and not failed_ids and scored_count == len(eligible):
            status = "complete"
        else:
            status = "incomplete"
        warmups = manifest.get("warmups", [])
        if not isinstance(warmups, list):
            warmups = []
        else:
            warmups = [item for item in warmups if isinstance(item, dict)]
        warmup_count, warmup_load_durations = _warmup_load_info(model, track, samples, warmups)
        warmup_load_total = sum(warmup_load_durations) if warmup_load_durations else None
        if warmup_load_total is not None:
            cold_load_duration = warmup_load_total
            cold_load_source = "warmup"
        elif load_durations:
            cold_load_duration = sum(load_durations)
            cold_load_source = "sample_records"
        else:
            cold_load_duration = None
            cold_load_source = None
        failure_denominator = scored_count + len(set(failed_ids))
        total_latency = sum(latencies) if latencies else None
        summaries.append(
            {
                "model": model,
                "actual_model": next(
                    (
                        model_records[sample_id].get("actual_model")
                        for sample_id in eligible
                        if sample_id in model_records
                        and model_records[sample_id].get("actual_model")
                    ),
                    None,
                ),
                "status": status,
                "unsupported_reason": reason,
                "unsupported_sample_ids": sorted(set(unsupported_ids)),
                "unsupported_sample_count": len(set(unsupported_ids)),
                "rank": None,
                "eligible_sample_count": len(eligible),
                "scored_sample_count": scored_count,
                "failed_sample_count": len(set(failed_ids)),
                "failure_rate": (
                    len(set(failed_ids)) / failure_denominator if failure_denominator else None
                ),
                "missing_sample_count": len(missing_ids),
                "missing_sample_ids": missing_ids,
                "failed_sample_ids": sorted(set(failed_ids)),
                "sample_coverage": scored_count / len(eligible) if eligible else None,
                "success_count": scored_count,
                "empty_count": empty_count,
                "cer": cer,
                "wer": wer,
                "exact_match_rate": exact,
                "reference_chars": reference_chars,
                "char_edits": char_edits,
                "mean_latency_seconds": (sum(latencies) / len(latencies) if latencies else None),
                "median_latency_seconds": (_percentile(latencies, 0.5) if latencies else None),
                "p95_latency_seconds": _percentile(latencies, 0.95),
                "total_latency_seconds": total_latency,
                "samples_per_minute": (
                    scored_count * 60 / total_latency
                    if total_latency is not None and total_latency > 0
                    else None
                ),
                "mean_inference_latency_seconds": (
                    sum(inference_latencies) / len(inference_latencies)
                    if inference_latencies
                    else None
                ),
                "median_inference_latency_seconds": (
                    _percentile(inference_latencies, 0.5) if inference_latencies else None
                ),
                "p95_inference_latency_seconds": _percentile(inference_latencies, 0.95),
                "total_inference_latency_seconds": (
                    sum(inference_latencies) if inference_latencies else None
                ),
                "inference_latency_count": len(inference_latencies),
                "load_duration_seconds": (sum(load_durations) if load_durations else None),
                "load_event_count": len(load_durations),
                "cold_load_duration_seconds": cold_load_duration,
                "cold_load_source": cold_load_source,
                "warmup_count": warmup_count,
                "warmup_load_duration_seconds": warmup_load_total,
                "warmup_load_event_count": len(warmup_load_durations),
                "truncated_count": truncated_count,
            }
        )

    rankable = [row for row in summaries if row["status"] == "complete" and row["cer"] is not None]
    rankable.sort(
        key=lambda row: (
            row["cer"],
            row["wer"] if row["wer"] is not None else math.inf,
            row["model"],
        )
    )
    for rank, row in enumerate(rankable, start=1):
        row["rank"] = rank

    common_ranked = [
        {"rank": row["rank"], "model": row["model"], "cer": row["cer"], "wer": row["wer"]}
        for row in rankable
    ]
    summary_by_model = {row["model"]: row for row in summaries}
    pairs: list[dict[str, Any]] = []
    for left_model, right_model in itertools.combinations(models, 2):
        left_reason = (
            _unsupported_reason(manifest, left_model, track)
            or (summary_by_model[left_model]["unsupported_reason"])
        )
        right_reason = (
            _unsupported_reason(manifest, right_model, track)
            or (summary_by_model[right_model]["unsupported_reason"])
        )
        if left_reason or right_reason:
            pairs.append(
                {
                    "left_model": left_model,
                    "right_model": right_model,
                    "status": "unsupported",
                    "unsupported_models": [
                        model
                        for model, reason in (
                            (left_model, left_reason),
                            (right_model, right_reason),
                        )
                        if reason
                    ],
                    "paired_sample_count": 0,
                    "corpus_cer_difference_right_minus_left": None,
                    "bootstrap": {
                        "unit": None,
                        "groups": 0,
                        "replicates": bootstrap_replicates,
                        "ci95": None,
                    },
                }
            )
            continue
        left_by_id = {row["sample_id"]: row for row in model_rows[left_model]}
        right_by_id = {row["sample_id"]: row for row in model_rows[right_model]}
        paired_ids = sorted(left_by_id.keys() & right_by_id.keys())
        left_rows = [left_by_id[sample_id] for sample_id in paired_ids]
        right_rows = [right_by_id[sample_id] for sample_id in paired_ids]
        left_cer = _ratio(left_rows, "cer")
        right_cer = _ratio(right_rows, "cer")
        bootstrap = _paired_bootstrap(
            model_rows[left_model],
            model_rows[right_model],
            bootstrap_seed,
            track,
            left_model,
            right_model,
            bootstrap_replicates,
        )
        pairs.append(
            {
                "left_model": left_model,
                "right_model": right_model,
                "status": "available" if paired_ids else "no_paired_predictions",
                "paired_sample_count": len(paired_ids),
                "left_corpus_cer": left_cer,
                "right_corpus_cer": right_cer,
                "corpus_cer_difference_right_minus_left": (
                    right_cer - left_cer if right_cer is not None and left_cer is not None else None
                ),
                "bootstrap": bootstrap,
            }
        )
    return {
        "track": track,
        "metric_interpretation": (
            "Literal equation transcription CER and normalized exact match; not mathematical equivalence."
            if track == "equation"
            else "Single-word transcription; CER and exact match are primary, while WER is a coarse word-level diagnostic."
            if track == "word"
            else "Whitespace-normalized prose CER/WER; case and punctuation are preserved."
        ),
        "eligible_sample_ids": eligible,
        "eligible_sample_count": len(eligible),
        "excluded_samples": [entry for entry in excluded if entry["track"] == track],
        "model_results": summaries,
        "common_eligible_ranking": common_ranked,
        "ranking_status": "available" if rankable else "no_complete_supported_models",
        "paired_differences": pairs,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "grouping": "source document when available; otherwise sample",
        },
    }


def _cost_configuration(manifest: dict[str, Any]) -> dict[str, Any]:
    research = manifest.get("research")
    if isinstance(research, dict) and isinstance(research.get("costs"), dict):
        return research["costs"]
    costs = manifest.get("costs")
    if isinstance(costs, dict):
        return costs
    return {}


def _usage_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _api_observed_unit_cost(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[float | None, dict[str, Any]]:
    direct = _finite_number(config.get("cost_per_sample_usd"))
    selector = config.get("model")
    if direct is not None:
        return direct, {
            "source": "explicit_cost_per_sample",
            "model": selector,
            "usage_record_count": 0,
            "estimated_observed_cost_usd": None,
            "mean_cost_per_usage_record_usd": None,
        }

    input_rate = _finite_number(config.get("input_per_million_usd"))
    output_rate = _finite_number(config.get("output_per_million_usd"))
    cached_rate = _finite_number(config.get("cached_input_per_million_usd"))
    if selector is None or input_rate is None or output_rate is None:
        return None, {
            "source": "unknown_missing_cost_or_usage_inputs",
            "model": selector,
            "usage_record_count": 0,
            "estimated_observed_cost_usd": None,
            "mean_cost_per_usage_record_usd": None,
        }

    usage_record_count = 0
    unpriced_attempt_count = 0
    input_tokens_total = output_tokens_total = cached_tokens_total = 0
    observed_total = 0.0
    for row in records:
        if row.get("model") != selector or row.get("status") == "unsupported":
            continue
        usage = row.get("usage")
        if not isinstance(usage, dict):
            unpriced_attempt_count += 1
            continue
        input_tokens = _usage_count(usage.get("input_tokens"))
        output_tokens = _usage_count(usage.get("output_tokens"))
        details = usage.get("input_tokens_details", {})
        cached_tokens = _usage_count(
            details.get("cached_tokens") if isinstance(details, dict) else None
        )
        if input_tokens is None or output_tokens is None:
            unpriced_attempt_count += 1
            continue
        cached_tokens = cached_tokens or 0
        if cached_tokens > input_tokens or (cached_tokens and cached_rate is None):
            unpriced_attempt_count += 1
            continue
        uncached_tokens = input_tokens - cached_tokens
        cost = (
            uncached_tokens * input_rate
            + cached_tokens * (cached_rate or 0)
            + output_tokens * output_rate
        ) / 1_000_000
        input_tokens_total += input_tokens
        output_tokens_total += output_tokens
        cached_tokens_total += cached_tokens
        observed_total += cost
        usage_record_count += 1
    complete_usage = usage_record_count > 0 and unpriced_attempt_count == 0
    unit_cost = observed_total / usage_record_count if complete_usage else None
    return unit_cost, {
        "source": (
            "observed_tokens_and_explicit_per_million_rates"
            if complete_usage
            else "unknown_incomplete_usage_or_cached_rate"
            if usage_record_count or unpriced_attempt_count
            else "unknown_no_usage_records"
        ),
        "model": selector,
        "usage_record_count": usage_record_count,
        "unpriced_attempt_count": unpriced_attempt_count,
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total,
        "cached_input_tokens": cached_tokens_total,
        "estimated_observed_cost_usd": observed_total if complete_usage else None,
        "mean_cost_per_usage_record_usd": unit_cost,
        "cached_input_rate_source": (
            "explicit_cached_input_rate"
            if cached_rate is not None
            else "unknown_if_cached_tokens_present"
        ),
    }


def _cost_report(manifest: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    costs = _cost_configuration(manifest)
    local = costs.get("local", {}) if isinstance(costs.get("local", {}), dict) else {}
    api = costs.get("api", {}) if isinstance(costs.get("api", {}), dict) else {}
    upfront = _finite_number(local.get("upfront_cost_usd"))
    local_variable = _finite_number(local.get("operating_cost_per_sample_usd"))
    api_variable, api_usage = _api_observed_unit_cost(api, records)
    volumes_value = costs.get("volumes", [100, 1_000, 10_000])
    volumes = (
        [
            int(value)
            for value in volumes_value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ]
        if isinstance(volumes_value, list)
        else []
    )
    volumes = sorted(set(volumes))
    complete = upfront is not None and local_variable is not None and api_variable is not None
    scenarios = []
    for volume in volumes:
        local_total = upfront + local_variable * volume if complete else None
        api_total = api_variable * volume if complete else None
        scenarios.append(
            {
                "sample_volume": volume,
                "local_total_cost_usd": local_total,
                "api_total_cost_usd": api_total,
                "local_cost_per_sample_usd": local_total / volume
                if local_total is not None and volume
                else None,
                "api_cost_per_sample_usd": api_variable if complete else None,
                "local_minus_api_cost_usd": (
                    local_total - api_total
                    if local_total is not None and api_total is not None
                    else None
                ),
                "inputs_complete": complete,
            }
        )
    if not complete:
        break_even = {"status": "unknown_missing_cost_inputs", "samples": None}
    elif local_variable < api_variable:
        break_even = {
            "status": "available",
            "samples": max(0, math.ceil(upfront / (api_variable - local_variable))),
        }
    else:
        break_even = {"status": "local_variable_cost_not_lower", "samples": None}
    return {
        "inputs": {
            "local_upfront_cost_usd": upfront,
            "local_operating_cost_per_sample_usd": local_variable,
            "api_cost_per_sample_usd": api_variable,
            "volumes": volumes,
            "local_model": local.get("model"),
            "api_model": api.get("model"),
        },
        "api_usage_estimate": api_usage,
        "scenarios": scenarios,
        "break_even": break_even,
        "interpretation": (
            "Projected costs use explicit experiment inputs. API per-sample costs may be "
            "derived from that exact model selector's observed token usage and supplied rates. "
            "Missing values are unknown; ChatGPT subscription access is not assigned an "
            "invented per-request price."
        ),
    }


def research_reports(records: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Build JSON-safe research reports for prose and equation task tracks.

    `manifest["samples"]` may carry `content_type`, `document_id` (or
    `source_document`), `split`, and `verified` metadata. Samples explicitly
    unverified or excluded are omitted; when explicit split labels exist, only
    test/evaluation samples are ranked. Legacy manifests without such metadata
    treat their frozen sample list as the evaluation set.
    """
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise TypeError("records must be a list of dictionaries")
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")

    manifest_samples = manifest.get("samples", [])
    has_frozen_samples = isinstance(manifest_samples, list)
    samples: dict[str, dict[str, Any]] = {}
    if has_frozen_samples:
        for item in manifest_samples:
            if isinstance(item, dict) and item.get("id") is not None:
                sample = dict(item)
                nested = sample.get("metadata")
                if isinstance(nested, dict):
                    for key, value in nested.items():
                        if sample.get(key) in (None, ""):
                            sample[key] = value
                sample_id = str(item["id"])
                if sample_id in samples:
                    raise ValueError(f"duplicate sample ID in run manifest: {sample_id!r}")
                samples[sample_id] = sample
    for row in records:
        sample_id = row.get("sample_id")
        if sample_id is None:
            continue
        sample_id = str(sample_id)
        if has_frozen_samples and sample_id not in samples:
            continue
        target = samples.setdefault(sample_id, {"id": sample_id})
        nested = row.get("metadata")
        if isinstance(nested, dict):
            for key, value in nested.items():
                if target.get(key) in (None, ""):
                    target[key] = value
        for key in (
            "content_type",
            "task",
            "track",
            "document_id",
            "source_document",
            "split",
            "verified",
            "verification_status",
            "excluded",
            "eligible",
        ):
            if key in row and key not in target:
                target[key] = row[key]
        if "reference" not in target and isinstance(row.get("reference"), str):
            target["reference"] = row["reference"]

    split_labels = {
        str(sample.get("split", "")).strip().lower().replace("_", "-")
        for sample in samples.values()
        if isinstance(sample.get("split"), str) and sample.get("split", "").strip()
    }
    has_split_labels = bool(split_labels.intersection(_NON_TEST_SPLITS | _TEST_SPLITS))
    excluded: list[dict[str, str]] = []
    eligible_samples: dict[str, dict[str, Any]] = {}
    for sample_id, sample in samples.items():
        reason = _sample_exclusion(sample, has_split_labels)
        if not str(sample.get("reference", "")).strip():
            reason = reason or "missing_reference"
        if reason:
            excluded.append(
                {"sample_id": sample_id, "track": _sample_kind(sample), "reason": reason}
            )
        else:
            eligible_samples[sample_id] = sample

    model_records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unexpected_result_pairs: set[tuple[str, str]] = set()
    for row in records:
        model = _model_name(row.get("model"))
        sample_id = row.get("sample_id")
        if not model or sample_id is None:
            continue
        sample_id = str(sample_id)
        if has_frozen_samples and sample_id not in samples:
            unexpected_result_pairs.add((model, sample_id))
            continue
        if sample_id in model_records[model]:
            raise ValueError(f"duplicate result for model {model!r}, sample {sample_id!r}")
        model_records[model][sample_id] = row

    model_names = _configured_models(manifest, records)
    research = manifest.get("research", {})
    if not isinstance(research, dict):
        research = {}
    repetitions = research.get("bootstrap_replicates", manifest.get("bootstrap_replicates", 1000))
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    seed = research.get("bootstrap_seed", manifest.get("seed", 42))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")

    tracks = {
        track: _track_report(
            track,
            eligible_samples,
            excluded,
            model_names,
            manifest,
            model_records,
            repetitions,
            seed,
        )
        for track in _TRACKS
    }
    return {
        "schema_version": 1,
        "unexpected_result_pairs": [
            {"model": model, "sample_id": sample_id}
            for model, sample_id in sorted(unexpected_result_pairs)
        ],
        "tracks": tracks,
        "cost_scenarios": _cost_report(manifest, records),
    }
