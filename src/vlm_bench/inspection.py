"""Offline inspection and subgroup reports for saved benchmark runs."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .metrics import normalize_text, score
from .research import _sample_kind, normalize_equation, research_reports, score_equation
from .runner import _digest, _lock, read_records
from .snapshot import fingerprint

_MAX_ALIGNMENT_CELLS = 250_000
_GROUP_FIELDS = {"writer_id", "difficulty", "source_document", "sample_type"}
_COMMENTARY_PATTERNS = (
    re.compile(r"(?:\bhere(?:'s| is) (?:the )?(?:transcription|text)\b|\btranscription\s*:)", re.I),
    re.compile(r"\bthe image (?:shows|contains|depicts)\b", re.I),
    re.compile(r"\bthe handwriting (?:says|reads)\b", re.I),
    re.compile(r"\b(?:i think|i can see|it appears to be)\b", re.I),
)


def _sample_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata")
    result = dict(metadata) if isinstance(metadata, dict) else {}
    for name in (
        "source_document",
        "document_id",
        "writer_id",
        "difficulty",
        "sample_type",
        "content_type",
        "split",
        "verified",
        "verification_status",
        "reference",
        "crop_path",
    ):
        value = sample.get(name)
        if value not in (None, ""):
            result[name] = value
    return result


def _sample_text(sample: dict[str, Any], field: str) -> str | None:
    value = _sample_metadata(sample).get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _load_run(run_dir: str | os.PathLike[str]):
    directory = Path(run_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Run manifest is missing: {manifest_path}")
    with _lock(directory):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = read_records(directory / "results.jsonl")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read saved run data from {directory}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Run manifest must contain a JSON object: {manifest_path}")
    configured_models = manifest.get("models")
    if not isinstance(configured_models, list) or any(
        not isinstance(model, str) or not model for model in configured_models
    ):
        raise ValueError(f"Run manifest has no valid model list: {manifest_path}")
    configured_model_set = set(configured_models)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Run manifest has no frozen sample list: {manifest_path}")
    if manifest.get("schema_version") == 2:
        integrity = manifest.get("integrity")
        immutable = {
            key: value
            for key, value in manifest.items()
            if key not in {"integrity", "status", "updated_at"}
        }
        if not isinstance(integrity, str) or _digest(immutable) != integrity:
            raise ValueError(f"Run manifest integrity check failed: {manifest_path}")
        preprocess = manifest.get("preprocess")
        if not isinstance(preprocess, str) or fingerprint(samples, preprocess) != manifest.get(
            "benchmark_fingerprint"
        ):
            raise ValueError(
                f"Run benchmark fingerprint does not match its frozen samples: {directory}"
            )

    sample_map: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Frozen sample {index} must be an object: {manifest_path}")
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Frozen sample {index} has no non-empty ID: {manifest_path}")
        if sample_id in sample_map:
            raise ValueError(f"Duplicate frozen sample ID {sample_id!r}: {manifest_path}")
        if not isinstance(sample.get("reference"), str):
            raise ValueError(f"Frozen sample {sample_id!r} has no string reference")
        sample_map[sample_id] = sample

    result_map: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Result record {index} must be an object: {directory}")
        model = row.get("model")
        sample_id = row.get("sample_id")
        status = row.get("status")
        if not isinstance(model, str) or not model:
            raise ValueError(f"Result record {index} has no non-empty model name: {directory}")
        if model not in configured_model_set:
            raise ValueError(
                f"Result record {index} references unknown model {model!r}: {directory}"
            )
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Result record {index} has no non-empty sample ID: {directory}")
        if sample_id not in sample_map:
            raise ValueError(
                f"Result record {index} references unknown frozen sample {sample_id!r}: {directory}"
            )
        if not isinstance(status, str) or not status:
            raise ValueError(f"Result record {index} has no status: {directory}")
        key = (model, sample_id)
        if key in result_map:
            raise ValueError(f"Duplicate result for model {model!r}, sample {sample_id!r}")
        if status == "success" and not isinstance(row.get("prediction"), str):
            raise ValueError(
                f"Successful result for model {model!r}, sample {sample_id!r} has no string prediction"
            )
        row_reference = row.get("reference")
        if row_reference is not None and row_reference != sample_map[sample_id]["reference"]:
            raise ValueError(
                f"Result reference for sample {sample_id!r} differs from the frozen reference"
            )
        result_map[key] = row

    eligibility = research_reports(records, manifest)
    eligible_ids = {
        sample_id
        for track in eligibility["tracks"].values()
        for sample_id in track["eligible_sample_ids"]
    }
    excluded_ids = {
        item["sample_id"]
        for track in eligibility["tracks"].values()
        for item in track["excluded_samples"]
    }
    return directory, manifest, sample_map, result_map, eligibility, eligible_ids, excluded_ids


def _has_repeated_text(text: str) -> bool:
    words = normalize_text(text).split()
    for width in range(1, min(8, len(words) // 2 + 1)):
        repeats_needed = 3 if width == 1 else 2
        for start in range(0, len(words) - width * repeats_needed + 1):
            phrase = words[start : start + width]
            if all(
                words[offset : offset + width] == phrase
                for offset in range(start + width, start + width * repeats_needed, width)
            ):
                return True
    return False


def _flags(record: dict[str, Any], prediction: str) -> list[str]:
    flags = []
    if not normalize_text(prediction):
        flags.append("empty_output")
    if record.get("truncated") is True:
        flags.append("truncated")
    if _has_repeated_text(prediction):
        flags.append("repeated_text")
    if "```" in prediction or any(pattern.search(prediction) for pattern in _COMMENTARY_PATTERNS):
        flags.append("extra_commentary")
    return flags


def minimum_edit_alignment(
    reference: str,
    prediction: str,
    *,
    equation: bool = False,
    max_cells: int = _MAX_ALIGNMENT_CELLS,
) -> dict[str, Any]:
    """Return a metric-consistent character alignment within a strict cell cap.

    Ties prefer a diagonal match/substitution, then a reference deletion, then
    a prediction insertion, matching :func:`vlm_bench.metrics._edit_counts`.
    """
    if not isinstance(equation, bool):
        raise ValueError("equation must be a boolean")
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells <= 0:
        raise ValueError("max_cells must be a positive integer")
    normalizer = normalize_equation if equation else normalize_text
    normalized_reference = normalizer(reference)
    normalized_prediction = normalizer(prediction)
    cells = (len(normalized_reference) + 1) * (len(normalized_prediction) + 1)
    if cells > max_cells:
        return {
            "status": "omitted_cell_limit",
            "cell_count": cells,
            "cell_limit": max_cells,
            "operations": [],
            "counts": None,
        }

    rows = len(normalized_reference) + 1
    columns = len(normalized_prediction) + 1
    costs = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        costs[i][0] = i
    for j in range(columns):
        costs[0][j] = j
    for i in range(1, rows):
        reference_char = normalized_reference[i - 1]
        for j in range(1, columns):
            prediction_char = normalized_prediction[j - 1]
            diagonal = costs[i - 1][j - 1] + (reference_char != prediction_char)
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            costs[i][j] = min(diagonal, deletion, insertion)

    operations: list[dict[str, str]] = []
    substitutions = deletions = insertions = 0
    i, j = rows - 1, columns - 1
    while i or j:
        if i and j:
            equal = normalized_reference[i - 1] == normalized_prediction[j - 1]
            if costs[i][j] == costs[i - 1][j - 1] + (not equal):
                operations.append(
                    {
                        "operation": "match" if equal else "substitution",
                        "reference": normalized_reference[i - 1],
                        "prediction": normalized_prediction[j - 1],
                    }
                )
                substitutions += not equal
                i -= 1
                j -= 1
                continue
        if i and costs[i][j] == costs[i - 1][j] + 1:
            operations.append(
                {
                    "operation": "deletion",
                    "reference": normalized_reference[i - 1],
                    "prediction": "",
                }
            )
            deletions += 1
            i -= 1
            continue
        if j and costs[i][j] == costs[i][j - 1] + 1:
            operations.append(
                {
                    "operation": "insertion",
                    "reference": "",
                    "prediction": normalized_prediction[j - 1],
                }
            )
            insertions += 1
            j -= 1
            continue
        raise RuntimeError("Could not reconstruct minimum edit alignment")
    operations.reverse()
    return {
        "status": "available",
        "cell_count": cells,
        "cell_limit": max_cells,
        "operations": operations,
        "counts": {
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "edits": substitutions + deletions + insertions,
        },
    }


def _row_error_rate(
    row: dict[str, Any], reference: str, prediction: str, *, equation: bool
) -> float | None:
    metrics = row.get("metrics")
    stored = _finite_nonnegative(metrics.get("cer")) if isinstance(metrics, dict) else None
    normalizer = normalize_equation if equation else normalize_text
    cells = (len(normalizer(reference)) + 1) * (len(normalizer(prediction)) + 1)
    if cells > _MAX_ALIGNMENT_CELLS:
        return stored
    try:
        scorer = score_equation if equation else score
        return scorer(prediction, reference)["cer"]
    except (TypeError, ValueError):
        return stored


def inspect_run(run_dir: str | os.PathLike[str], worst: int = 20) -> dict[str, Any]:
    """Select the worst eligible model/sample results for offline review."""
    if isinstance(worst, bool) or not isinstance(worst, int) or worst <= 0:
        raise ValueError("worst must be a positive integer")
    directory, manifest, samples, records, eligibility, eligible_ids, excluded_ids = _load_run(
        run_dir
    )
    candidates = []
    ignored_unsupported = 0
    for (model, sample_id), record in records.items():
        if record["status"] == "unsupported":
            ignored_unsupported += 1
            continue
        if sample_id not in eligible_ids:
            continue
        sample = samples[sample_id]
        reference = sample["reference"]
        prediction = record.get("prediction")
        if not isinstance(prediction, str):
            prediction = ""
        equation = _track_for_sample(sample) == "equation"
        error_rate = (
            _row_error_rate(record, reference, prediction, equation=equation)
            if record["status"] == "success"
            else None
        )
        candidates.append(
            {
                "model": model,
                "sample_id": sample_id,
                "status": record["status"],
                "error": record.get("error"),
                "reference": reference,
                "prediction": prediction,
                "cer": error_rate,
                "flags": _flags(record, prediction),
                "truncated": record.get("truncated") is True,
                "alignment": None,
                "sample": sample,
            }
        )

    candidates.sort(
        key=lambda item: (
            0 if item["status"] != "success" else 1,
            -(item["cer"] if item["cer"] is not None else -1.0),
            item["model"],
            item["sample_id"],
        )
    )
    selected = candidates[:worst]
    for item in selected:
        item["alignment"] = (
            minimum_edit_alignment(
                item["reference"],
                item["prediction"],
                equation=_track_for_sample(item["sample"]) == "equation",
            )
            if item["status"] == "success"
            else {
                "status": "unavailable",
                "cell_count": 0,
                "cell_limit": _MAX_ALIGNMENT_CELLS,
                "operations": [],
                "counts": None,
            }
        )
    return {
        "schema_version": 1,
        "run_directory": str(directory),
        "requested_worst_count": worst,
        "selected_count": len(selected),
        "eligible_result_count": len(candidates),
        "eligible_sample_count": len(eligible_ids),
        "excluded_sample_count": len(excluded_ids),
        "ignored_unsupported_result_count": ignored_unsupported,
        "alignment_cell_limit": _MAX_ALIGNMENT_CELLS,
        "evaluation": eligibility.get("evaluation", {}),
        "results": selected,
    }


def _image_url(run_dir: Path, output_path: Path, sample: dict[str, Any]) -> str | None:
    crop_path = sample.get("crop_path")
    if not isinstance(crop_path, str) or not crop_path:
        return None
    image_path = Path(crop_path)
    if not image_path.is_absolute():
        image_path = run_dir / image_path
    try:
        image_path = image_path.resolve(strict=True)
        image_path.relative_to(run_dir.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Image path for sample {sample.get('id')!r} is missing or outside the run"
        ) from exc
    if not image_path.is_file():
        raise ValueError(f"Image path for sample {sample.get('id')!r} is not a file: {image_path}")
    hashes = sample.get("hashes")
    expected_hash = hashes.get("crop") if isinstance(hashes, dict) else None
    if isinstance(expected_hash, str):
        actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Saved image hash does not match the frozen sample {sample.get('id')!r}"
            )
    relative = os.path.relpath(image_path, output_path.parent.resolve())
    return quote(Path(relative).as_posix(), safe="/.-_")


def _render_alignment(alignment: dict[str, Any]) -> str:
    if alignment["status"] != "available":
        if alignment["status"] == "omitted_cell_limit":
            return (
                '<p class="notice">Alignment omitted: '
                f"{alignment['cell_count']:,} cells exceed the "
                f"{alignment['cell_limit']:,} cell limit.</p>"
            )
        return '<p class="notice">Alignment unavailable for this result.</p>'
    rows = []
    for operation in alignment["operations"]:
        name = html.escape(operation["operation"])
        reference = html.escape(operation["reference"]) or "&nbsp;"
        prediction = html.escape(operation["prediction"]) or "&nbsp;"
        rows.append(
            f'<tr class="{name}"><td>{name}</td><td>{reference}</td><td>{prediction}</td></tr>'
        )
    counts = alignment["counts"]
    summary = (
        f"substitutions {counts['substitutions']}, deletions {counts['deletions']}, "
        f"insertions {counts['insertions']}"
    )
    return (
        f"<p>{html.escape(summary)}</p><details><summary>Character alignment</summary>"
        '<table class="alignment"><thead><tr><th>Operation</th><th>Reference</th>'
        f"<th>Prediction</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details>"
    )


def write_inspection_html(
    run_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    worst: int = 20,
) -> tuple[Path, dict[str, Any]]:
    """Write a local HTML review and return its path and report data."""
    report = inspect_run(run_dir, worst=worst)
    directory = Path(report["run_directory"])
    output = (
        Path(output_path).expanduser().resolve() if output_path else directory / "inspection.html"
    )
    if os.path.lexists(output):
        raise FileExistsError(f"Inspection output already exists: {output}")
    cards = []
    for item in report["results"]:
        image = _image_url(directory, output, item["sample"])
        image_html = (
            f'<img src="{html.escape(image, quote=True)}" alt="Sample {html.escape(item["sample_id"])}">'
            if image
            else '<p class="notice">No local image was recorded for this sample.</p>'
        )
        flags = ", ".join(item["flags"]) if item["flags"] else "none"
        cer = "N/A" if item["cer"] is None else f"{item['cer']:.2%}"
        cards.append(
            "<article>"
            f"<h2>{html.escape(item['model'])} · {html.escape(item['sample_id'])}</h2>"
            f"<p>Status: {html.escape(item['status'])}; CER: {cer}; flags: {html.escape(flags)}</p>"
            f"{image_html}"
            f"<h3>Reference</h3><pre>{html.escape(item['reference'])}</pre>"
            f"<h3>Prediction</h3><pre>{html.escape(item['prediction'])}</pre>"
            f"{_render_alignment(item['alignment'])}"
            f'<p class="error">{html.escape(str(item["error"])) if item["error"] else ""}</p>'
            "</article>"
        )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>VLM Bench error inspection</title><style>"
        "body{font:16px system-ui,sans-serif;margin:2rem;max-width:1100px;color:#18202b}"
        "article{border:1px solid #ccd3dc;border-radius:10px;padding:1rem;margin:1.5rem 0}"
        "img{max-width:100%;max-height:420px;object-fit:contain;background:#f5f5f5}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:.75rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccd3dc;padding:.25rem .5rem}"
        ".substitution{background:#ffe0de}.deletion{background:#fff0c2}.insertion{background:#dff3e4}"
        ".notice{color:#805700}.error{color:#a11}details{margin-top:1rem}"
        "</style></head><body><h1>VLM Bench error inspection</h1>"
        f"<p>{report['selected_count']} model/sample results selected from "
        f"{report['eligible_result_count']} eligible results across "
        f"{report['eligible_sample_count']} eligible samples; "
        f"{report['excluded_sample_count']} excluded by research eligibility rules. "
        f"Alignment limit: {report['alignment_cell_limit']:,} cells per result. "
        "Review flags are heuristic prompts, not verified error labels.</p>"
        f"{''.join(cards)}</body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(document)
    except FileExistsError:
        raise FileExistsError(f"Inspection output already exists: {output}") from None
    return output, report


def _track_for_sample(sample: dict[str, Any]) -> str:
    return _sample_kind(_sample_metadata(sample))


def group_report(run_dir: str | os.PathLike[str], group_by: str) -> dict[str, Any]:
    """Summarize eligible prediction coverage and accuracy by metadata field."""
    if group_by not in _GROUP_FIELDS:
        raise ValueError("group_by must be writer_id, difficulty, source_document, or sample_type")
    directory, manifest, samples, records, eligibility, eligible_ids, excluded_ids = _load_run(
        run_dir
    )
    groups: dict[tuple[bool, str], list[str]] = defaultdict(list)
    document_by_sample = {}
    for sample_id in sorted(eligible_ids):
        sample = samples[sample_id]
        value = _sample_text(sample, group_by)
        groups[(value is None, value or "Unknown")].append(sample_id)
        document_by_sample[sample_id] = _sample_text(sample, "source_document") or _sample_text(
            sample, "document_id"
        )

    model_names = manifest.get("models")
    if not isinstance(model_names, list):
        model_names = sorted({model for model, _ in records})
    model_names = [model for model in model_names if isinstance(model, str) and model]
    model_names = list(dict.fromkeys(model_names))
    models_with_records = {model for model, _ in records}
    excluded_by_reason = Counter(
        item["reason"]
        for track in eligibility["tracks"].values()
        for item in track["excluded_samples"]
    )

    def summarize_group(sample_ids: list[str]) -> list[dict[str, Any]]:
        summaries = []
        for model in model_names:
            scored = []
            failed = unsupported = missing = 0
            for sample_id in sample_ids:
                record = records.get((model, sample_id))
                if record is None:
                    missing += 1
                    continue
                if record["status"] == "unsupported":
                    unsupported += 1
                    continue
                if record["status"] != "success":
                    failed += 1
                    continue
                prediction = record.get("prediction")
                if not isinstance(prediction, str):
                    failed += 1
                    continue
                reference = samples[sample_id]["reference"]
                metrics = (
                    score_equation(prediction, reference)
                    if _track_for_sample(samples[sample_id]) == "equation"
                    else score(prediction, reference)
                )
                scored.append(metrics)
            reference_chars = sum(row["reference_chars"] for row in scored)
            reference_words = sum(row["reference_words"] for row in scored)
            char_edits = sum(row["char_edits"] for row in scored)
            word_edits = sum(row["word_edits"] for row in scored)
            summaries.append(
                {
                    "model": model,
                    "eligible_sample_count": len(sample_ids),
                    "scored_sample_count": len(scored),
                    "success_count": len(scored),
                    "failed_sample_count": failed,
                    "unsupported_sample_count": unsupported,
                    "missing_sample_count": missing,
                    "sample_coverage": len(scored) / len(sample_ids) if sample_ids else None,
                    "cer": char_edits / reference_chars if reference_chars else None,
                    "wer": word_edits / reference_words if reference_words else None,
                    "exact_match_rate": (
                        sum(bool(row["exact_match"]) for row in scored) / len(scored)
                        if scored
                        else None
                    ),
                    "status": "complete"
                    if sample_ids and len(scored) == len(sample_ids)
                    else "not_run"
                    if model not in models_with_records
                    else "incomplete",
                }
            )
        return summaries

    rendered_groups = []
    for (is_unknown, value), sample_ids in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1].casefold())
    ):
        documents = {document_by_sample[item] for item in sample_ids if document_by_sample[item]}
        rendered_groups.append(
            {
                "value": value,
                "is_unknown": is_unknown,
                "eligible_sample_count": len(sample_ids),
                "document_count": len(documents),
                "samples_without_document_metadata": sum(
                    document_by_sample[item] is None for item in sample_ids
                ),
                "sample_ids": sample_ids,
                "models": summarize_group(sample_ids),
            }
        )
    known_documents = {document for document in document_by_sample.values() if document is not None}
    return {
        "schema_version": 1,
        "run_directory": str(directory),
        "group_by": group_by,
        "evaluation": eligibility.get("evaluation", {}),
        "eligible_sample_count": len(eligible_ids),
        "excluded_sample_count": len(excluded_ids),
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "eligible_document_count": len(known_documents),
        "eligible_samples_without_document_metadata": sum(
            document is None for document in document_by_sample.values()
        ),
        "groups": rendered_groups,
    }
