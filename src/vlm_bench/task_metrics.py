"""Optional, task-specific metrics kept separate from transcription accuracy."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from .metrics import normalize_text

TASK_METRICS_VERSION = 1
FORMULA_RENDERER_NAME = "matplotlib-mathtext"
FORMULA_RENDERER_VERSION = "3.10.3"
MAX_EXPRESSION_LENGTH = 4096
MAX_RENDER_DIMENSION = 4096
_EQUATION_TYPES = {"equation", "math", "mathematics", "latex", "formula"}
_METRIC_NAMES = (
    "formula_render_similarity",
    "critical_expression_accuracy",
    "reading_order_accuracy",
)
_SCORED_STATUSES = {"scored"}


def _normalize(value: str) -> str:
    return normalize_text(unicodedata.normalize("NFC", value))


def _kind(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    value = next(
        (
            candidate
            for candidate in (
                sample.get("content_type"),
                sample.get("task"),
                metadata.get("content_type"),
                metadata.get("task"),
            )
            if candidate not in (None, "")
        ),
        "prose",
    )
    normalized = str(value).strip().lower().replace("_", "-")
    return "equation" if normalized in _EQUATION_TYPES or "equation" in normalized else "prose"


def _annotations(sample: dict[str, Any]) -> dict[str, Any] | None:
    metadata = sample.get("metadata")
    nested = metadata.get("annotations") if isinstance(metadata, dict) else None
    value = nested if nested is not None else sample.get("annotations")
    return value if isinstance(value, dict) else None


def _annotation_metric(
    metric_name: str,
    prediction: str,
    reference: str,
    annotation: Any,
) -> dict[str, Any]:
    if annotation is None:
        return {"status": "not_annotated", "value": None}
    if not isinstance(annotation, list):
        return {"status": "error", "value": None, "message": "annotation must be a list"}
    if not annotation:
        return {"status": "not_annotated", "value": None}

    normalized_prediction = _normalize(prediction)
    normalized_reference = _normalize(reference)
    details: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, entry in enumerate(annotation):
        if metric_name == "critical_expression_accuracy":
            if not isinstance(entry, str) or not entry.strip():
                errors.append(f"annotation {index} must be a non-empty string")
                continue
            expression = _normalize(entry)
            ref_count = normalized_reference.count(expression)
            pred_count = normalized_prediction.count(expression)
            if ref_count < 1:
                errors.append(f"critical expression {entry!r} is absent from the reference")
                continue
            details.append(
                {
                    "expression": entry,
                    "reference_occurrences": ref_count,
                    "prediction_occurrences": pred_count,
                    "passed": pred_count == ref_count,
                }
            )
        else:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or any(not isinstance(anchor, str) or not anchor.strip() for anchor in entry)
            ):
                errors.append(
                    f"reading-order annotation {index} must be a pair of non-empty strings"
                )
                continue
            earlier, later = (_normalize(anchor) for anchor in entry)
            ref_earlier = normalized_reference.find(earlier)
            ref_later = normalized_reference.find(later)
            if (
                ref_earlier < 0
                or ref_later < 0
                or normalized_reference.count(earlier) != 1
                or normalized_reference.count(later) != 1
                or ref_earlier >= ref_later
            ):
                errors.append(
                    f"reading-order pair {entry!r} must occur exactly once in the declared order in the reference"
                )
                continue
            pred_earlier = normalized_prediction.find(earlier)
            pred_later = normalized_prediction.find(later)
            passed = (
                pred_earlier >= 0
                and pred_later >= 0
                and normalized_prediction.count(earlier) == 1
                and normalized_prediction.count(later) == 1
                and pred_earlier < pred_later
            )
            details.append(
                {
                    "earlier": entry[0],
                    "later": entry[1],
                    "passed": passed,
                }
            )

    if errors:
        return {"status": "error", "value": None, "items": details, "errors": errors}
    passed = sum(item["passed"] for item in details)
    total = len(details)
    return {
        "status": "scored",
        "value": passed / total,
        "matching": "normalized occurrence count must match the reference",
        "passed": passed,
        "total": total,
        "items": details,
    }


def preflight_renderer() -> dict[str, Any]:
    """Confirm the pinned MathText renderer is installed and importable.

    The returned identity is recorded in run manifests. No TeX installation or
    external process is used; Matplotlib MathText supports a documented subset
    of TeX math syntax.
    """
    try:
        import matplotlib
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import MathTextParser
    except ImportError as exc:
        raise RuntimeError(
            "Formula rendering requires the optional dependency; install with "
            "`uv sync --extra formula-render`"
        ) from exc
    if matplotlib.__version__ != FORMULA_RENDERER_VERSION:
        raise RuntimeError(
            "Formula rendering requires matplotlib=="
            f"{FORMULA_RENDERER_VERSION}; found {matplotlib.__version__}"
        )
    # Construct now so a broken Matplotlib font/mathtext installation fails
    # during preflight, before a provider is contacted.
    try:
        with matplotlib.rc_context(
            {
                "mathtext.fontset": "dejavusans",
                "mathtext.default": "it",
                "text.usetex": False,
            }
        ):
            MathTextParser("agg").parse(
                "$x$", dpi=96, prop=FontProperties(family="DejaVu Sans", size=18)
            )
    except Exception as exc:
        raise RuntimeError(f"Cannot initialize the pinned MathText renderer: {exc}") from exc
    return {
        "name": FORMULA_RENDERER_NAME,
        "version": FORMULA_RENDERER_VERSION,
        "task_metrics_version": TASK_METRICS_VERSION,
        "engine": "MathTextParser(agg), DejaVu Sans, 18 pt, 96 dpi",
    }


def _math_source(expression: str) -> str:
    value = expression.strip()
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        return value
    if value.startswith(r"\(") and value.endswith(r"\)"):
        value = value[2:-2]
    elif value.startswith(r"\[") and value.endswith(r"\]"):
        value = value[2:-2]
    return f"${value}$"


def _render_math(expression: str, parser: Any, font_properties: Any, numpy: Any):
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"formula expression exceeds the {MAX_EXPRESSION_LENGTH}-character rendering limit"
        )
    if not expression.strip():
        return numpy.zeros((1, 1), dtype=bool), 0
    parsed = parser.parse(_math_source(expression), dpi=96, prop=font_properties)
    image = parsed.image
    pixels = image.as_array() if hasattr(image, "as_array") else numpy.asarray(image)
    pixels = numpy.asarray(pixels)
    if pixels.ndim == 3:
        pixels = pixels[..., -1]
    if pixels.ndim != 2:
        raise ValueError(f"renderer returned an unsupported raster shape {pixels.shape!r}")
    height, width = pixels.shape
    if width > MAX_RENDER_DIMENSION or height > MAX_RENDER_DIMENSION:
        raise ValueError(
            f"rendered formula is {width}x{height}px; each dimension must be at most "
            f"{MAX_RENDER_DIMENSION}px"
        )
    mask = pixels > 0
    baseline = max(0, round(float(parsed.height) - float(parsed.depth)))
    if mask.any():
        ys, xs = numpy.nonzero(mask)
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1
        mask = mask[top:bottom, left:right]
        baseline = max(0, baseline - top)
    return mask, baseline


def _foreground_iou(
    reference: Any, reference_baseline: int, prediction: Any, prediction_baseline: int
):
    import numpy

    ref_height, ref_width = reference.shape
    pred_height, pred_width = prediction.shape
    above = max(reference_baseline, prediction_baseline)
    below = max(ref_height - reference_baseline, pred_height - prediction_baseline)
    height = max(1, above + below)
    width = max(1, ref_width, pred_width)
    if width > MAX_RENDER_DIMENSION or height > MAX_RENDER_DIMENSION:
        raise ValueError(
            f"aligned formula canvas is {width}x{height}px; each dimension must be at most "
            f"{MAX_RENDER_DIMENSION}px"
        )
    ref_canvas = numpy.zeros((height, width), dtype=bool)
    pred_canvas = numpy.zeros((height, width), dtype=bool)
    ref_top = above - reference_baseline
    pred_top = above - prediction_baseline
    ref_canvas[ref_top : ref_top + ref_height, :ref_width] = reference
    pred_canvas[pred_top : pred_top + pred_height, :pred_width] = prediction
    intersection = int(numpy.logical_and(ref_canvas, pred_canvas).sum())
    union = int(numpy.logical_or(ref_canvas, pred_canvas).sum())
    return intersection / union if union else 1.0


def _formula_score(prediction: str, reference: str) -> dict[str, Any]:
    identity = {
        "renderer": FORMULA_RENDERER_NAME,
        "renderer_version": FORMULA_RENDERER_VERSION,
        "method": "foreground-pixel intersection over union after baseline alignment",
        "supported_syntax": "Matplotlib MathText subset; not full LaTeX",
    }
    try:
        preflight_renderer()
        import matplotlib
        import numpy
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import MathTextParser

        with matplotlib.rc_context(
            {
                "mathtext.fontset": "dejavusans",
                "mathtext.default": "it",
                "text.usetex": False,
            }
        ):
            parser = MathTextParser("agg")
            font = FontProperties(family="DejaVu Sans", size=18)
            reference_image, reference_baseline = _render_math(reference, parser, font, numpy)
            prediction_image, prediction_baseline = _render_math(prediction, parser, font, numpy)
        value = _foreground_iou(
            reference_image, reference_baseline, prediction_image, prediction_baseline
        )
        return {"status": "scored", "value": value, **identity}
    except Exception as exc:
        return {"status": "error", "value": None, "message": str(exc), **identity}


def score_task(
    prediction: str,
    reference: str,
    sample: dict[str, Any],
    formula_rendering: bool = False,
) -> dict[str, Any]:
    """Compute optional formula and annotation metrics without changing CER/WER.

    Formula rendering is opt-in. Critical expressions and reading-order pairs
    are scored only when annotations exist and always report their own coverage
    status. The returned values must not be combined into a model ranking.
    """
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise TypeError("prediction and reference must be strings")
    if not isinstance(sample, dict):
        raise TypeError("sample must be a dictionary")
    if not isinstance(formula_rendering, bool):
        raise ValueError("formula_rendering must be a boolean")

    annotations = _annotations(sample)
    if sample.get("annotations") is not None and annotations is None:
        malformed_annotations = sample.get("annotations")
    else:
        metadata = sample.get("metadata")
        malformed_annotations = (
            metadata.get("annotations")
            if isinstance(metadata, dict)
            and metadata.get("annotations") is not None
            and annotations is None
            else None
        )
    scores: dict[str, Any] = {}
    if not formula_rendering:
        scores["formula_render_similarity"] = {"status": "disabled", "value": None}
    elif _kind(sample) != "equation":
        scores["formula_render_similarity"] = {"status": "not_applicable", "value": None}
    else:
        scores["formula_render_similarity"] = _formula_score(prediction, reference)

    if malformed_annotations is not None:
        message = "annotations must be an object with critical_expressions and reading_order lists"
        scores["critical_expression_accuracy"] = {
            "status": "error",
            "value": None,
            "message": message,
        }
        scores["reading_order_accuracy"] = {
            "status": "error",
            "value": None,
            "message": message,
        }
        errors = [
            {"metric": "critical_expression_accuracy", "message": message},
            {"metric": "reading_order_accuracy", "message": message},
        ]
    else:
        annotations = annotations or {}
        critical = annotations.get("critical_expressions")
        order = annotations.get("reading_order")
        scores["critical_expression_accuracy"] = _annotation_metric(
            "critical_expression_accuracy", prediction, reference, critical
        )
        scores["reading_order_accuracy"] = _annotation_metric(
            "reading_order_accuracy", prediction, reference, order
        )
        errors = []
        for metric_name, value in scores.items():
            if value.get("status") == "error":
                messages = value.get("errors", [value.get("message", "metric failed")])
                for message in messages:
                    errors.append({"metric": metric_name, "message": str(message)})
    return {"version": TASK_METRICS_VERSION, "scores": scores, "errors": errors}


def _sample_track(sample: dict[str, Any]) -> str:
    kind = _kind(sample)
    if kind == "equation":
        return "equation"
    metadata = sample.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    sample_type = str(sample.get("sample_type", metadata.get("sample_type", ""))).lower()
    return "word" if sample_type == "word" else "prose"


def _annotation_list(sample: dict[str, Any], name: str) -> list[Any]:
    annotations = _annotations(sample) or {}
    value = annotations.get(name, [])
    return value if isinstance(value, list) else []


def _empty_metric() -> dict[str, Any]:
    return {
        "status_counts": Counter(),
        "values": [],
        "passed": 0,
        "items": 0,
        "errors": [],
        "recorded": 0,
    }


def _finalize_metric(
    metric: dict[str, Any], denominator: int, annotated_samples: int
) -> dict[str, Any]:
    values = metric["values"]
    item_count = metric["items"]
    passed = metric["passed"]
    return {
        "status_counts": dict(sorted(metric["status_counts"].items())),
        "recorded_sample_count": metric["recorded"],
        "scored_sample_count": len(values),
        "annotated_sample_count": annotated_samples,
        "coverage_denominator": denominator,
        "coverage": len(values) / denominator if denominator else None,
        "mean_sample_score": sum(values) / len(values) if values else None,
        "passed_items": passed,
        "total_items": item_count,
        "item_accuracy": passed / item_count if item_count else None,
        "error_count": sum(1 for _ in metric["errors"]),
        "errors": metric["errors"],
    }


def aggregate_task_metrics(
    records: list[dict[str, Any]],
    samples: dict[str, dict[str, Any]],
    models: list[str],
    *,
    formula_rendering_enabled: bool,
) -> dict[str, Any]:
    """Aggregate saved version-matched task metrics with explicit coverage.

    `samples` should contain only eligible frozen samples. The aggregate is
    descriptive and intentionally contains no combined score or ranking.
    """
    if not isinstance(records, list) or not isinstance(samples, dict):
        raise TypeError("records and samples must be a list and dictionary")
    if not isinstance(formula_rendering_enabled, bool):
        raise ValueError("formula_rendering_enabled must be a boolean")
    by_track: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    track_samples: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for sample_id, sample in samples.items():
        track_samples[_sample_track(sample)][str(sample_id)] = sample
    for row in records:
        if not isinstance(row, dict) or row.get("sample_id") is None:
            continue
        sample_id = str(row["sample_id"])
        sample = samples.get(sample_id)
        if sample is None:
            continue
        model = str(row.get("model", "")).strip()
        if model:
            by_track[_sample_track(sample)].setdefault(model, {})[sample_id] = row

    track_reports: dict[str, Any] = {}
    for track in ("prose", "word", "equation"):
        eligible = track_samples.get(track, {})
        model_reports: list[dict[str, Any]] = []
        for model in models:
            model_rows = by_track.get(track, {}).get(model, {})
            accumulators = {name: _empty_metric() for name in _METRIC_NAMES}
            version_mismatch_count = 0
            missing_task_metric_count = 0
            for sample_id, row in model_rows.items():
                metrics = row.get("metrics")
                task_metrics = metrics.get("task_metrics") if isinstance(metrics, dict) else None
                if not isinstance(task_metrics, dict):
                    missing_task_metric_count += 1
                    continue
                if task_metrics.get("version") != TASK_METRICS_VERSION:
                    version_mismatch_count += 1
                    continue
                scores = task_metrics.get("scores")
                if not isinstance(scores, dict):
                    version_mismatch_count += 1
                    continue
                for metric_name in _METRIC_NAMES:
                    value = scores.get(metric_name)
                    if not isinstance(value, dict):
                        continue
                    acc = accumulators[metric_name]
                    status = str(value.get("status", "unknown"))
                    acc["recorded"] += 1
                    acc["status_counts"][status] += 1
                    score_value = value.get("value")
                    if status in _SCORED_STATUSES and isinstance(score_value, (int, float)):
                        if math.isfinite(float(score_value)):
                            acc["values"].append(float(score_value))
                    if isinstance(value.get("passed"), int) and isinstance(value.get("total"), int):
                        acc["passed"] += value["passed"]
                        acc["items"] += value["total"]
                    if status == "error":
                        messages = value.get("errors")
                        if not isinstance(messages, list):
                            messages = [value.get("message", "metric failed")]
                        for message in messages:
                            acc["errors"].append({"sample_id": sample_id, "message": str(message)})

            metric_reports: dict[str, Any] = {}
            for metric_name, accumulator in accumulators.items():
                if metric_name == "formula_render_similarity":
                    denominator = (
                        len(eligible) if formula_rendering_enabled and track == "equation" else 0
                    )
                    annotated = denominator
                elif metric_name == "critical_expression_accuracy":
                    annotated = sum(
                        bool(_annotation_list(sample, "critical_expressions"))
                        for sample in eligible.values()
                    )
                    denominator = annotated
                else:
                    annotated = sum(
                        bool(_annotation_list(sample, "reading_order"))
                        for sample in eligible.values()
                    )
                    denominator = annotated
                metric_reports[metric_name] = _finalize_metric(accumulator, denominator, annotated)
            model_reports.append(
                {
                    "model": model,
                    "eligible_sample_count": len(eligible),
                    "result_sample_count": len(model_rows),
                    "missing_task_metric_count": missing_task_metric_count,
                    "version_mismatch_count": version_mismatch_count,
                    "metrics": metric_reports,
                }
            )
        track_reports[track] = {
            "eligible_sample_count": len(eligible),
            "models": model_reports,
        }

    result = {
        "version": TASK_METRICS_VERSION,
        "formula_rendering_enabled": formula_rendering_enabled,
        "tracks": track_reports,
        "ranking": None,
    }
    if formula_rendering_enabled:
        result["formula_renderer"] = {
            "name": FORMULA_RENDERER_NAME,
            "version": FORMULA_RENDERER_VERSION,
        }
    return result
