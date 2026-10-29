"""Deterministic anchored accuracy bands and latency tie-breaking."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0 <= number <= 100 else None


def rank_track(
    model_rows: Sequence[Mapping[str, Any]],
    accuracy_margin_points: float = 0.5,
    finalizable: bool = True,
) -> dict[str, Any]:
    """Rank one track, anchoring each band at its highest remaining score.

    Rows without a finite score remain visible in ``unranked_models``. Every
    band is ordered by median successful uncached latency, then score, then
    model name. A band with missing latency has no speed-resolved placements.
    The winner is withheld unless the evaluation is finalizable and all timing
    needed to resolve the top band is available.
    """
    if isinstance(accuracy_margin_points, bool):
        raise TypeError("accuracy_margin_points must be numeric")
    try:
        margin = float(accuracy_margin_points)
    except (TypeError, ValueError) as exc:
        raise TypeError("accuracy_margin_points must be numeric") from exc
    if not math.isfinite(margin) or margin < 0 or margin > 100:
        raise ValueError("accuracy_margin_points must be finite and between 0 and 100")
    if not isinstance(finalizable, bool):
        raise TypeError("finalizable must be a boolean")

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    unranked: list[str] = []
    incomplete_models: list[str] = []
    for row in model_rows:
        if not isinstance(row, Mapping):
            raise TypeError("each model row must be a mapping")
        model = row.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("each model row must have a non-empty model name")
        if model in seen:
            raise ValueError(f"duplicate model row: {model!r}")
        seen.add(model)
        row_status = row.get("status")
        if row_status == "unsupported":
            unranked.append(model)
            continue
        model_score = _score(row.get("score", row.get("accuracy_score")))
        if model_score is None:
            unranked.append(model)
            continue
        candidate = {
            "model": model,
            "score": model_score,
            "median_latency_seconds": _finite_nonnegative(row.get("median_latency_seconds")),
        }
        candidates.append(candidate)
        if row.get("terminal_complete") is False:
            incomplete_models.append(model)

    candidates.sort(key=lambda row: (-row["score"], row["model"]))
    bands: list[dict[str, Any]] = []
    remaining = list(candidates)
    all_band_members: list[dict[str, Any]] = []
    while remaining:
        anchor = remaining[0]["score"]
        included = [row for row in remaining if anchor - row["score"] <= margin]
        included_models = {row["model"] for row in included}
        remaining = [row for row in remaining if row["model"] not in included_models]
        timing_complete = all(row["median_latency_seconds"] is not None for row in included)
        if timing_complete:
            included.sort(
                key=lambda row: (
                    row["median_latency_seconds"],
                    -row["score"],
                    row["model"],
                )
            )
        else:
            # Keep output deterministic while making clear this is not a
            # speed-resolved order.
            included.sort(key=lambda row: (-row["score"], row["model"]))
        band_index = len(bands) + 1
        band_models: list[dict[str, Any]] = []
        for position, row in enumerate(included, start=1):
            item = {
                **row,
                "band": band_index,
                "band_anchor_score": anchor,
                "within_band_position": position if timing_complete else None,
                "rank": None,
            }
            band_models.append(item)
            all_band_members.append(item)
        bands.append(
            {
                "band": band_index,
                "anchor_score": anchor,
                "margin_points": margin,
                "timing_complete": timing_complete,
                "ordering": "median_latency_then_score_then_model"
                if timing_complete
                else "unresolved_missing_latency",
                "models": band_models,
            }
        )

    # Rank positions are assigned for fully resolved bands. An unresolved
    # band's members have no internal ranks, although later band offsets remain
    # known because the number of members is still fixed.
    rank_cursor = 1
    for band in bands:
        if band["timing_complete"]:
            for item in band["models"]:
                item["rank"] = rank_cursor
                rank_cursor += 1
        else:
            rank_cursor += len(band["models"])

    winner = None
    winner_reason = None
    top_band = bands[0] if bands else None
    if not candidates:
        winner_reason = "no_scored_models"
    elif not finalizable or incomplete_models:
        winner_reason = "incomplete_evaluation"
    elif top_band is not None and not top_band["timing_complete"]:
        winner_reason = "missing_top_band_latency"
    else:
        winner = top_band["models"][0]["model"] if top_band else None
        if winner is None:
            winner_reason = "no_scored_models"

    if not candidates:
        status = "no_scored_models"
    elif winner is not None:
        status = (
            "available"
            if all(band["timing_complete"] for band in bands)
            else "partial_missing_latency"
        )
    elif winner_reason == "incomplete_evaluation":
        status = "incomplete"
    elif winner_reason == "missing_top_band_latency":
        status = "missing_top_band_latency"
    else:
        status = "unavailable"

    return {
        "status": status,
        "accuracy_margin_points": margin,
        "winner": winner,
        "winner_status": "available" if winner is not None else "withheld",
        "winner_reason": winner_reason,
        "bands": bands,
        "unranked_models": sorted(unranked),
        "incomplete_models": sorted(set(incomplete_models)),
        "ranked_models": all_band_members,
    }
