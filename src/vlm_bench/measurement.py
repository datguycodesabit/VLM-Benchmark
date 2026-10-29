"""Normalized timing evidence shared by local and remote backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

TIMING_SOURCES = {
    "client_measured",
    "provider_reported",
    "adapter_instrumented",
    "unavailable",
}


@dataclass(frozen=True)
class TimingObservation:
    """One timing value with an explicit unit, source, and boundary."""

    metric_name: str
    value: float | None
    unit: str
    source: str
    start_boundary: str | None
    end_boundary: str | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ValueError("Timing metric_name must be a non-empty string")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("Timing unit must be a non-empty string")
        if self.source not in TIMING_SOURCES:
            raise ValueError(f"Timing source must be one of {', '.join(sorted(TIMING_SOURCES))}")
        for name in ("start_boundary", "end_boundary"):
            boundary = getattr(self, name)
            if boundary is not None and (not isinstance(boundary, str) or not boundary.strip()):
                raise ValueError(f"Timing {name} must be a non-empty string or null")
        if self.value is None:
            if self.source != "unavailable":
                raise ValueError("A null timing value must use source 'unavailable'")
            if not isinstance(self.unavailable_reason, str) or not self.unavailable_reason.strip():
                raise ValueError("Unavailable timing values require an unavailable_reason")
        else:
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(float(self.value))
                or self.value < 0
            ):
                raise ValueError("Timing value must be a finite non-negative number or null")
            if self.source == "unavailable":
                raise ValueError("A timing with a numeric value cannot use source 'unavailable'")
            if self.unavailable_reason is not None:
                raise ValueError("Available timing values cannot have an unavailable_reason")
            if not self.start_boundary or not self.end_boundary:
                raise ValueError("Available timing values require start and end boundaries")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "start_boundary": self.start_boundary,
            "end_boundary": self.end_boundary,
            "unavailable_reason": self.unavailable_reason,
        }


def timing_observation(
    metric_name: str,
    value: float | int | None,
    *,
    unit: str,
    source: str,
    start_boundary: str | None,
    end_boundary: str | None,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Build and validate a JSON-serializable observation dictionary."""
    if value is not None and not isinstance(value, bool):
        value = float(value)
    return TimingObservation(
        metric_name=metric_name,
        value=value,
        unit=unit,
        source=source,
        start_boundary=start_boundary,
        end_boundary=end_boundary,
        unavailable_reason=unavailable_reason,
    ).to_dict()


def unavailable_timing(
    metric_name: str,
    reason: str,
    *,
    unit: str = "seconds",
    start_boundary: str | None = None,
    end_boundary: str | None = None,
) -> dict[str, Any]:
    return timing_observation(
        metric_name,
        None,
        unit=unit,
        source="unavailable",
        start_boundary=start_boundary,
        end_boundary=end_boundary,
        unavailable_reason=reason,
    )


def normalize_timing_observations(
    observations: Iterable[Mapping[str, Any]], *, default_source: str | None = None
) -> list[dict[str, Any]]:
    """Validate plugin/provider observation mappings into the normalized shape.

    ``default_source`` may be used for an adapter hook that returns an otherwise
    complete observation without its source. Inputs are copied so callers keep
    their own data and cannot mutate the normalized evidence after validation.
    """
    normalized: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise TypeError(f"Timing observation {index} must be a mapping")
        data = dict(observation)
        if "source" not in data and default_source is not None:
            data["source"] = default_source
        required = {
            "metric_name",
            "value",
            "unit",
            "source",
            "start_boundary",
            "end_boundary",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(
                f"Timing observation {index} is missing fields: {', '.join(sorted(missing))}"
            )
        normalized.append(
            timing_observation(
                data["metric_name"],
                data["value"],
                unit=data["unit"],
                source=data["source"],
                start_boundary=data["start_boundary"],
                end_boundary=data["end_boundary"],
                unavailable_reason=data.get("unavailable_reason"),
            )
        )
    return normalized
