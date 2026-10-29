"""Versioned configuration for the new weighted leaderboard score.

The legacy CER summary in :mod:`vlm_bench.metrics` is deliberately unchanged.
This policy is consumed only by the offline report model and future experiment
manifests.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

SCORING_POLICY_VERSION = 1
NORMALIZATION_VERSION = "whitespace-v1"


@dataclass(frozen=True, slots=True)
class ScoringPolicy:
    """Frozen scoring and ranking choices for a benchmark evaluation."""

    version: int = SCORING_POLICY_VERSION
    character_accuracy_weight: float = 0.70
    exact_match_weight: float = 0.30
    accuracy_margin_points: float = 0.5
    missing_and_failed_as_empty: bool = True
    pass_aggregation: str = "arithmetic_mean"
    normalization: str = NORMALIZATION_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("scoring policy version must be an integer")
        if self.version != SCORING_POLICY_VERSION:
            raise ValueError(f"unsupported scoring policy version: {self.version!r}")
        weights = (self.character_accuracy_weight, self.exact_match_weight)
        if any(isinstance(value, bool) for value in weights):
            raise TypeError("scoring weights must be numeric")
        try:
            character_weight, exact_weight = (float(value) for value in weights)
        except (TypeError, ValueError) as exc:
            raise TypeError("scoring weights must be numeric") from exc
        if not all(
            math.isfinite(value) and 0 <= value <= 1 for value in (character_weight, exact_weight)
        ):
            raise ValueError("scoring weights must be finite values from 0 through 1")
        if not math.isclose(character_weight + exact_weight, 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("character accuracy and exact match weights must sum to 1")
        if isinstance(self.accuracy_margin_points, bool):
            raise TypeError("accuracy margin must be numeric")
        try:
            margin = float(self.accuracy_margin_points)
        except (TypeError, ValueError) as exc:
            raise TypeError("accuracy margin must be numeric") from exc
        if not math.isfinite(margin) or margin < 0 or margin > 100:
            raise ValueError("accuracy margin must be finite and between 0 and 100 points")
        if not isinstance(self.missing_and_failed_as_empty, bool):
            raise TypeError("missing_and_failed_as_empty must be a boolean")
        if not self.missing_and_failed_as_empty:
            raise ValueError(
                "missing and failed predictions must receive the empty-prediction penalty"
            )
        if self.pass_aggregation != "arithmetic_mean":
            raise ValueError("pass_aggregation must be 'arithmetic_mean'")
        if self.normalization != NORMALIZATION_VERSION:
            raise ValueError(f"unsupported normalization version: {self.normalization!r}")

        # Store normalized floats even when callers supplied integers/Decimals.
        object.__setattr__(self, "character_accuracy_weight", character_weight)
        object.__setattr__(self, "exact_match_weight", exact_weight)
        object.__setattr__(self, "accuracy_margin_points", margin)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation for run manifests."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> ScoringPolicy:
        """Parse the supported manifest and configuration spellings.

        ``scoring`` and ``ranking`` wrappers are accepted to make the policy
        easy to consume from a larger configuration object. Unknown fields are
        ignored so manifests can also carry UI-only scoring notes.
        """
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("scoring policy must be a mapping")
        nested = value.get("scoring_policy")
        if not isinstance(nested, Mapping):
            nested = value.get("scoring")
        if isinstance(nested, Mapping):
            value = nested

        aliases = {
            "version": ("version", "policy_version", "scoring_policy_version"),
            "character_accuracy_weight": (
                "character_accuracy_weight",
                "char_weight",
                "character_weight",
                "character_accuracy",
            ),
            "exact_match_weight": ("exact_match_weight", "exact_weight", "exact_match"),
            "accuracy_margin_points": (
                "accuracy_margin_points",
                "accuracy_margin",
                "practical_accuracy_margin",
                "tie_margin_points",
            ),
            "missing_and_failed_as_empty": ("missing_and_failed_as_empty", "failure_penalty"),
            "pass_aggregation": ("pass_aggregation", "repetition_aggregation"),
            "normalization": ("normalization", "normalization_version"),
        }
        accepted = {name for names in aliases.values() for name in names}
        unknown = set(value) - accepted
        if unknown:
            raise ValueError(f"unknown scoring policy field(s): {', '.join(sorted(unknown))}")
        parsed: dict[str, Any] = {}
        for field, names in aliases.items():
            for name in names:
                if name in value:
                    candidate = value[name]
                    if field == "missing_and_failed_as_empty" and name == "failure_penalty":
                        if isinstance(candidate, str):
                            if candidate not in {"empty", "all_reference_deletions"}:
                                raise ValueError(
                                    "failure_penalty must be 'empty' or 'all_reference_deletions'"
                                )
                            candidate = True
                        else:
                            if not isinstance(candidate, bool):
                                raise TypeError("failure_penalty must be a boolean")
                            candidate = candidate
                    parsed[field] = candidate
                    break
        return cls(**parsed)
