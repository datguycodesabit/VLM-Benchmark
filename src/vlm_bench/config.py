"""Versioned experiment configuration; credentials never belong in an experiment."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .backends import parse_model

_EXPERIMENT_FIELDS = {
    "data",
    "prepared",
    "models",
    "limit",
    "seed",
    "layout",
    "preprocess",
    "split",
    "content_type",
    "warmup",
    "preprocessing",
}
_SELECTION_FIELDS = {
    "data",
    "layout",
    "preprocess",
    "preprocessing",
    "limit",
    "seed",
    "split",
    "content_type",
}
_MODEL_SETTINGS = {
    "device",
    "revision",
    "num_predict",
    "num_beams",
    "reasoning_effort",
    "image_detail",
}
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
_SPLITS = {"train", "validation", "test"}
_LAYOUTS = {"auto", "paired", "iam-words", "iam-lines", "iam-forms"}
_PROFILES = {"original", "enhanced"}
_CONTENT_TYPES = {"prose", "equation", "word"}
_MAX_SEED = 2**32 - 1


def _selector(value: Any, context: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty model selector")
    try:
        provider, model = parse_model(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model selector in {context}: {value!r}") from exc
    if not model.strip():
        raise ValueError(f"{context} must be a non-empty model selector")
    return provider, model


def _canonical_selector(value: Any, context: str) -> str:
    provider, model = _selector(value, context)
    return f"{provider}:{model}"


def _selector_list(value: Any, context: str) -> list[tuple[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{context} must be a non-empty list of model selectors")
    parsed = [_selector(item, context) for item in value]
    canonical = [f"{provider}:{model}" for provider, model in parsed]
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{context} must not contain duplicate model selectors")
    return parsed


def _positive_integer(value: Any, name: str, *, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def _validate_model_settings(settings: Any, context: str) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise ValueError(f"Settings for {context} must be a table")
    unsupported = set(settings) - _MODEL_SETTINGS
    if unsupported:
        # This also blocks credentials such as api_key, token, and password.
        raise ValueError(f"Unsupported settings for {context}; credentials must not be in TOML")

    result = dict(settings)
    if "device" in result and (
        not isinstance(result["device"], str) or result["device"] not in {"auto", "cpu", "mps"}
    ):
        raise ValueError(f"device for {context} must be one of auto, cpu, or mps")
    if "revision" in result and (
        not isinstance(result["revision"], str) or not result["revision"].strip()
    ):
        raise ValueError(f"revision for {context} must be a non-empty string")
    if "num_predict" in result:
        _positive_integer(result["num_predict"], f"num_predict for {context}")
    if "num_beams" in result:
        _positive_integer(result["num_beams"], f"num_beams for {context}", maximum=10)
    if "image_detail" in result and (
        not isinstance(result["image_detail"], str)
        or result["image_detail"] not in {"auto", "low", "high"}
    ):
        raise ValueError(f"image_detail for {context} must be one of auto, low, or high")
    if "reasoning_effort" in result and (
        not isinstance(result["reasoning_effort"], str)
        or result["reasoning_effort"] not in _REASONING_EFFORTS
    ):
        raise ValueError(f"Unsupported reasoning_effort for {context}")
    return result


def validate_settings(
    models: Sequence[str], settings: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate and return settings only for the canonically selected models.

    Config files may contain settings for a default model list that is replaced
    by explicit CLI selectors. Those unused entries are ignored here. Returned
    keys use the spelling from ``models`` so callers can pass them directly to
    the engine.
    """
    selected = _selector_list(models, "models")
    if not isinstance(settings, Mapping):
        raise ValueError("Model settings must be a table keyed by model selector")

    configured: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    selected_by_canonical = {
        f"{provider}:{model}": original for original, (provider, model) in zip(models, selected)
    }
    for selector, options in settings.items():
        canonical = _canonical_selector(selector, "model settings")
        if canonical in seen:
            raise ValueError(f"Duplicate model settings after provider resolution: {selector!r}")
        seen.add(canonical)
        if canonical not in selected_by_canonical:
            continue
        provider, _ = _selector(selector, "model settings")
        canonical_options = _validate_model_settings(options, str(selector))
        if provider == "trocr":
            inapplicable = set(canonical_options) & {"reasoning_effort", "image_detail"}
        elif provider in {"openai", "chatgpt"}:
            inapplicable = set(canonical_options) & {"device", "revision", "num_beams"}
        else:
            inapplicable = set(canonical_options) & {
                "device",
                "revision",
                "num_beams",
                "reasoning_effort",
                "image_detail",
            }
        if inapplicable:
            names = ", ".join(sorted(inapplicable))
            raise ValueError(f"Settings {names} are not supported for provider {provider!r}")
        configured[canonical] = canonical_options

    return {
        selected_by_canonical[canonical]: configured[canonical]
        for canonical in selected_by_canonical
        if canonical in configured
    }


def _path_from_config(value: Any, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"experiment.{name} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _validate_experiment(experiment: Any, base: Path) -> dict[str, Any]:
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment must be a table")
    unknown = set(experiment) - _EXPERIMENT_FIELDS
    if unknown:
        raise ValueError("Unknown experiment option: " + ", ".join(sorted(unknown)))

    result = dict(experiment)
    if "prepared" in result:
        conflicts = set(result) & _SELECTION_FIELDS
        if conflicts:
            raise ValueError(
                "experiment.prepared freezes selection and cannot be combined with: "
                + ", ".join(sorted(conflicts))
            )
        result["prepared"] = _path_from_config(result["prepared"], "prepared", base)
    if "data" in result:
        result["data"] = _path_from_config(result["data"], "data", base)
    if "models" in result:
        _selector_list(result["models"], "experiment.models")
    if "limit" in result:
        _positive_integer(result["limit"], "experiment.limit")
    if "seed" in result and (
        isinstance(result["seed"], bool)
        or not isinstance(result["seed"], int)
        or not 0 <= result["seed"] <= _MAX_SEED
    ):
        raise ValueError(f"experiment.seed must be an integer from 0 to {_MAX_SEED}")
    if "layout" in result and (
        not isinstance(result["layout"], str) or result["layout"] not in _LAYOUTS
    ):
        raise ValueError("experiment.layout must be a supported dataset layout")
    for name in ("preprocess",):
        if name in result and (not isinstance(result[name], str) or result[name] not in _PROFILES):
            raise ValueError(f"experiment.{name} must be original or enhanced")
    if "preprocessing" in result:
        profiles = result["preprocessing"]
        if (
            not isinstance(profiles, list)
            or not profiles
            or any(not isinstance(profile, str) or profile not in _PROFILES for profile in profiles)
        ):
            raise ValueError("experiment.preprocessing must be a non-empty list of valid profiles")
        if len(profiles) != len(set(profiles)):
            raise ValueError("experiment.preprocessing must not contain duplicate profiles")
    if "split" in result and (
        not isinstance(result["split"], str) or result["split"] not in _SPLITS
    ):
        raise ValueError("experiment.split must be train, validation, or test")
    if "content_type" in result and (
        not isinstance(result["content_type"], str) or result["content_type"] not in _CONTENT_TYPES
    ):
        raise ValueError("experiment.content_type must be prose, equation, or word")
    if "warmup" in result and not isinstance(result["warmup"], bool):
        raise ValueError("experiment.warmup must be a boolean")
    return result


def _validate_costs(costs: Any) -> dict[str, Any]:
    if not isinstance(costs, Mapping):
        raise ValueError("costs must be a table")
    if set(costs) - {"volumes", "local", "api"}:
        raise ValueError("Unknown cost assumptions")
    result = dict(costs)
    if "volumes" in result:
        volumes = result["volumes"]
        if (
            not isinstance(volumes, list)
            or not volumes
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in volumes
            )
        ):
            raise ValueError("costs.volumes must be a non-empty list of positive integers")
        if len(volumes) != len(set(volumes)):
            raise ValueError("costs.volumes must not contain duplicate values")

    cost_fields = {
        "local": {"model", "upfront_cost_usd", "operating_cost_per_sample_usd"},
        "api": {
            "model",
            "cost_per_sample_usd",
            "input_per_million_usd",
            "output_per_million_usd",
            "cached_input_per_million_usd",
        },
    }
    for kind, allowed_fields in cost_fields.items():
        assumptions = result.get(kind, {})
        if not isinstance(assumptions, Mapping) or set(assumptions) - allowed_fields:
            raise ValueError(f"Unknown {kind} cost assumptions")
        clean = dict(assumptions)
        if "model" in clean:
            _selector(clean["model"], f"costs.{kind}.model")
        for name, amount in clean.items():
            if name == "model":
                continue
            if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount < 0:
                raise ValueError(f"costs.{kind}.{name} must be finite and non-negative")
            try:
                finite = math.isfinite(amount)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"costs.{kind}.{name} must be finite and non-negative")
        result[kind] = clean
    return result


def load_config(path: Path | None) -> dict[str, Any]:
    """Load and validate a version-1 experiment TOML file.

    Data and prepared snapshot paths in TOML are resolved against the TOML
    file's directory, making the configuration portable as a project file.
    """
    if path is None:
        return {"version": 1, "experiment": {}, "models": {}, "costs": {}}
    path = Path(path).expanduser()
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    if (
        not isinstance(value.get("version"), int)
        or isinstance(value.get("version"), bool)
        or value["version"] != 1
    ):
        raise ValueError("Experiment config requires version = 1")
    if set(value) - {"version", "experiment", "models", "costs"}:
        raise ValueError("Unknown experiment config section")

    base = path.resolve().parent
    experiment = _validate_experiment(value.get("experiment", {}), base)
    model_settings = value.get("models", {})
    if not isinstance(model_settings, Mapping):
        raise ValueError("models must be a table keyed by model selector")
    validated_models: dict[str, dict[str, Any]] = {}
    canonical_model_keys: set[str] = set()
    for selector, options in model_settings.items():
        canonical = _canonical_selector(selector, "models")
        if canonical in canonical_model_keys:
            raise ValueError(f"Duplicate model settings after provider resolution: {selector!r}")
        canonical_model_keys.add(canonical)
        validated_models[str(selector)] = _validate_model_settings(options, str(selector))

    costs = _validate_costs(value.get("costs", {}))
    return {"version": 1, "experiment": experiment, "models": validated_models, "costs": costs}
