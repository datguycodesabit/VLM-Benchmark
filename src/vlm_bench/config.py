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
    "strict_research",
    "protocol",
    "formula_rendering",
    "max_retries",
    "concurrency",
    "max_requests",
    "max_spend_usd",
    "cache_dir",
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
    "temperature",
    "seed",
    "reasoning_effort",
    "image_detail",
}
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
_SPLITS = {"train", "validation", "test"}
_LAYOUTS = {"auto", "paired", "iam-words", "iam-lines", "iam-forms"}
_PROFILES = {"original", "enhanced"}
_CONTENT_TYPES = {"prose", "equation", "word"}
_PROTOCOLS = {"document-disjoint", "writer-disjoint"}
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
    if "temperature" in result:
        temperature = result["temperature"]
        try:
            finite_temperature = math.isfinite(temperature)
        except (TypeError, OverflowError):
            finite_temperature = False
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not finite_temperature
            or not 0 <= temperature <= 2
        ):
            raise ValueError(f"temperature for {context} must be finite and between 0 and 2")
    if "seed" in result and (
        isinstance(result["seed"], bool)
        or not isinstance(result["seed"], int)
        or not 0 <= result["seed"] <= _MAX_SEED
    ):
        raise ValueError(f"seed for {context} must be an integer from 0 to {_MAX_SEED}")
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
        if provider == "external" and canonical_options:
            raise ValueError("Settings are not supported for provider 'external'")
        if provider == "trocr":
            inapplicable = set(canonical_options) & {
                "reasoning_effort",
                "image_detail",
                "temperature",
                "seed",
            }
        elif provider in {"openai", "chatgpt"}:
            inapplicable = set(canonical_options) & {
                "device",
                "revision",
                "num_beams",
                "temperature",
                "seed",
            }
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
    if "cache_dir" in result:
        result["cache_dir"] = _path_from_config(result["cache_dir"], "cache_dir", base)
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
    if "strict_research" in result and not isinstance(result["strict_research"], bool):
        raise ValueError("experiment.strict_research must be a boolean")
    if "formula_rendering" in result and not isinstance(result["formula_rendering"], bool):
        raise ValueError("experiment.formula_rendering must be a boolean")
    if "protocol" in result and (
        not isinstance(result["protocol"], str) or result["protocol"] not in _PROTOCOLS
    ):
        raise ValueError("experiment.protocol must be document-disjoint or writer-disjoint")
    if "max_retries" in result and (
        isinstance(result["max_retries"], bool)
        or not isinstance(result["max_retries"], int)
        or result["max_retries"] < 0
    ):
        raise ValueError("experiment.max_retries must be a non-negative integer")
    if "concurrency" in result:
        _positive_integer(result["concurrency"], "experiment.concurrency")
    if "max_requests" in result:
        _positive_integer(result["max_requests"], "experiment.max_requests")
    if "max_spend_usd" in result:
        amount = result["max_spend_usd"]
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount <= 0
        ):
            raise ValueError("experiment.max_spend_usd must be a positive finite number")
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


def _optional_prompt(condition: Mapping[str, Any], name: str) -> str | None:
    if name not in condition:
        return None
    value = condition[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"suite condition {name} must be a non-empty string")
    return value


def _validate_suite_settings(settings: Any, models: Sequence[str], context: str) -> None:
    if not isinstance(settings, Mapping):
        raise ValueError(f"{context} must be a table keyed by selected model selector")
    selected = {f"{provider}:{model}" for provider, model in _selector_list(models, "models")}
    seen: set[str] = set()
    for selector in settings:
        canonical = _canonical_selector(selector, context)
        if canonical in seen:
            raise ValueError(f"Duplicate settings for {context} selector {selector!r}")
        seen.add(canonical)
        if canonical not in selected:
            raise ValueError(f"{context} contains settings for unselected model {selector!r}")


def _validate_suite(
    suite: Any,
    experiment: Mapping[str, Any],
    global_settings: Mapping[str, Any],
    base: Path,
) -> dict[str, Any]:
    if not isinstance(suite, Mapping):
        raise ValueError("version = 2 requires a [suite] table")
    unknown = set(suite) - {"repeats", "conditions"}
    if unknown:
        raise ValueError("Unknown suite option: " + ", ".join(sorted(unknown)))

    source_options = {
        "data",
        "limit",
        "seed",
        "layout",
        "preprocess",
        "preprocessing",
        "split",
        "content_type",
    }
    conflicts = set(experiment) & source_options
    if conflicts:
        raise ValueError(
            "Experiment suites require frozen prepared snapshots and cannot use source selection: "
            + ", ".join(sorted(conflicts))
        )
    if "models" in experiment:
        raise ValueError(
            "Each suite condition must define its own models; remove experiment.models"
        )

    suite_repeats = suite.get("repeats", 1)
    _positive_integer(suite_repeats, "suite.repeats")
    conditions = suite.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("suite.conditions must be a non-empty list of conditions")

    condition_fields = {
        "name",
        "models",
        "prepared",
        "repeats",
        "prose_prompt",
        "math_prompt",
        "settings",
        "warmup",
        "strict_research",
        "protocol",
        "formula_rendering",
    }
    names: set[str] = set()
    validated_conditions = []
    for index, raw_condition in enumerate(conditions):
        context = f"suite.conditions[{index}]"
        if not isinstance(raw_condition, Mapping):
            raise ValueError(f"{context} must be a table")
        unknown = set(raw_condition) - condition_fields
        if unknown:
            invalid_selection = unknown & source_options
            if invalid_selection:
                raise ValueError(
                    f"{context} must use a separate prepared snapshot; cannot set: "
                    + ", ".join(sorted(invalid_selection))
                )
            raise ValueError(f"Unknown {context} option: " + ", ".join(sorted(unknown)))

        name = raw_condition.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(f"{context}.name must be a non-empty trimmed string")
        if name in names:
            raise ValueError(f"suite condition names must be unique: {name!r}")
        names.add(name)

        models = raw_condition.get("models")
        parsed_models = _selector_list(models, f"{context}.models")
        external = [model for provider, model in parsed_models if provider == "external"]
        if external:
            raise ValueError(
                f"{context}.models cannot include external prediction selectors; "
                "external systems are import-only"
            )

        configured_prepared = raw_condition.get("prepared", experiment.get("prepared"))
        if configured_prepared is None:
            raise ValueError(f"{context}.prepared or experiment.prepared is required")
        if "prepared" in raw_condition:
            prepared = _path_from_config(configured_prepared, f"{context}.prepared", base)
        else:
            prepared = configured_prepared

        repeats = raw_condition.get("repeats", suite_repeats)
        _positive_integer(repeats, f"{context}.repeats")

        condition_settings = raw_condition.get("settings", {})
        _validate_suite_settings(condition_settings, models, f"{context}.settings")
        global_selected = validate_settings(models, global_settings)
        local_selected = validate_settings(models, condition_settings)
        merged_settings = {}
        for model in models:
            merged = dict(global_selected.get(model, {}))
            merged.update(local_selected.get(model, {}))
            if merged:
                merged_settings[model] = merged
        validated_settings = validate_settings(models, merged_settings)

        warmup = raw_condition.get("warmup", experiment.get("warmup", True))
        strict_research = raw_condition.get(
            "strict_research", experiment.get("strict_research", False)
        )
        protocol = raw_condition.get("protocol", experiment.get("protocol", "document-disjoint"))
        formula_rendering = raw_condition.get(
            "formula_rendering", experiment.get("formula_rendering", False)
        )
        if not isinstance(warmup, bool):
            raise ValueError(f"{context}.warmup must be a boolean")
        if not isinstance(strict_research, bool):
            raise ValueError(f"{context}.strict_research must be a boolean")
        if not isinstance(protocol, str) or protocol not in _PROTOCOLS:
            raise ValueError(f"{context}.protocol must be document-disjoint or writer-disjoint")
        if not isinstance(formula_rendering, bool):
            raise ValueError(f"{context}.formula_rendering must be a boolean")

        validated_conditions.append(
            {
                "name": name,
                "models": list(models),
                "prepared": prepared,
                "repeats": repeats,
                "settings": validated_settings,
                "prose_prompt": _optional_prompt(raw_condition, "prose_prompt"),
                "math_prompt": _optional_prompt(raw_condition, "math_prompt"),
                "warmup": warmup,
                "strict_research": strict_research,
                "protocol": protocol,
                "formula_rendering": formula_rendering,
            }
        )

    return {"version": 2, "repeats": suite_repeats, "conditions": validated_conditions}


def load_config(path: Path | None) -> dict[str, Any]:
    """Load and validate a versioned experiment TOML file.

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
        or value["version"] not in {1, 2}
    ):
        raise ValueError("Experiment config requires version = 1 or 2")
    config_version = value["version"]
    allowed_sections = {"version", "experiment", "models", "costs"}
    if config_version == 2:
        allowed_sections.add("suite")
    if set(value) - allowed_sections:
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
        provider, _ = _selector(selector, "models")
        validated_options = _validate_model_settings(options, str(selector))
        if provider == "external" and validated_options:
            raise ValueError("Settings are not supported for provider 'external'")
        validated_models[str(selector)] = validated_options

    costs = _validate_costs(value.get("costs", {}))
    result = {
        "version": config_version,
        "experiment": experiment,
        "models": validated_models,
        "costs": costs,
    }
    if config_version == 2:
        result["suite"] = _validate_suite(value.get("suite"), experiment, validated_models, base)
    return result
