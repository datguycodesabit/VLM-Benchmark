"""Versioned experiment configuration; credentials never belong in an experiment."""

import math
import tomllib
from pathlib import Path


def load_config(path: Path | None) -> dict:
    if path is None:
        return {"version": 1, "experiment": {}, "models": {}, "costs": {}}
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    if value.get("version") != 1:
        raise ValueError("Experiment config requires version = 1")
    if set(value) - {"version", "experiment", "models", "costs"}:
        raise ValueError("Unknown experiment config section")
    allowed = {"device", "revision", "num_predict", "num_beams", "reasoning_effort", "image_detail"}
    for name, settings in value.get("models", {}).items():
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise ValueError(f"Unsupported settings for {name}; credentials must not be in TOML")
        for key in ("num_predict", "num_beams"):
            if key in settings and (
                isinstance(settings[key], bool)
                or not isinstance(settings[key], int)
                or settings[key] <= 0
            ):
                raise ValueError(f"{key} must be a positive integer for {name}")
    experiment = value.setdefault("experiment", {})
    if set(experiment) - {
        "data",
        "models",
        "limit",
        "seed",
        "layout",
        "preprocess",
        "split",
        "content_type",
        "warmup",
        "preprocessing",
    }:
        raise ValueError("Unknown experiment option")
    value.setdefault("models", {})
    value.setdefault("costs", {})
    if "models" in experiment and (
        not isinstance(experiment["models"], list)
        or not all(isinstance(m, str) and m for m in experiment["models"])
    ):
        raise ValueError("experiment.models must be a list of model selectors")
    if "preprocessing" in experiment and (
        not isinstance(experiment["preprocessing"], list) or not experiment["preprocessing"]
    ):
        raise ValueError("experiment.preprocessing must be a nonempty list")
    if set(value["costs"]) - {"volumes", "local", "api"}:
        raise ValueError("Unknown cost assumptions")
    for kind, allowed_fields in {
        "local": {"model", "upfront_cost_usd", "operating_cost_per_sample_usd"},
        "api": {
            "model",
            "cost_per_sample_usd",
            "input_per_million_usd",
            "output_per_million_usd",
            "cached_input_per_million_usd",
        },
    }.items():
        assumptions = value["costs"].get(kind, {})
        if not isinstance(assumptions, dict) or set(assumptions) - allowed_fields:
            raise ValueError(f"Unknown {kind} cost assumptions")

    def validate_numbers(item):
        if isinstance(item, dict):
            for child in item.values():
                validate_numbers(child)
        elif isinstance(item, list):
            for child in item:
                validate_numbers(child)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            if not math.isfinite(item) or item < 0:
                raise ValueError("Cost assumptions must be finite and non-negative")

    validate_numbers(value["costs"])
    return value
