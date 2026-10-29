"""Capability declarations and import helpers for Python model adapters.

Use a selector such as ``python:my_ocr.adapter:Adapter``. The named factory is
called with no arguments. Its adapter instance declares ``capabilities`` and
implements ``load(model, *, device, settings)``, ``preprocess(image, *, track,
prompt, controls)``, ``infer(prepared, *, controls)``, and ``decode(output)``.
Optional ``identity(model, *, settings, runtime)``, ``unload(model)``,
``close()``, and ``timing_observations()`` hooks provide reproducibility and
lifecycle details. Adapter methods receive the image and prompt only; the
benchmark never passes reference text or the full sample record.

The benchmark fingerprints the selected module and factory source files. That
does not cover transitive imported helpers or dependency versions. A custom
adapter must return ``checkpoint_sha256`` (or an immutable ``revision``) from
its identity hook if it needs resume checks to detect changed checkpoint
contents; a selector/settings value alone cannot fingerprint mutable weights.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


def _string_set(value: Any, name: str) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a collection of strings or null")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a collection of strings or null") from exc
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{name} values must be non-empty strings")
    return frozenset(item.strip().lower() for item in values)


@dataclass(frozen=True)
class BackendCapabilities:
    """Backend workload/timing guarantees used for pre-run eligibility checks.

    A ``None`` supported set means the backend makes no track/sample-type
    restrictions. Empty sets reject every value.
    """

    supported_tracks: frozenset[str] | None = None
    supported_sample_types: frozenset[str] | None = None
    batching: bool = False
    streaming: bool = False
    concurrency_safe: bool = False
    timing_hooks: tuple[str, ...] = ()
    usage_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("supported_tracks", "supported_sample_types"):
            value = getattr(self, name)
            if value is not None:
                normalized = _string_set(value, name)
                object.__setattr__(self, name, normalized)
        for name in ("batching", "streaming", "concurrency_safe"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} capability must be a boolean")
        for name in ("timing_hooks", "usage_fields"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
                raise TypeError(f"{name} must be a sequence of strings")
            if any(not isinstance(item, str) or not item.strip() for item in value):
                raise ValueError(f"{name} values must be non-empty strings")
            object.__setattr__(self, name, tuple(dict.fromkeys(item.strip() for item in value)))

    def supports(self, track: str | None, sample_type: str | None) -> bool:
        if self.supported_tracks is not None and (
            not isinstance(track, str) or track.lower() not in self.supported_tracks
        ):
            return False
        if self.supported_sample_types is not None and (
            not isinstance(sample_type, str)
            or sample_type.lower() not in self.supported_sample_types
        ):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_tracks": (
                sorted(self.supported_tracks) if self.supported_tracks is not None else None
            ),
            "supported_sample_types": (
                sorted(self.supported_sample_types)
                if self.supported_sample_types is not None
                else None
            ),
            "batching": self.batching,
            "streaming": self.streaming,
            "concurrency_safe": self.concurrency_safe,
            "timing_hooks": list(self.timing_hooks),
            "usage_fields": list(self.usage_fields),
        }

    @classmethod
    def from_value(cls, value: Any) -> "BackendCapabilities":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Adapter capabilities must be BackendCapabilities or a mapping")
        allowed = {
            "supported_tracks",
            "supported_sample_types",
            "batching",
            "streaming",
            "concurrency_safe",
            "timing_hooks",
            "usage_fields",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("Unknown adapter capability fields: " + ", ".join(sorted(unknown)))
        return cls(**dict(value))


def selector_parts(selector: str) -> tuple[str, str]:
    """Parse a ``module:Factory`` selector without accepting filesystem paths."""
    if not isinstance(selector, str) or selector.count(":") != 1:
        raise ValueError("Python backend selector must have the form python:package.module:Factory")
    module_name, factory_name = selector.rsplit(":", 1)
    if not module_name or not factory_name:
        raise ValueError("Python backend selector must include both a module and factory name")
    if any(not part.isidentifier() for part in module_name.split(".")):
        raise ValueError("Python backend module must be an importable dotted package name")
    if not factory_name.isidentifier():
        raise ValueError("Python backend factory name must be a Python identifier")
    return module_name, factory_name


def import_adapter_factory(selector: str) -> tuple[Any, ModuleType, str]:
    """Import the selected local factory and fingerprint its Python source."""
    module_name, factory_name = selector_parts(selector)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import custom adapter module {module_name!r} ({type(exc).__name__})"
        ) from None
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise RuntimeError(
            f"Custom adapter factory {factory_name!r} was not found in module {module_name!r}"
        )
    source_paths: set[Path] = set()
    for candidate in (
        getattr(module, "__file__", None),
        inspect.getsourcefile(factory),
    ):
        if isinstance(candidate, str):
            path = Path(candidate)
            if path.is_file():
                source_paths.add(path.resolve())
    if source_paths:
        digest = hashlib.sha256()
        for path in sorted(source_paths, key=lambda item: str(item)):
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        source_hash = digest.hexdigest()
    else:
        try:
            source = inspect.getsource(factory).encode("utf-8")
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot fingerprint custom adapter factory {selector!r}; install it from source"
            ) from exc
        source_hash = hashlib.sha256(source).hexdigest()
    return factory, module, source_hash


def safe_adapter_identity(value: Any) -> dict[str, Any]:
    """Keep a narrow, JSON-safe identity allowlist from third-party adapters."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Adapter identity hook must return a mapping")
    allowed = {
        "model",
        "revision",
        "checkpoint_sha256",
        "adapter_version",
        "precision",
        "task",
        "weights_sha256",
    }
    safe: dict[str, Any] = {}
    for key in sorted(allowed & set(value)):
        item = value[key]
        if item is None or isinstance(item, (str, bool, int)):
            safe[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            safe[key] = item
    return safe
