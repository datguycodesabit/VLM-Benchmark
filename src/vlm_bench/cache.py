"""Content-addressed, opt-in response cache for deterministic inference."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CACHE_SCHEMA_VERSION = 1
_KEY_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-fA-F]{7,64}$")
_FLOATING_IDENTITIES = {"latest", "main", "master", "stable", "nightly", "production"}
_SAFE_PROVIDER_DETAIL_FIELDS = {
    "actual_device",
    "device",
    "model_revision",
    "provider",
    "revision",
    "version",
}


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Cache values must be finite JSON data") from exc


def _valid_cache_key(key: Any) -> str:
    if not isinstance(key, str) or not _SHA256.fullmatch(key):
        raise ValueError("cache key must be a lowercase SHA-256 hex digest")
    return key


def _crop_hash(sample: Mapping[str, Any]) -> str | None:
    hashes = sample.get("hashes")
    value = hashes.get("crop") if isinstance(hashes, Mapping) else sample.get("crop_sha256")
    if isinstance(value, str) and _SHA256.fullmatch(value):
        return value
    return None


def _provider(selector: str, model_info: Mapping[str, Any]) -> str:
    provider = model_info.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip().lower()
    if ":" in selector:
        return selector.split(":", 1)[0].strip().lower()
    return "ollama"


def _immutable_identity(model_info: Mapping[str, Any]) -> dict[str, str]:
    identity: dict[str, str] = {}
    digest = model_info.get("digest")
    if isinstance(digest, str):
        normalized = digest.strip().lower()
        if normalized.startswith("sha256:"):
            normalized = normalized.removeprefix("sha256:")
        if _SHA256.fullmatch(normalized):
            identity["digest"] = normalized

    revision = model_info.get("revision")
    if (
        isinstance(revision, str)
        and revision.strip().lower() not in _FLOATING_IDENTITIES
        and _COMMIT.fullmatch(revision.strip())
    ):
        identity["revision"] = revision.strip().lower()

    checkpoint_hash = model_info.get("checkpoint_sha256")
    if isinstance(checkpoint_hash, str):
        normalized = checkpoint_hash.strip().lower()
        if _SHA256.fullmatch(normalized):
            identity["checkpoint_sha256"] = normalized
    return identity


def cache_key(
    sample: Mapping[str, Any],
    selector: str,
    model_info: Mapping[str, Any],
    prompt: str,
    effective_controls: Mapping[str, Any],
    repeated_measurement: bool = False,
) -> tuple[str | None, str | None]:
    """Return a deterministic cache key, or an explicit bypass reason.

    Only a frozen crop digest, an immutable model revision/digest, a prompt, and
    the complete effective controls can define a reusable response.
    """
    if not isinstance(sample, Mapping):
        raise TypeError("sample must be a mapping")
    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("selector must be a non-empty string")
    if not isinstance(model_info, Mapping):
        raise TypeError("model_info must be a mapping")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(effective_controls, Mapping):
        raise TypeError("effective_controls must be a mapping")
    if not isinstance(repeated_measurement, bool):
        raise TypeError("repeated_measurement must be a boolean")
    if repeated_measurement:
        return None, "repeated_measurement"

    temperature = effective_controls.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            return None, "invalid_temperature"
        if not math.isfinite(temperature):
            return None, "invalid_temperature"
        if temperature < 0 or temperature > 2:
            return None, "invalid_temperature"
        if temperature > 0:
            return None, "stochastic_temperature"
    do_sample = effective_controls.get("do_sample")
    if do_sample is True:
        return None, "stochastic_sampling"
    if do_sample not in (None, False):
        return None, "invalid_sampling_control"

    selector_provider = selector.split(":", 1)[0].strip().lower() if ":" in selector else "ollama"
    reported_provider = model_info.get("provider")
    if (
        isinstance(reported_provider, str)
        and reported_provider.strip()
        and reported_provider.strip().lower() != selector_provider
    ):
        return None, "model_identity_provider_mismatch"
    provider = _provider(selector, model_info)
    if provider == "ollama":
        if temperature is None:
            return None, "missing_deterministic_temperature"
        seed = effective_controls.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            return None, "missing_fixed_seed"
    identity = _immutable_identity(model_info)
    if not identity:
        return None, "missing_immutable_model_identity"
    image_hash = _crop_hash(sample)
    if image_hash is None:
        return None, "missing_crop_hash"
    if not prompt.strip():
        return None, "missing_prompt"

    payload = {
        "key_version": _KEY_VERSION,
        "crop_sha256": image_hash,
        "selector": selector,
        "provider": provider,
        "model_identity": identity,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "effective_controls": dict(effective_controls),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest(), None


def _safe_response(raw_response: Any) -> dict[str, Any]:
    if not isinstance(raw_response, Mapping):
        raise TypeError("cached response must be a mapping")
    if raw_response.get("status") not in (None, "success"):
        raise ValueError("Only successful responses can be cached")
    message = raw_response.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("Only successful responses with string message.content can be cached")
    response: dict[str, Any] = {"message": {"content": message["content"]}}
    model = raw_response.get("model")
    if isinstance(model, str) and model:
        response["model"] = model
    # Keep only the minimal provider diagnostics needed to reconstruct output
    # truncation on cache hits. Timing and billing fields (including load
    # duration) deliberately remain uncached.
    for field in ("done_reason", "finish_reason"):
        value = raw_response.get(field)
        if isinstance(value, str) and value and len(value) <= 128:
            response[field] = value
    eval_count = raw_response.get("eval_count")
    if isinstance(eval_count, int) and not isinstance(eval_count, bool) and eval_count >= 0:
        response["eval_count"] = eval_count
    details = raw_response.get("provider_details")
    if isinstance(details, Mapping):
        safe_details = {
            key: value
            for key, value in details.items()
            if key in _SAFE_PROVIDER_DETAIL_FIELDS
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        if safe_details:
            # Also rejects NaN/Infinity and protects the on-disk JSON contract.
            _canonical_json(safe_details)
            response["provider_details"] = safe_details
    _canonical_json(response)
    return response


class ResponseCache:
    """Integrity-checked response entries stored as immutable JSON files."""

    def __init__(self, root: str | Path) -> None:
        if not isinstance(root, (str, Path)) or not str(root).strip():
            raise ValueError("cache root must be a non-empty path")
        self.root = Path(root).expanduser().resolve()

    def _entry_path(self, key: str, *, create: bool) -> Path:
        key = _valid_cache_key(key)
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(f"Cache root is not a directory: {self.root}")
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.exists():
            return self.root / key[:2] / f"{key}.json"
        bucket = self.root / key[:2]
        if create:
            bucket.mkdir(parents=True, exist_ok=True)
        if bucket.exists() and not bucket.is_dir():
            raise ValueError(f"Cache bucket is not a directory: {bucket}")
        if bucket.is_symlink():
            raise ValueError(f"Cache bucket must not be a symlink: {bucket}")
        path = bucket / f"{key}.json"
        if path.is_symlink():
            raise ValueError(f"Cache entry must not be a symlink: {path}")
        try:
            path.resolve(strict=False).relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Cache entry escapes cache root: {path}") from exc
        return path

    @staticmethod
    def _entry(key: str, response: dict[str, Any]) -> dict[str, Any]:
        body = {"schema_version": _CACHE_SCHEMA_VERSION, "key": key, "response": response}
        return {**body, "integrity": hashlib.sha256(_canonical_json(body)).hexdigest()}

    def lookup(self, key: str) -> dict[str, Any] | None:
        key = _valid_cache_key(key)
        path = self._entry_path(key, create=False)
        if not path.exists():
            return None
        if not path.is_file():
            raise ValueError(f"Cache entry is not a regular file: {path}")
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cache entry is unreadable or corrupt: {path}") from exc
        if not isinstance(entry, dict) or set(entry) != {
            "schema_version",
            "key",
            "response",
            "integrity",
        }:
            raise ValueError(f"Cache entry has an invalid envelope: {path}")
        if entry.get("schema_version") != _CACHE_SCHEMA_VERSION or entry.get("key") != key:
            raise ValueError(f"Cache entry version or key mismatch: {path}")
        body = {name: entry[name] for name in ("schema_version", "key", "response")}
        expected_integrity = hashlib.sha256(_canonical_json(body)).hexdigest()
        if entry.get("integrity") != expected_integrity:
            raise ValueError(f"Cache entry integrity check failed: {path}")
        response = _safe_response(entry.get("response"))
        if response != entry["response"]:
            raise ValueError(f"Cache entry contains disallowed response data: {path}")
        return response

    def store(self, key: str, raw_response: Mapping[str, Any]) -> None:
        """Store a response once, refusing conflicting writes for the same key."""
        key = _valid_cache_key(key)
        response = _safe_response(raw_response)
        path = self._entry_path(key, create=True)
        if path.exists():
            if self.lookup(key) != response:
                raise ValueError(f"Refusing to replace a different response for cache key {key}")
            return

        entry_bytes = (
            json.dumps(
                self._entry(key, response),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{key}.", suffix=".tmp", dir=path.parent, delete=False
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(entry_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                if self.lookup(key) != response:
                    raise ValueError(
                        f"Refusing to replace a different response for cache key {key}"
                    )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
