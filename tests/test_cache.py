from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_bench.cache import ResponseCache, cache_key
from vlm_bench.runner import _response_timings

_CROP_HASH = "a" * 64
_MODEL_DIGEST = "sha256:" + "b" * 64


def _inputs():
    return (
        {"id": "sample-1", "hashes": {"crop": _CROP_HASH}},
        "ollama:vision-model",
        {"provider": "ollama", "model": "vision-model", "digest": _MODEL_DIGEST},
        "Read this image.",
        {"temperature": 0, "seed": 42, "num_predict": 512},
    )


def test_cache_key_changes_with_sample_model_prompt_or_effective_controls():
    inputs = _inputs()
    key, reason = cache_key(*inputs)
    assert reason is None
    assert key is not None

    changed = []
    sample = dict(inputs[0], hashes={"crop": "c" * 64})
    changed.append((sample, *inputs[1:]))
    changed.append((inputs[0], "ollama:other-model", *inputs[2:]))
    changed.append(
        (inputs[0], inputs[1], {**inputs[2], "digest": "sha256:" + "d" * 64}, *inputs[3:])
    )
    changed.append((*inputs[:3], "Read this image differently.", inputs[4]))
    changed.append((*inputs[:4], {**inputs[4], "num_predict": 256}))

    for case in changed:
        changed_key, changed_reason = cache_key(*case)
        assert changed_reason is None
        assert changed_key != key


@pytest.mark.parametrize(
    ("updates", "repeated", "expected_reason"),
    [
        ({}, True, "repeated_measurement"),
        ({"temperature": 0.2}, False, "stochastic_temperature"),
        ({"do_sample": True}, False, "stochastic_sampling"),
        ({"seed": None}, False, "missing_fixed_seed"),
        ({"temperature": None}, False, "missing_deterministic_temperature"),
    ],
)
def test_cache_key_returns_explicit_bypass_reasons(updates, repeated, expected_reason):
    inputs = _inputs()
    controls = {**inputs[4], **updates}

    key, reason = cache_key(*inputs[:4], controls, repeated_measurement=repeated)

    assert key is None
    assert reason == expected_reason


def test_cache_key_bypasses_missing_crop_or_immutable_model_identity():
    inputs = _inputs()
    missing_crop = {"id": "sample-1"}
    assert cache_key(missing_crop, *inputs[1:])[1] == "missing_crop_hash"
    unversioned = {"provider": "ollama", "model": "vision-model"}
    assert cache_key(inputs[0], inputs[1], unversioned, *inputs[3:])[1] == (
        "missing_immutable_model_identity"
    )


def test_response_cache_returns_only_safe_response_fields(tmp_path: Path):
    key, reason = cache_key(*_inputs())
    assert key is not None and reason is None
    cache = ResponseCache(tmp_path / "responses")
    assert cache.lookup(key) is None

    cache.store(
        key,
        {
            "message": {"content": "recognized text", "role": "assistant"},
            "model": "vision-model:latest",
            "done_reason": "length",
            "eval_count": 512,
            "load_duration": 1_200_000_000,
            "provider_details": {
                "device": "gpu",
                "revision": "b" * 40,
                "api_key": "must-not-persist",
                "raw_request": {"authorization": "must-not-persist"},
            },
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "latency_seconds": 1.25,
            "inference_latency_seconds": 1.1,
        },
    )

    assert cache.lookup(key) == {
        "message": {"content": "recognized text"},
        "model": "vision-model:latest",
        "done_reason": "length",
        "eval_count": 512,
        "provider_details": {"device": "gpu", "revision": "b" * 40},
    }
    assert _response_timings(cache.lookup(key), output_limit=512) == {
        "truncated": True,
        "truncation_unknown": False,
        "load_duration_seconds": None,
    }
    serialized = next((tmp_path / "responses").rglob("*.json")).read_text(encoding="utf-8")
    assert "must-not-persist" not in serialized
    assert '"usage"' not in serialized
    assert '"latency_seconds"' not in serialized
    assert '"load_duration"' not in serialized


def test_response_cache_preserves_finish_reason_for_truncation_diagnostics(tmp_path: Path):
    key, reason = cache_key(*_inputs())
    assert key is not None and reason is None
    cache = ResponseCache(tmp_path / "responses")

    cache.store(
        key,
        {
            "message": {"content": "recognized text"},
            "finish_reason": "stop",
            "eval_count": 12,
            "provider_details": {"generation_seconds": 0.5},
        },
    )

    cached = cache.lookup(key)
    assert cached is not None
    assert cached["finish_reason"] == "stop"
    assert cached["eval_count"] == 12
    assert "generation_seconds" not in cached.get("provider_details", {})


def test_response_cache_fails_on_corruption_and_conflicting_replacement(tmp_path: Path):
    key, reason = cache_key(*_inputs())
    assert key is not None and reason is None
    cache = ResponseCache(tmp_path / "responses")
    response = {"message": {"content": "first"}}
    cache.store(key, response)
    with pytest.raises(ValueError, match="different response"):
        cache.store(key, {"message": {"content": "second"}})

    entry_path = next((tmp_path / "responses").rglob("*.json"))
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["response"]["message"]["content"] = "tampered"
    entry_path.write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check failed"):
        cache.lookup(key)


def test_response_cache_rejects_non_response_payload(tmp_path: Path):
    key, reason = cache_key(*_inputs())
    assert key is not None and reason is None
    cache = ResponseCache(tmp_path / "responses")
    with pytest.raises(ValueError, match="Only successful responses"):
        cache.store(key, {"status": "error", "error": "failed"})
    with pytest.raises(ValueError, match="Only successful responses"):
        cache.store(key, {"status": "error", "message": {"content": "partial"}})


def test_response_cache_rejects_a_non_directory_root(tmp_path: Path):
    root = tmp_path / "cache-root"
    root.write_text("not a cache directory", encoding="utf-8")
    key, reason = cache_key(*_inputs())
    assert key is not None and reason is None

    with pytest.raises(ValueError, match="Cache root is not a directory"):
        ResponseCache(root).lookup(key)
