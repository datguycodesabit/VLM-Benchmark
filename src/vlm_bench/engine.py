"""Provider-neutral execution over frozen benchmark inputs."""

import hashlib
import json
import math
import platform
import tempfile
import time
import uuid
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .metrics import score
from .runner import (
    PROMPT,
    _append,
    _atomic_json,
    _digest,
    _lock,
    _now,
    _response_timings,
    read_records,
)

MATH_PROMPT = "Transcribe only the visible handwritten equation as LaTeX. Preserve symbols and errors. Return only LaTeX without delimiters, explanations, or Markdown."


def _factory(provider, base_url, timeout, settings):
    from .backends import create_backend

    return create_backend(provider, base_url=base_url, timeout=timeout, settings=settings)


def _parse(selector):
    from .backends import parse_model

    return parse_model(selector)


def _identity(info):
    return {
        key: info.get(key)
        for key in ("digest", "revision", "checkpoint_sha256", "model", "device", "version")
    }


def _eligible(provider, sample):
    if provider != "trocr":
        return True
    return (sample.get("content_type") or "prose") in {"prose", "line", "prose-line", "word"} and (
        sample.get("sample_type") or "line"
    ) != "page"


SCORING_VERSION = "1"


def _validate(models, settings, *, timeout, num_predict, seed):
    from .config import validate_settings

    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError("Select at least one model")
    parsed = [_parse(m) for m in models]
    if len(set(parsed)) != len(parsed):
        raise ValueError("Duplicate model selectors after provider resolution")
    normalized_settings = validate_settings(models, settings if settings is not None else {})
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("Timeout must be a positive finite number")
    if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
        raise ValueError("Token limit must be a positive integer")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("Seed must be an integer from 0 to 4294967295")
    return parsed, normalized_settings


def _validate_url(base_url):
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Ollama base URL must be an HTTP or HTTPS address")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Base URLs must not contain credentials, query parameters, or fragments")


def _controls(provider, options, settings, info):
    requested = dict(options, **settings)
    supported = {
        "ollama": {"num_predict", "temperature", "seed"},
        "trocr": {"num_predict", "num_beams", "device", "revision"},
        "openai": {"num_predict", "image_detail", "reasoning_effort"},
        "chatgpt": {"image_detail", "reasoning_effort"},
    }[provider]
    effective = {k: v for k, v in requested.items() if k in supported}
    if provider == "trocr":
        effective.update(
            device=info.get("device"),
            revision=info.get("revision"),
            num_beams=settings.get("num_beams", 1),
        )
        effective["num_predict"] = min(effective["num_predict"], 4096)
        effective.update(do_sample=False, processor_use_fast=False)
    elif provider in {"chatgpt", "openai"}:
        effective.setdefault("image_detail", "auto")
        if provider == "openai":
            effective["max_output_tokens"] = effective.pop("num_predict")
    return {
        "requested": requested,
        "effective": effective,
        "unsupported": sorted(set(requested) - supported),
    }


def _environment():
    dependencies = {}
    for name in (
        "Pillow",
        "httpx",
        "openpyxl",
        "torch",
        "transformers",
        "huggingface-hub",
        "keyring",
        "PyJWT",
    ):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "dependencies": dependencies,
    }


@contextmanager
def _inputs(data_dir, prepared, selection):
    from .snapshot import freeze, load

    if prepared is not None:
        if data_dir is not None or any(v is not None for v in selection.values()):
            raise ValueError(
                "--prepared cannot be combined with source data, selection, or preprocessing options"
            )
        path = Path(prepared).resolve()
        yield path, load(path)
        return
    if data_dir is None:
        data_dir = Path("data")
    options = {k: v for k, v in selection.items() if v is not None}
    with tempfile.TemporaryDirectory(prefix="vlm-inputs-") as folder:
        path = Path(folder) / "benchmark"
        freeze(Path(data_dir), path, **options)
        yield path, load(path)


def preview(
    data_dir=None,
    models=None,
    *,
    prepared=None,
    layout=None,
    preprocess=None,
    split=None,
    content_type=None,
    limit=None,
    seed=None,
    base_url="http://localhost:11434",
    timeout=300,
    num_predict=4096,
    settings=None,
    backend_factory=_factory,
):
    _validate_url(base_url)
    parsed, settings = _validate(
        models, settings, timeout=timeout, num_predict=num_predict, seed=seed
    )
    selection = dict(
        layout=layout,
        preprocess=preprocess,
        split=split,
        content_type=content_type,
        limit=limit,
        seed=seed,
    )
    with _inputs(data_dir, prepared, selection) as (_, snapshot):
        samples = snapshot["samples"]
        choices = []
        options = {"num_predict": num_predict, "temperature": 0, "seed": snapshot.get("seed", 42)}
        for selector, (provider, name) in zip(models, parsed):
            configured = (settings or {}).get(selector, {})
            with backend_factory(provider, base_url, timeout, configured) as backend:
                info = backend.validate_model(name)
            choices.append(
                {
                    "selector": selector,
                    "provider": provider,
                    "execution": "cloud" if provider in {"openai", "chatgpt"} else "local",
                    "model_info": info,
                    "controls": _controls(provider, options, configured, info),
                    "eligible_samples": sum(_eligible(provider, s) for s in samples),
                    "unsupported_sample_ids": [
                        s["id"] for s in samples if not _eligible(provider, s)
                    ],
                }
            )
        return {
            "sample_count": len(samples),
            "sample_ids": [s["id"] for s in samples],
            "benchmark_fingerprint": snapshot["benchmark_fingerprint"],
            "verification_status_counts": dict(
                Counter(
                    s.get("verification_status")
                    or (s.get("metadata") or {}).get("verification_status")
                    or "unknown"
                    for s in samples
                )
            ),
            "layout": snapshot["layout"],
            "preprocess": snapshot["preprocess"],
            "scoring_version": SCORING_VERSION,
            "models": choices,
            "inference": False,
            "determinism_note": "The sampling seed freezes selection; it does not guarantee identical model predictions.",
        }


def run(
    data_dir=None,
    models=None,
    output_dir=Path("runs"),
    *,
    prepared=None,
    limit=None,
    seed=None,
    layout=None,
    preprocess=None,
    split=None,
    content_type=None,
    base_url="http://localhost:11434",
    timeout=300,
    num_predict=4096,
    warmup=True,
    settings=None,
    costs=None,
    backend_factory=_factory,
    progress=print,
):
    from .snapshot import copy_inputs

    _validate_url(base_url)
    parsed, settings = _validate(
        models, settings, timeout=timeout, num_predict=num_predict, seed=seed
    )
    if not isinstance(warmup, bool):
        raise ValueError("Warmup must be a boolean")
    selection = dict(
        layout=layout,
        preprocess=preprocess,
        split=split,
        content_type=content_type,
        limit=limit,
        seed=seed,
    )
    with _inputs(data_dir, prepared, selection) as (snapshot_path, snapshot):
        info = {}
        for selector, (provider, name) in zip(models, parsed):
            with backend_factory(
                provider, base_url, timeout, (settings or {}).get(selector, {})
            ) as backend:
                info[selector] = backend.validate_model(name)
        directory = Path(output_dir).resolve() / (
            datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        directory.mkdir(parents=True)
        with _lock(directory):
            samples, copied = copy_inputs(snapshot_path, directory)
            options = {"num_predict": num_predict, "temperature": 0, "seed": copied.get("seed", 42)}
            manifest = {
                "schema_version": 2,
                "benchmark_version": __version__,
                "created_at": _now(),
                "benchmark_fingerprint": copied["benchmark_fingerprint"],
                "scoring_version": SCORING_VERSION,
                "models": list(models),
                "model_info": info,
                "samples": samples,
                "data_dir": str(Path(data_dir).resolve()) if data_dir is not None else None,
                "prepared": str(Path(prepared).resolve()) if prepared is not None else None,
                "layout": copied["layout"],
                "preprocess": copied["preprocess"],
                "split": copied.get("split"),
                "content_type": copied.get("content_type"),
                "limit": copied.get("limit"),
                "seed": copied.get("seed", 42),
                "base_url": base_url,
                "timeout": timeout,
                "prompt": PROMPT,
                "math_prompt": MATH_PROMPT,
                "prompt_hashes": {
                    "prose": hashlib.sha256(PROMPT.encode()).hexdigest(),
                    "equation": hashlib.sha256(MATH_PROMPT.encode()).hexdigest(),
                },
                "options": options,
                "settings": settings or {},
                "costs": costs or {},
                "warmup": warmup,
                "provider_controls": {
                    selector: _controls(
                        provider, options, (settings or {}).get(selector, {}), info[selector]
                    )
                    for selector, (provider, _) in zip(models, parsed)
                },
                "host": _environment(),
            }
            manifest["integrity"] = _digest(manifest)
            _atomic_json(directory / "manifest.json", manifest)
            progress(f"Run directory: {directory}")
            return _execute(directory, manifest, backend_factory, progress)


def resume(directory, *, retry_failed=False, backend_factory=_factory, progress=print):
    directory = directory.resolve()
    with _lock(directory):
        manifest = json.loads((directory / "manifest.json").read_text())
        immutable = {
            k: v for k, v in manifest.items() if k not in {"integrity", "status", "updated_at"}
        }
        if _digest(immutable) != manifest["integrity"]:
            raise ValueError("Run manifest changed; create a new run")
        if manifest.get("scoring_version", SCORING_VERSION) != SCORING_VERSION:
            raise ValueError("Scoring rules changed; create a new run instead of mixing scores")
        if manifest.get("benchmark_fingerprint"):
            from .snapshot import fingerprint

            if (
                fingerprint(manifest["samples"], manifest["preprocess"])
                != manifest["benchmark_fingerprint"]
            ):
                raise ValueError("Frozen benchmark fingerprint does not match the run manifest")
        if manifest["prompt"] != PROMPT or manifest.get("math_prompt") != MATH_PROMPT:
            raise ValueError("Prompt changed; create a new run")
        # Resume consumes saved inputs, not a newly sampled live dataset.
        for sample in manifest["samples"]:
            path = Path(sample["crop_path"])
            if (
                not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != sample["hashes"]["crop"]
            ):
                raise ValueError(f"Saved image changed: {sample['id']}")
        if retry_failed:
            rows = read_records(directory / "results.jsonl", repair=True)
            failures = [r for r in rows if r["status"] not in {"success", "unsupported"}]
            if failures:
                for row in failures:
                    _append(directory / "attempts.jsonl", row)
                temporary = directory / "results.retry.tmp"
                temporary.write_text(
                    "".join(
                        json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r not in failures
                    )
                )
                temporary.replace(directory / "results.jsonl")
        return _execute(directory, manifest, backend_factory, progress, verify=True)


def _execute(directory, manifest, backend_factory, progress, verify=False):
    from .backends import BackendPaused

    rows_path = directory / "results.jsonl"
    existing = read_records(rows_path, repair=True)
    expected = {(m, s["id"]) for m in manifest["models"] for s in manifest["samples"]}
    completed = {(r["model"], r["sample_id"]) for r in existing}
    if len(completed) != len(existing) or not completed.issubset(expected):
        raise ValueError("Duplicate or unknown result pairs")
    rows_path.touch(exist_ok=True)
    manifest["status"] = "running"
    _atomic_json(directory / "manifest.json", manifest)
    paused = False
    try:
        for selector in manifest["models"]:
            provider, name = _parse(selector)
            pending = [s for s in manifest["samples"] if (selector, s["id"]) not in completed]
            if not pending:
                continue
            settings = dict(manifest["settings"].get(selector, {}))
            if provider == "trocr" and manifest["model_info"][selector].get("revision"):
                settings["revision"] = manifest["model_info"][selector]["revision"]
            with ExitStack() as stack:
                backend = stack.enter_context(
                    backend_factory(provider, manifest["base_url"], manifest["timeout"], settings)
                )
                current = backend.validate_model(name)
                if _identity(current) != _identity(manifest["model_info"][selector]):
                    raise ValueError(f"Model identity changed: {selector}; create a new run")
                options = dict(manifest["options"], **settings)
                try:
                    eligible = [s for s in pending if _eligible(provider, s)]
                    if eligible and manifest["warmup"] and provider in {"ollama", "trocr"}:
                        start = time.perf_counter()
                        try:
                            raw = backend.transcribe(
                                name,
                                Path(eligible[0]["crop_path"]),
                                MATH_PROMPT
                                if eligible[0].get("content_type") in {"equation", "math"}
                                else PROMPT,
                                options,
                            )
                            warmup = {"model": selector, "status": "success", "response": raw}
                        except (ValueError, RuntimeError, OSError) as exc:
                            warmup = {"model": selector, "status": "error", "error": str(exc)}
                        warmup["sample_id"] = eligible[0]["id"]
                        warmup["latency_seconds"] = time.perf_counter() - start
                        _append(directory / "warmups.jsonl", warmup)
                    for sample in pending:
                        row = {
                            "model": selector,
                            "provider": provider,
                            "sample_id": sample["id"],
                            "reference": sample["reference"],
                            "reference_source": sample["reference_source"],
                            "content_type": sample.get("content_type") or "prose",
                            "source_document": sample.get("source_document"),
                            "prediction": None,
                            "metrics": None,
                            "error": None,
                            "timestamp": _now(),
                            "status": "error",
                            "truncated": False,
                        }
                        start = time.perf_counter()
                        if not _eligible(provider, sample):
                            row.update(
                                status="unsupported",
                                error="TrOCR requires a single prose-line image",
                            )
                        else:
                            try:
                                math = sample.get("content_type") in {"equation", "math"}
                                raw = backend.transcribe(
                                    name,
                                    Path(sample["crop_path"]),
                                    MATH_PROMPT if math else PROMPT,
                                    options,
                                )
                                row.update(
                                    raw_response=raw,
                                    prediction=raw["message"]["content"],
                                    usage=raw.get("usage"),
                                    actual_model=raw.get("model", name),
                                    actual_device=(raw.get("provider_details") or {}).get("device"),
                                )
                                row.update(
                                    _response_timings(
                                        raw,
                                        float("inf")
                                        if provider == "chatgpt"
                                        else options["num_predict"],
                                    )
                                )
                                if math:
                                    from .research import score_equation

                                    row["metrics"] = score_equation(
                                        row["prediction"], row["reference"]
                                    )
                                else:
                                    row["metrics"] = score(row["prediction"], row["reference"])
                                row["status"] = "success"
                            except BackendPaused:
                                paused = True
                                progress(f"{selector}: usage unavailable; resume this run later")
                                break
                            except (RuntimeError, OSError, ValueError) as exc:
                                row["error"] = str(exc)
                        row["latency_seconds"] = time.perf_counter() - start
                        if row["status"] == "success":
                            load = row.get("load_duration_seconds")
                            if isinstance(load, (int, float)) and not isinstance(load, bool):
                                row["inference_latency_seconds"] = max(
                                    0, row["latency_seconds"] - load
                                )
                            elif provider in {"openai", "chatgpt"}:
                                row["inference_latency_seconds"] = row["latency_seconds"]
                        _append(rows_path, row)
                        completed.add((selector, sample["id"]))
                        progress(
                            f"[{len(completed)}/{len(expected)}] {selector} {sample['id']}: {row['status']}"
                        )
                finally:
                    backend.unload(name)
    except BaseException:
        manifest.update(status="interrupted", updated_at=_now())
        _atomic_json(directory / "manifest.json", manifest)
        raise
    rows = read_records(rows_path)
    manifest.update(
        status="paused"
        if paused
        else "complete"
        if all(r["status"] in {"success", "unsupported"} for r in rows)
        else "complete_with_errors",
        updated_at=_now(),
    )
    _atomic_json(directory / "manifest.json", manifest)
    return directory
