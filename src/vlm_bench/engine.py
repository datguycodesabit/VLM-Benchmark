"""Provider-neutral execution over frozen benchmark inputs."""

import hashlib
import json
import platform
import time
import uuid
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from . import __version__
from .dataset import prepare_dataset
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


def preview(
    data_dir,
    models,
    *,
    layout="auto",
    preprocess="original",
    split=None,
    content_type=None,
    limit=None,
    seed=42,
    base_url="http://localhost:11434",
    timeout=300,
    settings=None,
    backend_factory=_factory,
):
    import tempfile

    with tempfile.TemporaryDirectory(prefix="vlm-preview-") as folder:
        samples = prepare_dataset(
            data_dir,
            Path(folder),
            limit=limit,
            seed=seed,
            layout=layout,
            preprocess=preprocess,
            split=split,
            content_type=content_type,
        )
    choices = []
    for selector in models:
        provider, name = _parse(selector)
        with backend_factory(
            provider, base_url, timeout, (settings or {}).get(selector, {})
        ) as backend:
            info = backend.validate_model(name)
        choices.append(
            {
                "selector": selector,
                "execution": "cloud" if provider in {"openai", "chatgpt"} else "local",
                "model_info": info,
                "eligible_samples": sum(_eligible(provider, s) for s in samples),
            }
        )
    return {
        "sample_count": len(samples),
        "sample_ids": [s["id"] for s in samples],
        "layout": layout,
        "preprocess": preprocess,
        "models": choices,
        "inference": False,
    }


def run(
    data_dir,
    models,
    output_dir=Path("runs"),
    *,
    limit=None,
    seed=42,
    layout="auto",
    preprocess="original",
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
    if not models or len(set(models)) != len(models):
        raise ValueError("Select unique model selectors")
    if timeout <= 0 or num_predict <= 0:
        raise ValueError("Timeout and token limit must be positive")
    parsed = [_parse(m) for m in models]
    if len(set(parsed)) != len(parsed):
        raise ValueError("Duplicate model selectors after provider resolution")
    directory = output_dir.resolve() / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    directory.mkdir(parents=True)
    with _lock(directory):
        samples = prepare_dataset(
            data_dir,
            directory,
            limit=limit,
            seed=seed,
            layout=layout,
            preprocess=preprocess,
            split=split,
            content_type=content_type,
        )
        info = {}
        for selector, (provider, name) in zip(models, parsed):
            with backend_factory(
                provider, base_url, timeout, (settings or {}).get(selector, {})
            ) as backend:
                info[selector] = backend.validate_model(name)
        manifest = {
            "schema_version": 2,
            "benchmark_version": __version__,
            "created_at": _now(),
            "models": models,
            "model_info": info,
            "samples": samples,
            "data_dir": str(data_dir.resolve()),
            "layout": layout,
            "preprocess": preprocess,
            "split": split,
            "content_type": content_type,
            "limit": limit,
            "seed": seed,
            "base_url": base_url,
            "timeout": timeout,
            "prompt": PROMPT,
            "math_prompt": MATH_PROMPT,
            "options": {"num_predict": num_predict, "temperature": 0, "seed": seed},
            "settings": settings or {},
            "costs": costs or {},
            "warmup": warmup,
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
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
                                )
                                row.update(_response_timings(raw, options["num_predict"]))
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
