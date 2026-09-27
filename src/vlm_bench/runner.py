"""Durable sequential evaluation with immutable inputs and resumable results."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .dataset import prepare_dataset
from .metrics import score
from .ollama import OllamaClient

PROMPT = (
    "You are an exact English handwriting OCR engine. Read only the visible "
    "handwriting in the image and transcribe it exactly as written. Preserve "
    "spelling, capitalization, punctuation, apostrophes, and visible errors. "
    "Return only the transcription. Never describe the image, explain your "
    "answer, translate, summarize, correct, or complete the text. Do not add "
    "labels, numbering, quotes, Markdown, or a preamble. For a single word or "
    "punctuation mark, return exactly that word or mark and nothing else. If the "
    "writing is uncertain, make your best character-level transcription instead "
    "of returning a description."
)
UPSTREAM_COMMIT = "fd4bd0ae44db0f57f7dcb0e301a0a718d3e6159f"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _lock(run_dir):
    """OS lock releases automatically even if the process is killed."""
    with (run_dir / ".lock").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("This run is already being used by another process") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_records(path: Path, repair=False):
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result, position = [], 0
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if repair and index == len(lines) - 1 and not line.endswith(b"\n"):
                # A power loss can interrupt the last append; retain the evidence.
                path.with_suffix(".interrupted-tail").write_bytes(line)
                with path.open("r+b") as handle:
                    handle.truncate(position)
                break
            raise ValueError(f"Invalid result record at line {index + 1}: {path}")
        position += len(line)
    if repair and data and not data.endswith(b"\n") and position == len(data):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return result


def _nonnegative_int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _response_timings(raw, output_limit):
    count = _nonnegative_int(raw.get("eval_count"))
    duration = _nonnegative_int(raw.get("load_duration"))
    reason = raw.get("done_reason")
    truncated = reason == "length" or (count is not None and count >= output_limit)
    return {
        "truncated": truncated,
        "truncation_unknown": reason not in ("stop", "length") and count is None,
        "load_duration_seconds": duration / 1e9 if duration is not None else None,
    }


def _sample_identity(samples):
    return [{k: s[k] for k in ("id", "reference", "hashes", "crop_bbox")} for s in samples]


def _execute(run_dir, manifest, client, progress):
    results_path = run_dir / "results.jsonl"
    existing = read_records(results_path, repair=True)
    read_records(run_dir / "warmups.jsonl", repair=True)
    valid_keys = {(m, s["id"]) for m in manifest["models"] for s in manifest["samples"]}
    completed = set()
    for row in existing:
        key = (row["model"], row["sample_id"])
        if key not in valid_keys or key in completed:
            raise ValueError("Results contain duplicate or unknown model/sample pairs")
        completed.add(key)
    results_path.touch(exist_ok=True)
    total = len(valid_keys)
    manifest["status"] = "running"
    _atomic_json(run_dir / "manifest.json", manifest)
    try:
        for model in manifest["models"]:
            pending = [s for s in manifest["samples"] if (model, s["id"]) not in completed]
            if not pending:
                continue
            try:
                if manifest["warmup"]:
                    start = time.perf_counter()
                    warmup = {"model": model, "sample_id": pending[0]["id"], "timestamp": _now()}
                    try:
                        warmup["response"] = client.transcribe(
                            model,
                            Path(pending[0]["crop_path"]),
                            manifest["prompt"],
                            manifest["options"],
                        )
                        warmup["status"] = "success"
                    except (RuntimeError, OSError, ValueError) as exc:
                        warmup.update(status="error", error=str(exc))
                        progress(f"Warmup failed for {model}: {exc}")
                    warmup["latency_seconds"] = time.perf_counter() - start
                    _append(run_dir / "warmups.jsonl", warmup)
                for sample in pending:
                    row = {
                        "model": model,
                        "sample_id": sample["id"],
                        "reference": sample["reference"],
                        "prediction": None,
                        "reference_source": sample["reference_source"],
                        "timestamp": _now(),
                        "status": "error",
                        "metrics": None,
                        "truncated": False,
                        "error": None,
                        "raw_response": None,
                        "load_duration_seconds": None,
                    }
                    start = time.perf_counter()
                    try:
                        raw = client.transcribe(
                            model,
                            Path(sample["crop_path"]),
                            manifest["prompt"],
                            manifest["options"],
                        )
                        elapsed = time.perf_counter() - start
                        row["raw_response"] = raw
                        row["prediction"] = raw["message"]["content"]
                        row.update(_response_timings(raw, manifest["options"]["num_predict"]))
                        row["metrics"] = score(row["prediction"], row["reference"])
                        row["status"] = "success"
                    except (RuntimeError, OSError, ValueError) as exc:
                        elapsed = time.perf_counter() - start
                        row["error"] = str(exc)
                    row["latency_seconds"] = elapsed
                    _append(results_path, row)
                    completed.add((model, sample["id"]))
                    progress(f"[{len(completed)}/{total}] {model} {sample['id']}: {row['status']}")
            finally:
                try:
                    client.unload(model)
                except (RuntimeError, OSError, ValueError) as exc:
                    progress(f"Could not unload {model}: {exc}")
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["updated_at"] = _now()
        _atomic_json(run_dir / "manifest.json", manifest)
        raise
    rows = read_records(results_path)
    manifest["status"] = (
        "complete" if all(r["status"] == "success" for r in rows) else "complete_with_errors"
    )
    manifest["updated_at"] = _now()
    _atomic_json(run_dir / "manifest.json", manifest)
    return run_dir


def run_benchmark(
    data_dir: Path,
    models: list[str],
    output_dir: Path = Path("runs"),
    limit=None,
    seed=42,
    base_url="http://localhost:11434",
    timeout=300,
    num_predict=4096,
    warmup=True,
    progress=print,
    client_factory=OllamaClient,
):
    if not models or len(set(models)) != len(models):
        raise ValueError("Select at least one model; model names must be unique")
    if timeout <= 0 or num_predict <= 0:
        raise ValueError("Timeout and output-token limit must be positive")
    run_dir = output_dir.resolve() / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    run_dir.mkdir(parents=True)
    try:
        with _lock(run_dir):
            samples = prepare_dataset(data_dir.resolve(), run_dir, limit=limit, seed=seed)
            with client_factory(base_url=base_url, timeout=timeout) as client:
                info = {m: client.validate_model(m) for m in models}
                manifest = {
                    "schema_version": 1,
                    "benchmark_version": __version__,
                    "upstream_commit": UPSTREAM_COMMIT,
                    "created_at": _now(),
                    "data_dir": str(data_dir.resolve()),
                    "models": models,
                    "model_info": info,
                    "ollama_version": client.version(),
                    "base_url": base_url,
                    "timeout": timeout,
                    "prompt": PROMPT,
                    "options": {"temperature": 0, "seed": seed, "num_predict": num_predict},
                    "warmup": warmup,
                    "limit": limit,
                    "seed": seed,
                    "host": {
                        "platform": platform.platform(),
                        "machine": platform.machine(),
                        "python": platform.python_version(),
                    },
                    "samples": samples,
                    "resource_measurement": None,
                }
                manifest["integrity"] = _digest(manifest)
                _atomic_json(run_dir / "manifest.json", manifest)
                progress(f"Run directory: {run_dir}")
                return _execute(run_dir, manifest, client, progress)
    except BaseException:
        if not (run_dir / "manifest.json").exists():
            # This unique directory contains only this failed setup's artifacts.
            shutil.rmtree(run_dir)
        raise


def resume_benchmark(run_dir: Path, progress=print, client_factory=OllamaClient):
    run_dir = run_dir.resolve()
    with _lock(run_dir):
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        immutable = {
            k: v for k, v in manifest.items() if k not in {"integrity", "status", "updated_at"}
        }
        if _digest(immutable) != manifest["integrity"]:
            raise ValueError("Run manifest changed; create a new run instead")
        if manifest["benchmark_version"] != __version__ or manifest["prompt"] != PROMPT:
            raise ValueError("Benchmark version or prompt changed; create a new run")
        with tempfile.TemporaryDirectory(prefix="vlm-bench-check-") as temporary:
            current = prepare_dataset(
                Path(manifest["data_dir"]),
                Path(temporary),
                limit=manifest["limit"],
                seed=manifest["seed"],
            )
            if _sample_identity(current) != _sample_identity(manifest["samples"]):
                raise ValueError("Dataset or references changed; create a new run")
        for sample in manifest["samples"]:
            crop = Path(sample["crop_path"])
            if (
                not crop.exists()
                or hashlib.sha256(crop.read_bytes()).hexdigest() != sample["hashes"]["crop"]
            ):
                raise ValueError(f"Saved crop changed or is missing: {sample['id']}")
        with client_factory(base_url=manifest["base_url"], timeout=manifest["timeout"]) as client:
            if client.version() != manifest["ollama_version"]:
                raise ValueError("Ollama version changed; create a new run")
            for model in manifest["models"]:
                info = client.validate_model(model)
                if info["digest"] != manifest["model_info"][model]["digest"]:
                    raise ValueError(f"Model digest changed: {model}; create a new run")
            return _execute(run_dir, manifest, client, progress)
