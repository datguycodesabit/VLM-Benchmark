"""Resumable orchestration for repeated standalone benchmark passes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import runner

_FINISHED = {"complete", "complete_with_errors"}
_RESUMABLE = {"paused", "interrupted", "running", "incomplete"}
_SAMPLE_METADATA_FIELDS = (
    "writer_id",
    "source_document",
    "split",
    "content_type",
    "sample_type",
    "difficulty",
    "verification_status",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    runner._atomic_json(path, state)


def _run_path(value: Any, experiment_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Experiment run paths must be non-empty strings")
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        resolved = path.resolve()
    elif path.is_absolute():
        # Old CLI versions stored absolute paths. If the enclosing experiment
        # folder was moved, keep the run's relative suffix under its new root.
        parts = path.parts
        try:
            marker = parts.index(experiment_dir.name)
        except ValueError as exc:
            raise FileNotFoundError(f"Saved experiment run is missing: {path}") from exc
        resolved = experiment_dir.joinpath(*parts[marker + 1 :]).resolve()
    else:
        resolved = (experiment_dir / path).resolve()
    try:
        resolved.relative_to(experiment_dir)
    except ValueError as exc:
        raise ValueError(f"Saved experiment run escapes its directory: {resolved}") from exc
    if not resolved.is_dir():
        raise FileNotFoundError(f"Saved experiment run is missing: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_run_manifest(run_dir: Path, *, rebase_crops: bool = True) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read experiment run manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError(f"Experiment run requires a schema-v2 manifest: {run_dir}")
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"integrity", "status", "updated_at"}
    }
    if (
        not isinstance(manifest.get("integrity"), str)
        or runner._digest(immutable) != manifest["integrity"]
    ):
        raise ValueError(f"Run manifest integrity check failed: {run_dir}")

    from .snapshot import fingerprint

    samples = manifest.get("samples")
    preprocess = manifest.get("preprocess")
    if not isinstance(samples, list) or not isinstance(preprocess, str):
        raise ValueError(f"Run manifest has no frozen samples: {run_dir}")
    if fingerprint(samples, preprocess) != manifest.get("benchmark_fingerprint"):
        raise ValueError(f"Run benchmark fingerprint does not match its samples: {run_dir}")

    changed_paths = False
    crop_root = (run_dir / "crops").resolve()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(f"Run manifest contains an invalid sample: {run_dir}")
        crop_value = sample.get("crop_path")
        if not isinstance(crop_value, str) or not crop_value:
            raise ValueError(f"Run sample {sample.get('id')!r} has no crop path")
        crop = Path(crop_value).expanduser()
        if crop.exists():
            crop = crop.resolve()
        elif rebase_crops:
            old_parts = crop.parts
            crop_markers = [index for index, part in enumerate(old_parts) if part == "crops"]
            if not crop_markers:
                raise FileNotFoundError(f"Saved image is missing: {crop}")
            crop = crop_root.joinpath(*old_parts[crop_markers[-1] + 1 :]).resolve()
            changed_paths = True
        else:
            raise FileNotFoundError(f"Saved image is missing: {crop}")
        try:
            crop.relative_to(crop_root)
        except ValueError as exc:
            raise ValueError(f"Saved image escapes the run crop directory: {crop}") from exc
        expected_hash = (sample.get("hashes") or {}).get("crop")
        if not crop.is_file() or _sha256(crop) != expected_hash:
            raise ValueError(f"Saved image changed for sample {sample.get('id')!r}")
        if str(crop) != sample["crop_path"]:
            sample["crop_path"] = str(crop)
            changed_paths = True

    if changed_paths:
        manifest["integrity"] = runner._digest(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"integrity", "status", "updated_at"}
            }
        )
        runner._atomic_json(manifest_path, manifest)
    return manifest


def _snapshot_from_run(
    first_run: str | os.PathLike[str], experiment_dir: str | os.PathLike[str]
) -> Path:
    """Create or validate one portable v2 snapshot from a run's frozen inputs."""
    run_dir = Path(first_run).expanduser().resolve()
    root = Path(experiment_dir).expanduser().resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("First pass must be stored inside the experiment directory") from exc
    manifest = _read_run_manifest(run_dir)
    output = root / "prepared-inputs"
    from .snapshot import _manifest_integrity, fingerprint, load

    expected_fingerprint = manifest["benchmark_fingerprint"]
    if output.exists():
        saved = load(output)
        if saved.get("benchmark_fingerprint") != expected_fingerprint:
            raise ValueError("Saved experiment prepared snapshot does not match pass 1")
        return output

    output.mkdir(parents=True)
    (output / "crops").mkdir()
    snapshot_samples = []
    run_crop_root = (run_dir / "crops").resolve()
    try:
        for sample in manifest["samples"]:
            source = Path(sample["crop_path"])
            relative = source.relative_to(run_crop_root)
            target = output / "crops" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if _sha256(target) != sample["hashes"]["crop"]:
                raise ValueError(f"Copied crop hash mismatch for sample {sample['id']!r}")
            frozen = {
                "id": sample["id"],
                "crop_path": (Path("crops") / relative).as_posix(),
                "reference": sample["reference"],
                "reference_source": sample.get("reference_source"),
                "metadata": sample.get("metadata") or {},
                "preprocess": manifest["preprocess"],
                "crop_bbox": sample.get("crop_bbox"),
                "hashes": dict(sample["hashes"]),
            }
            for field in _SAMPLE_METADATA_FIELDS:
                frozen[field] = sample.get(field)
            snapshot_samples.append(frozen)
        snapshot_manifest = {
            "manifest_version": 2,
            "created_at": _now(),
            "layout": manifest.get("layout", "auto"),
            "preprocess": manifest["preprocess"],
            "seed": manifest.get("seed", 42),
            "limit": manifest.get("limit"),
            "split": manifest.get("split"),
            "content_type": manifest.get("content_type"),
            "sample_ids": [sample["id"] for sample in snapshot_samples],
            "samples": snapshot_samples,
            "provenance": {"source": "resumed-experiment-pass", "pass": 1},
        }
        snapshot_manifest["benchmark_fingerprint"] = fingerprint(
            snapshot_samples, manifest["preprocess"]
        )
        if snapshot_manifest["benchmark_fingerprint"] != expected_fingerprint:
            raise ValueError("Copied experiment inputs do not preserve the benchmark fingerprint")
        snapshot_manifest["integrity"] = _manifest_integrity(snapshot_manifest)
        runner._atomic_json(output / "dataset-manifest.json", snapshot_manifest)
        load(output)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output


def _same_pass_identity(reference: dict[str, Any], current: dict[str, Any]) -> bool:
    fields = (
        "planned_passes",
        "benchmark_fingerprint",
        "models",
        "settings",
        "base_url",
        "timeout",
        "options",
        "prompt_hashes",
        "strict_research",
        "protocol",
        "formula_rendering",
        "warmup",
        "scoring_policy",
        "evaluation_schema_version",
        "measurement_profile",
        "measurement_schema_version",
        "model_capabilities",
        "host",
        "source_audit",
        "validation_scope",
        "preprocess",
        "layout",
        "split",
        "content_type",
        "limit",
        "seed",
        "concurrency",
        "cache_dir",
        "max_retries",
        "max_requests",
        "max_spend_usd",
        "costs",
        "formula_renderer",
        "task_metrics_version",
        "research_protocol_version",
        "prompt_version",
    )
    if any(reference.get(field) != current.get(field) for field in fields):
        return False
    for selector, info in reference.get("model_info", {}).items():
        other = current.get("model_info", {}).get(selector)
        if other is None:
            return False
        identity_keys = (
            "digest",
            "revision",
            "checkpoint_sha256",
            "model",
            "device",
            "requested_device",
            "runtime_identity",
            "adapter_sha256",
            "adapter_module",
            "adapter_factory",
            "adapter_version",
            "version",
        )
        if {key: info.get(key) for key in identity_keys} != {
            key: other.get(key) for key in identity_keys
        }:
            return False
    return True


def _state_for(directory: Path) -> tuple[Path, dict[str, Any]]:
    state_path = directory / "experiment.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read experiment state: {state_path}") from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Unsupported experiment state schema")
    planned = state.get("planned_passes")
    if isinstance(planned, bool) or not isinstance(planned, int) or planned <= 0:
        raise ValueError("Experiment planned_passes must be a positive integer")
    runs = state.get("runs")
    if not isinstance(runs, list) or len(runs) > planned:
        raise ValueError("Experiment run list does not match planned_passes")
    return state_path, state


def _resume_experiment_locked(
    directory: str | os.PathLike[str],
    *,
    retry_failed: bool = False,
    max_requests: int | None = None,
    max_spend_usd: float | None = None,
    progress: Callable[[str], None] = print,
    run_fn: Callable | None = None,
    resume_fn: Callable | None = None,
    backend_factory: Callable | None = None,
) -> Path:
    """Continue saved passes and create any missing passes from their frozen inputs."""
    from . import engine

    experiment_dir = Path(directory).expanduser().resolve()
    state_path, state = _state_for(experiment_dir)
    run_fn = run_fn or engine.run
    resume_fn = resume_fn or engine.resume
    paths = [_run_path(value, experiment_dir) for value in state["runs"]]
    if len(set(paths)) != len(paths):
        raise ValueError("Experiment contains duplicate run paths")
    manifests = [_read_run_manifest(path) for path in paths]
    planned = state["planned_passes"]
    for index, manifest in enumerate(manifests, start=1):
        if manifest.get("repetition") != index or manifest.get("suite_condition") != "baseline":
            raise ValueError(f"Saved run does not match experiment pass {index}")
        if manifest.get("planned_passes", 1) != planned:
            raise ValueError("Saved run planned_passes does not match experiment.json")
        if index > 1 and not _same_pass_identity(manifests[0], manifest):
            raise ValueError(f"Saved pass {index} changed the frozen experiment identity")
    if not manifests and planned:
        raise ValueError(
            "No pass was saved, so there are no frozen inputs to resume; rerun the original command"
        )
    reference = manifests[0]

    def save_paths() -> None:
        state["runs"] = [os.path.relpath(path, experiment_dir) for path in paths]
        if state.get("status") not in _FINISHED:
            state["status"] = "incomplete"
        _write_state(state_path, state)

    for index, (run_dir, manifest) in enumerate(zip(paths, manifests, strict=True), start=1):
        status = manifest.get("status")
        retry_completed_errors = retry_failed and status == "complete_with_errors"
        if status in _FINISHED and not retry_completed_errors:
            continue
        if status not in _RESUMABLE and not retry_completed_errors:
            raise ValueError(f"Experiment pass {index} has non-resumable status {status!r}")
        progress(f"Resuming experiment pass {index}: {run_dir}")
        kwargs: dict[str, Any] = {
            "retry_failed": retry_failed,
            "max_requests": max_requests,
            "max_spend_usd": max_spend_usd,
            "progress": progress,
        }
        if backend_factory is not None:
            kwargs["backend_factory"] = backend_factory
        resume_fn(run_dir, **kwargs)
        manifests[index - 1] = _read_run_manifest(run_dir)
        if manifests[index - 1].get("status") == "paused":
            save_paths()
            return experiment_dir

    prepared = _snapshot_from_run(paths[0], experiment_dir)
    for repetition in range(len(paths) + 1, planned + 1):
        execution = reference.get("execution_controls") or {}
        controls = {
            "prepared": prepared,
            "settings": reference.get("settings", {}),
            "costs": reference.get("costs", {}),
            "base_url": reference.get("base_url", "http://localhost:11434"),
            "timeout": reference.get("timeout", 300),
            "num_predict": (reference.get("options") or {}).get("num_predict", 4096),
            "warmup": reference.get("warmup", True),
            "max_retries": execution.get("max_retries", reference.get("max_retries", 2)),
            "concurrency": execution.get("concurrency", reference.get("concurrency", 1)),
            "max_requests": execution.get("max_requests", reference.get("max_requests")),
            "max_spend_usd": execution.get("max_spend_usd", reference.get("max_spend_usd")),
            "cache_dir": execution.get("cache_dir", reference.get("cache_dir")),
            "strict_research": reference.get("strict_research", False),
            "protocol": reference.get("protocol", "document-disjoint"),
            "formula_rendering": reference.get("formula_rendering", False),
            "frozen_source_audit": reference.get("source_audit"),
            "frozen_validation_scope": reference.get("validation_scope"),
            "prose_prompt": reference.get("prompt"),
            "math_prompt": reference.get("math_prompt"),
            "suite_condition": "baseline",
            "repetition": repetition,
            "repeated_measurement": True,
            "planned_passes": planned,
            "progress": progress,
        }
        if max_requests is not None or max_spend_usd is not None:
            controls["invocation_controls"] = {
                "max_requests": max_requests
                if max_requests is not None
                else execution.get("max_requests", reference.get("max_requests")),
                "max_spend_usd": max_spend_usd
                if max_spend_usd is not None
                else execution.get("max_spend_usd", reference.get("max_spend_usd")),
            }
        if "scoring_policy" in reference:
            controls["scoring_policy"] = reference["scoring_policy"]
        if "measurement_profile" in reference:
            controls["measurement_profile"] = reference["measurement_profile"]
        if backend_factory is not None:
            controls["backend_factory"] = backend_factory

        captured_run: list[Path] = []

        def record_progress(message: str) -> None:
            progress(message)
            if message.startswith("Run directory: "):
                run_path = Path(message.removeprefix("Run directory: ")).expanduser().resolve()
                if run_path not in paths:
                    paths.append(run_path)
                    captured_run.append(run_path)
                    save_paths()
                    candidate = _read_run_manifest(run_path)
                    if (
                        candidate.get("repetition") != repetition
                        or candidate.get("suite_condition") != "baseline"
                    ):
                        raise ValueError(f"New run does not match experiment pass {repetition}")
                    if not _same_pass_identity(reference, candidate):
                        raise ValueError(
                            f"New pass {repetition} changed the frozen experiment identity"
                        )

        controls["progress"] = record_progress
        run_dir = Path(run_fn(None, list(reference["models"]), experiment_dir / "runs", **controls))
        run_dir = run_dir.resolve()
        if run_dir not in paths:
            paths.append(run_dir)
            save_paths()
        current = _read_run_manifest(run_dir)
        if current.get("repetition") != repetition or current.get("suite_condition") != "baseline":
            raise ValueError(f"New run does not match experiment pass {repetition}")
        if not _same_pass_identity(reference, current):
            raise ValueError(f"New pass {repetition} changed the frozen experiment identity")
        manifests.append(current)
        if current.get("status") == "paused":
            save_paths()
            return experiment_dir

    state["status"] = (
        "complete_with_errors"
        if any(manifest.get("status") == "complete_with_errors" for manifest in manifests)
        else "complete"
    )
    save_paths()
    return experiment_dir


def resume_experiment(
    directory: str | os.PathLike[str],
    *,
    retry_failed: bool = False,
    max_requests: int | None = None,
    max_spend_usd: float | None = None,
    progress: Callable[[str], None] = print,
    run_fn: Callable | None = None,
    resume_fn: Callable | None = None,
    backend_factory: Callable | None = None,
) -> Path:
    """Resume an experiment while holding its orchestration lock."""
    experiment_dir = Path(directory).expanduser().resolve()
    with runner._lock(experiment_dir):
        return _resume_experiment_locked(
            experiment_dir,
            retry_failed=retry_failed,
            max_requests=max_requests,
            max_spend_usd=max_spend_usd,
            progress=progress,
            run_fn=run_fn,
            resume_fn=resume_fn,
            backend_factory=backend_factory,
        )
