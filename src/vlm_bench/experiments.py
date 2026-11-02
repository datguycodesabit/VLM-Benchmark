"""Resumable orchestration for repeated standalone benchmark passes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
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
_PLAN_KWARGS = {
    "base_url",
    "timeout",
    "num_predict",
    "warmup",
    "settings",
    "costs",
    "max_retries",
    "concurrency",
    "max_requests",
    "max_spend_usd",
    "cache_dir",
    "strict_research",
    "protocol",
    "formula_rendering",
    "prose_prompt",
    "math_prompt",
    "scoring_policy",
    "measurement_profile",
    "planned_passes",
    "frozen_source_audit",
    "frozen_validation_scope",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    runner._atomic_json(path, state)


def _plan_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plan_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plan_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Experiment plan values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Experiment plan contains unsupported value: {type(value).__name__}")


def initialize_experiment_plan(
    directory: str | os.PathLike[str],
    *,
    planned_passes: int,
    prepared: str | os.PathLike[str],
    models: list[str] | tuple[str, ...],
    run_kwargs: dict[str, Any],
) -> Path:
    """Persist the frozen inputs and execution plan before starting pass one.

    The returned snapshot lives under the experiment root and is safe to use if
    the process stops before an engine run directory has been committed.
    """
    experiment_dir = Path(directory).expanduser().resolve()
    if (
        isinstance(planned_passes, bool)
        or not isinstance(planned_passes, int)
        or planned_passes <= 0
    ):
        raise ValueError("planned_passes must be a positive integer")
    if (
        not isinstance(models, (list, tuple))
        or not models
        or any(not isinstance(model, str) or not model for model in models)
    ):
        raise ValueError("Experiment plan models must be non-empty strings")
    if len(set(models)) != len(models):
        raise ValueError("Experiment plan models must be unique")
    if not isinstance(run_kwargs, dict):
        raise ValueError("Experiment plan run_kwargs must be an object")
    unsupported = set(run_kwargs) - _PLAN_KWARGS
    if unsupported:
        raise ValueError("Unsupported experiment plan options: " + ", ".join(sorted(unsupported)))
    normalized_kwargs = _plan_jsonable(run_kwargs)
    if "planned_passes" in normalized_kwargs:
        run_planned = normalized_kwargs["planned_passes"]
        if (
            isinstance(run_planned, bool)
            or not isinstance(run_planned, int)
            or run_planned != planned_passes
        ):
            raise ValueError("Experiment plan planned_passes does not match run_kwargs")
    normalized_kwargs["planned_passes"] = planned_passes

    from .snapshot import load as load_snapshot

    experiment_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = Path(prepared).expanduser().resolve()
    source = load_snapshot(source_snapshot)
    try:
        source_relative = source_snapshot.relative_to(experiment_dir)
    except ValueError:
        source_relative = None
    destination = experiment_dir / "prepared-inputs"
    with runner._lock(experiment_dir):
        if source_relative is not None:
            plan_snapshot = source_snapshot
        else:
            if destination.exists():
                saved = load_snapshot(destination)
                if saved.get("benchmark_fingerprint") != source.get("benchmark_fingerprint"):
                    raise ValueError(
                        "Saved experiment plan snapshot does not match the selected inputs"
                    )
            else:
                staging = Path(tempfile.mkdtemp(prefix=".prepared-inputs.tmp-", dir=experiment_dir))
                staging.rmdir()
                try:
                    shutil.copytree(source_snapshot, staging)
                    copied = load_snapshot(staging)
                    if copied.get("benchmark_fingerprint") != source.get("benchmark_fingerprint"):
                        raise ValueError(
                            "Copied experiment plan snapshot changed its benchmark fingerprint"
                        )
                    os.replace(staging, destination)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
            plan_snapshot = destination

        plan = {
            "schema_version": 1,
            "planned_passes": planned_passes,
            "prepared": plan_snapshot.relative_to(experiment_dir).as_posix(),
            "benchmark_fingerprint": source["benchmark_fingerprint"],
            "models": list(models),
            "run_kwargs": normalized_kwargs,
        }
        state_path = experiment_dir / "experiment.json"
        if state_path.exists():
            state_path_value, state = _state_for(experiment_dir)
            if state.get("planned_passes") != planned_passes:
                raise ValueError("Experiment state planned_passes does not match the frozen plan")
            existing_plan = state.get("plan")
            if existing_plan is not None and (
                existing_plan != plan or state.get("plan_integrity") != runner._digest(plan)
            ):
                raise ValueError("Experiment already has a different frozen plan")
            state_path = state_path_value
        else:
            state = {"schema_version": 1, "planned_passes": planned_passes, "runs": []}
        state["plan"] = plan
        state["plan_integrity"] = runner._digest(plan)
        _write_state(state_path, state)
    return plan_snapshot


def _run_path(value: Any, experiment_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Experiment run paths must be non-empty strings")
    path = Path(value.replace("\\", "/")).expanduser()
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


def _validated_plan(directory: Path, state: dict[str, Any]) -> tuple[dict[str, Any], Path] | None:
    plan = state.get("plan")
    if plan is None:
        return None
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("Unsupported frozen experiment plan schema")
    if state.get("plan_integrity") != runner._digest(plan):
        raise ValueError("Experiment plan integrity check failed")
    planned = state.get("planned_passes")
    plan_planned = plan.get("planned_passes")
    if (
        isinstance(plan_planned, bool)
        or not isinstance(plan_planned, int)
        or plan_planned != planned
    ):
        raise ValueError("Experiment plan planned_passes does not match experiment.json")
    models = plan.get("models")
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(model, str) or not model for model in models)
        or len(models) != len(set(models))
    ):
        raise ValueError("Experiment plan models are invalid")
    run_kwargs = plan.get("run_kwargs")
    if not isinstance(run_kwargs, dict) or set(run_kwargs) - _PLAN_KWARGS:
        raise ValueError("Experiment plan run_kwargs are invalid")
    run_planned = run_kwargs.get("planned_passes", planned)
    if isinstance(run_planned, bool) or not isinstance(run_planned, int) or run_planned != planned:
        raise ValueError("Experiment plan run_kwargs planned_passes does not match experiment.json")
    prepared_value = plan.get("prepared")
    if not isinstance(prepared_value, str) or not prepared_value:
        raise ValueError("Experiment plan has no prepared snapshot path")
    prepared_relative = Path(prepared_value)
    if prepared_relative.is_absolute() or ".." in prepared_relative.parts:
        raise ValueError("Experiment plan snapshot path must be relative to the experiment")
    prepared = (directory / prepared_relative).resolve()
    try:
        prepared.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError("Experiment plan snapshot path escapes its directory") from exc
    from .snapshot import load as load_snapshot

    snapshot = load_snapshot(prepared)
    if snapshot.get("benchmark_fingerprint") != plan.get("benchmark_fingerprint"):
        raise ValueError("Experiment plan snapshot fingerprint changed")
    return plan, prepared


def _verify_plan_run(plan: dict[str, Any], manifest: dict[str, Any]) -> None:
    from . import engine

    kwargs = plan["run_kwargs"]
    if manifest.get("models") != plan["models"]:
        raise ValueError("Saved run models do not match the frozen experiment plan")
    if manifest.get("benchmark_fingerprint") != plan["benchmark_fingerprint"]:
        raise ValueError("Saved run inputs do not match the frozen experiment plan")
    if manifest.get("planned_passes", 1) != plan["planned_passes"]:
        raise ValueError("Saved run planned_passes does not match the frozen experiment plan")

    expected_manifest_fields = (
        "settings",
        "costs",
        "base_url",
        "timeout",
        "warmup",
        "strict_research",
        "protocol",
        "formula_rendering",
        "max_retries",
        "concurrency",
        "max_requests",
        "max_spend_usd",
    )
    for field in expected_manifest_fields:
        if field in kwargs and manifest.get(field) != kwargs[field]:
            raise ValueError(f"Saved run {field} does not match the frozen experiment plan")
    if "cache_dir" in kwargs:
        expected_cache = (
            str(Path(kwargs["cache_dir"]).expanduser().resolve())
            if kwargs["cache_dir"] is not None
            else None
        )
        if manifest.get("cache_dir") != expected_cache:
            raise ValueError("Saved run cache_dir does not match the frozen experiment plan")
    if (
        "num_predict" in kwargs
        and (manifest.get("options") or {}).get("num_predict") != kwargs["num_predict"]
    ):
        raise ValueError("Saved run num_predict does not match the frozen experiment plan")
    for source, target in (
        ("frozen_source_audit", "source_audit"),
        ("frozen_validation_scope", "validation_scope"),
    ):
        if source in kwargs and manifest.get(target) != kwargs[source]:
            raise ValueError(f"Saved run {target} does not match the frozen experiment plan")

    prose_prompt, math_prompt = engine._resolve_prompts(
        kwargs.get("prose_prompt"), kwargs.get("math_prompt")
    )
    if manifest.get("prompt") != prose_prompt or manifest.get("math_prompt") != math_prompt:
        raise ValueError("Saved run prompts do not match the frozen experiment plan")
    if "scoring_policy" in kwargs:
        expected_scoring = engine._scoring_policy(kwargs["scoring_policy"]).to_dict()
        if manifest.get("scoring_policy") != expected_scoring:
            raise ValueError("Saved run scoring policy does not match the frozen experiment plan")
        if manifest.get("evaluation_schema_version") != engine.EVALUATION_SCHEMA_VERSION:
            raise ValueError(
                "Saved run evaluation schema does not match the frozen experiment plan"
            )
    if "measurement_profile" in kwargs:
        expected_measurement = engine._measurement_profile(
            kwargs["measurement_profile"], concurrency=kwargs.get("concurrency", 1)
        )
        if manifest.get("measurement_profile") != expected_measurement:
            raise ValueError(
                "Saved run measurement profile does not match the frozen experiment plan"
            )
        if manifest.get("measurement_schema_version") != expected_measurement["schema_version"]:
            raise ValueError(
                "Saved run measurement schema does not match the frozen experiment plan"
            )


def _recover_untracked_runs(
    directory: Path,
    paths: list[Path],
    manifests: list[dict[str, Any]],
    *,
    planned_passes: int,
    plan: dict[str, Any] | None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    runs_root = directory / "runs"
    if not runs_root.is_dir():
        return paths, manifests
    by_repetition = {
        manifest.get("repetition"): (path, manifest)
        for path, manifest in zip(paths, manifests, strict=True)
    }
    recovered: dict[int, tuple[Path, dict[str, Any]]] = {}
    for candidate in runs_root.iterdir():
        if not candidate.is_dir() or not (candidate / "manifest.json").is_file():
            continue
        manifest = _read_run_manifest(candidate)
        repetition = manifest.get("repetition")
        if manifest.get("suite_condition") != "baseline" or repetition is None:
            continue
        if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition <= 0:
            continue
        if repetition > planned_passes:
            raise ValueError(f"Unexpected saved experiment pass {repetition} exceeds the plan")
        if repetition in by_repetition:
            if candidate.resolve() != by_repetition[repetition][0].resolve():
                raise ValueError(
                    f"Multiple saved run directories claim experiment pass {repetition}"
                )
            continue
        if manifest.get("planned_passes", 1) != planned_passes:
            raise ValueError(
                f"Saved experiment pass {repetition} planned_passes does not match the plan"
            )
        if plan is not None:
            _verify_plan_run(plan, manifest)
        if repetition in recovered:
            raise ValueError(
                f"Multiple unrecorded run directories claim experiment pass {repetition}"
            )
        recovered[repetition] = (candidate.resolve(), manifest)
    if not recovered:
        return paths, manifests
    combined = dict(by_repetition)
    combined.update(recovered)
    ordered_repetitions = sorted(combined)
    if ordered_repetitions != list(range(1, max(ordered_repetitions) + 1)):
        raise ValueError("Saved experiment passes are not contiguous starting from pass 1")
    return (
        [combined[number][0] for number in ordered_repetitions],
        [combined[number][1] for number in ordered_repetitions],
    )


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
    plan_info = _validated_plan(experiment_dir, state)
    plan = plan_info[0] if plan_info else None
    paths = [_run_path(value, experiment_dir) for value in state["runs"]]
    if len(set(paths)) != len(paths):
        raise ValueError("Experiment contains duplicate run paths")
    manifests = [_read_run_manifest(path) for path in paths]
    planned = state["planned_passes"]
    paths, manifests = _recover_untracked_runs(
        experiment_dir,
        paths,
        manifests,
        planned_passes=planned,
        plan=plan,
    )
    for index, manifest in enumerate(manifests, start=1):
        if manifest.get("repetition") != index or manifest.get("suite_condition") != "baseline":
            raise ValueError(f"Saved run does not match experiment pass {index}")
        if manifest.get("planned_passes", 1) != planned:
            raise ValueError("Saved run planned_passes does not match experiment.json")
        if plan is not None:
            _verify_plan_run(plan, manifest)
        if index > 1 and not _same_pass_identity(manifests[0], manifest):
            raise ValueError(f"Saved pass {index} changed the frozen experiment identity")
    if not manifests and plan is None:
        raise ValueError(
            "No pass was saved, so there are no frozen inputs to resume; rerun the original command"
        )
    reference = manifests[0] if manifests else None

    def save_paths() -> None:
        state["runs"] = [Path(os.path.relpath(path, experiment_dir)).as_posix() for path in paths]
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

    if len(paths) < planned:
        if reference is not None:
            prepared = (
                plan_info[1]
                if plan_info is not None
                else _snapshot_from_run(paths[0], experiment_dir)
            )
            execution = reference.get("execution_controls") or {}
            models = list(reference["models"])
            base_controls = {
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
            }
            if "scoring_policy" in reference:
                base_controls["scoring_policy"] = reference["scoring_policy"]
            if "measurement_profile" in reference:
                base_controls["measurement_profile"] = reference["measurement_profile"]
        else:
            assert plan is not None and plan_info is not None
            prepared = plan_info[1]
            models = list(plan["models"])
            base_controls = dict(plan["run_kwargs"])

        for repetition in range(len(paths) + 1, planned + 1):
            controls = dict(base_controls)
            controls.update(
                prepared=prepared,
                suite_condition="baseline",
                repetition=repetition,
                repeated_measurement=planned > 1,
                planned_passes=planned,
            )
            if max_requests is not None or max_spend_usd is not None:
                controls["invocation_controls"] = {
                    "max_requests": max_requests
                    if max_requests is not None
                    else controls.get("max_requests"),
                    "max_spend_usd": max_spend_usd
                    if max_spend_usd is not None
                    else controls.get("max_spend_usd"),
                }
            controls["progress"] = progress
            if backend_factory is not None:
                controls["backend_factory"] = backend_factory

            def record_progress(message: str) -> None:
                if message.startswith("Run directory: "):
                    run_path = Path(message.removeprefix("Run directory: ")).expanduser().resolve()
                    try:
                        run_path.relative_to(experiment_dir)
                    except ValueError as exc:
                        raise ValueError("New experiment run escaped its directory") from exc
                    if run_path not in paths:
                        paths.append(run_path)
                        save_paths()
                progress(message)

            controls["progress"] = record_progress
            run_dir = Path(run_fn(None, models, experiment_dir / "runs", **controls)).resolve()
            try:
                run_dir.relative_to(experiment_dir)
            except ValueError as exc:
                raise ValueError("New experiment run escaped its directory") from exc
            if run_dir not in paths:
                paths.append(run_dir)
                save_paths()
            current = _read_run_manifest(run_dir)
            if (
                current.get("repetition") != repetition
                or current.get("suite_condition") != "baseline"
            ):
                raise ValueError(f"New run does not match experiment pass {repetition}")
            if current.get("planned_passes", 1) != planned:
                raise ValueError("New run planned_passes does not match experiment.json")
            if plan is not None:
                _verify_plan_run(plan, current)
            if reference is None:
                reference = current
            elif not _same_pass_identity(reference, current):
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
