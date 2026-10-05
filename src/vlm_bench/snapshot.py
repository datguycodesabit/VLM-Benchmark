"""Portable, integrity-checked prepared dataset snapshots.

Snapshots contain the selected crops, references, and metadata needed to run a
benchmark without consulting the original dataset again.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .dataset import prepare_dataset

_MANIFEST_NAME = "dataset-manifest.json"
_SNAPSHOT_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYOUTS = {"auto", "paired", "iam-words", "iam-lines", "iam-forms"}
_PREPROCESSING = {"original", "enhanced"}
_SPLITS = {"train", "validation", "test"}
_CONTENT_TYPES = {"prose", "equation", "word"}
_MAX_SEED = 2**32 - 1
_TOP_LEVEL_METADATA = (
    "writer_id",
    "source_document",
    "split",
    "content_type",
    "sample_type",
    "difficulty",
    "verification_status",
)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Snapshot data is not canonical JSON: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _snapshot_fingerprint(samples: Iterable[dict[str, Any]], preprocess: str) -> str:
    if not isinstance(preprocess, str) or not preprocess:
        raise ValueError("preprocess must be a non-empty string")

    projected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each snapshot sample must be a JSON object")
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Each snapshot sample must have a non-empty string ID")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID {sample_id!r}")
        seen.add(sample_id)

        hashes = sample.get("hashes")
        crop_hash = hashes.get("crop") if isinstance(hashes, dict) else None
        if not _is_sha256(crop_hash):
            raise ValueError(f"Sample {sample_id!r} has no valid crop SHA-256 hash")
        reference = sample.get("reference")
        if not isinstance(reference, str):
            raise ValueError(f"Sample {sample_id!r} has no string reference")
        metadata = sample.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Sample {sample_id!r} metadata must be a JSON object")
        effective_metadata = dict(metadata)
        for field in _TOP_LEVEL_METADATA:
            value = sample.get(field)
            if value is None:
                value = metadata.get(field)
            if value is not None or field in metadata or field in sample:
                effective_metadata[field] = value

        projected.append(
            {
                "id": sample_id,
                "crop_sha256": crop_hash,
                "reference": reference,
                "metadata": effective_metadata,
            }
        )

    projected.sort(key=lambda sample: sample["id"])
    payload = {"preprocess": preprocess, "samples": projected}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def fingerprint(samples: Iterable[dict[str, Any]], preprocess: str) -> str:
    """Return a stable fingerprint for prepared samples.

    The fingerprint covers sorted sample IDs, crop hashes, reference text,
    metadata, and preprocessing. It deliberately ignores timestamps and file
    paths so copied or moved snapshots keep the same identity.
    """
    return _snapshot_fingerprint(samples, preprocess)


def _absolute_output_path(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.name:
        raise ValueError("Snapshot output must name a directory")
    return candidate.parent.resolve() / candidate.name


def _has_entry(path: Path) -> bool:
    return os.path.lexists(path)


def _acquire_publish_lock(output: Path) -> tuple[Path, int]:
    lock_path = output.parent / f".{output.name}.freeze.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Another snapshot freeze may be publishing to {output}; "
            f"if no freeze is active, remove stale lock {lock_path}"
        ) from exc
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    return lock_path, descriptor


def _validate_prepare_options(options: dict[str, Any]) -> None:
    seed = options.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be an integer from 0 to {_MAX_SEED}")

    limit = options.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer when provided")

    layout = options.get("layout", "auto")
    if not isinstance(layout, str) or layout not in _LAYOUTS:
        raise ValueError("layout must be auto, paired, iam-words, iam-lines, or iam-forms")

    preprocess = options.get("preprocess", "original")
    if not isinstance(preprocess, str) or preprocess not in _PREPROCESSING:
        raise ValueError("preprocess must be original or enhanced")

    split = options.get("split")
    if split is not None and (not isinstance(split, str) or split not in _SPLITS):
        raise ValueError("split must be train, validation, or test")

    content_type = options.get("content_type")
    if content_type is not None and (
        not isinstance(content_type, str) or content_type not in _CONTENT_TYPES
    ):
        raise ValueError("content_type must be prose, equation, or word")


def _relative_crop_path(path_value: Any, root: Path, sample_id: str) -> str:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Sample {sample_id!r} has no crop path")
    if "\\" in path_value or "\x00" in path_value or ":" in path_value:
        raise ValueError(f"Sample {sample_id!r} has an invalid portable crop path")
    parts = path_value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Sample {sample_id!r} crop path must not traverse directories")
    pure = PurePosixPath(path_value)
    if pure.is_absolute() or len(parts) < 2 or parts[0] != "crops":
        raise ValueError(f"Sample {sample_id!r} crop path must be relative to crops/")

    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Sample {sample_id!r} crop path escapes the snapshot or is missing"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"Sample {sample_id!r} crop path is not a file")
    return PurePosixPath(*parts).as_posix()


def _sample_for_manifest(sample: dict[str, Any], root: Path) -> dict[str, Any]:
    sample_id = sample.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Prepared sample has no non-empty string ID")
    relative = Path(sample["crop_path"]).resolve(strict=True).relative_to(root)
    if relative.parts[0] != "crops":
        raise ValueError(f"Prepared crop for sample {sample_id!r} is outside crops/")
    hashes = sample.get("hashes")
    if not isinstance(hashes, dict) or not _is_sha256(hashes.get("crop")):
        raise ValueError(f"Prepared sample {sample_id!r} has no valid crop hash")

    result = {
        "id": sample_id,
        "crop_path": PurePosixPath(*relative.parts).as_posix(),
        "reference": sample.get("reference"),
        "reference_source": sample.get("reference_source"),
        "metadata": sample.get("metadata") or {},
        "preprocess": sample.get("preprocess"),
        "crop_bbox": sample.get("crop_bbox"),
        "hashes": dict(hashes),
    }
    for field in _TOP_LEVEL_METADATA:
        result[field] = sample.get(field)
    return result


def _manifest_integrity(manifest: dict[str, Any]) -> str:
    contents = {key: value for key, value in manifest.items() if key != "integrity"}
    return hashlib.sha256(_canonical_json(contents)).hexdigest()


def freeze(
    data_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    **prepare_options: Any,
) -> dict[str, Any]:
    """Prepare and atomically publish a portable v2 dataset snapshot.

    ``output_dir`` must not exist. Preparation occurs in a sibling temporary
    directory so an incomplete operation never appears as a usable snapshot.
    ``write_references`` retains ``prepare_dataset``'s default of ``False``.
    """
    _validate_prepare_options(prepare_options)
    output = _absolute_output_path(output_dir)
    if _has_entry(output):
        raise FileExistsError(f"Snapshot output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if _has_entry(output):
        raise FileExistsError(f"Snapshot output already exists: {output}")

    lock_path, lock_descriptor = _acquire_publish_lock(output)
    staging: Path | None = None
    try:
        if _has_entry(output):
            raise FileExistsError(f"Snapshot output already exists: {output}")
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        prepared = prepare_dataset(data_dir, staging, **prepare_options)
        v1_path = staging / _MANIFEST_NAME
        try:
            legacy = json.loads(v1_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Dataset preparation did not produce a readable manifest: {exc}"
            ) from exc
        if not isinstance(legacy, dict) or legacy.get("manifest_version") != 1:
            raise ValueError("Dataset preparation produced an unexpected manifest version")

        staging_root = staging.resolve(strict=True)
        samples = [_sample_for_manifest(sample, staging_root) for sample in prepared]
        if not samples:
            raise ValueError("Cannot freeze an empty prepared dataset")
        sample_ids = [sample["id"] for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Prepared dataset contains duplicate sample IDs")
        if legacy.get("sample_ids") != sample_ids:
            raise ValueError("Prepared samples do not match the dataset manifest order")

        preprocess = legacy.get("preprocess")
        selected_layout = legacy.get("layout")
        if not isinstance(preprocess, str) or not preprocess:
            raise ValueError("Prepared dataset manifest has no preprocessing mode")
        if not isinstance(selected_layout, str) or not selected_layout:
            raise ValueError("Prepared dataset manifest has no resolved layout")

        options = {
            "limit": prepare_options.get("limit"),
            "seed": prepare_options.get("seed", 42),
            "split": prepare_options.get("split"),
            "content_type": prepare_options.get("content_type"),
        }
        manifest: dict[str, Any] = {
            "manifest_version": _SNAPSHOT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "layout": selected_layout,
            "preprocess": preprocess,
            "seed": legacy.get("seed", 42),
            "limit": legacy.get("limit"),
            "split": prepare_options.get("split"),
            "content_type": prepare_options.get("content_type"),
            "sample_ids": sample_ids,
            "samples": samples,
            "provenance": {
                "requested_layout": prepare_options.get("layout", "auto"),
                "selection": options,
                "write_references": prepare_options.get("write_references", False),
            },
        }
        manifest["benchmark_fingerprint"] = fingerprint(samples, preprocess)
        manifest["integrity"] = _manifest_integrity(manifest)
        v1_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        if _has_entry(output):
            raise FileExistsError(f"Snapshot output already exists: {output}")
        os.rename(staging, output)
        return load(output)
    finally:
        if staging is not None and _has_entry(staging):
            shutil.rmtree(staging, ignore_errors=True)
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def _safe_manifest_path(snapshot_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    candidate = Path(snapshot_path).expanduser()
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Snapshot directory cannot be accessed: {candidate}") from exc
    if not root.is_dir():
        raise ValueError(f"Snapshot path is not a directory: {candidate}")
    manifest_path = root / _MANIFEST_NAME
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
        resolved_manifest.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Snapshot manifest is missing or escapes the snapshot directory") from exc
    if not resolved_manifest.is_file():
        raise ValueError("Snapshot manifest is not a file")
    return root, resolved_manifest


def _manifest_samples_for_load(manifest: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(manifest, dict):
        raise ValueError("Snapshot manifest must be a JSON object")
    version = manifest.get("manifest_version")
    if version == 1:
        raise ValueError(
            "This is a legacy dataset-manifest.json v1. It lacks portable crop paths; "
            "re-prepare the source as a v2 snapshot with "
            "`vlm-bench prepare --data SOURCE --output NEW_FOLDER`."
        )
    if version != _SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported dataset manifest version {version!r}; expected snapshot version 2"
        )
    if not isinstance(manifest.get("layout"), str) or not manifest["layout"]:
        raise ValueError("Snapshot manifest has no layout")
    if not isinstance(manifest.get("preprocess"), str) or not manifest["preprocess"]:
        raise ValueError("Snapshot manifest has no preprocessing mode")
    samples = manifest.get("samples")
    sample_ids = manifest.get("sample_ids")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Snapshot manifest samples must be a non-empty ordered list")
    if not isinstance(sample_ids, list) or len(sample_ids) != len(samples):
        raise ValueError("Snapshot manifest sample_ids do not match its samples")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    relative_paths: list[str] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Snapshot sample {index} must be a JSON object")
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Snapshot sample {index} has no non-empty string ID")
        if sample_id in seen_ids:
            raise ValueError(f"Snapshot manifest has duplicate sample ID {sample_id!r}")
        seen_ids.add(sample_id)
        if sample_ids[index] != sample_id:
            raise ValueError("Snapshot manifest sample_ids do not preserve sample order")

        relative = sample.get("crop_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Snapshot sample {sample_id!r} has no relative crop_path")
        if "\\" in relative or "\x00" in relative or ":" in relative:
            raise ValueError(f"Snapshot sample {sample_id!r} has an invalid crop_path")
        parts = relative.split("/")
        if (
            any(part in {"", ".", ".."} for part in parts)
            or PurePosixPath(relative).is_absolute()
            or len(parts) < 2
            or parts[0] != "crops"
        ):
            raise ValueError(f"Snapshot sample {sample_id!r} crop_path escapes the snapshot")
        if relative in seen_paths:
            raise ValueError(f"Snapshot samples share crop_path {relative!r}")
        seen_paths.add(relative)
        relative_paths.append(PurePosixPath(*parts).as_posix())

        reference = sample.get("reference")
        if not isinstance(reference, str):
            raise ValueError(f"Snapshot sample {sample_id!r} has no string reference")
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Snapshot sample {sample_id!r} metadata must be an object")
        hashes = sample.get("hashes")
        if not isinstance(hashes, dict) or not _is_sha256(hashes.get("crop")):
            raise ValueError(f"Snapshot sample {sample_id!r} has no valid crop SHA-256 hash")
        if sample.get("preprocess") != manifest["preprocess"]:
            raise ValueError(f"Snapshot sample {sample_id!r} preprocessing does not match manifest")

    return samples, relative_paths


def _load_validated(
    snapshot_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], Path, list[str]]:
    root, manifest_path = _safe_manifest_path(snapshot_path)
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read snapshot manifest: {exc}") from exc

    disk_samples, relative_paths = _manifest_samples_for_load(raw_manifest)
    stored_integrity = raw_manifest.get("integrity")
    if not _is_sha256(stored_integrity) or _manifest_integrity(raw_manifest) != stored_integrity:
        raise ValueError("Snapshot manifest integrity check failed")
    expected_fingerprint = raw_manifest.get("benchmark_fingerprint")
    computed_fingerprint = fingerprint(disk_samples, raw_manifest["preprocess"])
    if not _is_sha256(expected_fingerprint) or expected_fingerprint != computed_fingerprint:
        raise ValueError("Snapshot benchmark fingerprint does not match its samples")

    materialized = deepcopy(raw_manifest)
    loaded_samples: list[dict[str, Any]] = []
    for sample, relative in zip(disk_samples, relative_paths):
        candidate = root.joinpath(*PurePosixPath(relative).parts)
        try:
            crop = candidate.resolve(strict=True)
            crop.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Snapshot crop for sample {sample['id']!r} escapes the snapshot or is missing"
            ) from exc
        if not crop.is_file():
            raise ValueError(f"Snapshot crop for sample {sample['id']!r} is not a file")
        if _sha256_file(crop) != sample["hashes"]["crop"]:
            raise ValueError(f"Snapshot crop hash mismatch for sample {sample['id']!r}")

        loaded = dict(sample)
        loaded["crop_path"] = str(crop)
        metadata = loaded.get("metadata", {})
        for field in _TOP_LEVEL_METADATA:
            if loaded.get(field) is None:
                loaded[field] = metadata.get(field)
        loaded_samples.append(loaded)
    materialized["samples"] = loaded_samples
    return materialized, root, relative_paths


def load(snapshot_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a v2 snapshot after checking its manifest and every crop."""
    manifest, _, _ = _load_validated(snapshot_path)
    return manifest


def copy_inputs(
    snapshot_path: str | os.PathLike[str], run_dir: str | os.PathLike[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy validated frozen inputs into a run directory and return engine records.

    The destination ``run_dir/crops`` must be absent. Existing run directory
    contents such as the engine's lock file are preserved.
    """
    manifest, _, relative_paths = _load_validated(snapshot_path)
    run_root = Path(run_dir).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    destination = run_root / "crops"
    if _has_entry(destination):
        raise FileExistsError(f"Run crop directory already exists: {destination}")

    staging = Path(tempfile.mkdtemp(prefix=".crops.tmp-", dir=run_root))
    copied_samples: list[dict[str, Any]] = []
    try:
        for sample, relative in zip(manifest["samples"], relative_paths):
            parts = PurePosixPath(relative).parts
            inner = Path(*parts[1:])
            target = staging / inner
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sample["crop_path"], target)
            if _sha256_file(target) != sample["hashes"]["crop"]:
                raise ValueError(f"Copied crop hash mismatch for sample {sample['id']!r}")
            copied = dict(sample)
            copied["crop_path"] = str(destination / inner)
            copied_samples.append(copied)

        if _has_entry(destination):
            raise FileExistsError(f"Run crop directory already exists: {destination}")
        os.rename(staging, destination)
    finally:
        if _has_entry(staging):
            shutil.rmtree(staging, ignore_errors=True)

    # The second tuple item remains the validated snapshot manifest. The first
    # item points at the run-local copies consumed by the execution engine.
    return copied_samples, manifest
