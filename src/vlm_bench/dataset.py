"""Dataset discovery and IAM handwritten-form preparation.

The IAM line selection, signature filtering, bounding-box union, and crop
padding are adapted from ``scripts/crop_handwritten.py`` and
``scripts/generate_ground_truth.py`` in PyaesoneP/vlm-ocr-research (Apache-2.0).
Source snapshot: https://github.com/PyaesoneP/vlm-ocr-research/tree/fd4bd0ae44db0f57f7dcb0e301a0a718d3e6159f/scripts
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_PADDING = 20
_MODEL_SCALE = 3
_MODEL_BORDER = 16
_METADATA_FIELDS = {
    "source_document",
    "writer_id",
    "split",
    "content_type",
    "difficulty",
    "verification_status",
    "sample_type",
}
_SPLITS = {"train", "validation", "test"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_by_stem(paths: list[Path], kind: str, suffix: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        if path.suffix.lower() != suffix:
            continue
        sample_id = path.stem
        previous = indexed.get(sample_id)
        if previous is not None:
            raise ValueError(f"Duplicate {kind} ID {sample_id!r}: {previous} and {path}")
        indexed[sample_id] = path
    return indexed


def _files_with_suffix(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )


def _load_metadata(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Read optional JSONL metadata keyed by sample ID."""
    path = data_dir / "metadata.jsonl"
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError(f"Metadata path is not a file: {path}")
    records: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read metadata JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Metadata record at {path}:{line_number} must be a JSON object")
        sample_id = value.get("id", value.get("sample_id"))
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Metadata record at {path}:{line_number} has no sample ID")
        sample_id = sample_id.strip()
        if sample_id in records:
            raise ValueError(f"Duplicate metadata ID {sample_id!r} in {path}")
        records[sample_id] = {key: value[key] for key in _METADATA_FIELDS if key in value}
    return records


def _metadata_for(
    sample_id: str,
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    values = dict(metadata.get(sample_id, {}))
    return {key: values.get(key) for key in sorted(_METADATA_FIELDS)}


def _filter_metadata_records(
    records: list[dict[str, Any]], split: str | None, content_type: str | None
) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        metadata = record.get("metadata") or {}
        if split is not None and metadata.get("split") != split:
            continue
        if content_type is not None and metadata.get("content_type") != content_type:
            continue
        filtered.append(record)
    if not filtered and records:
        filters = []
        if split is not None:
            filters.append(f"split={split!r}")
        if content_type is not None:
            filters.append(f"content_type={content_type!r}")
        raise ValueError("No samples match the requested metadata filter: " + ", ".join(filters))
    return filtered


def _manifest_sample(sample: dict[str, Any], preprocess: str) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "reference": sample["reference"],
        "reference_source": sample.get("reference_source"),
        "hashes": sample["hashes"],
        "metadata": sample.get("metadata", {}),
        "preprocess": preprocess,
    }


def _write_prepared_manifest(
    output_dir: Path,
    samples: list[dict[str, Any]],
    *,
    layout: str,
    preprocess: str,
    seed: int,
    limit: int | None,
) -> None:
    manifest = {
        "manifest_version": 1,
        "layout": layout,
        "preprocess": preprocess,
        "seed": seed,
        "limit": limit,
        "sample_ids": [sample["id"] for sample in samples],
        "samples": [_manifest_sample(sample, preprocess) for sample in samples],
    }
    path = output_dir / "dataset-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_coordinate(element: ET.Element, attribute: str, sample_id: str) -> int:
    value = element.get(attribute)
    if value is None:
        raise ValueError(f"Sample {sample_id!r} has a <cmp> without {attribute!r} coordinate")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sample {sample_id!r} has invalid <cmp> {attribute}={value!r}") from exc


def _writer_id(root: ET.Element) -> str | None:
    for key in ("writer-id", "writer_id", "writer"):
        value = root.get(key)
        if value:
            return value
    writer = root.find(".//writer")
    if writer is not None:
        return writer.get("id") or writer.get("writer-id")
    return None


def _load_xml_sample(
    xml_path: Path, sample_id: str, image_size: tuple[int, int]
) -> tuple[str, tuple[int, int, int, int], str | None]:
    """Return retained handwritten text, its union box, and writer identifier."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Cannot parse XML for sample {sample_id!r}: {exc}") from exc

    handwritten = root.find("handwritten-part")
    if handwritten is None:
        raise ValueError(f"Sample {sample_id!r} XML has no <handwritten-part>")
    lines = handwritten.findall("line")
    if not lines:
        raise ValueError(f"Sample {sample_id!r} XML has no handwritten lines")

    width, height = image_size
    raw_height = root.get("height")
    try:
        form_height = int(raw_height) if raw_height is not None else 3542
    except ValueError as exc:
        raise ValueError(f"Sample {sample_id!r} has invalid form height {raw_height!r}") from exc
    if form_height <= 0:
        raise ValueError(f"Sample {sample_id!r} has non-positive form height")

    retained_text: list[str] = []
    retained_boxes: list[tuple[int, int, int, int]] = []
    for index, line in enumerate(lines):
        text = line.get("text", "").strip()
        components = list(line.iter("cmp"))
        if not components:
            continue

        component_boxes: list[tuple[int, int, int, int]] = []
        for component in components:
            x = _parse_coordinate(component, "x", sample_id)
            y = _parse_coordinate(component, "y", sample_id)
            box_width = _parse_coordinate(component, "width", sample_id)
            box_height = _parse_coordinate(component, "height", sample_id)
            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                raise ValueError(
                    f"Sample {sample_id!r} has out-of-range or empty <cmp> box "
                    f"[{x}, {y}, {x + box_width}, {y + box_height}]"
                )
            x2, y2 = x + box_width, y + box_height
            if x2 > width or y2 > height:
                raise ValueError(
                    f"Sample {sample_id!r} <cmp> box [{x}, {y}, {x2}, {y2}] "
                    f"exceeds image bounds {width}x{height}"
                )
            component_boxes.append((x, y, x2, y2))

        line_box = (
            min(box[0] for box in component_boxes),
            min(box[1] for box in component_boxes),
            max(box[2] for box in component_boxes),
            max(box[3] for box in component_boxes),
        )
        if not text:
            continue

        # This mirrors the upstream IAM rule: omit a final "Name:" line or a
        # one/two-word line whose top edge falls in the bottom 15% of the form.
        is_last = index == len(lines) - 1
        is_short = len(text.split()) <= 2
        in_signature_band = line_box[1] > form_height * 0.85
        if (is_last and "Name:" in text) or (is_short and in_signature_band):
            continue

        retained_text.append(text)
        retained_boxes.extend(component_boxes)

    if not retained_text or not retained_boxes:
        raise ValueError(f"Sample {sample_id!r} has no retained handwritten text")

    union_box = (
        min(box[0] for box in retained_boxes),
        min(box[1] for box in retained_boxes),
        max(box[2] for box in retained_boxes),
        max(box[3] for box in retained_boxes),
    )
    return "\n".join(retained_text), union_box, _writer_id(root)


def _read_reference(path: Path, sample_id: str) -> tuple[str, bytes]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Reference for sample {sample_id!r} is not readable UTF-8: {exc}"
        ) from exc
    if not text.strip():
        raise ValueError(f"Reference for sample {sample_id!r} is empty")
    return text, raw


def _validate_image(path: Path, sample_id: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"Image for sample {sample_id!r} cannot be decoded: {exc}") from exc


def _save_model_crop(
    source: Path,
    destination: Path,
    bbox: tuple[int, int, int, int],
    preprocess: str,
) -> None:
    """Save either source pixels or the optional enhanced model crop."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        crop = image.crop(bbox)
        if preprocess == "enhanced":
            crop = crop.convert("L")
            crop = ImageOps.autocontrast(crop)
            crop = crop.resize(
                (crop.width * _MODEL_SCALE, crop.height * _MODEL_SCALE),
                Image.Resampling.LANCZOS,
            )
            crop = ImageOps.expand(crop, border=_MODEL_BORDER, fill=255)
        crop.save(destination, format="PNG")


def _save_full_source(
    source: Path,
    destination: Path,
    preprocess: str,
) -> None:
    """Copy an image byte-for-byte or save its enhanced derivative."""
    if preprocess == "original":
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return
    width, height = _validate_image(source, source.stem)
    _save_model_crop(source, destination, (0, 0, width, height), preprocess)


def _word_annotations(path: Path) -> dict[str, str]:
    """Read IAM's word table, whose final field is the word transcription."""
    annotations: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read IAM word annotations: {path}: {exc}") from exc
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        sample_id = fields[0]
        if len(fields) > 2 and fields[1].lower() != "ok":
            continue
        text = fields[-1]
        previous = annotations.get(sample_id)
        if previous is not None and previous != text:
            raise ValueError(f"Conflicting word annotations for {sample_id!r} in {path}")
        annotations[sample_id] = text
    if not annotations:
        raise ValueError(f"IAM word annotation file contains no samples: {path}")
    return annotations


def _line_annotations(path: Path) -> dict[str, str]:
    """Read IAM's ASCII line table, including pipe-delimited exports."""
    annotations: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read IAM line annotations: {path}: {exc}") from exc
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.count("|") >= 8:
            fields = [field.strip() for field in line.strip().strip("|").split("|")]
        else:
            fields = line.split()
        # ID, status, graylevel, components, x, y, width, height, text...
        if len(fields) < 9:
            continue
        try:
            [int(value) for value in fields[2:8]]
        except ValueError:
            continue
        sample_id, status = fields[0], fields[1].lower()
        if status != "ok":
            continue
        text = " ".join(fields[8:]).replace("|", " ")
        text = " ".join(text.split())
        if not text:
            continue
        previous = annotations.get(sample_id)
        if previous is not None and previous != text:
            raise ValueError(f"Conflicting line annotations for {sample_id!r} in {path}")
        annotations[sample_id] = text
    if not annotations:
        raise ValueError(f"IAM line annotation file contains no samples: {path}")
    return annotations


def _relative_sample_id(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).with_suffix("").as_posix()


def _paired_paths(data_dir: Path) -> tuple[str, dict[str, Path], dict[str, Path]]:
    image_dir = data_dir / "images"
    if not image_dir.is_dir():
        raise ValueError(f"Paired dataset requires an images/ directory: {image_dir}")
    text_dir = data_dir / "text"
    references_dir = data_dir / "references"
    if text_dir.exists() and references_dir.exists():
        raise ValueError("Paired dataset is ambiguous: provide text/ or references/, not both")
    reference_dir = text_dir if text_dir.exists() else references_dir
    reference_name = "text" if reference_dir == text_dir else "references"
    if not reference_dir.is_dir():
        raise ValueError("Paired dataset requires text/ or references/ alongside images/")

    image_by_id: dict[str, Path] = {}
    for path in _files_with_suffix(image_dir, _IMAGE_SUFFIXES):
        sample_id = _relative_sample_id(path, image_dir)
        if sample_id in image_by_id:
            raise ValueError(
                f"Duplicate image ID {sample_id!r}: {image_by_id[sample_id]} and {path}"
            )
        image_by_id[sample_id] = path
    reference_by_id: dict[str, Path] = {}
    for path in _files_with_suffix(reference_dir, {".txt"}):
        sample_id = _relative_sample_id(path, reference_dir)
        if sample_id in reference_by_id:
            raise ValueError(
                f"Duplicate reference ID {sample_id!r}: {reference_by_id[sample_id]} and {path}"
            )
        reference_by_id[sample_id] = path
    return reference_name, image_by_id, reference_by_id


def _prepare_paired_dataset(
    data_dir: Path,
    output_dir: Path,
    limit: int | None,
    seed: int,
    preprocess: str,
    split: str | None,
    content_type: str | None,
) -> list[dict[str, Any]]:
    """Prepare generic images and references matched by relative path and stem."""
    reference_name, image_by_id, reference_by_id = _paired_paths(data_dir)
    image_ids, reference_ids = set(image_by_id), set(reference_by_id)
    missing_text = sorted(image_ids - reference_ids)
    missing_images = sorted(reference_ids - image_ids)
    if missing_text or missing_images:
        raise ValueError(
            "Paired dataset files do not match: "
            f"{len(missing_text)} images lack references and "
            f"{len(missing_images)} references lack images"
        )
    if not image_by_id:
        raise ValueError(f"No image/reference pairs found under {data_dir}")

    metadata_by_id = _load_metadata(data_dir)
    unknown_metadata = sorted(set(metadata_by_id) - image_ids)
    if unknown_metadata:
        raise ValueError(f"Metadata refers to unknown sample ID {unknown_metadata[0]!r}")
    records: list[dict[str, Any]] = []
    for sample_id in sorted(image_ids):
        image_path = image_by_id[sample_id]
        reference_path = reference_by_id[sample_id]
        width, height = _validate_image(image_path, sample_id)
        reference, reference_bytes = _read_reference(reference_path, sample_id)
        sample_metadata = _metadata_for(sample_id, metadata_by_id)
        sample_metadata["sample_type"] = sample_metadata.get("sample_type") or "line"
        records.append(
            {
                "id": sample_id,
                "image_path": image_path,
                "reference_path": reference_path,
                "reference": reference,
                "reference_bytes": reference_bytes,
                "reference_source": reference_name,
                "writer_id": sample_metadata.get("writer_id"),
                "metadata": sample_metadata,
                "crop_bbox": [0, 0, width, height],
            }
        )
    records = _filter_metadata_records(records, split, content_type)
    if limit is not None:
        if limit > len(records):
            raise ValueError(f"Requested {limit} samples, but only {len(records)} are available")
        chosen = set(random.Random(seed).sample([record["id"] for record in records], limit))
        records = [record for record in records if record["id"] in chosen]

    crop_dir = output_dir / "crops"
    results: list[dict[str, Any]] = []
    for record in records:
        sample_id = record["id"]
        source_path: Path = record["image_path"]
        extension = source_path.suffix.lower() if preprocess == "original" else ".png"
        crop_path = crop_dir / f"{sample_id}{extension}"
        _save_full_source(source_path, crop_path, preprocess)
        results.append(
            {
                "id": sample_id,
                "image_path": str(source_path),
                "crop_path": str(crop_path.resolve()),
                "reference": record["reference"],
                "reference_source": record["reference_source"],
                "writer_id": record["writer_id"],
                "source_document": record["metadata"].get("source_document"),
                "split": record["metadata"].get("split"),
                "content_type": record["metadata"].get("content_type"),
                "sample_type": record["metadata"].get("sample_type"),
                "difficulty": record["metadata"].get("difficulty"),
                "verification_status": record["metadata"].get("verification_status"),
                "metadata": record["metadata"],
                "preprocess": preprocess,
                "crop_bbox": record["crop_bbox"],
                "hashes": {
                    "image": _sha256_file(source_path),
                    "reference": _sha256_bytes(record["reference_bytes"]),
                    "crop": _sha256_file(crop_path),
                    "metadata": _sha256_bytes(
                        json.dumps(record["metadata"], sort_keys=True, ensure_ascii=False).encode(
                            "utf-8"
                        )
                    ),
                },
            }
        )
    _write_prepared_manifest(
        output_dir,
        results,
        layout="paired",
        preprocess=preprocess,
        seed=seed,
        limit=limit,
    )
    return results


def _prepare_line_dataset(
    data_dir: Path,
    output_dir: Path,
    limit: int | None,
    seed: int,
    preprocess: str,
    split: str | None,
    content_type: str | None,
) -> list[dict[str, Any]]:
    """Prepare IAM line images paired with lines.txt transcriptions."""
    all_images = _all_data_images(data_dir, output_dir)
    image_by_id: dict[str, Path] = {}
    for image_path in all_images:
        previous = image_by_id.get(image_path.stem)
        if previous is not None:
            raise ValueError(f"Duplicate image ID {image_path.stem!r}: {previous} and {image_path}")
        image_by_id[image_path.stem] = image_path

    matches: list[tuple[int, Path, dict[str, str]]] = []
    for candidate in sorted(data_dir.rglob("lines*.txt")):
        try:
            annotations = _line_annotations(candidate)
        except ValueError:
            continue
        count = len(set(annotations) & set(image_by_id))
        if count:
            matches.append((count, candidate, annotations))
    if not matches:
        raise ValueError(
            "No IAM lines.txt matching the pasted images was found. "
            "Expected line images and a lines.txt annotation file."
        )
    _, annotation_path, annotations = max(matches, key=lambda item: (item[0], str(item[1])))

    override_dirs = list(
        dict.fromkeys([data_dir / "references", annotation_path.parent / "references"])
    )
    override_paths: list[Path] = []
    for directory in override_dirs:
        override_paths.extend(_files_with_suffix(directory, {".txt"}))
    reference_by_id = _index_by_stem(override_paths, "reference", ".txt") if override_paths else {}

    metadata_by_id = _load_metadata(data_dir)
    unknown_metadata = sorted(set(metadata_by_id) - set(annotations))
    if unknown_metadata:
        raise ValueError(f"Metadata refers to unknown sample ID {unknown_metadata[0]!r}")
    records: list[dict[str, Any]] = []
    for sample_id in sorted(set(annotations) & set(image_by_id)):
        image_path = image_by_id[sample_id]
        try:
            width, height = _validate_image(image_path, sample_id)
        except ValueError as exc:
            warnings.warn(f"Skipping invalid IAM line sample {sample_id!r}: {exc}", stacklevel=2)
            continue
        reference = annotations[sample_id]
        reference_source = "lines.txt"
        reference_bytes = reference.encode("utf-8")
        reference_path = reference_by_id.get(sample_id)
        if reference_path is not None:
            reference, reference_bytes = _read_reference(reference_path, sample_id)
            reference_source = "txt"
        sample_metadata = _metadata_for(sample_id, metadata_by_id)
        sample_metadata["sample_type"] = sample_metadata.get("sample_type") or "line"
        records.append(
            {
                "id": sample_id,
                "image_path": image_path,
                "reference": reference,
                "reference_bytes": reference_bytes,
                "reference_source": reference_source,
                "writer_id": sample_metadata.get("writer_id"),
                "metadata": sample_metadata,
                "crop_bbox": [0, 0, width, height],
                "annotation_path": annotation_path,
            }
        )

    if not records:
        raise ValueError("No valid IAM line images matched lines.txt")
    records = _filter_metadata_records(records, split, content_type)
    if limit is not None:
        if limit > len(records):
            raise ValueError(f"Requested {limit} samples, but only {len(records)} are available")
        chosen = set(random.Random(seed).sample([record["id"] for record in records], limit))
        records = [record for record in records if record["id"] in chosen]

    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for record in records:
        crop_path = crop_dir / f"{record['id']}.png"
        _save_model_crop(record["image_path"], crop_path, tuple(record["crop_bbox"]), preprocess)
        sample_metadata = record["metadata"]
        results.append(
            {
                "id": record["id"],
                "image_path": str(record["image_path"]),
                "crop_path": str(crop_path.resolve()),
                "reference": record["reference"],
                "reference_source": record["reference_source"],
                "writer_id": record["writer_id"],
                "source_document": sample_metadata.get("source_document"),
                "split": sample_metadata.get("split"),
                "content_type": sample_metadata.get("content_type"),
                "sample_type": sample_metadata.get("sample_type"),
                "difficulty": sample_metadata.get("difficulty"),
                "verification_status": sample_metadata.get("verification_status"),
                "metadata": sample_metadata,
                "preprocess": preprocess,
                "crop_bbox": record["crop_bbox"],
                "hashes": {
                    "image": _sha256_file(record["image_path"]),
                    "xml": _sha256_file(record["annotation_path"]),
                    "reference": _sha256_bytes(record["reference_bytes"]),
                    "crop": _sha256_file(crop_path),
                    "metadata": _sha256_bytes(
                        json.dumps(sample_metadata, sort_keys=True, ensure_ascii=False).encode(
                            "utf-8"
                        )
                    ),
                },
            }
        )
    _write_prepared_manifest(
        output_dir,
        results,
        layout="iam-lines",
        preprocess=preprocess,
        seed=seed,
        limit=limit,
    )
    return results


def _prepare_word_dataset(
    data_dir: Path,
    output_dir: Path,
    limit: int | None,
    seed: int,
    write_references: bool,
    preprocess: str,
    split: str | None,
    content_type: str | None,
) -> list[dict[str, Any]]:
    """Prepare pasted IAM word images when no form/XML layout is present."""
    all_images = _all_data_images(data_dir, output_dir)
    image_by_id: dict[str, Path] = {}
    for image_path in all_images:
        previous = image_by_id.get(image_path.stem)
        if previous is not None:
            raise ValueError(f"Duplicate image ID {image_path.stem!r}: {previous} and {image_path}")
        image_by_id[image_path.stem] = image_path
    if not image_by_id:
        raise ValueError(f"No image files found under {data_dir}")

    candidates = sorted(data_dir.rglob("words*.txt"))
    matches: list[tuple[int, Path, dict[str, str]]] = []
    for candidate in candidates:
        try:
            annotations = _word_annotations(candidate)
        except ValueError:
            continue
        count = len(set(annotations) & set(image_by_id))
        if count:
            matches.append((count, candidate, annotations))
    if not matches:
        raise ValueError(
            "No IAM words.txt matching the pasted images was found. "
            "For form data, provide images/ and matching xml/ directories."
        )
    _, annotation_path, annotations = max(matches, key=lambda item: (item[0], str(item[1])))

    override_dirs = list(
        dict.fromkeys([data_dir / "references", annotation_path.parent / "references"])
    )
    override_paths: list[Path] = []
    for directory in override_dirs:
        override_paths.extend(_files_with_suffix(directory, {".txt"}))
    reference_by_id = _index_by_stem(override_paths, "reference", ".txt") if override_paths else {}

    metadata_by_id = _load_metadata(data_dir)
    unknown_metadata = sorted(set(metadata_by_id) - set(annotations))
    if unknown_metadata:
        raise ValueError(f"Metadata refers to unknown sample ID {unknown_metadata[0]!r}")
    records: list[dict[str, Any]] = []
    for sample_id in sorted(set(annotations) & set(image_by_id)):
        image_path = image_by_id[sample_id]
        try:
            width, height = _validate_image(image_path, sample_id)
        except ValueError as exc:
            # Word archives in the wild occasionally contain an empty or
            # truncated crop. It cannot be scored, so omit only that sample
            # and make the omission visible to the caller.
            warnings.warn(f"Skipping invalid IAM word sample {sample_id!r}: {exc}", stacklevel=2)
            continue
        reference = annotations[sample_id]
        reference_source = "words.txt"
        reference_bytes = reference.encode("utf-8")
        reference_path = reference_by_id.get(sample_id)
        if reference_path is not None:
            reference, reference_bytes = _read_reference(reference_path, sample_id)
            reference_source = "txt"
        if not reference.strip():
            raise ValueError(f"Reference for sample {sample_id!r} is empty")
        sample_metadata = _metadata_for(sample_id, metadata_by_id)
        sample_metadata["sample_type"] = sample_metadata.get("sample_type") or "word"
        records.append(
            {
                "id": sample_id,
                "image_path": image_path,
                "reference": reference,
                "reference_bytes": reference_bytes,
                "reference_source": reference_source,
                "reference_path": reference_path,
                "writer_id": sample_metadata.get("writer_id"),
                "metadata": sample_metadata,
                "crop_bbox": [0, 0, width, height],
                "annotation_path": annotation_path,
            }
        )

    records = _filter_metadata_records(records, split, content_type)
    if limit is not None:
        if limit > len(records):
            raise ValueError(f"Requested {limit} samples, but only {len(records)} are available")
        chosen = set(random.Random(seed).sample([record["id"] for record in records], limit))
        records = [record for record in records if record["id"] in chosen]

    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for record in records:
        crop_path = crop_dir / f"{record['id']}.png"
        _save_model_crop(record["image_path"], crop_path, tuple(record["crop_bbox"]), preprocess)
        sample_metadata = record["metadata"]
        results.append(
            {
                "id": record["id"],
                "image_path": str(record["image_path"]),
                "crop_path": str(crop_path.resolve()),
                "reference": record["reference"],
                "reference_source": record["reference_source"],
                "writer_id": record["writer_id"],
                "source_document": sample_metadata.get("source_document"),
                "split": sample_metadata.get("split"),
                "content_type": sample_metadata.get("content_type"),
                "sample_type": sample_metadata.get("sample_type"),
                "difficulty": sample_metadata.get("difficulty"),
                "verification_status": sample_metadata.get("verification_status"),
                "metadata": sample_metadata,
                "preprocess": preprocess,
                "crop_bbox": record["crop_bbox"],
                "hashes": {
                    "image": _sha256_file(record["image_path"]),
                    "xml": _sha256_file(record["annotation_path"]),
                    "reference": _sha256_bytes(record["reference_bytes"]),
                    "crop": _sha256_file(crop_path),
                    "metadata": _sha256_bytes(
                        json.dumps(sample_metadata, sort_keys=True, ensure_ascii=False).encode(
                            "utf-8"
                        )
                    ),
                },
            }
        )
    _write_prepared_manifest(
        output_dir,
        results,
        layout="iam-words",
        preprocess=preprocess,
        seed=seed,
        limit=limit,
    )
    return results


def _all_data_images(data_dir: Path, output_dir: Path | None = None) -> list[Path]:
    paths = _files_with_suffix(data_dir, _IMAGE_SUFFIXES)
    paths = [path for path in paths if not {"prepared", "runs"}.intersection(path.parts)]
    if output_dir is not None and output_dir.is_relative_to(data_dir):
        paths = [path for path in paths if output_dir not in path.parents]
    return paths


def _detect_annotation_layout(
    data_dir: Path,
    form_images: list[Path],
    output_dir: Path | None = None,
) -> str:
    """Resolve non-form IAM layouts using overlap between IDs and images."""
    if form_images and not (data_dir / "text").exists() and not (data_dir / "references").exists():
        return "iam-forms"
    images = _all_data_images(data_dir, output_dir)
    image_by_id: dict[str, Path] = {}
    for image in images:
        if image.stem in image_by_id:
            raise ValueError(
                f"Duplicate image ID {image.stem!r}: {image_by_id[image.stem]} and {image}"
            )
        image_by_id[image.stem] = image
    matches: dict[str, list[tuple[int, Path]]] = {"iam-lines": [], "iam-words": []}
    for annotation in sorted(data_dir.rglob("lines*.txt")):
        try:
            parsed = _line_annotations(annotation)
        except ValueError:
            continue
        overlap = len(set(parsed) & set(image_by_id))
        if overlap:
            matches["iam-lines"].append((overlap, annotation))
    for annotation in sorted(data_dir.rglob("words*.txt")):
        try:
            parsed = _word_annotations(annotation)
        except ValueError:
            continue
        overlap = len(set(parsed) & set(image_by_id))
        if overlap:
            matches["iam-words"].append((overlap, annotation))
    available = [layout for layout, items in matches.items() if items]
    if len(available) > 1:
        raise ValueError(
            "Both IAM line and word annotations match images; select --layout explicitly"
        )
    if available:
        return available[0]
    raise ValueError(
        "Cannot detect dataset layout. Use images/ with text/ or references/, "
        "IAM words*.txt, IAM lines*.txt, or images/ with xml/."
    )


def _add_issue(issues: list[dict[str, Any]], code: str, message: str, sample_id: str | None = None):
    issue: dict[str, Any] = {"code": code, "message": message}
    if sample_id is not None:
        issue["id"] = sample_id
    issues.append(issue)


def _duplicate_groups(values: dict[str, str]) -> list[list[str]]:
    grouped: dict[str, list[str]] = {}
    for sample_id, digest in values.items():
        grouped.setdefault(digest, []).append(sample_id)
    return [sorted(ids) for ids in grouped.values() if len(ids) > 1]


def check_dataset(data_dir: Path, layout: str = "auto") -> dict[str, Any]:
    """Inspect a dataset without writing crops or modifying its files."""
    data_dir = Path(data_dir).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []
    duplicate_image_hashes: dict[str, str] = {}
    duplicate_reference_hashes: dict[str, str] = {}
    image_count = 0
    reference_count = 0
    if layout not in {"auto", "paired", "iam-words", "iam-lines", "iam-forms"}:
        raise ValueError("layout must be auto, paired, iam-words, iam-lines, or iam-forms")
    try:
        if layout == "auto":
            form_images = _files_with_suffix(data_dir / "images", _IMAGE_SUFFIXES)
            if form_images and _files_with_suffix(data_dir / "xml", {".xml"}):
                layout = "iam-forms"
            elif form_images and (
                (data_dir / "text").exists() or (data_dir / "references").exists()
            ):
                layout = "paired"
            else:
                layout = _detect_annotation_layout(data_dir, form_images)
        metadata = _load_metadata(data_dir)
    except ValueError as exc:
        message = str(exc)
        code = "duplicate_id" if "duplicate" in message.lower() else "layout"
        if "ambiguous" in message.lower():
            code = "ambiguous_layout"
        _add_issue(issues, code, message)
        return {
            "valid": False,
            "layout": layout,
            "counts": {
                "samples": 0,
                "issues": len(issues),
                "excluded": 0,
                "images": 0,
                "references": 0,
            },
            "samples": [],
            "issues": issues,
            "excluded": excluded,
            "duplicate_content": {"images": [], "references": []},
            "findings": [],
        }

    def record_sample(
        sample_id: str,
        image_path: Path,
        reference: str,
        reference_bytes: bytes,
        metadata_record: dict[str, Any] | None = None,
    ) -> None:
        try:
            width, height = _validate_image(image_path, sample_id)
        except ValueError as exc:
            _add_issue(issues, "corrupt_image", str(exc), sample_id)
            excluded.append({"id": sample_id, "reason": "corrupt_image", "path": str(image_path)})
            return
        if not reference.strip():
            _add_issue(issues, "empty_reference", "Reference is empty", sample_id)
            excluded.append({"id": sample_id, "reason": "empty_reference", "path": str(image_path)})
            return
        metadata_record = metadata_record or _metadata_for(sample_id, metadata)
        metadata_record["sample_type"] = metadata_record.get("sample_type") or {
            "paired": "line",
            "iam-lines": "line",
            "iam-words": "word",
            "iam-forms": "page",
        }.get(layout)
        image_hash = _sha256_file(image_path)
        reference_hash = _sha256_bytes(reference_bytes)
        duplicate_image_hashes[sample_id] = image_hash
        duplicate_reference_hashes[sample_id] = reference_hash
        samples.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "reference": reference,
                "metadata": metadata_record,
                "sample_type": metadata_record.get("sample_type"),
                "hashes": {"image": image_hash, "reference": reference_hash},
            }
        )

    try:
        if layout == "paired":
            reference_name, images, references = _paired_paths(data_dir)
            image_count, reference_count = len(images), len(references)
            image_ids, reference_ids = set(images), set(references)
            if not image_ids and not reference_ids:
                _add_issue(issues, "empty_dataset", "No image/reference pairs were found")
            for sample_id in sorted(image_ids - reference_ids):
                _add_issue(issues, "missing_reference", f"No {reference_name}/ pair", sample_id)
                excluded.append(
                    {"id": sample_id, "reason": "missing_reference", "path": str(images[sample_id])}
                )
                try:
                    _validate_image(images[sample_id], sample_id)
                except ValueError as exc:
                    _add_issue(issues, "corrupt_image", str(exc), sample_id)
            for sample_id in sorted(reference_ids - image_ids):
                _add_issue(issues, "missing_image", "No matching image", sample_id)
                excluded.append(
                    {"id": sample_id, "reason": "missing_image", "path": str(references[sample_id])}
                )
                try:
                    _read_reference(references[sample_id], sample_id)
                except ValueError as exc:
                    code = "empty_reference" if "empty" in str(exc).lower() else "invalid_reference"
                    _add_issue(issues, code, str(exc), sample_id)
            for sample_id in sorted(image_ids & reference_ids):
                try:
                    reference, reference_bytes = _read_reference(references[sample_id], sample_id)
                except ValueError as exc:
                    code = "empty_reference" if "empty" in str(exc).lower() else "invalid_reference"
                    _add_issue(issues, code, str(exc), sample_id)
                    excluded.append(
                        {"id": sample_id, "reason": code, "path": str(references[sample_id])}
                    )
                    continue
                record_sample(sample_id, images[sample_id], reference, reference_bytes)
        elif layout in {"iam-words", "iam-lines"}:
            images = _all_data_images(data_dir)
            image_count = len(images)
            by_id: dict[str, Path] = {}
            for image in images:
                if image.stem in by_id:
                    _add_issue(
                        issues, "duplicate_id", f"Duplicate image ID {image.stem!r}", image.stem
                    )
                else:
                    by_id[image.stem] = image
            candidates = sorted(
                data_dir.rglob("words*.txt" if layout == "iam-words" else "lines*.txt")
            )
            parser = _word_annotations if layout == "iam-words" else _line_annotations
            matches = []
            for candidate in candidates:
                try:
                    annotations = parser(candidate)
                except ValueError:
                    continue
                overlap = set(annotations) & set(by_id)
                if overlap:
                    matches.append((len(overlap), candidate, annotations))
            if not matches:
                _add_issue(
                    issues, "annotations_missing", f"No matching IAM annotation file for {layout}"
                )
            else:
                _, annotation_path, annotations = max(
                    matches, key=lambda item: (item[0], str(item[1]))
                )
                reference_count = len(annotations)
                override_dirs = list(
                    dict.fromkeys([data_dir / "references", annotation_path.parent / "references"])
                )
                override_paths = [
                    path
                    for directory in override_dirs
                    for path in _files_with_suffix(directory, {".txt"})
                ]
                reference_overrides = (
                    _index_by_stem(override_paths, "reference", ".txt") if override_paths else {}
                )
                matched_ids = set(annotations) & set(by_id)
                for sample_id in sorted(set(annotations) - set(by_id)):
                    _add_issue(
                        issues, "missing_image", "Annotation has no matching image", sample_id
                    )
                for sample_id in sorted(set(by_id) - set(annotations)):
                    excluded.append(
                        {
                            "id": sample_id,
                            "reason": "unannotated_image",
                            "path": str(by_id[sample_id]),
                        }
                    )
                for sample_id in sorted(matched_ids):
                    reference = annotations[sample_id]
                    reference_bytes = reference.encode("utf-8")
                    override = reference_overrides.get(sample_id)
                    if override is not None:
                        try:
                            reference, reference_bytes = _read_reference(override, sample_id)
                        except ValueError as exc:
                            code = (
                                "empty_reference"
                                if "empty" in str(exc).lower()
                                else "invalid_reference"
                            )
                            _add_issue(issues, code, str(exc), sample_id)
                            excluded.append(
                                {"id": sample_id, "reason": code, "path": str(override)}
                            )
                            continue
                    record_sample(sample_id, by_id[sample_id], reference, reference_bytes)
        elif layout == "iam-forms":
            images = _files_with_suffix(data_dir / "images", _IMAGE_SUFFIXES)
            image_count = len(images)
            xml_dir = data_dir / "xml"
            image_by_id: dict[str, Path] = {}
            for image in images:
                if image.stem in image_by_id:
                    _add_issue(
                        issues, "duplicate_id", f"Duplicate image ID {image.stem!r}", image.stem
                    )
                image_by_id.setdefault(image.stem, image)
            xmls = _files_with_suffix(xml_dir, {".xml"})
            reference_count = len(xmls)
            xml_by_id: dict[str, Path] = {}
            for xml in xmls:
                if xml.stem in xml_by_id:
                    _add_issue(issues, "duplicate_id", f"Duplicate XML ID {xml.stem!r}", xml.stem)
                xml_by_id.setdefault(xml.stem, xml)
            refs = _files_with_suffix(data_dir / "references", {".txt"})
            ref_by_id: dict[str, Path] = {}
            for ref in refs:
                if ref.stem in ref_by_id:
                    _add_issue(
                        issues, "duplicate_id", f"Duplicate reference ID {ref.stem!r}", ref.stem
                    )
                ref_by_id.setdefault(ref.stem, ref)
            for sample_id, image in sorted(image_by_id.items()):
                xml_path = xml_by_id.get(sample_id)
                if xml_path is None:
                    _add_issue(issues, "missing_xml", "No matching XML", sample_id)
                    excluded.append({"id": sample_id, "reason": "missing_xml", "path": str(image)})
                    continue
                try:
                    size = _validate_image(image, sample_id)
                    xml_reference, _, writer = _load_xml_sample(xml_path, sample_id, size)
                    ref_path = ref_by_id.get(sample_id)
                    if ref_path:
                        reference, raw = _read_reference(ref_path, sample_id)
                    else:
                        reference, raw = xml_reference, xml_reference.encode("utf-8")
                    sample_metadata = _metadata_for(sample_id, metadata)
                    if sample_metadata.get("writer_id") is None:
                        sample_metadata["writer_id"] = writer
                    record_sample(sample_id, image, reference, raw, sample_metadata)
                except ValueError as exc:
                    _add_issue(issues, "invalid_sample", str(exc), sample_id)
                    excluded.append(
                        {"id": sample_id, "reason": "invalid_sample", "path": str(image)}
                    )
    except (OSError, ValueError) as exc:
        message = str(exc)
        code = "duplicate_id" if "duplicate" in message.lower() else "dataset"
        if "ambiguous" in message.lower():
            code = "ambiguous_layout"
        _add_issue(issues, code, message)

    known_ids = {sample["id"] for sample in samples} | {item["id"] for item in excluded}
    for sample_id in sorted(set(metadata) - known_ids):
        _add_issue(
            issues, "orphan_metadata", "Metadata has no corresponding dataset sample", sample_id
        )

    duplicate_images = _duplicate_groups(duplicate_image_hashes)
    duplicate_references = _duplicate_groups(duplicate_reference_hashes)
    return {
        "valid": not issues,
        "layout": layout,
        "counts": {
            "samples": len(samples),
            "issues": len(issues),
            "excluded": len(excluded),
            "images": image_count,
            "references": reference_count,
        },
        "samples": samples,
        "issues": issues,
        "excluded": excluded,
        "duplicate_content": {"images": duplicate_images, "references": duplicate_references},
        "findings": [{"type": "duplicate_image_content", "ids": ids} for ids in duplicate_images]
        + [{"type": "duplicate_reference_content", "ids": ids} for ids in duplicate_references],
    }


def split_dataset(
    samples: list[dict[str, Any]],
    train: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Assign whole source documents to reproducible data splits.

    Every sample must identify its source document in its top-level fields or
    metadata. Exact duplicate image bytes assigned to different documents are
    rejected as potential leakage.
    """
    ratios = {"train": train, "validation": validation, "test": test}
    if any(value < 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("train, validation, and test ratios must be non-negative and sum to 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    image_document: dict[str, str] = {}
    seen_ids: set[str] = set()
    for sample in samples:
        sample_id = sample.get("id")
        metadata = sample.get("metadata") or {}
        document = sample.get("source_document") or metadata.get("source_document")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Every sample needs a non-empty id")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID {sample_id!r}")
        seen_ids.add(sample_id)
        if not isinstance(document, str) or not document.strip():
            raise ValueError(
                f"Sample {sample_id!r} needs source_document metadata before document-level splitting"
            )
        document = document.strip()
        groups.setdefault(document, []).append(sample)
        image_hash = (sample.get("hashes") or {}).get("image")
        if not image_hash:
            image_path = sample.get("image_path")
            if not isinstance(image_path, str) or not Path(image_path).is_file():
                raise ValueError(
                    f"Sample {sample_id!r} needs an image hash or readable image_path for leakage checks"
                )
            image_hash = _sha256_file(Path(image_path))
        if image_hash:
            previous_document = image_document.get(image_hash)
            if previous_document is not None and previous_document != document:
                raise ValueError(
                    f"Duplicate image content crosses documents {previous_document!r} and {document!r}"
                )
            image_document[image_hash] = document
    if not groups:
        raise ValueError("Cannot split an empty dataset")
    if len(groups) < 3 and all(ratios[name] > 0 for name in ratios):
        raise ValueError(
            "At least three source documents are required for non-empty train/validation/test splits"
        )
    randomizer = random.Random(seed)
    document_names = list(groups)
    randomizer.shuffle(document_names)
    tie_order = list(ratios)
    randomizer.shuffle(tie_order)
    assigned: dict[str, str] = {}
    counts = {name: 0 for name in ratios}
    targets = {name: len(samples) * ratio for name, ratio in ratios.items()}
    # Seed each requested split once, then place larger remaining documents
    # where they reduce the distance from the target sample counts.
    initial_documents = document_names[: len([name for name in ratios if ratios[name] > 0])]
    for document, split_name in zip(
        initial_documents, [name for name in tie_order if ratios[name] > 0], strict=True
    ):
        assigned[document] = split_name
        counts[split_name] += len(groups[document])
    remaining = document_names[len(initial_documents) :]
    remaining.sort(key=lambda document: (-len(groups[document]), document))
    for document in remaining:
        split_name = max(
            (name for name in tie_order if ratios[name] > 0),
            key=lambda name: targets[name] - counts[name],
        )
        assigned[document] = split_name
        counts[split_name] += len(groups[document])
    result = []
    for document, group in groups.items():
        for sample in group:
            metadata = sample.get("metadata") or {}
            result.append(
                {
                    "id": sample["id"],
                    "source_document": document,
                    "split": assigned[document],
                    "writer_id": sample.get("writer_id", metadata.get("writer_id")),
                    "content_type": sample.get("content_type", metadata.get("content_type")),
                    "difficulty": sample.get("difficulty", metadata.get("difficulty")),
                    "verification_status": sample.get(
                        "verification_status", metadata.get("verification_status")
                    ),
                    "sample_type": sample.get("sample_type", metadata.get("sample_type")),
                }
            )
    return sorted(result, key=lambda record: record["id"])


def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    seed: int = 42,
    write_references: bool = False,
    layout: str = "auto",
    preprocess: str = "original",
    split: str | None = None,
    content_type: str | None = None,
) -> list[dict[str, Any]]:
    """Validate and prepare generic paired or IAM handwriting data.

    Paired data uses ``images/`` with matching relative paths under ``text/``
    or ``references/``. IAM layouts are selected automatically or by name.
    ``original`` preserves input pixels; ``enhanced`` applies grayscale,
    contrast normalization, 3x enlargement, and white padding. A prepared
    manifest freezes selected IDs, references, hashes, and metadata.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer when provided")
    if preprocess not in {"original", "enhanced"}:
        raise ValueError("preprocess must be 'original' or 'enhanced'")
    if split is not None and split not in _SPLITS:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    if content_type is not None and not content_type.strip():
        raise ValueError("content_type must be non-empty when provided")

    image_dir = data_dir / "images"
    xml_dir = data_dir / "xml"
    reference_dir = data_dir / "references"
    form_images = _files_with_suffix(image_dir, _IMAGE_SUFFIXES)
    form_xml = _files_with_suffix(xml_dir, {".xml"})
    if layout not in {"auto", "paired", "iam-words", "iam-lines", "iam-forms"}:
        raise ValueError("layout must be auto, paired, iam-words, iam-lines, or iam-forms")
    selected_layout = layout
    if selected_layout == "auto":
        if form_images and form_xml:
            selected_layout = "iam-forms"
        elif form_images and ((data_dir / "text").exists() or reference_dir.exists()):
            selected_layout = "paired"
        else:
            selected_layout = _detect_annotation_layout(data_dir, form_images, output_dir)
    if selected_layout == "paired":
        return _prepare_paired_dataset(
            data_dir, output_dir, limit, seed, preprocess, split, content_type
        )
    if selected_layout == "iam-lines":
        return _prepare_line_dataset(
            data_dir, output_dir, limit, seed, preprocess, split, content_type
        )
    if selected_layout == "iam-words":
        return _prepare_word_dataset(
            data_dir,
            output_dir,
            limit,
            seed,
            write_references,
            preprocess,
            split,
            content_type,
        )
    # IAM forms retain a useful missing-XML diagnostic when images/ was supplied.
    if reference_dir.exists() and not reference_dir.is_dir():
        raise ValueError(f"Reference path is not a directory: {reference_dir}")

    image_paths = form_images
    if not image_paths:
        raise ValueError(f"No PNG, JPG, or JPEG images found in {image_dir}")

    image_by_id: dict[str, Path] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_path in image_paths:
        sample_id = image_path.stem
        if sample_id in image_by_id:
            raise ValueError(
                f"Duplicate image ID {sample_id!r}: {image_by_id[sample_id]} and {image_path}"
            )
        image_by_id[sample_id] = image_path
        # Every image is decoded here, before any subset is selected.
        image_sizes[sample_id] = _validate_image(image_path, sample_id)

    xml_by_id = _index_by_stem(_files_with_suffix(xml_dir, {".xml"}), "XML", ".xml")
    reference_by_id = _index_by_stem(
        _files_with_suffix(reference_dir, {".txt"}), "reference", ".txt"
    )

    metadata_by_id = _load_metadata(data_dir)
    unknown_metadata = sorted(set(metadata_by_id) - set(image_by_id))
    if unknown_metadata:
        raise ValueError(f"Metadata refers to unknown sample ID {unknown_metadata[0]!r}")
    prepared: list[dict[str, Any]] = []
    for sample_id in sorted(image_by_id):
        image_path = image_by_id[sample_id]
        xml_path = xml_by_id.get(sample_id)
        if xml_path is None:
            raise ValueError(f"Missing XML for image ID {sample_id!r}")

        xml_reference, union_box, writer_id = _load_xml_sample(
            xml_path, sample_id, image_sizes[sample_id]
        )
        reference_path = reference_by_id.get(sample_id)
        if reference_path is not None:
            reference, reference_bytes = _read_reference(reference_path, sample_id)
            reference_source = "txt"
        else:
            reference = xml_reference
            reference_bytes = reference.encode("utf-8")
            reference_source = "xml"
        if not reference.strip():
            raise ValueError(f"Reference for sample {sample_id!r} is empty")
        sample_metadata = _metadata_for(sample_id, metadata_by_id)
        sample_metadata["sample_type"] = sample_metadata.get("sample_type") or "page"
        if sample_metadata.get("writer_id") is not None:
            writer_id = sample_metadata["writer_id"]

        prepared.append(
            {
                "id": sample_id,
                "image_path": image_path,
                "xml_path": xml_path,
                "reference_path": reference_path,
                "reference": reference,
                "reference_bytes": reference_bytes,
                "reference_source": reference_source,
                "writer_id": writer_id,
                "metadata": sample_metadata,
                "union_box": union_box,
            }
        )

    prepared = _filter_metadata_records(prepared, split, content_type)
    all_prepared = prepared
    if limit is not None:
        if limit > len(prepared):
            raise ValueError(f"Requested {limit} samples, but only {len(prepared)} are available")
        selected_ids = set(random.Random(seed).sample([item["id"] for item in prepared], limit))
        prepared = [item for item in prepared if item["id"] in selected_ids]

    # Reference files are generated only after the full dataset has passed
    # validation, and exclusive creation guarantees existing files are kept.
    missing_reference_items = [
        item for item in all_prepared if write_references and item["reference_source"] == "xml"
    ]
    if missing_reference_items:
        reference_dir.mkdir(parents=True, exist_ok=True)
        for item in missing_reference_items:
            path = reference_dir / f"{item['id']}.txt"
            try:
                with path.open("x", encoding="utf-8", newline="") as stream:
                    stream.write(item["reference"])
            except FileExistsError:
                # Never overwrite a reference created by another process.
                pass

    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for item in prepared:
        sample_id = item["id"]
        image_path: Path = item["image_path"]
        xml_path: Path = item["xml_path"]
        width, height = image_sizes[sample_id]
        x1, y1, x2, y2 = item["union_box"]
        crop_bbox = (
            max(0, x1 - _PADDING),
            max(0, y1 - _PADDING),
            min(width, x2 + _PADDING),
            min(height, y2 + _PADDING),
        )
        if crop_bbox[0] >= crop_bbox[2] or crop_bbox[1] >= crop_bbox[3]:
            raise ValueError(f"Sample {sample_id!r} has an empty crop after clipping")

        crop_path = crop_dir / f"{sample_id}.png"
        try:
            _save_model_crop(image_path, crop_path, crop_bbox, preprocess)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise ValueError(f"Cannot crop image for sample {sample_id!r}: {exc}") from exc

        sample_metadata = item["metadata"]
        results.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "crop_path": str(crop_path.resolve()),
                "reference": item["reference"],
                "reference_source": item["reference_source"],
                "writer_id": item["writer_id"],
                "source_document": sample_metadata.get("source_document"),
                "split": sample_metadata.get("split"),
                "content_type": sample_metadata.get("content_type"),
                "sample_type": sample_metadata.get("sample_type"),
                "difficulty": sample_metadata.get("difficulty"),
                "verification_status": sample_metadata.get("verification_status"),
                "metadata": sample_metadata,
                "preprocess": preprocess,
                "crop_bbox": list(crop_bbox),
                "hashes": {
                    "image": _sha256_file(image_path),
                    "xml": _sha256_file(xml_path),
                    "reference": _sha256_bytes(item["reference_bytes"]),
                    "crop": _sha256_file(crop_path),
                    "metadata": _sha256_bytes(
                        json.dumps(sample_metadata, sort_keys=True, ensure_ascii=False).encode(
                            "utf-8"
                        )
                    ),
                },
            }
        )
    _write_prepared_manifest(
        output_dir,
        results,
        layout="iam-forms",
        preprocess=preprocess,
        seed=seed,
        limit=limit,
    )
    return results
