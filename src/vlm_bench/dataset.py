"""Dataset discovery and IAM handwritten-form preparation.

The IAM line selection, signature filtering, bounding-box union, and crop
padding are adapted from ``scripts/crop_handwritten.py`` and
``scripts/generate_ground_truth.py`` in PyaesoneP/vlm-ocr-research (Apache-2.0).
Source snapshot: https://github.com/PyaesoneP/vlm-ocr-research/tree/fd4bd0ae44db0f57f7dcb0e301a0a718d3e6159f/scripts
"""

from __future__ import annotations

import hashlib
import random
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_PADDING = 20
_MODEL_SCALE = 3
_MODEL_BORDER = 16


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


def _save_model_crop(source: Path, destination: Path, bbox: tuple[int, int, int, int]) -> None:
    """Make a consistent, enlarged grayscale crop for a vision model."""
    with Image.open(source) as image:
        crop = image.convert("L").crop(bbox)
        crop = ImageOps.autocontrast(crop)
        crop = crop.resize(
            (crop.width * _MODEL_SCALE, crop.height * _MODEL_SCALE), Image.Resampling.LANCZOS
        )
        crop = ImageOps.expand(crop, border=_MODEL_BORDER, fill=255)
        crop.save(destination, format="PNG")


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
        sample_id, text = fields[0], fields[-1]
        previous = annotations.get(sample_id)
        if previous is not None and previous != text:
            raise ValueError(f"Conflicting word annotations for {sample_id!r} in {path}")
        annotations[sample_id] = text
    if not annotations:
        raise ValueError(f"IAM word annotation file contains no samples: {path}")
    return annotations


def _line_annotations(path: Path) -> dict[str, str]:
    """Read IAM's ASCII line table (ID, status, geometry, transcription)."""
    annotations: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read IAM line annotations: {path}: {exc}") from exc
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        # ID, status, graylevel, components, x, y, width, height, text...
        if len(fields) < 9:
            continue
        try:
            [int(value) for value in fields[2:8]]
        except ValueError:
            continue
        sample_id, text = fields[0], " ".join(fields[8:]).strip()
        if not text:
            continue
        previous = annotations.get(sample_id)
        if previous is not None and previous != text:
            raise ValueError(f"Conflicting line annotations for {sample_id!r} in {path}")
        annotations[sample_id] = text
    if not annotations:
        raise ValueError(f"IAM line annotation file contains no samples: {path}")
    return annotations


def _prepare_line_dataset(
    data_dir: Path,
    output_dir: Path,
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Prepare IAM line images paired with lines.txt transcriptions."""
    all_images = _files_with_suffix(data_dir, _IMAGE_SUFFIXES)
    if output_dir.is_relative_to(data_dir):
        all_images = [path for path in all_images if output_dir not in path.parents]
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

    override_dirs = [data_dir / "references", annotation_path.parent / "references"]
    override_paths: list[Path] = []
    for directory in override_dirs:
        override_paths.extend(_files_with_suffix(directory, {".txt"}))
    reference_by_id = _index_by_stem(override_paths, "reference", ".txt") if override_paths else {}

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
        records.append(
            {
                "id": sample_id,
                "image_path": image_path,
                "reference": reference,
                "reference_bytes": reference_bytes,
                "reference_source": reference_source,
                "writer_id": sample_id.split("-", 1)[0],
                "crop_bbox": [0, 0, width, height],
                "annotation_path": annotation_path,
            }
        )

    if not records:
        raise ValueError("No valid IAM line images matched lines.txt")
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
        _save_model_crop(record["image_path"], crop_path, tuple(record["crop_bbox"]))
        results.append(
            {
                "id": record["id"],
                "image_path": str(record["image_path"]),
                "crop_path": str(crop_path.resolve()),
                "reference": record["reference"],
                "reference_source": record["reference_source"],
                "writer_id": record["writer_id"],
                "crop_bbox": record["crop_bbox"],
                "hashes": {
                    "image": _sha256_file(record["image_path"]),
                    "xml": _sha256_file(record["annotation_path"]),
                    "reference": _sha256_bytes(record["reference_bytes"]),
                    "crop": _sha256_file(crop_path),
                },
            }
        )
    return results


def _prepare_word_dataset(
    data_dir: Path,
    output_dir: Path,
    limit: int | None,
    seed: int,
    write_references: bool,
) -> list[dict[str, Any]]:
    """Prepare pasted IAM word images when no form/XML layout is present."""
    all_images = _files_with_suffix(data_dir, _IMAGE_SUFFIXES)
    if output_dir.is_relative_to(data_dir):
        all_images = [path for path in all_images if output_dir not in path.parents]
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

    override_dirs = [data_dir / "references", annotation_path.parent / "references"]
    override_paths: list[Path] = []
    for directory in override_dirs:
        override_paths.extend(_files_with_suffix(directory, {".txt"}))
    reference_by_id = _index_by_stem(override_paths, "reference", ".txt") if override_paths else {}

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
        records.append(
            {
                "id": sample_id,
                "image_path": image_path,
                "reference": reference,
                "reference_bytes": reference_bytes,
                "reference_source": reference_source,
                "reference_path": reference_path,
                "writer_id": sample_id.split("-", 1)[0],
                "crop_bbox": [0, 0, width, height],
                "annotation_path": annotation_path,
            }
        )

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
        _save_model_crop(record["image_path"], crop_path, tuple(record["crop_bbox"]))
        results.append(
            {
                "id": record["id"],
                "image_path": str(record["image_path"]),
                "crop_path": str(crop_path.resolve()),
                "reference": record["reference"],
                "reference_source": record["reference_source"],
                "writer_id": record["writer_id"],
                "crop_bbox": record["crop_bbox"],
                "hashes": {
                    "image": _sha256_file(record["image_path"]),
                    "xml": _sha256_file(record["annotation_path"]),
                    "reference": _sha256_bytes(record["reference_bytes"]),
                    "crop": _sha256_file(crop_path),
                },
            }
        )
    return results


def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    seed: int = 42,
    write_references: bool = False,
) -> list[dict[str, Any]]:
    """Validate, reproducibly select, and crop IAM handwriting samples.

    Form inputs are ``images/``, ``xml/``, and optionally ``references/`` beneath
    ``data_dir``. If no form images are present, the function auto-detects a
    pasted IAM line archive (``lines*.txt`` plus matching images) or word
    archive (``words*.txt`` plus matching images) recursively. Every model crop
    is converted to grayscale, contrast-normalized, enlarged 3x, and padded
    with a white border so small handwriting is easier for vision models to
    read. The returned ``crop_bbox`` records the source-image coordinates.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    image_dir = data_dir / "images"
    xml_dir = data_dir / "xml"
    reference_dir = data_dir / "references"

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer when provided")
    form_images = _files_with_suffix(image_dir, _IMAGE_SUFFIXES)
    form_xml = _files_with_suffix(xml_dir, {".xml"})
    # A populated images/ directory is an explicit form dataset request. Keep
    # its missing-XML error useful; only fall back to word auto-detection when
    # no form images were supplied at all.
    if not form_images:
        line_files = sorted(data_dir.rglob("lines*.txt"))
        if line_files:
            try:
                return _prepare_line_dataset(data_dir, output_dir, limit, seed)
            except ValueError as line_error:
                # A non-IAM lines file should not hide a valid word archive.
                if not any(data_dir.rglob("words*.txt")):
                    raise line_error
        if not xml_dir.exists() or not form_xml:
            return _prepare_word_dataset(data_dir, output_dir, limit, seed, write_references)
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
                "union_box": union_box,
            }
        )

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
            _save_model_crop(image_path, crop_path, crop_bbox)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise ValueError(f"Cannot crop image for sample {sample_id!r}: {exc}") from exc

        results.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "crop_path": str(crop_path.resolve()),
                "reference": item["reference"],
                "reference_source": item["reference_source"],
                "writer_id": item["writer_id"],
                "crop_bbox": list(crop_bbox),
                "hashes": {
                    "image": _sha256_file(image_path),
                    "xml": _sha256_file(xml_path),
                    "reference": _sha256_bytes(item["reference_bytes"]),
                    "crop": _sha256_file(crop_path),
                },
            }
        )

    return results
