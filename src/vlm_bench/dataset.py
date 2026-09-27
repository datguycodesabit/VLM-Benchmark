"""Dataset discovery and IAM handwritten-form preparation.

The IAM line selection, signature filtering, bounding-box union, and crop
padding are adapted from ``scripts/crop_handwritten.py`` and
``scripts/generate_ground_truth.py`` in PyaesoneP/vlm-ocr-research (Apache-2.0).
Source snapshot: https://github.com/PyaesoneP/vlm-ocr-research/tree/fd4bd0ae44db0f57f7dcb0e301a0a718d3e6159f/scripts
"""

from __future__ import annotations

import hashlib
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_PADDING = 20


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


def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    seed: int = 42,
    write_references: bool = False,
) -> list[dict[str, Any]]:
    """Validate, reproducibly select, and crop IAM handwriting samples.

    Expected inputs are ``images/``, ``xml/``, and optionally ``references/``
    beneath ``data_dir``. References in ``references/<ID>.txt`` override the
    text in XML. The returned ``crop_bbox`` is the actual clipped box saved to
    the crop, including 20 pixels of padding where image bounds allow it.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    image_dir = data_dir / "images"
    xml_dir = data_dir / "xml"
    reference_dir = data_dir / "references"

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer when provided")
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if not xml_dir.exists() or not xml_dir.is_dir():
        raise ValueError(f"XML directory does not exist: {xml_dir}")
    if reference_dir.exists() and not reference_dir.is_dir():
        raise ValueError(f"Reference path is not a directory: {reference_dir}")

    image_paths = _files_with_suffix(image_dir, _IMAGE_SUFFIXES)
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
            with Image.open(image_path) as image:
                image.crop(crop_bbox).save(crop_path, format="PNG")
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
