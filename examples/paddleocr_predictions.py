"""Generate an external-prediction JSONL file with the PaddleOCR 3.x pipeline.

Reproducible CPU example (creates an isolated uv environment, not a host install):

    uv run --with "paddleocr==3.7.0" --with "paddlepaddle==3.2.0" \\
      python examples/paddleocr_predictions.py --prepared SNAPSHOT --output paddle.jsonl

The script checks those pinned runtime versions by default and records them,
model identifiers, disabled orientation/unwarping transforms, the prepared
snapshot preprocessing, line/page/word scope, and text joining policy. The
official PaddleOCR API uses ``PaddleOCR(...).predict(image_path)`` and returns
result objects containing recognized ``rec_texts``; those are retained in the
reported order. First use may download the pinned named model checkpoints.
Import with ``vlm-bench import --system paddleocr --provenance-file FILE``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from vlm_bench.snapshot import load

PINNED_PADDLEOCR_VERSION = "3.7.0"
PINNED_PADDLE_VERSION = "3.2.0"
_MODEL_SIZES = {"medium", "small", "tiny"}
_SAMPLE_TYPES = {"line", "word", "page"}
_SEPARATORS = {"line": " ", "word": "", "page": "\n"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepared", type=Path, required=True, help="Validated frozen snapshot")
    result.add_argument("--output", type=Path, required=True, help="New prediction JSONL path")
    result.add_argument("--provenance-output", type=Path)
    result.add_argument("--scope", choices=["auto", "line", "word", "page"], default="auto")
    result.add_argument("--lang", default="en", help="PaddleOCR language code")
    result.add_argument("--device", default="cpu", help="PaddleOCR device, e.g. cpu or gpu:0")
    result.add_argument("--model-size", choices=sorted(_MODEL_SIZES), default="medium")
    result.add_argument("--expected-paddleocr-version", default=PINNED_PADDLEOCR_VERSION)
    result.add_argument("--expected-paddle-version", default=PINNED_PADDLE_VERSION)
    return result


def _sample_scope(sample: dict[str, Any], requested_scope: str) -> str:
    if requested_scope != "auto":
        return requested_scope
    metadata = sample.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    sample_type = sample.get("sample_type", metadata.get("sample_type"))
    if not isinstance(sample_type, str) or sample_type.strip().lower() not in _SAMPLE_TYPES:
        raise ValueError(
            f"Sample {sample.get('id')!r} needs sample_type line, word, or page; "
            "specify --scope to override it"
        )
    return sample_type.strip().lower()


def _provenance_path(output: Path, requested: Path | None) -> Path:
    return (
        requested if requested is not None else output.with_name(output.name + ".provenance.json")
    )


def _check_new_outputs(output: Path, provenance: Path) -> None:
    if output.resolve() == provenance.resolve():
        raise ValueError("prediction output and provenance output must be different paths")
    if output.exists() or provenance.exists():
        existing = output if output.exists() else provenance
        raise FileExistsError(f"Refusing to overwrite existing output: {existing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)


def _write_outputs(
    output: Path, provenance: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    created_output = False
    created_provenance = False
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            created_output = True
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with provenance.open("x", encoding="utf-8", newline="\n") as stream:
            created_provenance = True
            json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except Exception:
        if created_output:
            output.unlink(missing_ok=True)
        if created_provenance:
            provenance.unlink(missing_ok=True)
        raise


def _recognized_text(result: Any, separator: str) -> str:
    payload = getattr(result, "json", None)
    if not isinstance(payload, dict):
        raise ValueError("PaddleOCR result.json must be an object")
    body = payload.get("res")
    if not isinstance(body, dict):
        raise ValueError("PaddleOCR result.json has no res object")
    texts = body.get("rec_texts")
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise ValueError("PaddleOCR result res.rec_texts must be a list of strings")
    return separator.join(texts)


def _import_runtime() -> tuple[Any, str, str]:
    try:
        import paddle
        import paddleocr
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR example requires its pinned optional runtime. Run with "
            '`uv run --with "paddleocr==3.7.0" --with "paddlepaddle==3.2.0" '
            "python examples/paddleocr_predictions.py ...`"
        ) from exc
    return PaddleOCR, str(paddleocr.__version__), str(paddle.__version__)


def _new_pipeline(
    factory: Callable[..., Any], *, language: str, device: str, model_size: str
) -> Any:
    if model_size not in _MODEL_SIZES:
        raise ValueError(f"unsupported PaddleOCR model size: {model_size!r}")
    return factory(
        lang=language,
        ocr_version="PP-OCRv6",
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name=f"PP-OCRv6_{model_size}_det",
        text_recognition_model_name=f"PP-OCRv6_{model_size}_rec",
        text_det_limit_side_len=736,
        text_det_limit_type="min",
        text_det_thresh=0.3,
        text_det_box_thresh=0.6,
        text_det_unclip_ratio=1.5,
        text_rec_score_thresh=0.0,
    )


def generate(
    prepared: Path,
    output: Path,
    *,
    provenance_output: Path | None = None,
    scope: str = "auto",
    language: str = "en",
    device: str = "cpu",
    model_size: str = "medium",
    expected_paddleocr_version: str = PINNED_PADDLEOCR_VERSION,
    expected_paddle_version: str = PINNED_PADDLE_VERSION,
    factory: Callable[..., Any] | None = None,
    paddleocr_version: str | None = None,
    paddle_version: str | None = None,
    stderr: Any = sys.stderr,
) -> tuple[Path, Path]:
    """Create prediction JSONL plus its provenance sidecar without overwriting."""
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be non-empty")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be non-empty")
    if scope not in {"auto", *_SAMPLE_TYPES}:
        raise ValueError("scope must be auto, line, word, or page")
    if model_size not in _MODEL_SIZES:
        raise ValueError(f"model_size must be one of {sorted(_MODEL_SIZES)}")

    manifest = load(prepared)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Prepared snapshot contains no samples")
    sidecar = _provenance_path(output, provenance_output)
    _check_new_outputs(output, sidecar)

    if factory is None or paddleocr_version is None or paddle_version is None:
        factory, runtime_ocr_version, runtime_paddle_version = _import_runtime()
        paddleocr_version = paddleocr_version or runtime_ocr_version
        paddle_version = paddle_version or runtime_paddle_version
    if paddleocr_version != expected_paddleocr_version:
        raise RuntimeError(
            f"Expected paddleocr=={expected_paddleocr_version}, found {paddleocr_version}"
        )
    if paddle_version != expected_paddle_version:
        raise RuntimeError(
            f"Expected paddlepaddle=={expected_paddle_version}, found {paddle_version}"
        )

    model_started = time.perf_counter()
    pipeline = _new_pipeline(factory, language=language, device=device, model_size=model_size)
    model_initialization_seconds = round(time.perf_counter() - model_started, 6)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    scope_counts: dict[str, int] = {name: 0 for name in sorted(_SAMPLE_TYPES)}
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("id"), str):
            raise ValueError("Prepared snapshot has a malformed sample entry")
        image_path = sample.get("crop_path")
        if not isinstance(image_path, str) or not Path(image_path).is_file():
            raise ValueError(f"Prepared sample {sample['id']!r} has no readable crop path")
        sample_scope = _sample_scope(sample, scope)
        scope_counts[sample_scope] += 1
        started = time.perf_counter()
        try:
            results = list(pipeline.predict(image_path))
            latency = round(time.perf_counter() - started, 6)
            if len(results) != 1:
                raise ValueError(
                    f"PaddleOCR expected one result for one image; received {len(results)}"
                )
            prediction = _recognized_text(results[0], _SEPARATORS[sample_scope])
            row = {
                "sample_id": sample["id"],
                "prediction": prediction,
                "status": "success",
                "latency_seconds": latency,
            }
            error = None
        except Exception as exc:
            latency = round(time.perf_counter() - started, 6)
            row = {
                "sample_id": sample["id"],
                "prediction": "",
                "status": "error",
                "latency_seconds": latency,
            }
            error = str(exc)
            errors.append({"sample_id": sample["id"], "message": error})
            print(f"PaddleOCR failed for {sample['id']}: {error}", file=stderr)
        rows.append(row)

    provenance = {
        "schema_version": 1,
        "system": "paddleocr",
        "implementation": {
            "name": "PaddleOCR",
            "version": paddleocr_version,
            "paddlepaddle_version": paddle_version,
        },
        "runtime": {"python": sys.version.split()[0], "device": device, "engine": "paddle_static"},
        "models": {
            "ocr_version": "PP-OCRv6",
            "model_size": model_size,
            "text_detection_model": f"PP-OCRv6_{model_size}_det",
            "text_recognition_model": f"PP-OCRv6_{model_size}_rec",
            "model_initialization_seconds": model_initialization_seconds,
        },
        "prepared_snapshot": {
            "benchmark_fingerprint": manifest["benchmark_fingerprint"],
            "preprocess": manifest["preprocess"],
            "sample_count": len(samples),
            "sample_ids": [sample["id"] for sample in samples],
        },
        "preprocessing": {
            "snapshot_preprocess": manifest["preprocess"],
            "additional_preprocessing": "PaddleOCR pipeline defaults except orientation and unwarping disabled",
            "text_detection": {
                "limit_side_len": 736,
                "limit_type": "min",
                "threshold": 0.3,
                "box_threshold": 0.6,
                "unclip_ratio": 1.5,
                "recognition_score_threshold": 0.0,
            },
        },
        "task_scope": {
            "requested": scope,
            "samples_by_type": scope_counts,
            "orientation_classification": False,
            "document_unwarping": False,
            "textline_orientation": False,
            "text_join_separator": _SEPARATORS,
            "recognized_text_order": "PaddleOCR result rec_texts order",
        },
        "controls": {"language": language, "device": device},
        "results": {"completed_count": len(rows) - len(errors), "error_count": len(errors)},
        "errors": errors,
    }
    _write_outputs(output, sidecar, rows, provenance)
    return output, sidecar


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    generate(
        args.prepared,
        args.output,
        provenance_output=args.provenance_output,
        scope=args.scope,
        language=args.lang,
        device=args.device,
        model_size=args.model_size,
        expected_paddleocr_version=args.expected_paddleocr_version,
        expected_paddle_version=args.expected_paddle_version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
