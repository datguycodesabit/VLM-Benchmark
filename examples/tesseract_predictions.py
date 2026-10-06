"""Generate an external-prediction JSONL file from a prepared snapshot.

Requires an existing Tesseract executable (no binary is installed by this
example). The saved provenance captures its reported version, language, OEM,
page segmentation, snapshot preprocessing, and task scope. Import the outputs
with ``vlm-bench import --system tesseract --provenance-file FILE``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from vlm_bench.snapshot import load

_PSM_DEFAULTS = {"line": 7, "word": 8, "page": 6}
_SAMPLE_TYPES = set(_PSM_DEFAULTS)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepared", type=Path, required=True, help="Validated frozen snapshot")
    result.add_argument("--output", type=Path, required=True, help="New prediction JSONL path")
    result.add_argument("--provenance-output", type=Path)
    result.add_argument("--binary", default="tesseract", help="Existing Tesseract binary")
    result.add_argument(
        "--expected-version",
        help="Require the exact first line reported by --version (otherwise record the actual build)",
    )
    result.add_argument("--lang", default="eng", help="Tesseract language code")
    result.add_argument("--oem", type=int, choices=range(0, 4), default=1)
    result.add_argument("--psm-line", type=int, choices=range(0, 14), default=7)
    result.add_argument("--psm-word", type=int, choices=range(0, 14), default=8)
    result.add_argument("--psm-page", type=int, choices=range(0, 14), default=6)
    result.add_argument(
        "--scope",
        choices=["auto", "line", "word", "page"],
        default="auto",
        help="Use metadata sample_type with separate PSM values, or override all samples",
    )
    result.add_argument("--timeout", type=float, default=120.0, help="Per-sample timeout seconds")
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


def _run_sample(
    binary: str,
    image_path: str,
    language: str,
    oem: int,
    psm: int,
    timeout: float,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> tuple[dict[str, Any], str | None]:
    command = [
        binary,
        image_path,
        "stdout",
        "--oem",
        str(oem),
        "--psm",
        str(psm),
        "-l",
        language,
    ]
    started = time.perf_counter()
    try:
        completed = run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        return (
            {"prediction": "", "status": "error", "latency_seconds": round(elapsed, 6)},
            f"Tesseract timed out after {timeout:g} seconds",
        )
    except OSError as exc:
        elapsed = time.perf_counter() - started
        return (
            {"prediction": "", "status": "error", "latency_seconds": round(elapsed, 6)},
            f"Tesseract process failed: {exc}",
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        return (
            {"prediction": "", "status": "error", "latency_seconds": round(elapsed, 6)},
            stderr or f"Tesseract exited with status {completed.returncode}",
        )
    return (
        {
            "prediction": (completed.stdout or "").rstrip("\r\n"),
            "status": "success",
            "latency_seconds": round(elapsed, 6),
        },
        None,
    )


def generate(
    prepared: Path,
    output: Path,
    *,
    provenance_output: Path | None = None,
    binary: str = "tesseract",
    expected_version: str | None = None,
    language: str = "eng",
    oem: int = 1,
    psm_line: int = 7,
    psm_word: int = 8,
    psm_page: int = 6,
    scope: str = "auto",
    timeout: float = 120.0,
    run: Callable[..., Any] = subprocess.run,
    locate: Callable[[str], str | None] = shutil.which,
    stderr: Any = sys.stderr,
) -> tuple[Path, Path]:
    """Create prediction JSONL plus its provenance sidecar without overwriting."""
    if not isinstance(binary, str) or not binary.strip():
        raise ValueError("binary must be non-empty")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be non-empty")
    if isinstance(oem, bool) or not isinstance(oem, int) or oem not in range(4):
        raise ValueError("oem must be an integer from 0 through 3")
    if scope not in {"auto", *_SAMPLE_TYPES}:
        raise ValueError("scope must be auto, line, word, or page")
    for name, value in (("psm_line", psm_line), ("psm_word", psm_word), ("psm_page", psm_page)):
        if isinstance(value, bool) or not isinstance(value, int) or value not in range(14):
            raise ValueError(f"{name} must be an integer from 0 through 13")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be positive")
    executable = locate(binary)
    if executable is None:
        raise RuntimeError(f"Tesseract executable {binary!r} was not found on PATH")
    version_result = run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if version_result.returncode != 0:
        raise RuntimeError(
            f"Cannot query Tesseract version: {(version_result.stderr or '').strip()}"
        )
    version_text = (version_result.stdout or version_result.stderr or "").strip()
    if not version_text:
        raise RuntimeError("Tesseract returned an empty version string")
    reported_version = version_text.splitlines()[0]
    if expected_version is not None and reported_version != expected_version:
        raise RuntimeError(
            f"Expected Tesseract version {expected_version!r}, found {reported_version!r}"
        )

    manifest = load(prepared)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Prepared snapshot contains no samples")
    sidecar = _provenance_path(output, provenance_output)
    _check_new_outputs(output, sidecar)

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
        psm = {"line": psm_line, "word": psm_word, "page": psm_page}[sample_scope]
        row, error = _run_sample(executable, image_path, language, oem, psm, timeout, run=run)
        row["sample_id"] = sample["id"]
        rows.append(row)
        if error:
            errors.append({"sample_id": sample["id"], "message": error})
            print(f"Tesseract failed for {sample['id']}: {error}", file=stderr)

    provenance = {
        "schema_version": 1,
        "system": "tesseract",
        "implementation": {"name": "Tesseract OCR", "version": reported_version},
        "runtime": {"binary": executable, "version_output": version_text},
        "reproducibility": {
            "version_policy": "exact first-line version recorded; optionally enforced with --expected-version",
            "expected_version": expected_version,
        },
        "prepared_snapshot": {
            "benchmark_fingerprint": manifest["benchmark_fingerprint"],
            "preprocess": manifest["preprocess"],
            "sample_count": len(samples),
            "sample_ids": [sample["id"] for sample in samples],
        },
        "preprocessing": {
            "snapshot_preprocess": manifest["preprocess"],
            "additional_preprocessing": "none; reads frozen prepared crops directly",
        },
        "task_scope": {
            "requested": scope,
            "samples_by_type": scope_counts,
            "page_segmentation_modes": {"line": psm_line, "word": psm_word, "page": psm_page},
        },
        "controls": {"language": language, "oem": oem, "timeout_seconds": timeout},
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
        binary=args.binary,
        expected_version=args.expected_version,
        language=args.lang,
        oem=args.oem,
        psm_line=args.psm_line,
        psm_word=args.psm_word,
        psm_page=args.psm_page,
        scope=args.scope,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
