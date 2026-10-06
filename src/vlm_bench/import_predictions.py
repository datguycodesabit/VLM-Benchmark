"""Import externally generated OCR/VLM predictions as immutable benchmark runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .backends import parse_model
from .metrics import score
from .runner import _digest
from .snapshot import copy_inputs, load

_PREDICTION_FIELDS = {
    "sample_id",
    "prediction",
    "status",
    "latency_seconds",
    "usage",
}
_STATUSES = {"success", "error", "unsupported"}
_MAX_TOKEN_COUNT = 2**63 - 1


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON number {value}")


def _validate_json(value: Any, context: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{context} contains an empty or non-string object key")
            _validate_json(item, f"{context}.{key}")
        return
    raise ValueError(f"{context} contains a non-JSON value")


def _read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    try:
        content = path.read_bytes()
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Could not read provenance JSON {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError("Provenance file must contain a non-empty JSON object")
    _validate_json(value, "provenance")
    return value, hashlib.sha256(content).hexdigest()


def _load_provenance(description: str | None, provenance_file: Path | None):
    if (description is None) == (provenance_file is None):
        raise ValueError("Supply exactly one of a provenance description or provenance file")
    if provenance_file is not None:
        provenance, digest = _read_json_object(Path(provenance_file))
        return provenance, digest
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Provenance description must not be empty")
    return {"description": description.strip()}, None


def _read_predictions(path: Path, sample_ids: set[str]):
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read prediction file {path}: {exc}") from exc

    predictions: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid prediction JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Prediction line {line_number} must contain a JSON object")
        unknown = set(row) - _PREDICTION_FIELDS
        if unknown:
            raise ValueError(
                f"Prediction line {line_number} has unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Prediction line {line_number} needs a non-empty string sample_id")
        if sample_id not in sample_ids:
            raise ValueError(f"Prediction line {line_number} has unknown sample_id {sample_id!r}")
        if sample_id in predictions:
            raise ValueError(f"Duplicate prediction for sample_id {sample_id!r}")
        prediction = row.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError(f"Prediction line {line_number} needs a string prediction")
        status = row.get("status", "success")
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError(
                f"Prediction line {line_number} status must be one of: "
                + ", ".join(sorted(_STATUSES))
            )
        latency = row.get("latency_seconds")
        if "latency_seconds" in row and (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or latency < 0
            or not _finite_number(latency)
        ):
            raise ValueError(
                f"Prediction line {line_number} latency_seconds must be finite and non-negative"
            )
        usage = row.get("usage")
        if "usage" in row:
            if not isinstance(usage, dict):
                raise ValueError(f"Prediction line {line_number} usage must be a JSON object")
            _validate_json(usage, f"prediction line {line_number} usage")
            for token_field in ("input_tokens", "output_tokens"):
                token_count = usage.get(token_field)
                if token_count is not None and (
                    isinstance(token_count, bool)
                    or not isinstance(token_count, int)
                    or token_count < 0
                    or token_count > _MAX_TOKEN_COUNT
                ):
                    raise ValueError(
                        f"Prediction line {line_number} usage.{token_field} "
                        "must be a non-negative integer"
                    )
            details = usage.get("input_tokens_details")
            if details is not None and not isinstance(details, dict):
                raise ValueError(
                    f"Prediction line {line_number} usage.input_tokens_details must be an object"
                )
            if isinstance(details, dict) and "cached_tokens" in details:
                cached = details["cached_tokens"]
                if (
                    isinstance(cached, bool)
                    or not isinstance(cached, int)
                    or cached < 0
                    or cached > _MAX_TOKEN_COUNT
                ):
                    raise ValueError(
                        f"Prediction line {line_number} usage.input_tokens_details.cached_tokens "
                        "must be a non-negative integer"
                    )

        predictions[sample_id] = row

    return predictions, hashlib.sha256(content).hexdigest()


def _sample_kind(sample: dict[str, Any]) -> str:
    value = sample.get("content_type") or sample.get("task") or sample.get("track") or "prose"
    normalized = str(value).strip().lower().replace("_", "-")
    return (
        "equation"
        if normalized in {"equation", "math", "mathematics", "formula", "latex"}
        else "prose"
    )


def _finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _external_records(samples, predictions, selector, *, formula_rendering):
    from .research import score_equation
    from .task_metrics import score_task

    records = []
    for sample in samples:
        sample_id = sample["id"]
        imported = predictions.get(sample_id)
        if imported is None:
            continue
        prediction = imported["prediction"]
        reference = sample["reference"]
        status = imported.get("status", "success")
        record: dict[str, Any] = {
            "model": selector,
            "sample_id": sample_id,
            "status": status,
            "prediction": prediction,
            "reference": reference,
            "latency_seconds": imported.get("latency_seconds"),
            "load_duration_seconds": None,
            "truncated": False,
            "error": None,
        }
        if "latency_seconds" in imported:
            record["timing_source"] = "external"
        if "usage" in imported:
            record["usage"] = imported["usage"]
            record["usage_source"] = "external"
        if status == "success":
            try:
                metrics = (
                    score_equation(prediction, reference)
                    if _sample_kind(sample) == "equation"
                    else score(prediction, reference)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot score imported prediction for sample {sample_id!r}: {exc}"
                ) from exc
            metrics["task_metrics"] = score_task(
                prediction,
                reference,
                sample,
                formula_rendering=formula_rendering,
            )
            record["metrics"] = metrics
        else:
            record["metrics"] = None
            record["error"] = f"External prediction status: {status}"
            if status == "unsupported":
                record["unsupported_reason"] = "External system marked this sample unsupported"
        records.append(record)
    return records


def import_predictions(
    prepared: Path,
    predictions_file: Path,
    output: Path,
    *,
    system: str,
    provenance: str | None = None,
    provenance_file: Path | None = None,
    formula_rendering: bool = False,
) -> Path:
    """Create a schema-v2 run from saved predictions and a prepared snapshot.

    Imported predictions are never sent to a backend. Missing IDs remain absent
    from ``results.jsonl`` so normal coverage reporting marks the run incomplete.
    """
    from .eligibility import validate_research
    from .task_metrics import TASK_METRICS_VERSION, preflight_renderer

    if (
        not isinstance(system, str)
        or not system.strip()
        or ":" in system
        or any(ord(character) < 32 or ord(character) == 127 for character in system)
    ):
        raise ValueError("System identity must be a non-empty plain name without ':' or controls")
    system = system.strip()
    selector = f"external:{system}"
    if parse_model(selector) != ("external", system):
        raise ValueError(f"System identity does not form a valid external selector: {system!r}")
    if not isinstance(formula_rendering, bool):
        raise ValueError("formula_rendering must be a boolean")

    requested_output = Path(output).expanduser()
    if not requested_output.name:
        raise ValueError("Output must name a new run directory")
    if os.path.lexists(requested_output):
        raise FileExistsError(f"Import output already exists: {requested_output}")
    output_path = requested_output.parent.resolve() / requested_output.name
    if os.path.lexists(output_path):
        raise FileExistsError(f"Import output already exists: {output_path}")
    prepared_path = Path(prepared).expanduser().resolve()
    if prepared_path == output_path or prepared_path in output_path.parents:
        raise ValueError("Import output cannot be inside the prepared snapshot")

    snapshot = load(prepared_path)
    samples = snapshot.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Prepared snapshot contains no samples")
    sample_ids = set()
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("id"), str):
            raise ValueError("Prepared snapshot contains a malformed sample")
        if sample["id"] in sample_ids:
            raise ValueError(f"Prepared snapshot contains duplicate sample ID {sample['id']!r}")
        sample_ids.add(sample["id"])
    prediction_rows, predictions_digest = _read_predictions(
        Path(predictions_file).expanduser(), sample_ids
    )
    provenance_value, provenance_digest = _load_provenance(provenance, provenance_file)
    renderer = preflight_renderer() if formula_rendering else None
    protocol = snapshot.get("protocol", "document-disjoint")
    if protocol not in {"document-disjoint", "writer-disjoint"}:
        raise ValueError(f"Prepared snapshot has an unsupported research protocol: {protocol!r}")
    eligibility = validate_research(samples, strict_research=False, protocol=protocol)
    records = _external_records(
        samples,
        prediction_rows,
        selector,
        formula_rendering=formula_rendering,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reservation = output_path.with_name(f".{output_path.name}.import-lock")
    try:
        reservation_fd = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Another import is preparing this output (or left a lock): {reservation}"
        ) from exc
    os.close(reservation_fd)
    stage = None
    try:
        if os.path.lexists(output_path):
            raise FileExistsError(f"Import output already exists: {output_path}")
        stage = Path(
            tempfile.mkdtemp(prefix=f".{output_path.name}.import-tmp-", dir=output_path.parent)
        )
        copied_samples, copied_snapshot = copy_inputs(prepared_path, stage)
        for sample in copied_samples:
            sample_path = Path(sample["crop_path"])
            sample["crop_path"] = str(output_path / sample_path.relative_to(stage))

        from .engine import RESEARCH_PROTOCOL_VERSION, SCORING_VERSION
        from .runner import _now

        created_at = _now()
        complete = len(prediction_rows) == len(samples) and all(
            row.get("status", "success") in {"success", "unsupported"}
            for row in prediction_rows.values()
        )
        status = (
            "complete"
            if complete
            else "incomplete"
            if len(prediction_rows) < len(samples)
            else "complete_with_errors"
        )
        manifest = {
            "schema_version": 2,
            "research_protocol_version": RESEARCH_PROTOCOL_VERSION,
            "benchmark_version": __version__,
            "created_at": created_at,
            "updated_at": created_at,
            "status": status,
            "benchmark_fingerprint": copied_snapshot["benchmark_fingerprint"],
            "scoring_version": SCORING_VERSION,
            "task_metrics_version": TASK_METRICS_VERSION,
            "models": [selector],
            "model_info": {
                selector: {
                    "provider": "external",
                    "system": system,
                    "provenance": provenance_value,
                }
            },
            "model_capabilities": {selector: {"provider": "external", "system": system}},
            "samples": copied_samples,
            "data_dir": None,
            "prepared": str(prepared_path),
            "layout": copied_snapshot.get("layout"),
            "preprocess": copied_snapshot["preprocess"],
            "split": copied_snapshot.get("split"),
            "strict_research": False,
            "protocol": protocol,
            "formula_rendering": formula_rendering,
            "formula_renderer": renderer,
            "research": eligibility,
            "source_audit": {
                "status": "unavailable",
                "reason": "Imported predictions use a prepared snapshot; source data was not re-audited",
            },
            "validation_scope": "prepared-snapshot-only",
            "content_type": copied_snapshot.get("content_type"),
            "limit": None,
            "seed": copied_snapshot.get("seed", 42),
            "timeout": None,
            "prompt": None,
            "math_prompt": None,
            "prompt_hashes": {},
            "options": {"source": "external_import"},
            "settings": {},
            "costs": {},
            "warmup": False,
            "provider_controls": {selector: {"source": "external"}},
            "external_predictions": {
                "system": system,
                "selector": selector,
                "provenance": provenance_value,
                "provenance_file_sha256": provenance_digest,
                "predictions_file": Path(predictions_file).name,
                "predictions_sha256": predictions_digest,
                "timing_source": "external"
                if any("latency_seconds" in row for row in prediction_rows.values())
                else None,
                "usage_source": "external"
                if any("usage" in row for row in prediction_rows.values())
                else None,
            },
        }
        manifest["integrity"] = _digest(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"integrity", "status", "updated_at"}
            }
        )
        (stage / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records),
            encoding="utf-8",
        )
        (stage / "warmups.jsonl").write_text("", encoding="utf-8")
        (stage / ".lock").touch()
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if os.path.lexists(output_path):
            raise FileExistsError(f"Import output already exists: {output_path}")
        stage.rename(output_path)
        stage = None
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        reservation.unlink(missing_ok=True)
    return output_path
