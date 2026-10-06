from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_example(name: str) -> Any:
    path = ROOT / "examples" / f"{name}_predictions.py"
    spec = importlib.util.spec_from_file_location(f"{name}_predictions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load example module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(tmp_path: Path, sample_types: list[str]) -> dict[str, Any]:
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"test image bytes")
    return {
        "benchmark_fingerprint": "a" * 64,
        "preprocess": "original",
        "samples": [
            {"id": f"sample-{index}", "crop_path": str(crop), "sample_type": sample_type}
            for index, sample_type in enumerate(sample_types)
        ],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tesseract_generates_rows_and_scope_provenance_without_installing_binary(
    tmp_path: Path,
) -> None:
    module = _load_example("tesseract")
    manifest = _snapshot(tmp_path, ["line", "word", "page"])
    module.load = lambda _path: manifest
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="tesseract 5.4.1\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="recognized\n", stderr="")

    output = tmp_path / "tesseract.jsonl"
    rows_path, provenance_path = module.generate(
        tmp_path / "prepared",
        output,
        run=fake_run,
        locate=lambda _binary: "/mock/bin/tesseract",
    )

    rows = _read_jsonl(rows_path)
    assert rows == [
        {
            "sample_id": f"sample-{index}",
            "prediction": "recognized",
            "status": "success",
            "latency_seconds": rows[index]["latency_seconds"],
        }
        for index in range(3)
    ]
    assert all(set(row) == {"sample_id", "prediction", "status", "latency_seconds"} for row in rows)
    assert [command[command.index("--psm") + 1] for command in commands[1:]] == ["7", "8", "6"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["system"] == "tesseract"
    assert provenance["implementation"]["version"] == "tesseract 5.4.1"
    assert provenance["reproducibility"]["expected_version"] is None
    assert "optionally enforced" in provenance["reproducibility"]["version_policy"]
    assert provenance["prepared_snapshot"]["benchmark_fingerprint"] == "a" * 64
    assert provenance["task_scope"]["samples_by_type"] == {"line": 1, "page": 1, "word": 1}
    assert provenance["results"] == {"completed_count": 3, "error_count": 0}


def test_tesseract_keeps_failed_samples_explicit_and_refuses_overwrite(tmp_path: Path) -> None:
    module = _load_example("tesseract")
    module.load = lambda _path: _snapshot(tmp_path, ["line"])

    def fake_run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="tesseract 5.4.1\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="bad crop")

    output = tmp_path / "failed.jsonl"
    stderr = SimpleNamespace(write=lambda _value: None)
    module.generate(
        tmp_path / "prepared",
        output,
        run=fake_run,
        locate=lambda _binary: "/mock/bin/tesseract",
        stderr=stderr,
    )
    row = _read_jsonl(output)[0]
    assert row["status"] == "error"
    assert row["prediction"] == ""
    metadata = json.loads((tmp_path / "failed.jsonl.provenance.json").read_text())
    assert metadata["results"] == {"completed_count": 0, "error_count": 1}
    assert metadata["errors"] == [{"sample_id": "sample-0", "message": "bad crop"}]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.generate(
            tmp_path / "prepared",
            output,
            run=fake_run,
            locate=lambda _binary: "/mock/bin/tesseract",
        )


def test_tesseract_can_enforce_an_exact_recorded_version(tmp_path: Path) -> None:
    module = _load_example("tesseract")
    module.load = lambda _path: _snapshot(tmp_path, ["line"])

    def fake_run(_command: list[str], **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="tesseract 5.4.1\n", stderr="")

    with pytest.raises(RuntimeError, match="Expected Tesseract version"):
        module.generate(
            tmp_path / "prepared",
            tmp_path / "version-guard.jsonl",
            expected_version="tesseract 5.5.0",
            run=fake_run,
            locate=lambda _binary: "/mock/bin/tesseract",
        )
    assert not (tmp_path / "version-guard.jsonl").exists()


def test_paddleocr_uses_explicit_controls_and_task_specific_joining(tmp_path: Path) -> None:
    module = _load_example("paddleocr")
    manifest = _snapshot(tmp_path, ["line", "word", "page"])
    module.load = lambda _path: manifest
    factory_calls: list[dict[str, Any]] = []
    predictions = iter([["alpha", "beta"], ["7", "x"], ["one", "two"]])

    class FakePipeline:
        def predict(self, _image_path: str) -> list[Any]:
            return [SimpleNamespace(json={"res": {"rec_texts": next(predictions)}})]

    def factory(**kwargs: Any) -> FakePipeline:
        factory_calls.append(kwargs)
        return FakePipeline()

    output = tmp_path / "paddle.jsonl"
    rows_path, provenance_path = module.generate(
        tmp_path / "prepared",
        output,
        factory=factory,
        paddleocr_version=module.PINNED_PADDLEOCR_VERSION,
        paddle_version=module.PINNED_PADDLE_VERSION,
    )

    rows = _read_jsonl(rows_path)
    assert [row["prediction"] for row in rows] == ["alpha beta", "7x", "one\ntwo"]
    assert [row["sample_id"] for row in rows] == ["sample-0", "sample-1", "sample-2"]
    assert all(row["status"] == "success" for row in rows)
    assert all(set(row) == {"sample_id", "prediction", "status", "latency_seconds"} for row in rows)
    assert factory_calls == [
        {
            "lang": "en",
            "ocr_version": "PP-OCRv6",
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_detection_model_name": "PP-OCRv6_medium_det",
            "text_recognition_model_name": "PP-OCRv6_medium_rec",
            "text_det_limit_side_len": 736,
            "text_det_limit_type": "min",
            "text_det_thresh": 0.3,
            "text_det_box_thresh": 0.6,
            "text_det_unclip_ratio": 1.5,
            "text_rec_score_thresh": 0.0,
        }
    ]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["implementation"]["version"] == module.PINNED_PADDLEOCR_VERSION
    assert provenance["implementation"]["paddlepaddle_version"] == module.PINNED_PADDLE_VERSION
    assert provenance["prepared_snapshot"]["preprocess"] == "original"
    assert provenance["task_scope"]["text_join_separator"] == {
        "line": " ",
        "word": "",
        "page": "\n",
    }


def test_paddleocr_records_pipeline_errors_and_rejects_unpinned_runtime(tmp_path: Path) -> None:
    module = _load_example("paddleocr")
    module.load = lambda _path: _snapshot(tmp_path, ["page"])

    class FailingPipeline:
        def predict(self, _image_path: str) -> list[Any]:
            raise RuntimeError("inference failed")

    output = tmp_path / "paddle-error.jsonl"
    stderr = SimpleNamespace(write=lambda _value: None)
    module.generate(
        tmp_path / "prepared",
        output,
        factory=lambda **_kwargs: FailingPipeline(),
        paddleocr_version=module.PINNED_PADDLEOCR_VERSION,
        paddle_version=module.PINNED_PADDLE_VERSION,
        stderr=stderr,
    )
    assert _read_jsonl(output)[0]["status"] == "error"
    metadata = json.loads((tmp_path / "paddle-error.jsonl.provenance.json").read_text())
    assert metadata["errors"] == [{"sample_id": "sample-0", "message": "inference failed"}]

    with pytest.raises(RuntimeError, match="Expected paddleocr=="):
        module.generate(
            tmp_path / "prepared",
            tmp_path / "bad-version.jsonl",
            factory=lambda **_kwargs: pytest.fail("factory must not be initialized"),
            paddleocr_version="0.0.0",
            paddle_version=module.PINNED_PADDLE_VERSION,
        )
    assert not (tmp_path / "bad-version.jsonl").exists()


@pytest.mark.parametrize("name", ["tesseract", "paddleocr"])
def test_baseline_examples_help_does_not_require_ocr_runtime(name: str) -> None:
    module = _load_example(name)
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0
