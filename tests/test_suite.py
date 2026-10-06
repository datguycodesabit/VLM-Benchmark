from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vlm_bench import config, runner, suite
from vlm_bench.metrics import score
from vlm_bench.snapshot import fingerprint


def _config(tmp_path: Path, *, conditions=None, repeats=1) -> tuple[Path, dict]:
    path = tmp_path / "suite.toml"
    path.write_text("version = 2\n", encoding="utf-8")
    normalized = {
        "version": 2,
        "experiment": {},
        "models": {},
        "costs": {},
        "suite": {
            "version": 2,
            "repeats": repeats,
            "conditions": conditions
            or [
                {
                    "name": "baseline",
                    "models": ["ollama:fake"],
                    "prepared": tmp_path / "prepared",
                    "repeats": repeats,
                    "settings": {},
                    "prose_prompt": None,
                    "math_prompt": None,
                    "warmup": False,
                    "strict_research": False,
                    "protocol": "document-disjoint",
                    "formula_rendering": False,
                }
            ],
        },
    }
    return path, normalized


def _samples():
    return [
        {
            "id": "s1",
            "reference": "abc",
            "hashes": {"crop": hashlib.sha256(b"crop1").hexdigest()},
            "source_document": "doc-a",
            "content_type": "prose",
            "verified": True,
            "split": "test",
        },
        {
            "id": "s2",
            "reference": "def",
            "hashes": {"crop": hashlib.sha256(b"crop2").hexdigest()},
            "source_document": "doc-b",
            "content_type": "prose",
            "verified": True,
            "split": "test",
        },
    ]


def _preview(*_args, **kwargs):
    return {
        "benchmark_fingerprint": fingerprint(_samples(), "original"),
        "prompt_hashes": {"prose": "prompt-hash", "equation": "math-hash"},
        "inference": False,
        "models": [{"selector": "ollama:fake"}],
    }


def _write_fake_run(
    tmp_path: Path,
    condition: dict,
    repetition: int,
    output_dir: Path,
    progress,
    *,
    status="complete",
    fingerprint_value=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"{condition['name']}-{repetition}"
    run_dir.mkdir()
    samples = _samples()
    actual_fingerprint = fingerprint(samples, "original")
    manifest = {
        "schema_version": 2,
        "benchmark_fingerprint": fingerprint_value or actual_fingerprint,
        "preprocess": "original",
        "models": condition["models"],
        "model_info": {model: {"revision": "test"} for model in condition["models"]},
        "samples": samples,
        "settings": condition["settings"],
        "base_url": "http://localhost:11434",
        "timeout": 300,
        "options": {"num_predict": 4096},
        "execution_controls": {
            "max_retries": 2,
            "concurrency": 1,
            "max_requests": None,
            "max_spend_usd": None,
            "cache_dir": None,
        },
        "prompt_hashes": {"prose": "prompt-hash", "equation": "math-hash"},
        "suite_condition": condition["name"],
        "repetition": repetition,
        "repeated_measurement": condition["repeats"] > 1,
        "strict_research": condition["strict_research"],
        "protocol": condition["protocol"],
        "formula_rendering": condition["formula_rendering"],
        "warmup": condition["warmup"],
        "status": status,
    }
    manifest["integrity"] = runner._digest(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"integrity", "status", "updated_at"}
        }
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    predictions = ["abc", "def"] if repetition == 1 else ["axc", "def"]
    rows = [
        {
            "model": model,
            "sample_id": sample["id"],
            "reference": sample["reference"],
            "prediction": prediction,
            "status": "success",
            "metrics": score(prediction, sample["reference"]),
        }
        for model in condition["models"]
        for sample, prediction in zip(samples, predictions, strict=True)
    ]
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "warmups.jsonl").write_text("", encoding="utf-8")
    progress(f"Run directory: {run_dir}")
    return run_dir, manifest


def _install_config(monkeypatch, normalized):
    monkeypatch.setattr(config, "load_config", lambda _path: normalized)


def test_dry_run_enumerates_each_named_repetition_without_output(tmp_path, monkeypatch):
    path, normalized = _config(tmp_path, repeats=3)
    _install_config(monkeypatch, normalized)
    calls = []
    plan = suite.run_suite(
        path,
        tmp_path / "suite-out",
        dry_run=True,
        preview_fn=lambda *args, **kwargs: calls.append((args, kwargs)) or _preview(),
        run_fn=lambda *args, **kwargs: pytest.fail("dry-run must not infer"),
    )
    assert len(calls) == 1
    assert [run["repetition"] for run in plan["conditions"][0]["runs"]] == [1, 2, 3]
    assert plan["conditions"][0]["benchmark_fingerprint"] == fingerprint(_samples(), "original")
    assert not (tmp_path / "suite-out").exists()


def test_preview_receives_configured_costs_for_spend_limit_validation(tmp_path, monkeypatch):
    path, normalized = _config(tmp_path)
    normalized["costs"] = {"api": {"model": "openai:fake", "input_per_million": 1}}
    _install_config(monkeypatch, normalized)
    calls = []
    suite.run_suite(
        path,
        tmp_path / "suite-out",
        dry_run=True,
        max_spend_usd=2.0,
        preview_fn=lambda *args, **kwargs: calls.append(kwargs) or _preview(),
    )
    assert calls[0]["costs"] == normalized["costs"]


def test_repetitions_persist_run_paths_and_preserve_independent_counts(tmp_path, monkeypatch):
    path, normalized = _config(tmp_path, repeats=2)
    _install_config(monkeypatch, normalized)
    output = tmp_path / "suite-out"
    immediate_paths = []

    # Inspect the already-wrapped callback to verify write-through before inference.
    def run_fn_with_callback(_data, _models, run_output, **kwargs):
        condition = normalized["suite"]["conditions"][0]
        repetition = kwargs["repetition"]
        run_dir, _manifest = _write_fake_run(
            tmp_path, condition, repetition, run_output, lambda message: None
        )
        kwargs["progress"](f"Run directory: {run_dir}")
        stored = json.loads((output / "suite.json").read_text(encoding="utf-8"))
        key = suite._run_key(condition["name"], repetition)
        immediate_paths.append(stored["runs"][key]["run_dir"])
        return run_dir

    result = suite.run_suite(
        path,
        output,
        preview_fn=lambda *_args, **_kwargs: _preview(),
        run_fn=run_fn_with_callback,
        progress=lambda _message: None,
        compare_fn=lambda *_args, **_kwargs: pytest.fail("one condition needs no comparison"),
    )
    assert len(immediate_paths) == 2
    variability = result["conditions"]["baseline"]["variability"]["prose"]["ollama:fake"]
    assert variability["scored_repetition_count"] == 2
    assert variability["mean_cer"] == pytest.approx(1 / 12)
    assert variability["sample_stdev_cer"] is not None
    assert variability["independent_sample_count"] == 2
    assert variability["independent_document_count"] == 2
    assert variability["bootstrap_unit_count"] == 2
    assert all(value["bootstrap"] for value in variability["per_repetition"])
    assert len({value["run_id"] for value in variability["per_repetition"]}) == 2


def test_resume_rejects_changed_config_before_preview(tmp_path, monkeypatch):
    path, normalized = _config(tmp_path)
    _install_config(monkeypatch, normalized)
    output = tmp_path / "suite-out"

    def run_fn(_data, _models, run_output, **kwargs):
        return _write_fake_run(
            tmp_path,
            normalized["suite"]["conditions"][0],
            kwargs["repetition"],
            run_output,
            kwargs["progress"],
        )[0]

    suite.run_suite(
        path,
        output,
        preview_fn=lambda *_args, **_kwargs: _preview(),
        run_fn=run_fn,
        progress=lambda _message: None,
        compare_fn=lambda *_args, **_kwargs: {},
    )
    path.write_text("version = 2 # changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config changed"):
        suite.run_suite(
            path,
            output,
            resume=True,
            preview_fn=lambda *_args, **_kwargs: pytest.fail("preview ran before config check"),
        )


def test_resume_skips_completed_runs_and_reuses_comparison_artifacts(tmp_path, monkeypatch):
    path, normalized = _config(
        tmp_path,
        conditions=[
            {
                "name": name,
                "models": ["ollama:fake"],
                "prepared": tmp_path / f"prepared-{name}",
                "repeats": 1,
                "settings": {},
                "prose_prompt": None,
                "math_prompt": None,
                "warmup": False,
                "strict_research": False,
                "protocol": "document-disjoint",
                "formula_rendering": False,
            }
            for name in ("base", "adapted")
        ],
    )
    _install_config(monkeypatch, normalized)
    output = tmp_path / "suite-out"
    run_calls = []
    compare_calls = []

    def run_fn(_data, _models, run_output, **kwargs):
        run_calls.append(kwargs["suite_condition"])
        return _write_fake_run(
            tmp_path,
            next(
                c
                for c in normalized["suite"]["conditions"]
                if c["name"] == kwargs["suite_condition"]
            ),
            kwargs["repetition"],
            run_output,
            kwargs["progress"],
        )[0]

    def compare_fn(run_dirs, compare_output):
        compare_calls.append(tuple(run_dirs))
        compare_output.mkdir(parents=True)
        (compare_output / "comparison.json").write_text(
            json.dumps(
                {
                    "benchmark_fingerprint": fingerprint(_samples(), "original"),
                    "comparison_schema_version": 1,
                }
            ),
            encoding="utf-8",
        )
        return {"comparison_schema_version": 1}

    kwargs = {
        "preview_fn": lambda *_args, **_kwargs: _preview(),
        "run_fn": run_fn,
        "progress": lambda _message: None,
        "compare_fn": compare_fn,
    }
    suite.run_suite(path, output, **kwargs)
    assert len(run_calls) == 2 and len(compare_calls) == 1
    suite.run_suite(path, output, resume=True, **kwargs)
    assert len(run_calls) == 2 and len(compare_calls) == 1


def test_resume_rejects_missing_saved_run_directory(tmp_path, monkeypatch):
    path, normalized = _config(tmp_path)
    _install_config(monkeypatch, normalized)
    output = tmp_path / "suite-out"
    condition = normalized["suite"]["conditions"][0]

    def run_fn(_data, _models, run_output, **kwargs):
        return _write_fake_run(
            tmp_path, condition, kwargs["repetition"], run_output, kwargs["progress"]
        )[0]

    suite.run_suite(
        path,
        output,
        preview_fn=lambda *_args, **_kwargs: _preview(),
        run_fn=run_fn,
        progress=lambda _message: None,
        compare_fn=lambda *_args, **_kwargs: {},
    )
    manifest_path = output / "suite.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = suite._run_key("baseline", 1)
    manifest["runs"][key]["run_dir"] = str(output / "runs" / "missing")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing"):
        suite.run_suite(
            path,
            output,
            resume=True,
            preview_fn=lambda *_args, **_kwargs: pytest.fail(
                "missing path should fail before preview"
            ),
        )


def test_resume_rejects_changed_execution_controls_before_preview(tmp_path, monkeypatch):
    path, normalized = _config(tmp_path)
    _install_config(monkeypatch, normalized)
    output = tmp_path / "suite-out"
    condition = normalized["suite"]["conditions"][0]

    def run_fn(_data, _models, run_output, **kwargs):
        return _write_fake_run(
            tmp_path, condition, kwargs["repetition"], run_output, kwargs["progress"]
        )[0]

    suite.run_suite(
        path,
        output,
        preview_fn=lambda *_args, **_kwargs: _preview(),
        run_fn=run_fn,
        progress=lambda _message: None,
        compare_fn=lambda *_args, **_kwargs: {},
    )
    with pytest.raises(ValueError, match="execution controls changed"):
        suite.run_suite(
            path,
            output,
            resume=True,
            base_url="http://localhost:11435",
            preview_fn=lambda *_args, **_kwargs: pytest.fail("preview ran before controls check"),
        )


def test_paused_run_resumes_without_counting_repetition_twice(tmp_path, monkeypatch):
    path, normalized = _config(tmp_path)
    _install_config(monkeypatch, normalized)
    output = tmp_path / "suite-out"
    condition = normalized["suite"]["conditions"][0]
    run_calls = []
    resume_calls = []

    def run_fn(_data, _models, run_output, **kwargs):
        run_calls.append(kwargs["repetition"])
        return _write_fake_run(
            tmp_path,
            condition,
            kwargs["repetition"],
            run_output,
            kwargs["progress"],
            status="paused",
        )[0]

    paused = suite.run_suite(
        path,
        output,
        preview_fn=lambda *_args, **_kwargs: _preview(),
        run_fn=run_fn,
        progress=lambda _message: None,
        compare_fn=lambda *_args, **_kwargs: {},
    )
    model_summary = paused["conditions"]["baseline"]["variability"]["prose"]["ollama:fake"]
    assert model_summary["scored_repetition_count"] == 0
    assert model_summary["mean_cer"] is None
    assert model_summary["per_repetition"][0]["run_status"] == "paused"

    def resume_fn(run_dir, *, progress):
        resume_calls.append(run_dir)
        run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        run_manifest["status"] = "complete"
        (run_dir / "manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
        progress("resumed")
        return run_dir

    completed = suite.run_suite(
        path,
        output,
        resume=True,
        preview_fn=lambda *_args, **_kwargs: _preview(),
        run_fn=lambda *_args, **_kwargs: pytest.fail("paused run must resume in place"),
        resume_fn=resume_fn,
        progress=lambda _message: None,
        compare_fn=lambda *_args, **_kwargs: {},
    )
    model_summary = completed["conditions"]["baseline"]["variability"]["prose"]["ollama:fake"]
    assert run_calls == [1]
    assert len(resume_calls) == 1
    assert model_summary["scored_repetition_count"] == 1
    assert model_summary["independent_sample_count"] == 2


def test_run_manifest_integrity_is_checked_before_reuse(tmp_path, monkeypatch):
    path, normalized = _config(tmp_path)
    _install_config(monkeypatch, normalized)
    output = tmp_path / "suite-out"
    condition = normalized["suite"]["conditions"][0]

    def run_fn(_data, _models, run_output, **kwargs):
        return _write_fake_run(
            tmp_path, condition, kwargs["repetition"], run_output, kwargs["progress"]
        )[0]

    suite.run_suite(
        path,
        output,
        preview_fn=lambda *_args, **_kwargs: _preview(),
        run_fn=run_fn,
        progress=lambda _message: None,
        compare_fn=lambda *_args, **_kwargs: {},
    )
    manifest = json.loads((output / "suite.json").read_text(encoding="utf-8"))
    run_dir = Path(manifest["runs"][suite._run_key("baseline", 1)]["run_dir"])
    run_manifest_path = run_dir / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["models"] = ["ollama:tampered"]
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check failed"):
        suite._load_run_state(run_dir)


def test_run_fingerprint_must_match_condition_preview(tmp_path, monkeypatch):
    path, normalized = _config(tmp_path)
    _install_config(monkeypatch, normalized)
    output = tmp_path / "suite-out"
    condition = normalized["suite"]["conditions"][0]

    def run_fn(_data, _models, run_output, **kwargs):
        return _write_fake_run(
            tmp_path, condition, kwargs["repetition"], run_output, kwargs["progress"]
        )[0]

    preview = dict(_preview(), benchmark_fingerprint="different-preview-fingerprint")
    with pytest.raises(ValueError, match="differs from preview"):
        suite.run_suite(
            path,
            output,
            preview_fn=lambda *_args, **_kwargs: preview,
            run_fn=run_fn,
            progress=lambda _message: None,
            compare_fn=lambda *_args, **_kwargs: {},
        )
    manifest = json.loads((output / "suite.json").read_text(encoding="utf-8"))
    entry = manifest["runs"][suite._run_key("baseline", 1)]
    assert entry["status"] == "fingerprint_mismatch"
    assert entry["run_dir"]
