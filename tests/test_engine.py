import json
from pathlib import Path

import pytest
from PIL import Image

from vlm_bench import engine
from vlm_bench.runner import read_records


@pytest.fixture
def paired(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    for i in range(3):
        Image.new("RGB", (80 + i, 20), "white").save(data / "images" / f"s{i}.png")
        (data / "text" / f"s{i}.txt").write_text("hello")
    return data


class Backend:
    calls = []
    failure = None
    identity = "fixed"

    def __init__(self, provider, *args):
        self.provider = provider

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def validate_model(self, model):
        return {"model": model, "revision": self.identity}

    def transcribe(self, model, path, prompt, options):
        assert "hello" not in prompt
        if self.failure == "interrupt" and len(self.calls) == 1:
            raise KeyboardInterrupt
        if self.failure == "error" and path.stem == "s1":
            raise RuntimeError("unavailable")
        if self.failure == "quota" and self.provider == "chatgpt":
            from vlm_bench.backends import BackendPaused

            raise BackendPaused(self.provider, model, "limit")
        self.calls.append((self.provider, path.stem))
        return {"message": {"content": "hello"}, "done_reason": "stop"}

    def unload(self, model):
        pass


@pytest.fixture(autouse=True)
def reset():
    Backend.calls = []
    Backend.failure = None
    Backend.identity = "fixed"


def test_cross_provider_same_frozen_inputs(paired, tmp_path):
    directory = engine.run(
        paired,
        ["ollama:first", "trocr:second"],
        tmp_path / "runs",
        warmup=False,
        backend_factory=Backend,
    )
    rows = read_records(directory / "results.jsonl")
    assert len(rows) == 6
    assert {r["sample_id"] for r in rows if r["provider"] == "ollama"} == {
        r["sample_id"] for r in rows if r["provider"] == "trocr"
    }
    assert all(r["metrics"]["cer"] == 0 for r in rows)


def test_resume_frozen_inputs_without_live_dataset(paired, tmp_path):
    Backend.failure = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        engine.run(
            paired, ["ollama:first"], tmp_path / "runs", warmup=False, backend_factory=Backend
        )
    directory = next((tmp_path / "runs").iterdir())
    (paired / "text" / "s0.txt").write_text("changed after freezing")
    Backend.failure = None
    engine.resume(directory, backend_factory=Backend)
    assert len(read_records(directory / "results.jsonl")) == 3
    assert len(Backend.calls) == 3


def test_retry_failed_preserves_history_and_successes(paired, tmp_path):
    Backend.failure = "error"
    directory = engine.run(
        paired, ["ollama:first"], tmp_path / "runs", warmup=False, backend_factory=Backend
    )
    Backend.failure = None
    engine.resume(directory, retry_failed=True, backend_factory=Backend)
    assert len(Backend.calls) == 3
    assert len(read_records(directory / "results.jsonl")) == 3
    assert len(read_records(directory / "attempts.jsonl")) == 1


def test_quota_pauses_only_cloud_and_never_switches_billing(paired, tmp_path):
    Backend.failure = "quota"
    directory = engine.run(
        paired,
        ["chatgpt:first", "ollama:first"],
        tmp_path / "runs",
        warmup=False,
        backend_factory=Backend,
    )
    assert json.loads((directory / "manifest.json").read_text())["status"] == "paused"
    assert len(read_records(directory / "results.jsonl")) == 3
    assert all(p == "ollama" for p, _ in Backend.calls)


def test_resume_rejects_changed_model_or_image(paired, tmp_path):
    directory = engine.run(
        paired, ["ollama:first"], tmp_path / "runs", warmup=False, backend_factory=Backend
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    Path(manifest["samples"][0]["crop_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Saved image changed"):
        engine.resume(directory, backend_factory=Backend)


def test_dry_run_never_transcribes(paired):
    report = engine.preview(paired, ["ollama:first", "openai:second"], backend_factory=Backend)
    assert report["sample_count"] == 3
    assert report["models"][1]["execution"] == "cloud"
    assert Backend.calls == []


def test_local_request_timing_separates_known_model_load(paired, tmp_path):
    class TimedBackend(Backend):
        def transcribe(self, *args):
            raw = super().transcribe(*args)
            raw["load_duration"] = 0
            return raw

    directory = engine.run(
        paired, ["ollama:first"], tmp_path / "runs", warmup=False, backend_factory=TimedBackend
    )
    for row in read_records(directory / "results.jsonl"):
        assert row["inference_latency_seconds"] == row["latency_seconds"]
        assert row["load_duration_seconds"] == 0
