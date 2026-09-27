import json

import pytest
from PIL import Image

from vlm_bench.runner import read_records, resume_benchmark, run_benchmark


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    for folder in ("images", "xml", "references"):
        (root / folder).mkdir(parents=True)
    for i in range(3):
        name = f"a01-{i:03}"
        Image.new("RGB", (200, 300), "white").save(root / "images" / f"{name}.png")
        (root / "xml" / f"{name}.xml").write_text(
            '<form height="300" writer-id="1"><handwritten-part><line text="Hello, world."><word><cmp x="30" y="100" width="100" height="30"/></word></line></handwritten-part></form>'
        )
    return root


class FakeClient:
    calls = []
    fail_after = None
    digest = "sha256:fixed"

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def version(self):
        return "test-1"

    def validate_model(self, model):
        return {"digest": self.digest, "capabilities": ["vision"]}

    def unload(self, model):
        pass

    def transcribe(self, model, image_path, prompt, options):
        assert "Hello" not in prompt
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise KeyboardInterrupt()
        self.calls.append((model, image_path.stem))
        return {
            "message": {"content": "Hello, world."},
            "done": True,
            "done_reason": "stop",
            "load_duration": 100,
            "eval_count": 5,
        }


@pytest.fixture(autouse=True)
def reset():
    FakeClient.calls = []
    FakeClient.fail_after = None
    FakeClient.digest = "sha256:fixed"


def run(dataset, tmp_path, **kwargs):
    return run_benchmark(
        dataset,
        ["first", "second"],
        tmp_path / "runs",
        progress=lambda _: None,
        client_factory=FakeClient,
        **kwargs,
    )


def test_full_run_all_records_warmups_separate(dataset, tmp_path):
    directory = run(dataset, tmp_path)
    rows = read_records(directory / "results.jsonl")
    assert len(rows) == 6
    assert len(read_records(directory / "warmups.jsonl")) == 2
    assert len(FakeClient.calls) == 8
    assert all(r["metrics"]["cer"] == 0 for r in rows)
    assert json.loads((directory / "manifest.json").read_text())["status"] == "complete"


def test_interrupt_resume_no_duplicate_measured_calls(dataset, tmp_path):
    FakeClient.fail_after = 2
    with pytest.raises(KeyboardInterrupt):
        run(dataset, tmp_path, warmup=False)
    directory = next((tmp_path / "runs").iterdir())
    assert len(read_records(directory / "results.jsonl")) == 2
    FakeClient.fail_after = None
    resume_benchmark(directory, progress=lambda _: None, client_factory=FakeClient)
    assert len(FakeClient.calls) == 6
    assert len(read_records(directory / "results.jsonl")) == 6


def test_resume_rejects_changed_data_model_and_settings(dataset, tmp_path):
    directory = run(dataset, tmp_path)
    FakeClient.digest = "changed"
    with pytest.raises(ValueError, match="digest changed"):
        resume_benchmark(directory, client_factory=FakeClient)
    FakeClient.digest = "sha256:fixed"
    (dataset / "references" / "a01-000.txt").write_text("Different")
    with pytest.raises(ValueError, match="Dataset or references changed"):
        resume_benchmark(directory, client_factory=FakeClient)
    (dataset / "references" / "a01-000.txt").unlink()
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["options"]["temperature"] = 1
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed"):
        resume_benchmark(directory, client_factory=FakeClient)


def test_failure_persisted_and_continue(dataset, tmp_path):
    class Failing(FakeClient):
        def transcribe(self, model, image_path, prompt, options):
            if image_path.stem == "a01-001":
                raise RuntimeError("timeout")
            return super().transcribe(model, image_path, prompt, options)

    directory = run_benchmark(
        dataset,
        ["first"],
        tmp_path / "runs",
        warmup=False,
        client_factory=Failing,
        progress=lambda _: None,
    )
    rows = read_records(directory / "results.jsonl")
    assert [r["status"] for r in rows] == ["success", "error", "success"]
    assert rows[1]["metrics"] is None
    assert json.loads((directory / "manifest.json").read_text())["status"] == "complete_with_errors"


def test_recover_partial_last_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"sample_id":"one"}\n{"sam')
    assert read_records(path, repair=True) == [{"sample_id": "one"}]
    assert path.read_bytes().endswith(b"\n")
    assert path.with_suffix(".interrupted-tail").read_bytes() == b'{"sam'


def test_failed_setup_removes_unusable_directory(dataset, tmp_path):
    class MissingModel(FakeClient):
        def validate_model(self, model):
            raise RuntimeError("not installed")

    with pytest.raises(RuntimeError, match="not installed"):
        run_benchmark(dataset, ["missing"], tmp_path / "runs", client_factory=MissingModel)
    assert list((tmp_path / "runs").iterdir()) == []


@pytest.mark.parametrize("value", [None, "not a number", -1, True])
def test_malformed_timing_metadata_keeps_successful_prediction(dataset, tmp_path, value):
    class BadTiming(FakeClient):
        def transcribe(self, *args):
            raw = super().transcribe(*args)
            raw.update(eval_count=value, load_duration=value)
            return raw

    directory = run_benchmark(
        dataset,
        ["first"],
        tmp_path / "runs",
        warmup=False,
        client_factory=BadTiming,
        progress=lambda _: None,
    )
    rows = read_records(directory / "results.jsonl")
    assert all(row["status"] == "success" for row in rows)
    assert all(row["load_duration_seconds"] is None for row in rows)


def test_truncated_response_scored_and_flagged(dataset, tmp_path):
    class Truncated(FakeClient):
        def transcribe(self, *args):
            return {"message": {"content": "Hello"}, "done": True, "done_reason": "length"}

    directory = run_benchmark(
        dataset,
        ["first"],
        tmp_path / "runs",
        warmup=False,
        client_factory=Truncated,
        progress=lambda _: None,
    )
    rows = read_records(directory / "results.jsonl")
    assert all(row["truncated"] and row["metrics"]["cer"] > 0 for row in rows)
