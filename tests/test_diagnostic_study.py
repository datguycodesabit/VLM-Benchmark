import json

import httpx
import pytest
from PIL import Image

from vlm_bench import engine
from vlm_bench.backends import OllamaBackend
from vlm_bench.config import load_config, validate_settings
from vlm_bench.diagnostic_study import MODELS, prepare_study, report_study
from vlm_bench.runner import read_records
from vlm_bench.snapshot import freeze, load
from vlm_bench.suite import run_suite


class StudyBackend:
    resource_calls = 0

    def __init__(self, provider, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def validate_model(self, name):
        return {"model": name, "revision": "fixture"}

    def transcribe(self, model, path, prompt, options):
        assert "hello" not in prompt
        assert options["num_predict"] == 32
        with Image.open(path) as image:
            text = "hello" if model.endswith("7b") or image.width > 250 else "helo"
        return {"message": {"content": text}, "done_reason": "stop"}

    def observe_resources(self, model):
        type(self).resource_calls += 1
        return {"source": "fixture", "is_peak": False, "allocated_bytes": 1000}

    def unload(self, model):
        pass


@pytest.fixture
def study(tmp_path):
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "text").mkdir()
    sizes = [
        (100, 100),
        (150, 100),
        (10, 10),
        (12, 12),
        (20, 20),
        (40, 40),
        (42, 42),
        (45, 45),
        (70, 70),
        (72, 72),
        (75, 75),
    ]
    for i, size in enumerate(sizes):
        Image.new("RGB", size, (100, 110, 120)).save(source / "images" / f"a01-000-00-{i:02}.png")
        (source / "text" / f"a01-000-00-{i:02}.txt").write_text("hello")
    baseline = tmp_path / "filtered"
    freeze(source, baseline, layout="paired", min_pixels=10000)
    calibration = engine.run(
        source,
        ["ollama:calibration"],
        tmp_path / "calibration",
        warmup=False,
        num_predict=32,
        backend_factory=StudyBackend,
        progress=lambda _: None,
    )
    output = tmp_path / "study"
    plan = prepare_study(baseline, calibration, output, challenge_size=6)
    return output, plan


def test_study_freezes_identical_ids_and_references_across_preparation(study):
    root, plan = study
    assert plan["planned_predictions"] == 32
    assert not set(plan["cohorts"]["filtered"]) & set(plan["cohorts"]["small-challenge"])
    config = load_config(root / "suite.toml")
    assert len(config["suite"]["conditions"]) == 4
    for cohort in plan["cohorts"]:
        original = load(root / f"{cohort}-original")
        padded = load(root / f"{cohort}-padded")
        assert original["sample_ids"] == padded["sample_ids"]
        for left, right in zip(original["samples"], padded["samples"]):
            assert left["reference"] == right["reference"]
            with Image.open(left["crop_path"]) as a, Image.open(right["crop_path"]) as b:
                assert b.size == (a.width * 3 + 32, a.height * 3 + 32)
                assert b.getpixel((16 + a.width, 16 + a.height)) == (100, 110, 120)
                assert b.getpixel((0, 0)) == (255, 255, 255)


def test_mocked_study_reports_pairs_missing_data_and_review_safely(study, tmp_path):
    root, plan = study
    StudyBackend.resource_calls = 0
    suite_output = tmp_path / "runs"
    result = run_suite(
        root / "suite.toml",
        suite_output,
        preview_fn=lambda *a, **kw: engine.preview(*a, backend_factory=StudyBackend, **kw),
        run_fn=lambda *a, **kw: engine.run(*a, backend_factory=StudyBackend, **kw),
        progress=lambda _: None,
    )
    assert all(row["status"] == "complete" for row in result["conditions"].values())
    assert StudyBackend.resource_calls == 8  # One per warmup/model, not per ranked request.
    assert sum(len(read_records(p)) for p in suite_output.glob("runs/*/results.jsonl")) == 32
    report_dir = tmp_path / "report"
    report = report_study(root, suite_output, report_dir)
    assert len(report["summaries"]) == 8
    assert all(row["missing_pair"] == 0 for row in report["pairs"])
    assert len(report["resources"]) == 8
    observations = json.loads((report_dir / "observations.json").read_text())
    assert all(row["ttfo_seconds"] is None for row in observations)
    assert all(row["input_tokens"] is None for row in observations)
    assert any(row["right_only_correct"] for row in report["pairs"])
    template = json.loads((report_dir / "review-template.json").read_text())
    assert len(template["reviews"]) == 32
    assert "data-review" in (report_dir / "review.html").read_text()
    # An absent condition remains missing, with denominator and paired coverage visible.
    first_run = next(suite_output.glob("runs/*/manifest.json")).parent
    (first_run / "manifest.json").rename(first_run / "manifest.hidden")
    partial = report_study(root, suite_output, tmp_path / "partial")
    assert any(row["missing_pair"] for row in partial["pairs"])


def test_changed_study_config_is_rejected(study, tmp_path):
    root, _ = study
    with (root / "suite.toml").open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="configuration changed"):
        report_study(root, tmp_path / "runs", tmp_path / "report")


def test_ollama_resource_snapshot_is_best_effort_and_not_generation():
    def respond(request):
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [{"name": "test", "size": 100, "size_vram": 80, "context_length": 4096}]
            },
        )

    backend = OllamaBackend(
        "http://localhost:11434", 1, {"transport": httpx.MockTransport(respond)}
    )
    snapshot = backend.observe_resources("test")
    assert snapshot["allocated_bytes"] == 100
    assert snapshot["size_vram_bytes"] == 80
    assert snapshot["is_peak"] is False
    assert backend.observe_resources("missing")["allocated_bytes"] is None
    backend.close()


def test_resource_setting_only_accepts_ollama_boolean():
    assert validate_settings([MODELS[0]], {MODELS[0]: {"collect_resources": True}})
    with pytest.raises(ValueError, match="boolean"):
        validate_settings([MODELS[0]], {MODELS[0]: {"collect_resources": "yes"}})
    with pytest.raises(ValueError, match="only supported"):
        validate_settings(["openai:test"], {"openai:test": {"collect_resources": True}})


def test_resource_control_is_not_sent_as_a_generation_option(tmp_path):
    def respond(request):
        assert request.url.path == "/api/chat"
        assert "collect_resources" not in json.loads(request.content)["options"]
        return httpx.Response(200, json={"message": {"content": "hello"}, "done": True})

    image = tmp_path / "sample.png"
    Image.new("RGB", (10, 10), "white").save(image)
    with OllamaBackend(
        "http://localhost:11434", 1, {"transport": httpx.MockTransport(respond)}
    ) as backend:
        response = backend.transcribe("test", image, "Transcribe", {"collect_resources": True})
        assert response["message"]["content"] == "hello"


def test_review_escapes_model_output(study, tmp_path):
    from vlm_bench.diagnostic_study import _write_review

    root, plan = study
    cohort = "filtered"
    sample_id = plan["cohorts"][cohort][0]
    evidence = [
        {
            "cohort": cohort,
            "sample_id": sample_id,
            "profile": "original",
            "model": MODELS[0],
            "reference": "hello",
            "prediction": "<script>alert(1)</script>",
            "status": "success",
            "exact_match": False,
            "native_sha256": "hash",
            "flags": ["mismatch"],
        }
    ]
    out = tmp_path / "review"
    out.mkdir()
    _write_review(out, root, plan, evidence, {"evidence_id": "x", "summaries": []}, 25)
    contents = (out / "review.html").read_text()
    assert "<script>alert(1)</script>" not in contents
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in contents
