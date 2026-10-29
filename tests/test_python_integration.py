import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vlm_bench import engine
from vlm_bench.runner import read_records


@pytest.fixture
def paired(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    for index in range(2):
        Image.new("RGB", (60 + index, 20), "white").save(data / "images" / f"s{index}.png")
        (data / "text" / f"s{index}.txt").write_text("reference text", encoding="utf-8")
    return data


@pytest.fixture
def fake_torch(monkeypatch):
    torch = SimpleNamespace(
        __version__="mock-torch-1",
        version=SimpleNamespace(cuda=None, hip=None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _adapter_source(*, track="prose", interrupt=False, suffix=""):
    return f"""import os
import weakref

instances = []

class Adapter:
    capabilities = {{
        "supported_tracks": [{track!r}],
        "supported_sample_types": ["line"],
        "batching": True,
        "streaming": False,
        "concurrency_safe": False,
        "timing_hooks": ["preprocessing", "inference"],
        "usage_fields": [],
    }}

    def __init__(self):
        self.events = []
        self.loaded_model = None
        self.image_ref = None
        self.observed = []
        instances.append(self)

    def load(self, model, *, device, settings):
        self.events.append("load")
        self.loaded_model = (model, device)

    def preprocess(self, image, *, track, prompt, controls):
        self.events.append("preprocess")
        self.image_ref = weakref.ref(image)
        self.observed.append((track, prompt, dict(controls)))
        return image.size

    def infer(self, prepared, *, controls):
        self.events.append("infer")
        if {interrupt!r} and os.environ.pop("VLM_BENCH_TEST_INTERRUPT", "") == "1":
            raise KeyboardInterrupt("simulated interruption")
        return prepared

    def decode(self, output):
        self.events.append("decode")
        return "recognized handwriting"

    def identity(self, model, *, settings, runtime):
        return {{"adapter_version": "fixture-v1", "checkpoint_sha256": "a" * 64}}

    def timing_observations(self):
        return []

    def unload(self, model):
        self.events.append("unload")
        self.loaded_model = None

    def close(self):
        self.events.append("close")

{suffix}
"""


def _write_adapter(path: Path, *, track="prose", interrupt=False, suffix="") -> None:
    path.write_text(
        _adapter_source(track=track, interrupt=interrupt, suffix=suffix),
        encoding="utf-8",
    )


def _module_name(tmp_path: Path, stem: str) -> str:
    suffix = "".join(char if char.isalnum() else "_" for char in tmp_path.name)
    return f"{stem}_{suffix}"


def test_python_backend_runs_through_engine_and_persists_timing_without_reference_leak(
    paired, tmp_path, monkeypatch, fake_torch
):
    module_name = _module_name(tmp_path, "toy_adapter")
    adapter_path = tmp_path / f"{module_name}.py"
    _write_adapter(adapter_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    run_dir = engine.run(
        paired,
        [f"python:{module_name}:Adapter"],
        tmp_path / "runs",
        warmup=False,
    )

    module = sys.modules[module_name]
    instances = module.instances
    rows = read_records(run_dir / "results.jsonl")
    assert len(rows) == 2
    assert all(row["status"] == "success" for row in rows)
    assert all(row["prediction"] == "recognized handwriting" for row in rows)
    assert all(row["model"] == f"python:{module_name}:Adapter" for row in rows)
    assert all(row["actual_device"] == "cpu" for row in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    model_identity = manifest["model_info"][f"python:{module_name}:Adapter"]
    assert model_identity["adapter_sha256"]
    assert model_identity["checkpoint_sha256"] == "a" * 64
    for row in rows:
        observations = row["timing_observations"]
        assert {item["metric_name"] for item in observations} >= {
            "model_load",
            "image_preprocessing",
            "inference",
            "output_conversion",
        }
        assert all(item["unit"] and item["source"] for item in observations)
        assert row["provider_details"]["timing_observations"] == observations

    active_instances = [instance for instance in instances if "preprocess" in instance.events]
    assert len(active_instances) == 1
    active = active_instances[0]
    assert len(active.observed) == 2
    assert all("reference" not in controls for _, _, controls in active.observed)
    assert all("reference text" not in prompt for _, prompt, _ in active.observed)
    assert active.events == [
        "load",
        "preprocess",
        "infer",
        "decode",
        "preprocess",
        "infer",
        "decode",
        "unload",
        "close",
    ]
    assert active.loaded_model is None
    assert active.image_ref() is None


def test_custom_python_capability_rejection_precedes_run_creation(
    paired, tmp_path, monkeypatch, fake_torch
):
    module_name = _module_name(tmp_path, "equation_only")
    adapter_path = tmp_path / f"{module_name}.py"
    _write_adapter(adapter_path, track="equation")
    monkeypatch.syspath_prepend(str(tmp_path))
    output_dir = tmp_path / "runs"

    with pytest.raises(ValueError, match="has no supported samples"):
        engine.run(
            paired,
            [f"python:{module_name}:Adapter"],
            output_dir,
            warmup=False,
        )

    assert not output_dir.exists()
    adapter = sys.modules[module_name].instances[0]
    assert adapter.events == ["close"]


def test_batch_measurement_is_rejected_even_when_adapter_declares_batching(
    paired, tmp_path, monkeypatch, fake_torch
):
    module_name = _module_name(tmp_path, "batched_adapter")
    adapter_path = tmp_path / f"{module_name}.py"
    _write_adapter(adapter_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    output_dir = tmp_path / "runs"

    with pytest.raises(ValueError, match="Batch sizes above 1 are not yet supported"):
        engine.run(
            paired,
            [f"python:{module_name}:Adapter"],
            output_dir,
            warmup=False,
            measurement_profile={"profile": "throughput", "batch_size": 2, "concurrency": 1},
        )

    assert not output_dir.exists()
    assert module_name not in sys.modules


def test_resume_rejects_changed_custom_adapter_source(paired, tmp_path, monkeypatch, fake_torch):
    module_name = _module_name(tmp_path, "resume_adapter")
    adapter_path = tmp_path / f"{module_name}.py"
    _write_adapter(adapter_path, interrupt=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("VLM_BENCH_TEST_INTERRUPT", "1")

    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        engine.run(
            paired,
            [f"python:{module_name}:Adapter"],
            tmp_path / "runs",
            warmup=False,
        )

    run_dir = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    selector = f"python:{module_name}:Adapter"
    assert manifest["model_info"][selector]["adapter_sha256"]
    assert read_records(run_dir / "results.jsonl") == []

    with adapter_path.open("a", encoding="utf-8") as stream:
        stream.write("# changed adapter source after the frozen run\n")
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    with pytest.raises(ValueError, match=f"Model identity changed: {selector}"):
        engine.resume(run_dir)
