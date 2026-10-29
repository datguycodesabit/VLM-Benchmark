"""Exercise saved-artifact orchestration at scale using a synthetic backend.

This is a workflow capacity check, not a model-quality or hardware benchmark.
Run: python scripts/validate_workflow.py --output runs/validation --images 1000
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from PIL import Image, ImageDraw

from vlm_bench import engine
from vlm_bench.reporting import write_report
from vlm_bench.runner import _atomic_json, read_records
from vlm_bench.scoring_policy import ScoringPolicy
from vlm_bench.snapshot import freeze


class SyntheticBackend:
    calls = 0

    def __init__(self, provider, *args):
        self.provider = provider

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def validate_model(self, model):
        return {"model": model, "revision": "synthetic-fixture-v1"}

    def transcribe(self, model, path, prompt, options):
        type(self).calls += 1
        index = int(path.stem.removeprefix("s"))
        text = f"x+{index}" if index % 2 else f"Synthetic sample {index}"
        # Deliberate errors exercise difficult examples and weighted scoring.
        if model == "fixture-b" and index % 17 == 0:
            text += "!"
        return {"message": {"content": text}, "done_reason": "stop"}

    def unload(self, model):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=1000)
    args = parser.parse_args()
    if args.images < 2:
        parser.error("--images must be at least 2")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = root / "synthetic-data"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    metadata = []
    for index in range(args.images):
        sample_id = f"s{index:04d}"
        text = f"x+{index}" if index % 2 else f"Synthetic sample {index}"
        image = Image.new("RGB", (360, 45), "white")
        ImageDraw.Draw(image).text((8, 12), text, fill="black")
        image.save(data / "images" / f"{sample_id}.png")
        (data / "text" / f"{sample_id}.txt").write_text(text)
        metadata.append(
            {
                "id": sample_id,
                "source_document": f"document-{index // 10:03d}",
                "content_type": "equation" if index % 2 else "prose",
                "sample_type": "equation" if index % 2 else "line",
                "verified": True,
                "split": "test",
                "variable_domains": {"x": [-1, 0, 1]},
            }
        )
    (data / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata))
    freeze(data, root / "prepared", layout="paired")
    prepared_seconds = time.perf_counter() - started
    state = {"schema_version": 1, "planned_passes": 3, "runs": []}
    _atomic_json(root / "experiment.json", state)
    policy = ScoringPolicy().to_dict()
    for repetition in range(1, 4):

        def progress(message):
            if message.startswith("Run directory: "):
                directory = Path(message.removeprefix("Run directory: "))
                state["runs"].append(str(directory.relative_to(root)))
                _atomic_json(root / "experiment.json", state)

        engine.run(
            None,
            ["ollama:fixture-a", "ollama:fixture-b"],
            root / "runs",
            prepared=root / "prepared",
            warmup=False,
            backend_factory=SyntheticBackend,
            scoring_policy=policy,
            measurement_profile={"profile": "latency"},
            planned_passes=3,
            suite_condition="baseline",
            repetition=repetition,
            repeated_measurement=True,
            progress=progress,
        )
        print(f"Completed synthetic pass {repetition}/3", flush=True)
    inference_finished = time.perf_counter()
    paths = write_report(root)
    elapsed = time.perf_counter() - started
    records = sum(len(read_records(root / path / "results.jsonl")) for path in state["runs"])
    evaluation = json.loads((root / "report" / "evaluation.json").read_text())
    result = {
        "kind": "synthetic-workflow-only",
        "images": args.images,
        "passes": 3,
        "models": 2,
        "result_records": records,
        "backend_calls": SyntheticBackend.calls,
        "prepare_seconds": prepared_seconds,
        "execution_seconds": inference_finished - started - prepared_seconds,
        "report_seconds": time.perf_counter() - inference_finished,
        "total_seconds": elapsed,
        "host": platform.platform(),
        "python": platform.python_version(),
        "evaluation_id": evaluation["evaluation_id"],
        "reports": [str(path.relative_to(root)) for path in paths],
        "limitations": "No live provider, real model, handwriting quality, or GPU measured.",
    }
    assert records == args.images * 6 == SyntheticBackend.calls
    assert evaluation["conditions"]["baseline"]["finalizable"]
    assert all(
        len(track["models"]) == 2
        for track in evaluation["conditions"]["baseline"]["tracks"].values()
    )
    _atomic_json(root / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
