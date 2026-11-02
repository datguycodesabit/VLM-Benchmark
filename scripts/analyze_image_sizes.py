"""Analyze saved predictions by native crop area without making model calls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from vlm_bench.export import _csv_value


def analyze(run: Path, thresholds: list[int]) -> tuple[list[dict], list[dict]]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    predictions = {}
    for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["model"], row["sample_id"])
        if key in predictions:
            raise ValueError(f"Duplicate prediction: {key}")
        predictions[key] = row
    if manifest.get("preprocess") != "original":
        raise ValueError("Use an original-pixel run; enlarged crops cannot establish native area")
    evidence = []
    for sample in manifest["samples"]:
        # Run-local frozen crops remain usable after the source data moves.
        crop = run / "crops" / Path(sample["crop_path"]).name
        if not crop.is_file():
            crop = Path(sample["crop_path"])
        with Image.open(crop) as image:
            width, height = image.size
        for model in manifest["models"]:
            row = predictions.get((model, sample["id"]))
            if row is None:
                raise ValueError("Size calibration requires complete saved predictions")
            metrics = row["metrics"]
            evidence.append(
                {
                    "model": model,
                    "sample_id": sample["id"],
                    "width": width,
                    "height": height,
                    "pixels": width * height,
                    "reference": row["reference"],
                    "prediction": row["prediction"],
                    "exact_match": metrics["exact_match"],
                    "char_edits": metrics["char_edits"],
                    "reference_chars": metrics["reference_chars"],
                    "truncated": row["truncated"],
                }
            )
    summaries = []
    for model in manifest["models"]:
        model_rows = [row for row in evidence if row["model"] == model]
        for threshold in sorted(set([0, *thresholds])):
            retained = [row for row in model_rows if row["pixels"] >= threshold]
            count = len(retained)
            reference_chars = sum(row["reference_chars"] for row in retained)
            summaries.append(
                {
                    "model": model,
                    "min_pixels": threshold,
                    "retained": count,
                    "excluded": len(model_rows) - count,
                    "exact_matches": sum(row["exact_match"] for row in retained),
                    "exact_match_rate": sum(row["exact_match"] for row in retained) / count
                    if count
                    else None,
                    "corpus_cer": sum(row["char_edits"] for row in retained) / reference_chars
                    if reference_chars
                    else None,
                }
            )
    return summaries, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000],
    )
    args = parser.parse_args()
    if any(value < 0 for value in args.thresholds):
        parser.error("Thresholds must be nonnegative")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    summaries, evidence = analyze(args.run.resolve(), args.thresholds)
    args.output.mkdir(parents=True)
    for name, rows in (("thresholds.csv", summaries), ("samples.csv", evidence)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
    (args.output / "analysis.json").write_text(
        json.dumps(
            {
                "source_run": str(args.run.resolve()),
                "limitations": "Exploratory thresholds selected using observed outcomes; sample length, "
                "punctuation and handwriting confound image area. Validate on fresh samples. "
                "Filtered CER is not an improvement on the original benchmark.",
                "thresholds": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in summaries:
        rate = row["exact_match_rate"]
        print(
            f"{row['min_pixels']:>6} pixels: {row['retained']:>3} retained, "
            f"{row['excluded']:>3} excluded; exact matches "
            + (f"{rate:.1%}" if rate is not None else "N/A")
        )


if __name__ == "__main__":
    main()
