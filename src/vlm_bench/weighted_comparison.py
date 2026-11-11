"""Compare saved weighted evaluations without collapsing passes or changing policies."""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from .export import _csv_value
from .report_model import REPORT_MODEL_VERSION, _load_validated_run, build_report, evaluate_run
from .reporting import _evidence, _leaderboards, render_report
from .research import _sample_kind
from .runner import _atomic_json, _digest
from .sources import source_plan


def needs_weighted_comparison(paths):
    """Keep legacy CER comparison explicit when any new-policy evidence is present."""
    for value in paths:
        path = Path(value).expanduser()
        if (path / "experiment.json").is_file() or (path / "suite.json").is_file():
            return True
        marker = path / "manifest.json"
        if marker.is_file():
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            if manifest.get("scoring_policy") is not None:
                return True
    return False


def compare_weighted(sources, output, *, policy=None):
    paths = [Path(value).expanduser().resolve() for value in sources]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("Comparison requires at least two distinct saved sources")
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Comparison output already exists: {target}")
    if any(target == path or target.is_relative_to(path) for path in paths):
        raise ValueError("Comparison output cannot be inside a source")
    plans = [source_plan(path) for path in paths]
    reports = [build_report(path, policy=policy) for path in paths]
    conditions = set(reports[0].get("conditions", {}))
    if not conditions or any(set(report.get("conditions", {})) != conditions for report in reports):
        raise ValueError(
            "Comparison sources must contain the same named conditions; compare matching conditions"
        )
    labels, used = [], set()
    for path in paths:
        label, suffix = path.name, 2
        while label in used:
            label = f"{path.name}-{suffix}"
            suffix += 1
        labels.append(label)
        used.add(label)
    evaluations, entries = {}, {}
    for condition in sorted(conditions):
        evaluations_for_condition = [report["conditions"][condition] for report in reports]
        if any(
            not item.get("policy") or not item.get("tracks") for item in evaluations_for_condition
        ):
            raise ValueError(
                "Legacy sources require --weighted or --scoring-config for a derived comparison"
            )
        policies = [item["policy"] for item in evaluations_for_condition]
        if any(item != policies[0] for item in policies[1:]):
            raise ValueError(
                "Scoring policies differ; use --scoring-config to explicitly re-evaluate all sources"
            )
        repeats = [item["planned_passes"] for item in evaluations_for_condition]
        if len(set(repeats)) != 1:
            raise ValueError("Comparison sources must have the same planned pass count")
        fingerprints = set()
        for plan in plans:
            for member in plan["runs"]:
                if member["condition"] == condition:
                    manifest = json.loads(
                        (member["path"] / "manifest.json").read_text(encoding="utf-8")
                    )
                    fingerprints.add(manifest.get("benchmark_fingerprint"))
        if len(fingerprints) != 1 or None in fingerprints:
            raise ValueError(
                f"Condition {condition!r} does not use the same frozen benchmark snapshot"
            )
        merged_records, merged_models, capabilities, model_info = [], [], {}, {}
        samples = None
        for label, path, plan, evaluation in zip(
            labels, paths, plans, evaluations_for_condition, strict=True
        ):
            timing_eligible = {
                (row["model"], track_name): row.get("median_latency_seconds") is not None
                for track_name, track in evaluation["tracks"].items()
                for row in track.get("models", [])
            }
            mapping = {}
            for member in plan["runs"]:
                if member["condition"] != condition:
                    continue
                manifest, records = _load_validated_run(member["path"])
                if samples is None:
                    samples = manifest["samples"]
                sample_by_id = {item["id"]: item for item in manifest["samples"]}
                for original in manifest["models"]:
                    selector = f"{original}@{label}"
                    mapping[original] = selector
                    if selector not in merged_models:
                        merged_models.append(selector)
                    entries[selector] = {"source": label, "directory": str(path), "model": original}
                    capabilities[selector] = manifest.get("model_capabilities", {}).get(
                        original, {}
                    )
                    model_info[selector] = manifest.get("model_info", {}).get(original, {})
                for saved in records:
                    row = copy.deepcopy(saved)
                    row.update(model=mapping[saved["model"]], **{"pass": member["repetition"]})
                    sample = sample_by_id[row["sample_id"]]
                    track = "equation" if _sample_kind(sample) == "equation" else "prose"
                    if not timing_eligible.get((saved["model"], track), False):
                        row["latency_seconds"] = None
                    merged_records.append(row)
        if samples is None:
            raise ValueError(f"No saved evidence for condition {condition!r}")
        merged = evaluate_run(
            samples,
            merged_records,
            models=merged_models,
            policy=policies[0],
            planned_passes=repeats[0],
            run_status="complete"
            if all(item.get("finalizable") for item in evaluations_for_condition)
            else "incomplete",
            manifest={
                "benchmark_fingerprint": next(iter(fingerprints)),
                "models": merged_models,
                "model_capabilities": capabilities,
                "model_info": model_info,
                "measurement_profile": {"profile": "latency", "concurrency": 1},
            },
        )
        for track in merged["tracks"].values():
            for row in track["models"]:
                entry = entries[row["model"]]
                row.update(
                    original_model=entry["model"], source=entry["source"], cer=row.get("corpus_cer")
                )
            track["models"].sort(
                key=lambda row: (
                    row.get("rank") is None,
                    row.get("rank") or 0,
                    -(row.get("score") or 0),
                    row["model"],
                )
            )
        merged["benchmark_fingerprint"] = next(iter(fingerprints))
        merged["sources"] = {
            label: evaluation
            for label, evaluation in zip(labels, evaluations_for_condition, strict=True)
        }
        evaluations[condition] = merged
    report = {
        "schema_version": REPORT_MODEL_VERSION,
        "report_model_version": REPORT_MODEL_VERSION,
        "comparison_schema_version": 2,
        "evaluation_schema_version": 2,
        "evaluation_id": _digest(
            {"sources": [item["evaluation_id"] for item in reports], "labels": labels}
        ),
        "raw_evidence_id": _digest(
            {
                "source_raw_evidence_ids": [item.get("raw_evidence_id") for item in reports],
                "labels": labels,
            }
        ),
        "provenance": {
            "kind": "weighted_comparison",
            "source_evaluation_ids": [item["evaluation_id"] for item in reports],
        },
        "conditions": evaluations,
        "entries": entries,
        "source": {"type": "comparison", "paths": [str(path) for path in paths]},
        "policy": reports[0].get("policy"),
    }
    if len(evaluations) == 1:
        report["tracks"] = next(iter(evaluations.values()))["tracks"]
    target.parent.mkdir(parents=True, exist_ok=True)
    reservation = target.with_name(f".{target.name}.comparison-lock")
    descriptor = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        if target.exists():
            raise FileExistsError(f"Comparison output already exists: {target}")
        for label, path in zip(labels, paths, strict=True):
            evidence = {}
            _evidence(evidence, path, stage)
            for key in (
                "difficult_examples",
                "measurement_context",
                "operational_summary",
                "timing_breakdowns",
                "usage_summary",
            ):
                report.setdefault(key, []).extend(
                    dict(item, source=label) for item in evidence.get(key, [])
                )
        render_report(report, stage)
        _atomic_json(stage / "comparison.json", report)
        rows = _leaderboards(report)
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["model", "score"]
        with (stage / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows({key: _csv_value(row.get(key)) for key in fields} for row in rows)
        paired = [
            dict(item, condition=condition, track=track)
            for condition, evaluation in evaluations.items()
            for track, data in evaluation["tracks"].items()
            for item in data["uncertainty"].get("paired_score_differences", [])
        ]
        for name, table, default_fields in (
            ("paired.csv", paired, ["condition", "track", "left_model", "right_model", "ci95"]),
            (
                "costs.csv",
                report.get("operational_summary", []),
                ["source", "pass", "estimated_cost_usd", "cost_status"],
            ),
        ):
            columns = list(dict.fromkeys(key for row in table for key in row)) or default_fields
            with (stage / name).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, columns)
                writer.writeheader()
                writer.writerows(
                    {key: _csv_value(row.get(key)) for key in columns} for row in table
                )
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        reservation.unlink(missing_ok=True)
    return report
