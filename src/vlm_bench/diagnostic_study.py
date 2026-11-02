"""Controlled, offline-auditable local OCR diagnostic studies."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import shutil
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .config import load_config
from .export import _csv_value
from .inspection import _flags, _load_run
from .metrics import normalize_text, score
from .runner import PROMPT, _digest, read_records
from .snapshot import freeze, load

MODELS = ["ollama:qwen2.5vl:3b", "ollama:qwen2.5vl:7b"]
COHORTS = ("filtered", "small-challenge")
PROFILES = ("original", "padded")


def _json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_plan(root):
    plan = json.loads((root / "study.json").read_text(encoding="utf-8"))
    if plan.get("integrity") != _digest({k: v for k, v in plan.items() if k != "integrity"}):
        raise ValueError("Diagnostic study manifest integrity check failed")
    return plan


def prepare_study(filtered_snapshot, challenge_run, output, *, challenge_size=30, seed=44):
    """Freeze paired original/padded snapshots without new inference or data changes."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Study output already exists: {output}")
    if (
        isinstance(challenge_size, bool)
        or not isinstance(challenge_size, int)
        or challenge_size < 1
    ):
        raise ValueError("challenge_size must be a positive integer")
    filtered = load(filtered_snapshot)
    if filtered["preprocess"] != "original":
        raise ValueError("Filtered snapshot must contain original pixels")
    _, old_manifest, old_samples, _, _, _, _ = _load_run(challenge_run)
    if old_manifest.get("preprocess") != "original":
        raise ValueError("Challenge source must contain original pixels")
    cutoff = filtered.get("provenance", {}).get("selection", {}).get("min_pixels")
    if not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError("Filtered snapshot must declare its native min_pixels cutoff")
    native = {}
    pools = defaultdict(list)
    for sample in [*filtered["samples"], *old_samples.values()]:
        with Image.open(sample["crop_path"]) as image:
            width, height = image.size
        native[sample["id"]] = {
            "width": width,
            "height": height,
            "pixels": width * height,
            "aspect_ratio": width / height,
            "sha256": sample["hashes"]["crop"],
        }
    for sample in old_samples.values():
        area = native[sample["id"]]["pixels"]
        if area < cutoff and sample["id"] not in filtered["sample_ids"]:
            pools[
                "under1000" if area < 1000 else "1000to2999" if area < 3000 else "3000plus"
            ].append(sample)
    available = sum(map(len, pools.values()))
    if available < challenge_size:
        raise ValueError(
            f"Requested {challenge_size} challenge samples; only {available} available"
        )
    rng = random.Random(seed)
    for pool in pools.values():
        pool.sort(key=lambda s: s["id"])
        rng.shuffle(pool)
    challenge = []
    while len(challenge) < challenge_size:
        for band in sorted(pools):
            if pools[band] and len(challenge) < challenge_size:
                challenge.append(pools[band].pop())
    challenge.sort(key=lambda s: s["id"])
    cohorts = {"filtered": filtered["samples"], "small-challenge": challenge}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        conditions, snapshots = [], {}
        for cohort, samples in cohorts.items():
            paired = staging / f".{cohort}-source"
            (paired / "images").mkdir(parents=True)
            (paired / "text").mkdir()
            metadata = []
            for sample in samples:
                sample_id = sample["id"]
                source = Path(sample["crop_path"])
                if hashlib.sha256(source.read_bytes()).hexdigest() != native[sample_id]["sha256"]:
                    raise ValueError(f"Frozen crop changed: {sample_id}")
                image_path = paired / "images" / f"{sample_id}.png"
                text_path = paired / "text" / f"{sample_id}.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, image_path)
                text_path.write_text(sample["reference"], encoding="utf-8")
                metadata.append(
                    {
                        "id": sample_id,
                        "sample_type": "word",
                        "content_type": "prose",
                        "source_document": sample_id.rsplit("-", 2)[0],
                        "verification_status": sample.get("verification_status"),
                        "writer_id": sample.get("writer_id"),
                    }
                )
            (paired / "metadata.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
            )
            for profile in PROFILES:
                name = f"{cohort}-{profile}"
                snapshot = freeze(paired, staging / name, layout="paired", preprocess=profile)
                snapshots[name] = snapshot["benchmark_fingerprint"]
                conditions.append(
                    f'[[suite.conditions]]\nname = "{name}"\n'
                    f'prepared = "{name}"\nmodels = {json.dumps(MODELS)}\n'
                    f"prose_prompt = {json.dumps(PROMPT)}\n"
                )
            shutil.rmtree(paired)
        config_text = (
            "version = 3\n[experiment]\nwarmup = true\nstrict_research = false\n"
            "[suite]\nrepeats = 1\n"
        )
        for model in MODELS:
            config_text += f'\n[models."{model}"]\nnum_predict = 32\ntemperature = 0\nseed = 42\nstreaming = true\ncollect_resources = true\n'
        config_text += '\n[measurement]\nprofile = "latency"\nconcurrency = 1\nbatch_size = 1\n\n'
        config_text += "\n".join(conditions)
        (staging / "suite.toml").write_text(config_text, encoding="utf-8")
        load_config(staging / "suite.toml")
        ids = {cohort: [sample["id"] for sample in samples] for cohort, samples in cohorts.items()}
        selected_ids = set(ids["filtered"] + ids["small-challenge"])
        plan = {
            "schema_version": 1,
            "models": MODELS,
            "profiles": list(PROFILES),
            "cohorts": ids,
            "native_images": {i: native[i] for i in sorted(selected_ids)},
            "min_pixels": cutoff,
            "seed": seed,
            "snapshots": snapshots,
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "filtered_source_fingerprint": filtered["benchmark_fingerprint"],
            "challenge_source_fingerprint": old_manifest["benchmark_fingerprint"],
            "challenge_selection": "area-stratified seeded sampling from an earlier unfiltered run; no outcome-based selection",
            "preprocessing": {
                "original": "source pixels",
                "padded": "3x Lanczos enlargement, 16-pixel white border; no contrast changes",
            },
            "planned_predictions": sum(map(len, ids.values())) * len(MODELS) * len(PROFILES),
            "limitations": [
                "Exploratory previously observed cohorts, not untouched test data",
                "One pass per condition: no repeat variability estimate",
                "Model order is fixed; thermal/order effects are not isolated",
                "Resource observations are after-warmup allocations, not peaks or power measurements",
                "No direct claim about general intelligence follows from this task",
            ],
        }
        plan["integrity"] = _digest(plan)
        _json(staging / "study.json", plan)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return plan


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def _error_flags(reference, prediction, record):
    flags = _flags(record, prediction)
    left, right = normalize_text(reference), normalize_text(prediction)
    if left == right and record.get("status") == "success":
        return ["exact_match"]
    if left != right and left.casefold() == right.casefold():
        flags.append("case_only")
    if any(c.isalpha() and "LATIN" not in unicodedata.name(c, "") for c in prediction):
        flags.append("non_latin_output")
    if len(prediction) > max(32, 4 * len(reference)):
        flags.append("possible_extra_content")
    if record.get("status") != "success":
        flags.append("request_failure_or_missing")
    return sorted(set(flags)) or ["character_mismatch"]


def report_study(study, suite_output, output, *, review_count=25):
    """Create paired diagnostics and a portable human-review sheet from saved evidence."""
    study, suite_output, output = map(lambda p: Path(p).resolve(), (study, suite_output, output))
    if output.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output}")
    plan = _read_plan(study)
    if hashlib.sha256((study / "suite.toml").read_bytes()).hexdigest() != plan["config_sha256"]:
        raise ValueError("Study configuration changed after preparation")
    runs = {}
    candidates = (
        [suite_output]
        if (suite_output / "manifest.json").is_file()
        else sorted(suite_output.glob("runs/*"))
    )
    for run in candidates:
        if not (run / "manifest.json").is_file():
            continue
        _, manifest, samples, results, _, _, _ = _load_run(run)
        name = manifest.get("suite_condition")
        if name not in plan["snapshots"]:
            continue
        if name in runs:
            raise ValueError(
                f"Multiple runs for {name}; repeated studies need explicit pass aggregation"
            )
        if manifest["benchmark_fingerprint"] != plan["snapshots"][name]:
            raise ValueError(f"Run does not match frozen study condition: {name}")
        if manifest["models"] != plan["models"] or manifest["prompt"] != PROMPT:
            raise ValueError("Run models or prompt differ from the controlled study")
        for model in plan["models"]:
            effective = manifest.get("provider_controls", {}).get(model, {}).get("effective", {})
            expected = {
                "num_predict": 32,
                "temperature": 0,
                "seed": 42,
                "streaming": True,
                "collect_resources": True,
            }
            if any(effective.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "Study requires identical token, prompt-sampling and measurement controls"
                )
        runs[name] = (run, manifest, samples, results)
    if not runs:
        raise ValueError("No matching study runs found")
    evidence, resources, identities = [], [], []
    for cohort in COHORTS:
        for profile in PROFILES:
            name = f"{cohort}-{profile}"
            snapshot = load(study / name)
            run, manifest, samples, results = runs.get(name, (None, {}, {}, {}))
            if run:
                identities.append(
                    {
                        "condition": name,
                        "host": manifest.get("host"),
                        "environment": manifest.get("environment"),
                        "model_info": manifest.get("model_info"),
                        "controls": manifest.get("provider_controls"),
                    }
                )
                for warmup in read_records(run / "warmups.jsonl"):
                    resources.append(
                        {
                            "condition": name,
                            "model": warmup["model"],
                            "status": warmup.get("status"),
                            "observation": warmup.get(
                                "resource_observation", {"unavailable_reason": "Not collected"}
                            ),
                        }
                    )
            for sample in snapshot["samples"]:
                with Image.open(sample["crop_path"]) as image:
                    sent_width, sent_height = image.size
                for model in plan["models"]:
                    record = results.get((model, sample["id"]), {"status": "missing"})
                    prediction = record.get("prediction") or ""
                    scored_prediction = prediction if record.get("status") == "success" else ""
                    metrics = score(scored_prediction, sample["reference"])
                    timings = {
                        item["metric_name"]: item for item in record.get("timing_observations", [])
                    }
                    native = plan["native_images"][sample["id"]]
                    evidence.append(
                        {
                            "cohort": cohort,
                            "profile": profile,
                            "condition": name,
                            "model": model,
                            "sample_id": sample["id"],
                            "reference": sample["reference"],
                            "prediction": prediction,
                            "status": record["status"],
                            "exact_match": metrics["exact_match"],
                            "cer": metrics["cer"],
                            "char_edits": metrics["char_edits"],
                            "reference_chars": metrics["reference_chars"],
                            "insertions": metrics["char_insertions"],
                            "deletions": metrics["char_deletions"],
                            "substitutions": metrics["char_substitutions"],
                            "native_width": native["width"],
                            "native_height": native["height"],
                            "native_pixels": native["pixels"],
                            "native_aspect_ratio": native["aspect_ratio"],
                            "native_sha256": native["sha256"],
                            "sent_width": sent_width,
                            "sent_height": sent_height,
                            "reference_length": len(sample["reference"]),
                            "output_length": len(prediction),
                            "truncated": record.get("truncated"),
                            "flags": _error_flags(sample["reference"], prediction, record),
                            "latency_seconds": record.get("latency_seconds"),
                            "ttfo_seconds": timings.get("time_to_first_output", {}).get("value"),
                            "ttfo_source": timings.get("time_to_first_output", {}).get(
                                "source", "unavailable"
                            ),
                            "generation_seconds": timings.get("generation", {}).get("value"),
                            "model_load_seconds": timings.get("model_load", {}).get("value"),
                            "prompt_evaluation_seconds": timings.get("prompt_evaluation", {}).get(
                                "value"
                            ),
                            "image_encoding_seconds": timings.get("image_encoding", {}).get(
                                "value"
                            ),
                            "generation_tokens_per_second": timings.get(
                                "tokens_per_second", {}
                            ).get("value"),
                            "input_tokens": (record.get("usage") or {}).get("input_tokens"),
                            "output_tokens": (record.get("usage") or {}).get("output_tokens"),
                            "cache_hit": record.get("cache_hit", False),
                            "error": record.get("error"),
                        }
                    )
    aggregates = []
    for cohort in COHORTS:
        for profile in PROFILES:
            for model in plan["models"]:
                rows = [
                    r
                    for r in evidence
                    if (r["cohort"], r["profile"], r["model"]) == (cohort, profile, model)
                ]
                latency = sorted(
                    r["latency_seconds"]
                    for r in rows
                    if r["status"] == "success"
                    and not r["cache_hit"]
                    and r["latency_seconds"] is not None
                )
                aggregates.append(
                    {
                        "cohort": cohort,
                        "profile": profile,
                        "model": model,
                        "expected": len(rows),
                        "completed": sum(r["status"] != "missing" for r in rows),
                        "successful": sum(r["status"] == "success" for r in rows),
                        "exact_matches": sum(r["exact_match"] for r in rows),
                        "exact_match_rate": statistics.mean(r["exact_match"] for r in rows),
                        "corpus_cer": sum(r["char_edits"] for r in rows)
                        / sum(r["reference_chars"] for r in rows),
                        "truncated": sum(r["truncated"] is True for r in rows),
                        "p50_seconds": statistics.median(latency) if latency else None,
                        "p95_seconds": latency[math.ceil(0.95 * len(latency)) - 1]
                        if latency
                        else None,
                        "flags": dict(Counter(flag for r in rows for flag in r["flags"])),
                    }
                )
    pairs = _paired_summary(evidence, plan)
    report = {
        "schema_version": 1,
        "study_id": plan["integrity"],
        "summaries": aggregates,
        "pairs": pairs,
        "resources": resources,
        "identities": identities,
        "limitations": plan["limitations"]
        + ["Automatic flags are review suggestions, not established causes"],
    }
    report["evidence_id"] = _digest(evidence)
    output.mkdir(parents=True)
    _json(output / "diagnostics.json", report)
    _json(output / "observations.json", evidence)
    _write_csv(output / "observations.csv", evidence)
    _write_csv(output / "summary.csv", aggregates)
    _write_csv(output / "paired.csv", pairs)
    subgroups = []
    for cohort in COHORTS:
        for profile in PROFILES:
            for model in plan["models"]:
                for kind, predicate in (
                    ("punctuation", lambda r: not any(c.isalnum() for c in r["reference"])),
                    ("single_character", lambda r: len(r["reference"]) == 1),
                    ("2_to_4_characters", lambda r: 2 <= len(r["reference"]) <= 4),
                    ("5_plus_characters", lambda r: len(r["reference"]) >= 5),
                    ("under_1000_pixels", lambda r: r["native_pixels"] < 1000),
                    ("1000_to_9999_pixels", lambda r: 1000 <= r["native_pixels"] < 10000),
                    ("10000_plus_pixels", lambda r: r["native_pixels"] >= 10000),
                ):
                    rows = [
                        r
                        for r in evidence
                        if (r["cohort"], r["profile"], r["model"]) == (cohort, profile, model)
                        and predicate(r)
                    ]
                    if rows:
                        subgroups.append(
                            {
                                "cohort": cohort,
                                "profile": profile,
                                "model": model,
                                "group": kind,
                                "expected": len(rows),
                                "exact_match_rate": statistics.mean(r["exact_match"] for r in rows),
                                "corpus_cer": sum(r["char_edits"] for r in rows)
                                / sum(r["reference_chars"] for r in rows),
                            }
                        )
    _write_csv(output / "subgroups.csv", subgroups)
    _write_review(output, study, plan, evidence, report, review_count)
    return report


def _paired_summary(evidence, plan):
    by_key = {(r["cohort"], r["profile"], r["model"], r["sample_id"]): r for r in evidence}
    pairs = []
    for cohort in COHORTS:
        contrasts = [("model_size", profile, MODELS[0], profile, MODELS[1]) for profile in PROFILES]
        contrasts += [("image_preparation", "original", model, "padded", model) for model in MODELS]
        for kind, lp, lm, rp, rm in contrasts:
            counts = Counter()
            for sample_id in plan["cohorts"][cohort]:
                left, right = by_key[cohort, lp, lm, sample_id], by_key[cohort, rp, rm, sample_id]
                if "missing" in {left["status"], right["status"]}:
                    counts["missing_pair"] += 1
                else:
                    counts[
                        "both_correct"
                        if left["exact_match"] and right["exact_match"]
                        else "right_only_correct"
                        if right["exact_match"]
                        else "left_only_correct"
                        if left["exact_match"]
                        else "both_wrong"
                    ] += 1
            pairs.append(
                {
                    "cohort": cohort,
                    "contrast": kind,
                    "left": f"{lm}/{lp}",
                    "right": f"{rm}/{rp}",
                    **{
                        key: counts[key]
                        for key in (
                            "both_correct",
                            "right_only_correct",
                            "left_only_correct",
                            "both_wrong",
                            "missing_pair",
                        )
                    },
                }
            )
    return pairs


def _write_review(output, study, plan, evidence, report, count):
    groups = defaultdict(list)
    for row in evidence:
        groups[row["cohort"], row["sample_id"]].append(row)
    buckets = defaultdict(list)
    for key, rows in sorted(groups.items()):
        present = [r for r in rows if r["status"] != "missing"]
        correct = sum(r["exact_match"] for r in present)
        category = (
            "disagreement"
            if 0 < correct < len(present)
            else "both_wrong"
            if present and not correct
            else "control"
        )
        buckets[key[0], category].append(key)
    selected = []
    rng = random.Random(44)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    while len(selected) < min(count, len(groups)):
        for bucket in sorted(buckets):
            if buckets[bucket] and len(selected) < count:
                selected.append(buckets[bucket].pop())
    assets = output / "images"
    assets.mkdir()
    snapshots = {
        f"{c}-{p}": {s["id"]: s for s in load(study / f"{c}-{p}")["samples"]}
        for c in COHORTS
        for p in PROFILES
    }
    cards, review_rows = [], []

    def escape(value):
        return html.escape(str(value), quote=True)

    categories = [
        "unreviewed",
        "correct",
        "wrong_characters",
        "instruction_failure",
        "image_ambiguous",
        "image_unreadable",
        "possible_reference_issue",
        "truncated",
        "uncertain",
    ]
    for index, (cohort, sample_id) in enumerate(selected):
        images = []
        for profile in PROFILES:
            sample = snapshots[f"{cohort}-{profile}"][sample_id]
            source = Path(sample["crop_path"])
            relative = f"images/{index}-{profile}.png"
            shutil.copyfile(source, output / relative)
            images.append(
                f'<figure><figcaption>{profile}</figcaption><img src="{relative}" alt="{escape(sample_id)}"></figure>'
            )
        predictions = []
        for row in groups[cohort, sample_id]:
            rid = len(review_rows)
            review_rows.append(
                {
                    "cohort": cohort,
                    "sample_id": sample_id,
                    "model": row["model"],
                    "profile": row["profile"],
                    "native_sha256": row["native_sha256"],
                    "human_category": "unreviewed",
                    "notes": "",
                }
            )
            options = "".join(f"<option>{category}</option>" for category in categories)
            predictions.append(
                f"<tr><td>{escape(row['model'])}/{row['profile']}</td>"
                f"<td><pre>{escape(row['prediction'])}</pre><small>{escape(row['flags'])}; {escape(row['status'])}</small></td>"
                f'<td><select data-review="{rid}">{options}</select><textarea data-notes="{rid}" aria-label="Review notes"></textarea></td></tr>'
            )
        native = plan["native_images"][sample_id]
        cards.append(
            f"<article><h2>{escape(cohort)} / {escape(sample_id)}</h2>"
            f"<p>Reference: <strong>{escape(groups[cohort, sample_id][0]['reference'])}</strong> · Native {native['width']} × {native['height']} pixels</p>"
            f'<div class="images">{"".join(images)}</div><table><tr><th>Model / preparation</th><th>Prediction / suggestions</th><th>Your assessment</th></tr>{"".join(predictions)}</table></article>'
        )
    payload = {
        "study_id": plan["integrity"],
        "evidence_id": report["evidence_id"],
        "reviews": review_rows,
    }
    _json(output / "review-template.json", payload)
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    table = "".join(
        f"<tr><td>{escape(r['cohort'])}</td><td>{r['profile']}</td><td>{escape(r['model'])}</td>"
        f"<td>{r['exact_matches']}/{r['expected']}</td><td>{r['successful']}/{r['expected']}</td><td>{r['corpus_cer']:.1%}</td><td>{escape(r['p50_seconds'])}</td></tr>"
        for r in report["summaries"]
    )
    body = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Qwen diagnostic review</title>
<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:16px;background:#f4f6fa;color:#17243c}article{background:white;padding:24px;margin:24px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:480px}.images{display:flex;gap:24px;flex-wrap:wrap}figure{margin:8px;max-width:45%}img{max-width:100%;max-height:180px;object-fit:contain}textarea{display:block;width:95%;margin-top:8px}button{padding:12px;cursor:pointer}small{color:#53627a}</style>
<h1>Qwen handwriting diagnostic study</h1><p>Separate cohorts; identical samples within each contrast. Flags suggest review, not proven causes. One pass provides no repeat variability. Missing predictions remain visible.</p>
<p>Memory is a provider-reported allocation snapshot after warmup, not peak usage. Full timings, usage, resource availability and identities are in diagnostics.json and observations.csv.</p>
<table><tr><th>Cohort</th><th>Preparation</th><th>Model</th><th>Exact matches</th><th>Successful requests</th><th>CER</th><th>Median seconds</th></tr>"""
    body += (
        table
        + "</table><h2>Paired changes</h2><p>Right-only correct means the right condition fixed an error; left-only correct means it introduced one. These counts describe this study, not a proven cause.</p>"
    )
    body += "<table><tr><th>Cohort</th><th>Left → right</th><th>Both correct</th><th>Right only</th><th>Left only</th><th>Both wrong</th><th>Missing pair</th></tr>"
    for pair in report.get("pairs", []):
        body += (
            f"<tr><td>{escape(pair['cohort'])}</td><td>{escape(pair['left'])} → {escape(pair['right'])}</td>"
            + "".join(
                f"<td>{pair[key]}</td>"
                for key in (
                    "both_correct",
                    "right_only_correct",
                    "left_only_correct",
                    "both_wrong",
                    "missing_pair",
                )
            )
            + "</tr>"
        )
    body += "</table><h2>Loaded-model allocation after warmup</h2><table><tr><th>Condition / model</th><th>Allocated GiB</th><th>Provider size_vram GiB</th><th>Availability</th></tr>"
    for resource in report.get("resources", []):
        observation = resource["observation"]
        values = [observation.get(key) for key in ("allocated_bytes", "size_vram_bytes")]
        body += (
            f"<tr><td>{escape(resource['condition'])} / {escape(resource['model'])}</td>"
            + "".join(
                f"<td>{value / 1024**3:.3f}</td>" if value is not None else "<td>Unavailable</td>"
                for value in values
            )
            + f"<td>{escape(observation.get('unavailable_reason') or 'Snapshot; not peak usage')}</td></tr>"
        )
    body += (
        '</table><p>Review categories and notes below, then export your assessments. Download them before closing; edits are not saved automatically.</p><button id="export">Download reviewed labels</button>'
        + "".join(cards)
    )
    body += f'<script id="review-data" type="application/json">{encoded}</script>'
    body += """<script>document.getElementById('export').onclick=()=>{const p=JSON.parse(document.getElementById('review-data').textContent);document.querySelectorAll('[data-review]').forEach(e=>p.reviews[Number(e.dataset.review)].human_category=e.value);document.querySelectorAll('[data-notes]').forEach(e=>p.reviews[Number(e.dataset.notes)].notes=e.value);const u=URL.createObjectURL(new Blob([JSON.stringify(p,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='reviewed-labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};</script></html>"""
    (output / "review.html").write_text(body, encoding="utf-8")
