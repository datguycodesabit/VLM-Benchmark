"""Create exam-grouped, portable training manifests and a frozen test snapshot.

No training or inference is performed. Existing output directories are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_bench.dataset import check_dataset  # noqa: E402
from vlm_bench.snapshot import freeze, load  # noqa: E402

SPLIT_YEARS = {
    "train": [2006, 2008, 2009, 2011, 2012],
    "validation": [2013],
    "test": [2014, 2015],
}
PROMPTS = {
    "prose": "Transcribe all visible printed text exactly. Preserve wording, numbers, "
    "units and punctuation. Represent inline mathematical notation with $...$ LaTeX. "
    "Do not solve the question or correct mistakes. Return only the transcription.",
    "equation": "Transcribe the visible mathematical content as LaTeX, one row per line, "
    "without outer math delimiters. Preserve signs, subscripts, superscripts, values "
    "and units. Include words within the target using \\mathrm{...}. Ignore frames, "
    "underlines and scan specks. Do not solve or correct the work. Return only the transcription.",
}
# These are readable source errors/context mismatches, not uncertain transcriptions.
SOURCE_NOTES = {
    "real/exam2009/page02-prose05": "Printed release height differs from adjacent source working.",
    "real/exam2009/page04-equation05": "Visible result is inconsistent with adjacent arithmetic.",
    "real/exam2009/page05-prose01": "Printed 24 m/s differs from 2.4 used in adjacent working.",
    "real/exam2011/page05-equation03": "Preserve the visible handwritten substitution; do not solve again.",
    "real/exam2014/page04-equation04": "Visible acceleration signs conflict with the stated motion.",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def safe_id(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError(f"Unsafe sample ID: {value!r}")
    return value


def assign_splits(records: list[dict]) -> list[dict]:
    """Keep an exam, its pages, and all derived pairs in exactly one split."""
    year_splits = {year: split for split, years in SPLIT_YEARS.items() for year in years}
    result, pairs, ids = [], defaultdict(list), set()
    for original in records:
        r = json.loads(json.dumps(original))
        sid = safe_id(r["id"])
        if sid in ids:
            raise ValueError(f"Duplicate sample ID: {sid}")
        ids.add(sid)
        match = re.fullmatch(r"(real|synthetic)/exam(\d{4})/(.+)", sid)
        if not match:
            raise ValueError(f"Unexpected Brothers sample ID: {sid}")
        origin, year, _ = match.groups()
        year = int(year)
        if year not in year_splits or r["origin"] != origin:
            raise ValueError(f"Unknown exam/origin: {sid}")
        ann = r.setdefault("annotations", {})
        pair = ann.get("pair_id")
        if pair != sid.split("/", 1)[1]:
            raise ValueError(f"Pair lineage disagrees with sample ID: {sid}")
        expected_source = f"Exam 1 Fall {year}.pdf"
        if r["source_document"] not in {
            expected_source, "synthetic-inspired-by-" + expected_source
        }:
            raise ValueError(f"Source document disagrees with exam year: {sid}")
        ann["original_source_document"] = r["source_document"]
        ann["exam_year"] = year
        ann["split_group"] = f"exam{year}"
        ann["label_audit"] = "structural_audit_with_prior_assistant_visual_review"
        ann["independent_human_review"] = False
        r["source_document"] = expected_source
        r["split"] = year_splits[year]
        pairs[pair].append(r)
        result.append(r)
    for pair, members in pairs.items():
        if len(members) != 2 or {r["origin"] for r in members} != {"real", "synthetic"}:
            raise ValueError(f"Missing real/synthetic pair: {pair}")
        if len({(r["split"], r["content_type"]) for r in members}) != 1:
            raise ValueError(f"Pair crosses split or content type: {pair}")
        real_id = "real/" + pair
        if any(r.get("paired_source_id") != real_id for r in members):
            raise ValueError(f"Incorrect paired_source_id: {pair}")
    return sorted(result, key=lambda r: r["id"])


def audit_label(text: str, kind: str) -> list[str]:
    findings = []
    if not text.strip():
        findings.append("empty_reference")
    if text != unicodedata.normalize("NFC", text):
        findings.append("non_nfc_unicode")
    if any(ord(c) < 32 and c not in "\n\t" for c in text):
        findings.append("control_character")
    if "\ufffd" in text or re.search(r"\b(?:TODO|TBD|UNREADABLE|ILLEGIBLE)\b", text):
        findings.append("unresolved_label")
    depth = 0
    for c in text:
        depth += (c == "{") - (c == "}")
        if depth < 0:
            break
    if depth != 0:
        findings.append("unbalanced_latex_braces")
    if kind == "prose" and text.count("$") % 2:
        findings.append("unbalanced_inline_math")
    if kind not in {"prose", "equation"}:
        findings.append("unknown_content_type")
    return findings


def verify_package(root: Path) -> dict:
    """Read-only check of content hashes, manifests, split membership and snapshot."""
    lock = json.loads((root / "package-lock.json").read_text())
    for name, expected in lock["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Package file changed: {name}")
    metadata = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in metadata}
    if len(by_id) != len(metadata):
        raise ValueError("Duplicate metadata IDs")
    groups, pairs = defaultdict(set), defaultdict(set)
    references, images = defaultdict(set), defaultdict(set)
    for split in SPLIT_YEARS:
        records = [json.loads(line) for line in (root / "manifests" / f"{split}.jsonl").read_text().splitlines()]
        expected_ids = {r["id"] for r in metadata if r["split"] == split}
        if len(records) != len(expected_ids) or {r["id"] for r in records} != expected_ids:
            raise ValueError(f"Manifest membership mismatch: {split}")
        for row in records:
            meta = by_id[row["id"]]
            if row["split"] != split or meta["split"] != split:
                raise ValueError(f"Wrong split: {row['id']}")
            if (root / row["reference_path"]).read_text().strip() != row["text"]:
                raise ValueError(f"Manifest target changed: {row['id']}")
            if digest(root / row["image"]) != row["image_sha256"]:
                raise ValueError(f"Image hash changed: {row['id']}")
            groups[meta["source_document"]].add(split)
            pairs[row["pair_id"]].add(split)
            images[row["image_sha256"]].add(split)
            normalized = " ".join(row["text"].split()).casefold()
            references[normalized].add(split)
    for label, buckets in [("exam", groups), ("pair", pairs), ("image", images), ("reference", references)]:
        if any(len(splits) > 1 for splits in buckets.values()):
            raise ValueError(f"Cross-split {label} leakage")
    snapshot = load(root / "frozen-test")
    expected_test = {r["id"] for r in metadata if r["split"] == "test"}
    if set(snapshot["sample_ids"]) != expected_test:
        raise ValueError("Frozen test membership mismatch")
    if snapshot["benchmark_fingerprint"] != lock["test_fingerprint"]:
        raise ValueError("Frozen test fingerprint changed")
    report = check_dataset(root, layout="paired")
    if not report["valid"]:
        raise ValueError(f"Dataset checker rejected package: {report['issues']}")
    return {
        "valid": True,
        "samples": len(metadata),
        "split_counts": dict(Counter(r["split"] for r in metadata)),
        "cross_split_exams": 0,
        "cross_split_pairs": 0,
        "cross_split_exact_images": 0,
        "cross_split_normalized_references": 0,
        "test_fingerprint": snapshot["benchmark_fingerprint"],
        "frozen_test_samples": len(snapshot["sample_ids"]),
        "perceptual_similarity_review_groups": len(report["findings"]),
        "model_inference_run": False,
    }


def near_text_audit(rows: list[dict]) -> list[dict]:
    """Flag close wording across real exam prose, ignoring numerical changes."""
    eligible = [r for r in rows if r["origin"] == "real" and r["content_type"] == "prose"
                and len(r["text"].split()) >= 12]
    findings = []
    for index, a in enumerate(eligible):
        for b in eligible[index + 1:]:
            if a["split"] == b["split"]:
                continue
            texts = [re.sub(r"\d+(?:\.\d+)?", "<number>", r["text"].casefold())
                     for r in [a, b]]
            similarity = SequenceMatcher(None, *texts, autojunk=False).ratio()
            if similarity >= 0.90:
                findings.append({"ids": [a["id"], b["id"]], "similarity": similarity,
                                 "status": "review", "blocking": False})
    return findings


def write_instructions(output: Path, rows: list[dict], report: dict) -> None:
    lines = ["# Physics transcription training and testing", "",
             "This version contains 546 image/transcription pairs. No model has been trained or run.",
             "", "| Split | Exams | Images | Prose | Equations | Real / synthetic |",
             "|---|---|---:|---:|---:|---:|"]
    for split, years in SPLIT_YEARS.items():
        selected = [r for r in rows if r["split"] == split]
        counts = Counter(r["content_type"] for r in selected)
        origins = Counter(r["origin"] for r in selected)
        lines.append(f"| {split} | {', '.join(map(str, years))} | {len(selected)} | "
                     f"{counts['prose']} | {counts['equation']} | "
                     f"{origins['real']} / {origins['synthetic']} |")
    lines += ["", "## Train", "",
              "Read `manifests/train.jsonl`. Each row supplies `image`, `text`, `content_type`, "
              "and `sample_type`. Image and reference paths are relative to this package root. "
              "Use the image as input and `text` as the target. Do not feed IDs, source years, "
              "reference files, audit sheets or target text into the model's input prompt.",
              "", "```python", "import json", "from pathlib import Path", "from PIL import Image",
              'root = Path("data/brothers-training-v1")',
              'with (root / "manifests/train.jsonl").open() as stream:',
              "    for line in stream:", "        example = json.loads(line)",
              '        image = Image.open(root / example["image"]).convert("RGB")',
              '        target = example["text"]',
              "        # Pass image and target to the selected model's training processor.", "```",
              "", "No architecture-specific resizing, tokenizer, learning rate or training loop "
              "is chosen here. Those depend on the model and hardware. Do not train from scratch "
              "on this small set. For a single-line prose recognizer, use `train.prose-lines.jsonl` "
              "and the matching validation/test variants; it is a separate, smaller benchmark. "
              "Do not send equations or page samples to a model that does not support them.",
              "", "## Validate and select", "",
              "Use only `manifests/validation.jsonl` to choose checkpoints, prompts, resizing, "
              "decoding and training settings. `prompts.json` provides fixed transcription prompts "
              "for prose and equations. If a chosen model requires changes, settle them using "
              "training/validation data and record them before test evaluation.",
              "", "## Test once settings are fixed", "",
              "`frozen-test/` is an integrity-checked v2 snapshot containing exactly the 134 held-out "
              "test samples. Use the same snapshot and supported sample subset for the baseline "
              "and fine-tuned model. `manifests/test.inputs.jsonl` omits the answers for inference. "
              "References are for the scoring stage only.", "",
              "From the VLM-Benchmark repository, inspect a compatible model without inference:",
              "", "```bash", "python -m vlm_bench run --prepared data/brothers-training-v1/frozen-test \\",
              "  --models PROVIDER:MODEL --dry-run", "```", "",
              "Replace `PROVIDER:MODEL` with the chosen supported model/checkpoint. Remove `--dry-run` "
              "only when evaluation is intended. No evaluation has been run by this preparation task.",
              "", "Report prose CER/WER and exact matches separately from equation transcription "
              "metrics. Also separate real scans from typeset controls, and report sample counts. "
              "Do not pool a prose-only model with a full-dataset model on different samples.",
              "", "## Audit and review", "",
              "`audit/labels.jsonl` records every image/reference check. No references were changed. "
              "No unresolved markers, invalid braces, control characters, blank images or missing "
              "pairs were found. `audit/review-queue.jsonl` lists 99 real equation crops for "
              "optional independent review. This recommendation is not a claim that those labels "
              "are wrong. Visible source physics mistakes are recorded and preserved.",
              "", "References were previously visually checked by the assistant; they are not "
              "independently human-certified. Two targeted source crops were rechecked during "
              "this setup. All labels received automated structural checks. If a label is later "
              "uncertain, quarantine it and its derived pair in a new version before training; "
              "do not silently change the frozen test labels.",
              "", "Open `audit/review.html` to inspect each queued image beside its transcription.",
              "", "`audit/near-text.json` checks real prose of at least 12 words for cross-split "
              "wording similarity at or above 0.90, with numbers replaced. This is a heuristic, "
              "not a semantic or pretraining-contamination guarantee. Common physics concepts "
              "naturally occur in multiple exams. Writer independence is not established.",
              "", "## Integrity and rebuilding", "",
              f"Frozen test fingerprint: `{report['test_fingerprint']}`", "",
              "Run `python verify_integrity.py` from this package for a standard-library-only "
              "file-integrity check. For a full dataset and snapshot verification, run from the repo:",
              "", "```bash", "python scripts/prepare_brothers_training.py --verify data/brothers-training-v1",
              "```", "",
              "Rebuild from the unchanged original dataset into a new output directory:", "",
              "```bash", "python scripts/prepare_brothers_training.py --source data/brothers \\",
              "  --output data/brothers-training-v2", "```", "",
              "Existing output directories are refused. `split-policy.json` pins the source metadata "
              "hash and exam assignments. `package-lock.json` hashes the package files. "
              "The original `data/brothers` dataset and its existing archive remain unchanged."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cards = []
    for row in rows:
        if row["origin"] != "real" or row["content_type"] != "equation":
            continue
        cards.append(
            f'<article><h2>{html.escape(row["id"])}</h2>'
            f'<p>Split: {row["split"]}. '
            f'{html.escape(SOURCE_NOTES.get(row["id"], "Independent second review recommended."))}</p>'
            f'<img src="../{html.escape(row["image"])}" alt="Equation transcription target">'
            f'<pre>{html.escape(row["text"])}</pre></article>'
        )
    (output / "audit" / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Physics reference review</title><style>'
        'body{font:16px system-ui;margin:32px;background:#f3f5f7}'
        'article{padding:24px;margin:24px 0;background:white;border:1px solid #ccc}'
        'h2{font-size:18px}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        '</style><h1>Real equation reference review</h1>'
        '<p>99 equation crops. Previously checked by the assistant; independent second review '
        'is recommended. Readable source physics mistakes are preserved. '
        'This page shows test answers: do not use it for model inputs or tune on test results.</p>'
        + ''.join(cards) + '</html>', encoding="utf-8"
    )
    (output / "verify_integrity.py").write_text('''"""Check packaged file hashes without third-party dependencies."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
lock = json.loads((root / "package-lock.json").read_text())
for name, expected in lock["files"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("Integrity check failed: " + name)
print("All", len(lock["files"]), "locked files verified.")
print("Frozen test fingerprint:", lock["test_fingerprint"])
''', encoding="utf-8")


def build(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new version: {output}")
    records = assign_splits([
        json.loads(line) for line in (source / "metadata.jsonl").read_text().splitlines()
    ])
    # Preflight labels before creating a usable dataset or test snapshot.
    errors = []
    for r in records:
        sid = r["id"]
        text = (source / "text" / f"{sid}.txt").read_text(encoding="utf-8")
        findings = audit_label(text, r["content_type"])
        if r["verification_status"] != "verified":
            findings.append("source_not_verified")
        if findings:
            errors.append({"id": sid, "findings": findings})
    if errors:
        raise ValueError(f"Label audit failed; quarantine/review before splitting: {errors}")
    output.mkdir(parents=True)
    rows, audit, review = [], [], []
    for r in records:
        sid = r["id"]
        for folder, ext in [("images", "png"), ("text", "txt")]:
            dest = output / folder / f"{sid}.{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / folder / f"{sid}.{ext}", dest)
        image = output / "images" / f"{sid}.png"
        target = output / "text" / f"{sid}.txt"
        with Image.open(image) as im:
            im.load()
            rgb = im.convert("RGB")
            if ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox() is None:
                raise ValueError(f"Blank image: {sid}")
            width, height = im.size
        text = target.read_text(encoding="utf-8").strip()
        rows.append({
            "id": sid, "image": f"images/{sid}.png", "reference_path": f"text/{sid}.txt",
            "text": text, "split": r["split"], "origin": r["origin"],
            "content_type": r["content_type"], "sample_type": r["sample_type"],
            "source_document": r["source_document"], "exam_year": r["annotations"]["exam_year"],
            "pair_id": r["annotations"]["pair_id"], "width": width, "height": height,
            "image_sha256": digest(image), "reference_sha256": digest(target),
            "verification_status": "verified", "independent_human_review": False,
        })
        audit.append({"id": sid, "split": r["split"], "structural_checks": "passed",
                      "unresolved_label": False, "reference_changed": False,
                      "source_reference_sha256": digest(source / "text" / f"{sid}.txt"),
                      "reference_sha256": digest(target), "source_note": SOURCE_NOTES.get(sid),
                      "verification_basis": r["annotations"]["verification_method"]})
        if r["origin"] == "real" and r["content_type"] == "equation":
            review.append({"id": sid, "split": r["split"], "image": f"images/{sid}.png",
                           "reference_path": f"text/{sid}.txt", "text": text,
                           "status": "second_review_recommended", "blocking": False,
                           "reason": SOURCE_NOTES.get(sid, "Real equation crop; independent review recommended.")})
    write_jsonl(output / "metadata.jsonl", records)
    write_jsonl(output / "audit" / "labels.jsonl", audit)
    write_jsonl(output / "audit" / "review-queue.jsonl", review)
    write_json(output / "audit" / "label-summary.json", {
        "samples_checked": len(rows), "structural_failures": 0, "unresolved_labels": 0,
        "references_modified": 0, "independent_human_review": False,
        "second_review_equations": len(review), "source_math_notes": SOURCE_NOTES,
        "scope": "Automated label/image checks, prior assistant visual review, and targeted rechecks. "
        "This does not certify that every transcription is error-free.",
    })
    for split in SPLIT_YEARS:
        selected = [r for r in rows if r["split"] == split]
        write_jsonl(output / "manifests" / f"{split}.jsonl", selected)
        for origin in ["real", "synthetic"]:
            for kind in ["prose", "equation"]:
                write_jsonl(output / "manifests" / f"{split}.{origin}.{kind}.jsonl", [
                    r for r in selected if r["origin"] == origin and r["content_type"] == kind
                ])
        write_jsonl(output / "manifests" / f"{split}.prose-lines.jsonl", [
            r for r in selected if r["content_type"] == "prose" and r["sample_type"] == "line"
        ])
    write_jsonl(output / "manifests" / "test.inputs.jsonl", [
        {key: r[key] for key in ["id", "image", "origin", "content_type", "sample_type"]}
        for r in rows if r["split"] == "test"
    ])
    write_json(output / "split-policy.json", {
        "version": 1, "protocol": "document-disjoint", "year_assignments": SPLIT_YEARS,
        "grouping": "Source exam year, including every page and synthetic counterpart",
        "writer_disjoint": False, "pretraining_contamination_assessed": False,
        "source_metadata_sha256": digest(source / "metadata.jsonl"),
    })
    write_json(output / "prompts.json", PROMPTS)
    write_json(output / "audit" / "near-text.json", {
        "scope": "real prose across splits; at least 12 words; numbers replaced; similarity >= 0.90",
        "findings": near_text_audit(rows),
    })
    snapshot = freeze(output, output / "frozen-test", layout="paired", preprocess="original",
                      split="test", seed=42)
    write_json(output / "package-lock.json", {
        "version": 1, "test_fingerprint": snapshot["benchmark_fingerprint"],
        "files": {p.relative_to(output).as_posix(): digest(p)
                  for p in sorted(output.rglob("*")) if p.is_file()},
    })
    report = verify_package(output)
    write_json(output / "audit" / "validation.json", report)
    write_instructions(output, rows, report)
    lock = json.loads((output / "package-lock.json").read_text())
    lock["files"] = {p.relative_to(output).as_posix(): digest(p)
                     for p in sorted(output.rglob("*"))
                     if p.is_file() and p != output / "package-lock.json"}
    write_json(output / "package-lock.json", lock)
    verify_package(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/brothers"))
    parser.add_argument("--output", type=Path, default=Path("data/brothers-training-v1"))
    parser.add_argument("--verify", type=Path, help="Read-only verification of an existing package")
    args = parser.parse_args()
    result = verify_package(args.verify) if args.verify else build(args.source, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
