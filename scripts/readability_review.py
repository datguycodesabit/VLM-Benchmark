"""Create model-blind readability reviews and freeze explicitly approved samples."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from vlm_bench.runner import _digest
from vlm_bench.snapshot import freeze, load


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _known_ids(runs):
    ids = set()
    if runs:
        for path in Path(runs).rglob("results.jsonl"):
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        ids.add(json.loads(line)["sample_id"])
    return ids


def _paired_source(samples, source, *, reviewed=None):
    (source / "images").mkdir(parents=True)
    (source / "text").mkdir()
    metadata = []
    for sample in samples:
        sid = sample["id"]
        if not sid or Path(sid).name != sid or sid in {".", ".."}:
            raise ValueError("This IAM review requires simple sample IDs")
        shutil.copyfile(sample["crop_path"], source / "images" / f"{sid}.png")
        reference = sample["reference"]
        if reviewed:
            decision = reviewed[sid]
            if decision["reference_status"] == "corrected":
                reference = decision["transcription"].strip()
        (source / "text" / f"{sid}.txt").write_text(reference, encoding="utf-8")
        kind = sample.get("sample_type") or "word"
        metadata.append(
            {
                "id": sid,
                "sample_type": kind,
                "content_type": "prose",
                "source_document": sid.rsplit("-", 2 if kind == "word" else 1)[0],
                "writer_id": sample.get("writer_id"),
                "verification_status": "verified" if reviewed else "unresolved",
                "split": "test" if reviewed else None,
            }
        )
    (source / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )


def prepare(
    data,
    output,
    *,
    layout="iam-words",
    count=50,
    min_pixels=10000,
    seed=20261102,
    exclude_runs=None,
):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be positive")
    seen = _known_ids(exclude_runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        # IDs exclude previously measured examples; prediction quality is never consulted.
        with tempfile.TemporaryDirectory(prefix="vlm-readable-pool-") as tmp:
            pool = freeze(
                data,
                Path(tmp) / "pool",
                layout=layout,
                limit=count * 2,
                min_pixels=min_pixels,
                seed=seed,
            )
            eligible = [sample for sample in pool["samples"] if sample["id"] not in seen]
            if len(eligible) < count:
                raise ValueError(
                    "Insufficient fresh candidates; choose another seed or smaller count"
                )
            selected = random.Random(seed).sample(eligible, count)
            paired = Path(tmp) / "paired"
            _paired_source(selected, paired)
            frozen = freeze(paired, staging / "candidates", layout="paired")
        records = []
        for sample in frozen["samples"]:
            with Image.open(sample["crop_path"]) as image:
                width, height = image.size
            records.append(
                {
                    "id": sample["id"],
                    "reference": sample["reference"],
                    "width": width,
                    "height": height,
                    "sha256": sample["hashes"]["crop"],
                }
            )
        manifest = {
            "schema_version": 1,
            "purpose": "model-blind readable IAM calibration review",
            "layout": layout,
            "min_pixels": min_pixels,
            "seed": seed,
            "previously_measured_ids_excluded": exclude_runs is not None,
            "candidate_fingerprint": frozen["benchmark_fingerprint"],
            "samples": records,
            "notes": "Readability and reference approval precede new inference. Prior IAM training overlap remains unknown.",
        }
        manifest["review_id"] = _digest(manifest)
        _write_json(staging / "review.json", manifest)
        _render(staging, manifest)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output / "review.html"


def _render(root, manifest):
    decisions, cards = [], []
    for i, sample in enumerate(manifest["samples"]):
        decisions.append(
            {
                "id": sample["id"],
                "sha256": sample["sha256"],
                "readability": "unreviewed",
                "reference_status": "unreviewed",
                "transcription": "",
                "notes": "",
            }
        )
        escape = html.escape
        image_id = quote(sample["id"], safe="")
        cards.append(f'''<article><h2>{i + 1}. {escape(sample["id"])}</h2>
<p>Native image: {sample["width"]} × {sample["height"]} pixels. No model predictions are shown.</p>
<a href="candidates/crops/{image_id}.png" target="_blank"><img src="candidates/crops/{image_id}.png" alt="Handwritten sample"></a>
<label>1. Can you confidently read this image without a sentence?<select data-key="readability" data-index="{i}"><option>unreviewed</option><option>clearly_readable</option><option>ambiguous</option><option>unreadable</option></select></label>
<label>2. Type what you read before revealing the reference:<input data-key="transcription" data-index="{i}" autocomplete="off"></label>
<details><summary>3. Reveal the IAM reference to verify it</summary><p class="reference">{escape(sample["reference"])}</p></details>
<label>Reference decision:<select data-key="reference_status" data-index="{i}"><option>unreviewed</option><option>confirmed</option><option>corrected</option><option>uncertain</option></select></label>
<p class="hint">Confirmed: IAM reference agrees with the writing. Corrected: your typed transcription is the verified replacement. Uncertain: do not score this image yet.</p>
<label>Notes:<textarea data-key="notes" data-index="{i}"></textarea></label></article>''')
    payload = {"review_id": manifest["review_id"], "decisions": decisions}
    _write_json(root / "review-template.json", payload)
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Readable handwriting review</title>
<style>body{font:17px system-ui;max-width:980px;margin:32px auto;padding:20px;background:#f4f6fa;color:#18273e}article{padding:25px;background:white;border-radius:12px;margin:24px 0}img{display:block;max-width:100%;max-height:280px;margin:20px 0;object-fit:contain;object-position:left}label{display:block;margin:16px 0}select,input,textarea{display:block;padding:9px;font:inherit;width:95%;margin-top:8px}.hint{color:#53627a;font-size:14px}.reference{font-size:24px}button{padding:14px;font:inherit;cursor:pointer;position:sticky;top:8px}</style>
<h1>Is this handwriting genuinely readable?</h1><p>Assess the image before seeing its reference. Judge ordinary cursive fairly; do not reject merely unfamiliar words. Reject ambiguous, masked, clipped or unreadable crops. Pixel count alone does not establish readability.</p>
<p>Only clearly readable images with confirmed or explicitly corrected references will enter the approved test. Models have not been run on this selection. Download your decisions before closing; edits are not saved automatically.</p>
<button id="export">Download readability decisions</button>"""
    page += "".join(cards) + f'<script id="data" type="application/json">{encoded}</script>'
    page += """<script>document.getElementById('export').onclick=()=>{const d=JSON.parse(document.getElementById('data').textContent);document.querySelectorAll('[data-key]').forEach(e=>d.decisions[Number(e.dataset.index)][e.dataset.key]=e.value);const u=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='readability-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};</script></html>"""
    (root / "review.html").write_text(page, encoding="utf-8")


def approve(review_dir, decisions_path, output):
    root = Path(review_dir).resolve()
    output = Path(output).expanduser().resolve()
    manifest = json.loads((root / "review.json").read_text())
    if manifest["review_id"] != _digest({k: v for k, v in manifest.items() if k != "review_id"}):
        raise ValueError("Review manifest changed")
    payload = json.loads(Path(decisions_path).read_text())
    if payload.get("review_id") != manifest["review_id"]:
        raise ValueError("Decisions belong to a different review")
    rows = payload.get("decisions", [])
    decisions = {row["id"]: row for row in rows}
    if len(decisions) != len(rows) or set(decisions) != {s["id"] for s in manifest["samples"]}:
        raise ValueError("Decisions must cover every candidate exactly once")
    snapshot = load(root / "candidates")
    if snapshot["benchmark_fingerprint"] != manifest["candidate_fingerprint"]:
        raise ValueError("Candidate evidence changed")
    selected, excluded = [], []
    for sample in snapshot["samples"]:
        row = decisions[sample["id"]]
        if row.get("sha256") != sample["hashes"]["crop"]:
            raise ValueError("Decision image identity mismatch")
        if row.get("readability") not in {
            "unreviewed",
            "clearly_readable",
            "ambiguous",
            "unreadable",
        } or row.get("reference_status") not in {
            "unreviewed",
            "confirmed",
            "corrected",
            "uncertain",
        }:
            raise ValueError("Unknown review decision")
        transcription = row.get("transcription")
        if not isinstance(transcription, str):
            raise ValueError("Transcription must be text")
        if row["readability"] == "clearly_readable" and row["reference_status"] in {
            "confirmed",
            "corrected",
        }:
            if not transcription.strip():
                raise ValueError("Approved images require your independently typed transcription")
            if (
                row["reference_status"] == "confirmed"
                and transcription.split() != sample["reference"].split()
            ):
                raise ValueError(
                    f"Typed transcription differs from the confirmed reference for {sample['id']}; use corrected or uncertain"
                )
            selected.append(sample)
        else:
            excluded.append(
                {
                    "id": sample["id"],
                    "readability": row["readability"],
                    "reference_status": row["reference_status"],
                }
            )
    if not selected:
        raise ValueError("No explicitly approved images")
    with tempfile.TemporaryDirectory(prefix="vlm-approved-readable-") as tmp:
        paired = Path(tmp) / "paired"
        _paired_source(selected, paired, reviewed=decisions)
        approved = freeze(paired, output, layout="paired", split="test")
    # Keep the human approval and exclusion record alongside the frozen snapshot.
    _write_json(
        Path(output) / "human-approval.json",
        {
            "review_id": manifest["review_id"],
            "decisions_sha256": hashlib.sha256(Path(decisions_path).read_bytes()).hexdigest(),
            "approved": approved["sample_ids"],
            "excluded": excluded,
            "reserved_test_documents": sorted({s["source_document"] for s in approved["samples"]}),
            "training_overlap": "Unknown for pretrained IAM checkpoints; do not claim an unseen IAM test",
        },
    )
    shutil.copyfile(decisions_path, Path(output) / "readability-decisions.json")
    return approved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--layout", choices=["iam-words", "iam-lines"], default="iam-words")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--min-pixels", type=int, default=10000)
    p.add_argument("--exclude-runs", type=Path, default=Path("runs"))
    p = sub.add_parser("approve")
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            prepare(
                args.data,
                args.output,
                layout=args.layout,
                count=args.count,
                min_pixels=args.min_pixels,
                exclude_runs=args.exclude_runs,
            )
        )
    else:
        approved = approve(args.review, args.decisions, args.output)
        print(f"Approved {len(approved['samples'])} images: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
