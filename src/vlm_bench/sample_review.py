"""Model-independent, explicitly completed human crop reviews for offline scoring."""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

_CODES = {"severe_fragmentation", "reference_mismatch", "unreadable", "other"}


def reference_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_review(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require complete decisions; never infer human approval from an automated flag."""
    if not isinstance(value, Mapping):
        raise ValueError("sample review must be a JSON object")
    allowed = {"schema_version", "benchmark_fingerprint", "reviewer", "reviewed_at", "samples"}
    if (
        set(value) != allowed
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("sample review requires schema_version=1 and the documented fields")
    for key in ("reviewer", "reviewed_at"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"sample review requires a completed {key}")
    try:
        reviewed_at = datetime.fromisoformat(value["reviewed_at"])
        if reviewed_at.tzinfo is None:
            raise ValueError("missing timezone")
    except ValueError as exc:
        raise ValueError(
            "sample review reviewed_at must be an ISO timestamp with timezone"
        ) from exc
    if not isinstance(value["benchmark_fingerprint"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["benchmark_fingerprint"]
    ):
        raise ValueError("sample review requires a benchmark fingerprint")
    if not isinstance(value["samples"], list) or not value["samples"]:
        raise ValueError("sample review requires sample decisions")
    seen = set()
    for row in value["samples"]:
        fields = {
            "sample_id",
            "crop_sha256",
            "reference_sha256",
            "decision",
            "reason_code",
            "reason",
        }
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("invalid sample review decision fields")
        sid = row["sample_id"]
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError("duplicate or invalid sample review ID")
        seen.add(sid)
        for key in ("crop_sha256", "reference_sha256"):
            if not isinstance(row[key], str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError(f"sample review requires valid {key}")
        if row["decision"] not in {"keep", "exclude"}:
            raise ValueError(f"sample review is unfinished: {sid}; choose keep or exclude")
        if not isinstance(row["reason"], str) or not isinstance(row["reason_code"], str):
            raise ValueError("sample review reasons must be strings")
        if row["decision"] == "exclude" and (
            row["reason_code"] not in _CODES or not row["reason"].strip()
        ):
            raise ValueError(f"excluded sample {sid} requires a reason code and explanation")
        if row["decision"] == "keep" and (row["reason_code"] or row["reason"]):
            raise ValueError(f"kept sample {sid} must have empty exclusion reasons")
    clean = json.loads(json.dumps(value))
    clean["samples"].sort(key=lambda row: row["sample_id"])
    return clean


def review_exclusions(review, samples, benchmark_fingerprint):
    """Bind every decision to the exact roster, image, and original reference."""
    review = validate_review(review)
    if benchmark_fingerprint != review["benchmark_fingerprint"]:
        raise ValueError("sample review benchmark fingerprint does not match the run")
    decisions = {row["sample_id"]: row for row in review["samples"]}
    if set(decisions) != set(samples):
        raise ValueError("sample review roster does not match the frozen sample roster")
    excluded = {}
    for sid, sample in samples.items():
        decision = decisions[sid]
        if (sample.get("hashes") or {}).get("crop") != decision["crop_sha256"]:
            raise ValueError(f"sample review crop hash mismatch: {sid}")
        if reference_hash(sample["reference"]) != decision["reference_sha256"]:
            raise ValueError(f"sample review reference hash mismatch: {sid}")
        if decision["decision"] == "exclude":
            excluded[sid] = f"human_review:{decision['reason_code']}: {decision['reason']}"
    return excluded


def create_review(prepared: Path, output: Path) -> Path:
    """Create a blind image/reference gallery and pending review file; no predictions."""
    from .snapshot import load

    snapshot = load(prepared)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Review output already exists: {output}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    review = {
        "schema_version": 1,
        "benchmark_fingerprint": snapshot["benchmark_fingerprint"],
        "reviewer": "",
        "reviewed_at": "",
        "samples": [],
    }
    articles = []
    for i, sample in enumerate(snapshot["samples"]):
        crop = Path(sample["crop_path"])
        name = f"{i:06d}{crop.suffix.lower()}"
        shutil.copyfile(crop, output / "images" / name)
        review["samples"].append(
            {
                "sample_id": sample["id"],
                "crop_sha256": sample["hashes"]["crop"],
                "reference_sha256": reference_hash(sample["reference"]),
                "decision": "pending",
                "reason_code": "",
                "reason": "",
            }
        )
        articles.append(f'''<article data-index="{i}"><h2>{html.escape(sample["id"])}</h2>
<img loading="lazy" src="images/{name}"><pre>Reference: {html.escape(sample["reference"])}</pre>
<label>Decision <select class="decision"><option>pending</option><option>keep</option><option>exclude</option></select></label>
<label>Reason <select class="code"><option value="">Choose for exclusion</option>{"".join(f"<option>{code}</option>" for code in sorted(_CODES))}</select></label>
<input class="reason" placeholder="Explain exclusion"></article>''')
    (output / "review.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    # Escape script delimiters even if a dataset's labels/IDs contain hostile markup.
    data = (
        json.dumps(review).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    page = (
        """<!doctype html><meta charset="utf-8"><title>Human crop review</title>
<style>body{font:16px system-ui;max-width:1600px;margin:30px auto;padding:20px}article{border:1px solid #aaa;padding:18px;margin:20px 0}img{max-width:100%;height:auto}pre{white-space:pre-wrap}label,input,button{margin:8px;padding:8px}input.reason{width:60%}header{position:sticky;top:0;background:white;padding:12px;border-bottom:2px solid #444}</style>
<h1>Human crop review</h1><p>Judge image quality and reference alignment, independently of model success. Keep hard but readable handwriting and ordinary crop seams. Exclude severe fragmentation, missing content, or invalid image/reference pairs with a reason. Predictions are deliberately absent. Decisions stay in this page until exported; export before closing.</p>
<header><label>Reviewer <input id="reviewer"></label><button id="export">Export completed review.json</button><span id="message"></span></header>
"""
        + "\n".join(articles)
        + f"""<script>const review={data};
document.getElementById('export').onclick=()=>{{
review.reviewer=document.getElementById('reviewer').value.trim();
for(const article of document.querySelectorAll('article')){{
const row=review.samples[Number(article.dataset.index)];
row.decision=article.querySelector('.decision').value;
row.reason_code=row.decision==='exclude'?article.querySelector('.code').value:'';
row.reason=row.decision==='exclude'?article.querySelector('.reason').value.trim():'';
}}
if(!review.reviewer||review.samples.some(r=>r.decision==='pending'||(r.decision==='exclude'&&(!r.reason_code||!r.reason)))){{document.getElementById('message').textContent='Enter your name, decide every image, and explain exclusions.';return;}}
review.reviewed_at=new Date().toISOString();
const url=URL.createObjectURL(new Blob([JSON.stringify(review,null,2)+'\\n'],{{type:'application/json'}}));
const a=document.createElement('a');a.href=url;a.download='review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
document.getElementById('message').textContent='Exported. Use the downloaded review.json with --review.';
}};</script>"""
    )
    (output / "review.html").write_text(page, encoding="utf-8")
    return output / "review.html"
