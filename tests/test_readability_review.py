import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

from vlm_bench.snapshot import load

api = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/readability_review.py"))


@pytest.fixture
def review(tmp_path):
    data = tmp_path / "iam"
    (data / "images").mkdir(parents=True)
    annotation = []
    for i in range(4):
        sid = f"a01-00{i}-00-00"
        Image.new("RGB", (150, 100), "white").save(data / "images" / f"{sid}.png")
        annotation.append(f"{sid} ok 154 1 0 0 150 100 NN hello")
    (data / "words.txt").write_text("\n".join(annotation))
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "results.jsonl").write_text(
        json.dumps({"sample_id": "a01-000-00-00", "prediction": "do not show this"}) + "\n"
    )
    output = tmp_path / "review"
    api["prepare"](data, output, count=2, exclude_runs=runs)
    payload = json.loads((output / "review-template.json").read_text())
    return output, payload


def test_blind_review_excludes_measured_ids_and_leaves_references_unverified(review):
    root, payload = review
    assert len(payload["decisions"]) == 2
    assert "a01-000-00-00" not in {r["id"] for r in payload["decisions"]}
    page = (root / "review.html").read_text()
    assert "do not show this" not in page
    assert "<details>" in page
    assert all(
        s["verification_status"] == "unresolved" for s in load(root / "candidates")["samples"]
    )


def test_approval_requires_readability_and_verified_reference_and_reserves_documents(
    review, tmp_path
):
    root, payload = review
    first, second = payload["decisions"]
    first.update(
        readability="clearly_readable", reference_status="confirmed", transcription="hello"
    )
    second.update(readability="ambiguous", reference_status="uncertain")
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(payload))
    output = tmp_path / "approved"
    approved = api["approve"](root, decisions, output)
    assert approved["sample_ids"] == [first["id"]]
    assert approved["samples"][0]["split"] == "test"
    assert approved["samples"][0]["verification_status"] == "verified"
    audit = json.loads((output / "human-approval.json").read_text())
    assert audit["reserved_test_documents"]
    assert audit["excluded"][0]["id"] == second["id"]


def test_disagreement_cannot_silently_confirm_bad_reference(review, tmp_path):
    root, payload = review
    payload["decisions"][0].update(
        readability="clearly_readable", reference_status="confirmed", transcription="different"
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Typed transcription differs"):
        api["approve"](root, decisions, tmp_path / "approved")
    assert not (tmp_path / "approved").exists()


def test_corrected_reference_and_image_hash_are_checked(review, tmp_path):
    root, payload = review
    row = payload["decisions"][0]
    row.update(
        readability="clearly_readable", reference_status="corrected", transcription="corrected word"
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(payload))
    approved = api["approve"](root, decisions, tmp_path / "approved")
    assert approved["samples"][0]["reference"] == "corrected word"
    row["sha256"] = "0" * 64
    decisions.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity mismatch"):
        api["approve"](root, decisions, tmp_path / "other")
