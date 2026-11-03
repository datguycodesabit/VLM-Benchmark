"""Strict preparation of pinned IAM line protocols, without segmentation filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

from .snapshot import freeze

LINE_ID = re.compile(r"[a-z]\d{2}-\d+[a-z]?-\d{2}")
COUNTS = {"aachen": 2915, "official": 1861}
ARCHIVES = {
    "lines.tar": "c17111f4c5b685958a1d02e372b377b5",
    "ascii.tar": "67eb823a5204e5924481e15089adbbdd",
}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = path.read_text(encoding="utf-8").split()
    if not ids or len(ids) != len(set(ids)) or any(not LINE_ID.fullmatch(i) for i in ids):
        raise ValueError(f"Empty, duplicate, or invalid line IDs: {path}")
    return ids


def read_references(text: str) -> dict[str, str]:
    """Keep err rows: IAM says this status concerns word segmentation, not lines."""
    references = {}
    for row in text.splitlines():
        if not row.strip() or row.lstrip().startswith("#"):
            continue
        fields = row.split(maxsplit=8)
        if len(fields) != 9 or not LINE_ID.fullmatch(fields[0]):
            raise ValueError(f"Malformed IAM annotation: {row[:100]}")
        sample_id = fields[0]
        reference = fields[8].replace("|", " ")
        if sample_id in references or not reference.strip():
            raise ValueError(f"Duplicate or empty IAM reference: {sample_id}")
        references[sample_id] = reference
    return references


def prepare_protocol(downloads: Path, protocols: Path, output: Path, protocol: str) -> dict:
    """Verify archive checksums and exact membership before publishing a snapshot."""
    expected = COUNTS[protocol]
    partition = protocols / protocol
    lock = json.loads((protocols / "sources.json").read_text(encoding="utf-8"))
    for name, expected_hash in lock["files"].items():
        if name.startswith(protocol + "/") and file_hash(protocols / name) != expected_hash:
            raise ValueError(f"Pinned partition checksum mismatch: {name}")
    ids = read_ids(partition / "test.txt")
    if len(ids) != expected:
        raise ValueError(f"{protocol} requires {expected} IDs, got {len(ids)}")
    for other in partition.glob("*.txt"):
        if other.name != "test.txt" and set(ids) & set(read_ids(other)):
            raise ValueError(f"Test overlap with {other}")
    for name, checksum in ARCHIVES.items():
        if file_hash(downloads / name, "md5") != checksum:
            raise ValueError(f"Official IAM checksum mismatch: {name}")
    target = output / protocol
    if target.exists():
        raise ValueError(f"Refusing to overwrite existing protocol: {target}")
    with tarfile.open(downloads / "ascii.tar") as archive:
        tables = [m for m in archive if Path(m.name).name == "lines.txt" and m.isfile()]
        if len(tables) != 1:
            raise ValueError("Expected exactly one lines.txt")
        references = read_references(archive.extractfile(tables[0]).read().decode("utf-8"))
    missing = set(ids) - references.keys()
    if missing:
        raise ValueError(f"Missing references: {sorted(missing)[:5]}")
    # Do not extract archive paths: copy only verified IDs to controlled filenames.
    members = {}
    with tarfile.open(downloads / "lines.tar") as archive:
        for member in archive:
            sample_id = Path(member.name).stem
            if member.isfile() and member.name.endswith(".png") and sample_id in ids:
                if sample_id in members:
                    raise ValueError(f"Duplicate image in IAM archive: {sample_id}")
                members[sample_id] = member
        if set(members) != set(ids):
            raise ValueError(f"Missing images: {sorted(set(ids) - members.keys())[:5]}")
        raw = target / "source"
        (raw / "images").mkdir(parents=True)
        (raw / "references").mkdir()
        metadata = []
        for sample_id in ids:
            (raw / "images" / f"{sample_id}.png").write_bytes(
                archive.extractfile(members[sample_id]).read()
            )
            (raw / "references" / f"{sample_id}.txt").write_text(
                references[sample_id], encoding="utf-8"
            )
            metadata.append(
                {
                    "id": sample_id,
                    "split": "test",
                    "sample_type": "line",
                    "content_type": "prose",
                    "source_document": sample_id.rsplit("-", 1)[0],
                }
            )
    (raw / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )
    snapshot = freeze(raw, target / "prepared", layout="paired", preprocess="original")
    samples = snapshot["samples"]
    if {s["id"] for s in samples} != set(ids) or len(samples) != expected:
        raise ValueError("Prepared membership differs from pinned test IDs")
    audit = {
        "protocol": protocol,
        "count": expected,
        "benchmark_fingerprint": snapshot["benchmark_fingerprint"],
        "test_ids_sha256": file_hash(partition / "test.txt"),
        "archives": {
            name: {"md5": checksum, "sha256": file_hash(downloads / name)}
            for name, checksum in ARCHIVES.items()
        },
        "reference_policy": "IAM lines.txt field 9; pipes replaced by spaces; keep ok and err rows",
        "selection": "all pinned test IDs; no sampling, readability or segmentation filtering",
        "published_score_reproduction": False,
        "limitations": [
            "Microsoft packaged references unavailable; cer2 parity unverified",
            "Hugging Face backend differs from original fairseq runtime",
        ],
    }
    (target / "protocol.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("data/iam-official/downloads"))
    parser.add_argument("--protocols", type=Path, default=Path("protocols/iam"))
    parser.add_argument("--output", type=Path, default=Path("data/iam-official"))
    parser.add_argument("--protocol", choices=COUNTS, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_protocol(args.downloads, args.protocols, args.output, args.protocol), indent=2
        )
    )


if __name__ == "__main__":
    main()
