import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from vlm_bench import iam_protocol
from vlm_bench.snapshot import load


def test_pinned_partitions_have_expected_counts_and_no_overlap():
    root = Path(__file__).resolve().parents[1] / "protocols/iam"
    for protocol, count in iam_protocol.COUNTS.items():
        test = set(iam_protocol.read_ids(root / protocol / "test.txt"))
        assert len(test) == count
        for path in (root / protocol).glob("*.txt"):
            if path.name != "test.txt":
                assert not test.intersection(iam_protocol.read_ids(path))


def make_archive(path, files):
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_preparation_keeps_err_lines_and_freezes_exact_ids(tmp_path, monkeypatch):
    monkeypatch.setitem(iam_protocol.COUNTS, "aachen", 2)
    partition = tmp_path / "part/aachen"
    partition.mkdir(parents=True)
    (partition / "test.txt").write_text("a01-000u-00\na01-000u-01\n")
    (partition / "train.txt").write_text("b01-001-00\n")
    (partition.parent / "sources.json").write_text(
        json.dumps(
            {
                "files": {
                    f"aachen/{f.name}": iam_protocol.file_hash(f) for f in partition.glob("*.txt")
                }
            }
        )
    )
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    image = io.BytesIO()
    Image.new("RGB", (64, 12), "white").save(image, format="PNG")
    make_archive(
        downloads / "lines.tar",
        {f"a01/a01-000u/a01-000u-{n:02d}.png": image.getvalue() for n in range(2)},
    )
    make_archive(
        downloads / "ascii.tar",
        {
            "lines.txt": (
                "a01-000u-00 ok 1 2 3 4 5 6 A|Word.\na01-000u-01 err 1 2 3 4 5 6 Other|Text\n"
            ).encode()
        },
    )
    for name in iam_protocol.ARCHIVES:
        monkeypatch.setitem(
            iam_protocol.ARCHIVES, name, iam_protocol.file_hash(downloads / name, "md5")
        )
    audit = iam_protocol.prepare_protocol(downloads, partition.parent, tmp_path / "out", "aachen")
    snapshot = load(tmp_path / "out/aachen/prepared")
    assert audit["count"] == 2
    assert [s["reference"] for s in snapshot["samples"]] == ["A Word.", "Other Text"]
    assert all(s["metadata"]["sample_type"] == "line" for s in snapshot["samples"])
    assert json.loads((tmp_path / "out/aachen/protocol.json").read_text()) == audit
    with pytest.raises(ValueError, match="overwrite"):
        iam_protocol.prepare_protocol(downloads, partition.parent, tmp_path / "out", "aachen")
    (downloads / "lines.tar").write_bytes(b"bad archive")
    with pytest.raises(ValueError, match="checksum mismatch"):
        iam_protocol.prepare_protocol(downloads, partition.parent, tmp_path / "new", "aachen")


@pytest.mark.parametrize("ids", ["", "../a\n", "a01-000u-00\na01-000u-00\n"])
def test_invalid_ids_are_rejected(tmp_path, ids):
    path = tmp_path / "ids"
    path.write_text(ids)
    with pytest.raises(ValueError):
        iam_protocol.read_ids(path)
