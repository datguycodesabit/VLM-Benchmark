# IAM line test setup

Two independent full test protocols are available. Neither uses the 100-word
exploratory selection or any pixel/readability filter.

| Configuration | Membership | Purpose |
|---|---|---|
| `examples/trocr-iam-aachen-lines.toml` | 2,915 lines | Aachen partition cited by TrOCR |
| `examples/trocr-iam-official-lines.toml` | 1,861 lines | IAM large writer-independent task |

Both use `microsoft/trocr-large-handwritten` pinned to
`e68501f437cd2587ae5d68ee457964cac824ddee`, deterministic ten-beam generation,
256 new tokens, and RGB input resized directly to 384×384 using PIL bicubic via
the checkpoint processor, with its 0.5 mean/std normalization. There is no
contrast enhancement, padding, deskewing, or aspect-preserving resize. The
existing word beam-10 configuration intentionally retains checkpoint resizing
and its original 32-token budget to isolate decoding.

## Provenance

- [IAM official archive and task](https://zenodo.org/records/19003721): source
  images, `lines.txt`, and official split ZIP. The published MD5 checksums are
  enforced before preparation; SHA-256 is recorded in each output audit.
- [Pinned Laia Aachen split](https://github.com/jpuigcerver/Laia/tree/1a4afa40f08f91f33d0d47a1d89b106091ad076a/egs/iam/data/part/lines/aachen):
  the partition cited by the [TrOCR paper](https://arxiv.org/html/2109.10282v5).
- `protocols/iam/` vendors all split IDs, with source URLs and SHA-256 checksums
  in `sources.json`. Preparation rejects changed lists, duplicate IDs,
  test/train or test/validation overlap, wrong counts, missing references,
  missing/duplicate images, corrupt images, and archive checksum mismatches.

References are the complete ninth field of IAM `lines.txt`, replacing word
separator pipes with spaces, retaining case and punctuation. Both `ok` and
`err` rows are retained: the IAM table explicitly says these flags describe
word segmentation and need not affect the whole line. Using the existing
`iam-lines` auto-selector would silently drop `err` rows, so these protocols use
a strict paired-data preparation path instead.

## Prepare or reproduce the setup

Run from the repository root. Dataset files stay in ignored `data/`; no IAM
images or transcriptions are committed. IAM permits non-commercial research;
see the source terms and cite Marti and Bunke (2002).

```sh
mkdir -p data/iam-official/downloads
curl -fL --retry 3 'https://zenodo.org/records/19003721/files/lines.tar?download=1' \
  -o data/iam-official/downloads/lines.tar
curl -fL --retry 3 'https://zenodo.org/records/19003721/files/ascii.tar?download=1' \
  -o data/iam-official/downloads/ascii.tar

.venv/bin/python -m vlm_bench.iam_protocol --protocol aachen
.venv/bin/python -m vlm_bench.iam_protocol --protocol official

.venv/bin/python -m vlm_bench run --config examples/trocr-iam-aachen-lines.toml \
  --output runs/trocr-iam-aachen-lines --dry-run
.venv/bin/python -m vlm_bench run --config examples/trocr-iam-official-lines.toml \
  --output runs/trocr-iam-official-lines --dry-run
```

Preparation refuses to overwrite an existing protocol directory. Each contains
`source/`, a portable, integrity-checked `prepared/` snapshot, and `protocol.json`
with the split hash, data hashes, count, and benchmark fingerprint. Retain the
snapshot for future model comparisons. An interrupted source preparation may
leave an incomplete protocol directory; move it aside before retrying.

## Run inference

After preparation, remove `--dry-run` from either command above. A full ten-beam
line run is substantially more work than the 100-word test. Setup itself does
not run inference. The two runs are separate experiments and their scores must
remain separately labeled. Use the normal run report/inspection to check all
samples completed and inspect every truncation flag; rerun with a larger budget
if any line reaches the cap without EOS. A 256-token budget is a starting
configuration, not evidence that no truncation can occur.

The primary accuracy measure is case-sensitive, corpus-weighted CER:
`100 × sum(character edits) / sum(reference characters)`, preserving punctuation
and spaces after the benchmark's documented whitespace collapsing. WER and the
project's composite score are supplementary; the composite is not a paper metric.

## Limits on the published 2.89% comparison

These are complete pinned line protocols using the Hugging Face adapter, not a
claim that the original fairseq experiment has been reproduced. On November 2,
2026 Microsoft's packaged IAM URL returned HTTP 409; its README's signed-URL
fallback returned HTTP 403. Thus we cannot confirm its packaged images and
reference strings against these original IAM sources byte for byte.

The [official evaluation README](https://github.com/microsoft/unilm/tree/master/trocr#evaluation-on-iam)
specifies `cer2`, but the upstream `trocr/scoring.py` currently registers `cer`
and does not define `cer2`. Exact normalization/scorer parity is unresolved.
HF and fairseq beam-search defaults, converted weights, precision and runtime
also need parity checks before attributing any remaining gap to model quality.
The 2.89% value is a published reference for Aachen, never a pass threshold or
an expected score for the 1,861-line protocol.

`strict_research=false` is intentional: this repository's strict research mode
requires independently verified reference/writer metadata not supplied by the
ASCII line table. Partition identity, membership and hashes are enforced by the
separate preparation checks; no manual-verification labels are fabricated.

## Completed local validation — November 2, 2026

Both archives passed the official MD5 checks. Both frozen snapshots passed the
snapshot loader's image/manifest integrity checks, and both CLI dry-runs
resolved the pinned checkpoint on MPS with all selected samples eligible and
`inference=false`. Dry-run JSON is saved beside each `protocol.json`.

| Protocol | Prepared / eligible lines | `err` rows retained | Reference characters |
|---|---:|---:|---:|
| Aachen | 2,915 / 2,915 | 577 | 125,616 |
| IAM official | 1,861 / 1,861 | 381 | 81,676 |

Snapshot fingerprints:

- Aachen: `1a8c3a40f47c7b2f93ab61c5cdc07f2b1196b65ba5e23e3ea34c0524de57c1b5`
- IAM official: `cfa1f6ae435b091e8ec149a1d89b66e5e1f82b2bc5a116f6a0c43c0496c6e1e6`

The bicubic processor tensor matched a direct PIL bicubic + 0.5/0.5
normalization calculation exactly (maximum absolute difference 0.0). The full
repository test suite passed: 497 passed, 1 skipped. Ruff and diff whitespace
checks passed. No model inference or accuracy measurement was performed.
