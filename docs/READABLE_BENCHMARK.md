# Readable handwriting benchmark

The goal is to measure recognition on human-approved, readable handwriting,
then investigate adaptation to one target writer. Word and line tasks must have
separate scores. Larger image dimensions are a screening rule, not proof of
readability.

## Current files and limits

- `data/readable-iam-words-review/review.html`: 50 fresh word candidates, with
  native area at least 10,000 pixels and no exact IDs found in previous results.
  The cutoff is an explicit choice for this readable subset, not a general IAM
  quality standard. Candidate sampling uses seed 20261102 and image IDs, never
  prediction correctness.
- No IAM line-image release or `lines.txt` was found in Downloads, Documents,
  Desktop, or the local IAM word archive. The line task remains pending genuine
  line images and references. Do not synthesize sentences by joining word crops.
- No target-writer training dataset has been identified. A separately found
  `Dr._Brothers_Exams.zip` contains 57 PDFs, but their contents and suitability
  have not been assessed.

## Review before inference

Open the review HTML. For each image:

1. Mark it clearly readable, ambiguous, or unreadable without sentence context.
2. Type the transcription before opening the reference disclosure.
3. Confirm the IAM reference, explicitly correct it using your typed text, or
   leave it uncertain. Add notes about clipping, masking or doubtful characters.
4. Download `readability-decisions.json` before closing. Browser edits are not
   saved automatically.

The sheet contains no model outputs. Existing source data and historical scores
are preserved. Review labels are annotations, not executable instructions.

Import approved decisions into a new frozen benchmark:

```bash
uv run python scripts/readability_review.py approve \
  --review data/readable-iam-words-review \
  --decisions /path/to/readability-decisions.json \
  --output data/readable-iam-words-test
```

Only clearly readable images with a confirmed or corrected reference are admitted.
Approval requires a typed transcription; a disagreement with a supposedly
confirmed reference must be resolved rather than silently accepted. The importer
checks the review identity, all candidate IDs, image hashes and frozen snapshot
identity. References are changed only in a new derived snapshot after explicit
correction. `human-approval.json` retains exclusions, the decisions hash and the
list of reserved test documents. The original decisions are copied alongside it.

## Compare models fairly

Run TrOCR, Qwen and the subscription model against the exact approved snapshot,
using saved prompts and controls and separate output directories. Report both
strict fidelity and separately labeled case-insensitive diagnostics; missing,
extra and substituted letters remain errors. Review correctness using the image,
not the model's own explanation. Retain rejection/coverage counts, timing, failure
counts, exact-match rate, CER and WER. This is a selected readable subset, not an
estimate of performance on all IAM handwriting.

When IAM lines become available, use the same review script with `--layout
iam-lines`, a separate review directory and separate output snapshot. An
improvement on different line images does not alone prove a context effect;
that claim needs matched word/line examples from the same source documents.

## Writer-specific training later

Collect multiple documents from the intended writer, preserve original scans and
create verified references. Assign entire documents to training, validation and
test sets before tuning; words and lines from the same page/document must stay
in the same split. Do not train on the approved test documents listed above.
Use validation data to choose preprocessing, prompts and training settings.
Evaluate the original and adapted checkpoints on identical untouched writer-test
images; compare accuracy, correction time and compute cost.

IAM-trained checkpoints may already have seen IAM samples. A fresh review or
new local split does not establish absence of pretrained-model leakage. Keep
that uncertainty explicit and use newly collected target-writer test documents
for the eventual adaptation claim. This workflow prepares evaluation data; it
does not implement a training pipeline or claim training has occurred.
