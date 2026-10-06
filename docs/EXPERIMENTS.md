# Experiment setup

This guide covers a reproducible comparison of handwriting recognition models
using a paste-in image and transcription folder. Start with prose lines. Treat
equations as a separate task because character error rate measures literal
transcription and cannot establish mathematical equivalence.

## 1. Install the backends you plan to use

For the full set of providers, install the development, cloud, and TrOCR
extras:

```bash
uv sync --extra dev --extra cloud --extra trocr --frozen
```

Ollama and TrOCR can run locally. ChatGPT subscription and OpenAI API requests
send each image to their respective cloud provider. TrOCR downloads its model
the first time it is used; on a supported Mac it uses MPS, otherwise it records
the CPU fallback.

Select providers explicitly when listing models:

```bash
uv run vlm-bench models --provider ollama
uv run vlm-bench models --provider trocr
uv run vlm-bench auth login --provider chatgpt
uv run vlm-bench auth status
uv run vlm-bench models --provider chatgpt
```

Sign in through the official ChatGPT authentication flow. Available models
depend on account eligibility. API access is separate: configure
`OPENAI_API_KEY`, then use `openai:MODEL_ID`. ChatGPT subscription requests do
not silently fall back to paid API requests. Use
`uv run vlm-bench auth logout --provider chatgpt` to remove saved sign-in
credentials.

Model selectors use the prefix `ollama:`, `trocr:`, `chatgpt:`, or `openai:`.
For example:

```text
ollama:qwen2.5vl:3b
trocr:microsoft/trocr-base-handwritten
chatgpt:MODEL_ID_FROM_MODELS
openai:MODEL_ID
```

Unprefixed Ollama names remain supported for existing runs and scripts.

## 2. Organize and verify the images

For ordinary paired data, create one image and one UTF-8 text file per sample.
Keep matching relative paths and filename stems:

```text
data/brothers/
  images/
    exam2006/page03-line02.png
    exam2008/page01-line04.jpg
  text/
    exam2006/page03-line02.txt
    exam2008/page01-line04.txt
  metadata.jsonl
```

The `text/` directory may be named `references/`. Supported images are PNG and
JPEG. Each reference file contains the verified transcription for its image.
Keep the source images unchanged. Select `--preprocess original` to use source
pixels, or `--preprocess enhanced` to test grayscale conversion, contrast
normalization, enlargement, and padding as a separate condition.

The optional `metadata.jsonl` file contains one JSON object per sample. Its ID
must match the image and text filename stem. Common fields are:

```json
{"id":"exam2006/page03-line02","source_document":"exam2006","writer_id":"brothers","content_type":"prose","sample_type":"line","difficulty":"hard","verification_status":"verified"}
```

Use `content_type` values `prose` or `equation`; use `sample_type` values such
as `line`, `word`, or `page`. Record `source_document` for every sample when
you intend to split by document. Transcriptions should be checked by an
independent reader, with uncertain references resolved by Dr. Brothers or
marked `verification_status: "unresolved"`. Unresolved references are omitted
from the primary scored report.

Check the data before running models:

```bash
uv run vlm-bench dataset check --data data/brothers --layout paired --json
```

Resolve missing or duplicate pairs, unreadable images, empty references, and
duplicate content before the comparison. Use `--json` to review sample
dimensions and excluded items in the full audit. Do not quietly drop a missing pair to make a run
complete.

## 3. Split by document and freeze a test set

Training, validation, and test samples from the same exam or source document
would make the evaluation overly optimistic. Add `source_document` metadata
first, then assign whole documents to reproducible 70/15/15 splits. The default
protocol is `document-disjoint`:

```bash
uv run vlm-bench dataset split \
  --data data/brothers \
  --output /tmp/brothers-splits.jsonl \
  --seed 42 --protocol document-disjoint
```

The split command creates new metadata and refuses to overwrite an existing
file. Preserve the original metadata, then install the split metadata for
subsequent preparation and runs:

```bash
cp data/brothers/metadata.jsonl data/brothers/metadata.before-split.jsonl
cp /tmp/brothers-splits.jsonl data/brothers/metadata.jsonl
```

If you have no existing metadata file, copy the generated file into
`data/brothers/metadata.jsonl`. Re-run `dataset check` and inspect the split
counts. The splitter checks that exact duplicate image bytes do not cross
documents. For a writer-held-out study, require `writer_id` on every sample and
select the writer-disjoint protocol:

```bash
uv run vlm-bench dataset split --data data/brothers \
  --output /tmp/brothers-writer-splits.jsonl \
  --seed 42 --protocol writer-disjoint
```

Writer-disjoint splitting also preserves each source document as one unit. It
joins documents connected through shared writers, including pages containing
multiple writers, so those connected groups cannot cross a split. It fails
visibly when a writer ID or source document is missing. Do not change the test
set after looking at model results.

The shipped IAM archive is useful for public-data smoke tests. Use its
documented evaluation partitions where available. The standard handwritten
TrOCR checkpoint was trained on IAM, so a random IAM sample is not an
independent test of that checkpoint. Keep the public-data and Dr. Brothers
results distinct.

## 4. Freeze the benchmark once

Prepare a portable snapshot after checking references and installing the
document split. The snapshot freezes the selected images, references, metadata,
preprocessing profile, sample IDs, and their hashes:

```bash
uv run vlm-bench prepare --data data/brothers \
  --output data/brothers/prepared-test \
  --layout paired --preprocess original --split test \
  --content-type prose --limit 100 --seed 42
```

The output directory must not already exist. Prepared snapshots use manifest
version 2 and include the inputs needed to run elsewhere; keep the complete
folder together and treat it as immutable. To make another selection or
preprocessing condition, prepare a different snapshot directory. Check the
saved snapshot before inference:

```bash
uv run vlm-bench dataset check --prepared data/brothers/prepared-test
```

This check reports the snapshot version, sample count, and benchmark
fingerprint. The fingerprint identifies the frozen benchmark and is carried
into each run so separate model runs can be checked against the same inputs.
For source datasets, the audit also reports document/writer split overlaps and
perceptually similar images. Split overlaps are integrity findings; perceptual
matches are review suggestions, not automatic proof of leakage.

## 5. Preview and run each model on the snapshot

Start with a dry run. It validates the saved snapshot and selected models,
reports the benchmark fingerprint, sample and verification-status counts,
model eligibility, requested and effective controls, and confirms that no
inference will be sent:

```bash
uv run vlm-bench run --prepared data/brothers/prepared-test \
  --models ollama:qwen2.5vl:3b --dry-run
```

Run each model against that same snapshot, recording the run directory printed
by each command:

```bash
uv run vlm-bench run --prepared data/brothers/prepared-test \
  --models ollama:qwen2.5vl:3b

uv run vlm-bench run --prepared data/brothers/prepared-test \
  --models trocr:microsoft/trocr-base-handwritten
```

Use explicit selectors for subscription or API models. `doctor` checks
provider setup, dependencies, and model availability:

```bash
uv run vlm-bench doctor --provider chatgpt \
  --models chatgpt:MODEL_ID_FROM_MODELS
```

The run manifest records the snapshot fingerprint, frozen sample IDs and
references, model identities, settings, and available provider usage. Dry-run
provenance and verification-status counts make it possible to check that each
model run used the intended benchmark before sending images.

Compare completed runs only when they share the same benchmark fingerprint:

```bash
uv run vlm-bench compare \
  --runs runs/RUN_DIRECTORY_FOR_OLLAMA runs/RUN_DIRECTORY_FOR_TROCR \
  --output comparisons/brothers-test
```

The output directory must be new. It contains `comparison.json`,
`comparison.csv`, `paired.csv`, and `costs.csv`. The report includes coverage
and paired differences. Provider controls can differ; the run manifests record
what each provider accepted.

Resume an interrupted run from its original directory:

```bash
uv run vlm-bench resume --run runs/YOUR_RUN_DIRECTORY
```

Resume continues the saved run on its original fingerprint and predictions.
The optional `--retry-failed` applies only to saved errors; completed
predictions are retained. Subscription usage pauses are resumable.

For a quick one-off check, the source-data path remains available:

```bash
uv run vlm-bench run --data data/brothers \
  --models ollama:qwen2.5vl:3b --layout paired --split test --limit 3 --seed 42 \
  --dry-run
```

## 6. Configuration and cost assumptions

Use the versioned example in [`../examples/experiment.toml`](../examples/experiment.toml)
after creating the prepared snapshot it names:

```bash
uv run vlm-bench run --config examples/experiment.toml --dry-run
```

Paths in TOML are resolved relative to the TOML file, so the example's
`prepared` path points to `data/brothers/prepared-test` from the repository
root. A prepared experiment freezes selection: do not combine `prepared` with
`data`, `layout`, `preprocess`, `preprocessing`, `limit`, `seed`, `split`, or
`content_type` in the same experiment table. An explicit CLI `--data` switches
from a configured prepared snapshot to source data; a CLI `--prepared` selects
a snapshot instead of configured source data. With `--prepared`, source
selection flags such as `--split` or `--preprocess` are rejected.

The config can select models, set per-model options such as TrOCR device or
beam count, and record costs. Provider-inapplicable options are rejected for
selected models. ChatGPT accepts the shared `num_predict` config field for
recording, but the subscription adapter reports it as unsupported. Do not put
credentials in TOML. The OpenAI API key belongs in the process environment;
the ChatGPT sign-in token is stored through the credential store.

Cost projections are estimates, and the program never invents missing prices.
Provide local upfront hardware/training cost and operating cost per sample.
For an API model, either enter cost per sample directly or provide input,
output prices per million tokens; add a cached-input price if the provider
reports cached tokens. Token-based estimates use observed usage for the exact
model selector. If a request lacks usage or cached tokens have no explicit
cached-input price, projected API cost remains unknown.
A ChatGPT subscription allowance is not a per-request price; compare
subscription value separately rather than assigning it an invented call cost.

The included experiment file lists usage volumes and leaves prices blank for
you to supply from a quote, invoice, or measured local energy/hardware costs.
Local total cost is modeled as upfront cost plus operating cost per sample.
The API total is its estimated per-sample cost times volume. The reported
break-even volume is valid only if both cost models have sufficient inputs and
local variable cost is below the API variable cost. Report annotation and
maintenance hours separately; they are not converted to dollars unless you
choose and document an hourly rate.

## 7. Read the research reports

The report provides separate prose and equation tracks; IAM word data appears
as a separate diagnostic. CER and WER use the same verified eligible test
samples for each model. A model receives a rank
only after successful predictions cover the full eligible set; unsupported
tasks and failed or missing predictions remain visible. Paired differences
compare two models on samples both completed. The confidence interval resamples
source documents as groups when document IDs are available, and uses individual
samples otherwise. A negative `right minus left` CER difference means the
right-hand model had lower error in that pair.

Timing summaries use measured model-request latency; cached responses have no
inference latency and are excluded from latency averages and throughput.
`attempts.jsonl` records each retry, its status, duration, and provider retry
delay separately. Throughput is based on successful measured requests and
their recorded request durations. Cold model-load time comes from a successful
warmup when available, or from load timing on sample requests when warmup is
disabled. Missing measurements remain blank. Imported timing is marked
external and is not represented as native model inference timing.

Equation scoring uses Unicode NFC and whitespace normalization, while
preserving operators and LaTeX syntax. It measures literal transcription,
not whether equivalent equations have the same meaning. Do not include
ordinary TrOCR on the equation leaderboard unless a model configuration
explicitly declares equation support.

Accuracy should be considered with latency, failures, coverage, and costs. The
research question is whether specialization provides enough improvement for
students to justify data preparation, fine-tuning, and local operation. A
result showing that another method is more accurate or more economical is also
a useful finding.

## 8. Check research eligibility before inference

Exploratory runs remain the default. Add `--strict-research` when the run
should enforce the research protocol. Every evaluated sample must have a
nonempty reference, an explicit verified state (`verified: true` or a verified
`verification_status`), a test split, and a `source_document`. A
`writer-disjoint` evaluation also requires `writer_id` on every sample.
Conflicting verification fields are treated as unverified. Strict source-data
runs audit the complete dataset before selecting the test samples, so a
document that appears in multiple splits blocks the run. Writer overlap also
blocks writer-disjoint runs.

```bash
uv run vlm-bench run --data data/brothers \
  --models ollama:qwen2.5vl:3b --strict-research \
  --protocol document-disjoint --dry-run
```

For a prepared snapshot, strict validation is limited to the samples included
in that snapshot. It cannot establish that unseen training or validation data
are free of leakage. The manifest and research report retain the protocol,
eligibility settings, audit findings, and the validation boundary. Exact
duplicate images and document/writer split overlaps are reported as integrity
issues; perceptually similar images are review findings that require human
inspection.

## 9. Inspect errors and compare subgroups

Use a completed run to create an HTML error review without another model call.
The default file is `RUN/inspection.html`; use `--output` to choose another
local path.

```bash
uv run vlm-bench inspect --run runs/YOUR_RUN_DIRECTORY --worst 20
uv run vlm-bench inspect --run runs/YOUR_RUN_DIRECTORY --worst 50 \
  --output reviews/brothers.html
```

The terminal summary and HTML page show references, predictions, edit
alignments, insertions, deletions, substitutions, and flags for empty,
truncated, repeated, or commentary-like outputs. The HTML report uses the
frozen local images and escapes displayed text.

Aggregate the same saved results by a metadata field:

```bash
uv run vlm-bench report --run runs/YOUR_RUN_DIRECTORY --group-by writer_id
uv run vlm-bench report --run runs/YOUR_RUN_DIRECTORY --group-by difficulty
uv run vlm-bench report --run runs/YOUR_RUN_DIRECTORY --group-by source_document
uv run vlm-bench report --run runs/YOUR_RUN_DIRECTORY --group-by sample_type
```

Each group includes sample and document counts, eligible coverage, and model
metrics. Missing metadata forms an explicit `unknown` group. These commands
read saved run data and do not perform inference.

## 10. Score formulas and page annotations

Literal equation CER/WER remain the default. Optional rendered similarity
requires the pinned MathText dependency:

```bash
uv sync --extra formula-render
uv run vlm-bench run --prepared data/brothers/prepared-equations \
  --models ollama:qwen2.5vl:3b --formula-rendering
```

The renderer uses Matplotlib MathText, a fixed font/configuration, and aligned
foreground-pixel intersection-over-union. It supports MathText's TeX-like subset, not a full
LaTeX engine; unsupported expressions, oversized input, and rasterization
errors are recorded as metric errors rather than hidden. CER/WER remain
available when rendering fails. Renderer availability and pinned version are
checked before inference.

Optional semantic task annotations are supplied as fields in the sample's
`metadata.jsonl` record and are preserved inside each frozen sample's
`metadata.annotations`:

```json
{
  "id": "exam2006/page03-line02",
  "content_type": "equation",
  "annotations": {
    "critical_expressions": ["x^2", "= 0"],
    "reading_order": [["x^2", "= 0"]]
  }
}
```

`critical_expressions` scores each expression by requiring its normalized
occurrence count in the prediction to match the reference. Each `reading_order`
pair checks that two anchors occur exactly once in the reference in the
declared order, and then checks the same order in the prediction. Missing or
ambiguous reference anchors produce a metric error; missing or ambiguous
prediction anchors score as a mismatch. Annotations are optional, and missing
ones are reported as not annotated. Per-metric scores, errors, and annotation
coverage are reported separately. They are not combined with CER/WER into a
single ranking.

## 11. Import external predictions and OCR baselines

The importer accepts one JSON object per line, without a header. Each row
requires `sample_id` and `prediction`; `status`, `latency_seconds`, and
`usage` are optional. The importer rejects duplicate and unknown IDs. Missing
sample rows are retained as incomplete coverage and cannot receive a full
coverage rank. Imported timing and usage are labeled external.

```bash
uv run vlm-bench import --prepared data/brothers/prepared-test \
  --predictions predictions.jsonl --system tesseract \
  --provenance-file predictions.jsonl.provenance.json \
  --output runs/tesseract
```

Use either `--provenance "description"` or `--provenance-file FILE`; the
provenance file must contain a JSON object. The system name becomes the
selector `external:<system>`, so imported runs can be inspected, exported,
rescored, and compared with native runs on the same snapshot.

The Tesseract example runs an already-installed `tesseract` binary and
records its actual reported version, language, OEM, PSM, preprocessing, and
task scope. It selects PSM 7 for lines, 8 for words, and 6 for pages; use
`--expected-version` to require a known build. PaddleOCR's example uses pinned
`paddleocr==3.7.0` and `paddlepaddle==3.2.0` with named PP-OCRv6 models and
records its device, preprocessing, and text-joining policy. Run it in an
isolated uv invocation:

```bash
uv run --with "paddleocr==3.7.0" --with "paddlepaddle==3.2.0" \
  python examples/paddleocr_predictions.py \
  --prepared data/brothers/prepared-test --output paddle.jsonl
```

Both generators write a JSONL file and a `.provenance.json` sidecar. Their
automated tests mock OCR calls; real OCR inference was not run for this
implementation.

## 12. Compare named experiment conditions and repetitions

Version 1 remains the format for a single experiment in
[`../examples/experiment.toml`](../examples/experiment.toml). Version 2 adds
named suite conditions, per-condition model controls and prompts, and
repetitions. The example suite uses a frozen prepared snapshot and two prompt
conditions:

```bash
uv run vlm-bench suite --config examples/suite.toml \
  --output runs/prompt-suite --dry-run
uv run vlm-bench suite --config examples/suite.toml \
  --output runs/prompt-suite
uv run vlm-bench suite --config examples/suite.toml \
  --output runs/prompt-suite --resume
```

Preview lists the planned runs and benchmark fingerprints without inference.
Each condition/repetition has a distinct run identity; resuming reuses
completed runs and continues incomplete ones. A paired comparison is created
only for conditions with matching evaluation fingerprints; conditions with
different snapshots can run but are not compared as matched evaluations.
Shared sample and document counts are not multiplied by the repetition count,
and repeated-measurement variability is reported separately from
document-bootstrap uncertainty.
Incomplete repetitions do not contribute to variability summaries. Repeated
measurements bypass response caching. Choose prompts and controls on
validation data, then freeze them before test evaluation; do not select them
by repeatedly checking test results.

Each condition may use a different prepared snapshot for a genuinely
different preprocessing or sample condition. Only condition pairs with
matching fingerprints receive matched comparisons.

## 13. Retries, limits, caching, status, and JSON

Transient provider errors receive up to two retries by default; permanent
errors fail immediately. Provider retry delays are honored and every attempt
is recorded in `attempts.jsonl`. Cloud concurrency defaults to one; values
above one are supported for OpenAI and ChatGPT providers. Local backends stay
serial.

Use `--max-requests N` to stop scheduling after the configured number of
generation attempts, including warmups and retries. Use `--max-spend-usd N`
to stop based on observed or projected provider usage. A spend limit requires
applicable pricing for every selected cloud model. Unknown usage prevents
additional spend-limited calls. These limits are per invocation, including
each resume invocation; status distinguishes the current invocation from
cumulative totals. The spend limit is an estimate, not a hard invoice cap:
token usage is unknown until a response arrives, and concurrent in-flight
requests may overshoot it. Completed predictions are saved before pausing so
the run can be resumed with a new allowance.

```bash
uv run vlm-bench run --prepared data/brothers/prepared-test \
  --models openai:MODEL_ID --max-retries 2 --concurrency 2 \
  --max-requests 100
uv run vlm-bench status --run runs/YOUR_RUN_DIRECTORY
```

For a spend limit, provide rates for the same selected API model in the TOML
config. The model ID below is a placeholder; enter the provider's current
prices before running:

```toml
version = 1

[experiment]
prepared = "../data/brothers/prepared-test"
models = ["openai:MODEL_ID"]
max_spend_usd = 5.00

[costs.api]
model = "openai:MODEL_ID"
input_per_million_usd = 1.00
output_per_million_usd = 4.00
```

Save this as `experiments/budgeted-api.toml`, replace the model and sample
rates with current values, then run:

```bash
uv run vlm-bench run --config experiments/budgeted-api.toml
```

The configured cost model must match the selected API model; without
applicable rates, a spend-limited run stops before inference.

Response caching is disabled unless `--cache-dir PATH` or
`experiment.cache_dir` is supplied. The cache key includes image/crop content,
immutable model identity, prompt, and effective controls. It bypasses
repetitions, stochastic settings, missing image hashes, and model identities
without a fixed revision or digest. Cache hits still contribute their saved
predictions to accuracy and coverage, but have no provider usage and no
inference latency. They are excluded from latency and observed-usage
summaries; reports show cache-hit and measured-latency counts. Treat the cache
directory as local experiment data.

Put `--json` before a command to produce one JSON response. Progress remains
on standard error, leaving standard output parseable:

```bash
uv run vlm-bench --json status --run runs/YOUR_RUN_DIRECTORY
uv run vlm-bench --json report --run runs/YOUR_RUN_DIRECTORY --group-by writer_id
```
