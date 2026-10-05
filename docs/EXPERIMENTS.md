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
first, then assign whole documents to reproducible 70/15/15 splits:

```bash
uv run vlm-bench dataset split \
  --data data/brothers \
  --output /tmp/brothers-splits.jsonl \
  --seed 42
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
counts. The splitter checks that exact duplicate images do not cross documents.
Do not change the test set after looking at model results.

The shipped IAM archive is useful for public-data smoke tests. Use its
documented evaluation partitions where available. The standard handwritten
TrOCR checkpoint was trained on IAM, so a random IAM sample is not an
independent test of that checkpoint. Keep the public-data and Dr. Brothers
results distinct.

## 4. Run the same samples through each model

Use a small dry run first. It validates configuration and reports which models
would receive local or cloud inference without sending images:

```bash
uv run vlm-bench run --data data/brothers \
  --models ollama:qwen2.5vl:3b trocr:microsoft/trocr-base-handwritten \
  --layout paired --split test --content-type prose \
  --limit 3 --seed 42 --dry-run
```

Then run the comparison:

```bash
uv run vlm-bench run --data data/brothers \
  --models ollama:qwen2.5vl:3b trocr:microsoft/trocr-base-handwritten \
  --layout paired --split test --content-type prose \
  --limit 100 --seed 42
```

To include a subscription or API model, list its explicit selector with the
others. `doctor` checks provider setup, dependencies, and model availability:

```bash
uv run vlm-bench doctor --provider chatgpt \
  --models chatgpt:MODEL_ID_FROM_MODELS
```

Use the same seed, limit, split, content type, and preprocessing profile for
every model. The run manifest freezes the exact sample IDs, references, hashes,
settings, and model identities. Multiple preprocessing profiles can be
specified in the TOML file for controlled comparisons. Resume an interrupted
run with:

```bash
uv run vlm-bench resume --run runs/YOUR_RUN_DIRECTORY
```

The optional `--retry-failed` applies only to saved errors; completed
predictions are retained. Subscription usage pauses are resumable. Remote
requests and local inference may have different generation controls; recorded
settings describe what each provider actually accepted.

## 5. Configuration and cost assumptions

Use the versioned example in [`../examples/experiment.toml`](../examples/experiment.toml)
as a starting point:

```bash
uv run vlm-bench run --config examples/experiment.toml --dry-run
```

The config selects the data root, models, split, preprocessing, and seed. It
can set per-model options such as TrOCR device or beam count. Do not put
credentials in TOML. The OpenAI API key belongs in the process environment; the
ChatGPT sign-in token is stored through the credential store.

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

## 6. Read the research reports

The report provides separate prose and equation tracks; IAM word data appears
as a separate diagnostic. CER and WER use the same verified eligible test
samples for each model. A model receives a rank
only after successful predictions cover the full eligible set; unsupported
tasks and failed or missing predictions remain visible. Paired differences
compare two models on samples both completed. The confidence interval resamples
source documents as groups when document IDs are available, and uses individual
samples otherwise. A negative `right minus left` CER difference means the
right-hand model had lower error in that pair.

Timing summaries show end-to-end request latency and inference latency when
the provider reports it. For cloud calls, inference latency includes network
time and retries. Throughput is successful samples per minute divided by total
measured end-to-end request time. Cold model-load time comes from a successful
warmup when available, or from load timing on sample requests when warmup is
disabled. Missing measurements remain blank.

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
