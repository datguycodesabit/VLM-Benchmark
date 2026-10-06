# VLM Bench

VLM Bench compares handwriting recognition systems on the same images and
verified references. It supports local Ollama vision models, TrOCR, eligible
ChatGPT subscription models, and the OpenAI API. Results include transcription
accuracy, coverage, speed, and cost estimates when the required inputs are
available.

The goal is to test whether a model adapted to one writer can make difficult
handwritten teaching materials easier to use, and whether the improvement is
worth the setup and operating cost. The benchmark is a research tool; review
transcriptions before relying on them in coursework.

## Install

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --extra dev --extra cloud --extra trocr --frozen
```

Ollama models run locally. Pull a vision model and check it is available:

```bash
ollama pull qwen2.5vl:3b
uv run vlm-bench models --provider ollama
```

The command above installs all optional providers. For an Ollama-only setup,
`uv sync --extra dev` is enough; add `--extra cloud` and/or `--extra trocr` for
those providers. See [Experiment setup](docs/EXPERIMENTS.md) for device details.

ChatGPT subscription access and OpenAI API access are separate providers:

```bash
uv run vlm-bench auth login --provider chatgpt
uv run vlm-bench auth status
uv run vlm-bench models --provider chatgpt
```

Subscription access is available only to eligible accounts and compatible
models. It uses your account's subscription allowance; the benchmark does not
silently switch to paid API calls. The API provider requires a separately
configured API key and can incur usage charges.

## Add your images and references

Use matching relative paths and filename stems. No IAM files or XML are needed
for this paired layout:

```text
data/brothers/
  images/
    exam2006/page03-line02.png
  text/
    exam2006/page03-line02.txt
```

Each `.txt` file contains the verified transcription for its image. Use
`references/` instead of `text/` if preferred. Keep subfolders and basenames
the same under both folders. Supported images are PNG and JPEG. The source
images are preserved; `--preprocess original` sends the original image, while
`--preprocess enhanced` applies grayscale conversion, contrast normalization,
enlargement, and padding as a separate experimental condition.

Optionally add `metadata.jsonl` in the data root, with one JSON object per
sample. Include `id` (or `sample_id`), `source_document`, `writer_id`, `split`,
`content_type`, `sample_type`, `difficulty`, and `verification_status` as
available. Use `content_type: "prose"` or `"equation"`; equation references
should use verified LaTeX where practical. Samples marked unverified or
unresolved are excluded from scored research reports. Optional
`metadata.annotations` can hold critical expressions and reading-order pairs
for separate task metrics; see [Experiment setup](docs/EXPERIMENTS.md).

Check the folder before running a model:

```bash
uv run vlm-bench dataset check --data data/brothers --layout paired
```

Unmatched images and references, unreadable images, duplicate IDs, empty
references, duplicate content, split leakage, and perceptually similar images
to review are reported. Resolve blocking issues and inspect review findings
before freezing evaluation samples. Split by source document by default; use
the writer-disjoint protocol when writers must be held out.

## Run a comparison

Freeze the reviewed test set once, then use the resulting portable snapshot for
each model:

```bash
uv run vlm-bench prepare --data data/brothers \
  --output data/brothers/prepared-test \
  --layout paired --preprocess original --split test \
  --content-type prose --limit 100 --seed 42

uv run vlm-bench dataset check --prepared data/brothers/prepared-test
```

The prepared folder contains the frozen samples, references, metadata, and
version 2 manifest. Its fingerprint identifies the benchmark. Keep the folder
together and do not edit it; preparation refuses to overwrite an existing
snapshot.

Preview and run each model on the same snapshot. `--dry-run` reports the
fingerprint, sample and verification-status counts, model eligibility, and
effective controls without sending images:

```bash
uv run vlm-bench run --prepared data/brothers/prepared-test \
  --models ollama:qwen2.5vl:3b --dry-run

uv run vlm-bench run --prepared data/brothers/prepared-test \
  --models ollama:qwen2.5vl:3b

uv run vlm-bench run --prepared data/brothers/prepared-test \
  --models trocr:microsoft/trocr-base-handwritten
```

Use `openai:MODEL_ID` for API models and `chatgpt:MODEL_ID` for models shown by
the subscription provider's `models` command. Compare the saved runs after
recording the run directories printed by each command:

```bash
uv run vlm-bench compare \
  --runs runs/RUN_DIRECTORY_FOR_OLLAMA runs/RUN_DIRECTORY_FOR_TROCR \
  --output comparisons/brothers-test
```

The comparison writes `comparison.json`, `comparison.csv`, `paired.csv`, and
`costs.csv` after checking that the runs used the same benchmark fingerprint.
Resume an interrupted run with `uv run vlm-bench resume --run runs/YOUR_RUN_DIRECTORY`;
resume keeps the original snapshot and predictions.

The direct source-data form remains useful for small one-off checks:

```bash
uv run vlm-bench run --data data/brothers \
  --models ollama:qwen2.5vl:3b --layout paired --limit 3 --seed 42 --dry-run
```

For a strict research run, require verified test references and source-document
metadata. With source data, the complete dataset is audited before the test
selection is frozen:

```bash
uv run vlm-bench run --data data/brothers --models ollama:qwen2.5vl:3b \
  --strict-research --protocol document-disjoint --dry-run
```

Strict runs require a verified reference, test split, and `source_document` for
each selected sample. Writer-disjoint runs also require `writer_id`. A prepared
snapshot can validate only its frozen samples; it cannot prove that unseen
training and validation data are leakage-free.

Review saved errors, subgroup coverage, and read-only run status without new
inference:

```bash
uv run vlm-bench inspect --run runs/YOUR_RUN_DIRECTORY --worst 20
uv run vlm-bench report --run runs/YOUR_RUN_DIRECTORY --group-by writer_id
uv run vlm-bench status --run runs/YOUR_RUN_DIRECTORY
```

Inspection writes a local HTML review with images, references, predictions,
edit alignments, and output warnings. Group reports accept `writer_id`,
`difficulty`, `source_document`, or `sample_type`; missing values appear in an
explicit unknown group. Put `--json` before the command, for example
`uv run vlm-bench --json status --run runs/YOUR_RUN_DIRECTORY`, to emit one
JSON object while progress messages go to stderr.

External predictions can be imported against the same prepared snapshot.
Starter generators are in [`examples/tesseract_predictions.py`](examples/tesseract_predictions.py)
and [`examples/paddleocr_predictions.py`](examples/paddleocr_predictions.py):

```bash
uv run python examples/tesseract_predictions.py \
  --prepared data/brothers/prepared-test --output predictions.jsonl
uv run vlm-bench import --prepared data/brothers/prepared-test \
  --predictions predictions.jsonl --system tesseract \
  --provenance-file predictions.jsonl.provenance.json --output runs/tesseract
```

The Tesseract generator needs an existing system executable; PaddleOCR uses
pinned optional Python packages through `uv run --with`. Imported systems use
the selector `external:<system>`. Missing predictions remain incomplete and
unranked, and supplied timing/usage are labeled external. These example
pipelines have mocked tests; real OCR inference was not run for this release.

Optional rendered formula similarity uses the pinned MathText renderer:

```bash
uv sync --extra formula-render
uv run vlm-bench run --prepared data/brothers/prepared-test \
  --models ollama:qwen2.5vl:3b --formula-rendering
```

It supports Matplotlib MathText’s TeX subset, not full LaTeX. Rendering
failures are shown per sample while literal CER/WER remain available. Critical
expressions and relative reading-order annotations add separate metrics with
their own coverage; they do not alter model rankings.

Named prompt/control conditions and repetitions use a version-2 suite config.
Preview all planned runs first, then execute or resume the same output:

```bash
uv run vlm-bench suite --config examples/suite.toml \
  --output runs/prompt-suite --dry-run
uv run vlm-bench suite --config examples/suite.toml \
  --output runs/prompt-suite
uv run vlm-bench suite --config examples/suite.toml \
  --output runs/prompt-suite --resume
```

Each condition/repetition receives a distinct run identity. Reports count the
shared samples and documents once and keep repetition variability separate
from document-bootstrap uncertainty. Comparisons require matching evaluation
fingerprints. Select prompts on validation data, then freeze them before
evaluating the held-out test set.

The complete CLI examples, dataset split workflow, configuration format, and
research cautions are in [Experiment setup](docs/EXPERIMENTS.md).

## Results

Each run is stored in `runs/<run-id>/`. The exports include:

```text
results.xlsx       model and sample results with research sheets
summary.csv        per-model metrics
samples.csv        per-sample predictions and scores
paired.csv         paired model differences and confidence intervals
tracks.csv         prose, word diagnostic, and equation summaries
costs.csv          cost scenarios and break-even result
results.jsonl      canonical per-sample records
research.json      task-aware metrics, paired comparisons, costs, and coverage
manifest.json      frozen samples and experiment configuration
attempts.jsonl     individual generation attempts, retries, and outcomes
execution.json     request/spend/cache totals and invocation history
```

CER counts character edits divided by reference characters; WER does the same
for words. Lower values are better. Equation CER uses a conservative literal
normalization that collapses whitespace; it does not determine whether two
equations are mathematically equivalent. Optional formula and annotation
metrics remain separate, with per-metric errors and coverage. Unsupported
tasks, missing outputs, and failures remain visible and do not receive an
accuracy rank. Cost estimates are left unknown when prices or usage data were
not supplied.

The manifest stores the snapshot fingerprint, frozen sample IDs, references,
hashes, protocol and eligibility settings, prompt hashes, model identifiers,
and execution controls. Resume a stopped run with:

```bash
uv run vlm-bench resume --run runs/YOUR_RUN_DIRECTORY
```

Retries default to two retries for classified transient provider errors;
permanent errors fail without replay. Concurrency defaults to one and is only
available for OpenAI and ChatGPT providers. `--max-requests` counts generation
attempts, including warmups and retries but excluding metadata requests.
Request and spend budgets are per invocation: resuming starts a new allowance,
and current/cumulative totals are shown separately. A configured
`--max-spend-usd` cap requires applicable prices and is an estimated soft
limit. Actual token cost is unknown before provider usage returns, and
concurrent in-flight requests can overshoot the estimate.

Response caching is off by default. Enable it with `--cache-dir PATH` or
`experiment.cache_dir`. Cache keys include the image crop hash, immutable
model identity, prompt, and effective controls; stochastic settings, repeated
measurements, and unversioned models bypass the cache. Hits still count for
accuracy and coverage but have no provider usage or inference latency and are
excluded from those summaries. Reports expose cache-hit and measured-latency
counts. The cache stores transcription responses at the selected local path.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Most tests use synthetic images and mocked providers; they do not require the
IAM dataset, cloud credentials, or a running model.

The application code is Apache-2.0 licensed. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). IAM data and model weights have separate terms and are not
part of this repository. See [CONTRIBUTING.md](CONTRIBUTING.md),
[SECURITY.md](SECURITY.md), and [docs/PUBLISHING.md](docs/PUBLISHING.md) for
project maintenance details.
