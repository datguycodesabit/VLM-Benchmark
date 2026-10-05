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
sample. Include `id` (or `sample_id`), `source_document`, `split`,
`content_type`, `sample_type`, and `verification_status` as available. Use
`content_type: "prose"` or `"equation"`; equation references should use
verified LaTeX where practical. Samples marked unverified or unresolved are
excluded from scored research reports.
See [Experiment setup](docs/EXPERIMENTS.md) for metadata and data-quality
guidance.

Check the folder before running a model:

```bash
uv run vlm-bench dataset check --data data/brothers --layout paired
```

Unmatched images and references, unreadable images, duplicate IDs, empty
references, and duplicate content are reported. Resolve these issues before
freezing the evaluation samples. For a writer-specific study, split by source
document before selecting crops so pages from one exam do not appear in both
training and evaluation data.

## Run a comparison

Run three samples first, then increase the limit after checking the input and
outputs:

```bash
uv run vlm-bench run --data data/brothers \
  --models ollama:qwen2.5vl:3b trocr:microsoft/trocr-base-handwritten \
  --layout paired --preprocess original --limit 3 --seed 42
```

Use `openai:MODEL_ID` for API models and `chatgpt:MODEL_ID` for models shown by
the subscription provider's `models` command. Use `--dry-run` to validate the
dataset and settings without sending images to any model.

Keep a run's selected samples fixed while comparing models. A run records its
sample IDs, reference and image hashes, preprocessing, model identifiers, and
available provider usage. Do not use the evaluation split to select prompts or
fine-tuning settings. For IAM, choose documented evaluation data where
available; handwritten TrOCR checkpoints may have trained on IAM, so arbitrary
IAM samples do not establish writer-independent test performance.

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
research.json      task-aware rankings, paired comparisons, and costs
manifest.json      frozen samples and experiment configuration
```

CER counts character edits divided by reference characters; WER does the same
for words. Lower values are better. Equation CER uses a conservative literal
normalization that collapses whitespace; it does not determine whether two
equations are mathematically equivalent. Unsupported tasks, missing outputs,
and failures remain visible and do not receive an accuracy rank. Cost estimates
are left unknown when prices or usage data were not supplied.

Resume a stopped run with:

```bash
uv run vlm-bench resume --run runs/YOUR_RUN_DIRECTORY
```

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
