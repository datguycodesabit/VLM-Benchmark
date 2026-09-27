# VLM Bench

**A local Ollama handwriting benchmark for IAM forms.**

Compare installed Ollama vision models on IAM handwritten passages, using your reference transcriptions. Drop in any number of forms; 100 forms give 100 measured samples per model. All inference stays on your machine.

This project adapts IAM cropping and CER/WER scoring from [PyaesoneP/vlm-ocr-research](https://github.com/PyaesoneP/vlm-ocr-research), pinned to commit `fd4bd0ae44db0f57f7dcb0e301a0a718d3e6159f`. See [NOTICE](NOTICE) and [LICENSE](LICENSE). It uses Ollama instead of the upstream model-specific CUDA environments.

## At a glance

| Capability | Included |
| --- | --- |
| Input | IAM form images + XML, with optional reference-text overrides |
| Evaluation | All supplied forms, or a seeded subset such as 100 forms |
| Metrics | CER, WER, exact match, edit counts, latency, and failure rate |
| Output | Excel, CSV, JSONL, raw responses, and reproducibility metadata |
| Inference | Installed local Ollama vision models; no cloud API calls |
| Recovery | Resume interrupted runs with data/model integrity checks |

**Status:** early release (`0.1.0`). The automated suite uses synthetic fixtures;
real handwriting scores require your own IAM data and installed models.

[Setup](#install) · [Dataset](#add-your-samples) · [Run](#run-a-benchmark) ·
[Metrics](#metrics) · [Contributing](CONTRIBUTING.md) ·
[GitHub upload guide](docs/PUBLISHING.md)

## Install

Requirements: Python 3.11 or newer, [Ollama](https://ollama.com), and at least one locally installed vision model. On macOS, the system `python3` may be too old; use a current Python installation or `uv`.

From a downloaded or cloned copy of this repository, run:

```bash
uv sync --python 3.12 --extra dev --frozen
source .venv/bin/activate
```

Or with an existing Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows activate with `.venv\Scripts\Activate.ps1` in PowerShell. Start the Ollama app, or run `ollama serve` in a separate terminal. Install your chosen vision models using Ollama before benchmarking; this program never downloads models.

## Add your samples

Use IAM **offline handwritten form images and matching XML**, not the IAM online pen-trajectory database. IAM was first published in 1999; use the form images and XML from the same downloaded release. Obtain the dataset yourself from the [official IAM download page](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database), following its registration and usage terms. Dataset images are not bundled.

```text
data/
  images/
    a01-000u.png
    a01-000x.png
  xml/
    a01-000u.xml
    a01-000x.xml
  references/
    a01-000u.txt
    a01-000x.txt
```

Drag files into these folders with Finder/File Explorer. Nested directories are allowed, but each form ID must be unique. Supported image extensions: PNG, JPG, JPEG. Matching uses the filename without its extension.

Your UTF-8 reference text takes precedence over XML text. It must contain exactly the handwritten passage you want scored, including spelling mistakes and punctuation, without the printed prompt or signature. Reference text is never sent to the model and is not used for training.

If you do not already have text files:

```bash
vlm-bench prepare --data ./data
```

This validates all inputs, writes handwriting-only crops into `data/prepared/crops`, and creates missing text files from the XML. It never replaces existing reference files. Review the crops and generated references before running. Without `prepare`, `run` can use XML references directly.

Full IAM forms include a printed copy of the answer. The program removes it by cropping the union of handwritten XML components, adding 20 pixels of padding within image bounds. It uses the same retained lines for cropping and XML reference extraction. The upstream signature rule excludes a final line containing `Name:` or a short line in the bottom 15% of the page; inspect your crops if a form has unusual layout. Images with missing XML, invalid coordinates, empty references, or duplicate IDs fail validation before inference.

## Run a benchmark

List your installed models and whether they are eligible:

```bash
vlm-bench models
```

Substitute exact model names from that output:

```bash
vlm-bench run --data ./data --models MODEL_A MODEL_B
```

Every image in the folder is evaluated once per model. To choose exactly 100 from a larger folder:

```bash
vlm-bench run --data ./data --models MODEL_A MODEL_B --limit 100 --seed 42
```

Models run sequentially on the same frozen sample list. Each model receives one excluded warmup request, then one measured request per sample. Each request is a fresh conversation with the cropped image and the same verbatim-transcription prompt. No model-specific cleanup or spelling correction is applied.

Defaults: temperature 0, seed 42, 4,096 output tokens, 300-second timeout, `http://localhost:11434`. Change `--num-predict`, `--timeout`, `--seed`, or use `--no-warmup` as needed. `--base-url` supports another loopback port. Cloud-backed models and remote hosts are rejected. Reproducible settings do not guarantee bit-identical generation on every hardware/backend version.

Each run saves under `runs/TIMESTAMP-ID/`. The program prints progress, summaries, and output paths. Exit code 0 means successful completion, 2 means completed with sample failures, 1 means validation/setup/export failure, and 130 means interrupted.

## Metrics

- **CER:** total character substitutions + deletions + insertions, divided by total reference characters. Lower is better.
- **WER:** total word edit distance divided by total reference words.
- **Mean sample CER/WER:** average of each form's score; unlike corpus CER/WER, each form has equal weight.
- **Exact match:** fraction of successful normalized transcriptions identical to the reference.
- **Character error counts:** substitutions, deletions, insertions, and total edits.
- **Speed:** mean, median, p95 request latency and samples/minute. Ollama model-load duration is recorded separately for warmups and measured requests; wall latency still includes any load time within that measured request.
- **Reliability:** successful, failed, empty, and truncated outputs.

Primary metrics collapse whitespace and preserve case and punctuation. Raw CER also preserves original whitespace. CER/WER can exceed 100% because insertions add errors. Empty successful responses score as all deletions. Infrastructure failures receive no accuracy score and remain visible; incomplete models are not ranked alongside complete models. Truncated responses are scored as returned and flagged. Accuracy summaries for incomplete models describe only their successful samples.

Latency excludes warmups and local scoring, but includes image reading/encoding, HTTP overhead, and inference. No misleading Python-process VRAM measurement is reported for Ollama's separate process. Model metadata and available Ollama timing fields are retained.

This is a benchmark of your supplied forms, not an official IAM split or proof that a model has never seen IAM during training. Use a separate development set for prompt tuning. For broader conclusions, evaluate more writers and forms.

## Results and recovery

Every measured result is appended and flushed immediately to `results.jsonl`, including full raw response, reference, prediction, status, metrics, and timing. Warmups are stored separately in `warmups.jsonl`. `manifest.json` records the selected samples, input hashes, prompt, generation settings, model digest/quantization metadata, Ollama version, and host information. Crops are preserved in the run directory.

Resume after Ctrl-C or an interrupted process:

```bash
vlm-bench resume --run ./runs/RUN_ID
```

Resume processes only model/sample pairs with no saved result, with a fresh excluded warmup per pending model. Already recorded failures are not retried automatically; create a new run after fixing their cause. Changes to source data, references, crops, model digest, Ollama version, prompt, or settings prevent resume. Keep the original data and run folders in place. A partial final JSONL line from an interrupted write is backed up and repaired during resume.

Exports are generated automatically at completion. Regenerate them without inference:

```bash
vlm-bench export --run ./runs/RUN_ID --formats xlsx csv jsonl
vlm-bench rescore --run ./runs/RUN_ID
```

`rescore` recalculates metrics from saved predictions and frozen references; it does not adopt edited source references.

- `results.xlsx`: Summary, Samples, Errors, and Run Configuration sheets with filters, frozen headers, and percentage formats. Warmups appear in a separate sheet when present. Long text is preserved in a separate overflow sheet.
- `summary.csv` and `samples.csv`: tabular scores and sample details.
- `results.jsonl`: full-fidelity canonical records suitable for scripts and analysis.

Excel stores model text as literal strings; XML-illegal control characters are represented with explicit JSON encoding. CSV uses spreadsheet-safe escaping for formula-like text; JSONL retains the original content.

## Development and validation

```bash
uv sync --extra dev --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

GitHub Actions runs the test suite across Linux, macOS, and Windows, with Python
3.11–3.13 coverage. A separate job checks style, builds distributions, and tests
the installed wheel outside the source checkout. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the contribution workflow and [CHANGELOG.md](CHANGELOG.md) for changes.

Tests use synthetic images/XML and mocked Ollama responses. They cover cropping/reference alignment, validation, deterministic sampling, known edit distances, weighted aggregation, failure visibility, text-safe exports, request payloads, interrupted runs, and resume integrity. Running these tests does not require models, an Ollama server, or the IAM dataset.

A real acceptance benchmark requires your 100 IAM forms/XML/references and two installed local vision models. Synthetic tests do not measure model handwriting quality.


## Project layout

```text
src/vlm_bench/     CLI, dataset preparation, Ollama client, scoring, and exports
tests/            Synthetic and mocked tests; no dataset download required
data/             Drop your images, XML, and references here (Git ignores them)
docs/             Publishing instructions
.github/          CI, dependency updates, and issue/PR templates
```

## License and attribution

Code is distributed under [Apache-2.0](LICENSE). Adapted upstream components and
their source revision are identified in [NOTICE](NOTICE). IAM data and model
weights are not distributed with this repository; their own terms apply.

Report bugs using the repository's issue templates. For vulnerabilities, follow
[SECURITY.md](SECURITY.md). See [CONTRIBUTING.md](CONTRIBUTING.md) before sending
changes.
