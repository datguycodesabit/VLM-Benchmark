# VLM Bench

Run a local handwriting-recognition benchmark against Ollama vision models.
The program reads IAM images and references, sends only the images to Ollama,
and reports character error rate (CER), word error rate (WER), exact matches,
latency, and failures.

Everything runs locally. The repository does not upload your handwriting or
reference text.

## Run it on macOS

Install [Ollama](https://ollama.com) and [uv](https://docs.astral.sh/uv/):

```bash
brew install uv
```

Open the Ollama app, then open Terminal and run:

```bash
cd /Users/carmelodalio/Documents/ChatGPT/VLM-Benchmark
uv sync --extra dev --frozen
uv run vlm-bench models
```

The last command lists the vision models installed in Ollama. If you do not
have one, install an example model:

```bash
ollama pull qwen3.5:2b
```

Use the exact model name shown by `vlm-bench models`.

Start with three samples:

```bash
uv run vlm-bench run \
  --data data \
  --models qwen3.5:2b \
  --limit 3 \
  --seed 42 \
  --no-warmup \
  --num-predict 128
```

Then run 100 samples:

```bash
uv run vlm-bench run \
  --data data \
  --models qwen3.5:2b \
  --limit 100 \
  --seed 42
```

To compare models, list both names after `--models`:

```bash
uv run vlm-bench run \
  --data data \
  --models qwen3.5:2b another-model \
  --limit 100 \
  --seed 42
```

The same seed selects the same samples for every model.

## The data already in this repository

The supplied archive is the IAM word-image dataset. It is organized as:

```text
data/words/images/       word images with references
data/words/references/   one transcription per image
data/words/unlabeled/    images without a supplied transcription
data/words/metadata/     IAM words.txt files
```

The importer finds this layout automatically when you use `--data data`.
To prepare a reviewable 100-sample set without running a model:

```bash
uv run vlm-bench prepare \
  --data data \
  --output /tmp/vlm-prepared \
  --limit 100 \
  --seed 42
```

The importer skips unreadable or truncated word images and prints a warning.
The current archive contains one such image. IAM data remains local and is
excluded from Git by `.gitignore`.

The importer also accepts pasted IAM line or form data. For line data, keep
the line images and `lines.txt` anywhere below `data`; matching file stems are
paired automatically. For form data, put images in `data/images`, matching XML
files in `data/xml`, and optional reviewed text files in `data/references`.
Form crops exclude the printed prompt, so the model has to read the handwriting.
All crops are converted to grayscale, contrast-normalized, enlarged 3x, and
given a white border before they are sent to Ollama.

## Results

Each run creates a directory under `runs/` containing:

```text
results.xlsx     spreadsheet report
summary.csv      one row per model
samples.csv      one row per sample and model
results.jsonl    complete raw records
manifest.json    model, data, prompt, and reproducibility details
```

If a run stops, resume it with:

```bash
uv run vlm-bench resume --run runs/YOUR_RUN_DIRECTORY
```

Regenerate exports without running Ollama again:

```bash
uv run vlm-bench export --run runs/YOUR_RUN_DIRECTORY --formats xlsx csv jsonl
```

## Metrics

CER is the number of character edits divided by the number of reference
characters. WER uses words instead. Whitespace is normalized, while case and
punctuation are kept. Lower scores are better. The reports also include exact
match rate, substitution/deletion/insertion counts, mean/median/p95 latency,
throughput, truncated responses, and failed requests.

## Development

```bash
uv sync --extra dev --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

The tests use synthetic images and mocked Ollama responses. They do not require
the IAM dataset or a running model.

## License

The code is Apache-2.0 licensed. See [LICENSE](LICENSE) and [NOTICE](NOTICE)
for attribution to the upstream OCR benchmark code. IAM data and Ollama model
weights have their own terms and are not part of the repository release.

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[docs/PUBLISHING.md](docs/PUBLISHING.md) for project maintenance details.
