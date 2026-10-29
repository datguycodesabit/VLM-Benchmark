# VLM Benchmark CLI workflow

The product requirements are in [PRODUCT_SPEC.md](PRODUCT_SPEC.md); the engineering
contracts are in [ARCHITECTURE.md](ARCHITECTURE.md). New configuration version 3
freezes the weighted policy and measurement profile. Existing version 1/2
configurations and legacy scoring remain readable.

## Prepare, configure, and run

```bash
vlm-bench prepare --data data/my-dataset --layout paired --output data/prepared
vlm-bench init --prepared data/prepared --models ollama:qwen2.5vl:3b openai:MODEL_ID \
  --output benchmark.toml
vlm-bench suite --config benchmark.toml --output runs/comparison --dry-run
vlm-bench suite --config benchmark.toml --output runs/comparison
vlm-bench suite --config benchmark.toml --output runs/comparison --resume
vlm-bench report --run runs/comparison
```

`init` without model/path arguments prompts when connected to a terminal. In
scripts, supply `--prepared` and `--models`. It refuses to overwrite a file and
validates its output through the normal configuration loader. Paths in TOML are
relative to the configuration file.

Suites default to three full passes. Each pass gets a distinct run and repetition
identity. Direct `run` without a legacy configuration also defaults to three
passes; use `--repeats 1` for a single pass. Operational retry counts are separate
from passes. Request and estimated-spend limits apply per engine invocation,
including warmups; a three-pass experiment can use three separate allowances.

## Scoring policy

```toml
version = 3

[experiment]
prepared = "data/prepared"
warmup = true

[scoring]
character_accuracy_weight = 0.70
exact_match_weight = 0.30
accuracy_margin_points = 0.5

[measurement]
profile = "latency"
concurrency = 1
batch_size = 1

[suite]
repeats = 3

[[suite.conditions]]
name = "baseline"
models = ["ollama:qwen2.5vl:3b"]
```

Per track and pass, score is
`100 * (character_weight * max(0, 1 - corpus_CER) + exact_weight * exact_match_rate)`.
Corpus CER divides total character edits by total reference characters. Raw CER
can exceed 100%; only composite character accuracy is clamped. WER is diagnostic.
Missing and failed predictions incur reference-character deletions and zero exact
matches. Unsupported tracks are explicit exclusions, never successful blanks.

Pass scores are averaged. Prose and equation rankings are separate. Bands start
at the highest remaining unrounded score and include scores within the configured
margin. Within a band, pooled successful uncached sequential median request
latency resolves placement; residual ties use score and canonical model name.
Missing comparable timing withholds speed-resolved placement in the affected
band. Incomplete experiments cannot produce a final winner. Document uncertainty
is diagnostic and does not change practical bands.

Equation equivalence is a separate conservative diagnostic. Unsupported notation,
parsing failures, and inconclusive proofs remain undetermined; agreement at a
finite set of points cannot prove equivalence. Fidelity alone determines score.

## Offline reports

```bash
vlm-bench report --run runs/comparison --output reports/comparison
vlm-bench report --run RUN_DIRECTORY --formats html xlsx csv json
vlm-bench report --run RUN_DIRECTORY --group-by source_document
```

The complete report writes `report.html`, `report.xlsx`, `leaderboards.csv`, and
`evaluation.json` from one evaluation model. Each condition is evaluated separately,
so different prompts do not silently share one leaderboard. A new derived evaluation
leaves historical result rows and legacy exports intact. HTML escapes prediction
text; spreadsheet exports protect against formula-like transcription text.

`export` retains the historical per-run artifacts; `report` provides the new weighted
evaluation. Generating reports requires no provider authentication or inference.

## Local devices and custom models

TrOCR accepts `auto`, `cpu`, `mps`, `cuda`, and `rocm`. Explicit accelerator requests
fail if unavailable; `auto` uses a recorded selection/fallback policy. ROCm uses
PyTorch's CUDA-facing API while recording HIP runtime identity separately. Device
selection code and simulated tests do not establish validated hardware support.

Custom local selectors have the form `python:package.module:Factory`. Install your
adapter package in the same environment as the CLI. The factory is instantiated
without arguments and declares backend capabilities. Its lifecycle implements:

- `load(model, *, device, settings)`
- `preprocess(image, *, track, prompt, controls)`
- `infer(prepared, *, controls)`
- `decode(output)` returning transcription text
- Optional identity and unload hooks, as documented in `vlm_bench.adapters`

The adapter receives an image, prompt, track, and generation controls. Reference
text is withheld. Use the shared device layer for accelerator selection and timing
synchronization. Adapters must record model/checkpoint revision and effective
precision in their identity; a module source hash alone does not identify weights.

## Measurement limits

Ranking uses single-request wall latency, including work inside the adapter request.
Provider timings retain their source and boundaries. Missing cloud internals stay
null; time to first output is not called pure prefill. Cached results do not provide
measured inference speed. Throughput experiments are labeled separately and do not
supply sequential ranking latency. Unknown usage/cost is not displayed as free.

Actual GPU execution, live cloud reliability, real handwriting quality, and
hardware throughput require measurements on the intended models and machines.
See [UPDATE_REPORT.md](UPDATE_REPORT.md) for this change's validation evidence.

Legacy and imported runs without a frozen weighted policy need an explicit opt-in:
`vlm-bench report --run RUN_DIRECTORY --weighted`. This writes a separate derived
evaluation and preserves the historical results and rankings. Resume a direct
three-pass experiment with `vlm-bench resume --run EXPERIMENT_DIRECTORY`.
