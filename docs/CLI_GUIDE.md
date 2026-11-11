# VLM Benchmark CLI workflow

For the controlled 3B/7B handwriting diagnostic experiment and human review,
see [DIAGNOSTIC_STUDY.md](DIAGNOSTIC_STUDY.md).

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

## 0.2.0 command completion

See [COMPLETION_REPORT.md](COMPLETION_REPORT.md) for the completed workflow and
validation. `status` and `resume` accept an experiment or suite directory as well
as an individual run. For a relocated suite configuration, use
`resume --run SUITE --config CONFIG.toml`.

`compare` uses weighted ranking for new-policy sources. Matching suite condition
names are compared independently; mismatched snapshots, pass counts or frozen
policies fail clearly. `--weighted` explicitly derives default-policy scores for
legacy sources; `--scoring-config CONFIG.toml` explicitly re-evaluates all sources
under a v3 policy. These options also work with `report` on raw saved evidence.
New CLI imports freeze a weighted policy by default, so they no longer need a
second command to obtain weighted reports. Older imports remain readable.

Ollama model settings can include `streaming = true` to collect client TTFO while
buffering a complete prediction. Ollama and TrOCR rate observations include token
count source, tokenizer/model identity when available, and special-token policy.
Cloud prefill and decode rates stay unknown when the provider does not expose them.

CSV exports now include diagnostic tables alongside `leaderboards.csv`. Reports
can be regenerated from a saved `comparison.json` or `evaluation.json` without the
original inference directory. Such regeneration preserves the existing evaluation;
changing its policy requires the original raw artifacts.
# IAM word image-size filtering

`prepare --min-pixels N` excludes crops with native width × height below N,
before enhancement and random subset selection. The default is zero (no filter).
Excluded samples are listed with dimensions and reasons in
`PREPARED/image-size-filter.json`. The snapshot records the threshold, population
counts and audit hash; changing or removing the audit invalidates that snapshot.
The requested `--limit` is drawn from eligible samples, not the full population.

For an exploratory Ollama word test with a 32-token output cap:

```bash
uv run vlm-bench prepare --data data/words --layout iam-words \
  --min-pixels 10000 --limit 100 --seed 43 --output data/iam-min10000-seed43
uv run vlm-bench run --config examples/iam-ollama-filtered.toml \
  --repeats 1 --output runs/iam-filtered-test
```

This cutoff was calibrated using one 100-image run: at least 10,000 native
pixels retained 49 images with 26 exact matches (53.1%), versus 31/100 overall.
Higher cutoffs were not consistently better. Area is confounded by word length,
punctuation and handwriting difficulty; this is not a universal minimum or an
accuracy gain on the original benchmark. Validate on fresh samples and retain
the unfiltered baseline. A token cap contains runaway text but can truncate
answers; keep truncation flags visible. The cap is specific to word experiments,
not a new global default for lines, pages or equations.

Reproduce the exploratory threshold table without inference:

```bash
uv run python scripts/analyze_image_sizes.py --run RUN --output runs/size-analysis
```

## Content scoring and human crop review

Use an explicit scoring policy when punctuation and spacing should not count as
prose transcription errors. `examples/scoring-content.toml` selects
`unicode-nfc-no-punctuation-or-spacing-v1`: NFC-normalize text, remove Unicode
punctuation and all whitespace, and preserve capitalization, digits, and other
symbols. Thus `living room` and `livingroom` match. This is a content metric;
it does not measure faithful word boundaries. WER is unavailable under this
policy, because removing spaces removes those boundaries. Equation scoring
remains strict: its LaTeX punctuation, superscripts, signs, and braces are not
removed. Punctuation-only prose references have no scorable content and are
explicitly listed as `empty_reference_after_normalization` exclusions.

Review crop quality independently of model predictions. Create a portable gallery
from a frozen dataset:

```bash
.venv/bin/vlm-bench review \
  --prepared data/iam-official/aachen/prepared-100-seed42 \
  --output data/iam-official/aachen/human-review-v1
```

Open the generated `review.html`, enter your name, and choose `keep` or `exclude`
for every crop. Exclusions require a reason code and an explanation. Retain hard
but readable handwriting and ordinary crop seams; exclude severe fragmentation,
missing content, or invalid image/reference pairs. The page shows images and
references, with no model predictions. Export the completed `review.json` before
closing the page; page edits are not automatically saved. The browser downloads
a new JSON file; use its actual path in the commands below. The initial JSON
beside the gallery remains a pending template until explicitly replaced.
Alternatively, edit that template in a text editor: set `reviewer`, an ISO timestamp
in `reviewed_at`, and each sample's decision. Keep decisions require empty reason
fields; exclusions require `severe_fragmentation`, `reference_mismatch`,
`unreadable`, or `other` plus a non-empty explanation.

A declared completed review is not independent certification; record the actual
reviewer. Agent-generated flags and suggestions must be checked by the person
completing the review. For a defensible new benchmark, review before comparing
model outputs. Reviewing after seeing predictions is a post-hoc sensitivity
analysis and should be reported as such.

Re-score saved predictions without inference, retaining the original artifacts:

```bash
.venv/bin/vlm-bench report --run runs/YOUR_RUN_OR_EXPERIMENT \
  --scoring-config examples/scoring-content.toml \
  --review /absolute/path/to/completed-review.json \
  --output reports/content-reviewed
```

`--review` requires `--scoring-config` and a separate `--output`. The completed
review must cover exactly the frozen sample roster and match its dataset
fingerprint, crop hashes, and reference hashes. Pending decisions, missing
reviewer details, stale hashes, unknown IDs, or unexplained exclusions fail
rather than silently changing eligibility. Reviews never change references,
delete evidence, or turn a missing/failed prediction into an exclusion. The
same decisions apply to every model and pass. This offline filter does not skip
inference on future runs; it controls the scored roster.

Compare models with the same policy and reviewed roster:

```bash
.venv/bin/vlm-bench compare --runs runs/MODEL_A runs/MODEL_B \
  --scoring-config examples/scoring-content.toml \
  --review /absolute/path/to/completed-review.json \
  --output reports/content-comparison
```

The embedded review participates in the evaluation identity. Reports include the
normalization, eligible counts, and exclusions with reasons; CSV and Excel also
include an Exclusions table. Existing strict reports and raw legacy metrics
remain available. The difficult-example diagnostic retains its original strict
metrics and is labeled separately. Never compare a content score against an
older strict score as if they were the same evaluation. Comparisons still require
matching snapshots and planned pass counts; unfinished experiments remain
unfinished after exclusions.

## Inspect results during inference

Open a second terminal and inspect the physical pass directory (the directory
containing `manifest.json`), without pausing the benchmark:

```bash
.venv/bin/vlm-bench inspect --run runs/EXPERIMENT/PASS_DIRECTORY --live --worst 100
```

`--live` reads one snapshot of complete newline-terminated result records, without
taking the inference lock or repairing/changing source evidence. An incomplete
last write is left for the running process; malformed completed records still
raise an error. The generated timestamped HTML includes the images, references,
predictions, strict CER, and character alignments for saved results. It shows
recorded versus expected counts and is labeled a partial snapshot, not a final
benchmark. Unfinished requests are not included. `--worst 100` shows up to 100
saved results, ordered by error; use the default 20 for a shorter review.

Repeat the command for a newer snapshot. Existing pages do not update themselves.
Each default live output has a new timestamp; an explicit `--output` must be a
new filename. These inspection diagnostics use strict scoring; content-policy
leaderboards remain a separate `report --scoring-config` operation.
