# Qwen handwriting failure diagnosis

This exploratory study asks whether errors change with model size and image
presentation. It does not measure general intelligence or establish a universal
minimum image size.

## Design

| Factor | Conditions |
|---|---|
| Model | Ollama `qwen2.5vl:3b` and `qwen2.5vl:7b` |
| Dataset | Existing 100-image filtered benchmark; separate 30-image small-crop challenge |
| Image preparation | Original; 3× Lanczos enlargement plus 16-pixel white border |
| Controls | Identical English OCR prompt, 32-token cap, temperature 0, seed 42 |
| Execution | One pass, sequential requests, warmup per model/condition, streaming timings |

The four dataset/preparation conditions produce 520 scored predictions plus
eight warmups before operational retries. Runtime depends on the actual hardware
and model performance; 1–2 hours is the user's budget, not a measured guarantee.
The padded profile preserves color and contrast. It deliberately does not use
the existing `enhanced` profile, which also converts to grayscale and normalizes
contrast.

The challenge set is sampled reproducibly from the earlier unfiltered run,
balanced across native pixel-area bands where available. Selection uses image
area and seed, never prediction quality. It is a diagnostic challenge, not a
representative IAM sample. IDs and references match across preparation variants;
cohorts remain disjoint. The filtered cohort has already been observed. These
are calibration/diagnostic results, not an untouched test-set estimate.

## Prepare and run

The study is already prepared at `data/qwen-diagnostic-study` in this checkout.
Open Ollama and obtain both models:

```bash
ollama pull qwen2.5vl:3b
ollama pull qwen2.5vl:7b
uv run python scripts/ollama_diagnostic_study.py run \
  --study data/qwen-diagnostic-study --output runs/qwen-diagnostics
```

The script uses the normal suite engine and generates the standard report plus
`runs/qwen-diagnostics/diagnostics/review.html` automatically on completion.
To resume interrupted work, repeat the command with `--resume`. Completed
predictions are preserved. Do not edit the study configuration to resume;
prepare a separate study for changed settings.

To prepare another study in a new directory:

```bash
uv run python scripts/ollama_diagnostic_study.py prepare \
  --filtered data/iam-min10000-seed43 \
  --challenge-source runs/iam-ollama-test/20261101-192204-bfc5861d \
  --output data/qwen-diagnostic-study-v2
```

For a partial or regenerated diagnostic report, choose a new output directory:

```bash
uv run python scripts/ollama_diagnostic_study.py report \
  --study data/qwen-diagnostic-study --runs runs/qwen-diagnostics \
  --output runs/qwen-diagnostics/review-v2
```

## Evidence and review

- `observations.json` / `.csv`: per-image native and submitted dimensions,
  aspect ratio, reference/output lengths, status, exact match, edit operations,
  heuristic flags, client latency, TTFO, provider load/prompt/decode timings,
  observed token counts/rates and measurement sources. Missing values stay null.
- `summary.csv`: per-model/condition accuracy, coverage, truncation and latency.
- `paired.csv`: both correct, 7B-only correct, 3B-only correct, both wrong,
  and missing pairs; equivalent paired contrasts for image preparation.
- `subgroups.csv`: punctuation, reference-length and native-area breakdowns.
  Groups overlap and are not additive.
- `diagnostics.json`: source identities, effective controls, limitations, and
  resource observations. Ollama `/api/ps` allocation observations run only after
  warmup, outside ranked sample requests. They are provider-reported snapshots,
  not process peak RAM, electricity consumption or GPU utilization. Missing
  endpoint/fields are explicit; they do not turn into zero memory usage.
- `review.html`: up to 25 distinct images, sampling disagreements, shared errors
  and successful controls across both cohorts. Review all four predictions for
  each image. Suggested flags are not proven error causes. Categories include
  wrong characters, instruction failure, ambiguous/unreadable images, possible
  reference issues, truncation and uncertainty. Export the reviewed labels before
  closing; edits are not saved automatically. Exported labels retain the study
  and evidence IDs and original image hashes.

Use the paired evidence to interpret changes: 7B correcting a readable image that
3B misses supports a model-specific limitation; padding helping both supports an
input-presentation problem. Shared failures need human review to distinguish
ambiguous images, incorrect references and limitations shared by the models.
These are diagnostic clues, not causal proof. Fixed model order, quantization,
and single-pass timing limit stronger claims. Repeat shortlisted conditions and
validate choices on fresh data before reporting general performance gains.

The Ollama allocation endpoint is documented at
[List running models](https://docs.ollama.com/api/ps).
