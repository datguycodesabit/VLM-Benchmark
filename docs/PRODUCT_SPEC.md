# Benchmark product specification

Product decisions from the researcher interview on 2026-10-28. This document
describes the intended product; it does not claim that these features are all
implemented. See IMPLEMENTATION_STATUS.md for the existing implementation.

## Purpose and audience

Build a terminal-first handwriting recognition benchmark for researchers who
are comfortable with command-line tools. Researchers choose a dataset and
models, run a reproducible comparison, and receive clean results suitable for
presentation.

The benchmark answers which model transcribes handwriting best, with detailed
speed diagnostics. Target workloads contain up to approximately 1,000 images,
primarily individual prose lines and equations. This is a design target, not a
measured capacity guarantee.

## Agreed product requirements

- Provide separate prose and equation leaderboards and winners.
- Rank primarily by a configurable weighted accuracy score. Use speed only
  within a configurable practical accuracy margin; show uncertainty separately.
- Default to 70% character accuracy and 30% exact-match rate. Report WER
  separately.
- Calculate character accuracy across all reference characters, so longer
  references contribute more to that component.
- Default the practical accuracy margin to 0.5 points on the 0–100 score.
- Use median end-to-end latency per image as the speed tie-breaker, measured
  with one request at a time.
- Penalize missing and failed predictions in the accuracy score.
- Retry operational failures only. Incorrect but valid transcriptions stand;
  never select retries using their agreement with the reference.
- Execute three full passes by default; retain distinct repetition identities.
- Report equation transcription fidelity and mathematical equivalence
  separately. Fidelity determines the primary equation ranking.
- Initially use conservative automated equivalence checks with three outcomes:
  equivalent, different, and undetermined.
- Keep cloud providers essential to the product.
- Support built-in local TrOCR checkpoints and a documented Python adapter for
  other local PyTorch models. Custom local models are the initial extension
  priority; generic hosted custom-model endpoints are deferred.
- Provide an optional `init` wizard that generates editable configuration.
- Deliver polished HTML reports plus Excel and CSV exports. Standalone chart
  exports and slide generation are outside the agreed initial scope.

## Agreed platform scope

The first release targets macOS and Linux:

- macOS: PyTorch MPS acceleration where supported, plus CPU execution.
- Linux with NVIDIA GPUs: PyTorch CUDA.
- Linux with supported AMD GPUs: PyTorch ROCm.
- CPU execution remains available as a portable baseline.

Validate specific hardware, operating-system, and runtime combinations.
Supporting AMD GPUs does not mean every AMD GPU supports ROCm. Windows support
is a later phase with explicitly validated configurations.

The current TrOCR adapter/configuration accepts `auto`, `cpu`, and `mps`.
NVIDIA/AMD device selection and validation still require implementation.

## Recommended scoring details

The following operational details are recommendations to implement and validate;
they were not individually selected during the interview.

For each track and repetition:

```text
corpus_CER = sum(character_edits) / sum(reference_characters)
exact_match_rate = exact_matches / all_eligible_samples
score = 70 * max(0, 1 - corpus_CER) + 30 * exact_match_rate
```

Treat a missing or failed prediction as an empty transcription: count deletion
of its reference characters and no exact match. Reject empty references during
dataset validation. Keep raw CER visible even above 100%; clamp only the
character-accuracy component of the composite score. Preserve the configured
normalization/scorer version with the run.

Average the three per-pass scores and report repetition variability. Compute
the latency tie-breaker from successful measured requests, while separately
showing failure rate, retry counts, and retry-inclusive completion time.

For deterministic placement, form accuracy bands anchored at the highest
remaining score. Include models within the configured margin below that score,
order the band by median latency, then repeat for remaining models. Display raw
accuracy order as well as final placement. Define a stable final ordering for
identical scores and latencies. Version and freeze these rules before test runs.

Equivalence checks must respect expression domains and assumptions. Parser
failures, unsupported notation, or inconclusive checks return undetermined;
failure to prove equivalence does not establish that expressions differ.
Report outcome counts and evaluation coverage separately from fidelity.

## Recommended architecture

Extend the existing Python implementation and preserve portable run artifacts.

1. Shared device layer: device detection/selection, accelerator synchronization,
   runtime identity, memory diagnostics, and explicit fallback behavior. Do not
   silently substitute CPU for an explicitly requested GPU benchmark.
2. Backend capability contract: supported tasks, streaming, batching, available
   timing measurements, and model identity. Custom adapters implement loading,
   preprocessing, inference, output conversion, and optional timing hooks.
3. Execution orchestration: separate scheduling, retries, budgets, persistence,
   and scoring without weakening checkpoint/resume behavior.
4. Versioned scoring policy: weights, normalization, failure handling, accuracy
   margin, repetition aggregation, and tie rules frozen in run metadata.
5. Reporting pipeline: one shared report model for native, cloud, custom-adapter,
   and imported results. Keep report generation usable without new inference.

Do not introduce distributed workers or require a database for this workload
without profiling evidence. A rebuildable experiment index can be added later
if browsing saved runs becomes cumbersome.

## Measurement and reporting contract

Record end-to-end latency, median and tail latency, and workload throughput.
Where measurable, also record time to first output, input/prefill duration,
decode duration/rate, image preprocessing/encoding, and cold-start overhead.

Every timing field must identify whether it is client-measured,
provider-reported, adapter-instrumented, or unavailable. Time to first output
must not be labeled pure prefill time. Do not infer unavailable cloud internals
or assume backend timing boundaries are interchangeable. Explain tokenizer
differences when presenting tokens per second across models.

Use a controlled latency profile for ranking; provide separately labeled
throughput experiments with explicit concurrency and batch size. Record warmup
policy, precision, device, runtime, model revision, and cache behavior. Exclude
cached results from measured inference speed and repeated timing experiments.

Speed comparisons describe the model and its execution environment. Preserve
hardware and provider context prominently, especially across machines.

The HTML report should contain:

- Separate prose/equation leaderboards, formulas, tie margins, and coverage.
- Accuracy-versus-latency views and available timing breakdowns.
- Per-pass results, variability, and document-based uncertainty.
- Equation equivalence outcomes and undetermined coverage.
- Failure/retry counts, observed usage, and known/unknown cost estimates.
- Difficult examples with images, references, predictions, and edit details.
- Dataset fingerprint, configuration, environment, and reproducibility details.

## Suggested implementation sequence

1. Implement/version the scoring policy and verify failure penalties and ranking
   behavior using meaningful fixtures.
2. Extend device selection and the custom PyTorch adapter contract; validate
   actual execution on MPS, CUDA, and ROCm hardware.
3. Standardize timing fields and add streaming/instrumentation where supported.
4. Integrate three-pass defaults, ranking policy, and the `init` wizard.
5. Build the presentation-ready HTML/Excel/CSV reporting experience.
6. Add conservative equation equivalence evaluation with explicit coverage.
7. Validate a 1,000-image workflow, interruption/resume, and live cloud/provider
   behavior. Publish the tested platform matrix and measured limitations.

Existing mocked tests are not evidence of live-provider reliability or
cross-platform GPU support. Hardware-dependent verification must be completed
on the relevant hardware before claiming a supported configuration.
