# Implementation status

## Phase 1 — Research eligibility and leakage

Status: complete.

- Added `run --strict-research`, `run --protocol`, and `dataset split --protocol`, including validated TOML defaults and CLI overrides.
- Exposed protocol and validation scope in research and comparison reports; dataset checks print bounded integrity findings.
- Verification: phase gate passed with 215 tests plus lint and format checks.

## Phase 2 — Error inspection and subgroup reports

Status: complete.

- Added `inspect --run RUN --worst N` for terminal summaries and escaped local HTML review with frozen images/references, bounded edit alignments, and heuristic flags.
- Added `report --run RUN --group-by FIELD` with eligibility exclusions, prediction coverage, sample/document counts, and an explicit unknown group.
- Verification: phase gate passed with 227 tests plus lint and format checks.

## Phase 3 — Math and page evaluation

Status: complete.

- Added the opt-in `--formula-rendering` CLI/config setting and frozen-annotation offline rescoring. Rescoring verifies schema-v2 manifest integrity, snapshot fingerprint, and model/sample result pairs before replacing results.
- Added task-metric CSV and workbook output. Comparisons retain saved task metrics and show per-run scorer settings and aggregates without introducing task-metric rankings or mixing renderer options.
- Verification: phase gate passed with 250 tests, Ruff lint/format, and package build; the pinned MathText renderer smoke test passed.

## Phase 4 — External predictions and OCR baselines

Status: complete.

- Added `vlm-bench import` for validated prepared snapshots and JSONL predictions with required external system/provenance, frozen-reference scoring, finite timing/usage checks, and incomplete-coverage handling.
- Imported runs use normal schema-v2 artifacts and work with inspection, export, rescoring, and comparison. Added reproducible Tesseract and PaddleOCR example generators with provenance sidecars.
- Verification: phase gate passed with 274 tests plus lint/format checks; imported/native scoring equivalence and incomplete unranked coverage are tested.

## Phase 5 — Experiment variants and repetitions

Status: complete.

- Added version-2 TOML suites with named, validated conditions, model-control overrides, custom prompts, and per-condition repetitions.
- Added preview and resumable `suite` orchestration with a signed suite manifest, immediate persistence of engine run paths, matching benchmark fingerprints, native run resumption, repeated CER variability, independent sample/document counts, per-run bootstrap records, and same-snapshot comparisons.
- Suite execution controls and frozen run identity are checked before resuming; completed runs are reused, incomplete runs continue in place, and repeated measurements retain distinct run identities.
- Verification: phase gate passed with 304 tests plus lint and format checks.

## Phase 6 — Execution and automation

Status: complete.

- Added bounded retries for classified transient provider failures, provider retry-delay handling, attempt logs, and opt-in concurrency for supported cloud providers.
- Added per-invocation request and estimated-spend limits, preserving completed predictions for resume, and read-only `status --run` summaries with current and cumulative totals.
- Added opt-in response caching keyed by input/model/prompt/effective controls; repeated and stochastic measurements bypass it, and cache hits are excluded from provider usage and inference-latency summaries.
- Added consistent `--json` command output with progress on standard error.
- Verification: the final phase gate passed with 370 tests and Ruff lint/format clean across 57 files. The pinned MathText renderer smoke passed and emitted upstream PyParsing deprecation warnings. OCR generators were tested with mocks; real Tesseract/PaddleOCR inference and live cloud calls were not run.

## Final verification notes

- The full test suite passed with 370 tests. `ruff check .` and `ruff format --check .` passed.
- The optional formula renderer is pinned to Matplotlib 3.10.3; its real MathText smoke test passed.
- Tesseract/PaddleOCR examples have mocked integration tests only. Cloud execution was tested with mocked providers; live cloud inference was not run.
- Spend caps are estimates applied per invocation, and concurrent in-flight requests may overshoot them. Strict validation on a prepared snapshot covers only its frozen samples and cannot establish source-wide train/validation leakage status.
- Artifact readers remain backward-compatible with existing version-2 manifests; execution controls and protocol fields are additive, task metrics, `execution.json`, and cache entries use version 1, and experiment suites use TOML version 2.
- Final packaging passed: offline build produced both wheel and source distribution, the final wheel installed in a fresh temporary environment, and `vlm-bench --json --help` worked outside the checkout. Archive inspection confirmed documentation, suite configuration, and OCR examples are in the source distribution; runtime modules are in the wheel; local data, runs, and environment files are excluded.
