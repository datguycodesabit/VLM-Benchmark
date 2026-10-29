# VLM Benchmark CLI update report

Date: 2026-10-28. Implemented against [ARCHITECTURE.md](ARCHITECTURE.md) and
[PRODUCT_SPEC.md](PRODUCT_SPEC.md). Three sub-agents used `gpt-6-luna` with
`max` reasoning for scoring/report evaluation, devices/adapters, and orchestration.
Integration, renderer work, documentation, and release checks were performed in
the primary chat. Existing README edits and the two design documents were preserved.

The implementation is ready for local review and automated workflow testing.
Live cloud and accelerator certification remain outstanding; this report does
not claim a hardware-validated production release.

## Updates delivered

| Area | Changes | Main implementation |
|---|---|---|
| Weighted accuracy | Frozen/versioned 70% character accuracy + 30% exact match defaults; configurable weights and 0.5-point margin; corpus character weighting; raw CER retained above 100%; WER diagnostic | `scoring_policy.py`, `report_model.py` |
| Failure accounting | Missing/failed predictions contribute reference-character deletions and zero exact matches; expected rosters and coverage remain visible; unsupported capabilities produce exclusions | `report_model.py`, `engine.py` |
| Track ranking | Independent prose/equation scores and winners; word samples included with prose; anchored inclusive bands; pooled successful uncached sequential median request latency; deterministic score/model residual ties; raw accuracy order | `ranking.py`, `report_model.py` |
| Winner gating | Incomplete passes, incompatible eligible rosters, and missing comparable timing withhold appropriate final placement/winner claims | `ranking.py`, `report_model.py` |
| Repetitions | New direct CLI experiments and omitted suite repeats default to three passes; physical runs retain distinct IDs/repetition numbers; all passes reuse one frozen snapshot | `cli.py`, `suite.py`, `experiments.py` |
| Recovery | Resumable direct experiments, run and orchestration locks, immediate run-path persistence, frozen crops/controls/model identity checks, portable relative paths, and continuation of missing passes | `experiments.py`, `engine.py`, `suite.py` |
| Retry/cache controls | Operational retries remain separate from passes; valid wrong transcriptions never select retries; measurement/repeated runs bypass response caching; warmups remain separate and consume operational allowances | `engine.py`, `suite.py` |
| Configuration | Version 3 adds explicit scoring and measurement sections; legacy v1/v2 parsing and scoring remain available; unknown/unsupported versions and settings fail clearly | `config.py`, `engine.py` |
| Init wizard | Interactive terminal prompts or scriptable `--prepared`, `--models`, `--repeats`; creates editable, validated TOML; preserves existing files | `init.py`, `cli.py` |
| Device layer | Explicit CPU/MPS/CUDA/ROCm selection; accelerator availability failures; recorded auto policy; synchronization; runtime/device identity and memory diagnostics; HIP distinguished from NVIDIA CUDA | `devices.py`, `backends.py` |
| Custom local models | `python:module:Factory` lifecycle for loading, preprocessing, inference and decoding; capabilities checked before execution; image/prompt/controls supplied without references; adapter source fingerprint and optional checkpoint identity | `adapters.py`, `backends.py` |
| Timing evidence | Normalized metric/value/unit/source/boundary/availability records; local preprocessing/load/inference/decode timings; provider timing provenance; unavailable internals stay null | `measurement.py`, `backends.py`, `engine.py` |
| Latency/throughput | Sequential latency ranking separated from throughput experiments; explicit concurrency and batch-size controls; elapsed workload wall-time denominators; resumed rates use cumulative workload time; batch sizes above one fail before inference | `config.py`, `engine.py`, `report_model.py` |
| Uncertainty | Joint source-document bootstrap preserves pass structure; per-model and paired intervals; unavailable document metadata is labeled; mean, standard deviation, range, and terminal/completed pass counts | `report_model.py` |
| Equation equivalence | Bounded safe arithmetic parser, exact rational counterexamples under explicit variable-domain values, normalized identity proof, and `undetermined` for unsupported/inconclusive cases; outcome/attempted/applicable coverage never feeds fidelity | `equivalence.py`, `report_model.py` |
| Offline reports | One saved-evidence evaluation model renders HTML, Excel, CSV and JSON; independent suite conditions; accuracy/latency plots, per-pass evidence, operational counts, uncertainty, device/provider context, and difficult frozen-image examples | `report_model.py`, `reporting.py` |
| Export integrity | Escaped HTML; portable report image assets with hash verification; spreadsheet formula protection; long-cell overflow preservation; workbook leaderboard, pass, operational, timing and full-evidence sheets | `reporting.py`, existing `export.py` helpers |
| Compatibility | Legacy CER metrics and exports are not rewritten. Runs lacking a weighted policy require explicit `report --weighted` to derive a separate default-policy evaluation. Native/imported predictions use the same offline evaluation pipeline | `report_model.py`, `cli.py` |
| Documentation | New CLI guide, updated suite example, runnable custom-adapter example, changelog, and reproducible synthetic workflow validation script | `docs/CLI_GUIDE.md`, `examples/`, `scripts/validate_workflow.py` |

## CLI entry points

```bash
vlm-bench init --prepared data/prepared --models ollama:MODEL openai:MODEL \
  --output benchmark.toml
vlm-bench suite --config benchmark.toml --output runs/comparison --dry-run
vlm-bench suite --config benchmark.toml --output runs/comparison
vlm-bench suite --config benchmark.toml --output runs/comparison --resume

# Direct runs default to three passes; legacy v1/v2 direct configs retain one.
vlm-bench run --prepared data/prepared --models ollama:MODEL
vlm-bench resume --run runs/EXPERIMENT_DIRECTORY

vlm-bench report --run runs/comparison
vlm-bench report --run LEGACY_OR_IMPORTED_RUN --weighted
```

`report` writes `report/report.html`, `report/report.xlsx`,
`report/leaderboards.csv`, and `report/evaluation.json` by default.
`--output` selects another directory. Existing `report --group-by` and `export`
commands retain their previous purposes. `--json` remains machine-readable;
report paths include the complete report files.

Budgets apply per engine invocation, including warmups and operational retries.
Three passes can therefore consume three allowances. A pause stops further pass
scheduling. If execution fails before the first run is saved, rerun the original
command: no saved first-pass identity exists to resume.

## Verification evidence

- Baseline before implementation: **370 tests passed**.
- Integrated suite after implementation: **423 tests passed**, including policy,
  failures/missing coverage, ranking, equation-domain safety, device simulation,
  custom-adapter reference isolation, cache exclusion, frozen identity, recovery,
  source changes, generated v3 configurations, repeated suite execution and
  resume without remeasurement, matching report values, escaped output, and
  spreadsheet overflow/formula protection.
- `ruff check .` and `ruff format --check .` passed. The optional pinned MathText
  smoke test passed; its existing upstream PyParsing deprecation warnings remain.
- Source distribution and wheel built successfully. Build verification uses the
  standard setuptools backend. Final artifacts are placed in `dist/final/`.
- The wheel installed in a fresh temporary environment. Installed `--json --help`,
  `init`, and offline `report` commands passed outside the checkout.
- HTML inspected in the in-app browser, including leaderboards, plots, diagnostic
  disclosures, and example images. Workbook verification checks sheet structure,
  score agreement, protected text, and overflow preservation; no Excel desktop
  visual certification was performed.

The reproducible synthetic workload command is:

```bash
python scripts/validate_workflow.py --output runs/validation-1000-20261028 --images 1000
```

This run completed **1,000 synthetic images × 2 fixture models × 3 passes**, yielding
**6,000 durable terminal results and 6,000 backend calls**. Both tracks were
finalizable and all four report formats were generated. On the observed
macOS ARM64/Python 3.12 environment, preparation took about **0.95 seconds**,
execution **4.37 seconds**, and initial report generation **3.69 seconds**
(**8.97 seconds total**). These times describe fixture orchestration and file
handling, not model inference or handwriting recognition speed. The fixture
backend returns deterministic text without calling a live provider.

Saved evidence is under `runs/validation-1000-20261028/validation.json`; the
preview is `runs/validation-1000-20261028/report/report.html`. Report regeneration
is offline and does not add inference calls.

## Platform and provider matrix

| Configuration | Evidence in this workspace | Release limitation |
|---|---|---|
| macOS ARM64 host, Python 3.12 | CLI, file operations, tests, package builds, synthetic workload | Real PyTorch model inference not run |
| CPU PyTorch | Device and adapter simulations; lifecycle integration | PyTorch/Transformers are not installed in this workspace environment |
| macOS MPS | Selection, unavailable/fallback rules, synchronization simulations | Actual MPS model execution unvalidated |
| Linux NVIDIA CUDA | Explicit selection and runtime identity simulations | Actual Linux/CUDA execution unvalidated |
| Linux AMD ROCm | HIP identification and CUDA-facing API simulations | Actual Linux/ROCm execution unvalidated; hardware support depends on the specific GPU/runtime |
| Ollama / OpenAI / eligible ChatGPT | Existing and extended mocked-provider tests | No live cloud requests or real Ollama inference performed |
| Custom Python adapter | Real engine orchestration with mocked torch/adapter lifecycle | Example checkpoint not downloaded or executed |

## Remaining validation and explicit limits

1. Run actual models on the intended CPU/MPS/CUDA/ROCm machines and publish
   measured versions, precision, hardware identity, synchronization, memory use,
   and performance. Device detection is not hardware certification.
2. Run credentialed provider smoke tests and interruption/quota cases against live
   services. Existing mocks do not prove live reliability or account eligibility.
3. Measure approximately 1,000 real handwriting images with representative models;
   the synthetic workflow establishes capacity of the tested file/orchestration
   path only.
4. Time to first output and cloud prefill/decode internals remain unavailable when
   no instrumentation/provider observation exists. Streaming timing is not added
   to every provider by this update.
5. True batched inference is not implemented. Batch sizes greater than one are
   rejected. Supported cloud concurrency provides separately labeled throughput.
6. Equivalence is intentionally conservative: general LaTeX, symbolic rewrites,
   unrestricted domains and incomplete proofs produce undetermined. Correctness
   of equations should not be inferred from visual similarity.
7. Adapter source hashes do not cover transitive dependencies; adapters must
   supply an immutable revision/checkpoint hash to detect changed weights.
8. Imported scalar timings without an explicit sequential measurement profile
   remain diagnostic and cannot resolve speed-based winners.
9. A provider response followed by a crash before durable commit can be repeated
   on resume; local artifacts cannot guarantee exactly-once remote billing.
10. Unknown cost remains unknown in presentation. Internal operational allowance
   accounting for local models is not an electricity/hardware cost measurement.
11. Windows, generic hosted custom-model endpoints, distributed workers, standalone
    chart exports and slide generation remain outside the initial scope.
