# CLI completion report — 0.2.0

Date: 2026-10-29. This completes the command workflow and closes the implementation
gaps found after the [2026-10-28 update](UPDATE_REPORT.md). The architecture and
product specification remain the scope reference. Three Luna Max sub-agents
reviewed and completed orchestration, scoring, and provider measurement; the main
chat integrated the CLI, comparison, diagnostics, rendering, and packaging.

## Completed behavior

| Workflow | Completed implementation |
|---|---|
| Terminal results | `run`, `report`, `compare` and resumed experiments show weighted scores, variability, placement, median latency, coverage and separate prose/equation winner decisions. Provisional or non-comparable winners remain withheld. |
| Saved comparisons | New-policy comparisons use the weighted scorer, anchored bands and joint document bootstrap. Repetitions retain their identity. Matching named conditions, frozen datasets, pass counts and policies are validated. Mixed legacy/new sources need explicit `--weighted` or `--scoring-config`. |
| Before-first-pass recovery | Direct experiments persist an integrity-checked plan and portable prepared snapshot before inference. A saved plan can resume even with no registered run. A committed run missing from the coordinator index is discovered rather than repeated. |
| Suite recovery | Orphaned committed passes are discovered; paths and crop locations survive moving the suite; model identity, controls and capabilities are checked before continuing. `resume --run SUITE` uses its saved configuration; `--config` selects a relocated file. |
| Status | `status` understands individual runs, repeated experiments and suites. It includes planned/started/completed passes, expected and missing work, per-condition coverage and cumulative counters. |
| Strict research | Repeated CLI runs default to the test split while retaining the audit of the entire source dataset. Every pass uses the same frozen selection. |
| Provider timing | Optional Ollama `streaming=true` collects client TTFO and response-stream duration while returning a complete buffered prediction. Partial streams are rejected. Empty cloud text deltas do not trigger TTFO. TTFO is never called prefill. |
| Usage/rates | Ollama and TrOCR report observed counts and tokenization provenance. A token rate is emitted only with a valid count and positive matching generation duration. Missing counts, tokenizer identity or cloud internals remain unknown. |
| Mathematical checks | Bounded exact rational-polynomial proofs handle identities such as commutation and expansion. Unsupported notation, undefined expressions, ambiguous `0^0`, excessive complexity and inconclusive sampling remain undetermined. Equivalence remains separate from fidelity. |
| Eligibility | All-pass declared unsupported work is excluded; partially unsupported work cannot disappear from the scoring denominator or create a final winner. |
| Offline policy changes | `report --scoring-config` explicitly derives an evaluation from a v3 policy. Raw-evidence IDs stay separate from policy-specific evaluation IDs and rescore provenance. |
| Imported predictions | New CLI imports freeze the default or configured weighted policy and generate the common report. Historical Python import behavior stays compatible. External scalar timing without controlled measurement provenance cannot decide speed winners. |
| Portable derived reports | Saved comparison/evaluation reports can be rendered after original inference folders move or disappear. Image assets are copied and hash checked. Policy changes still require original evidence. |
| Report formats | HTML, workbook and CSV diagnostics use the same report model. Extra CSV tables cover passes, operations, timings, usage, equivalence and uncertainty where available. Long spreadsheet text is preserved through overflow sheets; text remains escaped/protected. |
| CLI contracts | `--version`, structured `doctor` checks including custom Python devices, useful JSON results/report paths, explicit suite override errors and shell-quoted setup instructions are implemented. |
| Packaging | Project, runtime and lockfile versions are aligned at `0.2.0`. The wheel and source archive are built under `dist/final/`. |

## Use the completed CLI

```bash
vlm-bench --version
vlm-bench doctor --provider ollama
vlm-bench init --prepared data/prepared --models ollama:MODEL --output benchmark.toml
vlm-bench suite --config benchmark.toml --output runs/experiment
vlm-bench status --run runs/experiment
vlm-bench resume --run runs/experiment
vlm-bench report --run runs/experiment
vlm-bench compare --runs runs/experiment-a runs/experiment-b --output comparisons/a-b
```

For token timing with an Ollama model, edit the ordinary model settings:

```toml
[models."ollama:MODEL"]
streaming = true
```

To deliberately apply a different scoring policy to saved raw predictions:

```bash
vlm-bench report --run RUN --scoring-config experiment-v3.toml --output reports/rescored
vlm-bench compare --runs A B --weighted --output comparisons/derived
```

Policy changes preserve provider output and historical results. A derived report
has its own evaluation identity linked to the raw evidence identity. Re-rendering
an already derived report preserves its evaluation rather than pretending to
perform a new scoring pass.

## Verification

- **460 tests passed**, including recovery, frozen identity, joint comparisons,
  policy changes, unsupported coverage, polynomial safety, streaming, token
  provenance, diagnostics, artifact portability, escaping and workbook preservation.
- Ruff lint and format checks passed; `git diff --check` passed.
- `uv lock --check --offline` passed without dependency drift.
- Wheel and source distribution build successfully; the wheel installs and runs
  outside the checkout in an isolated temporary environment.
- A real installed-executable HTTP check processed **1,000 images × 2 fixture
  models × 3 passes = 6,000 terminal results**, plus **6 warmups**. It exercised
  a budget pause and checkpoint resume, then generated all reports after the
  fixture server stopped. Total observed elapsed time was **58.05 seconds** on
  macOS ARM64/Python 3.12.14. These are synthetic workflow timings, not model speed.
- The HTTP workload exposed repeated raw timing expansion and a quadratic
  workbook-cell lookup. Reports now retain bounded per-metric timing summaries,
  while full raw observations stay in `results.jsonl`; report model version 2
  identifies the changed derived representation. A 1,000-observation regression
  fixture verifies bounded report size. Workbook writes access known columns
  directly rather than rescanning every populated cell per row.

Evidence: `runs/cli-completion-http-20261029-verified/validation.json`; the generated
HTML/workbook/CSV/JSON files are in its `offline-report/` directory.

The reproducible commands are:

```bash
python -m pytest -q
ruff check .
ruff format --check .
uv lock --check --offline
uv build --offline --out-dir dist/final

python scripts/validate_cli.py --cli /path/to/installed/vlm-bench \
  --output runs/cli-http-validation --images 1000
```

The HTTP validation runs the installed executable as a subprocess against a
local synthetic Ollama-compatible server. It exercises actual HTTP requests,
NDJSON streaming, configuration, request-budget pause, three-pass resume,
measurements, JSON output, aggregate status and report generation after the
server shuts down. Fixture predictions come from an encoded pixel index; this
is a workflow check and does not perform handwriting recognition.

## Validation boundary

The code implements CPU/MPS/CUDA/ROCm selection and synchronization, but this
workspace has neither PyTorch/Transformers nor an OpenAI API key, and no live
Ollama server was running. Real GPU execution, checkpoint quality and credentialed
cloud service reliability therefore remain unmeasured. The local HTTP fixture
is explicitly synthetic. Windows, hosted custom endpoints, distributed workers,
true batching, standalone chart exports and slide generation remain outside this
release; unsupported batch sizes fail before inference. Conservative equation
checks deliberately return undetermined for notation beyond the supported grammar.
