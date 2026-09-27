# Contributing

Thanks for helping improve the local Ollama handwriting benchmark. Bug reports, focused improvements, and documentation fixes are welcome.

## Development setup

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --frozen
```

Before opening a pull request, run the project checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The test suite uses synthetic fixtures and mocked Ollama responses. It does not need Ollama, benchmark models, or IAM data.

## Data and privacy

Use synthetic images, XML, references, and Ollama responses in tests and examples. Do not add IAM images, transcriptions, private dataset files, benchmark run manifests, result exports, or machine-specific paths to a commit or issue. Keep real benchmark data and generated runs local.

## Pull requests

Keep changes focused and describe the user-visible behavior they change. Add or update tests for behavior changes, and include the commands you ran in the pull request description. Preserve the existing project style and avoid adding dependencies unless the change needs them.

By submitting a contribution, you agree that it may be distributed under this repository's Apache-2.0 license.
