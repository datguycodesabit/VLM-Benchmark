# Publishing this repository on GitHub

The repository is ready for a first upload. It includes source code, synthetic
tests, a locked development environment, CI, contributor templates, and the
Apache-2.0 license with upstream attribution. No GitHub remote is configured by
these instructions.

## First upload

1. Create an **empty** GitHub repository with the name and visibility you want.
   Do not initialize it with another README, license, or `.gitignore`.
2. Open a terminal in this project's root and review the files that Git will add:

   ```bash
   git status --short --untracked-files=all
   git add .
   git diff --cached --stat
   git diff --cached --check
   ```

   The staged set should contain code, tests, documentation, configuration, and
   only `.gitkeep` placeholders inside `data/`. It should not contain your IAM
   images, reference text, model files, `.env`, `.venv`, or benchmark runs.
   `.gitignore` protects the standard input/output locations; if you used a
   custom output directory, exclude it before staging.

3. Commit and push, replacing `YOUR_USERNAME` and `YOUR_REPOSITORY`:

   ```bash
   git commit -m "Initial release of local Ollama handwriting benchmark"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   git push -u origin main
   ```

   These commands assume a new repository with no existing `origin`. If your
   checkout already has one, inspect `git remote -v` and use the correct existing
   remote rather than overwriting it.

Use Git rather than dragging the whole working directory into the browser:
Git honors `.gitignore` and includes the hidden `.github` CI/template files.
The optional `dist/vlm-benchmark-github.zip` archive, if supplied, contains the
reviewed source files only. Extract it before uploading; uploading the ZIP itself
does not install the repository's workflows.

## After the first push

- Check the **Actions** tab for the CI run. The matrix checks Linux, macOS, and
  Windows; local verification alone does not establish a passing hosted matrix.
- Set the description to: "Local Ollama handwriting benchmark for IAM forms with
  CER/WER scoring and Excel, CSV, and JSONL exports."
- Suggested topics: `ollama`, `ocr`, `handwriting-recognition`, `benchmark`,
  `vision-language-models`, `python`.
- For a public repository, enable GitHub's private vulnerability reporting so
  the workflow in `SECURITY.md` is available.
- If collaborating, protect `main` with pull requests and passing CI checks.
- Once you know the final repository URL, add it to `[project.urls]` in
  `pyproject.toml` as `Repository` and `Issues`. Keep the upstream repository
  link in `NOTICE` as attribution, not as this project's support address.

## Releases

The current package version is `0.1.0`; the changelog begins with **Unreleased**.
When you decide to publish a release:

1. Confirm CI passes, move the applicable changelog entries under the version
   and release date, and ensure the version in `pyproject.toml` agrees with
   `src/vlm_bench/__init__.py`.
2. Run `uv build` and inspect the source distribution and wheel. The Python
   packages intentionally omit IAM data, local runs, and machine-specific files.
3. Commit the release changes, tag the corresponding commit (for example,
   `v0.1.0`), and create a GitHub release with release notes.

GitHub hosting does not publish the package to PyPI. This repository does not
include automatic package publishing, credentials, or deployment workflows.
