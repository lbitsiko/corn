# Developer Guide

This guide is for contributors working directly in CORN and for developers using coding agents to extend it.

## Working model

Treat CORN as:

- a project-driven Streamlit application
- a UI and session orchestrator in `app.py`
- backend modules under `corn/`
- app-wide project/provider discovery files under `data/`
- project-local configuration, runtime cache, annotations, and LLM state
- an optional worker process for queued LLM annotation
- a portable desktop launcher that supervises the UI and worker

Backend behavior is divided by lifecycle concern, but the UI workflow remains centralized in `app.py`. Read `docs/architecture.md` before changing persistence or cross-process behavior.

## Development setup

CORN supports Python 3.11 and newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Run the UI from the repository root:

```bash
python -m streamlit run app.py
```

Run the worker in a separate process when testing LLM jobs:

```bash
python -m corn.llm_worker \
  --registry-path data/project_registry.yaml \
  --provider-settings-path data/provider_settings.json \
  --worker-id worker-1
```

The UI and worker must use the same checkout and data directory. Provider API keys must be present in both process environments; `provider_settings.json` stores non-secret settings only.

Run the desktop launcher from source:

```bash
python corn_desktop.py
```

The launcher sets `CORN_DATA_DIR` to the platform user-data directory. For isolated development or smoke testing, set it explicitly:

```bash
CORN_DATA_DIR="$(mktemp -d)" python -m corn.desktop --internal-smoke-test
```

## Portable build entry points

Desktop packaging is organized around:

- `corn/app_paths.py` for packaged resources and writable state
- `corn/desktop_runtime.py` for commands, health checks, and process supervision
- `corn/desktop.py` for internal UI, worker, smoke-test, and launcher dispatch
- `corn/launcher.py` for the Tkinter launcher
- `corn/packaging_manifest.py` for the resource allowlist
- `packaging/corn.spec` for PyInstaller
- `packaging/verify_bundle.py` for frozen smoke and secret-leak checks
- `.github/workflows/portable-build.yml` for native release artifacts

Build locally on the target operating system:

```bash
python -m pip install -r requirements-build.txt
pyinstaller --clean --noconfirm packaging/corn.spec
python packaging/verify_bundle.py dist/CORN/CORN
```

Use `dist/CORN/CORN.exe` on Windows. Build the complete `onedir` folder; distributing only its top-level executable will not work. See [Portable CORN Distribution](portable-distribution.md) for release verification and platform instructions.

## Change entry points

### Guided setup and project lifecycle

Use these files for project creation, config upload, source inspection, copying/reference behavior, cache status, or registration:

- `app.py`
- `corn/source_inspection.py`
- `corn/project_config.py`
- `corn/projects.py`
- `corn/registry.py`

Guided setup and uploaded YAML/JSON ultimately pass through the same canonical project-config validator.

### Source ingestion and runtime cache

Use these files for CSV/SQLite normalization, canonical document columns, or cache metadata:

- `corn/ingestion.py`
- `corn/projects.py`
- `corn/project_config.py`

A copied source is stored under the project; a referenced source remains at its configured path.

### Querying and document coverage

Use these files for filter behavior, document retrieval, human annotation status, or LLM coverage:

- `corn/querying.py`
- `corn/storage.py`
- `corn/llm_storage.py`
- `app.py`

### Human annotation and evidence

Use these files for editor behavior, canonical payloads, span capture, note-only evidence, or pending-save state:

- `app.py`
- `corn/annotations.py`
- `corn/components/document_selector/`
- `corn/storage.py`

`corn.annotations.save_annotation` writes JSON; `corn.storage.save_annotation_record` maintains the SQLite index.

### Tie attributes

Use these files for attribute schema definitions, editor widgets, validation, LLM output, import behavior, or agreement metrics:

- `corn/project_config.py`
- `corn/attribute_validation.py`
- `app.py`
- `corn/llm_prompts.py`
- `corn/llm_validation.py`
- `corn/annotation_io.py`
- `corn/reporting.py`

Do not duplicate value rules in a UI or provider module. `validate_tie_attributes` is the shared source of truth for categorical, numeric, and temporal values.

### Annotation portability and exports

Use these files for document-level JSON import/export or workspace CSV/JSON exports:

- `corn/annotation_io.py`
- `corn/storage.py`
- `corn/annotations.py`
- `app.py`

Imports validate document identity, tie labels, and tie attributes before writing. Workspace exports are built from indexed records; LLM export also consults valid LLM runs so older runs are not omitted.

### LLM-assisted annotation

Use these files for prompts, providers, validation, queue execution, persistence, or worker behavior:

- `corn/llm_prompts.py`
- `corn/llm_providers.py`
- `corn/provider_settings.py`
- `corn/llm_validation.py`
- `corn/llm_storage.py`
- `corn/llm_jobs.py`
- `corn/llm_logging.py`
- `corn/llm_worker.py`
- `app.py`

Provider adapters handle transport and provider response extraction. Project-specific semantic checks remain in CORN validation code.

### LLM reporting

Use these files for human/LLM scope selection, directionality, document coverage, tuple metrics, confusion matrices, or attribute agreement:

- `corn/reporting.py`
- `corn/llm_storage.py`
- `corn/storage.py`
- `app.py`

The selected human scope defines the evaluation population. LLM-only documents are excluded, while a UI control determines whether human documents without valid LLM runs remain in scope.

## Git and worktree conventions

For isolated feature work, prefer a worktree based on the current integration branch:

```bash
git worktree add .worktrees/my-feature -b feature/my-feature dev
```

This keeps the main checkout available, isolates feature state, and makes review and cleanup easier. If a user explicitly asks for direct work on the current checkout, inspect tracked and untracked state before editing and preserve unrelated files.

Use concise conventional commit messages. Do not rewrite shared branch history.

## Testing expectations

Before claiming a change is ready:

1. Run focused tests for the affected modules.
2. Run the full suite:

   ```bash
   pytest -q
   ```

3. Run a UI smoke test when `app.py` behavior changes.
4. When project setup changes, test both CSV and SQLite paths where relevant and cover copied and referenced sources.
5. When annotation or attribute behavior changes, test human save, import, LLM validation, and reporting boundaries as applicable.
6. When LLM workflow changes, verify queueing, worker execution, failure persistence, UI visibility, and duplication into the human editor.
7. When desktop runtime or packaging changes, run `python -m corn.desktop --internal-smoke-test`, build with PyInstaller, and run `packaging/verify_bundle.py` against the frozen executable.

The repository includes focused tests for project config, guided setup, ingestion, annotation storage, portability, providers, worker execution, and reporting.

## Documentation maintenance

When behavior or architecture changes:

- update `README.md` for user-facing workflow or configuration changes
- update `docs/architecture.md` for component, persistence, or data-flow changes
- update this guide when contributor entry points or verification commands change
- update all relevant TikZ sources under `docs/`
- regenerate tracked PDFs with `pdflatex`
- regenerate `docs/architecture-flowchart-dev.png` from its PDF at 150 DPI
- refresh screenshots when a visible workflow changes materially

Run diagram generation from `docs/`:

```bash
pdflatex architecture-flowchart-dev.tex
pdflatex architecture-flowchart.tex
pdflatex architecture-flowchart-simple.tex
pdftoppm -png -r 150 -singlefile \
  architecture-flowchart-dev.pdf architecture-flowchart-dev
```

Do not commit LaTeX auxiliary files such as `.aux` or `.log`.

## Common pitfalls

- Queued LLM jobs do not complete without a running worker.
- The UI and worker must see the same registry, project paths, provider settings, and API-key environment variables.
- Streamlit widget identity and render order can make session-state changes subtle.
- Changing a tie-attribute schema can make persisted values unknown or invalid; loaders and reports must remain defensive.
- Annotation export behavior depends on both payload shape and SQLite index/run data.
- Reference-mode projects depend on external source paths that must remain available after deployment or migration.
- Reports are computed on demand and are not stored as durable report artifacts.

## Documentation still worth adding

The current tracked docs do not yet provide:

- a complete annotation and LLM SQLite schema reference
- a dedicated troubleshooting guide for providers, workers, stale caches, and span selection
- a committed production deployment/operations guide
