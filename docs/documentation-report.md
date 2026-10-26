# Documentation Status Report

**Reviewed:** 2026-09-26

**Code baseline:** `dev` at `fd7a211`

## Purpose

This is a living status report for CORN's tracked user, contributor, and architecture documentation. It records what is authoritative, what was verified against the code, and what remains intentionally unresolved.

## Authoritative tracked documentation

### User and operator overview

`README.md` is the primary entry point. It covers:

- installation and local execution
- UI/worker process separation
- provider configuration and environment variables
- project creation and annotation workflow
- LLM jobs and reporting
- tie-attribute configuration and behavior
- exports and document portability
- code structure and documentation links

### Architecture

`docs/architecture.md` is the detailed component and data-flow reference. It covers:

- app-wide registry and provider settings
- copied versus referenced project sources
- runtime cache and stale-cache metadata
- scoped annotation JSON and project-local SQLite state
- human annotation and portability boundaries
- queued LLM execution and failure persistence
- human-defined LLM evaluation subsets
- tie-attribute validation and agreement reporting

### Contributor workflow

`docs/developer-guide.md` maps common changes to their implementation files and documents local setup, worker execution, worktree conventions, testing expectations, documentation maintenance, and common pitfalls.

### Visual architecture

Tracked architecture sources and generated artifacts are:

- `docs/architecture-flowchart-dev.tex`
- `docs/architecture-flowchart-dev.pdf`
- `docs/architecture-flowchart-dev.png`
- `docs/architecture-flowchart.tex`
- `docs/architecture-flowchart.pdf`
- `docs/architecture-flowchart-simple.tex`
- `docs/architecture-flowchart-simple.pdf`

The developer/operational PNG is embedded in README. The TeX source is authoritative; generated PDF and PNG files must be regenerated whenever it changes.

## Current screenshots

README uses these tracked screenshots:

- `docs/screenshots/startup.png`
- `docs/screenshots/query-results.png`
- `docs/screenshots/document-view.png`
- `docs/screenshots/annotation-editor.png`
- `docs/screenshots/LLM-annotation-panel.png`

These screenshots were captured on 2026-06-04. They remain illustrative but are not a complete representation of the current UI: they predate native Gemini, tie-attribute editing, attribute agreement, and guided tie-attribute setup. Some also display local development paths. They should be replaced in a dedicated capture pass using sanitized demo data and paths.

## Verification performed in this review

The 2026-09-26 review checked documentation against the implementation rather than relying only on prior prose.

Completed checks:

- all local Markdown links resolve
- all three TikZ sources compile with `pdflatex`
- worker CLI options match documented commands
- project source copy/reference behavior matches architecture text
- annotation JSON paths match scoped storage behavior
- reporting text matches the human-defined evaluation subset implemented by `corn/reporting.py`
- architecture includes the shared tie-attribute validator and current LLM/reporting paths
- full Python test suite passes

## Known follow-up work

### Screenshot refresh

Capture current, sanitized versions of startup, query results, document view, tie-attribute editing, LLM annotation, and reporting. Avoid absolute home/worktree paths in published images.

### Deployment guide decision

A Cloudflare Tunnel deployment guide and PDF exist locally but are currently untracked. They were not incorporated into this documentation set. Review their operational and security guidance separately before deciding whether to commit them.

### Schema and troubleshooting references

The project still lacks:

- a consolidated annotation JSON schema reference
- a table-level reference for annotation and LLM SQLite storage
- a troubleshooting guide for provider credentials, worker status, stale caches, imports, and span selection

### Gemini tie-attribute structured output

The prompt layer advertises dynamic tie attributes, while Gemini's provider-level structured-output schema currently declares only tie type and reasoning text. Confirm Gemini's behavior for undeclared properties before documentation claims native Gemini tie-attribute generation is guaranteed.

### Reporting terminology

The report payload and UI still use `union_documents`/“Union docs” for the final evaluation-document count, even though LLM-only documents are excluded. The documentation describes the implemented semantics; the code/UI terminology should be considered in a separate behavior-preserving cleanup.

## Maintenance checklist

When a feature changes user workflow, persistence, providers, or reporting:

1. Update README behavior and configuration sections.
2. Update `docs/architecture.md` component and critical-flow descriptions.
3. Update contributor entry points in `docs/developer-guide.md`.
4. Update applicable TikZ sources.
5. Regenerate PDFs and the README architecture PNG.
6. Refresh affected screenshots.
7. Verify Markdown links, TeX compilation, documented CLI commands, and tests.
