# ADR 0001: Portable desktop runtime boundary

- Status: Accepted
- Date: 2026-09-26

## Context

CORN currently runs as a Streamlit UI plus an optional LLM worker. Researchers must install Python and dependencies and start both processes themselves. The first desktop distribution should preserve the working Streamlit interface while making the application usable on machines without Python.

The distribution targets Windows x64, macOS Apple Silicon, and Linux x64. Early releases are portable archives rather than installers and may be unsigned.

## Decision

Package CORN as a platform-native PyInstaller `onedir` application with a small launcher that supervises two child roles:

1. the existing Streamlit application, bound only to `127.0.0.1`; and
2. the existing LLM worker.

The launcher selects an available port, starts both roles with the same environment and writable-data root, waits for Streamlit's health endpoint, and then opens the default browser. It remains responsible for status, worker restart, logs, and bounded shutdown.

Packaged code and assets are read-only resources. Mutable registries, projects, annotations, provider settings, and logs live under a platform-specific user-data directory selected through `CORN_DATA_DIR`. Source-mode development continues to default to the repository `data/` directory.

The bundle contains no real project data or credentials. A fresh workspace is empty, with an optional toy project. Provider API keys remain environment-variable-only and must not appear in logs or diagnostics.

Native GitHub Actions runners build each operating-system artifact. Releases include SHA-256 checksums. Signing, notarization, installers, automatic updates, and single-file packaging are deferred.

## Consequences

- The existing Streamlit UI and source-mode commands remain supported.
- Replacing the extracted application folder does not replace user data.
- Each operating system and architecture requires its own build.
- `onedir` produces an application folder rather than one standalone binary, but avoids fragile runtime extraction and makes Streamlit assets easier to package and diagnose.
- Unsigned Windows and macOS builds require documented SmartScreen or Gatekeeper approval during early testing.
