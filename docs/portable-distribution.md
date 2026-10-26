# Portable CORN Distribution

CORN portable releases package the existing Streamlit application, LLM worker, Python runtime, and required assets. Users do not need to install Python. Releases are native: download the archive matching the operating system and CPU architecture.

## Supported packages

| Platform | Archive | Launch command |
|---|---|---|
| Windows x64 | `CORN-windows-x64.zip` | `CORN\CORN.exe` |
| macOS Apple Silicon | `CORN-macos-arm64.zip` | `./CORN/CORN` |
| Linux x64 | `CORN-linux-x64.tar.gz` | `./CORN/CORN` |

Intel macOS, Windows ARM, and Linux ARM are not included in the first release.

## Download and verify

Download the archive and adjacent `.sha256` file from the GitHub Release. Verify it before extraction.

Windows PowerShell:

```powershell
(Get-FileHash .\CORN-windows-x64.zip -Algorithm SHA256).Hash.ToLower()
Get-Content .\CORN-windows-x64.zip.sha256
```

macOS:

```bash
shasum -a 256 -c CORN-macos-arm64.zip.sha256
```

Linux:

```bash
sha256sum -c CORN-linux-x64.tar.gz.sha256
```

The calculated and published hashes must match.

## Launch CORN

### Windows

Extract the complete ZIP and run `CORN\CORN.exe`. Do not move only the executable out of the `CORN` folder; the adjacent `_internal` directory is required.

Early builds are unsigned. If Microsoft Defender SmartScreen appears, select **More info**, confirm that the archive came from the project release page and its checksum matches, then select **Run anyway**.

### macOS Apple Silicon

Extract the ZIP and run from Terminal:

```bash
./CORN/CORN
```

Early builds are unsigned and not notarized. If Gatekeeper blocks the executable, verify the checksum first, then right-click the executable in Finder and choose **Open**, or approve it under **System Settings → Privacy & Security**. Only bypass Gatekeeper for a release you obtained from the project and verified.

### Linux x64

Extract and run:

```bash
tar -xzf CORN-linux-x64.tar.gz
./CORN/CORN
```

If archive handling removed the executable bit, run `chmod +x CORN/CORN` once.

## Launcher behavior

The launcher starts the LLM worker and Streamlit UI, binds the UI only to `127.0.0.1`, waits for the health endpoint, and opens the default browser. Keep the launcher open while using CORN.

The launcher provides:

- **Open Browser** to reopen the local UI
- **Open Logs** to inspect UI and worker logs
- **Restart Worker** to recover LLM processing without restarting the UI
- **Stop CORN** to terminate both child processes cleanly

Closing the launcher also stops CORN. Closing only the browser tab does not stop it.

## User data

Writable state is separate from the extracted application folder:

- Windows: `%LOCALAPPDATA%\CORN`
- macOS: `~/Library/Application Support/CORN`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/CORN`

Projects, annotations, the registry, provider settings, and logs live there. Replacing the extracted application folder during an upgrade does not remove user data. Back up this directory before important upgrades or research milestones.

A fresh installation starts with no projects. The home screen offers **Install demo project**, which creates a small synthetic project in user storage. Installing the demo is optional and never replaces an existing project.

## LLM credentials

The first portable release reads credentials only from environment variables. Keys are not stored by the launcher and must not be placed in the application folder.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY = "your-key"
.\CORN\CORN.exe
```

macOS or Linux:

```bash
export GEMINI_API_KEY="your-key"
./CORN/CORN
```

Use `OPENROUTER_API_KEY` for OpenRouter. Ollama normally needs only a reachable `OLLAMA_BASE_URL`; authenticated endpoints may use `OLLAMA_API_KEY`.

Applications launched from Finder may not inherit variables configured in a shell profile. For this release, launch CORN from the Terminal session where the key was exported. Never include real keys in bug reports or logs.

## Logs and troubleshooting

Select **Open Logs** in the launcher. `ui.log` records Streamlit startup and runtime failures; `worker.log` records worker process output. Logs should not contain provider-key values. Review them before sharing because research workflows may still produce project identifiers or error context.

If the browser does not open, wait for the launcher to report that the UI is running and select **Open Browser**. If LLM jobs remain queued, check the worker status and select **Restart Worker**.

## Developer build

Create a clean environment and install build dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-build.txt
pytest -q
pyinstaller --clean --noconfirm packaging/corn.spec
python packaging/verify_bundle.py dist/CORN/CORN
```

On Windows, verify `dist/CORN/CORN.exe`. Build each target on its native operating system; do not cross-compile release archives.

Clean generated products with:

```bash
rm -rf build dist
```

## Release verification checklist

Before publishing a platform archive:

1. Run the full test suite and frozen-bundle verifier.
2. Test on a clean machine without Python installed.
3. Start without API keys and install the demo project.
4. Create and reopen a user project.
5. Replace the application folder and confirm user data survives.
6. Start with an environment key and complete one LLM job.
7. Simulate worker failure and use **Restart Worker**.
8. Stop from the launcher and confirm no child process remains.
9. Inspect output and logs for secret leakage.
10. Create and publish the archive SHA-256 checksum.

Installers, automatic updates, system-tray behavior, code signing, and Apple notarization are deferred from this initial portable release.
