from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


SENTINEL_SECRET = "corn-packaging-secret-sentinel"


def verify_bundle(executable: Path, *, timeout: float = 60.0) -> None:
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"CORN executable not found: {executable}")

    bundle_root = executable.parent
    forbidden_names = {
        "project_registry.yaml",
        "provider_settings.json",
        "annotations.sqlite",
        "llm_events.jsonl",
        "failed_llms.log",
    }
    preexisting = [
        path for path in bundle_root.rglob("*") if path.name in forbidden_names
    ]
    if preexisting:
        raise RuntimeError(f"Mutable state found in bundle: {preexisting}")

    with tempfile.TemporaryDirectory(prefix="corn-bundle-smoke-") as temp_dir:
        data_dir = Path(temp_dir)
        environment = os.environ.copy()
        environment.update(
            {
                "CORN_DATA_DIR": str(data_dir),
                "GEMINI_API_KEY": SENTINEL_SECRET,
                "OPENROUTER_API_KEY": SENTINEL_SECRET,
            }
        )
        result = subprocess.run(
            [str(executable), "--internal-smoke-test"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        combined_output = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0:
            raise RuntimeError(
                f"Packaged CORN smoke test failed with {result.returncode}:\n"
                f"{combined_output}"
            )
        if not (data_dir / "project_registry.yaml").is_file():
            raise RuntimeError(
                "Packaged CORN did not initialize its user-data registry"
            )

        diagnostic_text = combined_output
        for log_path in (data_dir / "logs").glob("*"):
            if log_path.is_file():
                diagnostic_text += log_path.read_text(
                    encoding="utf-8", errors="replace"
                )
        if SENTINEL_SECRET in diagnostic_text:
            raise RuntimeError("Packaged CORN leaked a provider credential")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a packaged CORN bundle.")
    parser.add_argument("executable", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    arguments = parser.parse_args()
    verify_bundle(arguments.executable, timeout=arguments.timeout)
    print(f"Verified CORN bundle: {arguments.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
