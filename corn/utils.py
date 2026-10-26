from __future__ import annotations

import json
import re
from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(value: str) -> str:
    """Convert a document identifier into a filesystem-safe filename stub."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if sanitized:
        return sanitized[:180]
    return "annotation"


def load_tie_options(path: Path) -> list[str]:
    """Load tie type options from JSON."""
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list) or not all(isinstance(item, str) for item in content):
        raise ValueError(f"Invalid tie options payload in {path}")
    return content
