from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_provider_settings(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Provider settings file must contain a JSON object.")
    normalized: dict[str, dict[str, Any]] = {}
    for provider_id, settings in raw.items():
        if isinstance(provider_id, str) and isinstance(settings, dict):
            normalized[provider_id] = dict(settings)
    return normalized


def save_provider_settings(path: Path, settings_by_provider: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings_by_provider, indent=2, sort_keys=True), encoding="utf-8")

