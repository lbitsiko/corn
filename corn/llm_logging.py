from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _append_json_line(log_path: Path, event: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_llm_event(log_path: Path, event: dict[str, Any]) -> None:
    _append_json_line(log_path, event)


def append_failed_llm_run(log_path: Path, event: dict[str, Any]) -> None:
    _append_json_line(log_path, event)
