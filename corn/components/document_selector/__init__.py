from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components


_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_DOCUMENT_SELECTOR = components.declare_component(
    "document_selector",
    path=str(_FRONTEND_DIR),
)


def document_selection(
    text: str,
    key: str,
    height: int = 420,
    highlights: list[dict[str, int | str]] | None = None,
    focus_request: dict[str, int | str] | None = None,
    selection_mode: str = "create_entry",
) -> dict | None:
    """Render selectable document text and return the latest text selection."""
    return _DOCUMENT_SELECTOR(
        text=text,
        height=height,
        highlights=highlights or [],
        focus_request=focus_request,
        selection_mode=selection_mode,
        key=key,
        default=None,
    )
