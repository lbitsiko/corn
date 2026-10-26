from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from corn.utils import ensure_directory


class IncompleteAnnotationError(ValueError):
    """Raised when required annotation fields are missing."""


class DuplicateSpanMatchError(ValueError):
    """Raised when a reasoning span matches the document content more than once."""


HIGHLIGHT_PALETTE = [
    "#fef3c7",
    "#fde68a",
    "#bfdbfe",
    "#bbf7d0",
    "#fbcfe8",
    "#fecdd3",
    "#ddd6fe",
    "#fed7aa",
    "#a7f3d0",
    "#c7d2fe",
    "#f5d0fe",
    "#bae6fd",
]


def find_reasoning_spans(content: str, reasoning_text: str) -> list[tuple[int, int]]:
    """Return all exact start/end offsets for a reasoning text span."""
    if not reasoning_text:
        return []

    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        index = content.find(reasoning_text, start)
        if index == -1:
            break
        matches.append((index, index + len(reasoning_text)))
        start = index + 1
    return matches


def build_network_entry(
    source: str,
    target: str,
    tie_type: str,
    reasoning_text: str,
    reasoning_start: int,
    reasoning_end: int,
) -> dict:
    """Build a normalized network entry from a selected document span."""
    return {
        "source": source.strip(),
        "target": target.strip(),
        "ties": [
            {
                "type": tie_type.strip(),
                "reasoning_text": reasoning_text,
                "reasoning_text_span": reasoning_text,
                "reasoning_text_note": "",
                "reasoning_start": reasoning_start,
                "reasoning_end": reasoning_end,
            }
        ],
    }


def append_network_entry(annotation_state: dict, entry: dict) -> dict:
    """Append a normalized network entry to annotation state."""
    annotation_state.setdefault("network_data", [])
    annotation_state["network_data"].append(entry)
    return annotation_state


def _highlight_color(entry: dict, tie: dict, entry_index: int, tie_index: int) -> str:
    """Assign a stable highlight color for a saved annotation tie."""
    stable_key = "|".join(
        [
            str(entry.get("source", "")),
            str(entry.get("target", "")),
            str(tie.get("type", "")),
            str(tie.get("reasoning_text", "")),
            str(tie.get("reasoning_start")),
            str(tie.get("reasoning_end")),
            str(entry_index),
            str(tie_index),
        ]
    )
    digest = sha256(stable_key.encode("utf-8")).digest()
    palette_index = int.from_bytes(digest[:2], byteorder="big") % len(HIGHLIGHT_PALETTE)
    return HIGHLIGHT_PALETTE[palette_index]


def _highlight_annotation_summary(entry: dict, tie: dict, color: str) -> dict[str, str]:
    """Build the tooltip summary payload for a highlighted annotation."""
    return {
        "source": str(entry.get("source", "")).strip(),
        "target": str(entry.get("target", "")).strip(),
        "type": str(tie.get("type", "")).strip(),
        "color": color,
    }


def _entry_annotation_summaries(
    entry: dict, entry_index: int, prioritized_tie_index: int
) -> list[dict[str, str]]:
    """Build tooltip summaries for all ties in a single annotation entry."""
    ties = list(enumerate(entry.get("ties", [])))
    ordered_ties = sorted(
        ties,
        key=lambda item: (
            item[0] != prioritized_tie_index,
            item[0],
        ),
    )

    summaries: list[dict[str, str]] = []
    for tie_index, tie in ordered_ties:
        summaries.append(
            _highlight_annotation_summary(
                entry=entry,
                tie=tie,
                color=_highlight_color(entry, tie, entry_index, tie_index),
            )
        )
    return summaries


def build_highlight_spans(annotation_state: dict) -> list[dict[str, int | str | list[dict[str, str]]]]:
    """Collect document highlight spans while preserving per-annotation identity."""
    seen_spans: set[tuple[int, int, str]] = set()
    normalized_spans: list[dict[str, int | str | list[dict[str, str]]]] = []
    for entry_index, entry in enumerate(annotation_state.get("network_data", [])):
        for tie_index, tie in enumerate(entry.get("ties", [])):
            start = tie.get("reasoning_start")
            end = tie.get("reasoning_end")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start < 0 or end <= start:
                continue
            color = _highlight_color(entry, tie, entry_index, tie_index)
            dedupe_key = (start, end, color)
            if dedupe_key in seen_spans:
                continue
            seen_spans.add(dedupe_key)
            entry_summaries = _entry_annotation_summaries(entry, entry_index, tie_index)
            normalized_spans.append(
                {
                    "start": start,
                    "end": end,
                    "color": color,
                    "annotations": entry_summaries,
                }
            )

    normalized_spans.sort(
        key=lambda span: (
            int(span["start"]),
            int(span["end"]),
            str(span["color"]),
        )
    )
    return normalized_spans


def _validate_annotation_payload(payload: dict) -> None:
    required_root_fields = [
        "workspace_id",
        "annotator_id",
        "document_id",
        "created_at",
        "updated_at",
        "network_data",
    ]
    for field in required_root_fields:
        if field not in payload:
            raise IncompleteAnnotationError(f"Missing required field: {field}")

    if not payload["workspace_id"]:
        raise IncompleteAnnotationError("workspace_id is required")
    if not payload["annotator_id"]:
        raise IncompleteAnnotationError("annotator_id is required")

    if not isinstance(payload["network_data"], list):
        raise IncompleteAnnotationError("network_data must be a list")

    for entry_index, entry in enumerate(payload["network_data"], start=1):
        if not entry.get("source"):
            raise IncompleteAnnotationError(f"Entry {entry_index} is missing source")
        if not entry.get("target"):
            raise IncompleteAnnotationError(f"Entry {entry_index} is missing target")
        ties = entry.get("ties")
        if not isinstance(ties, list) or not ties:
            raise IncompleteAnnotationError(f"Entry {entry_index} must include at least one tie")

        for tie_index, tie in enumerate(ties, start=1):
            if not tie.get("type"):
                raise IncompleteAnnotationError(
                    f"Entry {entry_index}, tie {tie_index} is missing type"
                )
            span_text = str(tie.get("reasoning_text_span") or "").strip()
            note_text = str(tie.get("reasoning_text_note") or "").strip()
            legacy_reasoning_text = str(tie.get("reasoning_text") or "").strip()
            if not (span_text or note_text or legacy_reasoning_text):
                raise IncompleteAnnotationError(
                    f"Entry {entry_index}, tie {tie_index} is missing reasoning evidence"
                )


def save_annotation(output_dir: Path, relative_path: Path, payload: dict) -> Path:
    """Validate and save a single-document annotation payload as JSON."""
    _validate_annotation_payload(payload)
    output_path = output_dir / relative_path
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_annotation(path: Path) -> dict:
    """Load an annotation JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))
