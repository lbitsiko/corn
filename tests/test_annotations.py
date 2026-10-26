import json
from pathlib import Path

import pytest

from corn.annotations import (
    HIGHLIGHT_PALETTE,
    append_network_entry,
    build_highlight_spans,
    build_network_entry,
    IncompleteAnnotationError,
    find_reasoning_spans,
    load_annotation,
    save_annotation,
)


def test_find_reasoning_spans_returns_single_match() -> None:
    content = "Alpha and Beta coordinate once."

    assert find_reasoning_spans(content, "Beta") == [(10, 14)]


def test_find_reasoning_spans_returns_all_duplicate_matches() -> None:
    content = "Alpha supports Beta. Beta responds."

    assert find_reasoning_spans(content, "Beta") == [(15, 19), (21, 25)]


def test_build_network_entry_uses_selected_text_offsets() -> None:
    entry = build_network_entry(
        source="Org A",
        target="Org B",
        tie_type="Policy",
        reasoning_text="selected evidence",
        reasoning_start=4,
        reasoning_end=21,
    )

    assert entry == {
        "source": "Org A",
        "target": "Org B",
        "ties": [
            {
                "type": "Policy",
                "reasoning_text": "selected evidence",
                "reasoning_text_span": "selected evidence",
                "reasoning_text_note": "",
                "reasoning_start": 4,
                "reasoning_end": 21,
            }
        ],
    }


def test_save_annotation_accepts_note_only_reasoning_evidence(tmp_path: Path) -> None:
    payload = {
        "workspace_id": "default",
        "annotator_id": "annotator-a",
        "annotation_source": "human_from_llm",
        "document_id": "doc-1",
        "created_at": "2026-05-27T10:00:00+00:00",
        "updated_at": "2026-05-27T10:00:00+00:00",
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "LLM summary evidence",
                        "reasoning_text_span": "",
                        "reasoning_text_note": "LLM summary evidence",
                        "reasoning_start": None,
                        "reasoning_end": None,
                    }
                ],
            }
        ],
    }

    saved_path = save_annotation(
        tmp_path,
        Path("default/annotator-a/doc-1.json"),
        payload,
    )

    assert load_annotation(saved_path) == payload


def test_append_network_entry_preserves_existing_entries() -> None:
    annotation_state = {
        "network_data": [
            build_network_entry("Org A", "Org B", "Policy", "first", 0, 5),
        ]
    }

    append_network_entry(
        annotation_state,
        build_network_entry("Org C", "Org D", "Coordination", "second", 6, 12),
    )

    assert len(annotation_state["network_data"]) == 2
    assert annotation_state["network_data"][1]["source"] == "Org C"


def test_build_highlight_spans_collects_saved_reasoning_offsets() -> None:
    annotation_state = {
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "first",
                        "reasoning_start": 4,
                        "reasoning_end": 9,
                    },
                    {
                        "type": "Coordination",
                        "reasoning_text": "second",
                        "reasoning_start": 12,
                        "reasoning_end": 18,
                    },
                    {
                        "type": "Ignored",
                        "reasoning_text": "missing offsets",
                        "reasoning_start": None,
                        "reasoning_end": None,
                    },
                ],
            }
        ]
    }

    spans = build_highlight_spans(annotation_state)

    assert [(span["start"], span["end"]) for span in spans] == [(4, 9), (12, 18)]
    assert all(span["color"] in HIGHLIGHT_PALETTE for span in spans)


def test_build_highlight_spans_preserves_adjacent_annotations_as_distinct_entries() -> None:
    annotation_state = {
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "first",
                        "reasoning_start": 4,
                        "reasoning_end": 9,
                    },
                    {
                        "type": "Coordination",
                        "reasoning_text": "second",
                        "reasoning_start": 9,
                        "reasoning_end": 14,
                    },
                ],
            }
        ]
    }

    spans = build_highlight_spans(annotation_state)

    assert [span["start"] for span in spans] == [4, 9]
    assert [span["end"] for span in spans] == [9, 14]
    assert all("color" in span for span in spans)
    assert spans[0]["color"] != spans[1]["color"]


def test_build_highlight_spans_includes_annotation_summaries_for_hover_tooltips() -> None:
    annotation_state = {
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "shared policy agenda",
                        "reasoning_start": 12,
                        "reasoning_end": 32,
                    },
                    {
                        "type": "Coordination",
                        "reasoning_text": "shared policy agenda with extra detail",
                        "reasoning_start": 12,
                        "reasoning_end": 40,
                    },
                ],
            }
        ]
    }

    spans = build_highlight_spans(annotation_state)

    assert len(spans) == 2
    assert spans[0]["annotations"] == [
        {
            "source": "Org A",
            "target": "Org B",
            "type": "Policy",
            "color": spans[0]["color"],
        },
        {
            "source": "Org A",
            "target": "Org B",
            "type": "Coordination",
            "color": spans[1]["color"],
        },
    ]
    assert spans[1]["annotations"] == [
        {
            "source": "Org A",
            "target": "Org B",
            "type": "Coordination",
            "color": spans[1]["color"],
        },
        {
            "source": "Org A",
            "target": "Org B",
            "type": "Policy",
            "color": spans[0]["color"],
        },
    ]


def test_save_annotation_rejects_incomplete_network_entry(tmp_path: Path) -> None:
    with pytest.raises(IncompleteAnnotationError):
        save_annotation(
            output_dir=tmp_path,
            relative_path=Path("default/annotator-a/doc-1.json"),
            payload={
                "workspace_id": "default",
                "annotator_id": "annotator-a",
                "document_id": "doc-1",
                "created_at": "2026-05-27T10:00:00+00:00",
                "updated_at": "2026-05-27T10:00:00+00:00",
                "network_data": [{"source": "", "target": "B", "ties": []}],
            },
        )


def test_save_annotation_writes_json_and_loads_it_back(tmp_path: Path) -> None:
    payload = {
        "workspace_id": "default",
        "annotator_id": "annotator-a",
        "document_id": "doc-1",
        "created_at": "2026-05-27T10:00:00+00:00",
        "updated_at": "2026-05-27T10:00:00+00:00",
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "shared policy agenda",
                        "reasoning_start": 12,
                        "reasoning_end": 32,
                    }
                ],
            }
        ],
    }

    saved_path = save_annotation(
        tmp_path,
        Path("default/annotator-a/doc-1.json"),
        payload,
    )
    loaded = load_annotation(saved_path)

    assert saved_path == tmp_path / "default" / "annotator-a" / "doc-1.json"
    assert json.loads(saved_path.read_text(encoding="utf-8")) == payload
    assert loaded == payload
