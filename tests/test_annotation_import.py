from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from corn.annotation_io import export_document_annotation, import_document_annotation
from corn.storage import get_annotation, save_annotation_record

TIE_ATTRIBUTE_SCHEMA = [
    {
        "name": "valence",
        "type": "categorical",
        "description": "Whether the interaction is positive or negative",
        "options": ["positive", "negative"],
    },
    {
        "name": "sentiment",
        "type": "numeric",
        "description": "Sentiment intensity of the tie",
        "range": [0, 1],
    },
    {
        "name": "time",
        "type": "temporal",
        "description": "When the interaction occurred",
    },
]


def test_import_document_annotation_succeeds_for_existing_document_and_valid_ties(tmp_path: Path) -> None:
    runtime_db_path = tmp_path / "runtime.sqlite"
    annotations_dir = tmp_path / "annotations"
    annotations_db_path = tmp_path / "annotations.sqlite"
    _create_runtime_db(runtime_db_path, ["doc-1"])
    payload = _sample_payload()

    imported = import_document_annotation(
        runtime_db_path=runtime_db_path,
        annotations_dir=annotations_dir,
        annotations_db_path=annotations_db_path,
        payload=payload,
        valid_tie_labels={"Policy", "Coordination"},
        conflict_mode="replace",
    )

    stored = get_annotation(
        db_path=annotations_db_path,
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        document_id="doc-1",
    )
    assert imported == payload
    assert stored is not None
    assert stored["payload"] == payload


def test_import_document_annotation_rejects_missing_target_document(tmp_path: Path) -> None:
    runtime_db_path = tmp_path / "runtime.sqlite"
    _create_runtime_db(runtime_db_path, ["doc-2"])

    with pytest.raises(ValueError, match="Document ID not found"):
        import_document_annotation(
            runtime_db_path=runtime_db_path,
            annotations_dir=tmp_path / "annotations",
            annotations_db_path=tmp_path / "annotations.sqlite",
            payload=_sample_payload(),
            valid_tie_labels={"Policy"},
            conflict_mode="replace",
        )


def test_import_document_annotation_rejects_missing_tie_labels(tmp_path: Path) -> None:
    runtime_db_path = tmp_path / "runtime.sqlite"
    _create_runtime_db(runtime_db_path, ["doc-1"])

    with pytest.raises(ValueError, match="Unknown tie labels"):
        import_document_annotation(
            runtime_db_path=runtime_db_path,
            annotations_dir=tmp_path / "annotations",
            annotations_db_path=tmp_path / "annotations.sqlite",
            payload=_sample_payload(tie_type="Policy"),
            valid_tie_labels={"Coordination"},
            conflict_mode="replace",
        )


def test_import_document_annotation_replaces_existing_when_requested(tmp_path: Path) -> None:
    runtime_db_path = tmp_path / "runtime.sqlite"
    annotations_dir = tmp_path / "annotations"
    annotations_db_path = tmp_path / "annotations.sqlite"
    _create_runtime_db(runtime_db_path, ["doc-1"])
    original = _sample_payload(reasoning_text="old text")
    save_annotation_record(
        annotations_db_path,
        original,
        annotations_dir / "workspace-1" / "annotator-a" / "doc-1.json",
    )

    updated = _sample_payload(reasoning_text="new text", updated_at="2026-05-27T12:00:00+00:00")
    import_document_annotation(
        runtime_db_path=runtime_db_path,
        annotations_dir=annotations_dir,
        annotations_db_path=annotations_db_path,
        payload=updated,
        valid_tie_labels={"Policy"},
        conflict_mode="replace",
    )

    stored = get_annotation(
        db_path=annotations_db_path,
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        document_id="doc-1",
    )
    assert stored is not None
    assert stored["payload"]["network_data"][0]["ties"][0]["reasoning_text"] == "new text"


def test_import_document_annotation_aborts_on_conflict_when_requested(tmp_path: Path) -> None:
    runtime_db_path = tmp_path / "runtime.sqlite"
    annotations_dir = tmp_path / "annotations"
    annotations_db_path = tmp_path / "annotations.sqlite"
    _create_runtime_db(runtime_db_path, ["doc-1"])
    save_annotation_record(
        annotations_db_path,
        _sample_payload(reasoning_text="old text"),
        annotations_dir / "workspace-1" / "annotator-a" / "doc-1.json",
    )

    with pytest.raises(ValueError, match="Annotation conflict"):
        import_document_annotation(
            runtime_db_path=runtime_db_path,
            annotations_dir=annotations_dir,
            annotations_db_path=annotations_db_path,
            payload=_sample_payload(reasoning_text="new text"),
            valid_tie_labels={"Policy"},
            conflict_mode="abort",
        )


def test_import_document_annotation_round_trips_tie_attributes(tmp_path: Path) -> None:
    runtime_db_path = tmp_path / "runtime.sqlite"
    annotations_dir = tmp_path / "annotations"
    annotations_db_path = tmp_path / "annotations.sqlite"
    _create_runtime_db(runtime_db_path, ["doc-1"])
    attributes = {
        "valence": "positive",
        "sentiment": 1,
        "time": {"start": "2020-01", "end": "2020-06"},
    }
    payload = _sample_payload(attributes=attributes)

    import_document_annotation(
        runtime_db_path=runtime_db_path,
        annotations_dir=annotations_dir,
        annotations_db_path=annotations_db_path,
        payload=payload,
        valid_tie_labels={"Policy"},
        conflict_mode="replace",
        tie_attribute_schema=TIE_ATTRIBUTE_SCHEMA,
    )

    exported = export_document_annotation(
        annotations_db_path=annotations_db_path,
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        document_id="doc-1",
    )
    exported_attributes = exported["network_data"][0]["ties"][0]["attributes"]
    assert exported == payload
    assert exported_attributes == attributes
    # Imported payloads are stored verbatim: the numeric value keeps its imported type.
    assert isinstance(exported_attributes["sentiment"], int)


def test_import_document_annotation_rejects_unknown_tie_attributes(tmp_path: Path) -> None:
    runtime_db_path = tmp_path / "runtime.sqlite"
    _create_runtime_db(runtime_db_path, ["doc-1"])

    with pytest.raises(ValueError, match=r"Unknown tie attributes.*\(tie: Org A → Org B\)"):
        import_document_annotation(
            runtime_db_path=runtime_db_path,
            annotations_dir=tmp_path / "annotations",
            annotations_db_path=tmp_path / "annotations.sqlite",
            payload=_sample_payload(attributes={"weight": 1}),
            valid_tie_labels={"Policy"},
            conflict_mode="replace",
            tie_attribute_schema=TIE_ATTRIBUTE_SCHEMA,
        )


def test_import_document_annotation_rejects_invalid_tie_attribute_values(tmp_path: Path) -> None:
    runtime_db_path = tmp_path / "runtime.sqlite"
    _create_runtime_db(runtime_db_path, ["doc-1"])

    with pytest.raises(
        ValueError, match=r"Tie attribute valence must be one of.*\(tie: Org A → Org B\)"
    ):
        import_document_annotation(
            runtime_db_path=runtime_db_path,
            annotations_dir=tmp_path / "annotations",
            annotations_db_path=tmp_path / "annotations.sqlite",
            payload=_sample_payload(attributes={"valence": "sideways"}),
            valid_tie_labels={"Policy"},
            conflict_mode="replace",
            tie_attribute_schema=TIE_ATTRIBUTE_SCHEMA,
        )


def test_import_document_annotation_without_attributes_or_schema_still_works(tmp_path: Path) -> None:
    runtime_db_path = tmp_path / "runtime.sqlite"
    annotations_dir = tmp_path / "annotations"
    annotations_db_path = tmp_path / "annotations.sqlite"
    _create_runtime_db(runtime_db_path, ["doc-1"])
    payload = _sample_payload()

    imported = import_document_annotation(
        runtime_db_path=runtime_db_path,
        annotations_dir=annotations_dir,
        annotations_db_path=annotations_db_path,
        payload=payload,
        valid_tie_labels={"Policy"},
        conflict_mode="replace",
    )

    assert imported == payload


def _create_runtime_db(path: Path, document_ids: list[str]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE documents (
                document_id TEXT NOT NULL,
                document_text TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO documents (document_id, document_text) VALUES (?, ?)",
            [(document_id, f"Text for {document_id}") for document_id in document_ids],
        )
        connection.commit()
    finally:
        connection.close()


def _sample_payload(
    *,
    tie_type: str = "Policy",
    reasoning_text: str = "shared policy agenda",
    updated_at: str = "2026-05-27T10:00:00+00:00",
    attributes: dict | None = None,
) -> dict:
    tie: dict = {
        "type": tie_type,
        "reasoning_text": reasoning_text,
        "reasoning_start": 12,
        "reasoning_end": 32,
    }
    if attributes is not None:
        tie["attributes"] = attributes
    return {
        "workspace_id": "workspace-1",
        "annotator_id": "annotator-a",
        "document_id": "doc-1",
        "created_at": "2026-05-27T10:00:00+00:00",
        "updated_at": updated_at,
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [tie],
            }
        ],
    }
