import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

from corn.storage import (
    export_annotations_edge_list_csv,
    export_annotations_json_archive,
    get_annotation,
    list_annotation_scopes,
    list_annotations,
    save_annotation_record,
)


def _sample_payload(
    *,
    document_id: str = "doc-1",
    annotator_id: str = "annotator-a",
    workspace_id: str = "workspace-1",
    created_at: str = "2026-05-27T10:00:00+00:00",
    updated_at: str = "2026-05-27T10:00:00+00:00",
) -> dict:
    return {
        "workspace_id": workspace_id,
        "annotator_id": annotator_id,
        "document_id": document_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "annotation_source": "human_from_llm",
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "shared policy agenda",
                        "reasoning_text_span": "shared policy agenda",
                        "reasoning_text_note": "LLM summary evidence",
                        "reasoning_start": 12,
                        "reasoning_end": 32,
                    }
                ],
            }
        ],
    }


def test_save_annotation_record_persists_and_updates_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    json_path = tmp_path / "workspace-1" / "annotator-a" / "doc-1.json"
    original_payload = _sample_payload()

    save_annotation_record(db_path, original_payload, json_path)
    loaded = get_annotation(
        db_path=db_path,
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        document_id="doc-1",
    )

    assert loaded is not None
    assert loaded["payload"] == original_payload
    assert loaded["json_path"] == "workspace-1/annotator-a/doc-1.json"

    updated_payload = _sample_payload(updated_at="2026-05-27T11:00:00+00:00")
    save_annotation_record(db_path, updated_payload, json_path)
    reloaded = get_annotation(
        db_path=db_path,
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        document_id="doc-1",
    )

    assert reloaded is not None
    assert reloaded["payload"]["created_at"] == "2026-05-27T10:00:00+00:00"
    assert reloaded["payload"]["updated_at"] == "2026-05-27T11:00:00+00:00"


def test_list_annotations_can_filter_by_annotator_and_workspace(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    save_annotation_record(
        db_path,
        _sample_payload(document_id="doc-1", annotator_id="annotator-a", workspace_id="workspace-1"),
        tmp_path / "workspace-1" / "annotator-a" / "doc-1.json",
    )
    save_annotation_record(
        db_path,
        _sample_payload(document_id="doc-2", annotator_id="annotator-b", workspace_id="workspace-1"),
        tmp_path / "workspace-1" / "annotator-b" / "doc-2.json",
    )
    save_annotation_record(
        db_path,
        _sample_payload(document_id="doc-3", annotator_id="annotator-a", workspace_id="workspace-2"),
        tmp_path / "workspace-2" / "annotator-a" / "doc-3.json",
    )

    assert [row["document_id"] for row in list_annotations(db_path)] == ["doc-1", "doc-2", "doc-3"]
    assert [row["document_id"] for row in list_annotations(db_path, annotator_id="annotator-a")] == [
        "doc-1",
        "doc-3",
    ]
    assert [row["document_id"] for row in list_annotations(db_path, workspace_id="workspace-1")] == [
        "doc-1",
        "doc-2",
    ]


def test_list_annotations_can_filter_by_annotation_source(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"

    def _save(doc_id: str, source: str) -> None:
        payload = _sample_payload(document_id=doc_id)
        payload["annotation_source"] = source
        save_annotation_record(db_path, payload, tmp_path / f"{doc_id}.json")

    _save("doc-human", "human")
    _save("doc-llm", "llm")
    _save("doc-hfl", "human_from_llm")

    assert [r["document_id"] for r in list_annotations(db_path, annotation_source="human")] == ["doc-human"]
    assert [r["document_id"] for r in list_annotations(db_path, annotation_source="llm")] == ["doc-llm"]
    assert [r["document_id"] for r in list_annotations(db_path, annotation_source="human_from_llm")] == ["doc-hfl"]
    assert len(list_annotations(db_path)) == 3


def test_list_annotation_scopes_returns_distinct_workspace_annotator_pairs(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    save_annotation_record(
        db_path,
        _sample_payload(document_id="doc-1", annotator_id="annotator-a", workspace_id="workspace-1"),
        tmp_path / "workspace-1" / "annotator-a" / "doc-1.json",
    )
    save_annotation_record(
        db_path,
        _sample_payload(document_id="doc-2", annotator_id="annotator-b", workspace_id="workspace-1"),
        tmp_path / "workspace-1" / "annotator-b" / "doc-2.json",
    )
    save_annotation_record(
        db_path,
        _sample_payload(document_id="doc-3", annotator_id="annotator-a", workspace_id="workspace-2"),
        tmp_path / "workspace-2" / "annotator-a" / "doc-3.json",
    )

    assert list_annotation_scopes(db_path) == [
        ("workspace-1", "annotator-a"),
        ("workspace-1", "annotator-b"),
        ("workspace-2", "annotator-a"),
    ]


def test_export_annotations_edge_list_csv_flattens_ties(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    save_annotation_record(
        db_path,
        _sample_payload(),
        tmp_path / "workspace-1" / "annotator-a" / "doc-1.json",
    )

    csv_bytes = export_annotations_edge_list_csv(list_annotations(db_path))
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"))))
    normalized_annotation = json.loads(rows[0]["normalized_annotation"])

    assert rows == [
        {
            "workspace_id": "workspace-1",
            "annotator_id": "annotator-a",
            "document_id": "doc-1",
            "source": "Org A",
            "target": "Org B",
            "annotation_source": "human-from-llm",
            "tie_type": "Policy",
            "reasoning_text": "shared policy agenda",
            "reasoning_text_span": "shared policy agenda",
            "reasoning_text_note": "LLM summary evidence",
            "reasoning_start": "12",
            "reasoning_end": "32",
            "normalized_annotation": rows[0]["normalized_annotation"],
            "created_at": "2026-05-27T10:00:00+00:00",
            "updated_at": "2026-05-27T10:00:00+00:00",
        }
    ]
    assert normalized_annotation == {
        "annotation_source": "human-from-llm",
        "source": "Org A",
        "target": "Org B",
        "tie": {
            "type": "Policy",
            "reasoning_text": "shared policy agenda",
            "reasoning_text_span": "shared policy agenda",
            "reasoning_text_note": "LLM summary evidence",
            "reasoning_start": 12,
            "reasoning_end": 32,
        },
    }


def test_export_annotations_json_archive_includes_scoped_files(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    save_annotation_record(
        db_path,
        _sample_payload(document_id="doc-1"),
        tmp_path / "workspace-1" / "annotator-a" / "doc-1.json",
    )
    save_annotation_record(
        db_path,
        _sample_payload(document_id="doc-2", annotator_id="annotator-b"),
        tmp_path / "workspace-1" / "annotator-b" / "doc-2.json",
    )

    archive_bytes = export_annotations_json_archive(list_annotations(db_path, annotator_id="annotator-a"))
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        payload = json.loads(archive.read("workspace-1/annotator-a/doc-1.json").decode("utf-8"))

    assert names == ["workspace-1/annotator-a/doc-1.json"]
    assert payload["annotator_id"] == "annotator-a"


def test_export_annotations_json_archive_falls_back_for_empty_json_path() -> None:
    payload = _sample_payload(
        workspace_id="llm",
        annotator_id="openrouter/google/gemini-2.5-flash-lite",
        document_id="2",
    )

    archive_bytes = export_annotations_json_archive(
        [
            {
                "workspace_id": payload["workspace_id"],
                "annotator_id": payload["annotator_id"],
                "document_id": payload["document_id"],
                "json_path": "",
                "payload": payload,
            }
        ]
    )
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        exported_payload = json.loads(
            archive.read("llm/openrouter_google_gemini-2.5-flash-lite/2.json").decode("utf-8")
        )

    assert names == ["llm/openrouter_google_gemini-2.5-flash-lite/2.json"]
    assert exported_payload["document_id"] == "2"
