from __future__ import annotations

from pathlib import Path

from corn.annotation_io import export_document_annotation
from corn.storage import save_annotation_record


def test_export_document_annotation_returns_payload_for_one_scoped_document(tmp_path: Path) -> None:
    annotations_db_path = tmp_path / "annotations.sqlite"
    payload = {
        "workspace_id": "workspace-1",
        "annotator_id": "annotator-a",
        "document_id": "doc-1",
        "created_at": "2026-05-27T10:00:00+00:00",
        "updated_at": "2026-05-27T11:00:00+00:00",
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
    json_path = tmp_path / "workspace-1" / "annotator-a" / "doc-1.json"
    save_annotation_record(annotations_db_path, payload, json_path)

    exported = export_document_annotation(
        annotations_db_path=annotations_db_path,
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        document_id="doc-1",
    )

    assert exported == payload
