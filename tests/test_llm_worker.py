from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from corn.llm_providers import FakeLLMProvider
from corn.llm_storage import (
    create_llm_job,
    create_prompt,
    get_job,
    list_llm_runs_for_document,
    summarize_job,
)
from corn.llm_worker import process_next_job, process_registry_once, run_worker
from corn.registry import register_project


def test_run_worker_stops_when_requested(tmp_path: Path) -> None:
    calls = []
    stop_values = iter([False, True])

    run_worker(
        registry_path=tmp_path / "registry.yaml",
        provider_settings_path=tmp_path / "providers.json",
        worker_id="desktop-test",
        poll_seconds=0,
        should_stop=lambda: next(stop_values),
        process_once=lambda **kwargs: calls.append(kwargs),
        sleep=lambda _: None,
    )

    assert len(calls) == 1
    assert calls[0]["worker_id"] == "desktop-test"


def test_process_next_job_claims_and_executes_queued_job(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Project {{project_name}}\nText: {{document_text}}",
        created_by="user-a",
        label="Default prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["1"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-default",
        request_params={"temperature": 0.0, "max_output_tokens": 400},
        created_by="user-a",
    )

    result = process_next_job(
        db_path=db_path,
        worker_id="worker-1",
        settings_by_provider={},
        project_name="Toy Project",
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        resolve_document=lambda document_id: {
            "id": int(document_id),
            "link": f"doc-{document_id}",
            "document_id": f"doc-{document_id}",
            "content": "shared agenda appears here",
        },
        lease_seconds=60,
    )

    assert result is not None
    assert result["job_id"] == job["job_id"]
    assert summarize_job(db_path, job["job_id"]) == {
        "total": 1,
        "pending": 0,
        "running": 0,
        "succeeded": 1,
        "invalid": 0,
        "failed": 0,
        "cancelled": 0,
    }
    assert get_job(db_path, job["job_id"])["status"] == "completed"


def test_process_registry_once_normalizes_attributes_from_project_config(
    tmp_path: Path, monkeypatch
) -> None:
    project_path = tmp_path / "projects" / "toy-project"
    (project_path / "derived").mkdir(parents=True)
    (project_path / "annotations").mkdir(parents=True)
    (project_path / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "Toy Project", "slug": "toy-project"},
                "source": {"type": "csv", "mode": "copy", "path": "data/source.csv"},
                "document": {"id_field": "id", "text_field": "text"},
                "tie_schema": [
                    {"label": "support", "description": "Helpful relationship"}
                ],
                "tie_attribute_schema": [
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
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    runtime_db_path = project_path / "derived" / "runtime.sqlite"
    connection = sqlite3.connect(runtime_db_path)
    try:
        connection.execute(
            "CREATE TABLE documents (document_id TEXT NOT NULL, document_text TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO documents (document_id, document_text) VALUES (?, ?)",
            ("doc-1", "shared agenda appears here"),
        )
        connection.commit()
    finally:
        connection.close()

    annotations_db_path = project_path / "annotations" / "annotations.sqlite"
    prompt = create_prompt(
        db_path=annotations_db_path,
        template_text="Project {{project_name}}\nText: {{document_text}}",
        created_by="user-a",
        label="Default prompt",
    )
    job = create_llm_job(
        db_path=annotations_db_path,
        document_ids=["1"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-default",
        request_params={"temperature": 0.0, "max_output_tokens": 400},
        created_by="user-a",
    )

    registry_path = tmp_path / "registry.yaml"
    register_project(
        registry_path,
        name="Toy Project",
        slug="toy-project",
        project_path=project_path,
    )

    fake_provider = FakeLLMProvider(
        outputs_by_document_id={
            "1": {
                "relationships": [
                    {
                        "source": "Alice",
                        "target": "Bob",
                        "ties": [
                            {
                                "type": "support",
                                "reasoning_text": "shared agenda appears here",
                                "attributes": {
                                    "valence": "positive",
                                    "sentiment": "0.5",
                                },
                            }
                        ],
                    }
                ]
            }
        }
    )
    monkeypatch.setattr(
        "corn.llm_worker.get_provider", lambda *args, **kwargs: fake_provider
    )

    result = process_registry_once(
        registry_path=registry_path,
        provider_settings_path=tmp_path / "provider_settings.json",
        worker_id="worker-1",
    )

    assert result is not None
    assert result["project_slug"] == "toy-project"
    assert result["job_id"] == job["job_id"]

    runs = list_llm_runs_for_document(annotations_db_path, "1")
    assert len(runs) == 1
    run = runs[0]
    assert run["validation_status"] == "valid"
    tie = run["normalized_payload"]["network_data"][0]["ties"][0]
    # The schema comes from project.yaml; values are normalized ("0.5" -> 0.5).
    assert tie["attributes"] == {"valence": "positive", "sentiment": 0.5}
