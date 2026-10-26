from __future__ import annotations

import json
from pathlib import Path

from corn.llm_jobs import execute_claimed_job
from corn.llm_providers import FakeLLMProvider, ProviderExecutionError
from corn.llm_storage import (
    claim_next_job,
    create_llm_job,
    create_prompt,
    get_job,
    get_job_document_items,
    get_llm_run,
    summarize_job,
)
from corn.storage import get_annotation


def test_execute_claimed_job_processes_documents_sequentially_and_persists_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Project {{project_name}}\nText: {{document_text}}",
        created_by="user-a",
        label="Default prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2", "doc-3"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-model",
        request_params={"temperature": 0.1, "max_output_tokens": 400},
        created_by="user-a",
    )
    provider = FakeLLMProvider(
        outputs_by_document_id={
            "doc-1": {
                "relationships": [
                    {
                        "source": "Org A",
                        "target": "Org B",
                        "ties": [{"type": "support", "reasoning_text": "shared agenda"}],
                    }
                ]
            },
            "doc-2": {"foo": "bar"},
        },
        errors_by_document_id={"doc-3": "provider timeout"},
    )
    documents = {
        "doc-1": {"link": "doc-1", "content": "The shared agenda was noted.", "time_period": "2024"},
        "doc-2": {"link": "doc-2", "content": "No valid output.", "time_period": "2024"},
        "doc-3": {"link": "doc-3", "content": "This one fails.", "time_period": "2024"},
    }

    claimed = claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
        now="2026-05-31T12:00:00+00:00",
    )

    result = execute_claimed_job(
        db_path=db_path,
        job_id=job["job_id"],
        worker_id="worker-1",
        provider=provider,
        project_name="Toy Project",
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        resolve_document=lambda document_id: documents[document_id],
    )

    assert claimed is not None
    assert result["processed_document_ids"] == ["doc-1", "doc-2", "doc-3"]
    items = get_job_document_items(db_path, job["job_id"])
    assert [item["status"] for item in items] == ["succeeded", "invalid", "failed"]
    assert summarize_job(db_path, job["job_id"]) == {
        "total": 3,
        "pending": 0,
        "running": 0,
        "succeeded": 1,
        "invalid": 1,
        "failed": 1,
        "cancelled": 0,
    }

    valid_run = get_llm_run(db_path, items[0]["llm_run_id"])
    invalid_run = get_llm_run(db_path, items[1]["llm_run_id"])

    assert valid_run is not None
    assert valid_run["validation_status"] == "valid"
    assert valid_run["normalized_payload"]["network_data"][0]["ties"][0]["type"] == "support"
    assert invalid_run is not None
    assert invalid_run["validation_status"] == "invalid"
    assert invalid_run["validation_errors"] == ["relationships is required"]
    assert items[2]["error_message"] == "provider timeout"
    assert get_job(db_path, job["job_id"])["status"] == "completed_with_errors"

    # Valid run for doc-1 should be auto-committed to annotations table
    saved = get_annotation(
        db_path=db_path,
        workspace_id="llm",
        annotator_id="fake/fake-model",
        document_id="doc-1",
    )
    assert saved is not None
    assert saved["payload"]["annotation_source"] == "llm"
    assert saved["payload"]["network_data"][0]["ties"][0]["type"] == "support"
    # doc-2 (invalid) and doc-3 (failed) must not be auto-committed
    assert get_annotation(db_path=db_path, workspace_id="llm", annotator_id="fake/fake-model", document_id="doc-2") is None
    assert get_annotation(db_path=db_path, workspace_id="llm", annotator_id="fake/fake-model", document_id="doc-3") is None


TIE_ATTRIBUTE_SCHEMA = [
    {
        "name": "sentiment",
        "type": "numeric",
        "description": "Sentiment score",
        "range": [0, 1],
    },
]


def test_execute_claimed_job_validates_and_normalizes_tie_attributes(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Project {{project_name}}\nText: {{document_text}}",
        created_by="user-a",
        label="Default prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-model",
        request_params={"temperature": 0.1, "max_output_tokens": 400},
        created_by="user-a",
    )
    provider = FakeLLMProvider(
        outputs_by_document_id={
            "doc-1": {
                "relationships": [
                    {
                        "source": "Org A",
                        "target": "Org B",
                        "ties": [
                            {
                                "type": "support",
                                "reasoning_text": "shared agenda",
                                "attributes": {"sentiment": "0.7"},
                            }
                        ],
                    }
                ]
            },
        },
    )
    documents = {
        "doc-1": {"link": "doc-1", "content": "The shared agenda was noted.", "time_period": "2024"},
    }
    claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
        now="2026-05-31T12:00:00+00:00",
    )

    execute_claimed_job(
        db_path=db_path,
        job_id=job["job_id"],
        worker_id="worker-1",
        provider=provider,
        project_name="Toy Project",
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        tie_attribute_schema=TIE_ATTRIBUTE_SCHEMA,
        resolve_document=lambda document_id: documents[document_id],
    )

    items = get_job_document_items(db_path, job["job_id"])
    assert items[0]["status"] == "succeeded"
    run = get_llm_run(db_path, items[0]["llm_run_id"])
    assert run["validation_status"] == "valid"
    saved_tie = run["normalized_payload"]["network_data"][0]["ties"][0]
    assert saved_tie["attributes"] == {"sentiment": 0.7}


def test_execute_claimed_job_marks_unknown_tie_attribute_as_invalid(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Project {{project_name}}\nText: {{document_text}}",
        created_by="user-a",
        label="Default prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-model",
        request_params={"temperature": 0.1, "max_output_tokens": 400},
        created_by="user-a",
    )
    provider = FakeLLMProvider(
        outputs_by_document_id={
            "doc-1": {
                "relationships": [
                    {
                        "source": "Org A",
                        "target": "Org B",
                        "ties": [
                            {
                                "type": "support",
                                "reasoning_text": "shared agenda",
                                "attributes": {"bogus": "x"},
                            }
                        ],
                    }
                ]
            },
        },
    )
    documents = {
        "doc-1": {"link": "doc-1", "content": "The shared agenda was noted.", "time_period": "2024"},
    }
    claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
        now="2026-05-31T12:00:00+00:00",
    )

    execute_claimed_job(
        db_path=db_path,
        job_id=job["job_id"],
        worker_id="worker-1",
        provider=provider,
        project_name="Toy Project",
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        tie_attribute_schema=TIE_ATTRIBUTE_SCHEMA,
        resolve_document=lambda document_id: documents[document_id],
    )

    items = get_job_document_items(db_path, job["job_id"])
    assert items[0]["status"] == "invalid"
    run = get_llm_run(db_path, items[0]["llm_run_id"])
    assert run["validation_status"] == "invalid"
    assert run["validation_errors"] == ["Unknown tie attributes: bogus"]
    assert get_job(db_path, job["job_id"])["status"] == "completed_with_errors"
    # Invalid output must not be auto-committed as an annotation
    assert get_annotation(db_path=db_path, workspace_id="llm", annotator_id="fake/fake-model", document_id="doc-1") is None


def test_execute_claimed_job_marks_missing_document_as_invalid_instead_of_leaking_running_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Project {{project_name}}\nText: {{document_text}}",
        created_by="user-a",
        label="Default prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-model",
        request_params={"temperature": 0.1, "max_output_tokens": 400},
        created_by="user-a",
    )
    claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
        now="2026-05-31T12:00:00+00:00",
    )

    result = execute_claimed_job(
        db_path=db_path,
        job_id=job["job_id"],
        worker_id="worker-1",
        provider=FakeLLMProvider(),
        project_name="Toy Project",
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        resolve_document=lambda _document_id: None,
    )

    items = get_job_document_items(db_path, job["job_id"])
    assert result["processed_document_ids"] == ["doc-1"]
    assert items[0]["status"] == "invalid"
    assert "NoneType" in items[0]["error_message"]
    assert get_job(db_path, job["job_id"])["status"] == "completed_with_errors"


def test_execute_claimed_job_writes_request_response_and_error_log_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    log_path = tmp_path / "llm_events.jsonl"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Project {{project_name}}\nText: {{document_text}}",
        created_by="user-a",
        label="Default prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-model",
        request_params={"temperature": 0.1, "max_output_tokens": 400},
        created_by="user-a",
    )
    provider = FakeLLMProvider(
        outputs_by_document_id={
            "doc-1": {
                "relationships": [
                    {
                        "source": "Org A",
                        "target": "Org B",
                        "ties": [{"type": "support", "reasoning_text": "shared agenda"}],
                    }
                ]
            }
        },
        errors_by_document_id={"doc-2": "provider timeout"},
    )
    documents = {
        "doc-1": {"link": "doc-1", "content": "The shared agenda was noted.", "time_period": "2024"},
        "doc-2": {"link": "doc-2", "content": "This one fails.", "time_period": "2024"},
    }
    claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
        now="2026-05-31T12:00:00+00:00",
    )

    execute_claimed_job(
        db_path=db_path,
        job_id=job["job_id"],
        worker_id="worker-1",
        provider=provider,
        project_name="Toy Project",
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        resolve_document=lambda document_id: documents[document_id],
        log_path=log_path,
    )

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events] == [
        "llm_document_succeeded",
        "llm_document_failed",
    ]
    assert events[0]["rendered_prompt_text"].startswith("Project Toy Project")
    assert "relationships" in events[0]["raw_response"]
    assert events[1]["error"] == "provider timeout"


def test_execute_claimed_job_writes_failed_runs_to_separate_failed_llms_log(tmp_path: Path) -> None:
    class MixedFailureProvider:
        provider_id = "mixed"

        def annotate(
            self,
            *,
            rendered_prompt_text: str,
            document_id: str,
            model_id: str,
            request_params: dict[str, object],
        ) -> dict[str, object]:
            if document_id == "doc-1":
                return {
                    "raw_response": '{"foo":"bar"}',
                    "parsed_output": {"foo": "bar"},
                    "usage": {},
                    "cost": {},
                    "provider_id": self.provider_id,
                    "model_id": model_id,
                    "request_params": request_params,
                }
            raise ProviderExecutionError(
                "Provider returned non-JSON annotation content.",
                raw_response='{"choices":[{"message":{"content":"not json at all"}}]}',
            )

    db_path = tmp_path / "annotations.sqlite"
    failed_log_path = tmp_path / "failed_llms.log"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Project {{project_name}}\nText: {{document_text}}",
        created_by="user-a",
        label="Default prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2"],
        prompt_id=prompt["prompt_id"],
        provider_id="mixed",
        model_id="mixed-model",
        request_params={"temperature": 0.1, "max_output_tokens": 400},
        created_by="user-a",
    )
    documents = {
        "doc-1": {"link": "doc-1", "content": "Invalid schema output.", "time_period": "2024"},
        "doc-2": {"link": "doc-2", "content": "Non JSON output.", "time_period": "2024"},
    }
    claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
        now="2026-05-31T12:00:00+00:00",
    )

    execute_claimed_job(
        db_path=db_path,
        job_id=job["job_id"],
        worker_id="worker-1",
        provider=MixedFailureProvider(),
        project_name="Toy Project",
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        resolve_document=lambda document_id: documents[document_id],
        failed_log_path=failed_log_path,
    )

    failed_events = [json.loads(line) for line in failed_log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in failed_events] == [
        "llm_document_invalid",
        "llm_document_failed",
    ]
    assert failed_events[0]["raw_response"] == '{"foo":"bar"}'
    assert failed_events[0]["validation_status"] == "invalid"
    assert "not json at all" in failed_events[1]["raw_response"]
    assert failed_events[1]["validation_status"] == "failed"
