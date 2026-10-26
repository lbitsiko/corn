from __future__ import annotations

from pathlib import Path

from corn.llm_storage import (
    cancel_job,
    claim_next_job,
    create_llm_job,
    create_prompt,
    complete_job,
    fork_prompt,
    get_job,
    get_job_document_items,
    get_llm_annotated_document_ids,
    get_llm_models_by_document,
    get_llm_run,
    get_project_default_prompt_id,
    get_project_llm_settings,
    get_prompt,
    list_all_valid_llm_runs,
    list_jobs,
    list_latest_valid_runs_for_documents,
    list_latest_valid_runs_for_scope,
    list_llm_reporting_scopes,
    list_llm_runs_for_document,
    list_prompts,
    save_llm_run,
    set_job_document_status,
    set_project_default_prompt_id,
    set_project_llm_settings,
    summarize_job,
)


def test_prompt_records_can_be_created_forked_and_promoted(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"

    base_prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    forked_prompt = fork_prompt(
        db_path=db_path,
        parent_prompt_id=base_prompt["prompt_id"],
        template_text="Annotate {{document_text}} with stricter rules",
        created_by="user-a",
        label="Document fork",
    )
    set_project_default_prompt_id(db_path, base_prompt["prompt_id"])

    prompts = list_prompts(db_path)

    assert [prompt["prompt_id"] for prompt in prompts] == [
        base_prompt["prompt_id"],
        forked_prompt["prompt_id"],
    ]
    assert get_prompt(db_path, forked_prompt["prompt_id"]) == forked_prompt
    assert forked_prompt["parent_prompt_id"] == base_prompt["prompt_id"]
    assert get_project_default_prompt_id(db_path) == base_prompt["prompt_id"]


def test_llm_job_persists_documents_and_status_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2", "doc-3"],
        prompt_id=prompt["prompt_id"],
        provider_id="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
        request_params={"temperature": 0.2, "max_output_tokens": 800},
        created_by="user-a",
    )

    set_job_document_status(db_path, job["job_id"], "doc-1", "succeeded")
    set_job_document_status(db_path, job["job_id"], "doc-2", "invalid")
    set_job_document_status(db_path, job["job_id"], "doc-3", "failed", error_message="provider timeout")

    stored_job = get_job(db_path, job["job_id"])
    assert stored_job is not None
    assert stored_job["job_id"] == job["job_id"]
    assert stored_job["status"] == "queued"
    assert stored_job["prompt_id"] == job["prompt_id"]
    assert stored_job["provider_id"] == job["provider_id"]
    assert stored_job["model_id"] == job["model_id"]
    assert stored_job["request_params"] == job["request_params"]
    assert stored_job["updated_at"] >= job["updated_at"]
    assert [item["document_id"] for item in get_job_document_items(db_path, job["job_id"])] == [
        "doc-1",
        "doc-2",
        "doc-3",
    ]
    assert summarize_job(db_path, job["job_id"]) == {
        "total": 3,
        "pending": 0,
        "running": 0,
        "succeeded": 1,
        "invalid": 1,
        "failed": 1,
        "cancelled": 0,
    }


def test_claim_next_job_assigns_worker_and_running_status(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-default",
        request_params={"temperature": 0.0, "max_output_tokens": 400},
        created_by="user-a",
    )

    claimed = claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=120,
        now="2026-05-31T12:00:00+00:00",
    )

    assert claimed is not None
    assert claimed["job_id"] == job["job_id"]
    assert claimed["status"] == "running"
    assert claimed["worker_id"] == "worker-1"
    assert claimed["started_at"] == "2026-05-31T12:00:00+00:00"
    assert claimed["lease_expires_at"] == "2026-05-31T12:02:00+00:00"


def test_claim_next_job_can_reclaim_expired_running_job(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-default",
        request_params={"temperature": 0.0, "max_output_tokens": 400},
        created_by="user-a",
    )
    claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
        now="2026-05-31T12:00:00+00:00",
    )

    reclaimed = claim_next_job(
        db_path,
        worker_id="worker-2",
        lease_seconds=60,
        now="2026-05-31T12:02:00+00:00",
    )

    assert reclaimed is not None
    assert reclaimed["job_id"] == job["job_id"]
    assert reclaimed["worker_id"] == "worker-2"
    assert reclaimed["lease_expires_at"] == "2026-05-31T12:03:00+00:00"


def test_complete_job_marks_terminal_job_status(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-default",
        request_params={"temperature": 0.0, "max_output_tokens": 400},
        created_by="user-a",
    )
    claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
        now="2026-05-31T12:00:00+00:00",
    )
    set_job_document_status(db_path, job["job_id"], "doc-1", "succeeded")
    set_job_document_status(db_path, job["job_id"], "doc-2", "invalid")

    complete_job(
        db_path,
        job["job_id"],
        worker_id="worker-1",
        now="2026-05-31T12:01:00+00:00",
    )

    stored_job = get_job(db_path, job["job_id"])
    assert stored_job["status"] == "completed_with_errors"
    assert stored_job["finished_at"] == "2026-05-31T12:01:00+00:00"
    assert stored_job["worker_id"] is None


def test_cancel_job_marks_queued_job_and_pending_documents_as_cancelled(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-default",
        request_params={"temperature": 0.0, "max_output_tokens": 400},
        created_by="user-a",
    )

    cancel_job(
        db_path,
        job["job_id"],
        reason="Cancelled manually",
        now="2026-05-31T12:01:00+00:00",
    )

    stored_job = get_job(db_path, job["job_id"])
    items = get_job_document_items(db_path, job["job_id"])

    assert stored_job["status"] == "cancelled"
    assert stored_job["finished_at"] == "2026-05-31T12:01:00+00:00"
    assert stored_job["worker_id"] is None
    assert [item["status"] for item in items] == ["cancelled", "cancelled"]
    assert summarize_job(db_path, job["job_id"]) == {
        "total": 2,
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "invalid": 0,
        "failed": 0,
        "cancelled": 2,
    }


def test_cancel_job_requests_stop_for_running_job_but_preserves_inflight_item(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-default",
        request_params={"temperature": 0.0, "max_output_tokens": 400},
        created_by="user-a",
    )
    claim_next_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
        now="2026-05-31T12:00:00+00:00",
    )
    set_job_document_status(db_path, job["job_id"], "doc-1", "running")

    cancel_job(
        db_path,
        job["job_id"],
        reason="Cancelled manually",
        now="2026-05-31T12:01:00+00:00",
    )

    stored_job = get_job(db_path, job["job_id"])
    items = get_job_document_items(db_path, job["job_id"])

    assert stored_job["status"] == "cancelled"
    assert stored_job["worker_id"] == "worker-1"
    assert stored_job["finished_at"] is None
    assert [item["status"] for item in items] == ["running", "cancelled"]
    assert summarize_job(db_path, job["job_id"]) == {
        "total": 2,
        "pending": 0,
        "running": 1,
        "succeeded": 0,
        "invalid": 0,
        "failed": 0,
        "cancelled": 1,
    }


def test_llm_runs_persist_valid_and_invalid_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1"],
        prompt_id=prompt["prompt_id"],
        provider_id="ollama",
        model_id="llama3",
        request_params={"temperature": 0.1, "max_output_tokens": 600},
        created_by="user-a",
    )

    valid_run = save_llm_run(
        db_path=db_path,
        job_id=job["job_id"],
        document_id="doc-1",
        prompt_id=prompt["prompt_id"],
        provider_id="ollama",
        model_id="llama3",
        prompt_template_text=prompt["template_text"],
        rendered_prompt_text="Annotate document text",
        request_params={"temperature": 0.1, "max_output_tokens": 600},
        raw_response='{"relationships":[]}',
        normalized_payload={"network_data": []},
        validation_status="valid",
        validation_errors=[],
        usage={"prompt_tokens": 120, "completion_tokens": 45},
        cost={"usd": 0.0},
    )
    invalid_run = save_llm_run(
        db_path=db_path,
        job_id=job["job_id"],
        document_id="doc-1",
        prompt_id=prompt["prompt_id"],
        provider_id="ollama",
        model_id="llama3",
        prompt_template_text=prompt["template_text"],
        rendered_prompt_text="Annotate document text",
        request_params={"temperature": 0.1, "max_output_tokens": 600},
        raw_response='{"foo":"bar"}',
        normalized_payload=None,
        validation_status="invalid",
        validation_errors=["relationships is required"],
        usage={},
        cost={},
    )

    stored_valid = get_llm_run(db_path, valid_run["run_id"])
    stored_invalid = get_llm_run(db_path, invalid_run["run_id"])

    assert stored_valid["normalized_payload"] == {"network_data": []}
    assert stored_valid["usage"] == {"prompt_tokens": 120, "completion_tokens": 45}
    assert stored_invalid["raw_response"] == '{"foo":"bar"}'
    assert stored_invalid["validation_status"] == "invalid"
    assert stored_invalid["validation_errors"] == ["relationships is required"]


def test_project_llm_settings_persist_provider_model_and_generation_defaults(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"

    assert get_project_llm_settings(db_path) == {
        "default_prompt_id": None,
        "provider_id": "fake",
        "model_id": "fake-default",
        "temperature": 0.0,
        "max_output_tokens": 800,
    }

    set_project_llm_settings(
        db_path,
        provider_id="openrouter",
        model_id="anthropic/claude-3.5-sonnet",
        temperature=0.3,
        max_output_tokens=1200,
    )

    assert get_project_llm_settings(db_path) == {
        "default_prompt_id": None,
        "provider_id": "openrouter",
        "model_id": "anthropic/claude-3.5-sonnet",
        "temperature": 0.3,
        "max_output_tokens": 1200,
    }


def test_list_jobs_and_document_runs_return_recent_records(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    older_job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="model-a",
        request_params={"temperature": 0.1},
        created_by="user-a",
    )
    newer_job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-2"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="model-b",
        request_params={"temperature": 0.2},
        created_by="user-a",
    )
    older_run = save_llm_run(
        db_path=db_path,
        job_id=older_job["job_id"],
        document_id="doc-1",
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="model-a",
        prompt_template_text=prompt["template_text"],
        rendered_prompt_text="Annotate doc-1",
        request_params={"temperature": 0.1},
        raw_response='{"relationships":[]}',
        normalized_payload={"network_data": []},
        validation_status="valid",
        validation_errors=[],
        usage={},
        cost={},
    )
    newer_run = save_llm_run(
        db_path=db_path,
        job_id=newer_job["job_id"],
        document_id="doc-1",
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="model-b",
        prompt_template_text=prompt["template_text"],
        rendered_prompt_text="Annotate doc-1 again",
        request_params={"temperature": 0.2},
        raw_response='{"relationships":[]}',
        normalized_payload={"network_data": []},
        validation_status="valid",
        validation_errors=[],
        usage={},
        cost={},
    )

    jobs = list_jobs(db_path)
    runs = list_llm_runs_for_document(db_path, "doc-1")

    assert [job["job_id"] for job in jobs] == [newer_job["job_id"], older_job["job_id"]]
    assert [run["run_id"] for run in runs] == [newer_run["run_id"], older_run["run_id"]]


def test_list_latest_valid_runs_returns_most_recent_valid_run_per_document(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2", "doc-3"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="model-a",
        request_params={"temperature": 0.0},
        created_by="user-a",
    )

    def _run(document_id: str, status: str, payload) -> dict:
        return save_llm_run(
            db_path=db_path,
            job_id=job["job_id"],
            document_id=document_id,
            prompt_id=prompt["prompt_id"],
            provider_id="fake",
            model_id="model-a",
            prompt_template_text=prompt["template_text"],
            rendered_prompt_text=f"prompt for {document_id}",
            request_params={"temperature": 0.0},
            raw_response="{}",
            normalized_payload=payload,
            validation_status=status,
            validation_errors=[] if status == "valid" else ["bad"],
            usage={},
            cost={},
        )

    # doc-1: two valid runs — second should win
    _run("doc-1", "valid", {"network_data": [], "marker": "old"})
    newer_doc1 = _run("doc-1", "valid", {"network_data": [], "marker": "new"})
    # doc-2: only an invalid run — should be absent
    _run("doc-2", "invalid", None)
    # doc-3: one valid run — should be returned
    only_doc3 = _run("doc-3", "valid", {"network_data": []})

    result = list_latest_valid_runs_for_documents(db_path, ["doc-1", "doc-2", "doc-3", "doc-4"])

    assert set(result.keys()) == {"doc-1", "doc-3"}
    assert result["doc-1"]["run_id"] == newer_doc1["run_id"]
    assert result["doc-1"]["normalized_payload"]["marker"] == "new"
    assert result["doc-3"]["run_id"] == only_doc3["run_id"]


def test_list_latest_valid_runs_returns_empty_for_no_document_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    assert list_latest_valid_runs_for_documents(db_path, []) == {}


def test_list_all_valid_llm_runs_returns_only_valid_ordered_by_document(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="model-a",
        request_params={},
        created_by="user-a",
    )

    def _run(doc_id: str, status: str) -> dict:
        return save_llm_run(
            db_path=db_path,
            job_id=job["job_id"],
            document_id=doc_id,
            prompt_id=prompt["prompt_id"],
            provider_id="fake",
            model_id="model-a",
            prompt_template_text=prompt["template_text"],
            rendered_prompt_text=f"prompt for {doc_id}",
            request_params={},
            raw_response="{}",
            normalized_payload={"network_data": [], "document_id": doc_id} if status == "valid" else None,
            validation_status=status,
            validation_errors=[] if status == "valid" else ["bad"],
            usage={},
            cost={},
        )

    older = _run("doc-1", "valid")
    newer = _run("doc-1", "valid")  # second valid run for same doc
    _run("doc-2", "invalid")        # invalid — must be excluded
    doc2_valid = _run("doc-2", "valid")

    runs = list_all_valid_llm_runs(db_path)

    assert all(r["validation_status"] == "valid" for r in runs)
    # doc-1 newest first, then doc-2
    run_ids = [r["run_id"] for r in runs]
    assert run_ids.index(newer["run_id"]) < run_ids.index(older["run_id"])
    assert doc2_valid["run_id"] in run_ids


def test_get_llm_models_by_document_groups_valid_models_per_document(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="model-a",
        request_params={},
        created_by="user-a",
    )

    def _run(doc_id: str, model_id: str, status: str) -> None:
        save_llm_run(
            db_path=db_path,
            job_id=job["job_id"],
            document_id=doc_id,
            prompt_id=prompt["prompt_id"],
            provider_id="fake",
            model_id=model_id,
            prompt_template_text=prompt["template_text"],
            rendered_prompt_text=f"prompt for {doc_id}",
            request_params={},
            raw_response="{}",
            normalized_payload={"network_data": [], "document_id": doc_id} if status == "valid" else None,
            validation_status=status,
            validation_errors=[] if status == "valid" else ["bad"],
            usage={},
            cost={},
        )

    _run("1", "model-b", "valid")
    _run("1", "model-a", "valid")
    _run("1", "model-a", "valid")  # duplicate model should collapse via DISTINCT
    _run("2", "model-c", "invalid")
    _run("2", "model-d", "valid")

    assert get_llm_models_by_document(db_path) == {
        "1": ["model-a", "model-b"],
        "2": ["model-d"],
    }


def test_get_llm_annotated_document_ids_returns_only_documents_with_valid_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2", "doc-3"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="model-a",
        request_params={},
        created_by="user-a",
    )

    def _run(doc_id: str, status: str) -> None:
        save_llm_run(
            db_path=db_path,
            job_id=job["job_id"],
            document_id=doc_id,
            prompt_id=prompt["prompt_id"],
            provider_id="fake",
            model_id="model-a",
            prompt_template_text=prompt["template_text"],
            rendered_prompt_text=f"prompt for {doc_id}",
            request_params={},
            raw_response="{}",
            normalized_payload={"network_data": [], "document_id": doc_id} if status == "valid" else None,
            validation_status=status,
            validation_errors=[] if status == "valid" else ["bad"],
            usage={},
            cost={},
        )

    _run("1", "valid")
    _run("2", "invalid")
    _run("3", "valid")

    assert get_llm_annotated_document_ids(db_path) == {"1", "3"}


def test_list_llm_reporting_scopes_lists_distinct_valid_provider_model_pairs(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2", "doc-3"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="model-a",
        request_params={},
        created_by="user-a",
    )

    def _run(doc_id: str, provider_id: str, model_id: str, status: str) -> None:
        save_llm_run(
            db_path=db_path,
            job_id=job["job_id"],
            document_id=doc_id,
            prompt_id=prompt["prompt_id"],
            provider_id=provider_id,
            model_id=model_id,
            prompt_template_text=prompt["template_text"],
            rendered_prompt_text=f"prompt for {doc_id}",
            request_params={},
            raw_response="{}",
            normalized_payload={"network_data": [], "document_id": doc_id} if status == "valid" else None,
            validation_status=status,
            validation_errors=[] if status == "valid" else ["bad"],
            usage={},
            cost={},
        )

    _run("doc-1", "openrouter", "model-a", "valid")
    _run("doc-2", "openrouter", "model-a", "valid")
    _run("doc-3", "openrouter", "model-b", "valid")
    _run("doc-4", "ollama", "model-c", "valid")
    _run("doc-5", "openrouter", "model-d", "invalid")

    assert list_llm_reporting_scopes(db_path) == [
        {
            "provider_id": "ollama",
            "model_id": "model-c",
            "document_count": 1,
            "latest_created_at": list_llm_reporting_scopes(db_path)[0]["latest_created_at"],
        },
        {
            "provider_id": "openrouter",
            "model_id": "model-a",
            "document_count": 2,
            "latest_created_at": list_llm_reporting_scopes(db_path)[1]["latest_created_at"],
        },
        {
            "provider_id": "openrouter",
            "model_id": "model-b",
            "document_count": 1,
            "latest_created_at": list_llm_reporting_scopes(db_path)[2]["latest_created_at"],
        },
    ]


def test_list_latest_valid_runs_for_scope_returns_latest_valid_run_per_document(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1", "doc-2", "doc-3"],
        prompt_id=prompt["prompt_id"],
        provider_id="openrouter",
        model_id="model-a",
        request_params={},
        created_by="user-a",
    )

    def _run(doc_id: str, provider_id: str, model_id: str, status: str, marker: str) -> dict:
        return save_llm_run(
            db_path=db_path,
            job_id=job["job_id"],
            document_id=doc_id,
            prompt_id=prompt["prompt_id"],
            provider_id=provider_id,
            model_id=model_id,
            prompt_template_text=prompt["template_text"],
            rendered_prompt_text=f"prompt for {doc_id}",
            request_params={},
            raw_response="{}",
            normalized_payload={"network_data": [], "document_id": doc_id, "marker": marker}
            if status == "valid"
            else None,
            validation_status=status,
            validation_errors=[] if status == "valid" else ["bad"],
            usage={},
            cost={},
        )

    _run("doc-1", "openrouter", "model-a", "valid", "old")
    latest_doc1 = _run("doc-1", "openrouter", "model-a", "valid", "new")
    _run("doc-1", "openrouter", "model-b", "valid", "other-model")
    _run("doc-2", "openrouter", "model-a", "invalid", "invalid")
    latest_doc3 = _run("doc-3", "openrouter", "model-a", "valid", "only")
    _run("doc-4", "ollama", "model-a", "valid", "other-provider")

    result = list_latest_valid_runs_for_scope(
        db_path,
        provider_id="openrouter",
        model_id="model-a",
    )

    assert set(result.keys()) == {"doc-1", "doc-3"}
    assert result["doc-1"]["run_id"] == latest_doc1["run_id"]
    assert result["doc-1"]["normalized_payload"]["marker"] == "new"
    assert result["doc-3"]["run_id"] == latest_doc3["run_id"]
