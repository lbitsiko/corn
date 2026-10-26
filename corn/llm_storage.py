from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from corn.storage import initialize_annotations_db

JOB_DOCUMENT_STATUSES = {"pending", "running", "succeeded", "invalid", "failed", "cancelled"}
RUN_VALIDATION_STATUSES = {"valid", "invalid", "failed"}
JOB_STATUSES = {"queued", "running", "completed", "completed_with_errors", "cancelled"}
DEFAULT_LLM_SETTINGS = {
    "provider_id": "fake",
    "model_id": "fake-default",
    "temperature": 0.0,
    "max_output_tokens": 800,
}


def initialize_llm_storage(db_path: Path) -> Path:
    """Ensure prompt, job, and LLM run tables exist in the project-local SQLite db."""
    initialize_annotations_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_prompts (
                prompt_id TEXT PRIMARY KEY,
                parent_prompt_id TEXT,
                label TEXT NOT NULL,
                template_text TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_project_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'queued',
                prompt_id TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                request_params_json TEXT NOT NULL,
                created_by TEXT NOT NULL,
                worker_id TEXT,
                lease_expires_at TEXT,
                heartbeat_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_job_documents (
                job_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL,
                llm_run_id TEXT,
                error_message TEXT,
                started_at TEXT,
                finished_at TEXT,
                PRIMARY KEY (job_id, document_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_runs (
                run_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_template_text TEXT NOT NULL,
                rendered_prompt_text TEXT NOT NULL,
                request_params_json TEXT NOT NULL,
                raw_response_text TEXT NOT NULL,
                normalized_payload_json TEXT,
                validation_status TEXT NOT NULL,
                validation_errors_json TEXT NOT NULL,
                usage_json TEXT NOT NULL,
                cost_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_column(connection, "llm_jobs", "status", "TEXT NOT NULL DEFAULT 'queued'")
        _ensure_column(connection, "llm_jobs", "worker_id", "TEXT")
        _ensure_column(connection, "llm_jobs", "lease_expires_at", "TEXT")
        _ensure_column(connection, "llm_jobs", "heartbeat_at", "TEXT")
        _ensure_column(connection, "llm_jobs", "started_at", "TEXT")
        _ensure_column(connection, "llm_jobs", "finished_at", "TEXT")
        connection.commit()
    return db_path


def create_prompt(
    db_path: Path,
    *,
    template_text: str,
    created_by: str,
    label: str,
    parent_prompt_id: str | None = None,
) -> dict[str, Any]:
    initialize_llm_storage(db_path)
    timestamp = _iso_utcnow()
    prompt = {
        "prompt_id": _generate_id("prompt"),
        "parent_prompt_id": parent_prompt_id,
        "label": label,
        "template_text": template_text,
        "created_by": created_by,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO llm_prompts (
                prompt_id,
                parent_prompt_id,
                label,
                template_text,
                created_by,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prompt["prompt_id"],
                prompt["parent_prompt_id"],
                prompt["label"],
                prompt["template_text"],
                prompt["created_by"],
                prompt["created_at"],
                prompt["updated_at"],
            ),
        )
        connection.commit()
    return prompt


def fork_prompt(
    db_path: Path,
    *,
    parent_prompt_id: str,
    template_text: str,
    created_by: str,
    label: str,
) -> dict[str, Any]:
    parent_prompt = get_prompt(db_path, parent_prompt_id)
    if parent_prompt is None:
        raise ValueError(f"Prompt not found: {parent_prompt_id}")
    return create_prompt(
        db_path=db_path,
        template_text=template_text,
        created_by=created_by,
        label=label,
        parent_prompt_id=parent_prompt_id,
    )


def get_prompt(db_path: Path, prompt_id: str) -> dict[str, Any] | None:
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT prompt_id, parent_prompt_id, label, template_text, created_by, created_at, updated_at
            FROM llm_prompts
            WHERE prompt_id = ?
            """,
            (prompt_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_prompt(row)


def list_prompts(db_path: Path) -> list[dict[str, Any]]:
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT prompt_id, parent_prompt_id, label, template_text, created_by, created_at, updated_at
            FROM llm_prompts
            ORDER BY created_at, prompt_id
            """
        ).fetchall()
    return [_row_to_prompt(row) for row in rows]


def set_project_default_prompt_id(db_path: Path, prompt_id: str) -> None:
    if get_prompt(db_path, prompt_id) is None:
        raise ValueError(f"Prompt not found: {prompt_id}")
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO llm_project_settings (setting_key, setting_value)
            VALUES ('default_prompt_id', ?)
            ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value
            """,
            (prompt_id,),
        )
        connection.commit()


def get_project_default_prompt_id(db_path: Path) -> str | None:
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT setting_value
            FROM llm_project_settings
            WHERE setting_key = 'default_prompt_id'
            """
        ).fetchone()
    return row[0] if row is not None else None


def get_project_llm_settings(db_path: Path) -> dict[str, Any]:
    initialize_llm_storage(db_path)
    settings = dict(DEFAULT_LLM_SETTINGS)
    settings["default_prompt_id"] = get_project_default_prompt_id(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT setting_key, setting_value
            FROM llm_project_settings
            WHERE setting_key IN (
                'llm_default_provider_id',
                'llm_default_model_id',
                'llm_default_temperature',
                'llm_default_max_output_tokens'
            )
            """
        ).fetchall()

    for setting_key, setting_value in rows:
        if setting_key == "llm_default_provider_id":
            settings["provider_id"] = setting_value
        elif setting_key == "llm_default_model_id":
            settings["model_id"] = setting_value
        elif setting_key == "llm_default_temperature":
            settings["temperature"] = float(setting_value)
        elif setting_key == "llm_default_max_output_tokens":
            settings["max_output_tokens"] = int(setting_value)
    return settings


def set_project_llm_settings(
    db_path: Path,
    *,
    provider_id: str,
    model_id: str,
    temperature: float,
    max_output_tokens: int,
) -> None:
    initialize_llm_storage(db_path)
    settings = {
        "llm_default_provider_id": provider_id,
        "llm_default_model_id": model_id,
        "llm_default_temperature": str(float(temperature)),
        "llm_default_max_output_tokens": str(int(max_output_tokens)),
    }
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO llm_project_settings (setting_key, setting_value)
            VALUES (?, ?)
            ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value
            """,
            list(settings.items()),
        )
        connection.commit()


def create_llm_job(
    db_path: Path,
    *,
    document_ids: list[str],
    prompt_id: str,
    provider_id: str,
    model_id: str,
    request_params: dict[str, Any],
    created_by: str,
) -> dict[str, Any]:
    if get_prompt(db_path, prompt_id) is None:
        raise ValueError(f"Prompt not found: {prompt_id}")
    initialize_llm_storage(db_path)
    timestamp = _iso_utcnow()
    job = {
        "job_id": _generate_id("job"),
        "status": "queued",
        "prompt_id": prompt_id,
        "provider_id": provider_id,
        "model_id": model_id,
        "request_params": request_params,
        "created_by": created_by,
        "worker_id": None,
        "lease_expires_at": None,
        "heartbeat_at": None,
        "started_at": None,
        "finished_at": None,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO llm_jobs (
                job_id,
                status,
                prompt_id,
                provider_id,
                model_id,
                request_params_json,
                created_by,
                worker_id,
                lease_expires_at,
                heartbeat_at,
                started_at,
                finished_at,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["job_id"],
                job["status"],
                job["prompt_id"],
                job["provider_id"],
                job["model_id"],
                json.dumps(job["request_params"], sort_keys=True),
                job["created_by"],
                job["worker_id"],
                job["lease_expires_at"],
                job["heartbeat_at"],
                job["started_at"],
                job["finished_at"],
                job["created_at"],
                job["updated_at"],
            ),
        )
        connection.executemany(
            """
            INSERT INTO llm_job_documents (
                job_id, document_id, status, llm_run_id, error_message, started_at, finished_at
            ) VALUES (?, ?, 'pending', NULL, NULL, NULL, NULL)
            """,
            [(job["job_id"], document_id) for document_id in document_ids],
        )
        connection.commit()
    return job


def get_job(db_path: Path, job_id: str) -> dict[str, Any] | None:
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                job_id,
                status,
                prompt_id,
                provider_id,
                model_id,
                request_params_json,
                created_by,
                worker_id,
                lease_expires_at,
                heartbeat_at,
                started_at,
                finished_at,
                created_at,
                updated_at
            FROM llm_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_job(row)


def list_jobs(db_path: Path) -> list[dict[str, Any]]:
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                job_id,
                status,
                prompt_id,
                provider_id,
                model_id,
                request_params_json,
                created_by,
                worker_id,
                lease_expires_at,
                heartbeat_at,
                started_at,
                finished_at,
                created_at,
                updated_at
            FROM llm_jobs
            ORDER BY created_at DESC, job_id DESC
            """
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def claim_next_job(
    db_path: Path,
    *,
    worker_id: str,
    lease_seconds: int,
    now: str | None = None,
) -> dict[str, Any] | None:
    initialize_llm_storage(db_path)
    current_time = now or _iso_utcnow()
    lease_expires_at = _iso_plus_seconds(current_time, lease_seconds)
    with sqlite3.connect(db_path) as connection:
        connection.isolation_level = None
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT job_id
            FROM llm_jobs
            WHERE
                (status = 'queued')
                OR (
                    status = 'running'
                    AND lease_expires_at IS NOT NULL
                    AND lease_expires_at < ?
                )
            ORDER BY created_at, job_id
            LIMIT 1
            """,
            (current_time,),
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None

        job_id = row[0]
        result = connection.execute(
            """
            UPDATE llm_jobs
            SET status = 'running',
                worker_id = ?,
                lease_expires_at = ?,
                heartbeat_at = ?,
                started_at = COALESCE(started_at, ?),
                finished_at = NULL,
                updated_at = ?
            WHERE job_id = ?
              AND (
                status = 'queued'
                OR (
                    status = 'running'
                    AND lease_expires_at IS NOT NULL
                    AND lease_expires_at < ?
                )
              )
            """,
            (
                worker_id,
                lease_expires_at,
                current_time,
                current_time,
                current_time,
                job_id,
                current_time,
            ),
        )
        if result.rowcount != 1:
            connection.execute("ROLLBACK")
            return None
        connection.execute("COMMIT")
    return get_job(db_path, job_id)


def heartbeat_job_lease(
    db_path: Path,
    job_id: str,
    *,
    worker_id: str,
    lease_seconds: int,
    now: str | None = None,
) -> None:
    initialize_llm_storage(db_path)
    current_time = now or _iso_utcnow()
    lease_expires_at = _iso_plus_seconds(current_time, lease_seconds)
    with sqlite3.connect(db_path) as connection:
        result = connection.execute(
            """
            UPDATE llm_jobs
            SET heartbeat_at = ?,
                lease_expires_at = ?,
                updated_at = ?
            WHERE job_id = ? AND status = 'running' AND worker_id = ?
            """,
            (current_time, lease_expires_at, current_time, job_id, worker_id),
        )
        if result.rowcount != 1:
            raise ValueError(f"Job is not actively leased by worker {worker_id}: {job_id}")
        connection.commit()


def complete_job(
    db_path: Path,
    job_id: str,
    *,
    worker_id: str,
    now: str | None = None,
) -> None:
    initialize_llm_storage(db_path)
    current_time = now or _iso_utcnow()
    summary = summarize_job(db_path, job_id)
    if summary["pending"] or summary["running"]:
        raise ValueError(f"Cannot complete job with unfinished items: {job_id}")
    terminal_status = (
        "completed_with_errors"
        if summary["invalid"] or summary["failed"]
        else "completed"
    )
    with sqlite3.connect(db_path) as connection:
        result = connection.execute(
            """
            UPDATE llm_jobs
            SET status = ?,
                worker_id = NULL,
                lease_expires_at = NULL,
                heartbeat_at = ?,
                finished_at = ?,
                updated_at = ?
            WHERE job_id = ? AND worker_id = ?
            """,
            (terminal_status, current_time, current_time, current_time, job_id, worker_id),
        )
        if result.rowcount != 1:
            raise ValueError(f"Job is not owned by worker {worker_id}: {job_id}")
        connection.commit()


def cancel_job(
    db_path: Path,
    job_id: str,
    *,
    reason: str = "Cancelled manually",
    now: str | None = None,
) -> dict[str, Any]:
    initialize_llm_storage(db_path)
    current_time = now or _iso_utcnow()
    job = get_job(db_path, job_id)
    if job is None:
        raise ValueError(f"LLM job not found: {job_id}")
    if job["status"] not in {"queued", "running"}:
        return job

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE llm_job_documents
            SET status = 'cancelled',
                error_message = ?,
                finished_at = ?
            WHERE job_id = ? AND status = 'pending'
            """,
            (reason, current_time, job_id),
        )
        if job["status"] == "queued":
            connection.execute(
                """
                UPDATE llm_jobs
                SET status = 'cancelled',
                    worker_id = NULL,
                    lease_expires_at = NULL,
                    heartbeat_at = NULL,
                    finished_at = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (current_time, current_time, job_id),
            )
        else:
            connection.execute(
                """
                UPDATE llm_jobs
                SET status = 'cancelled',
                    updated_at = ?
                WHERE job_id = ?
                """,
                (current_time, job_id),
            )
        connection.commit()
    return get_job(db_path, job_id)


def finalize_cancelled_job(
    db_path: Path,
    job_id: str,
    *,
    worker_id: str,
    now: str | None = None,
) -> None:
    initialize_llm_storage(db_path)
    current_time = now or _iso_utcnow()
    summary = summarize_job(db_path, job_id)
    if summary["running"]:
        raise ValueError(f"Cannot finalize cancelled job with running items: {job_id}")
    with sqlite3.connect(db_path) as connection:
        result = connection.execute(
            """
            UPDATE llm_jobs
            SET worker_id = NULL,
                lease_expires_at = NULL,
                heartbeat_at = ?,
                finished_at = ?,
                updated_at = ?
            WHERE job_id = ? AND status = 'cancelled' AND worker_id = ?
            """,
            (current_time, current_time, current_time, job_id, worker_id),
        )
        if result.rowcount != 1:
            raise ValueError(f"Cancelled job is not owned by worker {worker_id}: {job_id}")
        connection.commit()


def get_job_document_items(db_path: Path, job_id: str) -> list[dict[str, Any]]:
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT job_id, document_id, status, llm_run_id, error_message, started_at, finished_at
            FROM llm_job_documents
            WHERE job_id = ?
            ORDER BY document_id
            """,
            (job_id,),
        ).fetchall()
    return [_row_to_job_document(row) for row in rows]


def set_job_document_status(
    db_path: Path,
    job_id: str,
    document_id: str,
    status: str,
    *,
    llm_run_id: str | None = None,
    error_message: str | None = None,
) -> None:
    if status not in JOB_DOCUMENT_STATUSES:
        raise ValueError(f"Invalid job document status: {status}")
    initialize_llm_storage(db_path)
    timestamp = _iso_utcnow()
    started_at = timestamp if status == "running" else None
    finished_at = timestamp if status in {"succeeded", "invalid", "failed", "cancelled"} else None
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE llm_job_documents
            SET status = ?,
                llm_run_id = COALESCE(?, llm_run_id),
                error_message = ?,
                started_at = COALESCE(?, started_at),
                finished_at = ?
            WHERE job_id = ? AND document_id = ?
            """,
            (status, llm_run_id, error_message, started_at, finished_at, job_id, document_id),
        )
        connection.execute(
            "UPDATE llm_jobs SET updated_at = ? WHERE job_id = ?",
            (timestamp, job_id),
        )
        connection.commit()


def summarize_job(db_path: Path, job_id: str) -> dict[str, int]:
    initialize_llm_storage(db_path)
    summary = {status: 0 for status in JOB_DOCUMENT_STATUSES}
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT status, COUNT(*)
            FROM llm_job_documents
            WHERE job_id = ?
            GROUP BY status
            """,
            (job_id,),
        ).fetchall()
    total = 0
    for status, count in rows:
        summary[str(status)] = int(count)
        total += int(count)
    return {
        "total": total,
        "pending": summary["pending"],
        "running": summary["running"],
        "succeeded": summary["succeeded"],
        "invalid": summary["invalid"],
        "failed": summary["failed"],
        "cancelled": summary["cancelled"],
    }


def save_llm_run(
    db_path: Path,
    *,
    job_id: str,
    document_id: str,
    prompt_id: str,
    provider_id: str,
    model_id: str,
    prompt_template_text: str,
    rendered_prompt_text: str,
    request_params: dict[str, Any],
    raw_response: str,
    normalized_payload: dict[str, Any] | None,
    validation_status: str,
    validation_errors: list[str],
    usage: dict[str, Any],
    cost: dict[str, Any],
) -> dict[str, Any]:
    if validation_status not in RUN_VALIDATION_STATUSES:
        raise ValueError(f"Invalid run validation status: {validation_status}")
    initialize_llm_storage(db_path)
    run = {
        "run_id": _generate_id("run"),
        "job_id": job_id,
        "document_id": document_id,
        "prompt_id": prompt_id,
        "provider_id": provider_id,
        "model_id": model_id,
        "prompt_template_text": prompt_template_text,
        "rendered_prompt_text": rendered_prompt_text,
        "request_params": request_params,
        "raw_response": raw_response,
        "normalized_payload": normalized_payload,
        "validation_status": validation_status,
        "validation_errors": validation_errors,
        "usage": usage,
        "cost": cost,
        "created_at": _iso_utcnow(),
    }
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO llm_runs (
                run_id,
                job_id,
                document_id,
                prompt_id,
                provider_id,
                model_id,
                prompt_template_text,
                rendered_prompt_text,
                request_params_json,
                raw_response_text,
                normalized_payload_json,
                validation_status,
                validation_errors_json,
                usage_json,
                cost_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run["run_id"],
                run["job_id"],
                run["document_id"],
                run["prompt_id"],
                run["provider_id"],
                run["model_id"],
                run["prompt_template_text"],
                run["rendered_prompt_text"],
                json.dumps(run["request_params"], sort_keys=True),
                run["raw_response"],
                json.dumps(run["normalized_payload"], sort_keys=True)
                if run["normalized_payload"] is not None
                else None,
                run["validation_status"],
                json.dumps(run["validation_errors"]),
                json.dumps(run["usage"], sort_keys=True),
                json.dumps(run["cost"], sort_keys=True),
                run["created_at"],
            ),
        )
        connection.commit()
    return run


def get_llm_run(db_path: Path, run_id: str) -> dict[str, Any] | None:
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                run_id,
                job_id,
                document_id,
                prompt_id,
                provider_id,
                model_id,
                prompt_template_text,
                rendered_prompt_text,
                request_params_json,
                raw_response_text,
                normalized_payload_json,
                validation_status,
                validation_errors_json,
                usage_json,
                cost_json,
                created_at
            FROM llm_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_run(row)


def list_llm_runs_for_document(db_path: Path, document_id: str) -> list[dict[str, Any]]:
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                run_id,
                job_id,
                document_id,
                prompt_id,
                provider_id,
                model_id,
                prompt_template_text,
                rendered_prompt_text,
                request_params_json,
                raw_response_text,
                normalized_payload_json,
                validation_status,
                validation_errors_json,
                usage_json,
                cost_json,
                created_at
            FROM llm_runs
            WHERE document_id = ?
            ORDER BY created_at DESC, run_id DESC
            """,
            (document_id,),
        ).fetchall()
    return [_row_to_run(row) for row in rows]


def get_llm_models_by_document(db_path: Path) -> dict[str, list[str]]:
    """Return {internal_document_id: [model_ids]} for all documents with valid runs."""
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT document_id, model_id
            FROM llm_runs
            WHERE validation_status = 'valid'
            ORDER BY document_id, model_id
            """
        ).fetchall()
    result: dict[str, list[str]] = {}
    for doc_id, model_id in rows:
        result.setdefault(str(doc_id), []).append(model_id)
    return result


def get_llm_annotated_document_ids(db_path: Path) -> set[str]:
    """Return the set of internal document IDs that have at least one valid LLM run."""
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT DISTINCT document_id FROM llm_runs WHERE validation_status = 'valid'"
        ).fetchall()
    return {str(row[0]) for row in rows}


def list_all_valid_llm_runs(db_path: Path) -> list[dict[str, Any]]:
    """Return all valid LLM runs, ordered by document_id then most recent first."""
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                run_id, job_id, document_id, prompt_id, provider_id, model_id,
                prompt_template_text, rendered_prompt_text, request_params_json,
                raw_response_text, normalized_payload_json, validation_status,
                validation_errors_json, usage_json, cost_json, created_at
            FROM llm_runs
            WHERE validation_status = 'valid'
            ORDER BY document_id, created_at DESC, run_id DESC
            """
        ).fetchall()
    return [_row_to_run(row) for row in rows]


def list_llm_reporting_scopes(db_path: Path) -> list[dict[str, Any]]:
    """Return distinct provider/model scopes that have at least one valid run."""
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                provider_id,
                model_id,
                COUNT(DISTINCT document_id) AS document_count,
                MAX(created_at) AS latest_created_at
            FROM llm_runs
            WHERE validation_status = 'valid'
            GROUP BY provider_id, model_id
            ORDER BY provider_id, model_id
            """
        ).fetchall()
    return [
        {
            "provider_id": row[0],
            "model_id": row[1],
            "document_count": int(row[2]),
            "latest_created_at": row[3],
        }
        for row in rows
    ]


def list_latest_valid_runs_for_documents(
    db_path: Path,
    document_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Return the most recent valid LLM run for each document, keyed by document_id."""
    if not document_ids:
        return {}
    initialize_llm_storage(db_path)
    placeholders = ",".join("?" * len(document_ids))
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT
                r.run_id,
                r.job_id,
                r.document_id,
                r.prompt_id,
                r.provider_id,
                r.model_id,
                r.prompt_template_text,
                r.rendered_prompt_text,
                r.request_params_json,
                r.raw_response_text,
                r.normalized_payload_json,
                r.validation_status,
                r.validation_errors_json,
                r.usage_json,
                r.cost_json,
                r.created_at
            FROM llm_runs r
            INNER JOIN (
                SELECT document_id, MAX(created_at || run_id) AS latest_key
                FROM llm_runs
                WHERE validation_status = 'valid' AND document_id IN ({placeholders})
                GROUP BY document_id
            ) latest
                ON r.document_id = latest.document_id
                AND (r.created_at || r.run_id) = latest.latest_key
            WHERE r.validation_status = 'valid'
            """,
            document_ids,
        ).fetchall()
    return {row[2]: _row_to_run(row) for row in rows}


def list_latest_valid_runs_for_scope(
    db_path: Path,
    *,
    provider_id: str,
    model_id: str,
) -> dict[str, dict[str, Any]]:
    """Return the most recent valid run per document for one provider/model scope."""
    initialize_llm_storage(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                r.run_id,
                r.job_id,
                r.document_id,
                r.prompt_id,
                r.provider_id,
                r.model_id,
                r.prompt_template_text,
                r.rendered_prompt_text,
                r.request_params_json,
                r.raw_response_text,
                r.normalized_payload_json,
                r.validation_status,
                r.validation_errors_json,
                r.usage_json,
                r.cost_json,
                r.created_at
            FROM llm_runs r
            INNER JOIN (
                SELECT document_id, MAX(created_at || run_id) AS latest_key
                FROM llm_runs
                WHERE validation_status = 'valid'
                  AND provider_id = ?
                  AND model_id = ?
                GROUP BY document_id
            ) latest
                ON r.document_id = latest.document_id
                AND (r.created_at || r.run_id) = latest.latest_key
            WHERE r.validation_status = 'valid'
              AND r.provider_id = ?
              AND r.model_id = ?
            """
            ,
            (provider_id, model_id, provider_id, model_id),
        ).fetchall()
    return {row[2]: _row_to_run(row) for row in rows}


def _generate_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _iso_utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_plus_seconds(timestamp: str, seconds: int) -> str:
    parsed = datetime.fromisoformat(timestamp)
    return (parsed + timedelta(seconds=seconds)).isoformat()


def _ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {row[1] for row in rows}
    if column_name not in existing:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def _row_to_prompt(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "prompt_id": row[0],
        "parent_prompt_id": row[1],
        "label": row[2],
        "template_text": row[3],
        "created_by": row[4],
        "created_at": row[5],
        "updated_at": row[6],
    }


def _row_to_job(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "job_id": row[0],
        "status": row[1],
        "prompt_id": row[2],
        "provider_id": row[3],
        "model_id": row[4],
        "request_params": json.loads(row[5]),
        "created_by": row[6],
        "worker_id": row[7],
        "lease_expires_at": row[8],
        "heartbeat_at": row[9],
        "started_at": row[10],
        "finished_at": row[11],
        "created_at": row[12],
        "updated_at": row[13],
    }


def _row_to_job_document(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "job_id": row[0],
        "document_id": row[1],
        "status": row[2],
        "llm_run_id": row[3],
        "error_message": row[4],
        "started_at": row[5],
        "finished_at": row[6],
    }


def _row_to_run(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "run_id": row[0],
        "job_id": row[1],
        "document_id": row[2],
        "prompt_id": row[3],
        "provider_id": row[4],
        "model_id": row[5],
        "prompt_template_text": row[6],
        "rendered_prompt_text": row[7],
        "request_params": json.loads(row[8]),
        "raw_response": row[9],
        "normalized_payload": json.loads(row[10]) if row[10] is not None else None,
        "validation_status": row[11],
        "validation_errors": json.loads(row[12]),
        "usage": json.loads(row[13]),
        "cost": json.loads(row[14]),
        "created_at": row[15],
    }
