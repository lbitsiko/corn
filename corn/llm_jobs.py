from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from corn.annotations import save_annotation
from corn.llm_logging import append_failed_llm_run, append_llm_event
from corn.llm_prompts import build_prompt_context, render_prompt_template
from corn.llm_storage import (
    complete_job,
    finalize_cancelled_job,
    get_job,
    get_job_document_items,
    get_prompt,
    heartbeat_job_lease,
    save_llm_run,
    set_job_document_status,
)
from corn.llm_validation import transform_llm_output_to_payload, validate_llm_intermediate_output
from corn.storage import save_annotation_record
from corn.utils import sanitize_filename


def _auto_save_llm_annotation(
    *,
    db_path: Path,
    normalized_payload: dict,
    job_id: str,
    document_id: str,
    log_path: Path | None,
) -> None:
    annotations_dir = db_path.parent
    doc_link = normalized_payload.get("document_id", document_id)
    relative_path = (
        Path(sanitize_filename(normalized_payload["workspace_id"]))
        / sanitize_filename(normalized_payload["annotator_id"])
        / f"{sanitize_filename(str(doc_link))}.json"
    )
    try:
        annotation_path = save_annotation(
            output_dir=annotations_dir,
            relative_path=relative_path,
            payload=normalized_payload,
        )
        save_annotation_record(
            db_path=db_path,
            payload=normalized_payload,
            json_path=annotation_path,
        )
    except Exception as exc:
        if log_path is not None:
            append_llm_event(
                log_path,
                {
                    "event_type": "llm_annotation_auto_save_failed",
                    "job_id": job_id,
                    "document_id": document_id,
                    "error": str(exc),
                },
            )


def execute_claimed_job(
    *,
    db_path,
    job_id: str,
    worker_id: str,
    provider,
    project_name: str,
    tie_schema: list[dict[str, str]],
    resolve_document: Callable[[str], dict[str, Any]],
    tie_attribute_schema: list[dict[str, Any]] | None = None,
    lease_seconds: int = 60,
    log_path=None,
    failed_log_path=None,
) -> dict[str, Any]:
    job = get_job(db_path, job_id)
    if job is None:
        raise ValueError(f"LLM job not found: {job_id}")
    if job["status"] != "running" or job["worker_id"] != worker_id:
        raise ValueError(f"Job is not actively claimed by worker {worker_id}: {job_id}")
    prompt = get_prompt(db_path, job["prompt_id"])
    if prompt is None:
        raise ValueError(f"Prompt not found: {job['prompt_id']}")

    processed_document_ids: list[str] = []
    valid_tie_labels = {item["label"] for item in tie_schema}

    for item in get_job_document_items(db_path, job_id):
        current_job = get_job(db_path, job_id)
        if current_job is None:
            raise ValueError(f"LLM job not found during execution: {job_id}")
        if current_job["status"] == "cancelled":
            break
        if item["status"] not in {"pending", "failed", "invalid"}:
            continue
        document_id = item["document_id"]
        processed_document_ids.append(document_id)
        heartbeat_job_lease(
            db_path,
            job_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        set_job_document_status(db_path, job_id, document_id, "running")
        provider_result: dict[str, Any] | None = None
        rendered_prompt_text = ""

        try:
            document = resolve_document(document_id)
            context = build_prompt_context(
                project_name=project_name,
                document=document,
                tie_schema=tie_schema,
                tie_attribute_schema=tie_attribute_schema,
            )
            rendered_prompt_text = render_prompt_template(prompt["template_text"], context)
            provider_result = provider.annotate(
                rendered_prompt_text=rendered_prompt_text,
                document_id=document_id,
                model_id=job["model_id"],
                request_params=job["request_params"],
            )
            parsed_output = provider_result["parsed_output"]
            validated_output = validate_llm_intermediate_output(
                parsed_output,
                valid_tie_labels=valid_tie_labels,
                tie_attribute_schema=tie_attribute_schema,
            )
            normalized_payload = transform_llm_output_to_payload(
                validated_output=validated_output,
                document=document,
                workspace_id="llm",
                annotator_id=f"{provider.provider_id}/{job['model_id']}",
                created_at=item["started_at"] or job["created_at"],
                updated_at=item["started_at"] or job["created_at"],
            )
            llm_run = save_llm_run(
                db_path=db_path,
                job_id=job_id,
                document_id=document_id,
                prompt_id=prompt["prompt_id"],
                provider_id=provider.provider_id,
                model_id=job["model_id"],
                prompt_template_text=prompt["template_text"],
                rendered_prompt_text=rendered_prompt_text,
                request_params=job["request_params"],
                raw_response=provider_result["raw_response"],
                normalized_payload=normalized_payload,
                validation_status="valid",
                validation_errors=[],
                usage=provider_result.get("usage", {}),
                cost=provider_result.get("cost", {}),
            )
            set_job_document_status(
                db_path,
                job_id,
                document_id,
                "succeeded",
                llm_run_id=llm_run["run_id"],
            )
            _auto_save_llm_annotation(
                db_path=db_path,
                normalized_payload=normalized_payload,
                job_id=job_id,
                document_id=document_id,
                log_path=log_path,
            )
            if log_path is not None:
                append_llm_event(
                    log_path,
                    {
                        "event_type": "llm_document_succeeded",
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "document_id": document_id,
                        "provider_id": provider.provider_id,
                        "model_id": job["model_id"],
                        "prompt_id": prompt["prompt_id"],
                        "rendered_prompt_text": rendered_prompt_text,
                        "raw_response": provider_result["raw_response"],
                        "validation_status": "valid",
                        "validation_errors": [],
                    },
                )
        except Exception as exc:
            if exc.__class__.__name__ == "ProviderExecutionError":
                raw_response = getattr(exc, "raw_response", "")
                set_job_document_status(
                    db_path,
                    job_id,
                    document_id,
                    "failed",
                    error_message=str(exc),
                )
                if log_path is not None:
                    append_llm_event(
                        log_path,
                        {
                            "event_type": "llm_document_failed",
                            "job_id": job_id,
                            "worker_id": worker_id,
                            "document_id": document_id,
                            "provider_id": provider.provider_id,
                            "model_id": job["model_id"],
                            "prompt_id": prompt["prompt_id"],
                            "rendered_prompt_text": rendered_prompt_text,
                            "raw_response": raw_response
                            or (provider_result.get("raw_response", "") if provider_result is not None else ""),
                            "validation_status": "failed",
                            "error": str(exc),
                        },
                    )
                if failed_log_path is not None:
                    append_failed_llm_run(
                        failed_log_path,
                        {
                            "event_type": "llm_document_failed",
                            "job_id": job_id,
                            "worker_id": worker_id,
                            "document_id": document_id,
                            "provider_id": provider.provider_id,
                            "model_id": job["model_id"],
                            "prompt_id": prompt["prompt_id"],
                            "rendered_prompt_text": rendered_prompt_text,
                            "raw_response": raw_response
                            or (provider_result.get("raw_response", "") if provider_result is not None else ""),
                            "validation_status": "failed",
                            "error": str(exc),
                        },
                    )
                continue

            raw_response = ""
            if provider_result is not None:
                raw_response = provider_result.get("raw_response", "")
            llm_run = save_llm_run(
                db_path=db_path,
                job_id=job_id,
                document_id=document_id,
                prompt_id=prompt["prompt_id"],
                provider_id=provider.provider_id,
                model_id=job["model_id"],
                prompt_template_text=prompt["template_text"],
                rendered_prompt_text=rendered_prompt_text,
                request_params=job["request_params"],
                raw_response=raw_response,
                normalized_payload=None,
                validation_status="invalid",
                validation_errors=[str(exc)],
                usage=provider_result.get("usage", {}) if provider_result is not None else {},
                cost=provider_result.get("cost", {}) if provider_result is not None else {},
            )
            set_job_document_status(
                db_path,
                job_id,
                document_id,
                "invalid",
                llm_run_id=llm_run["run_id"],
                error_message=str(exc),
            )
            if log_path is not None:
                append_llm_event(
                    log_path,
                    {
                        "event_type": "llm_document_invalid",
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "document_id": document_id,
                        "provider_id": provider.provider_id,
                        "model_id": job["model_id"],
                        "prompt_id": prompt["prompt_id"],
                        "rendered_prompt_text": rendered_prompt_text,
                        "raw_response": raw_response,
                        "validation_status": "invalid",
                        "validation_errors": [str(exc)],
                        "error": str(exc),
                    },
                )
            if failed_log_path is not None:
                append_failed_llm_run(
                    failed_log_path,
                    {
                        "event_type": "llm_document_invalid",
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "document_id": document_id,
                        "provider_id": provider.provider_id,
                        "model_id": job["model_id"],
                        "prompt_id": prompt["prompt_id"],
                        "rendered_prompt_text": rendered_prompt_text,
                        "raw_response": raw_response,
                        "validation_status": "invalid",
                        "validation_errors": [str(exc)],
                        "error": str(exc),
                    },
                )

    final_job = get_job(db_path, job_id)
    if final_job is None:
        raise ValueError(f"LLM job disappeared during completion: {job_id}")
    if final_job["status"] == "cancelled":
        finalize_cancelled_job(db_path, job_id, worker_id=worker_id)
    else:
        complete_job(db_path, job_id, worker_id=worker_id)
    return {"job_id": job_id, "processed_document_ids": processed_document_ids}
