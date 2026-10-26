from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

from corn.llm_jobs import execute_claimed_job
from corn.llm_providers import get_provider
from corn.llm_storage import claim_next_job
from corn.project_config import load_project_config_content
from corn.provider_settings import load_provider_settings
from corn.querying import get_project_document_by_id
from corn.registry import list_projects


def process_next_job(
    *,
    db_path: Path,
    worker_id: str,
    settings_by_provider: dict[str, dict[str, Any]],
    project_name: str,
    tie_schema: list[dict[str, str]],
    resolve_document: Callable[[str], dict[str, Any]],
    tie_attribute_schema: list[dict[str, Any]] | None = None,
    lease_seconds: int = 60,
    log_path: Path | None = None,
    failed_log_path: Path | None = None,
) -> dict[str, Any] | None:
    claimed = claim_next_job(
        db_path,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
    )
    if claimed is None:
        return None

    provider = get_provider(
        claimed["provider_id"], settings_by_provider=settings_by_provider
    )
    result = execute_claimed_job(
        db_path=db_path,
        job_id=claimed["job_id"],
        worker_id=worker_id,
        provider=provider,
        project_name=project_name,
        tie_schema=tie_schema,
        tie_attribute_schema=tie_attribute_schema,
        resolve_document=resolve_document,
        lease_seconds=lease_seconds,
        log_path=log_path,
        failed_log_path=failed_log_path,
    )
    return {"job_id": claimed["job_id"], **result}


def process_registry_once(
    *,
    registry_path: Path,
    provider_settings_path: Path,
    worker_id: str,
    lease_seconds: int = 60,
) -> dict[str, Any] | None:
    settings_by_provider = load_provider_settings(provider_settings_path)
    for project in list_projects(registry_path, include_archived=False):
        project_path = Path(project["project_path"])
        config = load_project_config_content(
            filename="project.yaml",
            raw_content=(project_path / "project.yaml").read_text(encoding="utf-8"),
        )
        runtime_db_path = project_path / "derived" / "runtime.sqlite"
        annotations_db_path = project_path / "annotations" / "annotations.sqlite"
        log_path = project_path / "annotations" / "llm_events.jsonl"
        failed_log_path = project_path / "annotations" / "failed_llms.log"

        result = process_next_job(
            db_path=annotations_db_path,
            worker_id=worker_id,
            settings_by_provider=settings_by_provider,
            project_name=project["name"],
            tie_schema=config["tie_schema"],
            tie_attribute_schema=config["tie_attribute_schema"],
            resolve_document=lambda internal_id, runtime_db_path=runtime_db_path: (
                get_project_document_by_id(
                    runtime_db_path,
                    int(internal_id),
                )
            ),
            lease_seconds=lease_seconds,
            log_path=log_path,
            failed_log_path=failed_log_path,
        )
        if result is not None:
            return {"project_slug": project["slug"], **result}
    return None


def run_worker(
    *,
    registry_path: Path,
    provider_settings_path: Path,
    worker_id: str,
    lease_seconds: int = 60,
    poll_seconds: float = 2.0,
    once: bool = False,
    should_stop: Callable[[], bool] = lambda: False,
    process_once: Callable[..., dict[str, Any] | None] = process_registry_once,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    while not should_stop():
        result = process_once(
            registry_path=registry_path,
            provider_settings_path=provider_settings_path,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        if once:
            return
        if result is None:
            sleep(poll_seconds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the CORN LLM annotation worker.")
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--provider-settings-path", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--lease-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    run_worker(
        registry_path=args.registry_path,
        provider_settings_path=args.provider_settings_path,
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
        poll_seconds=args.poll_seconds,
        once=args.once,
    )


if __name__ == "__main__":
    main()
