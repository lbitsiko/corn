from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import streamlit as st
import yaml

from corn.app_paths import build_app_paths, initialize_data_directory
from corn.annotations import (
    IncompleteAnnotationError,
    append_network_entry,
    build_highlight_spans,
    build_network_entry,
    find_reasoning_spans,
    load_annotation,
    save_annotation,
)
from corn.annotation_io import export_document_annotation, import_document_annotation
from corn.attribute_validation import validate_tie_attributes
from corn.components.document_selector import document_selection
from corn.demo_project import DEMO_PROJECT_SLUG, install_demo_project
from corn.llm_prompts import SUPPORTED_PROMPT_PLACEHOLDERS, build_prompt_context, render_prompt_template
from corn.llm_providers import (
    get_provider_config,
    list_available_models_for_provider,
    list_models_for_provider,
    list_provider_ids,
)
from corn.provider_settings import load_provider_settings, save_provider_settings
from corn.reporting import build_pairwise_llm_report
from corn.llm_storage import (
    cancel_job,
    create_llm_job,
    create_prompt,
    fork_prompt,
    get_llm_models_by_document,
    get_project_default_prompt_id,
    get_project_llm_settings,
    get_prompt,
    list_all_valid_llm_runs,
    list_jobs,
    list_latest_valid_runs_for_scope,
    list_llm_reporting_scopes,
    list_llm_runs_for_document,
    list_prompts,
    set_project_default_prompt_id,
    set_project_llm_settings,
    summarize_job,
)
from corn.project_config import (
    VALID_FILTER_TYPES,
    load_project_config_content,
    validate_project_config,
)
from corn.projects import (
    build_project_runtime,
    create_and_register_project,
    get_project_status,
)
from corn.querying import (
    fetch_project_filter_options,
    get_project_document_by_id,
    query_project_documents,
)
from corn.registry import list_projects
from corn.storage import (
    export_annotations_edge_list_csv,
    export_annotations_json_archive,
    get_annotation,
    list_annotations,
    save_annotation_record,
)
from corn.utils import ensure_directory, load_tie_options, sanitize_filename
from corn.source_inspection import (
    build_guided_project_config,
    discover_source_files,
    inspect_csv_columns,
    inspect_sqlite_columns,
    inspect_sqlite_tables,
)

logging.basicConfig(level=logging.INFO)

APP_PATHS = build_app_paths()
APP_ROOT = APP_PATHS.resource_dir
LOGOS_DIR = APP_PATHS.logos_dir
DATA_DIR = APP_PATHS.data_dir
PROJECTS_DIR = APP_PATHS.projects_dir
REGISTRY_PATH = APP_PATHS.registry_path
TIE_OPTIONS_PATH = APP_PATHS.tie_options_path
ANNOTATIONS_DIR = APP_PATHS.raw_annotations_dir
ANNOTATIONS_DB_PATH = APP_PATHS.legacy_annotations_db_path
DEFAULT_WORKSPACE_ID = "default"
GUIDED_UPLOADS_DIR = APP_PATHS.guided_uploads_dir
GUIDED_SERVER_SOURCE_IGNORED_ROOTS = (PROJECTS_DIR, ANNOTATIONS_DIR, GUIDED_UPLOADS_DIR)
GUIDED_SERVER_SOURCE_IGNORED_FILES = (ANNOTATIONS_DB_PATH, REGISTRY_PATH)
PROVIDER_SETTINGS_PATH = APP_PATHS.provider_settings_path
CORN_APP_NAME = "CORN"
CORN_APP_SUBTITLE = "Corpus-Oriented Relationship Notation"
DEFAULT_LLM_CREATED_BY = "system"
RESERVED_LLM_WORKSPACE_ID = "llm"
UNSET_ATTRIBUTE_LABEL = "(unset)"


def _annotation_relative_path(
    document: dict,
    workspace_id: str,
    annotator_id: str,
) -> Path:
    return (
        Path(sanitize_filename(workspace_id))
        / sanitize_filename(annotator_id)
        / f"{sanitize_filename(str(document['link']))}.json"
    )


def _annotation_path(document: dict, workspace_id: str, annotator_id: str) -> Path:
    return ANNOTATIONS_DIR / _annotation_relative_path(document, workspace_id, annotator_id)


def _project_annotation_path(
    annotations_dir: Path,
    document: dict,
    workspace_id: str,
    annotator_id: str,
) -> Path:
    return annotations_dir / _annotation_relative_path(document, workspace_id, annotator_id)


def _legacy_annotation_path(annotations_dir: Path, document: dict) -> Path:
    return annotations_dir / f"{sanitize_filename(str(document['link']))}.json"


def _iso_utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_annotation_payload(
    document: dict,
    workspace_id: str,
    annotator_id: str,
) -> dict:
    timestamp = _iso_utcnow()
    return {
        "workspace_id": workspace_id,
        "annotator_id": annotator_id,
        "annotation_source": "human",
        "document_id": document["link"],
        "created_at": timestamp,
        "updated_at": timestamp,
        "network_data": [],
    }


def _default_llm_prompt_template() -> str:
    return """You are annotating documents for a relationship annotation project.

Project: {{project_name}}
Document ID: {{document_id}}
Document title: {{document_title}}

Metadata:
{{metadata_json}}

Tie schema:
{{tie_schema_json}}

Tie attributes (optional, may be empty):
{{tie_attribute_schema_json}}

Each tie may include an "attributes" object. Only set an attribute when the document
text clearly supports it; omit it otherwise. Never invent values.
Categorical attribute values must be exactly one of the listed options, verbatim.
The "end" date is optional; omit it for a point in time. Dates must be zero-padded
(e.g. 2019-06, not 2019-6).

Document text:
{{document_text}}

Return JSON only. The JSON must match this schema exactly:
{{annotation_schema_json}}
"""


def _ensure_project_default_llm_prompt(db_path: Path, project_name: str) -> dict:
    default_prompt_id = get_project_default_prompt_id(db_path)
    if default_prompt_id:
        existing_prompt = get_prompt(db_path, default_prompt_id)
        if existing_prompt is not None:
            return existing_prompt

    prompt = create_prompt(
        db_path=db_path,
        template_text=_default_llm_prompt_template(),
        created_by=DEFAULT_LLM_CREATED_BY,
        label=f"{project_name} default LLM prompt",
    )
    set_project_default_prompt_id(db_path, prompt["prompt_id"])
    return prompt


def _persist_llm_prompt_template(
    db_path: Path,
    *,
    selected_prompt_id: str | None,
    template_text: str,
    created_by: str,
    label: str,
    promote_to_project_default: bool,
) -> dict:
    normalized_template = template_text.strip()
    if not normalized_template:
        raise ValueError("Prompt template cannot be empty.")

    prompt: dict | None = get_prompt(db_path, selected_prompt_id) if selected_prompt_id else None
    if prompt is not None and prompt["template_text"] == normalized_template:
        saved_prompt = prompt
    elif prompt is not None:
        saved_prompt = fork_prompt(
            db_path=db_path,
            parent_prompt_id=prompt["prompt_id"],
            template_text=normalized_template,
            created_by=created_by,
            label=label,
        )
    else:
        saved_prompt = create_prompt(
            db_path=db_path,
            template_text=normalized_template,
            created_by=created_by,
            label=label,
        )

    if promote_to_project_default:
        set_project_default_prompt_id(db_path, saved_prompt["prompt_id"])
    return saved_prompt


def _llm_prompt_select_options(
    prompts: list[dict],
    selected_prompt_id: str | None,
) -> tuple[dict[str, str], int]:
    options: dict[str, str] = {}
    for prompt in prompts:
        label = f"{prompt['label']} | {prompt['prompt_id']}"
        options[label] = prompt["prompt_id"]

    selected_index = 0
    values = list(options.values())
    if selected_prompt_id in values:
        selected_index = values.index(selected_prompt_id)
    return options, selected_index


def _query_result_document_ids(results: list[dict]) -> list[str]:
    return [str(row["id"]) for row in results]


def _single_document_job_ids(document: dict) -> list[str]:
    return [str(document["id"])]


def _provider_status_message(provider_id: str) -> tuple[str, str]:
    config = get_provider_config(
        provider_id,
        stored_settings=load_provider_settings(PROVIDER_SETTINGS_PATH).get(provider_id),
    )
    if config.configured:
        if config.base_url:
            return ("success", f"{provider_id} configured via {config.base_url}")
        return ("success", f"{provider_id} configured")
    env_hint = ", ".join(config.env_vars)
    return ("warning", f"{provider_id} is not configured. Set {env_hint}.")


def _provider_settings_summary(provider_id: str) -> dict:
    stored_settings = load_provider_settings(PROVIDER_SETTINGS_PATH).get(provider_id, {})
    config = get_provider_config(provider_id, stored_settings=stored_settings)
    return {
        "provider_id": provider_id,
        "configured": config.configured,
        "base_url": config.base_url,
        "has_api_key": bool(config.api_key),
        "requires_api_key": config.requires_api_key,
        "env_vars": list(config.env_vars),
        "uses_stored_settings": bool(stored_settings),
    }


def _provider_uses_freeform_model_id(provider_id: str) -> bool:
    return provider_id in {"gemini", "openrouter"}


def _save_provider_settings_from_form(
    provider_id: str,
    *,
    base_url: str,
) -> None:
    settings_by_provider = load_provider_settings(PROVIDER_SETTINGS_PATH)
    provider_settings: dict[str, str] = dict(settings_by_provider.get(provider_id, {}))
    if base_url.strip():
        provider_settings["base_url"] = base_url.strip()
    provider_settings.pop("api_key", None)
    settings_by_provider[provider_id] = provider_settings
    save_provider_settings(PROVIDER_SETTINGS_PATH, settings_by_provider)


def _clear_provider_settings(provider_id: str) -> None:
    settings_by_provider = load_provider_settings(PROVIDER_SETTINGS_PATH)
    settings_by_provider.pop(provider_id, None)
    save_provider_settings(PROVIDER_SETTINGS_PATH, settings_by_provider)


def _llm_worker_status(job_rows: list[dict]) -> tuple[str, str] | None:
    if not job_rows:
        return None

    queued_rows = [row for row in job_rows if row["status"] == "queued"]
    if not queued_rows:
        return None

    running_rows = [row for row in job_rows if row["status"] == "running" and row.get("worker_id")]
    if not running_rows:
        recent_worker_activity = False
        now = datetime.now(timezone.utc)
        for row in job_rows:
            timestamp = row.get("heartbeat_at") or row.get("finished_at")
            if not timestamp:
                continue
            try:
                activity_time = datetime.fromisoformat(timestamp)
            except ValueError:
                continue
            if activity_time.tzinfo is None:
                activity_time = activity_time.replace(tzinfo=timezone.utc)
            if (now - activity_time).total_seconds() <= 30:
                recent_worker_activity = True
                break
        if recent_worker_activity:
            return (
                "info",
                "LLM jobs are queued and a worker was active recently. Use Refresh in Recent LLM jobs to monitor progress.",
            )
        return (
            "warning",
            "LLM jobs are queued but no worker is currently processing them. "
            "Start `python -m corn.llm_worker --registry-path data/project_registry.yaml "
            "--provider-settings-path data/provider_settings.json --worker-id worker-1`.",
        )

    return (
        "info",
        "LLM jobs are queued and a worker appears to be running. Use Refresh in Recent LLM jobs to monitor progress.",
    )


def _queue_save_feedback(saved_links: list[str], errors: list[str]) -> None:
    st.session_state.session_save_feedback = {
        "saved_links": list(saved_links),
        "errors": list(errors),
    }


def _annotation_state_fingerprint(annotation_state: dict | None) -> str | None:
    if annotation_state is None:
        return None
    return json.dumps(annotation_state, sort_keys=True)


def _set_annotation_state_baseline(document_id: str, annotation_state: dict | None) -> None:
    if "annotation_state_baselines" not in st.session_state:
        st.session_state.annotation_state_baselines = {}
    st.session_state.annotation_state_baselines[document_id] = _annotation_state_fingerprint(annotation_state)


def _sync_dirty_state_for_document(document_id: str) -> None:
    if "annotation_states" not in st.session_state:
        st.session_state.annotation_states = {}
    if "annotation_state_baselines" not in st.session_state:
        st.session_state.annotation_state_baselines = {}
    if "dirty_document_ids" not in st.session_state:
        st.session_state.dirty_document_ids = set()
    current_state = st.session_state.annotation_states.get(document_id)
    baseline = st.session_state.annotation_state_baselines.get(document_id)
    current_fingerprint = _annotation_state_fingerprint(current_state)
    if current_fingerprint is None or baseline == current_fingerprint:
        st.session_state.dirty_document_ids.discard(document_id)
    else:
        st.session_state.dirty_document_ids.add(document_id)


def _sync_current_annotation_state_for_document(document_id: str, annotation_state: dict | None = None) -> None:
    if "annotation_states" not in st.session_state:
        st.session_state.annotation_states = {}
    if annotation_state is None:
        annotation_state = st.session_state.get("annotation_state")
    st.session_state.annotation_states[document_id] = annotation_state
    _sync_dirty_state_for_document(document_id)


def _rendered_llm_prompt_preview(
    *,
    project_name: str,
    document: dict,
    tie_schema: list[dict[str, str]],
    template_text: str,
    tie_attribute_schema: list[dict] | None = None,
) -> str:
    context = build_prompt_context(
        project_name=project_name,
        document=document,
        tie_schema=tie_schema,
        tie_attribute_schema=tie_attribute_schema,
    )
    return render_prompt_template(template_text, context)


def _llm_run_label(run: dict) -> str:
    return (
        f"LLM | {run['provider_id']}/{run['model_id']} | {run['prompt_id']} | "
        f"{run['validation_status']} | {run['created_at']}"
    )


def _human_annotation_label(workspace_id: str, annotator_id: str, payload: dict) -> str:
    updated_at = payload.get("updated_at") or payload.get("created_at") or "unsaved"
    owner = annotator_id or "unassigned"
    return f"Human | {workspace_id}/{owner} | {updated_at}"


def _annotation_selector_options(
    *,
    workspace_id: str,
    annotator_id: str,
    human_payload: dict,
    llm_runs: list[dict],
) -> tuple[dict[str, dict], str]:
    human_label = _human_annotation_label(workspace_id, annotator_id, human_payload)
    options: dict[str, dict] = {
        human_label: {
            "record_type": "human",
            "payload": human_payload,
        }
    }
    default_label = human_label
    for run in llm_runs:
        label = _llm_run_label(run)
        options[label] = {
            "record_type": "llm",
            "run": run,
        }
        if default_label == human_label and not human_payload.get("network_data"):
            default_label = label
    return options, default_label


def _pending_selector_override_key(selector_state_key: str) -> str:
    return f"{selector_state_key}__pending_override"


def _resolve_annotation_record_selection(
    *,
    session_state: dict,
    selector_state_key: str,
    selector_options: dict[str, dict],
    default_record_label: str,
) -> None:
    pending_key = _pending_selector_override_key(selector_state_key)
    pending_override = session_state.pop(pending_key, None)
    if pending_override in selector_options:
        session_state[selector_state_key] = pending_override
    current_selection = session_state.get(selector_state_key)
    if current_selection not in selector_options:
        session_state[selector_state_key] = default_record_label


def _llm_job_summary_rows(db_path: Path, limit: int = 10) -> list[dict]:
    rows: list[dict] = []
    for job in list_jobs(db_path)[:limit]:
        summary = summarize_job(db_path, job["job_id"])
        rows.append(
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "worker_id": job["worker_id"],
                "provider_id": job["provider_id"],
                "model_id": job["model_id"],
                "prompt_id": job["prompt_id"],
                "started_at": job["started_at"],
                "heartbeat_at": job["heartbeat_at"],
                "lease_expires_at": job["lease_expires_at"],
                "finished_at": job["finished_at"],
                "created_at": job["created_at"],
                **summary,
            }
        )
    return rows


def _duplicate_llm_payload_for_human_editing(
    normalized_payload: dict,
    *,
    workspace_id: str,
    annotator_id: str,
) -> dict:
    duplicated_payload = json.loads(json.dumps(normalized_payload))
    duplicated_payload["workspace_id"] = workspace_id
    duplicated_payload["annotator_id"] = annotator_id
    duplicated_payload["annotation_source"] = "human_from_llm"
    duplicated_payload["created_at"] = _iso_utcnow()
    duplicated_payload["updated_at"] = duplicated_payload["created_at"]
    return duplicated_payload


def _launch_llm_annotation_job(
    *,
    db_path: Path,
    runtime_db_path: Path,
    project_name: str,
    tie_schema: list[dict[str, str]],
    document_ids: list[str],
    selected_prompt_id: str,
    template_text: str,
    created_by: str,
    prompt_label: str,
    provider_id: str,
    model_id: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[dict, dict[str, int]]:
    run_prompt = _persist_llm_prompt_template(
        db_path,
        selected_prompt_id=selected_prompt_id,
        template_text=template_text,
        created_by=created_by,
        label=prompt_label,
        promote_to_project_default=False,
    )
    set_project_llm_settings(
        db_path,
        provider_id=provider_id,
        model_id=model_id,
        temperature=float(temperature),
        max_output_tokens=max_output_tokens,
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=document_ids,
        prompt_id=run_prompt["prompt_id"],
        provider_id=provider_id,
        model_id=model_id,
        request_params={
            "temperature": float(temperature),
            "max_output_tokens": max_output_tokens,
        },
        created_by=created_by,
    )
    return run_prompt, summarize_job(db_path, job["job_id"])


def _theme_base() -> str:
    theme_base = st.get_option("theme.base")
    if theme_base in {"light", "dark"}:
        return theme_base
    return "light"


def _branding_assets(theme_base: str) -> dict[str, Path]:
    return {
        "home_logo": LOGOS_DIR / "CORN-with-text-WITH-builtin-background.svg",
        "page_logo": LOGOS_DIR / "CORN-log-WITH-builtin-background.svg",
        "page_icon": LOGOS_DIR / "CORN-icon-white-background-square.svg",
    }


def _render_brand_header(show_home_screen: bool) -> None:
    assets = _branding_assets(_theme_base())
    logo_path = assets["home_logo"] if show_home_screen else assets["page_logo"]
    if show_home_screen:
        _left, center, _right = st.columns([1, 2, 1])
        with center:
            st.image(str(logo_path), width="stretch")
    else:
        st.image(str(logo_path), width=180)
    if show_home_screen:
        st.caption(CORN_APP_SUBTITLE)


def _annotation_scope_key(workspace_id: str, annotator_id: str) -> str:
    return f"{workspace_id}:{annotator_id}"


def _blank_tie(default_type: str | None) -> dict:
    if default_type:
        tie_type = default_type
    else:
        tie_type = ""
    return {
        "type": tie_type,
        "reasoning_text": "",
        "reasoning_text_span": "",
        "reasoning_text_note": "",
        "reasoning_start": None,
        "reasoning_end": None,
        "selected_match": None,
        "attributes": {},
    }


def _apply_validated_tie_attribute(tie: dict, attribute: dict, candidate: object) -> None:
    """Write a candidate attribute value only if the shared save-gate validator accepts it."""
    name = attribute["name"]
    try:
        normalized = validate_tie_attributes({name: candidate}, tie_attribute_schema=[attribute])
    except ValueError as error:
        st.warning(str(error))
    else:
        tie["attributes"][name] = normalized[name]


def _render_tie_attribute_input(tie: dict, attribute: dict, tie_key_prefix: str) -> None:
    name = attribute["name"]
    widget_key = f"{tie_key_prefix}_attr_{name}"
    current = tie["attributes"].get(name)

    if attribute["type"] == "categorical":
        options = [UNSET_ATTRIBUTE_LABEL, *attribute["options"]]
        if current is not None and current not in attribute["options"]:
            options.append(current)
            st.warning(f"{name} value '{current}' is not in the current schema options")
        index = options.index(current) if current in options[1:] else 0
        selected = st.selectbox(
            name, options=options, index=index, key=widget_key, help=attribute["description"],
        )
        if selected == UNSET_ATTRIBUTE_LABEL:
            tie["attributes"].pop(name, None)
        else:
            tie["attributes"][name] = selected
    elif attribute["type"] == "numeric":
        minimum, maximum = attribute["range"]
        raw = st.text_input(
            f"{name} ({minimum}–{maximum})",
            value="" if current is None else str(current),
            key=widget_key,
            help=attribute["description"],
        )
        if not raw.strip():
            tie["attributes"].pop(name, None)
        else:
            _apply_validated_tie_attribute(tie, attribute, raw.strip())
    else:  # temporal
        if current and not isinstance(current, dict):
            st.warning(
                f"{name} has an invalid stored value; set a new period or leave unset to discard it"
            )
            current = {}
        current_start = (current or {}).get("start", "")
        current_end = (current or {}).get("end", "")
        start = st.text_input(
            f"{name} start (YYYY[-MM[-DD]])", value=current_start,
            key=f"{widget_key}__start", help=attribute["description"],
        )
        end = st.text_input(
            f"{name} end (optional)", value=current_end, key=f"{widget_key}__end",
        )
        if not start.strip():
            tie["attributes"].pop(name, None)
        else:
            candidate = {"start": start.strip()}
            if end.strip():
                candidate["end"] = end.strip()
            _apply_validated_tie_attribute(tie, attribute, candidate)


def _render_unknown_tie_attributes(
    tie: dict, tie_attribute_schema: list[dict], tie_key_prefix: str
) -> None:
    """Surface attribute names not in the current schema so they can be removed.

    The lenient loader keeps such values (e.g. an attribute dropped from the
    schema after annotating), the schema-driven widgets never render them, and
    the save gate rejects them. This gives the annotator an in-app escape hatch.
    """
    schema_names = {attribute["name"] for attribute in tie_attribute_schema}
    unknown_names = sorted(set(tie.get("attributes", {})) - schema_names)
    if not unknown_names:
        return
    st.warning(
        f"Attributes not in the current schema: {', '.join(unknown_names)} "
        "(saving is blocked while present)"
    )
    if st.button("Remove unknown attributes", key=f"{tie_key_prefix}_remove_unknown_attrs"):
        for unknown_name in unknown_names:
            tie["attributes"].pop(unknown_name, None)
        st.rerun()


def _blank_network_entry(default_type: str | None) -> dict:
    return {"source": "", "target": "", "ties": [_blank_tie(default_type)]}


def _copied_tie(existing_tie: dict, default_type: str | None) -> dict:
    if default_type:
        tie_type = default_type
    else:
        tie_type = ""
    span_text = str(existing_tie.get("reasoning_text_span", "") or "").strip()
    note_text = str(existing_tie.get("reasoning_text_note", "") or "").strip()
    legacy_text = str(existing_tie.get("reasoning_text", "") or "").strip()
    if not span_text and not note_text:
        if legacy_text and existing_tie.get("reasoning_start") is not None and existing_tie.get("reasoning_end") is not None:
            span_text = legacy_text
        else:
            note_text = legacy_text
    return {
        "type": tie_type,
        "reasoning_text": span_text or note_text,
        "reasoning_text_span": span_text,
        "reasoning_text_note": note_text,
        "reasoning_start": existing_tie.get("reasoning_start"),
        "reasoning_end": existing_tie.get("reasoning_end"),
        "selected_match": existing_tie.get("selected_match"),
        "attributes": {},
    }


def _leniently_normalized_attributes(
    attributes: dict, tie_attribute_schema: list[dict]
) -> dict:
    """Normalize attribute values the way the editor render would write them back.

    Verbatim-imported payloads may carry unnormalized values (int for numeric,
    numeric strings, nulls). Normalizing per single-name schema entry keeps the
    editor baseline aligned with rendered state so untouched documents never
    look dirty. Invalid and unknown-name values are kept as-is so the editor
    can surface them and the save gate can report them.
    """
    schema_by_name = {entry["name"]: entry for entry in tie_attribute_schema}
    normalized: dict = {}
    for name, value in attributes.items():
        entry = schema_by_name.get(name)
        if entry is None:
            normalized[name] = value
            continue
        try:
            validated = validate_tie_attributes({name: value}, tie_attribute_schema=[entry])
        except ValueError:
            normalized[name] = value
        else:
            # A null value validates to an empty dict; unset means absent.
            if name in validated:
                normalized[name] = validated[name]
    return normalized


def _normalize_loaded_annotation(
    payload: dict,
    default_type: str | None,
    tie_attribute_schema: list[dict] | None = None,
) -> dict:
    payload.setdefault("workspace_id", "")
    payload.setdefault("annotator_id", "")
    payload.setdefault("annotation_source", "human")
    payload.setdefault("created_at", "")
    payload.setdefault("updated_at", payload["created_at"])
    network_data = payload.get("network_data", [])
    normalized_entries: list[dict] = []
    for entry in network_data:
        ties = entry.get("ties", [])
        normalized_ties = []
        for tie in ties:
            span_text = str(tie.get("reasoning_text_span", "") or "").strip()
            note_text = str(tie.get("reasoning_text_note", "") or "").strip()
            legacy_text = str(tie.get("reasoning_text", "") or "").strip()
            if not span_text and not note_text:
                if legacy_text and tie.get("reasoning_start") is not None and tie.get("reasoning_end") is not None:
                    span_text = legacy_text
                else:
                    note_text = legacy_text
            attributes = tie.get("attributes")
            if isinstance(attributes, dict):
                # Copy nested temporal dicts too, so editor mutations never alias the loaded payload.
                attributes = {
                    name: dict(value) if isinstance(value, dict) else value
                    for name, value in attributes.items()
                }
                attributes = _leniently_normalized_attributes(
                    attributes, tie_attribute_schema or []
                )
            else:
                attributes = {}
            normalized_ties.append(
                {
                    "type": tie.get("type", default_type or ""),
                    "reasoning_text": span_text or note_text,
                    "reasoning_text_span": span_text,
                    "reasoning_text_note": note_text,
                    "reasoning_start": tie.get("reasoning_start"),
                    "reasoning_end": tie.get("reasoning_end"),
                    "selected_match": None,
                    "attributes": attributes,
                }
            )
        if not normalized_ties:
            normalized_ties.append(_blank_tie(default_type))
        normalized_entries.append(
            {
                "source": entry.get("source", ""),
                "target": entry.get("target", ""),
                "ties": normalized_ties,
            }
        )
    payload["network_data"] = normalized_entries
    return payload


def _pending_span_target_template() -> dict:
    return {
        "document_id": None,
        "entry_index": None,
        "tie_index": None,
        "nonce": None,
    }


def _pending_create_entry_template() -> dict:
    return {
        "document_id": None,
        "text": "",
        "start": None,
        "end": None,
        "nonce": None,
    }


def _create_entry_form_defaults(default_type: str | None) -> dict:
    return {
        "source": "",
        "target": "",
        "tie_type": default_type or "",
    }


def _focus_request(document_id: int, start: int, end: int, nonce: int) -> dict:
    return {
        "document_id": document_id,
        "start": start,
        "end": end,
        "nonce": f"{document_id}:{start}:{end}:{nonce}",
    }


def _document_focus_request_for_document(
    request: dict | None, current_document_id: int
) -> dict | None:
    if not request:
        return None
    if request.get("document_id") != current_document_id:
        return None
    start = request.get("start")
    end = request.get("end")
    nonce = request.get("nonce")
    if not isinstance(start, int) or not isinstance(end, int) or not nonce:
        return None
    return request


def _queue_document_focus_request(document_id: int, start: int, end: int) -> None:
    next_nonce = st.session_state.get("document_focus_request_counter", 0) + 1
    st.session_state.document_focus_request_counter = next_nonce
    st.session_state.document_focus_request = _focus_request(
        document_id=document_id,
        start=start,
        end=end,
        nonce=next_nonce,
    )


def _reset_pending_span_target() -> None:
    st.session_state.pending_span_target = _pending_span_target_template()


def _reset_pending_create_entry(default_type: str | None) -> None:
    st.session_state.pending_create_entry = _pending_create_entry_template()
    st.session_state.create_entry_form = _create_entry_form_defaults(default_type)


def _queue_pending_span_target(document_id: int, entry_index: int, tie_index: int) -> None:
    st.session_state.pending_span_target = {
        "document_id": document_id,
        "entry_index": entry_index,
        "tie_index": tie_index,
        "nonce": f"{document_id}:{entry_index}:{tie_index}:{_iso_utcnow()}",
    }


def _queue_pending_create_entry(
    document_id: int, selection: dict, default_type: str | None
) -> None:
    st.session_state.pending_create_entry = {
        "document_id": document_id,
        "text": str(selection.get("text", "") or "").strip(),
        "start": selection.get("start"),
        "end": selection.get("end"),
        "nonce": selection.get("nonce"),
    }
    st.session_state.create_entry_form = _create_entry_form_defaults(default_type)


def _apply_pending_selection_to_target_tie(
    *,
    annotation_state: dict,
    pending_selection: dict,
    pending_span_target: dict,
    session_state: dict | None = None,
    widget_key_prefix: str | None = None,
) -> bool:
    document_id = pending_selection.get("document_id")
    if not document_id or pending_span_target.get("document_id") != document_id:
        return False
    entry_index = pending_span_target.get("entry_index")
    tie_index = pending_span_target.get("tie_index")
    if not isinstance(entry_index, int) or not isinstance(tie_index, int):
        return False

    try:
        tie = annotation_state["network_data"][entry_index]["ties"][tie_index]
    except (IndexError, KeyError, TypeError):
        return False

    span_text = str(pending_selection.get("text", "") or "").strip()
    if not span_text:
        return False
    tie["reasoning_text_span"] = span_text
    tie["reasoning_text"] = span_text
    tie["reasoning_start"] = pending_selection.get("start")
    tie["reasoning_end"] = pending_selection.get("end")
    tie["selected_match"] = 0
    if session_state is not None and widget_key_prefix:
        session_state[f"{widget_key_prefix}_reasoning_span"] = span_text
    return True


def _append_network_entry_from_selection(
    *,
    annotation_state: dict,
    selection: dict,
    source: str,
    target: str,
    tie_type: str,
) -> None:
    span_text = str(selection.get("text", "") or "").strip()
    if not span_text:
        return
    entry = {
        "source": source.strip(),
        "target": target.strip(),
        "ties": [
            {
                "type": tie_type.strip(),
                "reasoning_text": span_text,
                "reasoning_text_span": span_text,
                "reasoning_text_note": "",
                "reasoning_start": selection.get("start"),
                "reasoning_end": selection.get("end"),
                "selected_match": 0,
                "attributes": {},
            }
        ],
    }
    append_network_entry(annotation_state, entry)


@st.dialog("Create annotation entry", width="large")
def _render_create_entry_dialog(document: dict, tie_options: list[str]) -> None:
    pending_entry = st.session_state.pending_create_entry
    form_state = st.session_state.create_entry_form
    default_type = tie_options[0] if tie_options else None

    if pending_entry.get("document_id") != document["id"]:
        return

    st.caption("Selected span")
    st.write(pending_entry["text"])
    st.caption(f"Document span: {pending_entry['start']}-{pending_entry['end']}")

    with st.form("create_annotation_entry_form"):
        form_state["source"] = st.text_input("Source node", value=form_state["source"])
        form_state["target"] = st.text_input("Target node", value=form_state["target"])
        if tie_options:
            selected_index = 0
            if form_state["tie_type"] in tie_options:
                selected_index = tie_options.index(form_state["tie_type"])
            form_state["tie_type"] = st.selectbox(
                "Tie description",
                options=tie_options,
                index=selected_index,
            )
        else:
            form_state["tie_type"] = st.text_input("Tie description", value=form_state["tie_type"])
        create_annotation = st.form_submit_button("Done", type="primary")

    if st.button("Cancel", key="cancel_create_annotation_entry"):
        _reset_pending_create_entry(default_type)
        st.rerun()

    if create_annotation:
        _append_network_entry_from_selection(
            annotation_state=st.session_state.annotation_state,
            selection=pending_entry,
            source=form_state["source"],
            target=form_state["target"],
            tie_type=form_state["tie_type"] or (default_type or ""),
        )
        _sync_current_annotation_state_for_document(str(document["id"]), st.session_state.annotation_state)
        _reset_pending_create_entry(default_type)
        st.session_state.annotation_editor_open = True
        st.rerun()


def _read_annotation_payload(
    document: dict,
    workspace_id: str,
    annotator_id: str,
    annotations_dir: Path,
    annotations_db_path: Path,
) -> dict:
    path = _project_annotation_path(annotations_dir, document, workspace_id, annotator_id)
    if path.exists():
        payload = load_annotation(path)
    else:
        stored = get_annotation(
            db_path=annotations_db_path,
            workspace_id=workspace_id,
            annotator_id=annotator_id,
            document_id=str(document["link"]),
        )
        if stored is not None:
            payload = stored["payload"]
        else:
            legacy_path = _legacy_annotation_path(annotations_dir, document)
            if legacy_path.exists():
                payload = load_annotation(legacy_path)
            else:
                payload = _empty_annotation_payload(document, workspace_id, annotator_id)

    if not payload.get("created_at"):
        payload["created_at"] = _iso_utcnow()
    if not payload.get("updated_at"):
        payload["updated_at"] = payload["created_at"]
    payload["workspace_id"] = workspace_id
    payload["annotator_id"] = annotator_id
    return payload


def _load_annotation_state(
    document: dict,
    default_type: str | None,
    workspace_id: str,
    annotator_id: str,
    annotations_dir: Path,
    annotations_db_path: Path,
    tie_attribute_schema: list[dict] | None = None,
) -> None:
    payload = _read_annotation_payload(
        document,
        workspace_id,
        annotator_id,
        annotations_dir,
        annotations_db_path,
    )
    normalized = _normalize_loaded_annotation(
        payload, default_type, tie_attribute_schema=tie_attribute_schema
    )
    st.session_state.annotation_state = normalized
    st.session_state.annotation_states[str(document["id"])] = normalized
    _set_annotation_state_baseline(str(document["id"]), normalized)
    st.session_state.document_objects[str(document["id"])] = document
    st.session_state.annotation_document_id = document["id"]
    st.session_state.annotation_scope_key = _annotation_scope_key(workspace_id, annotator_id)
    _reset_pending_span_target()


def _build_save_payload(
    document: dict,
    annotation_state: dict,
    content: str,
    annotator_id: str,
    workspace_id: str,
    created_at: str,
    updated_at: str,
    tie_attribute_schema: list[dict] | None = None,
) -> dict:
    payload = {
        "workspace_id": workspace_id,
        "annotator_id": annotator_id,
        "annotation_source": annotation_state.get("annotation_source", "human"),
        "document_id": document["link"],
        "created_at": created_at,
        "updated_at": updated_at,
        "network_data": [],
    }

    for entry in annotation_state["network_data"]:
        payload_entry = {
            "source": entry.get("source", "").strip(),
            "target": entry.get("target", "").strip(),
            "ties": [],
        }
        for tie in entry.get("ties", []):
            span_text = str(tie.get("reasoning_text_span", "") or "").strip()
            note_text = str(tie.get("reasoning_text_note", "") or "").strip()
            if not span_text and not note_text:
                legacy_text = str(tie.get("reasoning_text", "") or "").strip()
                note_text = legacy_text
            status_message = _sync_tie_span(tie, content)
            if span_text and status_message and status_message.startswith("Multiple matches"):
                matches = find_reasoning_spans(content, span_text)
                selected_match = tie.get("selected_match")
                if selected_match is None or selected_match >= len(matches):
                    raise IncompleteAnnotationError(
                        "Choose the correct reasoning span for each duplicated match."
                    )
            if not span_text and not note_text:
                raise IncompleteAnnotationError(
                    "Each tie must include reasoning text in either the span or note field."
                )
            try:
                attributes = validate_tie_attributes(
                    tie.get("attributes"),
                    tie_attribute_schema=tie_attribute_schema or [],
                )
            except ValueError as error:
                raise IncompleteAnnotationError(
                    f"{payload_entry['source']} → {payload_entry['target']}: {error}"
                ) from error

            tie_payload = {
                "type": tie.get("type", "").strip(),
                "reasoning_text": span_text or note_text,
                "reasoning_text_span": span_text,
                "reasoning_text_note": note_text,
                "reasoning_start": tie.get("reasoning_start"),
                "reasoning_end": tie.get("reasoning_end"),
            }
            if attributes:
                tie_payload["attributes"] = attributes
            payload_entry["ties"].append(tie_payload)
        payload["network_data"].append(payload_entry)
    return payload


def _llm_runs_as_export_records(annotations_db_path: Path) -> list[dict]:
    """Build export records from valid LLM runs (most recent run per document × model)."""
    seen: set[tuple[str, str]] = set()
    records: list[dict] = []
    for run in list_all_valid_llm_runs(annotations_db_path):
        doc_id = run["document_id"]
        if run["normalized_payload"] is None:
            continue
        payload = run["normalized_payload"]
        annotator_id = payload.get("annotator_id", f"{run['provider_id']}/{run['model_id']}")
        document_id = payload.get("document_id", doc_id)
        json_path = (
            Path(sanitize_filename(str(payload.get("workspace_id", "llm"))))
            / sanitize_filename(str(annotator_id))
            / f"{sanitize_filename(str(document_id))}.json"
        ).as_posix()
        key = (doc_id, annotator_id)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "workspace_id": payload.get("workspace_id", "llm"),
            "annotator_id": annotator_id,
            "document_id": document_id,
            "created_at": payload.get("created_at", run["created_at"]),
            "updated_at": payload.get("updated_at", run["created_at"]),
            "json_path": json_path,
            "payload": payload,
        })
    return records


_EXPORT_SCOPES = [
    "All annotations",
    "Human annotations",
    "LLM annotations",
    "Human-from-LLM annotations",
]


def _export_records_for_scope(annotations_db_path: Path, scope: str) -> list[dict]:
    if scope == "Human annotations":
        return list_annotations(annotations_db_path, annotation_source="human")
    if scope == "LLM annotations":
        return _llm_runs_as_export_records(annotations_db_path)
    if scope == "Human-from-LLM annotations":
        return list_annotations(annotations_db_path, annotation_source="human_from_llm")
    # All annotations
    human = list_annotations(annotations_db_path, annotation_source="human")
    hfl = list_annotations(annotations_db_path, annotation_source="human_from_llm")
    llm = _llm_runs_as_export_records(annotations_db_path)
    return human + hfl + llm


def _render_export_controls(
    annotations_db_path: Path,
    workspace_id: str,
    annotator_id: str,
) -> None:
    st.header("Exports")
    export_scope = st.selectbox(
        "Export scope",
        options=_EXPORT_SCOPES,
        key="export_scope",
    )
    records = _export_records_for_scope(annotations_db_path, export_scope)

    if not records:
        st.caption("No annotations available for the selected export scope.")
        return

    scope_stub = sanitize_filename(export_scope.lower().replace(" ", "_"))
    st.download_button(
        "Download edge list CSV",
        data=export_annotations_edge_list_csv(records),
        file_name=f"{scope_stub}_edge_list.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download JSON archive",
        data=export_annotations_json_archive(records),
        file_name=f"{scope_stub}_annotations.zip",
        mime="application/zip",
        use_container_width=True,
    )


def _stored_span_matches(tie: dict, content: str) -> bool:
    reasoning_text = (
        str(tie.get("reasoning_text_span", "") or "").strip()
        or str(tie.get("reasoning_text", "") or "").strip()
    )
    stored_start = tie.get("reasoning_start")
    stored_end = tie.get("reasoning_end")
    return (
        bool(reasoning_text)
        and isinstance(stored_start, int)
        and isinstance(stored_end, int)
        and stored_start >= 0
        and stored_end > stored_start
        and stored_end <= len(content)
        and content[stored_start:stored_end] == reasoning_text
    )


def _requires_span_disambiguation(tie: dict, content: str) -> bool:
    if _stored_span_matches(tie, content):
        return False
    reasoning_text = (
        str(tie.get("reasoning_text_span", "") or "").strip()
        or str(tie.get("reasoning_text", "") or "").strip()
    )
    if not reasoning_text:
        return False
    return len(find_reasoning_spans(content, reasoning_text)) > 1


def _sync_tie_span(tie: dict, content: str) -> str | None:
    reasoning_text = (
        str(tie.get("reasoning_text_span", "") or "").strip()
        or str(tie.get("reasoning_text", "") or "").strip()
    )
    if not reasoning_text:
        tie["reasoning_start"] = None
        tie["reasoning_end"] = None
        tie["selected_match"] = None
        tie["reasoning_text"] = str(tie.get("reasoning_text_note", "") or "").strip()
        return "Add exact span text to locate evidence in the document."

    tie["reasoning_text"] = reasoning_text

    if _stored_span_matches(tie, content):
        stored_start = tie["reasoning_start"]
        stored_end = tie["reasoning_end"]
        return f"Using stored span at characters {stored_start}-{stored_end}."

    matches = find_reasoning_spans(content, reasoning_text)
    if not matches:
        tie["reasoning_start"] = None
        tie["reasoning_end"] = None
        tie["selected_match"] = None
        return "No exact match found in the document for this reasoning text."

    if len(matches) == 1:
        start, end = matches[0]
        tie["reasoning_start"] = start
        tie["reasoning_end"] = end
        tie["selected_match"] = 0
        return f"Matched one span at characters {start}-{end}."

    if tie.get("selected_match") is None or tie["selected_match"] >= len(matches):
        tie["selected_match"] = 0

    selected_start, selected_end = matches[tie["selected_match"]]
    tie["reasoning_start"] = selected_start
    tie["reasoning_end"] = selected_end
    return f"Multiple matches found. Select the correct occurrence below."


def _tie_needs_span_review(tie: dict, content: str) -> bool:
    note_text = str(tie.get("reasoning_text_note", "") or "").strip()
    span_text = str(tie.get("reasoning_text_span", "") or "").strip()
    if span_text and _stored_span_matches(tie, content):
        return False
    return bool(note_text or span_text)


def _render_document_portability_controls(
    document: dict,
    annotations_db_path: Path,
    annotations_dir: Path,
    runtime_db_path: Path,
    valid_tie_labels: set[str],
    tie_attribute_schema: list[dict],
    workspace_id: str,
    annotator_id: str,
) -> None:
    if workspace_id and annotator_id:
        try:
            payload = export_document_annotation(
                annotations_db_path=annotations_db_path,
                workspace_id=workspace_id,
                annotator_id=annotator_id,
                document_id=str(document["link"]),
            )
        except ValueError:
            st.caption("No saved annotation exists for this document in the current scope.")
        else:
            st.download_button(
                "Export current document JSON",
                data=json.dumps(payload, indent=2),
                file_name=f"{sanitize_filename(str(document['link']))}_annotation.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.caption("Enter workspace and annotator IDs to export the current document annotation.")

    uploaded_import = st.file_uploader(
        "Import document annotation JSON",
        type=["json"],
        key=f"document_import_{document['id']}",
    )
    conflict_mode = st.selectbox(
        "Import conflict mode",
        options=["replace", "abort"],
        key=f"document_import_conflict_{document['id']}",
    )
    if uploaded_import is not None and st.button("Import document annotation", key=f"import_document_{document['id']}"):
        try:
            payload = json.loads(uploaded_import.getvalue().decode("utf-8"))
            import_document_annotation(
                runtime_db_path=runtime_db_path,
                annotations_dir=annotations_dir,
                annotations_db_path=annotations_db_path,
                payload=payload,
                valid_tie_labels=valid_tie_labels,
                conflict_mode=conflict_mode,
                tie_attribute_schema=tie_attribute_schema,
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            st.error(f"Unable to import annotation: {exc}")
        else:
            st.success("Imported annotation.")
            st.rerun()


def _render_llm_annotation_tools(
    *,
    document: dict,
    query_results: list[dict],
    runtime_context: dict,
    workspace_id: str,
    annotator_id: str,
    default_type: str | None,
) -> None:

    db_path = runtime_context["annotations_db_path"]
    project = runtime_context["project"]
    project_slug = project["slug"]
    project_name = project["name"]
    tie_schema = runtime_context["config"]["tie_schema"]
    tie_attribute_schema = runtime_context["config"].get("tie_attribute_schema", [])
    created_by = annotator_id or DEFAULT_LLM_CREATED_BY

    _ensure_project_default_llm_prompt(db_path, project_name)
    llm_settings = get_project_llm_settings(db_path)
    prompts = list_prompts(db_path)
    selected_prompt_state_key = f"llm_selected_prompt_id_{project_slug}"
    if selected_prompt_state_key not in st.session_state:
        st.session_state[selected_prompt_state_key] = llm_settings["default_prompt_id"]

    prompt_options, prompt_index = _llm_prompt_select_options(
        prompts,
        st.session_state.get(selected_prompt_state_key),
    )
    selected_prompt_label = st.selectbox(
        "Prompt version",
        options=list(prompt_options.keys()),
        index=prompt_index,
        key=f"llm_prompt_choice_{project_slug}",
    )
    selected_prompt_id = prompt_options[selected_prompt_label]
    st.session_state[selected_prompt_state_key] = selected_prompt_id
    selected_prompt = get_prompt(db_path, selected_prompt_id)
    if selected_prompt is None:
        st.error("Selected prompt could not be loaded.")
        return

    prompt_editor_key = f"llm_prompt_editor_{project_slug}"
    prompt_editor_loaded_key = f"llm_prompt_editor_loaded_{project_slug}"
    if st.session_state.get(prompt_editor_loaded_key) != selected_prompt_id:
        st.session_state[prompt_editor_key] = selected_prompt["template_text"]
        st.session_state[prompt_editor_loaded_key] = selected_prompt_id

    prompt_label = st.text_input(
        "Prompt label",
        value=selected_prompt["label"],
        key=f"llm_prompt_label_{project_slug}",
        help="Used when saving a forked or promoted prompt version.",
    ).strip() or selected_prompt["label"]
    prompt_template_text = st.text_area(
        "Annotation prompt template",
        key=prompt_editor_key,
        height=260,
        help="Use the supported placeholders shown below.",
    )

    provider_options = list_provider_ids()
    provider_default = llm_settings["provider_id"]
    if provider_default not in provider_options:
        provider_default = provider_options[0]
    provider_id = st.selectbox(
        "Provider",
        options=provider_options,
        index=provider_options.index(provider_default),
        key=f"llm_provider_{project_slug}",
    )
    provider_status_level, provider_status_message = _provider_status_message(provider_id)
    if provider_status_level == "warning":
        st.warning(provider_status_message)
    else:
        st.caption(provider_status_message)
    st.caption(
        f"Provider settings file: `{PROVIDER_SETTINGS_PATH}`. API keys must come from environment variables for both the app and worker."
    )

    if _provider_uses_freeform_model_id(provider_id):
        model_id = (
            st.text_input(
                "Model",
                value=llm_settings["model_id"],
                key=f"llm_model_{project_slug}",
                help=(
                    "Enter the exact model identifier accepted by the selected provider, for example "
                    "gemini-2.5-flash or anthropic/claude-3.5-sonnet."
                ),
            ).strip()
            or llm_settings["model_id"]
        )
    else:
        model_options = list_available_models_for_provider(
            provider_id,
            settings_by_provider=load_provider_settings(PROVIDER_SETTINGS_PATH),
        )
        if llm_settings["model_id"] not in model_options:
            model_options = model_options + [llm_settings["model_id"]]
        model_id = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(llm_settings["model_id"]),
            key=f"llm_model_{project_slug}",
        )
    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(llm_settings["temperature"]),
        step=0.1,
        key=f"llm_temperature_{project_slug}",
    )
    max_output_tokens = int(
        st.number_input(
            "Max output tokens",
            min_value=1,
            value=int(llm_settings["max_output_tokens"]),
            step=50,
            key=f"llm_max_output_tokens_{project_slug}",
        )
    )

    with st.expander("Supported prompt placeholders"):
        st.code("\n".join(SUPPORTED_PROMPT_PLACEHOLDERS), language="text")

    try:
        rendered_preview = _rendered_llm_prompt_preview(
            project_name=project_name,
            document=document,
            tie_schema=tie_schema,
            template_text=prompt_template_text,
            tie_attribute_schema=tie_attribute_schema,
        )
    except ValueError as exc:
        st.error(f"Prompt preview error: {exc}")
        rendered_preview = None
    else:
        with st.expander("Rendered prompt preview for current document"):
            st.code(rendered_preview, language="text")

    save_default_col, save_fork_col = st.columns(2)
    with save_default_col:
        if st.button("Save as project default", key=f"save_llm_default_{project_slug}"):
            try:
                saved_prompt = _persist_llm_prompt_template(
                    db_path,
                    selected_prompt_id=selected_prompt_id,
                    template_text=prompt_template_text,
                    created_by=created_by,
                    label=prompt_label,
                    promote_to_project_default=True,
                )
                set_project_llm_settings(
                    db_path,
                    provider_id=provider_id,
                    model_id=model_id,
                    temperature=float(temperature),
                    max_output_tokens=max_output_tokens,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state[selected_prompt_state_key] = saved_prompt["prompt_id"]
                st.session_state[prompt_editor_loaded_key] = None
                st.success("Saved LLM project defaults.")
                st.rerun()

    with save_fork_col:
        if st.button("Save as document prompt fork", key=f"save_llm_fork_{project_slug}"):
            try:
                saved_prompt = _persist_llm_prompt_template(
                    db_path,
                    selected_prompt_id=selected_prompt_id,
                    template_text=prompt_template_text,
                    created_by=created_by,
                    label=prompt_label,
                    promote_to_project_default=False,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state[selected_prompt_state_key] = saved_prompt["prompt_id"]
                st.session_state[prompt_editor_loaded_key] = None
                st.success(f"Saved prompt fork {saved_prompt['prompt_id']}.")
                st.rerun()

    current_doc_clicked = False
    query_set_clicked = False
    if rendered_preview is None:
        st.caption("Fix the prompt template before launching an LLM annotation job.")
    else:
        current_doc_col, query_set_col = st.columns(2)
        current_doc_clicked = current_doc_col.button(
            "Annotate current document",
            key=f"launch_llm_job_single_{project_slug}_{document['id']}",
            use_container_width=True,
        )
        query_set_clicked = query_set_col.button(
            f"Annotate current query result set ({len(query_results)} docs)",
            key=f"launch_llm_job_{project_slug}",
            use_container_width=True,
            disabled=not query_results,
        )
        if not query_results:
            st.caption("Run a query to enable LLM annotation over the current result set.")

    if current_doc_clicked or query_set_clicked:
        try:
            document_ids = (
                _single_document_job_ids(document)
                if current_doc_clicked
                else _query_result_document_ids(query_results)
            )
            run_prompt, summary = _launch_llm_annotation_job(
                db_path=db_path,
                runtime_db_path=runtime_context["runtime_db_path"],
                project_name=project_name,
                tie_schema=tie_schema,
                document_ids=document_ids,
                selected_prompt_id=selected_prompt_id,
                template_text=prompt_template_text,
                created_by=created_by,
                prompt_label=prompt_label,
                provider_id=provider_id,
                model_id=model_id,
                temperature=float(temperature),
                max_output_tokens=max_output_tokens,
            )
        except (ValueError, RuntimeError) as exc:
            st.error(f"Unable to run LLM annotation job: {exc}")
        else:
            st.session_state[selected_prompt_state_key] = run_prompt["prompt_id"]
            st.success(
                "LLM job queued. "
                f"{summary['pending']} documents are waiting for a worker."
            )
            st.rerun()

    job_rows = _llm_job_summary_rows(db_path)
    worker_status = _llm_worker_status(job_rows)
    if worker_status is not None:
        level, message = worker_status
        if level == "warning":
            st.warning(message)
        else:
            st.info(message)
    if job_rows:
        with st.expander("Recent LLM jobs"):
            if st.button(
                "Refresh",
                key=f"refresh_llm_jobs_{project_slug}",
                use_container_width=True,
            ):
                st.rerun()
            queued_count = sum(1 for row in job_rows if row["status"] == "queued")
            running_count = sum(1 for row in job_rows if row["status"] == "running")
            cancelled_count = sum(1 for row in job_rows if row["status"] == "cancelled")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Queued jobs", queued_count)
            metric_col2.metric("Running jobs", running_count)
            metric_col3.metric("Cancelled jobs", cancelled_count)
            st.dataframe(job_rows, use_container_width=True)

            cancellable_rows = [row for row in job_rows if row["status"] in {"queued", "running"}]
            if cancellable_rows:
                cancellable_options = {
                    (
                        f"{row['job_id']} | {row['status']} | {row['provider_id']}/{row['model_id']} | "
                        f"worker={row['worker_id'] or 'unclaimed'}"
                    ): row["job_id"]
                    for row in cancellable_rows
                }
                selected_cancel_label = st.selectbox(
                    "Cancel queued/running job",
                    options=list(cancellable_options.keys()),
                    key=f"cancel_llm_job_select_{project_slug}",
                )
                if st.button(
                    "Cancel selected job",
                    key=f"cancel_llm_job_button_{project_slug}",
                    use_container_width=True,
                ):
                    cancel_job(db_path, cancellable_options[selected_cancel_label])
                    st.success("Job cancellation recorded.")
                    st.rerun()


def _create_project_from_uploaded_config(
    projects_root: Path,
    registry_path: Path,
    filename: str,
    raw_content: str | bytes,
) -> dict:
    config = load_project_config_content(filename=filename, raw_content=raw_content)
    return create_and_register_project(projects_root, registry_path, config)


def _project_runtime_context(project: dict) -> dict:
    project_path = Path(project["project_path"])
    config = load_project_config_content(
        filename="project.yaml",
        raw_content=(project_path / "project.yaml").read_text(encoding="utf-8"),
    )
    return {
        "project": project,
        "project_path": project_path,
        "config": config,
        "runtime_db_path": project_path / "derived" / "runtime.sqlite",
        "annotations_dir": project_path / "annotations",
        "annotations_db_path": project_path / "annotations" / "annotations.sqlite",
        "filter_schema": config["filters"],
    }


def _normalized_project_home_mode(mode: str | None) -> str:
    if mode in {"open", "create"}:
        return mode
    return "open"


def _guided_wizard_state() -> dict:
    defaults = {
        "step": 1,
        "project_name": "",
        "project_slug": "",
        "source_origin": "server",
        "source_type": "csv",
        "server_source_path": "",
        "source_mode": "copy",
        "uploaded_source_name": "",
        "uploaded_source_path": "",
        "sqlite_table": "",
        "id_field": "",
        "text_field": "",
        "title_field": "",
        "metadata_fields": [],
        "filter_types_by_field": {},
        "tie_schema_preset": "Blank",
        "tie_schema_text": "",
        "tie_attributes": [],
        "next_tie_attribute_id": 0,
    }
    state = st.session_state.setdefault("guided_setup", {})
    for key, value in defaults.items():
        state.setdefault(key, value)
    return state


def _reset_guided_wizard() -> None:
    st.session_state.guided_setup = {
        "step": 1,
        "project_name": "",
        "project_slug": "",
        "source_origin": "server",
        "source_type": "csv",
        "server_source_path": "",
        "source_mode": "copy",
        "uploaded_source_name": "",
        "uploaded_source_path": "",
        "sqlite_table": "",
        "id_field": "",
        "text_field": "",
        "title_field": "",
        "metadata_fields": [],
        "filter_types_by_field": {},
        "tie_schema_preset": "Blank",
        "tie_schema_text": "",
        "tie_attributes": [],
        "next_tie_attribute_id": 0,
    }


def _server_source_candidates(source_type: str) -> list[Path]:
    suffixes = {".csv"} if source_type == "csv" else {".sqlite", ".db"}
    return [
        path
        for path in discover_source_files(
            [DATA_DIR],
            ignored_roots=GUIDED_SERVER_SOURCE_IGNORED_ROOTS,
            ignored_files=GUIDED_SERVER_SOURCE_IGNORED_FILES,
        )
        if path.suffix.lower() in suffixes
    ]


def _server_source_options(source_type: str) -> dict[str, Path]:
    return {
        path.relative_to(DATA_DIR).as_posix(): path
        for path in _server_source_candidates(source_type)
    }


def _store_guided_upload(project_slug: str, uploaded_file) -> Path:
    ensure_directory(GUIDED_UPLOADS_DIR)
    slug_part = sanitize_filename(project_slug or "project")
    filename = sanitize_filename(uploaded_file.name)
    stored_path = GUIDED_UPLOADS_DIR / f"{slug_part}_{filename}"
    stored_path.write_bytes(uploaded_file.getvalue())
    return stored_path


def _cleanup_guided_upload(state: dict) -> None:
    raw_path = state.get("uploaded_source_path", "")
    if not raw_path:
        return
    path = Path(raw_path)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        logging.warning("Unable to clean up guided upload: %s", path)


def _guided_source_path(state: dict) -> Path | None:
    if state["source_origin"] == "upload":
        raw_path = state.get("uploaded_source_path", "")
    else:
        raw_path = state.get("server_source_path", "")
    if not raw_path:
        return None
    return Path(raw_path)


def _guided_source_columns(state: dict) -> list[str]:
    source_path = _guided_source_path(state)
    if source_path is None:
        return []
    if state["source_type"] == "csv":
        return inspect_csv_columns(source_path)
    if not state.get("sqlite_table"):
        return []
    return inspect_sqlite_columns(source_path, state["sqlite_table"])


def _available_metadata_columns(
    columns: list[str],
    id_field: str,
    text_field: str,
    title_field: str | None,
) -> list[str]:
    reserved = {id_field, text_field}
    if title_field:
        reserved.add(title_field)
    return [column for column in columns if column not in reserved]


def _tie_schema_presets() -> dict[str, list[dict[str, str]]]:
    presets = {"Blank": []}
    presets.update(_tie_schema_presets_from_project_configs())
    if TIE_OPTIONS_PATH.exists():
        labels = load_tie_options(TIE_OPTIONS_PATH)
        descriptions_lookup: dict[str, str] = {}
        descriptions_path = DATA_DIR / "tie_descriptions" / "tie_descriptions.json"
        if descriptions_path.exists():
            content = json.loads(descriptions_path.read_text(encoding="utf-8"))
            for item in content:
                label = item.get("network_type")
                description = item.get("function", {}).get("description", "")
                if isinstance(label, str) and label:
                    descriptions_lookup[label] = description if isinstance(description, str) else ""
        presets["Tie Options File"] = [
            {"label": label, "description": descriptions_lookup.get(label, label)}
            for label in labels
        ]
    return presets


def _tie_schema_presets_from_project_configs() -> dict[str, list[dict[str, str]]]:
    presets: dict[str, list[dict[str, str]]] = {}
    for config_path in sorted(DATA_DIR.glob("*.y*ml")) + sorted(DATA_DIR.glob("*.json")):
        if config_path.name == REGISTRY_PATH.name:
            continue
        try:
            config = load_project_config_content(
                filename=config_path.name,
                raw_content=config_path.read_text(encoding="utf-8"),
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        tie_schema = config.get("tie_schema", [])
        if not tie_schema:
            continue
        preset_name = config["project"]["name"]
        presets[preset_name] = [
            {
                "label": item["label"],
                "description": item["description"],
            }
            for item in tie_schema
        ]
    return presets


def _tie_schema_text(tie_schema: list[dict[str, str]]) -> str:
    return "\n".join(f'{item["label"]}|{item["description"]}' for item in tie_schema)


def _parse_tie_schema_text(raw_text: str) -> list[dict[str, str]]:
    tie_schema: list[dict[str, str]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        label, separator, description = line.partition("|")
        if not separator or not label.strip() or not description.strip():
            raise ValueError(f"Invalid tie schema line: {raw_line}")
        tie_schema.append(
            {
                "label": label.strip(),
                "description": description.strip(),
            }
        )
    if not tie_schema:
        raise ValueError("At least one tie schema entry is required")
    return tie_schema


def _add_guided_tie_attribute(state: dict) -> None:
    """Append a blank attribute card with a never-reused widget identity."""
    attribute_id = int(state.get("next_tie_attribute_id", 0))
    state.setdefault("tie_attributes", []).append(
        {
            "id": attribute_id,
            "name": "",
            "description": "",
            "type": "categorical",
            "options_text": "",
            "minimum": 0.0,
            "maximum": 1.0,
        }
    )
    state["next_tie_attribute_id"] = attribute_id + 1


def _remove_guided_tie_attribute(state: dict, attribute_id: int) -> None:
    """Remove one attribute card without recycling its widget identity."""
    state["tie_attributes"] = [
        card
        for card in state.get("tie_attributes", [])
        if card.get("id") != attribute_id
    ]


def _guided_tie_attribute_schema(cards: list[dict]) -> list[dict]:
    """Convert wizard card state into canonical tie attribute entries."""
    schema: list[dict] = []
    for card in cards:
        attribute_type = str(card.get("type", "")).strip()
        attribute = {
            "name": str(card.get("name", "")).strip(),
            "type": attribute_type,
            "description": str(card.get("description", "")).strip(),
        }
        if attribute_type == "categorical":
            options_text = str(card.get("options_text", ""))
            attribute["options"] = [
                option.strip() for option in options_text.splitlines() if option.strip()
            ]
        elif attribute_type == "numeric":
            attribute["range"] = [card.get("minimum"), card.get("maximum")]
        schema.append(attribute)
    return schema


def _guided_config_for_review(state: dict) -> dict:
    config = build_guided_project_config(
        project_name=state["project_name"],
        project_slug=state["project_slug"],
        source_type=state["source_type"],
        source_mode="copy" if state["source_origin"] == "upload" else state["source_mode"],
        source_path=str(_guided_source_path(state)),
        table_or_view=state.get("sqlite_table") or None,
        id_field=state["id_field"],
        text_field=state["text_field"],
        title_field=state.get("title_field") or None,
        metadata_fields=list(state.get("metadata_fields", [])),
        filter_types_by_field=dict(state.get("filter_types_by_field", {})),
        tie_schema=_parse_tie_schema_text(state.get("tie_schema_text", "")),
        tie_attribute_schema=_guided_tie_attribute_schema(
            list(state.get("tie_attributes", []))
        ),
    )
    return validate_project_config(config)


def _guided_review_summary(config: dict) -> dict[str, object]:
    return {
        "project_name": config["project"]["name"],
        "project_slug": config["project"]["slug"],
        "source_type": config["source"]["type"],
        "source_mode": config["source"]["mode"],
        "source_path": config["source"]["path"],
        "document_fields": dict(config["document"]),
        "metadata_fields": [field["name"] for field in config["metadata_fields"]],
        "filter_fields": [filter_config["field"] for filter_config in config["filters"]],
        "tie_schema_count": len(config["tie_schema"]),
        "tie_attribute_schema_count": len(config.get("tie_attribute_schema", [])),
    }


def _render_open_existing_projects(projects: list[dict]) -> None:
    st.subheader("Open Existing Project")
    st.caption("Open a registered project or create one from an existing config file.")

    uploaded_config = st.file_uploader(
        "Load config file",
        type=["yaml", "yml", "json"],
        key="project_config_upload",
    )
    if uploaded_config is not None:
        if st.button("Create project from config", key="create_project_from_upload", type="primary"):
            try:
                entry = _create_project_from_uploaded_config(
                    projects_root=PROJECTS_DIR,
                    registry_path=REGISTRY_PATH,
                    filename=uploaded_config.name,
                    raw_content=uploaded_config.getvalue(),
                )
            except (OSError, ValueError) as exc:
                st.error(f"Unable to create project: {exc}")
            else:
                st.session_state.active_project_slug = entry["slug"]
                st.success(f"Created project {entry['name']}.")
                st.rerun()

    if not projects:
        st.info("No projects registered yet.")
        return

    for project in projects:
        status = get_project_status(Path(project["project_path"]))
        with st.container(border=True):
            st.markdown(f"**{project['name']}**")
            st.caption(f"Slug: {project['slug']} | Status: {status}")
            st.caption(project["project_path"])
            if st.button("Open project", key=f"open_project_{project['slug']}"):
                st.session_state.active_project_slug = project["slug"]
                st.rerun()


def _annotation_scope_catalog(annotations_db_path: Path) -> dict[str, object]:
    scopes = [
        (record["workspace_id"], record["annotator_id"])
        for record in list_annotations(annotations_db_path)
        if record["workspace_id"] != RESERVED_LLM_WORKSPACE_ID
        and record["payload"].get("annotation_source") != "llm"
    ]
    workspaces = sorted({workspace_id for workspace_id, _annotator_id in scopes})
    annotators = sorted({annotator_id for _workspace_id, annotator_id in scopes})
    annotators_by_workspace: dict[str, list[str]] = {}
    for workspace_id, annotator_id in scopes:
        annotators_by_workspace.setdefault(workspace_id, []).append(annotator_id)
    for workspace_id, values in annotators_by_workspace.items():
        annotators_by_workspace[workspace_id] = sorted(set(values))
    return {
        "workspaces": workspaces,
        "annotators": annotators,
        "annotators_by_workspace": annotators_by_workspace,
    }


def _scope_select_options(
    existing_values: list[str],
    current_value: str,
    *,
    default_value: str | None = None,
) -> tuple[list[str], int]:
    options: list[str] = []
    seen: set[str] = set()

    for candidate in existing_values:
        if candidate and candidate not in seen:
            seen.add(candidate)
            options.append(candidate)

    if current_value and current_value not in seen:
        options.append(current_value)
        seen.add(current_value)
    elif not current_value and default_value and default_value not in seen:
        options.append(default_value)
        seen.add(default_value)

    options.append("Enter new...")
    selected_value = current_value or default_value or "Enter new..."
    if selected_value not in options:
        options.append(selected_value)
    index = options.index(selected_value) if selected_value in options else len(options) - 1
    return options, index


def _selected_scope_value(selection: str, typed_value: str, fallback: str = "") -> str:
    if selection == "Enter new...":
        return typed_value.strip() or fallback
    return selection


def _human_reporting_scope_options(
    scopes: list[tuple[str, str]],
    *,
    selected_scope: tuple[str, str] | None,
) -> tuple[dict[str, tuple[str, str]], int]:
    options = {
        f"{workspace_id} / {annotator_id}": (workspace_id, annotator_id)
        for workspace_id, annotator_id in scopes
    }
    values = list(options.values())
    if selected_scope in values:
        return options, values.index(selected_scope)
    return options, 0


def _llm_reporting_scope_options(
    scopes: list[dict],
    *,
    selected_scope: tuple[str, str] | None,
) -> tuple[dict[str, tuple[str, str]], int]:
    options = {
        f"{scope['provider_id']} / {scope['model_id']} ({scope['document_count']} docs)": (
            scope["provider_id"],
            scope["model_id"],
        )
        for scope in scopes
    }
    values = list(options.values())
    if selected_scope in values:
        return options, values.index(selected_scope)
    return options, 0


def _llm_reporting_direction_options(
    *,
    selected_mode: str | None,
) -> tuple[dict[str, str], int]:
    options = {
        "Directed": "directed",
        "Undirected": "undirected",
    }
    values = list(options.values())
    if selected_mode in values:
        return options, values.index(selected_mode)
    return options, 0


def _llm_reporting_human_document_coverage_options(
    *,
    selected_mode: str | None,
) -> tuple[dict[str, str], int]:
    options = {
        "Include all human annotated docs": "include_all_human_annotated_docs",
        "Exclude human docs without valid LLM runs": "exclude_human_docs_without_valid_llm_runs",
    }
    values = list(options.values())
    if selected_mode in values:
        return options, values.index(selected_mode)
    return options, 0


def _llm_reporting_empty_state_message(
    human_scopes: list[tuple[str, str]],
    llm_scopes: list[object],
) -> str | None:
    if not human_scopes and not llm_scopes:
        return "No human annotation scopes or valid LLM reporting scopes are available yet."
    if not human_scopes:
        return "No human annotation scopes are available yet."
    if not llm_scopes:
        return "No valid LLM reporting scopes are available yet."
    return None


def _llm_reporting_summary_stats(report: dict) -> dict[str, float | int]:
    overall = report["overall"]
    coverage = report["document_coverage"]
    return {
        "Precision": overall["precision"],
        "Recall": overall["recall"],
        "F1": overall["f1"],
        "Macro F1": overall["macro_f1"],
        "Human docs": coverage["human_documents"],
        "LLM docs": coverage["llm_documents"],
        "Union docs": coverage["union_documents"],
    }


def _llm_reporting_refresh_button(project_slug: str) -> None:
    if st.button(
        "Refresh and recalculate",
        key=f"refresh_llm_reporting_{project_slug}",
        use_container_width=True,
    ):
        st.rerun()


def _render_llm_reporting_panel(runtime_context: dict) -> None:
    annotations_db_path = runtime_context["annotations_db_path"]
    catalog = _annotation_scope_catalog(annotations_db_path)
    human_scopes = [
        (workspace_id, annotator_id)
        for workspace_id in catalog["workspaces"]
        for annotator_id in catalog["annotators_by_workspace"].get(workspace_id, [])
    ]
    llm_scopes = list_llm_reporting_scopes(annotations_db_path)
    empty_state = _llm_reporting_empty_state_message(human_scopes, llm_scopes)
    if empty_state is not None:
        st.caption(empty_state)
        return

    _llm_reporting_refresh_button(runtime_context["project"]["slug"])

    selected_human_scope = st.session_state.get("llm_reporting_human_scope")
    human_options, human_index = _human_reporting_scope_options(
        human_scopes,
        selected_scope=selected_human_scope,
    )
    selected_human_label = st.selectbox(
        "Human scope",
        options=list(human_options.keys()),
        index=human_index,
        key="llm_reporting_human_scope_label",
    )
    human_scope = human_options[selected_human_label]
    st.session_state.llm_reporting_human_scope = human_scope

    selected_llm_scope = st.session_state.get("llm_reporting_llm_scope")
    llm_options, llm_index = _llm_reporting_scope_options(
        llm_scopes,
        selected_scope=selected_llm_scope,
    )
    selected_llm_label = st.selectbox(
        "LLM scope",
        options=list(llm_options.keys()),
        index=llm_index,
        key="llm_reporting_llm_scope_label",
    )
    llm_scope = llm_options[selected_llm_label]
    st.session_state.llm_reporting_llm_scope = llm_scope

    selected_direction_mode = st.session_state.get("llm_reporting_relationship_direction")
    direction_options, direction_index = _llm_reporting_direction_options(
        selected_mode=selected_direction_mode,
    )
    selected_direction_label = st.selectbox(
        "Relationship direction",
        options=list(direction_options.keys()),
        index=direction_index,
        key="llm_reporting_relationship_direction_label",
    )
    relationship_direction = direction_options[selected_direction_label]
    st.session_state.llm_reporting_relationship_direction = relationship_direction

    selected_human_document_coverage = st.session_state.get("llm_reporting_human_document_coverage")
    human_document_coverage_options, human_document_coverage_index = (
        _llm_reporting_human_document_coverage_options(
            selected_mode=selected_human_document_coverage,
        )
    )
    selected_human_document_coverage_label = st.selectbox(
        "Human document coverage",
        options=list(human_document_coverage_options.keys()),
        index=human_document_coverage_index,
        key="llm_reporting_human_document_coverage_label",
    )
    human_document_coverage = human_document_coverage_options[
        selected_human_document_coverage_label
    ]
    st.session_state.llm_reporting_human_document_coverage = human_document_coverage

    st.caption("Uses the latest valid run per document for the selected provider/model.")

    human_records = list_annotations(
        annotations_db_path,
        annotator_id=human_scope[1],
        workspace_id=human_scope[0],
    )
    llm_runs = list(
        list_latest_valid_runs_for_scope(
            annotations_db_path,
            provider_id=llm_scope[0],
            model_id=llm_scope[1],
        ).values()
    )
    tie_labels = [tie["label"] for tie in runtime_context["config"]["tie_schema"]]
    report = build_pairwise_llm_report(
        human_records=human_records,
        llm_runs=llm_runs,
        tie_labels=tie_labels,
        relationship_direction=relationship_direction,
        human_document_coverage=human_document_coverage,
        tie_attribute_schema=runtime_context["config"].get("tie_attribute_schema", []),
    )

    summary_stats = _llm_reporting_summary_stats(report)
    metric_columns = st.columns(4)
    for index, metric_name in enumerate(["Precision", "Recall", "F1", "Macro F1"]):
        metric_columns[index].metric(metric_name, summary_stats[metric_name])

    coverage_columns = st.columns(3)
    for index, metric_name in enumerate(["Human docs", "LLM docs", "Union docs"]):
        coverage_columns[index].metric(metric_name, summary_stats[metric_name])

    per_label_rows = []
    for label in report["labels"]:
        metrics = report["per_label"][label]
        per_label_rows.append(
            {
                "label": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "support": metrics["support"],
            }
        )
    st.caption("Per-label metrics")
    st.dataframe(per_label_rows, use_container_width=True)

    matrix_rows = []
    for truth_label in report["confusion_matrix"]["labels"]:
        row = {"truth": truth_label}
        row.update(report["confusion_matrix"]["rows"][truth_label])
        matrix_rows.append(row)
    st.caption("Confusion matrix")
    st.dataframe(matrix_rows, use_container_width=True)

    _render_attribute_agreement_panel(report)


def _render_attribute_agreement_panel(report: dict) -> None:
    attribute_agreement = report.get("attribute_agreement")
    if not attribute_agreement:
        return
    with st.expander(
        f"Attribute agreement (over {attribute_agreement['matched_ties']} matched ties)"
    ):
        for name, metrics in attribute_agreement["attributes"].items():
            _render_attribute_agreement_metrics(name, metrics)


def _render_attribute_agreement_metrics(name: str, metrics: dict) -> None:
    st.markdown(f"**{name}**")
    coverage = metrics["coverage"]
    coverage_labels = {
        "both": "Both",
        "human_only": "Human only",
        "llm_only": "LLM only",
        "neither": "Neither",
    }
    coverage_columns = st.columns(4)
    for index, (coverage_key, coverage_label) in enumerate(coverage_labels.items()):
        coverage_columns[index].metric(coverage_label, coverage[coverage_key])
    if "accuracy" in metrics:
        st.metric("Accuracy", _format_agreement_metric(metrics["accuracy"]))
        matrix_rows = [
            {"truth": truth, **predicted}
            for truth, predicted in metrics["confusion_matrix"].items()
        ]
        st.caption("Confusion matrix (rows = human)")
        st.dataframe(matrix_rows, use_container_width=True)
    elif "mae" in metrics:
        numeric_columns = st.columns(2)
        numeric_columns[0].metric("MAE", _format_agreement_metric(metrics["mae"]))
        numeric_columns[1].metric(
            f"Within ±{metrics['tolerance']}",
            _format_agreement_metric(metrics["within_tolerance"]),
            help="Share of matched ties within tolerance",
        )
    else:
        st.metric(
            "Match at common granularity",
            _format_agreement_metric(metrics["match_at_common_granularity"]),
        )


def _format_agreement_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value}"


def _render_guided_project_setup(projects_root: Path, registry_path: Path) -> None:
    state = _guided_wizard_state()
    presets = _tie_schema_presets()

    st.subheader("Create New Project")
    st.caption("Use the guided wizard to inspect a CSV or SQLite source and generate the project config.")
    st.caption(f"Step {state['step']} of 7")

    if state["step"] == 1:
        state["project_name"] = st.text_input("Project name", value=state["project_name"])
        state["project_slug"] = st.text_input(
            "Project slug",
            value=state["project_slug"],
            help="Used as the project folder name. Lowercase letters, numbers, hyphens, and underscores only.",
        )
        previous_slug = state["project_slug"]
        if state["project_name"] and not previous_slug:
            generated_slug = sanitize_filename(state["project_name"].lower()).replace(".", "-")
            generated_slug = generated_slug.replace("-", "-")
            state["project_slug"] = generated_slug.replace("__", "_")

        if st.button("Next", key="guided_next_1", type="primary"):
            if not state["project_name"].strip():
                st.error("Project name is required.")
                return
            try:
                build_guided_project_config(
                    project_name=state["project_name"],
                    project_slug=state["project_slug"],
                    source_type="csv",
                    source_mode="copy",
                    source_path="/tmp/placeholder.csv",
                    id_field="id",
                    text_field="text",
                    title_field=None,
                    metadata_fields=[],
                    filter_types_by_field={},
                    tie_schema=[{"label": "placeholder", "description": "placeholder"}],
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            state["step"] = 2
            st.rerun()

    elif state["step"] == 2:
        state["source_origin"] = st.radio(
            "Source location",
            options=["server", "upload"],
            format_func=lambda value: "Server file" if value == "server" else "Upload file",
            index=0 if state["source_origin"] == "server" else 1,
        )
        state["source_type"] = st.selectbox(
            "Source type",
            options=["csv", "sqlite"],
            index=0 if state["source_type"] == "csv" else 1,
        )

        if state["source_origin"] == "server":
            options = _server_source_options(state["source_type"])
            option_labels = list(options)
            if option_labels:
                selected_index = 0
                if state["server_source_path"]:
                    current_path = Path(state["server_source_path"])
                    matching_labels = [
                        label for label, path in options.items() if path == current_path
                    ]
                    if matching_labels:
                        selected_index = option_labels.index(matching_labels[0])
                selected_label = st.selectbox(
                    "Server-side source file",
                    options=option_labels,
                    index=selected_index,
                )
                if selected_label:
                    state["server_source_path"] = str(options[selected_label])
            else:
                st.info("No matching server-side source files were found.")
                state["server_source_path"] = ""
            state["source_mode"] = st.selectbox(
                "How to use this file",
                options=["copy", "reference"],
                index=0 if state["source_mode"] == "copy" else 1,
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload data source",
                type=["csv"] if state["source_type"] == "csv" else ["sqlite", "db"],
                key=f"guided_upload_{state['source_type']}",
            )
            if uploaded_file is not None:
                stored_path = _store_guided_upload(state["project_slug"], uploaded_file)
                state["uploaded_source_name"] = uploaded_file.name
                state["uploaded_source_path"] = str(stored_path)
                st.caption(f"Imported file: {uploaded_file.name}")
            st.caption("Uploaded sources are always copied into the project.")

        source_path = _guided_source_path(state)
        if state["source_type"] == "sqlite" and source_path is not None and source_path.exists():
            tables = inspect_sqlite_tables(source_path)
            if tables:
                current_index = tables.index(state["sqlite_table"]) if state["sqlite_table"] in tables else 0
                state["sqlite_table"] = st.selectbox(
                    "SQLite table or view",
                    options=tables,
                    index=current_index,
                )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="guided_back_2"):
                state["step"] = 1
                st.rerun()
        with col2:
            if st.button("Next", key="guided_next_2", type="primary"):
                if source_path is None or not source_path.exists():
                    st.error("Select or upload a valid source file before continuing.")
                    return
                if state["source_type"] == "sqlite" and not state["sqlite_table"]:
                    st.error("Choose a SQLite table or view before continuing.")
                    return
                state["step"] = 3
                st.rerun()

    elif state["step"] == 3:
        columns = _guided_source_columns(state)
        if not columns:
            st.error("The selected source could not be inspected.")
            return

        state["id_field"] = st.selectbox(
            "Document ID field",
            options=columns,
            index=columns.index(state["id_field"]) if state["id_field"] in columns else 0,
        )
        state["text_field"] = st.selectbox(
            "Document text field",
            options=columns,
            index=columns.index(state["text_field"]) if state["text_field"] in columns else min(1, len(columns) - 1),
        )
        title_options = [""] + columns
        title_index = title_options.index(state["title_field"]) if state["title_field"] in title_options else 0
        state["title_field"] = st.selectbox(
            "Document title field (optional)",
            options=title_options,
            index=title_index,
            format_func=lambda value: "None" if value == "" else value,
        )
        st.caption(f"Available columns: {', '.join(columns)}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="guided_back_3"):
                state["step"] = 2
                st.rerun()
        with col2:
            if st.button("Next", key="guided_next_3", type="primary"):
                if state["id_field"] == state["text_field"]:
                    st.error("Document ID field and document text field must be different.")
                    return
                state["step"] = 4
                st.rerun()

    elif state["step"] == 4:
        columns = _guided_source_columns(state)
        metadata_options = _available_metadata_columns(
            columns,
            state["id_field"],
            state["text_field"],
            state.get("title_field") or None,
        )
        state["metadata_fields"] = st.multiselect(
            "Metadata fields",
            options=metadata_options,
            default=[field for field in state["metadata_fields"] if field in metadata_options],
        )
        filter_types_by_field: dict[str, str] = {}
        for field in state["metadata_fields"]:
            current = state["filter_types_by_field"].get(field, "select")
            filter_types_by_field[field] = st.selectbox(
                f"Filter type for {field}",
                options=sorted(VALID_FILTER_TYPES),
                index=sorted(VALID_FILTER_TYPES).index(current)
                if current in VALID_FILTER_TYPES
                else 0,
                key=f"filter_type_{field}",
            )
        state["filter_types_by_field"] = filter_types_by_field

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="guided_back_4"):
                state["step"] = 3
                st.rerun()
        with col2:
            if st.button("Next", key="guided_next_4", type="primary"):
                state["step"] = 5
                st.rerun()

    elif state["step"] == 5:
        preset_names = list(presets.keys())
        preset_index = preset_names.index(state["tie_schema_preset"]) if state["tie_schema_preset"] in preset_names else 0
        state["tie_schema_preset"] = st.selectbox(
            "Tie schema preset",
            options=preset_names,
            index=preset_index,
        )
        if st.button("Load preset into editor", key="guided_tie_schema_load"):
            state["tie_schema_text"] = _tie_schema_text(presets[state["tie_schema_preset"]])
            st.rerun()
        state["tie_schema_text"] = st.text_area(
            "Tie schema editor (one per line: label|description)",
            value=state["tie_schema_text"],
            height=220,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="guided_back_5"):
                state["step"] = 4
                st.rerun()
        with col2:
            if st.button("Next", key="guided_next_5", type="primary"):
                try:
                    _parse_tie_schema_text(state["tie_schema_text"])
                except ValueError as exc:
                    st.error(str(exc))
                    return
                state["step"] = 6
                st.rerun()

    elif state["step"] == 6:
        st.markdown("**Tie attributes (optional)**")
        st.caption(
            "Define reusable fields that annotators can set independently on each tie. "
            "Leave this step empty if the project does not need tie attributes."
        )

        for card in state.get("tie_attributes", []):
            attribute_id = card["id"]
            with st.container(border=True):
                heading_col, remove_col = st.columns([4, 1])
                with heading_col:
                    st.markdown(f"**Attribute {attribute_id + 1}**")
                with remove_col:
                    if st.button(
                        "Remove",
                        key=f"guided_tie_attribute_remove_{attribute_id}",
                    ):
                        _remove_guided_tie_attribute(state, attribute_id)
                        st.rerun()

                card["name"] = st.text_input(
                    "Name",
                    value=card.get("name", ""),
                    key=f"guided_tie_attribute_name_{attribute_id}",
                )
                card["description"] = st.text_input(
                    "Description",
                    value=card.get("description", ""),
                    key=f"guided_tie_attribute_description_{attribute_id}",
                )
                attribute_types = ["categorical", "numeric", "temporal"]
                current_type = card.get("type", "categorical")
                if current_type not in attribute_types:
                    current_type = "categorical"
                card["type"] = st.selectbox(
                    "Type",
                    options=attribute_types,
                    index=attribute_types.index(current_type),
                    key=f"guided_tie_attribute_type_{attribute_id}",
                )

                if card["type"] == "categorical":
                    card["options_text"] = st.text_area(
                        "Options (one per line)",
                        value=card.get("options_text", ""),
                        key=f"guided_tie_attribute_options_{attribute_id}",
                    )
                elif card["type"] == "numeric":
                    minimum_col, maximum_col = st.columns(2)
                    with minimum_col:
                        card["minimum"] = st.number_input(
                            "Minimum",
                            value=float(card.get("minimum", 0.0)),
                            key=f"guided_tie_attribute_minimum_{attribute_id}",
                        )
                    with maximum_col:
                        card["maximum"] = st.number_input(
                            "Maximum",
                            value=float(card.get("maximum", 1.0)),
                            key=f"guided_tie_attribute_maximum_{attribute_id}",
                        )
                else:
                    st.caption(
                        "Temporal values accept YYYY, YYYY-MM, or YYYY-MM-DD, with an "
                        "optional end value."
                    )

        if st.button("Add tie attribute", key="guided_tie_attribute_add"):
            _add_guided_tie_attribute(state)
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="guided_back_6"):
                state["step"] = 5
                st.rerun()
        with col2:
            if st.button("Next", key="guided_next_6", type="primary"):
                try:
                    _guided_config_for_review(state)
                except ValueError as exc:
                    st.error(str(exc))
                    return
                state["step"] = 7
                st.rerun()

    else:
        config: dict | None = None
        try:
            config = _guided_config_for_review(state)
        except ValueError as exc:
            st.error(f"Review failed: {exc}")

        if config is not None:
            summary = _guided_review_summary(config)
            st.json(summary)
            with st.expander("Generated config preview"):
                st.code(yaml.safe_dump(config, sort_keys=False), language="yaml")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Back", key="guided_back_7"):
                state["step"] = 6
                st.rerun()
        with col2:
            if st.button("Start over", key="guided_reset"):
                _reset_guided_wizard()
                st.rerun()
        with col3:
            if st.button("Create project", key="guided_create_project", type="primary"):
                try:
                    entry = create_and_register_project(projects_root, registry_path, _guided_config_for_review(state))
                except (OSError, ValueError) as exc:
                    st.error(f"Unable to create project: {exc}")
                else:
                    if state.get("source_origin") == "upload":
                        _cleanup_guided_upload(state)
                    _reset_guided_wizard()
                    st.session_state.active_project_slug = entry["slug"]
                    st.success(f"Created project {entry['name']}.")
                    st.rerun()


def _home_screen_state(projects: list[dict], active_project_slug: str | None) -> dict:
    visible_projects = [project for project in projects if not project.get("archived", False)]
    active_project = None
    if active_project_slug:
        active_project = next(
            (project for project in visible_projects if project.get("slug") == active_project_slug),
            None,
        )
    return {
        "projects": visible_projects,
        "active_project": active_project,
        "show_home_screen": active_project is None,
    }


def _render_project_home_screen(projects: list[dict]) -> None:
    with st.expander("Provider configuration", expanded=False):
        provider_id = st.selectbox(
            "Provider to configure",
            options=[provider for provider in list_provider_ids() if provider != "fake"],
            key="provider_config_provider_id",
        )
        summary = _provider_settings_summary(provider_id)
        if summary["configured"]:
            st.caption(f"Configured base URL: {summary['base_url']}")
        else:
            st.warning(
                f"{provider_id} is not fully configured. Expected env vars: {', '.join(summary['env_vars'])}"
            )
        st.caption(
            "Settings are stored at the app level for this server instance and are not written into project configs."
        )
        st.caption(
            "API keys are not stored by the app. Set provider credentials in the server environment before "
            "starting both Streamlit and the worker process."
        )
        base_url_default = ""
        if summary["uses_stored_settings"]:
            base_url_default = summary["base_url"]
        elif provider_id == "gemini":
            base_url_default = "https://generativelanguage.googleapis.com/v1beta"
        elif provider_id == "openrouter":
            base_url_default = "https://openrouter.ai/api/v1"
        elif provider_id == "ollama":
            base_url_default = "http://localhost:11434/api"
        base_url = st.text_input(
            "Base URL",
            value=base_url_default,
            key=f"provider_base_url_{provider_id}",
        )
        if provider_id == "gemini":
            st.code("Required environment variable: GEMINI_API_KEY", language="text")
        elif provider_id == "openrouter":
            st.code("Required environment variable: OPENROUTER_API_KEY", language="text")
        elif provider_id == "ollama":
            st.code(
                "Optional environment variable: OLLAMA_API_KEY\n"
                "Required only for authenticated/cloud Ollama endpoints.",
                language="text",
            )
        save_col, clear_col = st.columns(2)
        with save_col:
            if st.button("Save provider settings", key=f"save_provider_settings_{provider_id}"):
                _save_provider_settings_from_form(provider_id, base_url=base_url)
                st.success(f"Saved settings for {provider_id}.")
                st.rerun()
        with clear_col:
            if st.button("Clear stored provider settings", key=f"clear_provider_settings_{provider_id}"):
                _clear_provider_settings(provider_id)
                st.success(f"Cleared stored settings for {provider_id}.")
                st.rerun()

    st.subheader("Projects")
    mode = _normalized_project_home_mode(st.session_state.get("project_home_mode"))

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Open existing project", key="project_home_open_mode", type="primary" if mode == "open" else "secondary"):
            st.session_state.project_home_mode = "open"
            st.rerun()
    with col2:
        if st.button("Create new project", key="project_home_create_mode", type="primary" if mode == "create" else "secondary"):
            st.session_state.project_home_mode = "create"
            st.rerun()
    with col3:
        demo_installed = any(project.get("slug") == DEMO_PROJECT_SLUG for project in projects)
        if st.button(
            "Install demo project",
            key="project_home_install_demo",
            disabled=demo_installed,
        ):
            try:
                entry = install_demo_project(APP_PATHS)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                st.error(f"Unable to install demo project: {exc}")
            else:
                st.session_state.active_project_slug = entry["slug"]
                st.success("Installed the CORN demo project.")
                st.rerun()

    mode = _normalized_project_home_mode(st.session_state.get("project_home_mode"))
    if mode == "create":
        _render_guided_project_setup(PROJECTS_DIR, REGISTRY_PATH)
    else:
        _render_open_existing_projects(projects)


def _render_project_sidebar(runtime_context: dict) -> tuple[str, str, dict]:
    filter_options = fetch_project_filter_options(
        runtime_context["runtime_db_path"],
        runtime_context["filter_schema"],
    )
    scope_catalog = _annotation_scope_catalog(runtime_context["annotations_db_path"])

    with st.sidebar:
        st.header("Researcher")
        workspace_options, workspace_index = _scope_select_options(
            scope_catalog["workspaces"],
            st.session_state.get("workspace_id_input", DEFAULT_WORKSPACE_ID),
            default_value=DEFAULT_WORKSPACE_ID,
        )
        workspace_choice = st.selectbox(
            "Saved workspaces",
            options=workspace_options,
            index=workspace_index,
            key="workspace_choice_input",
        )
        workspace_entry_value = st.text_input(
            "Project workspace",
            key="workspace_id_input",
            disabled=workspace_choice != "Enter new...",
            help="Choose an existing workspace or enter a new one.",
        ).strip()
        workspace_id = _selected_scope_value(
            workspace_choice,
            workspace_entry_value,
            fallback=DEFAULT_WORKSPACE_ID,
        )

        annotator_candidates = scope_catalog["annotators_by_workspace"].get(workspace_id) or scope_catalog[
            "annotators"
        ]
        annotator_options, annotator_index = _scope_select_options(
            annotator_candidates,
            st.session_state.get("annotator_id_input", ""),
        )
        annotator_choice = st.selectbox(
            "Saved annotators",
            options=annotator_options,
            index=annotator_index,
            key="annotator_choice_input",
        )
        annotator_entry_value = st.text_input(
            "Annotator ID",
            key="annotator_id_input",
            disabled=annotator_choice != "Enter new...",
            help="Saved into each annotation and used for per-annotator filtering and export.",
        ).strip()
        annotator_id = _selected_scope_value(annotator_choice, annotator_entry_value)

        st.header("Filters")
        filter_values: dict[str, object] = {}
        for filter_config in runtime_context["filter_schema"]:
            field = filter_config["field"]
            label = filter_config["label"]
            filter_type = filter_config["type"]
            state_key = f"project_filter_{field}"
            if filter_type == "select":
                choice = st.selectbox(label, options=["All"] + filter_options.get(field, []), key=state_key)
                filter_values[field] = None if choice == "All" else choice
            elif filter_type == "multi_select":
                filter_values[field] = st.multiselect(
                    label,
                    options=filter_options.get(field, []),
                    key=state_key,
                )
            elif filter_type == "text_contains":
                filter_values[field] = st.text_input(label, key=state_key).strip() or None
            elif filter_type == "boolean":
                choice = st.selectbox(label, options=["All", "Yes", "No"], key=state_key)
                filter_values[field] = None if choice == "All" else choice == "Yes"

        annotation_status = st.selectbox(
            "Annotated by me",
            options=["Not yet", "Done", "All"],
            index=0,
            key="filter_annotation_status",
            help="Filter by whether the current annotator has saved an annotation for this document.",
        )
        llm_status = st.selectbox(
            "LLM coverage",
            options=["All", "Has runs", "Not yet run"],
            key="filter_llm_status",
            help="Filter by whether any LLM model has produced a valid annotation for this document.",
        )
        document_id_filter = st.text_area(
            "Document ID",
            key="filter_document_id",
            height=80,
            help="One ID for exact match. Paste multiple IDs (one per line) to find all of them.",
        ).strip() or None
        sample_size = st.number_input(
            "Random sample size",
            min_value=0,
            value=0,
            step=1,
            help="0 returns all matching rows.",
            key="filter_sample_size",
        )

        if st.button("Run query", use_container_width=True):
            if annotation_status != "All" and not annotator_id:
                st.error("Enter an annotator ID before filtering by annotation status.")
                return workspace_id, annotator_id, filter_values
            normalized_annotation_status = (
                "unannotated" if annotation_status == "Not yet"
                else "annotated" if annotation_status == "Done"
                else "all"
            )
            llm_annotation_status = (
                "has_llm" if llm_status == "Has runs"
                else "no_llm" if llm_status == "Not yet run"
                else "all"
            )

            st.session_state.query_results = query_project_documents(
                runtime_db_path=runtime_context["runtime_db_path"],
                filter_schema=runtime_context["filter_schema"],
                filter_values=filter_values,
                sample_size=sample_size or None,
                annotations_db_path=runtime_context["annotations_db_path"],
                annotator_id=annotator_id or None,
                workspace_id=workspace_id or None,
                annotation_status=normalized_annotation_status,
                document_id_filter=document_id_filter,
                llm_annotation_status=llm_annotation_status,
            )
            st.session_state.query_has_run = True
            st.session_state.selected_document_id = None
            st.session_state.annotation_document_id = None
            st.session_state.annotation_editor_open = False

        if st.button("Rebuild project cache", use_container_width=True):
            build_project_runtime(runtime_context["project_path"])
            st.session_state.query_results = []
            st.session_state.query_has_run = False
            st.session_state.selected_document_id = None
            st.session_state.annotation_document_id = None
            st.session_state.annotation_editor_open = False
            st.success("Project runtime cache rebuilt.")

        _render_export_controls(
            annotations_db_path=runtime_context["annotations_db_path"],
            workspace_id=workspace_id,
            annotator_id=annotator_id,
        )

    return workspace_id, annotator_id, filter_values


def _document_selector_state(
    results: list[dict],
    selected_document_id: int | None,
) -> tuple[dict[str, int | None], int, int | None]:
    options: dict[str, int | None] = {"Select a document": None}
    for row in results:
        label = f"{row['id']} | {row.get('title') or row['link']}"
        options[label] = row["id"]

    if selected_document_id in options.values():
        selected_label = next(
            label for label, document_id in options.items() if document_id == selected_document_id
        )
    else:
        selected_label = "Select a document"
        selected_document_id = None

    index = list(options.keys()).index(selected_label)
    return options, index, selected_document_id


def _document_table_rows(
    results: list[dict],
    document_config: dict[str, str],
    metadata_fields: list[str],
) -> list[dict]:
    rows: list[dict] = []
    id_field = document_config["id_field"]
    title_field = document_config.get("title_field")
    for row in results:
        table_row = {id_field: row.get("document_id", row.get("link", "")) or ""}
        if title_field:
            table_row[title_field] = row.get("document_title", row.get("title", "")) or ""
        for field_name in metadata_fields:
            if field_name in table_row:
                continue
            table_row[field_name] = row.get(field_name, "")
        rows.append(table_row)
    return rows


def _initial_query_results() -> list[dict]:
    return []


def _should_show_query_results(query_has_run: bool, query_results: list[dict]) -> bool:
    return query_has_run and bool(query_results)


def _should_open_annotation_editor(
    editor_open: bool,
    annotation_document_id: int | None,
    current_document_id: int,
) -> bool:
    return editor_open and annotation_document_id == current_document_id


def _render_project_document_selector(
    results: list[dict],
    runtime_db_path: Path,
    document_config: dict[str, str],
    metadata_fields: list[str],
    annotations_db_path: Path | None = None,
) -> dict | None:
    if not results:
        st.info("Run a query to load matching documents.")
        return None

    table_rows = _document_table_rows(results, document_config, metadata_fields)
    if annotations_db_path is not None:
        llm_models = get_llm_models_by_document(annotations_db_path)
        for i, row in enumerate(results):
            models = llm_models.get(str(row["id"]), [])
            table_rows[i]["LLM"] = ", ".join(models) if models else "—"
    st.subheader("Matching documents")
    st.dataframe(table_rows, use_container_width=True)

    options, default_index, _ = _document_selector_state(
        results,
        st.session_state.get("selected_document_id"),
    )
    selected_label = st.selectbox(
        "Open document",
        options=list(options.keys()),
        index=default_index,
    )
    selected_id = options[selected_label]
    st.session_state.selected_document_id = selected_id
    if selected_id is None:
        return None
    return get_project_document_by_id(runtime_db_path, selected_id)


def _render_annotation_editor(
    document: dict,
    tie_options: list[str],
    tie_attribute_schema: list[dict],
) -> None:
    if "annotation_states" not in st.session_state:
        st.session_state.annotation_states = {}
    document_id_str = str(document["id"])
    annotation_state = st.session_state.annotation_state
    default_type = tie_options[0] if tie_options else None
    content = document.get("content", "")

    st.subheader("Annotations")
    st.caption(
        f"Source: {str(annotation_state.get('annotation_source', 'human')).replace('_', ' ').title()}"
    )
    for entry_index, entry in enumerate(annotation_state["network_data"]):
        st.markdown(f"**Network entry {entry_index + 1}**")
        source_key = f"source_{entry_index}"
        target_key = f"target_{entry_index}"
        entry["source"] = st.text_input("Source node", value=entry["source"], key=source_key)
        entry["target"] = st.text_input("Target node", value=entry["target"], key=target_key)

        for tie_index, tie in enumerate(entry["ties"]):
            st.caption(f"Tie {tie_index + 1}")
            tie_key_prefix = f"entry_{entry_index}_tie_{tie_index}"
            if tie_options:
                initial_index = 0
                if tie["type"] in tie_options:
                    initial_index = tie_options.index(tie["type"])
                tie["type"] = st.selectbox(
                    "Tie type",
                    options=tie_options,
                    index=initial_index,
                    key=f"{tie_key_prefix}_type",
                )
            else:
                tie["type"] = st.text_input(
                    "Tie type",
                    value=tie["type"],
                    key=f"{tie_key_prefix}_type_text",
                )

            tie["reasoning_text_span"] = st.text_area(
                "Reasoning text (span)",
                value=tie.get("reasoning_text_span", ""),
                key=f"{tie_key_prefix}_reasoning_span",
                help="Optional exact text span from the document.",
            )
            tie["reasoning_text_note"] = st.text_area(
                "Reasoning text (note)",
                value=tie.get("reasoning_text_note", ""),
                key=f"{tie_key_prefix}_reasoning_note",
                help="Optional free-text explanation or LLM rationale.",
            )
            tie["reasoning_text"] = (
                str(tie.get("reasoning_text_span", "") or "").strip()
                or str(tie.get("reasoning_text_note", "") or "").strip()
            )
            if tie_attribute_schema:
                tie.setdefault("attributes", {})
                with st.expander("Attributes", expanded=bool(tie["attributes"])):
                    for attribute in tie_attribute_schema:
                        _render_tie_attribute_input(tie, attribute, tie_key_prefix)
                    _render_unknown_tie_attributes(tie, tie_attribute_schema, tie_key_prefix)
            status_message = _sync_tie_span(tie, content)
            if tie.get("reasoning_text_span", "").strip():
                if status_message.startswith("No exact match"):
                    st.warning(status_message)
                else:
                    st.info(status_message)
            if _tie_needs_span_review(tie, content):
                st.warning("Needs span review: this tie can be saved, but it does not yet have a resolved exact span.")

            pending_span_target = st.session_state.get("pending_span_target", {})
            is_pending_target = (
                pending_span_target.get("document_id") == document["id"]
                and pending_span_target.get("entry_index") == entry_index
                and pending_span_target.get("tie_index") == tie_index
            )
            if st.button(
                "Select span from document",
                key=f"{tie_key_prefix}_capture_span",
                use_container_width=True,
            ):
                _queue_pending_span_target(document["id"], entry_index, tie_index)
                st.info("Select text in the document panel to populate this tie's span.")
                st.rerun()
            if is_pending_target:
                st.caption("Pending span selection: select text in the document panel for this tie.")

            matches = find_reasoning_spans(content, tie["reasoning_text_span"].strip())
            if _requires_span_disambiguation(tie, content):
                options = [
                    f"{start}-{end}: {content[start:end][:80]}"
                    for start, end in matches
                ]
                selected_match = st.selectbox(
                    "Choose matching span",
                    options=list(range(len(options))),
                    format_func=lambda index: options[index],
                    index=tie.get("selected_match") or 0,
                    key=f"{tie_key_prefix}_match",
                )
                tie["selected_match"] = selected_match
                tie["reasoning_start"], tie["reasoning_end"] = matches[selected_match]
                st.caption(
                    f"Stored span: {tie['reasoning_start']}-{tie['reasoning_end']}"
                )
            elif tie.get("reasoning_start") is not None:
                st.caption(f"Stored span: {tie['reasoning_start']}-{tie['reasoning_end']}")

            can_focus_tie = isinstance(tie.get("reasoning_start"), int) and isinstance(
                tie.get("reasoning_end"), int
            )
            if st.button(
                "Locate in document",
                key=f"{tie_key_prefix}_focus",
                disabled=not can_focus_tie,
            ):
                _queue_document_focus_request(
                    document_id=document["id"],
                    start=tie["reasoning_start"],
                    end=tie["reasoning_end"],
                )
                st.rerun()

            if st.button("Remove tie", key=f"{tie_key_prefix}_remove"):
                entry["ties"].pop(tie_index)
                _sync_current_annotation_state_for_document(document_id_str, annotation_state)
                st.rerun()

        if st.button("Add tie", key=f"add_tie_{entry_index}"):
            if entry["ties"]:
                entry["ties"].append(_copied_tie(entry["ties"][0], default_type))
            else:
                entry["ties"].append(_blank_tie(default_type))
            _sync_current_annotation_state_for_document(document_id_str, annotation_state)
            st.rerun()

        if st.button("Remove network entry", key=f"remove_entry_{entry_index}"):
            annotation_state["network_data"].pop(entry_index)
            _sync_current_annotation_state_for_document(document_id_str, annotation_state)
            st.rerun()

        st.divider()

    if st.button("Add network entry"):
        annotation_state["network_data"].append(_blank_network_entry(default_type))
        _sync_current_annotation_state_for_document(document_id_str, annotation_state)
        st.rerun()

    _sync_current_annotation_state_for_document(document_id_str, annotation_state)


def _render_llm_annotation_viewer(
    run: dict,
    default_type: str | None,
    content: str,
) -> None:
    st.caption(f"Validation status: {run['validation_status']}")
    if run["validation_errors"]:
        st.error("\n".join(run["validation_errors"]))
    with st.expander("Rendered prompt used"):
        st.code(run["rendered_prompt_text"], language="text")
    with st.expander("Raw provider response"):
        st.code(run["raw_response"], language="json")
    if run["normalized_payload"] is None:
        st.caption("No normalized annotation payload available.")
        return
    normalized_payload = _normalize_loaded_annotation(
        json.loads(json.dumps(run["normalized_payload"])), default_type
    )
    if any(
        _tie_needs_span_review(tie, content)
        for entry in normalized_payload.get("network_data", [])
        for tie in entry.get("ties", [])
    ):
        st.warning("Needs span review: this annotation includes note-only or unresolved span evidence.")
    network_data = normalized_payload.get("network_data", [])
    if not network_data:
        st.caption("No network entries in this LLM annotation.")
        return
    st.caption("Network entries")
    for entry_index, entry in enumerate(network_data):
        source = entry.get("source", "")
        target = entry.get("target", "")
        st.markdown(f"**Entry {entry_index + 1}: {source} → {target}**")
        for tie_index, tie in enumerate(entry.get("ties", [])):
            st.caption(f"Tie {tie_index + 1} — {tie.get('type', '')}")
            span = str(tie.get("reasoning_text_span", "") or "").strip()
            note = str(tie.get("reasoning_text_note", "") or "").strip()
            if span:
                st.write(f"**Span:** {span}")
            if note:
                st.write(f"**Note:** {note}")
            if tie.get("attributes"):
                st.caption(f"Attributes: {json.dumps(tie['attributes'], sort_keys=True)}")
        st.divider()


def _save_all_pending(
    workspace_id: str,
    annotator_id: str,
    annotations_dir: Path,
    annotations_db_path: Path,
    default_type: str | None,
    tie_attribute_schema: list[dict] | None = None,
) -> bool:
    if not annotator_id:
        st.error("Enter an annotator ID before saving annotations.")
        return False
    if workspace_id == RESERVED_LLM_WORKSPACE_ID:
        st.error("The 'llm' workspace is reserved for worker-managed LLM annotations.")
        return False

    saved_links: list[str] = []
    errors: list[str] = []

    dirty_ids = list(st.session_state.get("dirty_document_ids", set()))
    for doc_id_str in dirty_ids:
        annotation_state = st.session_state.annotation_states.get(doc_id_str)
        document = st.session_state.document_objects.get(doc_id_str)
        if annotation_state is None or document is None:
            continue
        content = document.get("content", "")
        try:
            created_at = annotation_state.get("created_at") or _iso_utcnow()
            payload = _build_save_payload(
                document,
                annotation_state,
                content,
                annotator_id=annotator_id,
                workspace_id=workspace_id,
                created_at=created_at,
                updated_at=_iso_utcnow(),
                tie_attribute_schema=tie_attribute_schema,
            )
            relative_path = _annotation_relative_path(document, workspace_id, annotator_id)
            output_path = save_annotation(
                output_dir=annotations_dir,
                relative_path=relative_path,
                payload=payload,
            )
            save_annotation_record(
                db_path=annotations_db_path,
                payload=payload,
                json_path=output_path,
            )
            normalized = _normalize_loaded_annotation(
                payload, default_type, tie_attribute_schema=tie_attribute_schema
            )
            st.session_state.annotation_states[doc_id_str] = normalized
            _set_annotation_state_baseline(doc_id_str, normalized)
            if st.session_state.get("annotation_document_id") == document["id"]:
                st.session_state.annotation_state = normalized
            st.session_state.dirty_document_ids.discard(doc_id_str)
            saved_links.append(str(document["link"]))
        except IncompleteAnnotationError as exc:
            errors.append(f"{document['link']}: {exc}")

    if saved_links:
        _queue_save_feedback(saved_links, errors)
        return True
    if not errors:
        st.info("No pending annotations to save.")
    for err in errors:
        st.error(err)
    return False


def _refresh_annotations_from_storage(
    workspace_id: str,
    annotator_id: str,
    annotations_dir: Path,
    annotations_db_path: Path,
    default_type: str | None,
    tie_attribute_schema: list[dict] | None = None,
) -> tuple[int, int]:
    """Reload cached annotation states from storage so collaborator saves become visible.

    Documents with unsaved local edits are skipped so refresh never discards pending work.
    Returns (refreshed_count, skipped_dirty_count).
    """
    annotation_states = st.session_state.get("annotation_states", {})
    document_objects = st.session_state.get("document_objects", {})
    dirty_ids = st.session_state.get("dirty_document_ids", set())

    refreshed = 0
    skipped_dirty = 0
    for doc_id_str in list(annotation_states.keys()):
        document = document_objects.get(doc_id_str)
        if document is None:
            continue
        if doc_id_str in dirty_ids:
            skipped_dirty += 1
            continue
        payload = _read_annotation_payload(
            document,
            workspace_id,
            annotator_id,
            annotations_dir,
            annotations_db_path,
        )
        normalized = _normalize_loaded_annotation(
            payload, default_type, tie_attribute_schema=tie_attribute_schema
        )
        annotation_states[doc_id_str] = normalized
        _set_annotation_state_baseline(doc_id_str, normalized)
        if str(st.session_state.get("annotation_document_id")) == doc_id_str:
            st.session_state.annotation_state = normalized
            st.session_state.annotation_scope_key = _annotation_scope_key(workspace_id, annotator_id)
            st.session_state.annotation_editor_open = False
        refreshed += 1
    return refreshed, skipped_dirty


def _render_session_refresh_button(
    workspace_id: str,
    annotator_id: str,
    annotations_dir: Path,
    annotations_db_path: Path,
    default_type: str | None,
    tie_attribute_schema: list[dict] | None = None,
) -> None:
    feedback = st.session_state.pop("session_refresh_feedback", None)
    if feedback:
        st.success(f"Refreshed {feedback['refreshed']} annotation(s) from storage.")
        if feedback["skipped_dirty"]:
            st.warning(
                f"Kept {feedback['skipped_dirty']} document(s) with unsaved changes. "
                "Save them first to refresh those too."
            )

    if st.button(
        "Refresh annotations",
        use_container_width=True,
        key="session_refresh_button",
        help="Reload annotations saved by collaborators without leaving the project.",
    ):
        refreshed, skipped_dirty = _refresh_annotations_from_storage(
            workspace_id=workspace_id,
            annotator_id=annotator_id,
            annotations_dir=annotations_dir,
            annotations_db_path=annotations_db_path,
            default_type=default_type,
            tie_attribute_schema=tie_attribute_schema,
        )
        st.session_state.session_refresh_feedback = {
            "refreshed": refreshed,
            "skipped_dirty": skipped_dirty,
        }
        st.rerun()


def _render_session_save_button(
    workspace_id: str,
    annotator_id: str,
    annotations_dir: Path,
    annotations_db_path: Path,
    default_type: str | None,
    tie_attribute_schema: list[dict] | None = None,
) -> None:
    feedback = st.session_state.pop("session_save_feedback", None)
    if feedback:
        saved_links = feedback.get("saved_links", [])
        errors = feedback.get("errors", [])
        if saved_links:
            st.success(f"Saved {len(saved_links)} annotation(s).")
        for err in errors:
            st.error(err)

    dirty_ids = st.session_state.get("dirty_document_ids", set())
    dirty_count = len(dirty_ids)

    st.divider()
    if dirty_count:
        with st.popover(f"{dirty_count} annotation(s) pending save", use_container_width=True):
            doc_objects = st.session_state.get("document_objects", {})
            links = [
                str(doc_objects[doc_id]["link"]) if doc_id in doc_objects else f"id:{doc_id}"
                for doc_id in sorted(dirty_ids)
            ]
            st.caption("Click the copy icon to copy all IDs.")
            st.code("\n".join(links), language=None)

    if st.button(
        "Done / Save all",
        type="primary",
        use_container_width=True,
        key="session_save_button",
    ):
        if _save_all_pending(
            workspace_id=workspace_id,
            annotator_id=annotator_id,
            annotations_dir=annotations_dir,
            annotations_db_path=annotations_db_path,
            default_type=default_type,
            tie_attribute_schema=tie_attribute_schema,
        ):
            st.rerun()


def main() -> None:
    initialize_data_directory(APP_PATHS)
    st.set_page_config(
        page_title=f"{CORN_APP_NAME} | {CORN_APP_SUBTITLE}",
        page_icon=str(_branding_assets(_theme_base())["page_icon"]),
        layout="wide",
    )
    if "query_results" not in st.session_state:
        st.session_state.query_results = _initial_query_results()
    if "query_has_run" not in st.session_state:
        st.session_state.query_has_run = False
    if "selected_document_id" not in st.session_state:
        st.session_state.selected_document_id = None
    if "annotation_document_id" not in st.session_state:
        st.session_state.annotation_document_id = None
    if "annotation_editor_open" not in st.session_state:
        st.session_state.annotation_editor_open = False
    if "annotation_scope_key" not in st.session_state:
        st.session_state.annotation_scope_key = None
    if "workspace_id_input" not in st.session_state:
        st.session_state.workspace_id_input = DEFAULT_WORKSPACE_ID
    if "annotator_id_input" not in st.session_state:
        st.session_state.annotator_id_input = ""
    if "pending_span_target" not in st.session_state:
        st.session_state.pending_span_target = _pending_span_target_template()
    if "pending_create_entry" not in st.session_state:
        st.session_state.pending_create_entry = _pending_create_entry_template()
    if "create_entry_form" not in st.session_state:
        st.session_state.create_entry_form = _create_entry_form_defaults(None)
    if "handled_create_entry_nonce" not in st.session_state:
        st.session_state.handled_create_entry_nonce = None
    if "document_focus_request" not in st.session_state:
        st.session_state.document_focus_request = None
    if "document_focus_request_counter" not in st.session_state:
        st.session_state.document_focus_request_counter = 0
    if "active_project_slug" not in st.session_state:
        st.session_state.active_project_slug = None
    if "project_home_mode" not in st.session_state:
        st.session_state.project_home_mode = "open"
    if "annotation_states" not in st.session_state:
        st.session_state.annotation_states = {}
    if "annotation_state_baselines" not in st.session_state:
        st.session_state.annotation_state_baselines = {}
    if "document_objects" not in st.session_state:
        st.session_state.document_objects = {}
    if "dirty_document_ids" not in st.session_state:
        st.session_state.dirty_document_ids = set()

    project_state = _home_screen_state(
        list_projects(REGISTRY_PATH, include_archived=True),
        active_project_slug=st.session_state.get("active_project_slug"),
    )
    if project_state["show_home_screen"]:
        _render_brand_header(show_home_screen=True)
        _render_project_home_screen(project_state["projects"])
        return

    active_project = project_state["active_project"]
    _render_brand_header(show_home_screen=False)
    if st.button("Back to project home", key="back_to_project_home"):
        st.session_state.active_project_slug = None
        st.rerun()
    runtime_context = _project_runtime_context(active_project)
    project_status = get_project_status(runtime_context["project_path"])
    st.caption(f"Active project: {active_project['name']} ({active_project['slug']})")
    if project_status != "ready":
        st.warning(f"Project status: {project_status}")
        if st.button("Build project cache", key="build_project_cache"):
            build_project_runtime(runtime_context["project_path"])
            st.rerun()
        return

    tie_options = [tie["label"] for tie in runtime_context["config"]["tie_schema"]]
    default_type = tie_options[0] if tie_options else None

    workspace_id, annotator_id, _ = _render_project_sidebar(runtime_context)

    with st.expander("LLM reporting", expanded=False):
        _render_llm_reporting_panel(runtime_context)

    visible_results = (
        st.session_state.query_results
        if _should_show_query_results(
            st.session_state.get("query_has_run", False),
            st.session_state.query_results,
        )
        else []
    )
    document = _render_project_document_selector(
        visible_results,
        runtime_context["runtime_db_path"],
        runtime_context["config"]["document"],
        [field["name"] for field in runtime_context["config"]["metadata_fields"]],
        annotations_db_path=runtime_context["annotations_db_path"],
    )

    document_panel, annotation_panel = st.columns([3, 2])

    with document_panel:
        if document is None:
            st.info("Run a query, then select a document to read.")
            return

        current_scope_key = _annotation_scope_key(workspace_id, annotator_id)
        if (
            st.session_state.get("annotation_document_id") != document["id"]
            or st.session_state.get("annotation_scope_key") != current_scope_key
        ):
            _load_annotation_state(
                document,
                default_type,
                workspace_id,
                annotator_id,
                runtime_context["annotations_dir"],
                runtime_context["annotations_db_path"],
                tie_attribute_schema=runtime_context["config"].get("tie_attribute_schema", []),
            )
            st.session_state.annotation_editor_open = False

        st.subheader(document.get("title") or document["link"])
        metadata = {key: value for key, value in document.items() if key != "content"}
        st.json(metadata, expanded=False)
        st.caption("Use 'Select span from document' in the annotation panel to capture exact evidence spans.")
        focus_request = _document_focus_request_for_document(
            st.session_state.get("document_focus_request"),
            current_document_id=document["id"],
        )
        selection = document_selection(
            document.get("content", ""),
            key=f"document_selection_{document['id']}",
            height=520,
            highlights=build_highlight_spans(st.session_state.annotation_state),
            focus_request=focus_request,
            selection_mode=(
                "capture_span"
                if st.session_state.get("pending_span_target", {}).get("document_id") == document["id"]
                else "create_entry"
            ),
        )
        if isinstance(selection, dict):
            if selection.get("action") == "create_annotation_entry":
                selection_nonce = selection.get("nonce")
                if selection_nonce != st.session_state.get("handled_create_entry_nonce"):
                    _queue_pending_create_entry(
                        document["id"],
                        selection,
                        default_type,
                    )
                    st.session_state.handled_create_entry_nonce = selection_nonce
                    st.session_state.annotation_editor_open = True
                    st.rerun()
            pending_span_target = st.session_state.get("pending_span_target", {})
            if selection.get("action") == "use_selected_span":
                if _apply_pending_selection_to_target_tie(
                    annotation_state=st.session_state.annotation_state,
                    pending_selection={
                        "document_id": document["id"],
                        "text": selection.get("text"),
                        "start": selection.get("start"),
                        "end": selection.get("end"),
                        "nonce": selection.get("nonce"),
                    },
                    pending_span_target=pending_span_target,
                    session_state=st.session_state,
                    widget_key_prefix=(
                        f"entry_{pending_span_target.get('entry_index')}_tie_{pending_span_target.get('tie_index')}"
                        if pending_span_target
                        else None
                    ),
                ):
                    _sync_current_annotation_state_for_document(
                        str(document["id"]),
                        st.session_state.annotation_state,
                    )
                    _reset_pending_span_target()
                    st.session_state.annotation_editor_open = True
                    st.rerun()
        if st.session_state.get("pending_create_entry", {}).get("document_id") == document["id"]:
            _render_create_entry_dialog(document, tie_options)

    with annotation_panel:
        if document is None:
            return
        st.subheader("Annotation Panel")

        with st.expander("Annotations", expanded=True):
            llm_runs = list_llm_runs_for_document(runtime_context["annotations_db_path"], str(document["id"]))
            selector_options, default_record_label = _annotation_selector_options(
                workspace_id=workspace_id,
                annotator_id=annotator_id,
                human_payload=st.session_state.annotation_state,
                llm_runs=llm_runs,
            )
            selector_state_key = f"annotation_record_{active_project['slug']}_{document['id']}"
            _resolve_annotation_record_selection(
                session_state=st.session_state,
                selector_state_key=selector_state_key,
                selector_options=selector_options,
                default_record_label=default_record_label,
            )
            selected_record_label = st.selectbox(
                "Annotation record",
                options=list(selector_options.keys()),
                key=selector_state_key,
            )
            selected_record = selector_options[selected_record_label]

            if selected_record["record_type"] == "human":
                # Human annotation editor
                if _should_open_annotation_editor(
                    st.session_state.get("annotation_editor_open", False),
                    st.session_state.get("annotation_document_id"),
                    document["id"],
                ):
                    _render_annotation_editor(
                        document,
                        tie_options,
                        runtime_context["config"].get("tie_attribute_schema", []),
                    )
                else:
                    st.caption("Open the editor to add or revise annotations for this document.")
                    if st.button("Open annotation editor", key=f"open_editor_{document['id']}"):
                        st.session_state.annotation_editor_open = True
                        st.session_state.dirty_document_ids.add(str(document["id"]))
                        st.rerun()
            else:
                # LLM record selected — offer two actions
                selected_run = selected_record["run"]
                llm_panel_mode_key = f"llm_panel_{active_project['slug']}_{document['id']}"

                view_col, dup_col = st.columns(2)
                with view_col:
                    if st.button(
                        "View LLM annotation",
                        key=f"view_llm_{active_project['slug']}_{document['id']}",
                        use_container_width=True,
                    ):
                        st.session_state[llm_panel_mode_key] = "view"
                        st.rerun()
                with dup_col:
                    if st.button(
                        "Duplicate into human editor",
                        key=f"duplicate_llm_run_{active_project['slug']}_{document['id']}",
                        disabled=not annotator_id,
                        use_container_width=True,
                        help="Enter an annotator ID first so the duplicated annotation has a human owner.",
                    ):
                        duplicated_payload = _duplicate_llm_payload_for_human_editing(
                            selected_run["normalized_payload"],
                            workspace_id=workspace_id,
                            annotator_id=annotator_id,
                        )
                        normalized_dup = _normalize_loaded_annotation(duplicated_payload, default_type)
                        st.session_state.annotation_state = normalized_dup
                        st.session_state.annotation_states[str(document["id"])] = normalized_dup
                        st.session_state.annotation_document_id = document["id"]
                        st.session_state.annotation_scope_key = _annotation_scope_key(workspace_id, annotator_id)
                        st.session_state.annotation_editor_open = True
                        st.session_state.dirty_document_ids.add(str(document["id"]))
                        st.session_state[llm_panel_mode_key] = None
                        st.session_state[_pending_selector_override_key(selector_state_key)] = _human_annotation_label(
                            workspace_id,
                            annotator_id,
                            duplicated_payload,
                        )
                        st.success("Loaded LLM annotation into the human editor.")
                        st.rerun()

                llm_panel_mode = st.session_state.get(llm_panel_mode_key)
                if llm_panel_mode == "view":
                    # LLM annotation viewer
                    _render_llm_annotation_viewer(
                        selected_run,
                        default_type,
                        document.get("content", ""),
                    )
                else:
                    st.caption(f"Validation status: {selected_run['validation_status']}")
                    if selected_run["validation_errors"]:
                        st.error("\n".join(selected_run["validation_errors"]))

        with st.expander("Document portability", expanded=False):
            _render_document_portability_controls(
                document=document,
                annotations_db_path=runtime_context["annotations_db_path"],
                annotations_dir=runtime_context["annotations_dir"],
                runtime_db_path=runtime_context["runtime_db_path"],
                valid_tie_labels={tie["label"] for tie in runtime_context["config"]["tie_schema"]},
                tie_attribute_schema=runtime_context["config"].get("tie_attribute_schema", []),
                workspace_id=workspace_id,
                annotator_id=annotator_id,
            )

        with st.expander("LLM annotation tools", expanded=False):
            _render_llm_annotation_tools(
                document=document,
                query_results=visible_results,
                runtime_context=runtime_context,
                workspace_id=workspace_id,
                annotator_id=annotator_id,
                default_type=default_type,
            )

    with st.sidebar:
        _render_session_save_button(
            workspace_id=workspace_id,
            annotator_id=annotator_id,
            annotations_dir=runtime_context["annotations_dir"],
            annotations_db_path=runtime_context["annotations_db_path"],
            default_type=default_type,
            tie_attribute_schema=runtime_context["config"].get("tie_attribute_schema", []),
        )
        _render_session_refresh_button(
            workspace_id=workspace_id,
            annotator_id=annotator_id,
            annotations_dir=runtime_context["annotations_dir"],
            annotations_db_path=runtime_context["annotations_db_path"],
            default_type=default_type,
            tie_attribute_schema=runtime_context["config"].get("tie_attribute_schema", []),
        )


if __name__ == "__main__":
    main()
