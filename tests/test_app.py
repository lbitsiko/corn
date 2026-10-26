from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import app
from app import (
    _add_guided_tie_attribute,
    _append_network_entry_from_selection,
    _annotation_selector_options,
    _annotation_scope_catalog,
    _available_metadata_columns,
    _annotation_relative_path,
    _blank_tie,
    _branding_assets,
    _build_save_payload,
    _cleanup_guided_upload,
    _copied_tie,
    _default_llm_prompt_template,
    _duplicate_llm_payload_for_human_editing,
    _document_table_rows,
    _document_focus_request_for_document,
    _document_selector_state,
    _ensure_project_default_llm_prompt,
    _focus_request,
    _guided_config_for_review,
    _guided_review_summary,
    _guided_tie_attribute_schema,
    _human_annotation_label,
    _human_reporting_scope_options,
    _initial_query_results,
    _llm_job_summary_rows,
    _llm_reporting_human_document_coverage_options,
    _llm_reporting_direction_options,
    _llm_reporting_empty_state_message,
    _llm_reporting_refresh_button,
    _llm_reporting_scope_options,
    _llm_reporting_summary_stats,
    _llm_prompt_select_options,
    _llm_run_label,
    _llm_runs_as_export_records,
    _llm_worker_status,
    _launch_llm_annotation_job,
    _normalize_loaded_annotation,
    _pending_selector_override_key,
    _pending_create_entry_template,
    _pending_span_target_template,
    _parse_tie_schema_text,
    _persist_llm_prompt_template,
    _provider_settings_summary,
    _provider_status_message,
    _provider_uses_freeform_model_id,
    _queue_save_feedback,
    _refresh_annotations_from_storage,
    _remove_guided_tie_attribute,
    _save_provider_settings_from_form,
    _save_all_pending,
    _server_source_options,
    _set_annotation_state_baseline,
    _clear_provider_settings,
    _query_result_document_ids,
    _rendered_llm_prompt_preview,
    _render_annotation_editor,
    _render_tie_attribute_input,
    _render_unknown_tie_attributes,
    _sync_current_annotation_state_for_document,
    _sync_dirty_state_for_document,
    _apply_pending_selection_to_target_tie,
    _resolve_annotation_record_selection,
    _requires_span_disambiguation,
    _scope_select_options,
    _selected_scope_value,
    _single_document_job_ids,
    _should_open_annotation_editor,
    _should_show_query_results,
    _tie_needs_span_review,
    _tie_schema_presets_from_project_configs,
)
from corn.annotations import IncompleteAnnotationError
from corn.app_paths import build_app_paths
from corn.llm_storage import create_llm_job, create_prompt, save_llm_run
from corn.llm_providers import FakeLLMProvider
from corn.storage import save_annotation_record


class SessionStateDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def test_document_selector_state_starts_with_no_document_selected() -> None:
    results = [
        {"id": 1, "title": "Doc 1", "link": "https://example.com/1"},
        {"id": 2, "title": "Doc 2", "link": "https://example.com/2"},
    ]

    options, index, selected_id = _document_selector_state(results, None)

    assert options == {
        "Select a document": None,
        "1 | Doc 1": 1,
        "2 | Doc 2": 2,
    }
    assert index == 0
    assert selected_id is None


def test_document_selector_state_restores_existing_selection() -> None:
    results = [
        {"id": 1, "title": "Doc 1", "link": "https://example.com/1"},
        {"id": 2, "title": "Doc 2", "link": "https://example.com/2"},
    ]

    options, index, selected_id = _document_selector_state(results, 2)

    assert options["2 | Doc 2"] == 2
    assert index == 2
    assert selected_id == 2


def test_initial_query_results_are_empty() -> None:
    assert _initial_query_results() == []


def test_app_constants_use_configured_application_paths() -> None:
    paths = build_app_paths()

    assert app.APP_PATHS == paths
    assert app.APP_ROOT == paths.resource_dir
    assert app.DATA_DIR == paths.data_dir
    assert app.LOGOS_DIR == paths.logos_dir
    assert app.REGISTRY_PATH == paths.registry_path
    assert app.PROVIDER_SETTINGS_PATH == paths.provider_settings_path


def test_server_source_options_are_relative_to_data_directory(tmp_path, monkeypatch) -> None:
    source = tmp_path / "imports" / "source.csv"
    source.parent.mkdir()
    source.write_text("id,text\n1,hello\n", encoding="utf-8")
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(app, "GUIDED_SERVER_SOURCE_IGNORED_ROOTS", ())
    monkeypatch.setattr(app, "GUIDED_SERVER_SOURCE_IGNORED_FILES", ())

    options = _server_source_options("csv")

    assert options == {"imports/source.csv": source}


def test_app_subtitle_uses_notation_branding() -> None:
    assert app.CORN_APP_SUBTITLE == "Corpus-Oriented Relationship Notation"


def test_branding_assets_use_light_variants_for_light_theme() -> None:
    assets = _branding_assets("light")

    assert assets["home_logo"].name == "CORN-with-text-WITH-builtin-background.svg"
    assert assets["page_logo"].name == "CORN-log-WITH-builtin-background.svg"
    assert assets["page_icon"].name == "CORN-icon-white-background-square.svg"


def test_branding_assets_use_dark_variants_for_dark_theme() -> None:
    assets = _branding_assets("dark")

    assert assets["home_logo"].name == "CORN-with-text-WITH-builtin-background.svg"
    assert assets["page_logo"].name == "CORN-log-WITH-builtin-background.svg"
    assert assets["page_icon"].name == "CORN-icon-white-background-square.svg"


def test_default_llm_prompt_template_uses_supported_placeholders() -> None:
    template = _default_llm_prompt_template()

    assert "{{document_text}}" in template
    assert "{{tie_schema_json}}" in template
    assert "{{tie_attribute_schema_json}}" in template
    assert "{{annotation_schema_json}}" in template


def test_ensure_project_default_llm_prompt_creates_and_reuses_default_prompt(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"

    first_prompt = _ensure_project_default_llm_prompt(db_path, "Toy Project")
    second_prompt = _ensure_project_default_llm_prompt(db_path, "Toy Project")

    assert first_prompt["prompt_id"] == second_prompt["prompt_id"]
    assert first_prompt["label"] == "Toy Project default LLM prompt"


def test_persist_llm_prompt_template_forks_when_text_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    default_prompt = _ensure_project_default_llm_prompt(db_path, "Toy Project")

    forked_prompt = _persist_llm_prompt_template(
        db_path,
        selected_prompt_id=default_prompt["prompt_id"],
        template_text=default_prompt["template_text"] + "\nReturn only valid JSON.",
        created_by="annotator-a",
        label="Document fork",
        promote_to_project_default=False,
    )

    assert forked_prompt["prompt_id"] != default_prompt["prompt_id"]
    assert forked_prompt["parent_prompt_id"] == default_prompt["prompt_id"]


def test_query_result_document_ids_return_internal_ids_as_strings() -> None:
    assert _query_result_document_ids([{"id": 7}, {"id": 9}]) == ["7", "9"]


def test_single_document_job_ids_return_current_internal_id_as_string() -> None:
    assert _single_document_job_ids({"id": 7}) == ["7"]


def test_llm_prompt_select_options_restore_selected_prompt() -> None:
    prompts = [
        {"prompt_id": "prompt-1", "label": "Default"},
        {"prompt_id": "prompt-2", "label": "Fork"},
    ]

    options, index = _llm_prompt_select_options(prompts, "prompt-2")

    assert list(options.values()) == ["prompt-1", "prompt-2"]
    assert index == 1


def test_rendered_llm_prompt_preview_renders_document_context() -> None:
    preview = _rendered_llm_prompt_preview(
        project_name="Toy Project",
        document={
            "document_id": "doc-1",
            "document_title": "Doc title",
            "content": "Alpha supports Beta.",
            "topic": "support",
        },
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        template_text="Title={{document_title}}\nText={{document_text}}",
    )

    assert preview == "Title=Doc title\nText=Alpha supports Beta."


def test_default_llm_prompt_template_renders_echo_safe_attribute_examples() -> None:
    preview = _rendered_llm_prompt_preview(
        project_name="Toy Project",
        document={"document_id": "doc-1", "content": "Alpha supports Beta."},
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        template_text=_default_llm_prompt_template(),
        tie_attribute_schema=[
            {
                "name": "valence",
                "type": "categorical",
                "description": "d",
                "options": ["positive", "negative"],
            },
            {"name": "certainty", "type": "numeric", "description": "d", "range": [0.0, 1.0]},
            {"name": "period", "type": "temporal", "description": "d"},
        ],
    )

    assert '\\"positive\\" | \\"negative\\"' in preview
    assert "number between 0.0 and 1.0" in preview
    assert "YYYY or YYYY-MM or YYYY-MM-DD" in preview
    assert "(optional)" not in preview
    assert "exactly one of the listed options, verbatim" in preview
    assert 'The "end" date is optional' in preview
    assert "zero-padded" in preview


def test_provider_status_message_reports_missing_openrouter_key(monkeypatch) -> None:
    monkeypatch.setattr(app, "PROVIDER_SETTINGS_PATH", Path("/tmp/nonexistent-provider-settings.json"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    level, message = _provider_status_message("openrouter")

    assert level == "warning"
    assert "OPENROUTER_API_KEY" in message


def test_provider_settings_summary_uses_stored_app_level_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app, "PROVIDER_SETTINGS_PATH", tmp_path / "provider_settings.json")
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    _save_provider_settings_from_form(
        "openrouter",
        base_url="https://example.com/router",
    )

    summary = _provider_settings_summary("openrouter")

    assert summary["configured"]
    assert summary["base_url"] == "https://example.com/router"
    assert summary["has_api_key"]
    assert summary["uses_stored_settings"]


def test_provider_uses_freeform_model_id_for_remote_model_catalogs() -> None:
    assert _provider_uses_freeform_model_id("gemini")
    assert _provider_uses_freeform_model_id("openrouter")
    assert not _provider_uses_freeform_model_id("ollama")
    assert not _provider_uses_freeform_model_id("fake")


def test_clear_provider_settings_removes_stored_app_level_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app, "PROVIDER_SETTINGS_PATH", tmp_path / "provider_settings.json")
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    _save_provider_settings_from_form(
        "openrouter",
        base_url="https://example.com/router",
    )

    _clear_provider_settings("openrouter")

    summary = _provider_settings_summary("openrouter")
    assert not summary["uses_stored_settings"]


def test_save_provider_settings_from_form_does_not_persist_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app, "PROVIDER_SETTINGS_PATH", tmp_path / "provider_settings.json")

    _save_provider_settings_from_form(
        "openrouter",
        base_url="https://example.com/router",
    )

    assert app.load_provider_settings(app.PROVIDER_SETTINGS_PATH) == {
        "openrouter": {
            "base_url": "https://example.com/router",
        }
    }


def test_llm_worker_status_warns_when_jobs_are_queued_without_running_worker() -> None:
    status = _llm_worker_status(
        [
            {"status": "queued", "worker_id": None},
            {"status": "completed", "worker_id": None},
        ]
    )

    assert status is not None
    assert status[0] == "warning"
    assert "python -m corn.llm_worker" in status[1]


def test_llm_worker_status_reports_info_when_queued_jobs_have_running_worker() -> None:
    status = _llm_worker_status(
        [
            {"status": "queued", "worker_id": None},
            {"status": "running", "worker_id": "worker-1"},
        ]
    )

    assert status == (
        "info",
        "LLM jobs are queued and a worker appears to be running. Use Refresh in Recent LLM jobs to monitor progress.",
    )


def test_llm_worker_status_reports_info_when_worker_was_recently_active() -> None:
    recent_time = datetime.now(timezone.utc).isoformat()
    status = _llm_worker_status(
        [
            {"status": "queued", "worker_id": None, "heartbeat_at": None, "finished_at": None},
            {"status": "completed", "worker_id": None, "heartbeat_at": recent_time, "finished_at": recent_time},
        ]
    )

    assert status == (
        "info",
        "LLM jobs are queued and a worker was active recently. Use Refresh in Recent LLM jobs to monitor progress.",
    )
def test_render_annotation_editor_does_not_mark_document_dirty_on_render(monkeypatch) -> None:
    initial_state = {
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
                        "reasoning_text_note": "",
                        "reasoning_start": 17,
                        "reasoning_end": 37,
                        "selected_match": 0,
                    }
                ],
            }
        ],
    }

    class FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = SessionStateDict({
                "dirty_document_ids": set(),
                "annotation_state": initial_state,
                "annotation_states": {"1": initial_state},
                "annotation_state_baselines": {
                    "1": app._annotation_state_fingerprint(initial_state),
                },
                "pending_span_target": {},
            })

        def subheader(self, *_args, **_kwargs):
            return None

        def caption(self, *_args, **_kwargs):
            return None

        def markdown(self, *_args, **_kwargs):
            return None

        def text_input(self, _label, value="", **_kwargs):
            return value

        def selectbox(self, _label, options, index=0, **_kwargs):
            return options[index]

        def text_area(self, _label, value="", **_kwargs):
            return value

        def warning(self, *_args, **_kwargs):
            return None

        def info(self, *_args, **_kwargs):
            return None

        def button(self, *_args, **_kwargs):
            return False

        def divider(self):
            return None

        def rerun(self):
            raise AssertionError("rerun should not be called during passive render")

    fake_st = FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    _render_annotation_editor(
        document={"id": 1, "content": "Alpha Beta Gamma shared policy agenda"},
        tie_options=["Policy"],
        tie_attribute_schema=[],
    )

    assert fake_st.session_state["dirty_document_ids"] == set()


def test_save_all_pending_clears_dirty_documents_and_queues_feedback(tmp_path: Path, monkeypatch) -> None:
    fake_st = SimpleNamespace(
        session_state=SessionStateDict({
            "dirty_document_ids": {"1"},
            "annotation_states": {
                "1": {
                    "annotation_source": "human_from_llm",
                    "workspace_id": "workspace-1",
                    "annotator_id": "annotator-a",
                    "document_id": "doc-1",
                    "created_at": "2026-06-01T10:00:00+00:00",
                    "updated_at": "2026-06-01T10:00:00+00:00",
                    "network_data": [
                        {
                            "source": "Org A",
                            "target": "Org B",
                            "ties": [
                                {
                                    "type": "Policy",
                                    "reasoning_text": "shared policy agenda",
                                    "reasoning_text_span": "shared policy agenda",
                                    "reasoning_text_note": "",
                                    "reasoning_start": 12,
                                    "reasoning_end": 32,
                                }
                            ],
                        }
                    ],
                }
            },
            "document_objects": {
                "1": {
                    "id": 1,
                    "link": "doc-1",
                    "content": "Alpha shared policy agenda Beta",
                    "document_id": "doc-1",
                }
            },
            "annotation_document_id": 1,
            "annotation_state": {},
        }),
        error=lambda *_args, **_kwargs: None,
        success=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(app, "st", fake_st)

    did_save = _save_all_pending(
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        annotations_dir=tmp_path / "annotations",
        annotations_db_path=tmp_path / "annotations.sqlite",
        default_type="Policy",
    )

    assert did_save
    assert fake_st.session_state["dirty_document_ids"] == set()
    assert fake_st.session_state["session_save_feedback"]["saved_links"] == ["doc-1"]
    assert fake_st.session_state["session_save_feedback"]["errors"] == []


def test_save_all_pending_rejects_reserved_llm_workspace(tmp_path: Path, monkeypatch) -> None:
    errors: list[str] = []
    infos: list[str] = []
    fake_st = SimpleNamespace(
        session_state=SessionStateDict({
            "dirty_document_ids": {"1"},
            "annotation_states": {
                "1": {
                    "annotation_source": "human",
                    "workspace_id": "llm",
                    "annotator_id": "openrouter/claude-3.5-sonnet",
                    "document_id": "doc-1",
                    "created_at": "2026-06-01T10:00:00+00:00",
                    "updated_at": "2026-06-01T10:00:00+00:00",
                    "network_data": [
                        {
                            "source": "Org A",
                            "target": "Org B",
                            "ties": [
                                {
                                    "type": "Policy",
                                    "reasoning_text": "shared policy agenda",
                                    "reasoning_text_span": "shared policy agenda",
                                    "reasoning_text_note": "",
                                    "reasoning_start": 12,
                                    "reasoning_end": 32,
                                }
                            ],
                        }
                    ],
                }
            },
            "document_objects": {
                "1": {
                    "id": 1,
                    "link": "doc-1",
                    "content": "Alpha shared policy agenda Beta",
                    "document_id": "doc-1",
                }
            },
            "annotation_document_id": 1,
            "annotation_state": {},
        }),
        error=errors.append,
        success=lambda *_args, **_kwargs: None,
        info=infos.append,
    )
    monkeypatch.setattr(app, "st", fake_st)

    did_save = _save_all_pending(
        workspace_id="llm",
        annotator_id="openrouter/claude-3.5-sonnet",
        annotations_dir=tmp_path / "annotations",
        annotations_db_path=tmp_path / "annotations.sqlite",
        default_type="Policy",
    )

    assert not did_save
    assert errors == ["The 'llm' workspace is reserved for worker-managed LLM annotations."]
    assert infos == []
    assert fake_st.session_state["dirty_document_ids"] == {"1"}


def _collaborator_refresh_session_state(document: dict, stale_state: dict) -> "SessionStateDict":
    doc_id_str = str(document["id"])
    return SessionStateDict({
        "dirty_document_ids": set(),
        "annotation_states": {doc_id_str: stale_state},
        "annotation_state_baselines": {
            doc_id_str: app._annotation_state_fingerprint(stale_state),
        },
        "document_objects": {doc_id_str: document},
        "annotation_document_id": document["id"],
        "annotation_state": stale_state,
        "annotation_editor_open": True,
        "annotation_scope_key": None,
    })


def test_refresh_annotations_from_storage_loads_collaborator_saves(tmp_path: Path, monkeypatch) -> None:
    document = {
        "id": 1,
        "link": "doc-1",
        "content": "Alpha shared policy agenda Beta",
        "document_id": "doc-1",
    }
    stale_state = _normalize_loaded_annotation(
        {
            "workspace_id": "workspace-1",
            "annotator_id": "annotator-a",
            "document_id": "doc-1",
            "network_data": [],
        },
        "Policy",
    )
    db_path = tmp_path / "annotations.sqlite"
    save_annotation_record(
        db_path,
        {
            "workspace_id": "workspace-1",
            "annotator_id": "annotator-a",
            "document_id": "doc-1",
            "created_at": "2026-07-07T10:00:00+00:00",
            "updated_at": "2026-07-07T10:05:00+00:00",
            "annotation_source": "human",
            "network_data": [
                {
                    "source": "Org A",
                    "target": "Org B",
                    "ties": [
                        {
                            "type": "Policy",
                            "reasoning_text": "shared policy agenda",
                            "reasoning_text_span": "shared policy agenda",
                            "reasoning_text_note": "",
                            "reasoning_start": 6,
                            "reasoning_end": 26,
                        }
                    ],
                }
            ],
        },
        tmp_path / "annotations" / "doc-1.json",
    )

    fake_st = SimpleNamespace(
        session_state=_collaborator_refresh_session_state(document, stale_state),
    )
    monkeypatch.setattr(app, "st", fake_st)

    refreshed, skipped_dirty = _refresh_annotations_from_storage(
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        annotations_dir=tmp_path / "annotations",
        annotations_db_path=db_path,
        default_type="Policy",
    )

    assert refreshed == 1
    assert skipped_dirty == 0
    refreshed_state = fake_st.session_state["annotation_states"]["1"]
    assert refreshed_state["network_data"][0]["source"] == "Org A"
    assert fake_st.session_state["annotation_state"] is refreshed_state
    assert fake_st.session_state["annotation_editor_open"] is False
    assert fake_st.session_state["annotation_state_baselines"]["1"] == app._annotation_state_fingerprint(
        refreshed_state
    )
    assert fake_st.session_state["dirty_document_ids"] == set()


def test_refresh_annotations_from_storage_keeps_unsaved_local_edits(tmp_path: Path, monkeypatch) -> None:
    document = {
        "id": 1,
        "link": "doc-1",
        "content": "Alpha shared policy agenda Beta",
        "document_id": "doc-1",
    }
    local_state = _normalize_loaded_annotation(
        {
            "workspace_id": "workspace-1",
            "annotator_id": "annotator-a",
            "document_id": "doc-1",
            "network_data": [
                {
                    "source": "Local Org",
                    "target": "Other Org",
                    "ties": [],
                }
            ],
        },
        "Policy",
    )
    session_state = _collaborator_refresh_session_state(document, local_state)
    session_state["dirty_document_ids"] = {"1"}
    fake_st = SimpleNamespace(session_state=session_state)
    monkeypatch.setattr(app, "st", fake_st)

    refreshed, skipped_dirty = _refresh_annotations_from_storage(
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        annotations_dir=tmp_path / "annotations",
        annotations_db_path=tmp_path / "annotations.sqlite",
        default_type="Policy",
    )

    assert refreshed == 0
    assert skipped_dirty == 1
    assert fake_st.session_state["annotation_states"]["1"] is local_state
    assert fake_st.session_state["annotation_state"] is local_state
    assert fake_st.session_state["annotation_editor_open"] is True


def test_sync_dirty_state_for_document_marks_dirty_only_when_annotation_differs(monkeypatch) -> None:
    fake_st = SimpleNamespace(
        session_state=SessionStateDict({
            "annotation_states": {
                "1": {
                    "annotation_source": "human",
                    "network_data": [
                        {
                            "source": "Org A",
                            "target": "Org B",
                            "ties": [
                                {
                                    "type": "Policy",
                                    "reasoning_text": "shared policy agenda",
                                    "reasoning_text_span": "shared policy agenda",
                                    "reasoning_text_note": "",
                                    "reasoning_start": 12,
                                    "reasoning_end": 32,
                                }
                            ],
                        }
                    ],
                }
            },
            "annotation_state_baselines": {},
            "dirty_document_ids": set(),
        }),
    )
    monkeypatch.setattr(app, "st", fake_st)

    _set_annotation_state_baseline("1", fake_st.session_state["annotation_states"]["1"])
    _sync_dirty_state_for_document("1")
    assert fake_st.session_state["dirty_document_ids"] == set()

    fake_st.session_state["annotation_states"]["1"]["network_data"][0]["ties"][0]["reasoning_text_note"] = "note"
    _sync_dirty_state_for_document("1")
    assert fake_st.session_state["dirty_document_ids"] == {"1"}


def test_sync_current_annotation_state_for_document_marks_dirty_immediately(monkeypatch) -> None:
    annotation_state = {
        "annotation_source": "human",
        "network_data": [],
    }
    fake_st = SimpleNamespace(
        session_state=SessionStateDict({
            "annotation_states": {"1": {"annotation_source": "human", "network_data": []}},
            "annotation_state_baselines": {},
            "dirty_document_ids": set(),
        }),
    )
    monkeypatch.setattr(app, "st", fake_st)

    _set_annotation_state_baseline("1", annotation_state)
    _append_network_entry_from_selection(
        annotation_state=annotation_state,
        selection={
            "text": "Beta",
            "start": 21,
            "end": 25,
        },
        source="Org A",
        target="Org B",
        tie_type="Policy",
    )

    _sync_current_annotation_state_for_document("1", annotation_state)

    assert fake_st.session_state["dirty_document_ids"] == {"1"}


def test_llm_run_label_includes_provider_prompt_and_status() -> None:
    label = _llm_run_label(
        {
            "provider_id": "fake",
            "model_id": "fake-default",
            "prompt_id": "prompt-1",
            "validation_status": "valid",
            "created_at": "2026-05-31T12:00:00+00:00",
        }
    )

    assert label == "LLM | fake/fake-default | prompt-1 | valid | 2026-05-31T12:00:00+00:00"


def test_llm_runs_as_export_records_uses_latest_valid_run_per_document_and_model(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = create_prompt(
        db_path=db_path,
        template_text="Annotate {{document_text}}",
        created_by="user-a",
        label="Default project prompt",
    )
    job = create_llm_job(
        db_path=db_path,
        document_ids=["1", "2"],
        prompt_id=prompt["prompt_id"],
        provider_id="openrouter",
        model_id="claude-3.5-sonnet",
        request_params={},
        created_by="user-a",
    )

    older_payload = {
        "workspace_id": "llm",
        "annotator_id": "openrouter/claude-3.5-sonnet",
        "document_id": "1",
        "created_at": "2026-06-01T09:00:00+00:00",
        "updated_at": "2026-06-01T09:00:00+00:00",
        "annotation_source": "llm",
        "network_data": [],
    }
    newer_payload = {
        **older_payload,
        "updated_at": "2026-06-01T10:00:00+00:00",
    }
    other_model_payload = {
        **older_payload,
        "annotator_id": "openrouter/gpt-4.1",
        "document_id": "1",
    }

    save_llm_run(
        db_path=db_path,
        job_id=job["job_id"],
        document_id="1",
        prompt_id=prompt["prompt_id"],
        provider_id="openrouter",
        model_id="claude-3.5-sonnet",
        prompt_template_text=prompt["template_text"],
        rendered_prompt_text="prompt old",
        request_params={},
        raw_response="{}",
        normalized_payload=older_payload,
        validation_status="valid",
        validation_errors=[],
        usage={},
        cost={},
    )
    save_llm_run(
        db_path=db_path,
        job_id=job["job_id"],
        document_id="1",
        prompt_id=prompt["prompt_id"],
        provider_id="openrouter",
        model_id="claude-3.5-sonnet",
        prompt_template_text=prompt["template_text"],
        rendered_prompt_text="prompt new",
        request_params={},
        raw_response="{}",
        normalized_payload=newer_payload,
        validation_status="valid",
        validation_errors=[],
        usage={},
        cost={},
    )
    save_llm_run(
        db_path=db_path,
        job_id=job["job_id"],
        document_id="1",
        prompt_id=prompt["prompt_id"],
        provider_id="openrouter",
        model_id="gpt-4.1",
        prompt_template_text=prompt["template_text"],
        rendered_prompt_text="prompt model 2",
        request_params={},
        raw_response="{}",
        normalized_payload=other_model_payload,
        validation_status="valid",
        validation_errors=[],
        usage={},
        cost={},
    )
    save_llm_run(
        db_path=db_path,
        job_id=job["job_id"],
        document_id="2",
        prompt_id=prompt["prompt_id"],
        provider_id="openrouter",
        model_id="claude-3.5-sonnet",
        prompt_template_text=prompt["template_text"],
        rendered_prompt_text="prompt invalid",
        request_params={},
        raw_response="{}",
        normalized_payload=None,
        validation_status="invalid",
        validation_errors=["bad"],
        usage={},
        cost={},
    )

    records = _llm_runs_as_export_records(db_path)
    by_annotator = {row["annotator_id"]: row for row in records}

    assert {row["annotator_id"] for row in records} == {
        "openrouter/claude-3.5-sonnet",
        "openrouter/gpt-4.1",
    }
    assert all(row["document_id"] == "1" for row in records)
    assert by_annotator["openrouter/claude-3.5-sonnet"]["payload"]["updated_at"] == "2026-06-01T10:00:00+00:00"
    assert by_annotator["openrouter/claude-3.5-sonnet"]["json_path"] == (
        "llm/openrouter_claude-3.5-sonnet/1.json"
    )
    assert by_annotator["openrouter/gpt-4.1"]["json_path"] == "llm/openrouter_gpt-4.1/1.json"


def test_human_annotation_label_uses_scope_and_timestamp() -> None:
    label = _human_annotation_label(
        "workspace-1",
        "annotator-a",
        {"updated_at": "2026-05-31T13:00:00+00:00"},
    )

    assert label == "Human | workspace-1/annotator-a | 2026-05-31T13:00:00+00:00"


def test_annotation_selector_options_include_human_and_llm_records() -> None:
    options, default_label = _annotation_selector_options(
        workspace_id="workspace-1",
        annotator_id="annotator-a",
        human_payload={"updated_at": "2026-05-31T13:00:00+00:00", "network_data": []},
        llm_runs=[
            {
                "provider_id": "fake",
                "model_id": "fake-default",
                "prompt_id": "prompt-1",
                "validation_status": "valid",
                "created_at": "2026-05-31T12:00:00+00:00",
            }
        ],
    )

    assert list(options.keys()) == [
        "Human | workspace-1/annotator-a | 2026-05-31T13:00:00+00:00",
        "LLM | fake/fake-default | prompt-1 | valid | 2026-05-31T12:00:00+00:00",
    ]
    assert default_label == "LLM | fake/fake-default | prompt-1 | valid | 2026-05-31T12:00:00+00:00"


def test_resolve_annotation_record_selection_applies_pending_override_before_widget_render() -> None:
    selector_state_key = "annotation_record_test_2"
    human_label = "Human | workspace-1/annotator-a | 2026-05-31T13:00:00+00:00"
    llm_label = "LLM | openrouter/model | prompt-1 | valid | 2026-05-31T12:00:00+00:00"
    session_state = {
        selector_state_key: llm_label,
        _pending_selector_override_key(selector_state_key): human_label,
    }

    _resolve_annotation_record_selection(
        session_state=session_state,
        selector_state_key=selector_state_key,
        selector_options={
            human_label: {"record_type": "human"},
            llm_label: {"record_type": "llm"},
        },
        default_record_label=llm_label,
    )

    assert session_state[selector_state_key] == human_label
    assert _pending_selector_override_key(selector_state_key) not in session_state


def test_llm_job_summary_rows_include_status_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    prompt = _ensure_project_default_llm_prompt(db_path, "Toy Project")
    job = create_llm_job(
        db_path=db_path,
        document_ids=["doc-1"],
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-default",
        request_params={"temperature": 0.0, "max_output_tokens": 800},
        created_by="annotator-a",
    )
    save_llm_run(
        db_path=db_path,
        job_id=job["job_id"],
        document_id="doc-1",
        prompt_id=prompt["prompt_id"],
        provider_id="fake",
        model_id="fake-default",
        prompt_template_text=prompt["template_text"],
        rendered_prompt_text="Prompt",
        request_params={"temperature": 0.0, "max_output_tokens": 800},
        raw_response='{"relationships":[]}',
        normalized_payload={"network_data": []},
        validation_status="valid",
        validation_errors=[],
        usage={},
        cost={},
    )
    from corn.llm_storage import set_job_document_status

    set_job_document_status(db_path, job["job_id"], "doc-1", "succeeded")

    rows = _llm_job_summary_rows(db_path)

    assert rows[0]["job_id"] == job["job_id"]
    assert rows[0]["status"] == "queued"
    assert rows[0]["worker_id"] is None
    assert rows[0]["succeeded"] == 1
    assert rows[0]["cancelled"] == 0
    assert rows[0]["total"] == 1


def test_launch_llm_annotation_job_processes_current_document_batch(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "annotations.sqlite"
    runtime_db_path = tmp_path / "runtime.sqlite"
    prompt = _ensure_project_default_llm_prompt(db_path, "Toy Project")

    run_prompt, summary = _launch_llm_annotation_job(
        db_path=db_path,
        runtime_db_path=runtime_db_path,
        project_name="Toy Project",
        tie_schema=[{"label": "support", "description": "Helpful relationship"}],
        document_ids=["1"],
        selected_prompt_id=prompt["prompt_id"],
        template_text=prompt["template_text"],
        created_by="annotator-a",
        prompt_label=prompt["label"],
        provider_id="fake",
        model_id="fake-default",
        temperature=0.0,
        max_output_tokens=800,
    )

    assert run_prompt["template_text"] == prompt["template_text"].strip()
    assert summary == {
        "total": 1,
        "pending": 1,
        "running": 0,
        "succeeded": 0,
        "invalid": 0,
        "failed": 0,
        "cancelled": 0,
    }


def test_duplicate_llm_payload_for_human_editing_rescopes_annotation() -> None:
    duplicated = _duplicate_llm_payload_for_human_editing(
        {
            "workspace_id": "llm",
            "annotator_id": "fake/fake-default",
            "annotation_source": "llm",
            "document_id": "doc-1",
            "created_at": "2026-05-31T10:00:00+00:00",
            "updated_at": "2026-05-31T10:00:00+00:00",
            "network_data": [],
        },
        workspace_id="workspace-1",
        annotator_id="annotator-a",
    )

    assert duplicated["workspace_id"] == "workspace-1"
    assert duplicated["annotator_id"] == "annotator-a"
    assert duplicated["annotation_source"] == "human_from_llm"
    assert duplicated["document_id"] == "doc-1"
    assert duplicated["created_at"] == duplicated["updated_at"]


def test_annotation_editor_stays_closed_until_user_opens_it() -> None:
    assert not _should_open_annotation_editor(False, 1, 1)


def test_annotation_editor_closes_when_user_switches_documents() -> None:
    assert not _should_open_annotation_editor(True, 1, 2)


def test_query_results_do_not_show_before_user_runs_query() -> None:
    assert not _should_show_query_results(False, [{"id": 1}])


def test_query_results_show_after_user_runs_query() -> None:
    assert _should_show_query_results(True, [{"id": 1}])


def test_normalize_loaded_annotation_keeps_empty_network_data_for_new_document() -> None:
    payload = {
        "workspace_id": "workspace-1",
        "annotator_id": "annotator-a",
        "document_id": "doc-1",
        "created_at": "2026-05-27T10:00:00+00:00",
        "updated_at": "2026-05-27T10:00:00+00:00",
        "network_data": [],
    }

    normalized = _normalize_loaded_annotation(payload, "Policy")

    assert normalized["network_data"] == []


def test_focus_request_builds_scroll_target_payload() -> None:
    assert _focus_request(document_id=7, start=12, end=24, nonce=3) == {
        "document_id": 7,
        "start": 12,
        "end": 24,
        "nonce": "7:12:24:3",
    }


def test_document_focus_request_is_returned_only_for_current_document() -> None:
    request = {
        "document_id": 7,
        "start": 12,
        "end": 24,
        "nonce": "7:12:24:3",
    }

    assert _document_focus_request_for_document(request, current_document_id=7) == request
    assert _document_focus_request_for_document(request, current_document_id=8) is None


def test_copied_tie_reuses_existing_reasoning_span() -> None:
    existing_tie = {
        "type": "Policy",
        "reasoning_text": "shared policy agenda",
        "reasoning_start": 12,
        "reasoning_end": 32,
        "selected_match": 0,
        "attributes": {"valence": "positive"},
    }

    assert _copied_tie(existing_tie, default_type="Coordination") == {
        "type": "Coordination",
        "reasoning_text": "shared policy agenda",
        "reasoning_text_span": "shared policy agenda",
        "reasoning_text_note": "",
        "reasoning_start": 12,
        "reasoning_end": 32,
        "selected_match": 0,
        "attributes": {},
    }


def test_build_save_payload_keeps_existing_exact_span_for_duplicate_text() -> None:
    document = {"link": "doc-1"}
    content = "Alpha supports Beta. Beta responds."
    annotation_state = {
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "Beta",
                        "reasoning_text_span": "Beta",
                        "reasoning_text_note": "",
                        "reasoning_start": 21,
                        "reasoning_end": 25,
                        "selected_match": None,
                    }
                ],
            }
        ]
    }

    payload = _build_save_payload(
        document,
        annotation_state,
        content,
        annotator_id="annotator-a",
        workspace_id="workspace-1",
        created_at="2026-05-27T10:00:00+00:00",
        updated_at="2026-05-27T11:00:00+00:00",
    )

    assert payload["annotator_id"] == "annotator-a"
    assert payload["workspace_id"] == "workspace-1"
    assert payload["annotation_source"] == "human"
    assert payload["created_at"] == "2026-05-27T10:00:00+00:00"
    assert payload["updated_at"] == "2026-05-27T11:00:00+00:00"
    assert payload["network_data"][0]["ties"][0] == {
        "type": "Policy",
        "reasoning_text": "Beta",
        "reasoning_text_span": "Beta",
        "reasoning_text_note": "",
        "reasoning_start": 21,
        "reasoning_end": 25,
    }


def test_build_save_payload_allows_note_only_reasoning_text() -> None:
    document = {"link": "doc-1"}
    annotation_state = {
        "annotation_source": "human_from_llm",
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text_span": "",
                        "reasoning_text_note": "LLM summary evidence",
                        "reasoning_start": None,
                        "reasoning_end": None,
                        "selected_match": None,
                    }
                ],
            }
        ],
    }

    payload = _build_save_payload(
        document,
        annotation_state,
        "Alpha supports Beta. Beta responds.",
        annotator_id="annotator-a",
        workspace_id="workspace-1",
        created_at="2026-05-27T10:00:00+00:00",
        updated_at="2026-05-27T11:00:00+00:00",
    )

    assert payload["annotation_source"] == "human_from_llm"
    assert payload["network_data"][0]["ties"][0] == {
        "type": "Policy",
        "reasoning_text": "LLM summary evidence",
        "reasoning_text_span": "",
        "reasoning_text_note": "LLM summary evidence",
        "reasoning_start": None,
        "reasoning_end": None,
    }


def test_blank_tie_includes_empty_attributes() -> None:
    assert _blank_tie("Policy")["attributes"] == {}


def test_unset_attribute_label_constant() -> None:
    assert app.UNSET_ATTRIBUTE_LABEL == "(unset)"


class _FakeAttributeStreamlit:
    """Fake st module for _render_tie_attribute_input.

    Widgets echo back the current value unless a canned response is
    registered for their widget key, simulating user input.
    """

    def __init__(self, responses: dict | None = None) -> None:
        self.responses = responses or {}
        self.warnings: list[str] = []
        self.widget_keys: list[str] = []
        self.rerun_count = 0

    def selectbox(self, _label, options, index=0, **kwargs):
        self.widget_keys.append(kwargs.get("key"))
        return self.responses.get(kwargs.get("key"), options[index])

    def text_input(self, _label, value="", **kwargs):
        self.widget_keys.append(kwargs.get("key"))
        return self.responses.get(kwargs.get("key"), value)

    def button(self, _label, **kwargs):
        self.widget_keys.append(kwargs.get("key"))
        return self.responses.get(kwargs.get("key"), False)

    def rerun(self):
        self.rerun_count += 1

    def warning(self, message, *_args, **_kwargs):
        self.warnings.append(str(message))


_CATEGORICAL_ATTRIBUTE = {
    "name": "valence",
    "type": "categorical",
    "options": ["positive", "negative"],
    "description": "Tone of the tie",
}
_NUMERIC_ATTRIBUTE = {
    "name": "certainty",
    "type": "numeric",
    "range": [0.0, 1.0],
    "description": "Annotator confidence",
}
_TEMPORAL_ATTRIBUTE = {
    "name": "period",
    "type": "temporal",
    "description": "When the tie was active",
}


def _attribute_input_case(attribute: dict, tie_attributes: dict, responses: dict | None = None, monkeypatch=None):
    fake_st = _FakeAttributeStreamlit(responses)
    monkeypatch.setattr(app, "st", fake_st)
    tie = {"attributes": tie_attributes}
    _render_tie_attribute_input(tie, attribute, "entry_0_tie_0")
    return tie, fake_st


def test_attribute_input_categorical_unset_stays_absent(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(_CATEGORICAL_ATTRIBUTE, {}, monkeypatch=monkeypatch)
    assert tie["attributes"] == {}
    assert fake_st.warnings == []


def test_attribute_input_categorical_selection_stored_and_unset_pops(monkeypatch) -> None:
    tie, _ = _attribute_input_case(
        _CATEGORICAL_ATTRIBUTE,
        {},
        responses={"entry_0_tie_0_attr_valence": "negative"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"valence": "negative"}

    tie, _ = _attribute_input_case(
        _CATEGORICAL_ATTRIBUTE,
        {"valence": "positive"},
        responses={"entry_0_tie_0_attr_valence": app.UNSET_ATTRIBUTE_LABEL},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {}


def test_attribute_input_categorical_stale_value_survives_render(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _CATEGORICAL_ATTRIBUTE, {"valence": "neutral"}, monkeypatch=monkeypatch
    )
    assert tie["attributes"] == {"valence": "neutral"}
    assert fake_st.warnings == [
        "valence value 'neutral' is not in the current schema options"
    ]


def test_attribute_input_categorical_valid_selection_replaces_stale_value(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _CATEGORICAL_ATTRIBUTE,
        {"valence": "neutral"},
        responses={"entry_0_tie_0_attr_valence": "positive"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"valence": "positive"}
    assert fake_st.warnings == [
        "valence value 'neutral' is not in the current schema options"
    ]


def test_attribute_input_categorical_existing_value_round_trips(monkeypatch) -> None:
    tie, _ = _attribute_input_case(
        _CATEGORICAL_ATTRIBUTE, {"valence": "positive"}, monkeypatch=monkeypatch
    )
    assert tie["attributes"] == {"valence": "positive"}


def test_attribute_input_numeric_valid_stored_as_float(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _NUMERIC_ATTRIBUTE,
        {},
        responses={"entry_0_tie_0_attr_certainty": " 0.7 "},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"certainty": 0.7}
    assert fake_st.warnings == []


def test_attribute_input_numeric_blank_pops_existing(monkeypatch) -> None:
    tie, _ = _attribute_input_case(
        _NUMERIC_ATTRIBUTE,
        {"certainty": 0.7},
        responses={"entry_0_tie_0_attr_certainty": "  "},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {}


def test_attribute_input_numeric_invalid_warns_and_leaves_state(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _NUMERIC_ATTRIBUTE,
        {"certainty": 0.7},
        responses={"entry_0_tie_0_attr_certainty": "high"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"certainty": 0.7}
    assert fake_st.warnings == ["Tie attribute certainty must be a number"]


def test_attribute_input_numeric_out_of_range_warns_and_leaves_state(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _NUMERIC_ATTRIBUTE,
        {},
        responses={"entry_0_tie_0_attr_certainty": "1.5"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {}
    assert fake_st.warnings == ["Tie attribute certainty must be between 0.0 and 1.0"]


def test_attribute_input_temporal_start_only(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _TEMPORAL_ATTRIBUTE,
        {},
        responses={"entry_0_tie_0_attr_period__start": "2019-07"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"period": {"start": "2019-07"}}
    assert fake_st.warnings == []


def test_attribute_input_temporal_start_and_end(monkeypatch) -> None:
    tie, _ = _attribute_input_case(
        _TEMPORAL_ATTRIBUTE,
        {"period": {"start": "2019-07"}},
        responses={"entry_0_tie_0_attr_period__end": "2020-01-15"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"period": {"start": "2019-07", "end": "2020-01-15"}}


def test_attribute_input_temporal_blank_start_pops_existing(monkeypatch) -> None:
    tie, _ = _attribute_input_case(
        _TEMPORAL_ATTRIBUTE,
        {"period": {"start": "2019-07", "end": "2020-01"}},
        responses={"entry_0_tie_0_attr_period__start": ""},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {}


def test_attribute_input_temporal_valid_start_malformed_end_warns_and_leaves_state(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _TEMPORAL_ATTRIBUTE,
        {"period": {"start": "2019-07"}},
        responses={"entry_0_tie_0_attr_period__end": "13/2020"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"period": {"start": "2019-07"}}
    assert fake_st.warnings == [
        "Tie attribute period dates must be ISO-partial strings: YYYY, YYYY-MM, or YYYY-MM-DD"
    ]


def test_attribute_input_temporal_malformed_warns_and_leaves_state(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _TEMPORAL_ATTRIBUTE,
        {"period": {"start": "2019-07"}},
        responses={"entry_0_tie_0_attr_period__start": "July 2019"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"period": {"start": "2019-07"}}
    assert fake_st.warnings == [
        "Tie attribute period dates must be ISO-partial strings: YYYY, YYYY-MM, or YYYY-MM-DD"
    ]


def test_attribute_input_temporal_invalid_month_warns_and_leaves_state(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _TEMPORAL_ATTRIBUTE,
        {"period": {"start": "2019-07"}},
        responses={"entry_0_tie_0_attr_period__start": "2020-13"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"period": {"start": "2019-07"}}
    assert fake_st.warnings == [
        "Tie attribute period dates must use a month between 01 and 12"
    ]


def test_attribute_input_temporal_invalid_calendar_date_warns_and_leaves_state(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _TEMPORAL_ATTRIBUTE,
        {},
        responses={"entry_0_tie_0_attr_period__start": "2020-02-31"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {}
    assert fake_st.warnings == [
        "Tie attribute period dates must be valid calendar dates"
    ]


def test_attribute_input_temporal_end_before_start_warns_and_leaves_state(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _TEMPORAL_ATTRIBUTE,
        {"period": {"start": "2019-07"}},
        responses={"entry_0_tie_0_attr_period__end": "2018-01"},
        monkeypatch=monkeypatch,
    )
    assert tie["attributes"] == {"period": {"start": "2019-07"}}
    assert fake_st.warnings == ["Tie attribute period end must not precede start"]


def test_attribute_input_widget_keys_do_not_collide_across_sibling_names(monkeypatch) -> None:
    temporal_x = {"name": "x", "type": "temporal", "description": "d"}
    categorical_x_start = {
        "name": "x_start",
        "type": "categorical",
        "options": ["a"],
        "description": "d",
    }
    fake_st = _FakeAttributeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    tie = {"attributes": {}}

    _render_tie_attribute_input(tie, temporal_x, "entry_0_tie_0")
    _render_tie_attribute_input(tie, categorical_x_start, "entry_0_tie_0")

    assert len(fake_st.widget_keys) == len(set(fake_st.widget_keys))


def test_attribute_input_temporal_non_dict_stored_value_renders_with_warning(monkeypatch) -> None:
    tie, fake_st = _attribute_input_case(
        _TEMPORAL_ATTRIBUTE, {"period": "2019"}, monkeypatch=monkeypatch
    )
    assert fake_st.warnings == [
        "period has an invalid stored value; set a new period or leave unset to discard it"
    ]
    # Echo render leaves the inputs blank, so the invalid value is discarded.
    assert tie["attributes"] == {}


def test_unknown_tie_attributes_warn_without_removal(monkeypatch) -> None:
    fake_st = _FakeAttributeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    tie = {"attributes": {"valence": "positive", "zeta": 1, "alpha": "x"}}

    _render_unknown_tie_attributes(tie, [_CATEGORICAL_ATTRIBUTE], "entry_0_tie_0")

    assert fake_st.warnings == [
        "Attributes not in the current schema: alpha, zeta (saving is blocked while present)"
    ]
    assert tie["attributes"] == {"valence": "positive", "zeta": 1, "alpha": "x"}
    assert fake_st.rerun_count == 0


def test_unknown_tie_attributes_removed_via_button(monkeypatch) -> None:
    fake_st = _FakeAttributeStreamlit(
        responses={"entry_0_tie_0_remove_unknown_attrs": True}
    )
    monkeypatch.setattr(app, "st", fake_st)
    tie = {"attributes": {"valence": "positive", "zeta": 1, "alpha": "x"}}

    _render_unknown_tie_attributes(tie, [_CATEGORICAL_ATTRIBUTE], "entry_0_tie_0")

    assert tie["attributes"] == {"valence": "positive"}
    assert fake_st.rerun_count == 1


def test_unknown_tie_attributes_silent_when_all_names_known(monkeypatch) -> None:
    fake_st = _FakeAttributeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    tie = {"attributes": {"valence": "positive"}}

    _render_unknown_tie_attributes(tie, [_CATEGORICAL_ATTRIBUTE], "entry_0_tie_0")

    assert fake_st.warnings == []
    assert fake_st.widget_keys == []


def test_normalize_loaded_annotation_preserves_tie_attributes() -> None:
    loaded_period = {"start": "2019-07"}
    payload = _attribute_annotation_payload(
        {"valence": "positive", "certainty": 0.7, "period": loaded_period}
    )

    normalized = _normalize_loaded_annotation(payload, "Policy")

    normalized_attributes = normalized["network_data"][0]["ties"][0]["attributes"]
    assert normalized_attributes == {
        "valence": "positive",
        "certainty": 0.7,
        "period": {"start": "2019-07"},
    }
    # Nested dicts must be copies so in-place editor mutations never alias loaded payloads.
    assert normalized_attributes["period"] is not loaded_period


def test_normalize_loaded_annotation_resets_non_dict_attributes() -> None:
    payload = _attribute_annotation_payload("positive")

    normalized = _normalize_loaded_annotation(payload, "Policy")

    assert normalized["network_data"][0]["ties"][0]["attributes"] == {}


def test_normalize_loaded_annotation_defaults_missing_attributes_to_empty_dict() -> None:
    payload = _attribute_annotation_payload(None)
    del payload["network_data"][0]["ties"][0]["attributes"]

    normalized = _normalize_loaded_annotation(payload, "Policy")

    assert normalized["network_data"][0]["ties"][0]["attributes"] == {}


def test_normalize_loaded_annotation_normalizes_attribute_values_against_schema() -> None:
    payload = _attribute_annotation_payload({"certainty": 1, "valence": None})

    normalized = _normalize_loaded_annotation(
        payload, "Policy", tie_attribute_schema=_TIE_ATTRIBUTE_SCHEMA
    )

    attributes = normalized["network_data"][0]["ties"][0]["attributes"]
    # The baseline must already match what the editor render writes back:
    # numeric values become floats and null values are dropped, so verbatim
    # imports do not show up as pending saves with zero user edits.
    assert attributes == {"certainty": 1.0}
    assert isinstance(attributes["certainty"], float)


def test_normalize_loaded_annotation_keeps_invalid_and_unknown_attribute_values() -> None:
    payload = _attribute_annotation_payload({"valence": "sideways", "vibe": "chaotic"})

    normalized = _normalize_loaded_annotation(
        payload, "Policy", tie_attribute_schema=_TIE_ATTRIBUTE_SCHEMA
    )

    # Stale/invalid and unknown-name values stay untouched so the editor can
    # surface them and the save gate can report them.
    assert normalized["network_data"][0]["ties"][0]["attributes"] == {
        "valence": "sideways",
        "vibe": "chaotic",
    }


def _attribute_annotation_payload(attributes) -> dict:
    return {
        "workspace_id": "workspace-1",
        "annotator_id": "annotator-a",
        "document_id": "doc-1",
        "created_at": "2026-05-27T10:00:00+00:00",
        "updated_at": "2026-05-27T10:00:00+00:00",
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "shared policy agenda",
                        "reasoning_text_span": "shared policy agenda",
                        "reasoning_text_note": "",
                        "reasoning_start": 6,
                        "reasoning_end": 26,
                        "attributes": attributes,
                    }
                ],
            }
        ],
    }


_TIE_ATTRIBUTE_SCHEMA = [
    {
        "name": "valence",
        "type": "categorical",
        "description": "d",
        "options": ["positive", "negative"],
    },
    {"name": "certainty", "type": "numeric", "description": "d", "range": [0.0, 1.0]},
]


def _attribute_annotation_state(attributes) -> dict:
    return {
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "shared policy agenda",
                        "reasoning_text_span": "shared policy agenda",
                        "reasoning_text_note": "",
                        "reasoning_start": 6,
                        "reasoning_end": 26,
                        "selected_match": None,
                        "attributes": attributes,
                    }
                ],
            }
        ]
    }


def test_build_save_payload_normalizes_valid_tie_attributes() -> None:
    annotation_state = _attribute_annotation_state({"valence": "positive", "certainty": "0.7"})

    payload = _build_save_payload(
        {"link": "doc-1"},
        annotation_state,
        "Alpha shared policy agenda Beta",
        annotator_id="annotator-a",
        workspace_id="workspace-1",
        created_at="2026-05-27T10:00:00+00:00",
        updated_at="2026-05-27T11:00:00+00:00",
        tie_attribute_schema=_TIE_ATTRIBUTE_SCHEMA,
    )

    assert payload["network_data"][0]["ties"][0]["attributes"] == {
        "valence": "positive",
        "certainty": 0.7,
    }


def test_build_save_payload_omits_attributes_key_when_all_unset() -> None:
    annotation_state = _attribute_annotation_state({"valence": None, "certainty": None})

    payload = _build_save_payload(
        {"link": "doc-1"},
        annotation_state,
        "Alpha shared policy agenda Beta",
        annotator_id="annotator-a",
        workspace_id="workspace-1",
        created_at="2026-05-27T10:00:00+00:00",
        updated_at="2026-05-27T11:00:00+00:00",
        tie_attribute_schema=_TIE_ATTRIBUTE_SCHEMA,
    )

    assert "attributes" not in payload["network_data"][0]["ties"][0]


def test_build_save_payload_raises_incomplete_error_for_invalid_attribute_value() -> None:
    annotation_state = _attribute_annotation_state({"valence": "neutral"})

    with pytest.raises(IncompleteAnnotationError, match="Org A → Org B: Tie attribute valence"):
        _build_save_payload(
            {"link": "doc-1"},
            annotation_state,
            "Alpha shared policy agenda Beta",
            annotator_id="annotator-a",
            workspace_id="workspace-1",
            created_at="2026-05-27T10:00:00+00:00",
            updated_at="2026-05-27T11:00:00+00:00",
            tie_attribute_schema=_TIE_ATTRIBUTE_SCHEMA,
        )


def test_build_save_payload_raises_incomplete_error_for_unknown_attribute_name() -> None:
    annotation_state = _attribute_annotation_state({"legacy": "positive"})

    with pytest.raises(IncompleteAnnotationError, match="Unknown tie attributes: legacy"):
        _build_save_payload(
            {"link": "doc-1"},
            annotation_state,
            "Alpha shared policy agenda Beta",
            annotator_id="annotator-a",
            workspace_id="workspace-1",
            created_at="2026-05-27T10:00:00+00:00",
            updated_at="2026-05-27T11:00:00+00:00",
            tie_attribute_schema=_TIE_ATTRIBUTE_SCHEMA,
        )


def test_build_save_payload_without_schema_keeps_previous_shape() -> None:
    annotation_state = _attribute_annotation_state({})

    payload = _build_save_payload(
        {"link": "doc-1"},
        annotation_state,
        "Alpha shared policy agenda Beta",
        annotator_id="annotator-a",
        workspace_id="workspace-1",
        created_at="2026-05-27T10:00:00+00:00",
        updated_at="2026-05-27T11:00:00+00:00",
    )

    assert payload["network_data"][0]["ties"][0] == {
        "type": "Policy",
        "reasoning_text": "shared policy agenda",
        "reasoning_text_span": "shared policy agenda",
        "reasoning_text_note": "",
        "reasoning_start": 6,
        "reasoning_end": 26,
    }


def test_tie_needs_span_review_is_false_when_exact_span_exists_even_with_note() -> None:
    tie = {
        "type": "Policy",
        "reasoning_text": "Beta",
        "reasoning_text_span": "Beta",
        "reasoning_text_note": "LLM summary evidence",
        "reasoning_start": 21,
        "reasoning_end": 25,
    }

    assert not _tie_needs_span_review(tie, "Alpha supports Beta. Beta responds.")


def test_apply_pending_selection_to_target_tie_populates_span_fields() -> None:
    annotation_state = {
        "network_data": [
            {
                "source": "Org A",
                "target": "Org B",
                "ties": [
                    {
                        "type": "Policy",
                        "reasoning_text": "LLM summary evidence",
                        "reasoning_text_span": "",
                        "reasoning_text_note": "LLM summary evidence",
                        "reasoning_start": None,
                        "reasoning_end": None,
                        "selected_match": None,
                    }
                ],
            }
        ]
    }
    pending_selection = {
        "document_id": 7,
        "text": "Beta",
        "start": 21,
        "end": 25,
        "nonce": "sel-1",
    }
    pending_target = {
        "document_id": 7,
        "entry_index": 0,
        "tie_index": 0,
        "nonce": "target-1",
    }
    widget_state = {
        "entry_0_tie_0_reasoning_span": "",
    }

    changed = _apply_pending_selection_to_target_tie(
        annotation_state=annotation_state,
        pending_selection=pending_selection,
        pending_span_target=pending_target,
        session_state=widget_state,
        widget_key_prefix="entry_0_tie_0",
    )

    assert changed
    assert widget_state["entry_0_tie_0_reasoning_span"] == "Beta"
    assert annotation_state["network_data"][0]["ties"][0] == {
        "type": "Policy",
        "reasoning_text": "Beta",
        "reasoning_text_span": "Beta",
        "reasoning_text_note": "LLM summary evidence",
        "reasoning_start": 21,
        "reasoning_end": 25,
        "selected_match": 0,
    }


def test_pending_span_target_template_is_empty() -> None:
    assert _pending_span_target_template() == {
        "document_id": None,
        "entry_index": None,
        "tie_index": None,
        "nonce": None,
    }


def test_pending_create_entry_template_is_empty() -> None:
    assert _pending_create_entry_template() == {
        "document_id": None,
        "text": "",
        "start": None,
        "end": None,
        "nonce": None,
    }


def test_append_network_entry_from_selection_creates_blank_entry_with_selected_span() -> None:
    annotation_state = {"network_data": []}

    _append_network_entry_from_selection(
        annotation_state=annotation_state,
        selection={
            "text": "Beta",
            "start": 21,
            "end": 25,
        },
        source="Org A",
        target="Org B",
        tie_type="Policy",
    )

    assert annotation_state["network_data"] == [
        {
            "source": "Org A",
            "target": "Org B",
            "ties": [
                {
                    "type": "Policy",
                    "reasoning_text": "Beta",
                    "reasoning_text_span": "Beta",
                    "reasoning_text_note": "",
                    "reasoning_start": 21,
                    "reasoning_end": 25,
                    "selected_match": 0,
                    "attributes": {},
                }
            ],
        }
    ]


def test_annotation_relative_path_scopes_json_by_workspace_and_annotator() -> None:
    document = {"link": "https://example.com/doc-1"}

    assert _annotation_relative_path(document, "workspace-1", "annotator-a").as_posix() == (
        "workspace-1/annotator-a/https_example.com_doc-1.json"
    )


def test_requires_span_disambiguation_is_false_when_stored_span_is_valid() -> None:
    tie = {
        "type": "Policy",
        "reasoning_text": "Beta",
        "reasoning_start": 21,
        "reasoning_end": 25,
        "selected_match": None,
    }

    assert not _requires_span_disambiguation(tie, "Alpha supports Beta. Beta responds.")


def test_requires_span_disambiguation_is_true_when_duplicate_text_has_no_valid_span() -> None:
    tie = {
        "type": "Policy",
        "reasoning_text": "Beta",
        "reasoning_start": None,
        "reasoning_end": None,
        "selected_match": None,
    }

    assert _requires_span_disambiguation(tie, "Alpha supports Beta. Beta responds.")


def test_document_table_rows_includes_project_metadata_columns() -> None:
    rows = [
        {
            "id": 1,
            "title": None,
            "link": "doc-1",
            "document_id": "doc-1",
            "document_title": "",
            "time_period": "",
            "subreddit": "r/work",
            "person_a": "Priya",
            "person_b": "Tom",
            "sentiment": "positive",
        }
    ]

    table_rows = _document_table_rows(
        rows,
        document_config={
            "id_field": "id",
            "text_field": "comment",
            "title_field": "subreddit_title",
        },
        metadata_fields=["subreddit", "person_a", "person_b", "sentiment"],
    )

    assert table_rows == [
        {
            "id": "doc-1",
            "subreddit_title": "",
            "subreddit": "r/work",
            "person_a": "Priya",
            "person_b": "Tom",
            "sentiment": "positive",
        }
    ]


def test_document_table_rows_omits_title_column_when_project_has_no_title_field() -> None:
    rows = [
        {
            "id": 1,
            "link": "doc-1",
            "document_id": "doc-1",
            "topic": "policy",
        }
    ]

    table_rows = _document_table_rows(
        rows,
        document_config={
            "id_field": "doc_id",
            "text_field": "body",
        },
        metadata_fields=["topic"],
    )

    assert table_rows == [
        {
            "doc_id": "doc-1",
            "topic": "policy",
        }
    ]


def test_available_metadata_columns_excludes_document_fields() -> None:
    assert _available_metadata_columns(
        ["id", "body", "title", "topic", "sentiment"],
        id_field="id",
        text_field="body",
        title_field="title",
    ) == ["topic", "sentiment"]


def test_parse_tie_schema_text_reads_editable_lines() -> None:
    assert _parse_tie_schema_text("support|Helpful relationship\nconflict|Relationship tension") == [
        {"label": "support", "description": "Helpful relationship"},
        {"label": "conflict", "description": "Relationship tension"},
    ]


def test_guided_tie_attribute_cards_use_monotonic_stable_ids() -> None:
    state = {"tie_attributes": [], "next_tie_attribute_id": 0}

    _add_guided_tie_attribute(state)
    _add_guided_tie_attribute(state)
    _remove_guided_tie_attribute(state, 0)
    _add_guided_tie_attribute(state)

    assert [card["id"] for card in state["tie_attributes"]] == [1, 2]
    assert state["next_tie_attribute_id"] == 3
    assert state["tie_attributes"][1] == {
        "id": 2,
        "name": "",
        "description": "",
        "type": "categorical",
        "options_text": "",
        "minimum": 0.0,
        "maximum": 1.0,
    }


def test_guided_tie_attribute_schema_converts_type_specific_card_state() -> None:
    assert _guided_tie_attribute_schema(
        [
            {
                "id": 0,
                "name": "valence",
                "description": "Character of the tie",
                "type": "categorical",
                "options_text": "positive\nnegative\n",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            {
                "id": 1,
                "name": "strength",
                "description": "Intensity of the tie",
                "type": "numeric",
                "options_text": "",
                "minimum": 0.0,
                "maximum": 10.0,
            },
            {
                "id": 2,
                "name": "period",
                "description": "When the tie occurred",
                "type": "temporal",
                "options_text": "",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        ]
    ) == [
        {
            "name": "valence",
            "description": "Character of the tie",
            "type": "categorical",
            "options": ["positive", "negative"],
        },
        {
            "name": "strength",
            "description": "Intensity of the tie",
            "type": "numeric",
            "range": [0.0, 10.0],
        },
        {
            "name": "period",
            "description": "When the tie occurred",
            "type": "temporal",
        },
    ]


def test_guided_config_for_review_validates_tie_attribute_cards() -> None:
    state = {
        "project_name": "Toy",
        "project_slug": "toy",
        "source_origin": "server",
        "source_type": "csv",
        "source_mode": "copy",
        "server_source_path": "/tmp/toy.csv",
        "sqlite_table": "",
        "id_field": "id",
        "text_field": "body",
        "title_field": "",
        "metadata_fields": [],
        "filter_types_by_field": {},
        "tie_schema_text": "support|Helpful relationship",
        "tie_attributes": [
            {
                "id": 0,
                "name": "valence",
                "description": "Character of the tie",
                "type": "categorical",
                "options_text": "(unset)",
                "minimum": 0.0,
                "maximum": 1.0,
            }
        ],
    }

    with pytest.raises(ValueError, match="reserved label"):
        _guided_config_for_review(state)


def test_guided_review_summary_returns_human_readable_snapshot() -> None:
    summary = _guided_review_summary(
        {
            "project": {"name": "Toy", "slug": "toy"},
            "source": {"type": "csv", "mode": "copy", "path": "/tmp/toy.csv"},
            "document": {"id_field": "id", "text_field": "body"},
            "metadata_fields": [{"name": "topic", "source_field": "topic"}],
            "filters": [{"field": "topic", "label": "Topic", "type": "select"}],
            "tie_schema": [{"label": "support", "description": "Helpful relationship"}],
            "tie_attribute_schema": [
                {
                    "name": "valence",
                    "type": "categorical",
                    "description": "Character of the tie",
                    "options": ["positive", "negative"],
                }
            ],
        }
    )

    assert summary == {
        "project_name": "Toy",
        "project_slug": "toy",
        "source_type": "csv",
        "source_mode": "copy",
        "source_path": "/tmp/toy.csv",
        "document_fields": {"id_field": "id", "text_field": "body"},
        "metadata_fields": ["topic"],
        "filter_fields": ["topic"],
        "tie_schema_count": 1,
        "tie_attribute_schema_count": 1,
    }


def test_tie_schema_presets_from_project_configs_loads_data_dir_presets(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "toy-project.yaml").write_text(
        """
project:
  name: Toy Relationship Dataset
  slug: toy-relationships
source:
  type: csv
  mode: copy
  path: /tmp/toy.csv
document:
  id_field: id
  text_field: text
tie_schema:
  - label: support
    description: Helpful relationship
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "DATA_DIR", data_dir)
    monkeypatch.setattr(app, "REGISTRY_PATH", data_dir / "project_registry.yaml")

    presets = _tie_schema_presets_from_project_configs()

    assert presets == {
        "Toy Relationship Dataset": [
            {"label": "support", "description": "Helpful relationship"},
        ]
    }


def test_cleanup_guided_upload_removes_staged_file(tmp_path: Path) -> None:
    upload_path = tmp_path / "staged.sqlite"
    upload_path.write_bytes(b"sqlite")

    _cleanup_guided_upload({"uploaded_source_path": str(upload_path)})

    assert not upload_path.exists()


def test_annotation_scope_catalog_groups_annotators_by_workspace(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    save_annotation_record(
        db_path,
        {
            "workspace_id": "workspace-1",
            "annotator_id": "annotator-a",
            "document_id": "doc-1",
            "created_at": "2026-05-30T10:00:00+00:00",
            "updated_at": "2026-05-30T10:00:00+00:00",
            "network_data": [],
        },
        tmp_path / "workspace-1" / "annotator-a" / "doc-1.json",
    )
    save_annotation_record(
        db_path,
        {
            "workspace_id": "workspace-1",
            "annotator_id": "annotator-b",
            "document_id": "doc-2",
            "created_at": "2026-05-30T10:00:00+00:00",
            "updated_at": "2026-05-30T10:00:00+00:00",
            "network_data": [],
        },
        tmp_path / "workspace-1" / "annotator-b" / "doc-2.json",
    )

    catalog = _annotation_scope_catalog(db_path)

    assert catalog == {
        "workspaces": ["workspace-1"],
        "annotators": ["annotator-a", "annotator-b"],
        "annotators_by_workspace": {
            "workspace-1": ["annotator-a", "annotator-b"],
        },
    }


def test_annotation_scope_catalog_excludes_llm_managed_scopes(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite"
    save_annotation_record(
        db_path,
        {
            "workspace_id": "workspace-1",
            "annotator_id": "annotator-a",
            "document_id": "doc-1",
            "created_at": "2026-05-30T10:00:00+00:00",
            "updated_at": "2026-05-30T10:00:00+00:00",
            "annotation_source": "human",
            "network_data": [],
        },
        tmp_path / "workspace-1" / "annotator-a" / "doc-1.json",
    )
    save_annotation_record(
        db_path,
        {
            "workspace_id": "workspace-2",
            "annotator_id": "annotator-b",
            "document_id": "doc-2",
            "created_at": "2026-05-30T10:00:00+00:00",
            "updated_at": "2026-05-30T10:00:00+00:00",
            "annotation_source": "human_from_llm",
            "network_data": [],
        },
        tmp_path / "workspace-2" / "annotator-b" / "doc-2.json",
    )
    save_annotation_record(
        db_path,
        {
            "workspace_id": "llm",
            "annotator_id": "openrouter/claude-3.5-sonnet",
            "document_id": "doc-3",
            "created_at": "2026-05-30T10:00:00+00:00",
            "updated_at": "2026-05-30T10:00:00+00:00",
            "annotation_source": "llm",
            "network_data": [],
        },
        tmp_path / "llm" / "openrouter_claude-3.5-sonnet" / "doc-3.json",
    )

    catalog = _annotation_scope_catalog(db_path)

    assert catalog == {
        "workspaces": ["workspace-1", "workspace-2"],
        "annotators": ["annotator-a", "annotator-b"],
        "annotators_by_workspace": {
            "workspace-1": ["annotator-a"],
            "workspace-2": ["annotator-b"],
        },
    }


def test_scope_select_options_include_existing_or_new_entry_mode() -> None:
    options, index = _scope_select_options(
        ["workspace-1", "workspace-2"],
        "workspace-2",
        default_value="default",
    )

    assert options == ["workspace-1", "workspace-2", "Enter new..."]
    assert index == 1


def test_selected_scope_value_uses_typed_value_for_new_entry() -> None:
    assert _selected_scope_value("Enter new...", " workspace-3 ") == "workspace-3"
    assert _selected_scope_value("workspace-1", "ignored") == "workspace-1"


def test_human_reporting_scope_options_labels_workspace_and_annotator() -> None:
    options, index = _human_reporting_scope_options(
        [("workspace-1", "annotator-a"), ("workspace-2", "annotator-b")],
        selected_scope=("workspace-2", "annotator-b"),
    )

    assert options == {
        "workspace-1 / annotator-a": ("workspace-1", "annotator-a"),
        "workspace-2 / annotator-b": ("workspace-2", "annotator-b"),
    }
    assert index == 1


def test_llm_reporting_scope_options_labels_provider_model_and_document_count() -> None:
    options, index = _llm_reporting_scope_options(
        [
            {"provider_id": "ollama", "model_id": "llama3", "document_count": 3},
            {"provider_id": "openrouter", "model_id": "claude-3.5-sonnet", "document_count": 7},
        ],
        selected_scope=("openrouter", "claude-3.5-sonnet"),
    )

    assert options == {
        "ollama / llama3 (3 docs)": ("ollama", "llama3"),
        "openrouter / claude-3.5-sonnet (7 docs)": ("openrouter", "claude-3.5-sonnet"),
    }
    assert index == 1


def test_llm_reporting_direction_options_select_existing_mode() -> None:
    options, index = _llm_reporting_direction_options(selected_mode="undirected")

    assert options == {
        "Directed": "directed",
        "Undirected": "undirected",
    }
    assert index == 1


def test_llm_reporting_human_document_coverage_options_select_existing_mode() -> None:
    options, index = _llm_reporting_human_document_coverage_options(
        selected_mode="exclude_human_docs_without_valid_llm_runs"
    )

    assert options == {
        "Include all human annotated docs": "include_all_human_annotated_docs",
        "Exclude human docs without valid LLM runs": "exclude_human_docs_without_valid_llm_runs",
    }
    assert index == 1


def test_llm_reporting_empty_state_message_explains_missing_scope_side() -> None:
    assert _llm_reporting_empty_state_message([], []) == (
        "No human annotation scopes or valid LLM reporting scopes are available yet."
    )
    assert _llm_reporting_empty_state_message([("workspace-1", "annotator-a")], []) == (
        "No valid LLM reporting scopes are available yet."
    )
    assert _llm_reporting_empty_state_message([], [("provider", "model")]) == (
        "No human annotation scopes are available yet."
    )
    assert _llm_reporting_empty_state_message(
        [("workspace-1", "annotator-a")],
        [("provider", "model")],
    ) is None


def test_llm_reporting_summary_stats_reads_report_contract() -> None:
    report = {
        "overall": {
            "precision": 0.75,
            "recall": 0.6,
            "f1": 0.6667,
            "macro_f1": 0.55,
        },
        "document_coverage": {
            "human_documents": 8,
            "llm_documents": 7,
            "union_documents": 10,
        },
    }

    assert _llm_reporting_summary_stats(report) == {
        "Precision": 0.75,
        "Recall": 0.6,
        "F1": 0.6667,
        "Macro F1": 0.55,
        "Human docs": 8,
        "LLM docs": 7,
        "Union docs": 10,
    }


def test_llm_reporting_refresh_button_reruns_when_clicked(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def fake_button(label: str, **kwargs) -> bool:
        calls.append((label, kwargs))
        return True

    rerun_called = {"value": False}

    def fake_rerun() -> None:
        rerun_called["value"] = True

    fake_st = SimpleNamespace(
        button=fake_button,
        rerun=fake_rerun,
    )
    monkeypatch.setattr(app, "st", fake_st)

    _llm_reporting_refresh_button("project-slug")

    assert calls == [
        (
            "Refresh and recalculate",
            {
                "key": "refresh_llm_reporting_project-slug",
                "use_container_width": True,
            },
        )
    ]
    assert rerun_called["value"] is True


class _FakeAgreementStreamlit:
    """Fake st module recording calls made by the attribute-agreement panel."""

    def __init__(self) -> None:
        self.markdowns: list[str] = []
        self.metrics: list[tuple[str, object]] = []
        self.metric_helps: dict[str, object] = {}
        self.captions: list[str] = []
        self.dataframes: list[list[dict]] = []
        self.expander_labels: list[str] = []

    def markdown(self, text, **_kwargs) -> None:
        self.markdowns.append(text)

    def metric(self, label, value, **kwargs) -> None:
        self.metrics.append((label, value))
        self.metric_helps[label] = kwargs.get("help")

    def caption(self, text, **_kwargs) -> None:
        self.captions.append(text)

    def dataframe(self, data, **_kwargs) -> None:
        self.dataframes.append(data)

    def columns(self, count):
        return [self] * count

    def expander(self, label, **_kwargs):
        self.expander_labels.append(label)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def _agreement_metrics_case(name, metrics, monkeypatch):
    fake_st = _FakeAgreementStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    app._render_attribute_agreement_metrics(name, metrics)
    return fake_st


def test_format_agreement_metric_none_is_na() -> None:
    assert app._format_agreement_metric(None) == "n/a"
    assert app._format_agreement_metric(0.75) == "0.75"
    assert app._format_agreement_metric(0) == "0"


def test_attribute_agreement_categorical_renders_accuracy_and_matrix(monkeypatch) -> None:
    fake_st = _agreement_metrics_case(
        "valence",
        {
            "coverage": {"both": 2, "human_only": 1, "llm_only": 0, "neither": 3},
            "accuracy": 0.5,
            "confusion_matrix": {
                "negative": {"negative": 0, "neutral": 0, "positive": 0},
                "neutral": {"negative": 0, "neutral": 0, "positive": 1},
                "positive": {"negative": 0, "neutral": 0, "positive": 1},
            },
        },
        monkeypatch,
    )
    assert fake_st.markdowns == ["**valence**"]
    assert ("Both", 2) in fake_st.metrics
    assert ("Human only", 1) in fake_st.metrics
    assert ("LLM only", 0) in fake_st.metrics
    assert ("Neither", 3) in fake_st.metrics
    assert ("Accuracy", "0.5") in fake_st.metrics
    assert fake_st.dataframes == [
        [
            {"truth": "negative", "negative": 0, "neutral": 0, "positive": 0},
            {"truth": "neutral", "negative": 0, "neutral": 0, "positive": 1},
            {"truth": "positive", "negative": 0, "neutral": 0, "positive": 1},
        ]
    ]
    assert fake_st.captions == ["Confusion matrix (rows = human)"]


def test_attribute_agreement_categorical_none_accuracy_renders_na(monkeypatch) -> None:
    fake_st = _agreement_metrics_case(
        "valence",
        {
            "coverage": {"both": 0, "human_only": 0, "llm_only": 0, "neither": 1},
            "accuracy": None,
            "confusion_matrix": {"positive": {"positive": 0}},
        },
        monkeypatch,
    )
    assert ("Accuracy", "n/a") in fake_st.metrics


def test_attribute_agreement_numeric_none_renders_na(monkeypatch) -> None:
    fake_st = _agreement_metrics_case(
        "certainty",
        {
            "coverage": {"both": 0, "human_only": 0, "llm_only": 2, "neither": 0},
            "mae": None,
            "tolerance": 0.1,
            "within_tolerance": None,
        },
        monkeypatch,
    )
    assert ("MAE", "n/a") in fake_st.metrics
    assert ("Within ±0.1", "n/a") in fake_st.metrics
    assert fake_st.dataframes == []


def test_attribute_agreement_numeric_values_render_as_str(monkeypatch) -> None:
    fake_st = _agreement_metrics_case(
        "certainty",
        {
            "coverage": {"both": 3, "human_only": 0, "llm_only": 0, "neither": 0},
            "mae": 0.05,
            "tolerance": 0.1,
            "within_tolerance": 1.0,
        },
        monkeypatch,
    )
    assert ("MAE", "0.05") in fake_st.metrics
    assert ("Within ±0.1", "1.0") in fake_st.metrics
    assert fake_st.metric_helps["Within ±0.1"] == "Share of matched ties within tolerance"


def test_attribute_agreement_temporal_renders_granularity_metric(monkeypatch) -> None:
    fake_st = _agreement_metrics_case(
        "period",
        {
            "coverage": {"both": 4, "human_only": 0, "llm_only": 0, "neither": 0},
            "match_at_common_granularity": 0.75,
        },
        monkeypatch,
    )
    assert ("Match at common granularity", "0.75") in fake_st.metrics
    assert fake_st.dataframes == []


def test_attribute_agreement_panel_absent_section_renders_nothing(monkeypatch) -> None:
    fake_st = _FakeAgreementStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    app._render_attribute_agreement_panel({"labels": []})
    assert fake_st.expander_labels == []
    assert fake_st.markdowns == []
    assert fake_st.metrics == []


def test_attribute_agreement_panel_renders_expander_per_attribute(monkeypatch) -> None:
    fake_st = _FakeAgreementStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    app._render_attribute_agreement_panel(
        {
            "attribute_agreement": {
                "matched_ties": 5,
                "attributes": {
                    "period": {
                        "coverage": {"both": 1, "human_only": 0, "llm_only": 0, "neither": 4},
                        "match_at_common_granularity": 1.0,
                    },
                },
            }
        }
    )
    assert fake_st.expander_labels == ["Attribute agreement (over 5 matched ties)"]
    assert fake_st.markdowns == ["**period**"]
    assert ("Match at common granularity", "1.0") in fake_st.metrics
