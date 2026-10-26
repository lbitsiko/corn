from __future__ import annotations

import pytest

from corn.llm_prompts import (
    SUPPORTED_PROMPT_PLACEHOLDERS,
    build_prompt_context,
    render_prompt_template,
)


def test_build_prompt_context_returns_stable_generic_placeholders() -> None:
    context = build_prompt_context(
        project_name="Toy Project",
        document={
            "link": "doc-1",
            "title": "Document title",
            "content": "Document body",
            "topic": "policy",
        },
        tie_schema=[
            {"label": "support", "description": "Helpful relationship"},
        ],
    )

    assert set(context) == set(SUPPORTED_PROMPT_PLACEHOLDERS)
    assert context["{{project_name}}"] == "Toy Project"
    assert context["{{document_id}}"] == "doc-1"
    assert context["{{document_title}}"] == "Document title"
    assert context["{{document_text}}"] == "Document body"


def test_render_prompt_template_substitutes_supported_placeholders() -> None:
    prompt = render_prompt_template(
        "Project: {{project_name}}\nTitle: {{document_title}}\nText: {{document_text}}",
        {
            "{{project_name}}": "Toy Project",
            "{{document_title}}": "Title",
            "{{document_text}}": "Body",
        },
    )

    assert prompt == "Project: Toy Project\nTitle: Title\nText: Body"


def test_render_prompt_template_handles_missing_optional_title() -> None:
    prompt = render_prompt_template(
        "Title: {{document_title}}\nText: {{document_text}}",
        {
            "{{document_title}}": "",
            "{{document_text}}": "Body",
        },
    )

    assert prompt == "Title: \nText: Body"


def test_context_includes_tie_attribute_schema_json() -> None:
    context = build_prompt_context(
        project_name="P",
        document={"document_id": "1", "content": "text"},
        tie_schema=[{"label": "interaction", "description": "d"}],
        tie_attribute_schema=[
            {
                "name": "valence",
                "type": "categorical",
                "description": "d",
                "options": ["positive", "negative"],
            }
        ],
    )

    assert "valence" in context["{{tie_attribute_schema_json}}"]
    assert "valence" in context["{{annotation_schema_json}}"]
    assert '\\"positive\\" | \\"negative\\"' in context["{{annotation_schema_json}}"]


def test_annotation_schema_json_has_no_attributes_without_schema() -> None:
    context = build_prompt_context(
        project_name="P",
        document={"document_id": "1", "content": "text"},
        tie_schema=[],
    )

    assert "attributes" not in context["{{annotation_schema_json}}"]
    assert context["{{tie_attribute_schema_json}}"] == "[]"


def test_annotation_schema_json_describes_numeric_and_temporal_attributes() -> None:
    context = build_prompt_context(
        project_name="P",
        document={"document_id": "1", "content": "text"},
        tie_schema=[{"label": "interaction", "description": "d"}],
        tie_attribute_schema=[
            {"name": "certainty", "type": "numeric", "description": "d", "range": [0.0, 1.0]},
            {"name": "period", "type": "temporal", "description": "d"},
        ],
    )

    assert "number between 0.0 and 1.0" in context["{{annotation_schema_json}}"]
    assert "YYYY" in context["{{annotation_schema_json}}"]
    assert "(optional)" not in context["{{annotation_schema_json}}"]


def test_render_prompt_template_accepts_tie_attribute_schema_placeholder() -> None:
    context = build_prompt_context(
        project_name="P",
        document={"document_id": "1", "content": "text"},
        tie_schema=[],
    )

    prompt = render_prompt_template(
        "Attributes:\n{{tie_attribute_schema_json}}",
        context,
    )

    assert prompt == "Attributes:\n[]"


def test_render_prompt_template_rejects_unsupported_placeholders() -> None:
    with pytest.raises(ValueError, match="Unsupported prompt placeholders"):
        render_prompt_template(
            "Text: {{document_text}}\nField: {{sentiment}}",
            {
                "{{document_text}}": "Body",
            },
        )
