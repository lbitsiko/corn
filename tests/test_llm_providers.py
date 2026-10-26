from __future__ import annotations

import json

import pytest

from corn.llm_providers import (
    FakeLLMProvider,
    GeminiProvider,
    OllamaProvider,
    OpenRouterProvider,
    ProviderExecutionError,
    get_provider,
    get_provider_config,
    list_available_models_for_provider,
    list_models_for_provider,
    list_provider_ids,
    provider_registry,
)


def test_provider_registry_exposes_builtin_fake_provider() -> None:
    registry = provider_registry()

    assert "fake" in registry
    assert "gemini" in registry
    assert "ollama" in registry
    assert "openrouter" in registry
    assert get_provider("fake").provider_id == "fake"


def test_provider_registry_exposes_provider_and_model_options() -> None:
    assert list_provider_ids() == ["fake", "gemini", "ollama", "openrouter"]
    assert list_models_for_provider("fake") == ["fake-default", "fake-verbose"]
    assert "gemini-2.5-flash" in list_models_for_provider("gemini")
    assert "llama3.2" in list_models_for_provider("ollama")
    assert "openai/gpt-4.1-mini" in list_models_for_provider("openrouter")


def test_fake_provider_returns_deterministic_result_payload() -> None:
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
        }
    )

    result = provider.annotate(
        rendered_prompt_text="Annotate doc-1",
        document_id="doc-1",
        model_id="fake-model",
        request_params={"temperature": 0.1, "max_output_tokens": 400},
    )

    assert result["raw_response"] == (
        '{"relationships": [{"source": "Org A", "target": "Org B", "ties": '
        '[{"type": "support", "reasoning_text": "shared agenda"}]}]}'
    )
    assert result["parsed_output"]["relationships"][0]["source"] == "Org A"


def test_fake_provider_can_raise_provider_execution_error() -> None:
    provider = FakeLLMProvider(errors_by_document_id={"doc-2": "rate limited"})

    with pytest.raises(ProviderExecutionError, match="rate limited"):
        provider.annotate(
            rendered_prompt_text="Annotate doc-2",
            document_id="doc-2",
            model_id="fake-model",
            request_params={"temperature": 0.1, "max_output_tokens": 400},
        )


def test_openrouter_config_requires_api_key() -> None:
    config = get_provider_config("openrouter", environ={})

    assert config.base_url == "https://openrouter.ai/api/v1"
    assert not config.configured
    assert config.requires_api_key


def test_gemini_config_requires_api_key() -> None:
    config = get_provider_config("gemini", environ={})

    assert config.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert not config.configured
    assert config.requires_api_key
    assert config.env_vars == ("GEMINI_API_KEY", "GEMINI_BASE_URL")


def test_gemini_provider_parses_structured_interaction_json() -> None:
    captured: dict[str, object] = {}

    def fake_transport(url: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "id": "v1_test",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "relationships": [
                                        {
                                            "source": "Org A",
                                            "target": "Org B",
                                            "ties": [
                                                {
                                                    "type": "support",
                                                    "reasoning_text": "shared agenda",
                                                }
                                            ],
                                        }
                                    ]
                                }
                            ),
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 18,
                "total_tokens": 30,
            },
        }

    provider = GeminiProvider(
        config=get_provider_config("gemini", environ={"GEMINI_API_KEY": "test-key"}),
        transport=fake_transport,
    )

    result = provider.annotate(
        rendered_prompt_text="Annotate this document",
        document_id="doc-1",
        model_id="gemini-2.5-flash",
        request_params={"temperature": 0.2, "max_output_tokens": 500},
    )

    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "x-goog-api-key": "test-key",
    }
    payload = captured["payload"]
    assert payload["model"] == "gemini-2.5-flash"
    assert payload["input"] == "Annotate this document"
    assert payload["store"] is False
    assert payload["generation_config"] == {
        "temperature": 0.2,
        "max_output_tokens": 500,
    }
    response_format = payload["response_format"]
    assert response_format["type"] == "text"
    assert response_format["mime_type"] == "application/json"
    assert response_format["schema"]["required"] == ["relationships"]
    relationship_schema = response_format["schema"]["properties"]["relationships"]["items"]
    assert relationship_schema["required"] == ["source", "target", "ties"]
    assert relationship_schema["properties"]["ties"]["items"]["required"] == [
        "type",
        "reasoning_text",
    ]
    assert result["parsed_output"]["relationships"][0]["target"] == "Org B"
    assert result["usage"] == {
        "input_tokens": 12,
        "output_tokens": 18,
        "total_tokens": 30,
    }


def test_gemini_provider_requires_api_key_before_request() -> None:
    provider = GeminiProvider(
        config=get_provider_config("gemini", environ={}),
        transport=lambda _url, _headers, _payload: {},
    )

    with pytest.raises(ProviderExecutionError, match="GEMINI_API_KEY"):
        provider.annotate(
            rendered_prompt_text="Annotate this document",
            document_id="doc-1",
            model_id="gemini-2.5-flash",
            request_params={"temperature": 0.2, "max_output_tokens": 500},
        )


def test_gemini_provider_attaches_raw_response_when_candidate_text_is_missing() -> None:
    provider = GeminiProvider(
        config=get_provider_config("gemini", environ={"GEMINI_API_KEY": "test-key"}),
        transport=lambda _url, _headers, _payload: {
            "id": "v1_test",
            "status": "failed",
            "steps": [],
            "error": {"message": "Response blocked by safety policy"},
        },
    )

    with pytest.raises(ProviderExecutionError, match="candidate text") as exc_info:
        provider.annotate(
            rendered_prompt_text="Annotate this document",
            document_id="doc-1",
            model_id="gemini-2.5-flash",
            request_params={"temperature": 0.2, "max_output_tokens": 500},
        )

    assert "Response blocked by safety policy" in exc_info.value.raw_response


def test_ollama_config_defaults_to_local_without_auth() -> None:
    config = get_provider_config("ollama", environ={})

    assert config.base_url == "http://localhost:11434/api"
    assert config.configured
    assert not config.requires_api_key


def test_openrouter_provider_parses_chat_completion_json() -> None:
    captured: dict[str, object] = {}

    def fake_transport(url: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "relationships": [
                                    {
                                        "source": "Org A",
                                        "target": "Org B",
                                        "ties": [{"type": "support", "reasoning_text": "shared agenda"}],
                                    }
                                ]
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    provider = OpenRouterProvider(
        config=get_provider_config("openrouter", environ={"OPENROUTER_API_KEY": "test-key"}),
        transport=fake_transport,
    )

    result = provider.annotate(
        rendered_prompt_text="Annotate this document",
        document_id="doc-1",
        model_id="anthropic/claude-3.5-sonnet",
        request_params={"temperature": 0.2, "max_output_tokens": 500},
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert result["parsed_output"]["relationships"][0]["source"] == "Org A"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}


def test_openrouter_provider_requires_api_key_before_request() -> None:
    provider = OpenRouterProvider(
        config=get_provider_config("openrouter", environ={}),
        transport=lambda _url, _headers, _payload: {},
    )

    with pytest.raises(ProviderExecutionError, match="OPENROUTER_API_KEY"):
        provider.annotate(
            rendered_prompt_text="Annotate this document",
            document_id="doc-1",
            model_id="anthropic/claude-3.5-sonnet",
            request_params={"temperature": 0.2, "max_output_tokens": 500},
        )


def test_openrouter_provider_attaches_raw_response_when_annotation_content_is_not_json() -> None:
    provider = OpenRouterProvider(
        config=get_provider_config("openrouter", environ={"OPENROUTER_API_KEY": "test-key"}),
        transport=lambda _url, _headers, _payload: {
            "choices": [{"message": {"content": "not json at all"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )

    with pytest.raises(ProviderExecutionError, match="non-JSON annotation content") as exc_info:
        provider.annotate(
            rendered_prompt_text="Annotate this document",
            document_id="doc-1",
            model_id="anthropic/claude-3.5-sonnet",
            request_params={"temperature": 0.2, "max_output_tokens": 500},
        )

    assert "not json at all" in exc_info.value.raw_response


def test_ollama_provider_parses_generate_json() -> None:
    captured: dict[str, object] = {}

    def fake_transport(url: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "response": json.dumps(
                {
                    "relationships": [
                        {
                            "source": "Org A",
                            "target": "Org B",
                            "ties": [{"type": "support", "reasoning_text": "shared agenda"}],
                        }
                    ]
                }
            ),
            "prompt_eval_count": 15,
            "eval_count": 22,
            "total_duration": 12345,
        }

    provider = OllamaProvider(
        config=get_provider_config("ollama", environ={}),
        transport=fake_transport,
    )

    result = provider.annotate(
        rendered_prompt_text="Annotate this document",
        document_id="doc-1",
        model_id="llama3.2",
        request_params={"temperature": 0.0, "max_output_tokens": 300},
    )

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["options"] == {"temperature": 0.0, "num_predict": 300}
    assert result["parsed_output"]["relationships"][0]["target"] == "Org B"
    assert result["usage"] == {
        "prompt_tokens": 15,
        "completion_tokens": 22,
        "total_duration": 12345,
    }


def test_ollama_provider_lists_available_models_from_api() -> None:
    provider = OllamaProvider(
        config=get_provider_config("ollama", environ={}),
        get_transport=lambda _url, _headers: {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "qwen2.5:7b"},
            ]
        },
    )

    assert provider.list_available_models() == ["llama3.2:latest", "qwen2.5:7b"]


def test_list_available_models_for_provider_falls_back_when_ollama_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(
        "corn.llm_providers.get_provider",
        lambda _provider_id, settings_by_provider=None: OllamaProvider(
            config=get_provider_config("ollama", environ={}),
            get_transport=lambda _url, _headers: (_ for _ in ()).throw(ProviderExecutionError("offline")),
        ),
    )

    models = list_available_models_for_provider("ollama")

    assert models == ["llama3.2", "gemma3", "mistral"]


def test_provider_config_uses_stored_base_url_but_not_stored_api_key() -> None:
    config = get_provider_config(
        "openrouter",
        environ={},
        stored_settings={
            "base_url": "https://example.com/router",
            "api_key": "stored-key",
        },
    )

    assert config.base_url == "https://example.com/router"
    assert config.api_key is None
    assert not config.configured
