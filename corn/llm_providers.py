from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol
from urllib import error, request

PROVIDER_MODELS = {
    "fake": ["fake-default", "fake-verbose"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    "ollama": ["llama3.2", "gemma3", "mistral"],
    "openrouter": ["openai/gpt-4.1-mini", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.3-70b-instruct"],
}

GEMINI_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "ties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "reasoning_text": {"type": "string"},
                            },
                            "required": ["type", "reasoning_text"],
                        },
                    },
                },
                "required": ["source", "target", "ties"],
            },
        }
    },
    "required": ["relationships"],
}


class ProviderExecutionError(RuntimeError):
    """Raised when an LLM provider cannot complete a request."""

    def __init__(self, message: str, *, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


class LLMProvider(Protocol):
    provider_id: str

    def annotate(
        self,
        *,
        rendered_prompt_text: str,
        document_id: str,
        model_id: str,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    base_url: str
    api_key: str | None
    requires_api_key: bool
    configured: bool
    env_vars: tuple[str, ...]
    description: str


Transport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]
GetTransport = Callable[[str, dict[str, str]], dict[str, Any]]


@dataclass
class FakeLLMProvider:
    provider_id: str = "fake"
    outputs_by_document_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors_by_document_id: dict[str, str] = field(default_factory=dict)

    def annotate(
        self,
        *,
        rendered_prompt_text: str,
        document_id: str,
        model_id: str,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        if document_id in self.errors_by_document_id:
            raise ProviderExecutionError(self.errors_by_document_id[document_id])

        parsed_output = self.outputs_by_document_id.get(document_id, {"relationships": []})
        return {
            "raw_response": json.dumps(parsed_output),
            "parsed_output": parsed_output,
            "usage": {},
            "cost": {},
            "provider_id": self.provider_id,
            "model_id": model_id,
            "request_params": request_params,
        }


@dataclass
class GeminiProvider:
    config: ProviderConfig
    transport: Transport = None  # type: ignore[assignment]
    provider_id: str = "gemini"

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = _post_json

    def annotate(
        self,
        *,
        rendered_prompt_text: str,
        document_id: str,
        model_id: str,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.config.api_key:
            raise ProviderExecutionError(
                "Google Gemini is not configured. Set GEMINI_API_KEY to enable this provider."
            )

        payload = {
            "model": model_id,
            "input": rendered_prompt_text,
            "store": False,
            "generation_config": {
                "temperature": request_params.get("temperature", 0.0),
                "max_output_tokens": request_params.get("max_output_tokens", 800),
            },
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": GEMINI_ANNOTATION_SCHEMA,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key,
        }
        response_json = self.transport(
            f"{self.config.base_url.rstrip('/')}/interactions",
            headers,
            payload,
        )
        raw_response = json.dumps(response_json)
        content = _gemini_interaction_content(response_json)
        if content is None:
            raise ProviderExecutionError(
                "Google Gemini response did not include candidate text.",
                raw_response=raw_response,
            )

        try:
            parsed_output = _parse_json_content(content)
        except ProviderExecutionError as exc:
            raise ProviderExecutionError(str(exc), raw_response=raw_response) from exc
        return {
            "raw_response": raw_response,
            "parsed_output": parsed_output,
            "usage": response_json.get("usage", {}),
            "cost": {},
            "provider_id": self.provider_id,
            "model_id": model_id,
            "request_params": request_params,
        }


@dataclass
class OpenRouterProvider:
    config: ProviderConfig
    transport: Transport = None  # type: ignore[assignment]
    provider_id: str = "openrouter"

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = _post_json

    def annotate(
        self,
        *,
        rendered_prompt_text: str,
        document_id: str,
        model_id: str,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.config.api_key:
            raise ProviderExecutionError(
                "OpenRouter is not configured. Set OPENROUTER_API_KEY to enable this provider."
            )

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": rendered_prompt_text}],
            "temperature": request_params.get("temperature", 0.0),
            "max_completion_tokens": request_params.get("max_output_tokens", 800),
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_X_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        response_json = self.transport(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers,
            payload,
        )
        raw_response = json.dumps(response_json)
        try:
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderExecutionError(
                "OpenRouter response did not include a chat completion message.",
                raw_response=raw_response,
            ) from exc

        try:
            parsed_output = _parse_json_content(content)
        except ProviderExecutionError as exc:
            raise ProviderExecutionError(str(exc), raw_response=raw_response) from exc
        return {
            "raw_response": raw_response,
            "parsed_output": parsed_output,
            "usage": response_json.get("usage", {}),
            "cost": response_json.get("cost", {}),
            "provider_id": self.provider_id,
            "model_id": model_id,
            "request_params": request_params,
        }


@dataclass
class OllamaProvider:
    config: ProviderConfig
    transport: Transport = None  # type: ignore[assignment]
    get_transport: GetTransport = None  # type: ignore[assignment]
    provider_id: str = "ollama"

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = _post_json
        if self.get_transport is None:
            self.get_transport = _get_json

    def annotate(
        self,
        *,
        rendered_prompt_text: str,
        document_id: str,
        model_id: str,
        request_params: dict[str, Any],
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": model_id,
            "prompt": rendered_prompt_text,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": request_params.get("temperature", 0.0),
                "num_predict": request_params.get("max_output_tokens", 800),
            },
        }
        response_json = self.transport(
            f"{self.config.base_url.rstrip('/')}/generate",
            headers,
            payload,
        )
        raw_response = json.dumps(response_json)
        response_text = response_json.get("response")
        if not isinstance(response_text, str):
            raise ProviderExecutionError(
                "Ollama response did not include a response string.",
                raw_response=raw_response,
            )

        try:
            parsed_output = _parse_json_content(response_text)
        except ProviderExecutionError as exc:
            raise ProviderExecutionError(str(exc), raw_response=raw_response) from exc
        usage = {
            "prompt_tokens": response_json.get("prompt_eval_count"),
            "completion_tokens": response_json.get("eval_count"),
            "total_duration": response_json.get("total_duration"),
        }
        return {
            "raw_response": raw_response,
            "parsed_output": parsed_output,
            "usage": usage,
            "cost": {},
            "provider_id": self.provider_id,
            "model_id": model_id,
            "request_params": request_params,
        }

    def list_available_models(self) -> list[str]:
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response_json = self.get_transport(
            f"{self.config.base_url.rstrip('/')}/tags",
            headers,
        )
        models = response_json.get("models")
        if not isinstance(models, list):
            raise ProviderExecutionError("Ollama model list response did not include a models array.")

        discovered: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name:
                discovered.append(name)
        return discovered


def get_provider_config(
    provider_id: str,
    environ: Mapping[str, str] | None = None,
    stored_settings: Mapping[str, Any] | None = None,
) -> ProviderConfig:
    env = environ or os.environ
    stored = stored_settings or {}
    if provider_id == "fake":
        return ProviderConfig(
            provider_id="fake",
            base_url="",
            api_key=None,
            requires_api_key=False,
            configured=True,
            env_vars=(),
            description="Deterministic in-process provider for testing and development.",
        )
    if provider_id == "gemini":
        api_key = env.get("GEMINI_API_KEY")
        return ProviderConfig(
            provider_id="gemini",
            base_url=stored.get("base_url")
            or env.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            api_key=api_key,
            requires_api_key=True,
            configured=bool(api_key),
            env_vars=("GEMINI_API_KEY", "GEMINI_BASE_URL"),
            description="Sends structured annotation requests to the Google Gemini Interactions API.",
        )
    if provider_id == "openrouter":
        api_key = env.get("OPENROUTER_API_KEY")
        return ProviderConfig(
            provider_id="openrouter",
            base_url=stored.get("base_url")
            or env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
            requires_api_key=True,
            configured=bool(api_key),
            env_vars=("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
            description="Routes requests through OpenRouter's chat completions API.",
        )
    if provider_id == "ollama":
        base_url = stored.get("base_url") or env.get("OLLAMA_BASE_URL", "http://localhost:11434/api")
        api_key = env.get("OLLAMA_API_KEY")
        requires_api_key = base_url.startswith("https://ollama.com")
        return ProviderConfig(
            provider_id="ollama",
            base_url=base_url,
            api_key=api_key,
            requires_api_key=requires_api_key,
            configured=bool(api_key) if requires_api_key else True,
            env_vars=("OLLAMA_BASE_URL", "OLLAMA_API_KEY"),
            description="Runs against a local or cloud Ollama API endpoint.",
        )
    raise ValueError(f"Unknown provider: {provider_id}")


def provider_registry(settings_by_provider: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, LLMProvider]:
    return {
        "fake": FakeLLMProvider(),
        "gemini": GeminiProvider(
            config=get_provider_config("gemini", stored_settings=(settings_by_provider or {}).get("gemini"))
        ),
        "ollama": OllamaProvider(
            config=get_provider_config("ollama", stored_settings=(settings_by_provider or {}).get("ollama"))
        ),
        "openrouter": OpenRouterProvider(
            config=get_provider_config("openrouter", stored_settings=(settings_by_provider or {}).get("openrouter"))
        ),
    }


def list_provider_ids() -> list[str]:
    return sorted(provider_registry().keys())


def list_models_for_provider(provider_id: str) -> list[str]:
    if provider_id not in provider_registry():
        raise ValueError(f"Unknown provider: {provider_id}")
    return list(PROVIDER_MODELS.get(provider_id, []))


def list_available_models_for_provider(
    provider_id: str,
    settings_by_provider: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    if provider_id == "ollama":
        provider = get_provider(provider_id, settings_by_provider=settings_by_provider)
        assert isinstance(provider, OllamaProvider)
        try:
            models = provider.list_available_models()
        except ProviderExecutionError:
            return list(PROVIDER_MODELS.get(provider_id, []))
        return models or list(PROVIDER_MODELS.get(provider_id, []))
    return list_models_for_provider(provider_id)


def get_provider(
    provider_id: str,
    settings_by_provider: Mapping[str, Mapping[str, Any]] | None = None,
) -> LLMProvider:
    registry = provider_registry(settings_by_provider=settings_by_provider)
    if provider_id not in registry:
        raise ValueError(f"Unknown provider: {provider_id}")
    return registry[provider_id]


def _gemini_interaction_content(response_json: dict[str, Any]) -> str | list[Any] | None:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    steps = response_json.get("steps")
    if not isinstance(steps, list):
        return None
    for step in reversed(steps):
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        content = step.get("content")
        if isinstance(content, list) and content:
            return content
    return None


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        content = "".join(text_parts)
    if not isinstance(content, str):
        raise ProviderExecutionError("Provider response content was not JSON text.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProviderExecutionError("Provider returned non-JSON annotation content.") from exc
    if not isinstance(parsed, dict):
        raise ProviderExecutionError("Provider returned JSON that was not an object.")
    return parsed


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    request_body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=request_body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=60) as response:
            response_text = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise ProviderExecutionError(f"Provider HTTP error {exc.code}: {error_text}") from exc
    except error.URLError as exc:
        raise ProviderExecutionError(f"Provider connection failed: {exc.reason}") from exc

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ProviderExecutionError("Provider returned a non-JSON HTTP response.") from exc
    if not isinstance(parsed, dict):
        raise ProviderExecutionError("Provider returned a JSON response that was not an object.")
    return parsed


def _get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=30) as response:
            response_text = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise ProviderExecutionError(f"Provider HTTP error {exc.code}: {error_text}") from exc
    except error.URLError as exc:
        raise ProviderExecutionError(f"Provider connection failed: {exc.reason}") from exc

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ProviderExecutionError("Provider returned a non-JSON HTTP response.") from exc
    if not isinstance(parsed, dict):
        raise ProviderExecutionError("Provider returned a JSON response that was not an object.")
    return parsed
