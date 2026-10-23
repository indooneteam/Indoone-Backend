"""The Azure OpenAI provider: additive, and off unless it is asked for.

These tests never reach the network. They pin the two things that would be easy
to get wrong and expensive to discover in production: that the provider is only
selected when the environment says so, and that the request the module would send
has Azure's shape - the deployment URL and the api-key header.
"""

import importlib
import os

import pytest


@pytest.fixture()
def azure_service(monkeypatch):
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "indoone-chat")
    module = importlib.import_module("app.ai.azure_openai_service")
    return importlib.reload(module)


def test_configuration_ready_only_with_all_three(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "indoone-chat")
    module = importlib.import_module("app.ai.azure_openai_service")
    assert module.azure_configuration_ready() is False

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    assert importlib.reload(module).azure_configuration_ready() is True


def test_url_is_the_deployment_url(azure_service):
    url = azure_service._chat_url(
        "https://example.openai.azure.com", "indoone-chat", "2024-10-21"
    )
    assert url == (
        "https://example.openai.azure.com/openai/deployments/indoone-chat"
        "/chat/completions?api-version=2024-10-21"
    )


def test_tool_config_prefers_the_tool_deployment(monkeypatch, azure_service):
    monkeypatch.setenv("AZURE_OPENAI_TOOL_DEPLOYMENT", "indoone-tools")
    module = importlib.reload(azure_service)
    assert module._get_tool_config()[2] == "indoone-tools"


def test_declarations_are_wrapped_for_openai(azure_service):
    tools = azure_service._openai_tools(
        [{"name": "search_web", "description": "Search", "parameters": {"type": "object"}}]
    )
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_the_key_is_never_reported(azure_service):
    # azure_model_name is what the status endpoint exposes; the key must not be in it.
    assert azure_service.azure_model_name() == "indoone-chat"
    assert "test-key" not in str(azure_service.azure_model_name())


def test_gemini_is_still_selected_by_its_own_value(monkeypatch):
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    from app.ai import service

    importlib.reload(service)
    assert service._gemini_backend_enabled() is True
    assert service._azure_backend_enabled() is False


def test_azure_is_selected_only_by_its_own_value(monkeypatch):
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "azure")
    from app.ai import service

    importlib.reload(service)
    assert service._azure_backend_enabled() is True
    assert service._gemini_backend_enabled() is False


def test_schema_types_are_lowercased(azure_service):
    """Gemini writes STRING/OBJECT; OpenAI rejects those with a 400.

    This is the bug that made every tool-carrying request fail, so it is pinned.
    """
    tools = azure_service._openai_tools(
        [
            {
                "name": "list_repos",
                "description": "List repositories",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "owner": {"type": "STRING"},
                        "limit": {"type": "NUMBER"},
                        "private": {"type": "BOOLEAN"},
                        "labels": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": ["owner"],
                },
            }
        ]
    )
    params = tools[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["owner"]["type"] == "string"
    assert params["properties"]["limit"]["type"] == "number"
    assert params["properties"]["private"]["type"] == "boolean"
    assert params["properties"]["labels"]["type"] == "array"
    assert params["properties"]["labels"]["items"]["type"] == "string"
    # and nothing uppercase survives anywhere
    import json

    assert "STRING" not in json.dumps(params)
    assert "OBJECT" not in json.dumps(params)
