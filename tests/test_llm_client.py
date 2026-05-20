"""Tests for the LLM client across providers, using HTTP mocks."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from config import Settings
from src.llm_client import LLMClient, LLMError


class _Resp:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return self._body if isinstance(self._body, dict) else json.loads(self._body)


def _groq_settings(key="dummy-key") -> Settings:
    return Settings(llm_provider="groq", groq_api_key=key)


def test_none_provider_returns_stub_text():
    client = LLMClient(settings=Settings(llm_provider="none"))
    resp = client.complete(system="sys", user="hello")
    assert resp.provider == "none"
    assert "[No LLM configured]" in resp.text


def test_none_provider_returns_stub_json():
    client = LLMClient(settings=Settings(llm_provider="none"))
    resp = client.complete(system="sys", user="x", json_mode=True)
    payload = json.loads(resp.text)
    assert payload["study_type"] == "unknown"


def test_groq_provider_calls_chat_completions():
    body = {
        "choices": [{"message": {"content": "hello world"}}],
    }
    with patch(
        "src.http_utils.requests.request",
        return_value=_Resp(200, body),
    ) as mock_req:
        client = LLMClient(settings=_groq_settings())
        resp = client.complete(system="s", user="u")
    assert resp.text == "hello world"
    assert resp.provider == "groq"
    call = mock_req.call_args
    assert "groq.com" in call.kwargs["url"]
    payload = call.kwargs["json"]
    assert payload["model"].startswith("llama")
    assert payload["messages"][0]["role"] == "system"


def test_groq_provider_missing_key_raises():
    client = LLMClient(settings=Settings(llm_provider="groq", groq_api_key=""))
    with pytest.raises(LLMError):
        client.complete(system="s", user="u")


def test_groq_provider_translates_http_4xx_to_llmerror():
    with patch(
        "src.http_utils.requests.request",
        return_value=_Resp(401, "bad key"),
    ):
        client = LLMClient(settings=_groq_settings())
        with pytest.raises(LLMError):
            client.complete(system="s", user="u")


def test_ollama_provider_calls_chat_endpoint():
    body = {"message": {"content": "ollama hi"}}
    with patch(
        "src.http_utils.requests.request",
        return_value=_Resp(200, body),
    ) as mock_req:
        client = LLMClient(settings=Settings(llm_provider="ollama"))
        resp = client.complete(system="s", user="u")
    assert resp.text == "ollama hi"
    assert resp.provider == "ollama"
    assert "/api/chat" in mock_req.call_args.kwargs["url"]


def test_json_mode_adds_response_format_for_openai_family():
    body = {"choices": [{"message": {"content": '{"a":1}'}}]}
    with patch(
        "src.http_utils.requests.request",
        return_value=_Resp(200, body),
    ) as mock_req:
        client = LLMClient(settings=_groq_settings())
        client.complete(system="s", user="u", json_mode=True)
    assert (
        mock_req.call_args.kwargs["json"]["response_format"]
        == {"type": "json_object"}
    )
