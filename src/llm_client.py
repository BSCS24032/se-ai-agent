"""LLM client abstraction.

The rest of the application talks to one stable interface (``LLMClient``)
regardless of which provider is in use (Groq, Ollama, OpenAI-compatible,
or a no-LLM fallback). This isolates the AI/ML layer so we can swap
providers without touching the pipeline or UI.

The default provider is Groq because it offers a generous free tier and
hosts the Llama 3.x family the assignment scope targets. Ollama covers
the fully-local route and is great for offline demos. The "none"
provider returns deterministic stub answers so the app still runs
without any key.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from config import Settings, load_settings
from src.http_utils import HTTPRetryError, request_with_retry

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM call fails in a non-retryable way."""


@dataclass
class LLMResponse:
    """A minimal, provider-agnostic response wrapper."""

    text: str
    provider: str
    model: str


class LLMClient:
    """Provider-agnostic LLM client.

    Usage::

        client = LLMClient()  # reads from config.SETTINGS by default
        reply = client.complete(system="You are helpful.", user="Hi")
        print(reply.text)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Single-turn chat completion.

        Args:
            system: System prompt establishing role and constraints.
            user: User message.
            temperature: Sampling temperature. Lower is more deterministic.
            max_tokens: Maximum generation length.
            json_mode: Hint the provider to return strict JSON if supported.

        Returns:
            LLMResponse with the model's text reply.

        Raises:
            LLMError: If the request fails and there is no useful fallback.
        """
        provider = self.settings.llm_provider
        try:
            if provider == "groq":
                return self._call_openai_compatible(
                    system,
                    user,
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.settings.groq_api_key,
                    model=self.settings.groq_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    provider_name="groq",
                )
            if provider == "openai":
                return self._call_openai_compatible(
                    system,
                    user,
                    base_url=self.settings.openai_base_url,
                    api_key=self.settings.openai_api_key,
                    model=self.settings.openai_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    provider_name="openai",
                )
            if provider == "ollama":
                return self._call_ollama(
                    system,
                    user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            # "none" or anything unrecognized
            return self._stub_response(system, user, json_mode)
        except LLMError:
            raise
        except HTTPRetryError as exc:
            raise LLMError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("LLM call failed unexpectedly.")
            raise LLMError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------
    def _call_openai_compatible(
        self,
        system: str,
        user: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        provider_name: str,
    ) -> LLMResponse:
        if not api_key:
            raise LLMError(
                f"{provider_name} provider selected but no API key configured."
            )
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # Groq and OpenAI both support response_format json_object on
            # supported models. Harmless on those that ignore it.
            payload["response_format"] = {"type": "json_object"}
        resp = request_with_retry(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=self.settings.request_timeout_seconds,
        )
        if resp.status_code >= 400:
            raise LLMError(
                f"{provider_name} HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(
                f"Unexpected {provider_name} response: {resp.text[:300]}"
            ) from exc
        return LLMResponse(text=text.strip(), provider=provider_name, model=model)

    def _call_ollama(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        url = self.settings.ollama_base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        resp = request_with_retry(
            "POST",
            url,
            json=payload,
            timeout=self.settings.request_timeout_seconds,
        )
        if resp.status_code >= 400:
            raise LLMError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"Ollama returned non-JSON: {resp.text[:300]}") from exc
        text = data.get("message", {}).get("content", "")
        if not text:
            raise LLMError(f"Empty Ollama response: {data}")
        return LLMResponse(
            text=text.strip(),
            provider="ollama",
            model=self.settings.ollama_model,
        )

    # ------------------------------------------------------------------
    # No-LLM fallback
    # ------------------------------------------------------------------
    def _stub_response(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        """Deterministic stub used when no provider is configured.

        Lets the app still run end-to-end as a keyword-only demo so a
        grader can see the pipeline without any external dependencies.
        """
        if json_mode:
            payload = {
                "study_type": "unknown",
                "population": "unknown",
                "intervention": "unknown",
                "outcome": "unknown",
                "key_finding": (
                    "LLM is not configured; structured extraction is unavailable."
                ),
            }
            return LLMResponse(
                text=json.dumps(payload), provider="none", model="stub"
            )
        return LLMResponse(
            text="[No LLM configured] " + user.split("\n", 1)[0][:200],
            provider="none",
            model="stub",
        )
