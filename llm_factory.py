"""LLM client factory for Accordis inference."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract base for all LLM clients."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a completion request and return the response text."""


class OpenAIClient(BaseLLMClient):
    """OpenAI chat-completion client."""

    _MODEL = "gpt-4o"

    def __init__(self) -> None:
        from openai import OpenAI
        self._BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self._MODEL = os.getenv("MODEL_NAME")
        self._API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        self._client = OpenAI(
            base_url=self._BASE_URL,
            api_key=self._API_KEY
        )

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


class GeminiClient(BaseLLMClient):
    """Google Gemini client with the same `complete(system, user)` contract."""

    def __init__(self) -> None:
        from google import genai
        self._MODEL = os.getenv("MODEL_NAME")
        self._API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        self._client = genai.Client(api_key=self._API_KEY)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.models.generate_content(
            model=self._MODEL,
            contents=user,
            config={"system_instruction": system},
        )
        return self._extract_text(resp).strip()

    @staticmethod
    def _extract_text(resp: Any) -> str:
        text = getattr(resp, "text", None)
        if text:
            return text

        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            return ""

        parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
        texts = [part.text for part in parts if getattr(part, "text", None)]
        return "".join(texts)


class LLMClientFactory:
    """Factory that selects an LLM client based on available API keys."""

    @staticmethod
    def create() -> BaseLLMClient:
        if os.getenv("PROVIDER") == "openai":
            return OpenAIClient()
        if os.getenv("PROVIDER") == "gemini":
            return GeminiClient()
        raise EnvironmentError(
            "No Provider set. Set PROVIDER to 'openai' or 'gemini'."
        )
