"""LLM client factory for Accordis inference."""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract base for all LLM clients."""

    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """Send a completion request and return the response text."""

    async def close(self) -> None:
        """Release provider resources when a client supports explicit cleanup."""


class OpenAIClient(BaseLLMClient):
    """OpenAI chat-completion client."""

    def __init__(self, model: str) -> None:
        from openai import AsyncOpenAI
        self._BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self._MODEL = model
        self._API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        self._client = AsyncOpenAI(
            base_url=self._BASE_URL,
            api_key=self._API_KEY
        )

    async def complete(self, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if not callable(close):
            return

        result = close()
        if asyncio.iscoroutine(result):
            await result


class GeminiClient(BaseLLMClient):
    """Google Gemini client with the same `complete(system, user)` contract."""

    def __init__(self, model: str) -> None:
        from google import genai
        self._MODEL = model
        self._API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        self._client = genai.Client(api_key=self._API_KEY)
        self._async_client = getattr(self._client, "aio", None)

    async def complete(self, system: str, user: str) -> str:
        if self._async_client is not None:
            resp = await self._async_client.models.generate_content(
                model=self._MODEL,
                contents=user,
                config={"system_instruction": system},
            )
        else:
            resp = await asyncio.to_thread(
                self._client.models.generate_content,
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

    async def close(self) -> None:
        for target in (self._async_client, self._client):
            close = getattr(target, "close", None)
            if not callable(close):
                continue

            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except AttributeError as exc:
                if "_async_httpx_client" not in str(exc):
                    raise


class LLMClientFactory:
    """Factory that selects an LLM client based on available API keys."""

    @staticmethod
    def create(provider: str, model: str) -> BaseLLMClient:
        if provider == "openai":
            return OpenAIClient(model=model)
        if provider == "gemini":
            return GeminiClient(model=model)
        raise EnvironmentError(
            "No Provider set. Set PROVIDER to 'openai' or 'gemini'."
        )
