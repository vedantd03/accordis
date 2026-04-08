"""LLM client factory for Accordis inference."""

from __future__ import annotations

import asyncio
import os
import re
import time
from abc import ABC, abstractmethod
from collections import deque
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
        self._BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
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

    _REQUESTS_PER_MINUTE = 15
    _WINDOW_SECONDS = 60.0
    _KEY_PATTERN = re.compile(r"^GEMINI(?:_API)?_KEY_(\d+)$")

    def __init__(self, model: str) -> None:
        from google import genai

        self._MODEL = model
        self._rotation_lock = asyncio.Lock()
        self._next_key_index = 0
        self._client_entries: list[dict[str, Any]] = []
        self._request_windows: list[deque[float]] = []

        for api_key in self._discover_api_keys():
            client = genai.Client(api_key=api_key)
            self._client_entries.append(
                {
                    "api_key": api_key,
                    "client": client,
                    "async_client": getattr(client, "aio", None),
                }
            )
            self._request_windows.append(deque())

        if not self._client_entries:
            fallback_key = os.getenv("API_KEY")
            if not fallback_key:
                raise EnvironmentError(
                    "No Gemini API key configured. Set GEMINI_API_KEY_<n>, GEMINI_KEY_<n>, or API_KEY."
                )

            client = genai.Client(api_key=fallback_key)
            self._client_entries.append(
                {
                    "api_key": fallback_key,
                    "client": client,
                    "async_client": getattr(client, "aio", None),
                }
            )
            self._request_windows.append(deque())

    def _discover_api_keys(self) -> list[str]:
        keyed_entries: list[tuple[int, str]] = []
        for env_name, value in os.environ.items():
            if not value:
                continue

            match = self._KEY_PATTERN.match(env_name)
            if match is None:
                continue

            keyed_entries.append((int(match.group(1)), value))

        keyed_entries.sort(key=lambda item: item[0])
        return [api_key for _, api_key in keyed_entries]

    async def complete(self, system: str, user: str) -> str:
        entry = await self._acquire_client_entry()
        async_client = entry["async_client"]
        client = entry["client"]

        if async_client is not None:
            resp = await async_client.models.generate_content(
                model=self._MODEL,
                contents=user,
                config={"system_instruction": system},
            )
        else:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=self._MODEL,
                contents=user,
                config={"system_instruction": system},
            )
        return self._extract_text(resp).strip()

    async def _acquire_client_entry(self) -> dict[str, Any]:
        while True:
            async with self._rotation_lock:
                now = time.monotonic()
                for request_window in self._request_windows:
                    self._prune_request_window(request_window, now)

                entry = self._reserve_next_available_entry(now)
                if entry is not None:
                    return entry

                wait_for = self._seconds_until_next_slot(now)

            await asyncio.sleep(wait_for)

    def _reserve_next_available_entry(self, now: float) -> dict[str, Any] | None:
        total_entries = len(self._client_entries)
        for offset in range(total_entries):
            index = (self._next_key_index + offset) % total_entries
            request_window = self._request_windows[index]
            if len(request_window) >= self._REQUESTS_PER_MINUTE:
                continue

            request_window.append(now)
            if len(request_window) >= self._REQUESTS_PER_MINUTE:
                self._next_key_index = (index + 1) % total_entries
            else:
                self._next_key_index = index
            return self._client_entries[index]

        return None

    def _seconds_until_next_slot(self, now: float) -> float:
        waits = []
        for request_window in self._request_windows:
            if not request_window:
                return 0.0
            waits.append(max(0.0, self._WINDOW_SECONDS - (now - request_window[0])))
        return min(waits) if waits else 0.0

    def _prune_request_window(self, request_window: deque[float], now: float) -> None:
        while request_window and now - request_window[0] >= self._WINDOW_SECONDS:
            request_window.popleft()

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
        for entry in self._client_entries:
            for target in (entry["async_client"], entry["client"]):
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
