"""Tests for the LLM client factory and provider adapters."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import types


def _reload_llm_factory():
    module_name = "_test_llm_factory"
    sys.modules.pop(module_name, None)

    module_path = Path(__file__).resolve().parents[1] / "llm_factory.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_openai_client_complete_uses_chat_completions(monkeypatch):
    calls = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="  openai reply  ")
                    )
                ]
            )

    class FakeAsyncOpenAI:
        def __init__(self, *, base_url, api_key):
            calls["base_url"] = base_url
            calls["api_key"] = api_key
            self.chat = types.SimpleNamespace(completions=FakeCompletions())
            self.closed = False

        async def close(self):
            self.closed = True

    monkeypatch.setenv("API_KEY", "openai-key")
    monkeypatch.setenv("API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    llm_factory = _reload_llm_factory()
    client = llm_factory.OpenAIClient(model="gpt-4o")

    assert asyncio.run(client.complete("sys prompt", "user prompt")) == "openai reply"
    assert calls["api_key"] == "openai-key"
    assert calls["base_url"] == "https://api.openai.com/v1"
    assert calls["model"] == "gpt-4o"
    assert calls["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    asyncio.run(client.close())


def test_gemini_client_complete_uses_generate_content(monkeypatch):
    calls = {}

    class FakeModels:
        async def generate_content(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(text="  gemini reply  ")

    class FakeAsyncGenAIClient:
        def __init__(self):
            self.models = FakeModels()

        async def close(self):
            return None

    class FakeGenAIClient:
        def __init__(self, *, api_key):
            calls["api_key"] = api_key
            self.models = types.SimpleNamespace()
            self.aio = FakeAsyncGenAIClient()
            self.closed = False

        async def close(self):
            self.closed = True

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeGenAIClient
    google_module.genai = genai_module

    monkeypatch.setenv("API_KEY", "gemini-key")
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    llm_factory = _reload_llm_factory()
    client = llm_factory.GeminiClient(model="gemini-2.0-flash")

    assert asyncio.run(client.complete("sys prompt", "user prompt")) == "gemini reply"
    assert calls["api_key"] == "gemini-key"
    assert calls["model"] == "gemini-2.0-flash"
    assert calls["contents"] == "user prompt"
    assert calls["config"] == {"system_instruction": "sys prompt"}
    asyncio.run(client.close())


def test_gemini_client_extracts_text_from_parts(monkeypatch):
    class FakeModels:
        async def generate_content(self, **kwargs):
            return types.SimpleNamespace(
                text=None,
                candidates=[
                    types.SimpleNamespace(
                        content=types.SimpleNamespace(
                            parts=[
                                types.SimpleNamespace(text="{"),
                                types.SimpleNamespace(text='"node_0": {}'),
                                types.SimpleNamespace(text="}"),
                            ]
                        )
                    )
                ],
            )

    class FakeAsyncGenAIClient:
        def __init__(self):
            self.models = FakeModels()

    class FakeGenAIClient:
        def __init__(self, *, api_key):
            self.models = types.SimpleNamespace()
            self.aio = FakeAsyncGenAIClient()

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeGenAIClient
    google_module.genai = genai_module

    monkeypatch.setenv("API_KEY", "gemini-key")
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    llm_factory = _reload_llm_factory()
    client = llm_factory.GeminiClient(model="gemini-2.0-flash")

    assert asyncio.run(client.complete("sys prompt", "user prompt")) == '{"node_0": {}}'


def test_factory_uses_provider_to_select_openai(monkeypatch):
    class FakeOpenAI:
        def __init__(self, *, base_url, api_key):
            self.base_url = base_url
            self.api_key = api_key
            self.chat = types.SimpleNamespace(completions=None)

    class FakeGenAIClient:
        def __init__(self, *, api_key):
            self.api_key = api_key
            self.models = None

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeGenAIClient
    google_module.genai = genai_module

    monkeypatch.setenv("API_KEY", "openai-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    llm_factory = _reload_llm_factory()

    assert isinstance(llm_factory.LLMClientFactory.create("openai", "gpt-4o"), llm_factory.OpenAIClient)


def test_gemini_client_close_swallows_missing_async_httpx_client(monkeypatch):
    class FakeAsyncGenAIClient:
        async def close(self):
            raise AttributeError("'BaseApiClient' object has no attribute '_async_httpx_client'")

    class FakeGenAIClient:
        def __init__(self, *, api_key):
            self.models = None
            self.aio = FakeAsyncGenAIClient()

        async def close(self):
            raise AttributeError("'BaseApiClient' object has no attribute '_async_httpx_client'")

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeGenAIClient
    google_module.genai = genai_module

    monkeypatch.setenv("API_KEY", "gemini-key")
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    llm_factory = _reload_llm_factory()
    client = llm_factory.GeminiClient(model="gemini-2.0-flash")

    asyncio.run(client.close())
