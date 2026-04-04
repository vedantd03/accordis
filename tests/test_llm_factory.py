"""Tests for the LLM client factory and provider adapters."""

from __future__ import annotations

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
        def create(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="  openai reply  ")
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key):
            calls["api_key"] = api_key
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    llm_factory = _reload_llm_factory()
    client = llm_factory.OpenAIClient()

    assert client.complete("sys prompt", "user prompt") == "openai reply"
    assert calls["api_key"] == "openai-key"
    assert calls["model"] == "gpt-4o"
    assert calls["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_gemini_client_complete_uses_generate_content(monkeypatch):
    calls = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(text="  gemini reply  ")

    class FakeGenAIClient:
        def __init__(self, *, api_key):
            calls["api_key"] = api_key
            self.models = FakeModels()

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeGenAIClient
    google_module.genai = genai_module

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    llm_factory = _reload_llm_factory()
    client = llm_factory.GeminiClient()

    assert client.complete("sys prompt", "user prompt") == "gemini reply"
    assert calls["api_key"] == "gemini-key"
    assert calls["model"] == "gemini-2.0-flash"
    assert calls["contents"] == "user prompt"
    assert calls["config"] == {"system_instruction": "sys prompt"}


def test_gemini_client_extracts_text_from_parts(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
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

    class FakeGenAIClient:
        def __init__(self, *, api_key):
            self.models = FakeModels()

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeGenAIClient
    google_module.genai = genai_module

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    llm_factory = _reload_llm_factory()
    client = llm_factory.GeminiClient()

    assert client.complete("sys prompt", "user prompt") == '{"node_0": {}}'


def test_factory_prefers_openai_when_both_keys_are_set(monkeypatch):
    class FakeOpenAI:
        def __init__(self, *, api_key):
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

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    llm_factory = _reload_llm_factory()

    assert isinstance(llm_factory.LLMClientFactory.create(), llm_factory.OpenAIClient)
