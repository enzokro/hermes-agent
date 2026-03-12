"""Tests for flush_memories() working correctly across all provider modes.

Catches the bug where Codex mode called chat.completions.create on a
Responses-only client, which would fail silently or with a 404.
"""

import json
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, call

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key", "test")
        self.base_url = kwargs.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(monkeypatch, api_mode="chat_completions", provider="openrouter"):
    """Build an AIAgent with mocked internals, ready for flush_memories testing."""
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kw: [
        {
            "type": "function",
            "function": {
                "name": "memory",
                "description": "Manage memories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
    ])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "OpenAI", _FakeOpenAI)

    agent = run_agent.AIAgent(
        api_key="test-key",
        base_url="https://test.example.com/v1",
        provider=provider,
        api_mode=api_mode,
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    # Give it a valid memory store with mock success responses
    mock_store = MagicMock()
    mock_store._success_response.return_value = {
        "success": True, "entries": [], "usage": "0% — 0/2,200 chars", "entry_count": 0,
    }
    agent._memory_store = mock_store
    agent._memory_flush_min_turns = 1
    agent._flush_max_rounds = 1  # Default to single-round for existing tests
    agent._user_turn_count = 5
    return agent


def _chat_response_with_memory_call():
    """Simulated chat completions response with a memory tool call."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="call_flush_1",
                    function=SimpleNamespace(
                        name="memory",
                        arguments=json.dumps({
                            "action": "add",
                            "target": "notes",
                            "content": "User prefers dark mode.",
                        }),
                    ),
                )],
            ),
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )


class TestFlushMemoriesUsesAuxiliaryClient:
    """When an auxiliary client is available, flush_memories should use it
    instead of self.client -- especially critical in Codex mode."""

    def test_flush_uses_auxiliary_when_available(self, monkeypatch):
        agent = _make_agent(monkeypatch, api_mode="codex_responses", provider="openai-codex")

        mock_response = _chat_response_with_memory_call()

        with patch("agent.auxiliary_client.call_llm", return_value=mock_response) as mock_call:
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "Remember this"},
            ]
            with patch("tools.memory_tool.memory_tool", return_value="Saved.") as mock_memory:
                agent.flush_memories(messages)

        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs.get("task") == "flush_memories"

    def test_flush_uses_main_client_when_no_auxiliary(self, monkeypatch):
        """Non-Codex mode with no auxiliary falls back to self.client."""
        agent = _make_agent(monkeypatch, api_mode="chat_completions", provider="openrouter")
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _chat_response_with_memory_call()

        with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("no provider")):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "Save this"},
            ]
            with patch("tools.memory_tool.memory_tool", return_value="Saved."):
                agent.flush_memories(messages)

        agent.client.chat.completions.create.assert_called_once()

    def test_flush_executes_memory_tool_calls(self, monkeypatch):
        """Verify that memory tool calls from the flush response actually get executed."""
        agent = _make_agent(monkeypatch, api_mode="chat_completions", provider="openrouter")

        mock_response = _chat_response_with_memory_call()

        with patch("agent.auxiliary_client.call_llm", return_value=mock_response):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Note this"},
            ]
            with patch("tools.memory_tool.memory_tool", return_value="Saved.") as mock_memory:
                agent.flush_memories(messages)

        mock_memory.assert_called_once()
        call_kwargs = mock_memory.call_args
        assert call_kwargs.kwargs["action"] == "add"
        assert call_kwargs.kwargs["target"] == "notes"
        assert "dark mode" in call_kwargs.kwargs["content"]

    def test_flush_strips_artifacts_from_messages(self, monkeypatch):
        """After flush, the flush prompt and any response should be removed from messages."""
        agent = _make_agent(monkeypatch, api_mode="chat_completions", provider="openrouter")

        mock_response = _chat_response_with_memory_call()

        with patch("agent.auxiliary_client.call_llm", return_value=mock_response):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Remember X"},
            ]
            original_len = len(messages)
            with patch("tools.memory_tool.memory_tool", return_value="Saved."):
                agent.flush_memories(messages)

        # Messages should not grow from the flush
        assert len(messages) <= original_len
        # No flush sentinel should remain
        for msg in messages:
            assert "_flush_sentinel" not in msg


class TestFlushMemoriesCodexFallback:
    """When no auxiliary client exists and we're in Codex mode, flush should
    use the Codex Responses API path instead of chat.completions."""

    def test_codex_mode_no_aux_uses_responses_api(self, monkeypatch):
        agent = _make_agent(monkeypatch, api_mode="codex_responses", provider="openai-codex")

        codex_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="memory",
                    arguments=json.dumps({
                        "action": "add",
                        "target": "notes",
                        "content": "Codex flush test",
                    }),
                ),
            ],
            usage=SimpleNamespace(input_tokens=50, output_tokens=10, total_tokens=60),
            status="completed",
            model="gpt-5-codex",
        )

        with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("no provider")), \
             patch.object(agent, "_run_codex_stream", return_value=codex_response) as mock_stream, \
             patch.object(agent, "_build_api_kwargs") as mock_build, \
             patch("tools.memory_tool.memory_tool", return_value="Saved.") as mock_memory:
            mock_build.return_value = {
                "model": "gpt-5-codex",
                "instructions": "test",
                "input": [],
                "tools": [],
                "max_output_tokens": 4096,
            }
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Save this"},
            ]
            agent.flush_memories(messages)

        mock_stream.assert_called_once()
        mock_memory.assert_called_once()
        assert mock_memory.call_args.kwargs["content"] == "Codex flush test"


class TestFlushMemoriesMultiRound:
    """Multi-round flush: model can sequence remove → add across 2 turns."""

    def test_flush_multi_round_remove_then_add(self, monkeypatch):
        """First response removes an entry, second response adds a new one."""
        agent = _make_agent(monkeypatch)
        agent._flush_max_rounds = 2

        # Round 1: model returns a remove call
        remove_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        id="call_r1",
                        function=SimpleNamespace(
                            name="memory",
                            arguments=json.dumps({
                                "action": "remove",
                                "target": "memory",
                                "old_text": "stale entry",
                            }),
                        ),
                    )],
                ),
            )],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

        # Round 2: model returns an add call
        add_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        id="call_r2",
                        function=SimpleNamespace(
                            name="memory",
                            arguments=json.dumps({
                                "action": "add",
                                "target": "memory",
                                "content": "consolidated entry",
                            }),
                        ),
                    )],
                ),
            )],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=25, total_tokens=145),
        )

        call_count = {"n": 0}

        def mock_call_llm(**kwargs):
            call_count["n"] += 1
            return remove_response if call_count["n"] == 1 else add_response

        with patch("agent.auxiliary_client.call_llm", side_effect=mock_call_llm):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Done"},
            ]
            with patch("tools.memory_tool.memory_tool", return_value='{"success": true}') as mock_mem:
                agent.flush_memories(messages)

        assert call_count["n"] == 2
        assert mock_mem.call_count == 2
        # First call was remove, second was add
        calls = mock_mem.call_args_list
        assert calls[0].kwargs["action"] == "remove"
        assert calls[1].kwargs["action"] == "add"

    def test_flush_early_exit_no_tool_calls(self, monkeypatch):
        """If round 1 produces no tool calls, skip round 2."""
        agent = _make_agent(monkeypatch)
        agent._flush_max_rounds = 2

        empty_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Nothing to save.", tool_calls=None),
            )],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )

        with patch("agent.auxiliary_client.call_llm", return_value=empty_response) as mock_call:
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Bye"},
            ]
            agent.flush_memories(messages)

        mock_call.assert_called_once()  # Only 1 round, not 2

    def test_flush_sentinel_cleanup_after_multi_round(self, monkeypatch):
        """Sentinel cleanup works correctly after 2-round flush."""
        agent = _make_agent(monkeypatch)
        agent._flush_max_rounds = 2

        # Return tool calls on round 1, none on round 2
        responses = [
            _chat_response_with_memory_call(),
            SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="Done.", tool_calls=None),
                )],
                usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5, total_tokens=55),
            ),
        ]
        call_idx = {"n": 0}

        def mock_call_llm(**kwargs):
            idx = call_idx["n"]
            call_idx["n"] += 1
            return responses[min(idx, len(responses) - 1)]

        with patch("agent.auxiliary_client.call_llm", side_effect=mock_call_llm):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Save"},
            ]
            original_len = len(messages)
            with patch("tools.memory_tool.memory_tool", return_value='{"success": true}'):
                agent.flush_memories(messages)

        assert len(messages) <= original_len
        for msg in messages:
            assert "_flush_sentinel" not in msg
