"""Chat model adapters behind one small interface.

* ``OpenAIChat``: the current ``openai`` SDK (extra ``llm``). ``base_url`` can point to any
  OpenAI-compatible server. Temperature is always 0.
* ``ScriptedLLM``: a fake for tests. It returns fixed replies and records the prompts.
"""
from __future__ import annotations

import os
from typing import Protocol


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAIChat:
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, timeout_s: float = 30.0, api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("set OPENAI_API_KEY (or use FRIDGE2FORK_LLM=none)")
        try:
            from openai import OpenAI  # noqa: PLC0415  (optional extra, openai>=1)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError('the openai model needs: pip install "fridge2fork[llm]" (openai>=1)') from exc
        self.model = model
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout_s)

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - network
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=600,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class ScriptedLLM:
    name = "scripted"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.replies.pop(0) if self.replies else ""
