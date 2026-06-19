"""LLM 客户端 (LLM Client) —— 可插拔, 默认 OpenAI 兼容 (DeepSeek)。

Tier-2 创造层的 LLM 接入点。两个实现 (同接口):
  - MockLLM: 确定性桩, 本地测试用 (按提示返回预设响应, 不触网)。
  - OpenAICompatLLM: 调 OpenAI 兼容 API (DeepSeek v4-pro)。key 从环境变量读, 绝不硬编码。

接口: chat(messages) -> str。
安全: key 只从 env (DEEPSEEK_API_KEY) 读; base_url/model 可配; 不打印 key。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Protocol

__all__ = ["LLMClient", "MockLLM", "OpenAICompatLLM"]


class LLMClient(Protocol):
    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096) -> str: ...


class MockLLM:
    """确定性桩: 用一个 responder(messages)->str 注入预设响应。本地测试用。"""

    def __init__(self, responder: Callable[[list[dict]], str]):
        self.responder = responder
        self.call_count = 0

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        self.call_count += 1
        return self.responder(messages)


class OpenAICompatLLM:
    """OpenAI 兼容 Chat Completions 客户端 (DeepSeek)。纯 stdlib urllib, 零额外依赖。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        timeout: float = 120.0,
    ):
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self._api_key = os.environ.get(api_key_env, "")
        if not self._api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置 (key 不应硬编码)")
        self.timeout = timeout

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        payload = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"LLM API HTTP {e.code}: {body}") from None
