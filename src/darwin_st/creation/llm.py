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
    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 16384) -> str: ...


# 默认 max_tokens=16384: DeepSeek-v4 推理模型的 reasoning 先吃输出额度, 可见输出排在后面,
# 实测 API 接受 65536(flash)/32768(pro), 余额充足 —— 默认给足, 别再让 reasoning 把可见输出挤没
# (历史教训: 诊断 1024 空响应, 反思 4096/8192 finish_reason=length)。


class MockLLM:
    """确定性桩: 用一个 responder(messages)->str 注入预设响应。本地测试用。"""

    def __init__(self, responder: Callable[[list[dict]], str]):
        self.responder = responder
        self.call_count = 0

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 16384) -> str:
        self.call_count += 1
        return self.responder(messages)


class OpenAICompatLLM:
    """OpenAI 兼容 Chat Completions 客户端 (DeepSeek)。纯 stdlib urllib, 零额外依赖。

    可复现性留档 (E14/R9 防线): 环境变量 LLM_ARCHIVE 指向 jsonl 路径时, 每次调用追加
    {ts, model, temperature, max_tokens, messages, response} —— prompt/response 全量留档,
    防模型版本漂移质疑 (默认空=不留档, 行为不变)。追加写失败只告警不影响调用。
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        timeout: float = 120.0,
    ):
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self._api_key = os.environ.get(api_key_env, "")
        if not self._api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置 (key 不应硬编码)")
        self.timeout = timeout
        self._archive_path = os.environ.get("LLM_ARCHIVE", "") or None

    def _log_call(self, messages: list[dict], temperature: float, max_tokens: int,
                  response: str | None, error: str | None) -> None:
        if not self._archive_path:
            return
        try:
            from datetime import datetime, timezone
            rec = {"ts": datetime.now(timezone.utc).isoformat(), "model": self.model,
                   "temperature": temperature, "max_tokens": max_tokens,
                   "messages": messages, "response": response, "error": error}
            with open(self._archive_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"[LLM] ⚠️ 留档写失败 (不影响调用): {type(e).__name__}: {str(e)[:120]}")

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 16384) -> str:
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
            text = data["choices"][0]["message"]["content"]
            self._log_call(messages, temperature, max_tokens, text, None)
            return text
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", "replace")
            self._log_call(messages, temperature, max_tokens, None,
                           f"HTTP {e.code}: {body[:200]}")
            raise RuntimeError(f"LLM API HTTP {e.code}: {body}") from None
