"""LLM 客户端留档 (E14/R9: prompt/response 全量留档) 测试。"""

from __future__ import annotations

import json

from darwin_st.creation.llm import OpenAICompatLLM


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LLM_ARCHIVE", str(tmp_path / "llm_calls.jsonl"))
    return OpenAICompatLLM(model="deepseek-v4-flash")


def test_archive_written_on_success(tmp_path, monkeypatch):
    """LLM_ARCHIVE 设置时: 成功调用落一条完整留档 (messages+response)。"""
    import darwin_st.creation.llm as llm_mod

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return b"{}"

        @staticmethod
        def loads_ok():
            return True

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _Resp())
    # json.load(resp) 需要对象可读; 直接 patch json.load 更稳
    monkeypatch.setattr(json, "load",
                        lambda r: {"choices": [{"message": {"content": "回答文本"}}]})
    client = _client(tmp_path, monkeypatch)
    out = client.chat([{"role": "user", "content": "测试"}], temperature=0.3)
    assert out == "回答文本"
    lines = open(tmp_path / "llm_calls.jsonl", encoding="utf-8").readlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["model"] == "deepseek-v4-flash" and rec["temperature"] == 0.3
    assert rec["messages"][0]["content"] == "测试" and rec["response"] == "回答文本"
    assert rec["error"] is None and "ts" in rec


def test_archive_off_by_default(tmp_path, monkeypatch):
    """不设 LLM_ARCHIVE → 不留档 (行为不变)。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("LLM_ARCHIVE", raising=False)
    client = OpenAICompatLLM(model="m")
    client._log_call([{"role": "user", "content": "x"}], 0.7, 100, "r", None)
    assert not (tmp_path / "llm_calls.jsonl").exists()


def test_archive_write_failure_does_not_break_call(tmp_path, monkeypatch):
    """留档写失败 (目录只读) 只告警不炸调用。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("LLM_ARCHIVE", "/nonexistent_dir_xyz/llm.jsonl")
    client = OpenAICompatLLM(model="m")
    client._log_call([], 0.7, 100, "r", None)   # 不抛异常即通过
