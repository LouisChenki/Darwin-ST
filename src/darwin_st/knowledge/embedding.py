"""文本嵌入 (Text Embedding) —— MAC 向量检索的后端, 可插拔。

研究 (docs/TIER2_RESEARCH_FINDINGS.md §3): MAC 阶段用语义向量召回, 处理表达模糊
("长程依赖"/"long-range"/"远距离时序"向量都接近)。**只嵌 abstract_function+
preconditions(embedding_text), 绝不含表面描述/域名**, 否则退化为同域语义相似。

两个后端 (同一接口):
  - HashEmbedder: 确定性 token-hash 向量, 零依赖零下载, 本地/测试用。逻辑可验但语义弱。
  - SentenceTransformerEmbedder: 真实语义 embedding(需 sentence-transformers + 模型),
    服务器用。语义准确。

接口: embed(texts) -> np.ndarray [n, dim]; 行已 L2 归一化(便于余弦=点积)。
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

__all__ = ["Embedder", "HashEmbedder", "SentenceTransformerEmbedder", "cosine_sim_matrix",
           "ST_INSTALL_HINT", "sentence_transformers_available", "warn_hash_fallback"]


class Embedder(Protocol):
    """嵌入器接口。"""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """返回 [len(texts), dim] 的 L2 归一化向量矩阵。"""
        ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return m / norm


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """行归一化后, 余弦相似 = 点积。返回 [len(a), len(b)]。"""
    return _l2_normalize(a) @ _l2_normalize(b).T


# ---------------------------------------------------------------------------
# HashEmbedder: 确定性 token-hash 向量 (零依赖, 本地/测试)
# ---------------------------------------------------------------------------

_STOP = {"的", "了", "在", "是", "和", "与", "the", "a", "of", "to", "and", "in", "on"}


def _tokenize(text: str) -> list[str]:
    """中英混合: 英文按词, 中文 2-gram。"""
    text = text.lower()
    en = [w for w in re.findall(r"[a-z_]+", text) if w not in _STOP and len(w) > 1]
    toks = list(en)
    for seg in re.findall(r"[一-鿿]+", text):
        for i in range(max(len(seg) - 1, 1)):
            toks.append(seg[i:i + 2])
    return toks


class HashEmbedder:
    """token 哈希到固定维 bag-of-tokens 向量。确定性, 零依赖。

    语义弱(同义词不共享维度), 但用于本地验证检索【逻辑】足够; 真实语义换 SentenceTransformer。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        return _l2_normalize(m)


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder: 真实语义 embedding (服务器)
# ---------------------------------------------------------------------------

# 安装提示统一口径 (pyproject [dependency-groups].semantic), 各处警告/报错都引用它
ST_INSTALL_HINT = "uv sync --group semantic  (pip 等价: pip install 'sentence-transformers>=2.7.0')"


def sentence_transformers_available() -> bool:
    """sentence-transformers 是否已安装。只查 module spec, 不 import 不下载模型。"""
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None


def warn_hash_fallback(context: str = "") -> None:
    """退回 HashEmbedder 时打一次醒目警告: 检索质量显著退化 + 安装提示。

    背景 (docs/SERVER_VALIDATION.md): Hash 嵌入下检索语义分仅 0.00-0.30, 真实语义
    0.54-0.70 —— 静默退回等于 MAC 向量路径形同虚设, 必须让用户看见。
    """
    tag = f" ({context})" if context else ""
    print("!" * 72)
    print(f"[警告] sentence-transformers 不可用{tag}, 退回 HashEmbedder —— "
          "跨域检索质量将显著退化 (语义分 0.54-0.70 → 0.00-0.30)!")
    print(f"       生产检索请装真语义嵌入: {ST_INSTALL_HINT}")
    print("!" * 72, flush=True)


class SentenceTransformerEmbedder:
    """基于 sentence-transformers 的真实语义嵌入 (延迟加载模型)。

    默认多语言模型(中英都行)。需装 sentence-transformers (`uv sync --group semantic`)。
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    f"sentence-transformers 未安装, 无法加载真实语义嵌入模型。安装: {ST_INSTALL_HINT}"
                ) from e

            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        self._ensure()
        return self._dim  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)  # type: ignore[union-attr]
        return emb.astype(np.float32)
