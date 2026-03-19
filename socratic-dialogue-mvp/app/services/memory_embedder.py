import hashlib
import math
import re
from typing import Iterable

import httpx

from app.config import settings


ASCII_WORD_RE = re.compile(r"[a-z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
AUTO_EMBEDDING_DISABLED_ERROR: str | None = None


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return values
    return [value / norm for value in values]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0
    return sum(a * b for a, b in zip(left_values, right_values))


def _embedding_terms(text: str) -> list[str]:
    lowered = (text or "").lower()
    words = ASCII_WORD_RE.findall(lowered)
    cjk_chars = CJK_RE.findall(text or "")
    cjk_bigrams = ["".join(cjk_chars[idx: idx + 2]) for idx in range(max(0, len(cjk_chars) - 1))]
    terms = words + cjk_chars + cjk_bigrams
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        if term and term not in seen:
            ordered.append(term)
            seen.add(term)
    return ordered


def _hash_embed_one(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    terms = _embedding_terms(text)
    if not terms:
        return vector
    for term in terms:
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        primary = int.from_bytes(digest[:4], "big") % dimensions
        secondary = int.from_bytes(digest[4:8], "big") % dimensions
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        weight = 1.25 if len(term) > 1 else 0.8
        vector[primary] += sign * weight
        vector[secondary] += (weight * 0.5) * (-sign)
    return _normalize(vector)


def _hash_embed(texts: list[str]) -> tuple[list[list[float]], str, str | None]:
    dimensions = max(32, settings.memory_embedding_dimensions)
    vectors = [_hash_embed_one(text, dimensions) for text in texts]
    return vectors, "hash_local", None


def _openai_embed(texts: list[str]) -> tuple[list[list[float]], str, str | None]:
    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY is empty")

    body = {
        "model": settings.memory_embedding_model,
        "input": texts,
        "encoding_format": "float",
        "dimensions": settings.memory_embedding_dimensions,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.llm_base_url.rstrip('/')}/embeddings"

    with httpx.Client(timeout=settings.memory_embedding_timeout_seconds) as client:
        response = client.post(url, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()

    data = payload.get("data", [])
    vectors = [item.get("embedding", []) for item in data]
    if len(vectors) != len(texts):
        raise ValueError("embedding response size mismatch")
    return vectors, settings.memory_embedding_model, None


def embed_texts(texts: list[str]) -> tuple[list[list[float]], str, str | None]:
    global AUTO_EMBEDDING_DISABLED_ERROR
    mode = settings.memory_embedding_mode.lower()
    if mode == "hash":
        return _hash_embed(texts)

    if mode in {"auto", "openai"}:
        if mode == "auto" and AUTO_EMBEDDING_DISABLED_ERROR:
            vectors, _, _ = _hash_embed(texts)
            return vectors, "hash_fallback", AUTO_EMBEDDING_DISABLED_ERROR
        try:
            return _openai_embed(texts)
        except Exception as exc:
            if mode == "auto":
                AUTO_EMBEDDING_DISABLED_ERROR = str(exc)
            if mode == "openai":
                return _hash_embed(texts)[0], "hash_fallback", str(exc)
            vectors, _, _ = _hash_embed(texts)
            return vectors, "hash_fallback", str(exc)

    vectors, source, error = _hash_embed(texts)
    fallback_error = None if mode == "hash_local" else f"unsupported MEMORY_EMBEDDING_MODE={settings.memory_embedding_mode}"
    return vectors, source, fallback_error or error
