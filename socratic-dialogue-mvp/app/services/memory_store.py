import re
from collections.abc import Iterable

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document, DocumentChunk, MemoryRecord, Session as ChatSession, Turn
from .extractor import ExtractionResult
from .memory_embedder import embed_texts


ASCII_WORD_RE = re.compile(r"[a-z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EXPLICIT_MEMORY_MARKERS = ("记住", "记得", "别忘", "请保存", "长期记忆")
DIRECT_STYLE_MARKERS = ("直接一点", "直接些", "尖锐一点", "别安慰", "不要安慰", "直接质疑", "挑战我")
SOFT_STYLE_MARKERS = ("温和一点", "别太尖锐", "先别反驳", "慢一点")
PHILOSOPHY_SCHOOLS = ("存在主义", "功利主义", "儒家", "道家", "斯多葛", "康德", "尼采", "实用主义")
GOAL_MARKERS = ("我的目标是", "我想提升", "我想训练", "我希望学会", "我想弄清")


def _chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_chars - overlap_chars)
    while start < len(cleaned):
        piece = cleaned[start: start + chunk_chars].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def build_search_terms(text: str, extra_terms: Iterable[str] | None = None) -> list[str]:
    lowered = (text or "").lower()
    ascii_terms = ASCII_WORD_RE.findall(lowered)
    cjk_chars = CJK_RE.findall(text or "")
    cjk_bigrams = ["".join(cjk_chars[idx: idx + 2]) for idx in range(max(0, len(cjk_chars) - 1))]
    terms = ascii_terms + cjk_chars + cjk_bigrams + [term.strip().lower() for term in (extra_terms or []) if term]
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        if not term or term in seen:
            continue
        ordered.append(term)
        seen.add(term)
    return ordered[:72]


def build_search_text(text: str, extra_terms: Iterable[str] | None = None) -> str:
    return " ".join(build_search_terms(text, extra_terms=extra_terms))


def build_fts_query(text: str, extra_terms: Iterable[str] | None = None) -> str:
    terms = build_search_terms(text, extra_terms=extra_terms)
    if not terms:
        return ""
    quoted = ['"' + term.replace('"', '""') + '"' for term in terms[:12]]
    return " OR ".join(quoted)


def _replace_source_records(db: Session, session_id: str | None, source_type: str, source_id: str) -> None:
    existing_ids = [
        row[0]
        for row in db.query(MemoryRecord.id)
        .filter(
            MemoryRecord.session_id == session_id,
            MemoryRecord.source_type == source_type,
            MemoryRecord.source_id == source_id,
            MemoryRecord.scope == "session",
        )
        .all()
    ]
    if not existing_ids:
        return
    for record_id in existing_ids:
        db.execute(
            sql_text("DELETE FROM memory_record_fts WHERE record_id = :record_id"),
            {"record_id": record_id},
        )
    db.query(MemoryRecord).filter(MemoryRecord.id.in_(existing_ids)).delete(synchronize_session=False)


def _insert_fts_rows(db: Session, records: list[MemoryRecord]) -> None:
    for record in records:
        db.execute(
            sql_text(
                """
                INSERT INTO memory_record_fts (
                    record_id,
                    session_id,
                    user_id,
                    source_type,
                    source_id,
                    kind,
                    is_evergreen,
                    search_text
                ) VALUES (
                    :record_id,
                    :session_id,
                    :user_id,
                    :source_type,
                    :source_id,
                    :kind,
                    :is_evergreen,
                    :search_text
                )
                """
            ),
            {
                "record_id": record.id,
                "session_id": record.session_id or "",
                "user_id": record.user_id or "",
                "source_type": record.source_type,
                "source_id": record.source_id,
                "kind": record.kind,
                "is_evergreen": "1" if record.is_evergreen else "0",
                "search_text": record.search_text,
            },
        )


def _build_turn_payloads(extraction: ExtractionResult, raw_text: str) -> list[dict]:
    explicit_memory = any(marker in raw_text for marker in EXPLICIT_MEMORY_MARKERS)
    payloads: list[dict] = []
    if extraction.claim:
        payloads.append(
            {
                "kind": "claim",
                "term": extraction.focus_terms[0] if extraction.focus_terms else None,
                "profile_key": None,
                "text": extraction.claim,
                "importance": 0.68,
                "confidence": extraction.confidence,
                "stability": 0.62 if explicit_memory else 0.4,
                "meta": {"flags": extraction.flags, "attackable_points": extraction.attackable_points[:2], "explicit_memory": explicit_memory},
            }
        )
    for definition in extraction.definitions[:3]:
        term = definition.get("term", "") if isinstance(definition, dict) else ""
        definition_text = definition.get("definition", "") if isinstance(definition, dict) else ""
        if not term:
            continue
        text = f"{term}：{definition_text}".strip("：")
        payloads.append(
            {
                "kind": "definition",
                "term": term,
                "profile_key": "stable_definitions",
                "text": text,
                "importance": 0.9,
                "confidence": max(0.72, extraction.confidence),
                "stability": 0.94 if explicit_memory else 0.82,
                "meta": {"term": term, "explicit_memory": explicit_memory},
            }
        )
    for premise in extraction.value_premises[:2]:
        payloads.append(
            {
                "kind": "value",
                "term": extraction.focus_terms[0] if extraction.focus_terms else None,
                "profile_key": "value_hierarchy",
                "text": premise,
                "importance": 0.82,
                "confidence": extraction.confidence,
                "stability": 0.88 if explicit_memory else 0.74,
                "meta": {"focus_terms": extraction.focus_terms[:3], "explicit_memory": explicit_memory},
            }
        )
    for reason in extraction.reasons[:2]:
        payloads.append(
            {
                "kind": "reason",
                "term": extraction.focus_terms[0] if extraction.focus_terms else None,
                "profile_key": None,
                "text": reason,
                "importance": 0.62,
                "confidence": extraction.confidence,
                "stability": 0.34,
                "meta": {"explicit_memory": explicit_memory},
            }
        )

    if any(marker in raw_text for marker in DIRECT_STYLE_MARKERS):
        payloads.append(
            {
                "kind": "preference",
                "term": "dialogue_style",
                "profile_key": "dialogue_style",
                "text": "偏好更直接、更有挑战性的苏格拉底式追问。",
                "importance": 0.92,
                "confidence": 0.95,
                "stability": 0.95,
                "meta": {"style_key": "direct_challenge", "constraint": "避免安慰式回复", "explicit_memory": True},
            }
        )
    if any(marker in raw_text for marker in SOFT_STYLE_MARKERS):
        payloads.append(
            {
                "kind": "preference",
                "term": "dialogue_style",
                "profile_key": "dialogue_style",
                "text": "偏好更温和、循序渐进的追问方式。",
                "importance": 0.85,
                "confidence": 0.9,
                "stability": 0.9,
                "meta": {"style_key": "gentle_probe", "constraint": "避免过强对抗", "explicit_memory": True},
            }
        )
    for school in PHILOSOPHY_SCHOOLS:
        if school in raw_text:
            payloads.append(
                {
                    "kind": "philosophy",
                    "term": school,
                    "profile_key": "philosophical_tendency",
                    "text": f"用户明确提到自己倾向或正在讨论{school}视角。",
                    "importance": 0.76,
                    "confidence": 0.88,
                    "stability": 0.78,
                    "meta": {"school": school, "explicit_memory": explicit_memory},
                }
            )
    for marker in GOAL_MARKERS:
        if marker in raw_text:
            payloads.append(
                {
                    "kind": "goal",
                    "term": "goal",
                    "profile_key": "long_term_goals",
                    "text": raw_text.strip(),
                    "importance": 0.86,
                    "confidence": 0.84,
                    "stability": 0.82,
                    "meta": {"explicit_memory": explicit_memory},
                }
            )
            break
    return payloads


def _materialize_records(
    db: Session,
    *,
    session_id: str | None,
    user_id: str | None,
    source_type: str,
    source_id: str,
    payloads: list[dict],
    extra_terms: list[str] | None = None,
    scope: str = "session",
    status: str = "active",
    is_evergreen: bool = False,
) -> list[MemoryRecord]:
    if not payloads:
        return []
    expanded: list[dict] = []
    for payload in payloads:
        chunks = _chunk_text(
            payload["text"],
            chunk_chars=settings.memory_chunk_chars,
            overlap_chars=settings.memory_chunk_overlap_chars,
        ) or [payload["text"]]
        for index, chunk in enumerate(chunks, start=1):
            terms = list(extra_terms or [])
            if payload.get("term"):
                terms.append(payload["term"])
            expanded.append(
                {
                    **payload,
                    "chunk_index": index,
                    "text": chunk,
                    "search_text": build_search_text(chunk, extra_terms=terms),
                }
            )

    embeddings, embedding_source, embedding_error = embed_texts([item["text"] for item in expanded])
    records: list[MemoryRecord] = []
    for payload, embedding in zip(expanded, embeddings):
        record = MemoryRecord(
            user_id=user_id,
            session_id=session_id,
            source_type=source_type,
            source_id=source_id,
            scope=scope,
            status=status,
            chunk_index=payload["chunk_index"],
            kind=payload["kind"],
            term=payload.get("term"),
            profile_key=payload.get("profile_key"),
            origin_memory_id=payload.get("origin_memory_id"),
            text=payload["text"],
            search_text=payload["search_text"],
            importance=payload["importance"],
            confidence=payload["confidence"],
            stability=payload["stability"],
            is_evergreen=is_evergreen,
            embedding=embedding,
            embedding_model=settings.memory_embedding_model,
            embedding_source=embedding_source,
            last_confirmed_at=payload.get("last_confirmed_at"),
            promoted_at=payload.get("promoted_at"),
            meta={**payload["meta"], "embedding_error": embedding_error},
        )
        db.add(record)
        records.append(record)

    db.flush()
    _insert_fts_rows(db, records)
    return records


def capture_turn_memories(
    db: Session,
    session: ChatSession,
    turn: Turn,
    extraction: ExtractionResult,
    raw_text: str,
) -> list[MemoryRecord]:
    payloads = _build_turn_payloads(extraction, raw_text=raw_text)
    if not payloads:
        return []

    _replace_source_records(db, session.id, "turn", turn.id)
    return _materialize_records(
        db,
        session_id=session.id,
        user_id=session.user_id,
        source_type="turn",
        source_id=turn.id,
        payloads=payloads,
        extra_terms=extraction.focus_terms,
        scope="session",
        status="active",
        is_evergreen=False,
    )


def capture_document_memories(
    db: Session,
    session: ChatSession,
    document: Document,
    document_chunks: list[DocumentChunk],
) -> list[MemoryRecord]:
    if not document_chunks:
        return []

    _replace_source_records(db, session.id, "document", document.id)
    payloads = [
        {
            "kind": "document",
            "term": document.title,
            "profile_key": None,
            "text": chunk.content,
            "importance": 0.82,
            "confidence": 0.95,
            "stability": 0.72,
            "meta": {
                "document_id": document.id,
                "document_title": document.title,
                "chunk_index": chunk.chunk_index,
            },
        }
        for chunk in document_chunks
    ]
    return _materialize_records(
        db,
        session_id=session.id,
        user_id=session.user_id,
        source_type="document",
        source_id=document.id,
        payloads=payloads,
        extra_terms=[document.title] if document.title else None,
        scope="session",
        status="active",
        is_evergreen=False,
    )
