import math
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ArgumentUnit, Document, DocumentChunk, MemoryRecord, Session as ChatSession, UserProfile
from .extractor import ExtractionResult
from .memory_embedder import cosine_similarity, embed_texts
from .memory_store import build_fts_query


NEGATIVE_MARKERS = ("不是", "不", "不会", "没有", "无", "并非")
ABSOLUTIST_MARKERS = ("一定", "所有", "绝不", "必须", "总是", "永远")


@dataclass
class PlanningRAG:
    memory_conflicts: list[dict[str, Any]]
    memory_supports: list[dict[str, Any]]
    definition_hits: list[dict[str, Any]]
    counterexample_hits: list[dict[str, Any]]
    revision_hits: list[dict[str, Any]]
    doc_hits: list[dict[str, Any]]
    relevance_summary: dict[str, Any] = field(default_factory=dict)
    profile_hits: list[dict[str, Any]] = field(default_factory=list)
    profile_snapshot: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_negative(text: str) -> bool:
    return any(marker in (text or "") for marker in NEGATIVE_MARKERS)


def _has_overlap(current_claim: str | None, focus_terms: list[str], candidate_text: str) -> bool:
    candidate_text = candidate_text or ""
    if focus_terms and any(term and term in candidate_text for term in focus_terms):
        return True
    if current_claim:
        short = current_claim[:6]
        return bool(short and short in candidate_text)
    return False


def _score_hit(text: str, focus_terms: list[str], base: float = 0.55) -> float:
    score = base
    for term in focus_terms[:3]:
        if term and term in (text or ""):
            score += 0.12
    if any(marker in (text or "") for marker in ABSOLUTIST_MARKERS):
        score += 0.06
    return min(score, 0.99)


def _query_terms(extraction: ExtractionResult) -> list[str]:
    terms = [term for term in extraction.focus_terms if term]
    if extraction.claim:
        terms.append(extraction.claim[:12])
    return list(dict.fromkeys([term for term in terms if term]))[:5]


def _retrieve_document_hits(db: Session, session_id: str, extraction: ExtractionResult, limit: int = 3) -> list[dict[str, Any]]:
    terms = _query_terms(extraction)
    if not terms:
        return []

    rows = (
        db.query(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(DocumentChunk.session_id == session_id)
        .order_by(DocumentChunk.created_at.desc())
        .limit(60)
        .all()
    )
    hits: list[dict[str, Any]] = []
    for chunk, document in rows:
        score = 0.0
        for term in terms:
            if term in chunk.content:
                score += 0.18 if len(term) > 3 else 0.1
        if score <= 0:
            continue
        relation = "support"
        if extraction.flags.get("absolutist"):
            relation = "counterpoint"
        hits.append({
            "source": f"doc_{document.id}#chunk_{chunk.chunk_index}",
            "document_title": document.title,
            "text": chunk.content[:220],
            "relation": relation,
            "score": min(0.99, 0.55 + score),
        })
    hits = sorted(hits, key=lambda item: item["score"], reverse=True)
    dedup: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        if hit["source"] in seen:
            continue
        seen.add(hit["source"])
        dedup.append(hit)
        if len(dedup) >= limit:
            break
    return dedup


def _legacy_build_planning_rag(
    db: Session,
    session_id: str,
    extraction: ExtractionResult,
    exclude_turn_id: str | None = None,
    limit: int = 18,
) -> PlanningRAG:
    rows = (
        db.query(ArgumentUnit)
        .filter(ArgumentUnit.session_id == session_id)
        .order_by(ArgumentUnit.created_at.desc())
        .limit(limit)
        .all()
    )
    if exclude_turn_id:
        rows = [row for row in rows if row.turn_id != exclude_turn_id]

    memory_conflicts: list[dict[str, Any]] = []
    memory_supports: list[dict[str, Any]] = []
    definition_hits: list[dict[str, Any]] = []
    counterexample_hits: list[dict[str, Any]] = []
    revision_hits: list[dict[str, Any]] = []
    doc_hits: list[dict[str, Any]] = _retrieve_document_hits(db, session_id, extraction)

    current_claim = extraction.claim or ""
    focus_terms = extraction.focus_terms
    current_negative = _is_negative(current_claim)
    current_absolute = extraction.flags.get("absolutist", False) or any(
        point.get("type") == "absolute_claim" for point in extraction.attackable_points
    )

    for row in rows:
        raw_schema = row.raw_schema or {}
        past_claim = row.claim or ((raw_schema.get("claim") or {}).get("text") if isinstance(raw_schema.get("claim"), dict) else "")
        if past_claim and _has_overlap(current_claim, focus_terms, past_claim):
            past_negative = _is_negative(past_claim)
            base_hit = {
                "source": f"turn_{row.turn_id}",
                "text": past_claim,
                "score": _score_hit(past_claim, focus_terms),
            }
            if current_claim and current_claim != past_claim:
                revision_hits.append({**base_hit, "relation": "revision"})
            if current_negative != past_negative:
                conflict_hit = {**base_hit, "relation": "conflict"}
                memory_conflicts.append(conflict_hit)
                if current_absolute:
                    counterexample_hits.append({**base_hit, "relation": "counterexample"})
            else:
                memory_supports.append({**base_hit, "relation": "support"})

        for definition in row.definitions or []:
            term = definition.get("term", "") if isinstance(definition, dict) else ""
            definition_text = definition.get("definition", "") if isinstance(definition, dict) else ""
            if term and (term in focus_terms or term in current_claim):
                text = f"{term}: {definition_text}".strip()
                definition_hits.append(
                    {
                        "source": f"turn_{row.turn_id}",
                        "term": term,
                        "text": text,
                        "relation": "definition",
                        "score": _score_hit(text, focus_terms, base=0.6),
                    }
                )

    return _finalize_rag(
        memory_conflicts=memory_conflicts,
        memory_supports=memory_supports,
        definition_hits=definition_hits,
        counterexample_hits=counterexample_hits,
        revision_hits=revision_hits,
        doc_hits=doc_hits,
        profile_hits=[],
        profile_snapshot={},
        summary_override={"retrieval_mode": "legacy_rule"},
    )


def _build_query_text(extraction: ExtractionResult) -> str:
    parts = [
        extraction.claim or "",
        " ".join(extraction.focus_terms[:3]),
        extraction.reasons[0] if extraction.reasons else "",
        extraction.missing_links[0] if extraction.missing_links else "",
    ]
    return " ".join(part for part in parts if part).strip()


def _normalize_forward(values: list[float]) -> list[float]:
    if not values:
        return []
    high = max(values)
    low = min(values)
    if abs(high - low) <= 1e-9:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _normalize_reverse(values: list[float]) -> list[float]:
    if not values:
        return []
    high = max(values)
    low = min(values)
    if abs(high - low) <= 1e-9:
        return [1.0 for _ in values]
    return [(high - value) / (high - low) for value in values]


def _memory_scope_filter(
    db: Session,
    session_id: str,
) -> tuple[str | None, Any]:
    session = db.get(ChatSession, session_id)
    user_id = session.user_id if session else None
    if user_id:
        return user_id, (
            (MemoryRecord.session_id == session_id)
            | ((MemoryRecord.user_id == user_id) & (MemoryRecord.is_evergreen.is_(True)))
        )
    return None, (MemoryRecord.session_id == session_id)


def _search_bm25(
    db: Session,
    session_id: str,
    user_id: str | None,
    query_text: str,
    exclude_turn_id: str | None,
) -> tuple[dict[str, float], int]:
    fts_query = build_fts_query(query_text)
    if not fts_query:
        return {}, 0

    rows = db.execute(
        sql_text(
            """
            SELECT record_id, bm25(memory_record_fts) AS rank
            FROM memory_record_fts
            WHERE memory_record_fts MATCH :fts_query
              AND (
                session_id = :session_id
                OR (:user_id != '' AND user_id = :user_id AND is_evergreen = '1')
              )
              AND NOT (source_type = 'turn' AND source_id = :exclude_turn_id)
            ORDER BY rank ASC
            LIMIT :limit
            """
        ),
        {
            "fts_query": fts_query,
            "session_id": session_id,
            "user_id": user_id or "",
            "exclude_turn_id": exclude_turn_id or "",
            "limit": settings.memory_bm25_top_k,
        },
    ).all()
    if not rows:
        return {}, 0

    ranks = [float(row.rank) for row in rows]
    normalized = _normalize_reverse(ranks)
    return {row.record_id: score for row, score in zip(rows, normalized)}, len(rows)


def _search_vector(
    db: Session,
    session_filter: Any,
    query_text: str,
    exclude_turn_id: str | None,
) -> tuple[dict[str, float], str | None, str | None, int]:
    if not query_text:
        return {}, None, None, 0

    query_embeddings, embedding_source, embedding_error = embed_texts([query_text])
    query_embedding = query_embeddings[0] if query_embeddings else []
    if not query_embedding:
        return {}, embedding_source, embedding_error, 0

    rows = (
        db.query(MemoryRecord)
        .filter(
            session_filter,
            MemoryRecord.embedding.is_not(None),
        )
        .order_by(MemoryRecord.created_at.desc())
        .limit(240)
        .all()
    )
    filtered_rows = [
        row
        for row in rows
        if not (row.source_type == "turn" and row.source_id == exclude_turn_id)
    ]
    if not filtered_rows:
        return {}, embedding_source, embedding_error, 0

    scored = [
        (row.id, cosine_similarity(query_embedding, row.embedding or []))
        for row in filtered_rows
    ]
    scored = sorted(scored, key=lambda item: item[1], reverse=True)[: settings.memory_vector_top_k]
    normalized = _normalize_forward([score for _, score in scored])
    return {record_id: score for (record_id, _), score in zip(scored, normalized)}, embedding_source, embedding_error, len(filtered_rows)


def _decay_factor(record: MemoryRecord) -> float:
    if record.is_evergreen:
        return 1.0
    now = datetime.utcnow()
    age_days = max(0.0, (now - record.created_at).total_seconds() / 86400.0)
    half_life = settings.memory_document_half_life_days if record.source_type == "document" else settings.memory_half_life_days
    return math.exp(-math.log(2) * age_days / max(0.1, half_life))


def _hybrid_score(record: MemoryRecord, vector_score: float, bm25_score: float) -> float:
    if record.scope == "durable":
        return (
            0.50 * vector_score
            + 0.25 * bm25_score
            + 0.15 * max(0.0, min(1.0, record.stability))
            + 0.10 * max(0.0, min(1.0, record.importance))
        )
    return (
        0.45 * vector_score
        + 0.25 * bm25_score
        + 0.20 * _decay_factor(record)
        + 0.10 * max(0.0, min(1.0, record.importance))
    )


def _hybrid_candidates(
    db: Session,
    session_id: str,
    extraction: ExtractionResult,
    exclude_turn_id: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_text = _build_query_text(extraction)
    user_id, session_filter = _memory_scope_filter(db, session_id)
    bm25_scores, bm25_count = _search_bm25(db, session_id, user_id, query_text, exclude_turn_id)
    vector_scores, embedding_source, embedding_error, vector_count = _search_vector(
        db,
        session_filter=session_filter,
        query_text=query_text,
        exclude_turn_id=exclude_turn_id,
    )

    candidate_ids = list(dict.fromkeys(list(bm25_scores.keys()) + list(vector_scores.keys())))
    if not candidate_ids:
        return [], {
            "query_text": query_text,
            "retrieval_mode": "memory_hybrid",
            "bm25_hits": bm25_count,
            "vector_candidates": vector_count,
            "embedding_source": embedding_source,
            "embedding_error": embedding_error,
        }

    rows = db.query(MemoryRecord).filter(MemoryRecord.id.in_(candidate_ids)).all()
    by_id = {row.id: row for row in rows}
    candidates: list[dict[str, Any]] = []
    for record_id in candidate_ids:
        record = by_id.get(record_id)
        if not record:
            continue
        hybrid_score = _hybrid_score(
            record,
            vector_score=vector_scores.get(record_id, 0.0),
            bm25_score=bm25_scores.get(record_id, 0.0),
        )
        decay = _decay_factor(record)
        candidates.append(
            {
                "record": record,
                "vector_score": vector_scores.get(record_id, 0.0),
                "bm25_score": bm25_scores.get(record_id, 0.0),
                "hybrid_score": hybrid_score,
                "final_score": hybrid_score * decay,
                "decay": decay,
            }
        )
    candidates = sorted(candidates, key=lambda item: item["final_score"], reverse=True)
    return candidates, {
        "query_text": query_text,
        "retrieval_mode": "memory_hybrid",
        "bm25_hits": bm25_count,
        "vector_candidates": vector_count,
        "embedding_source": embedding_source,
        "embedding_error": embedding_error,
    }


def _mmr_rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < settings.memory_hybrid_top_k:
        best_idx = 0
        best_score = None
        for idx, candidate in enumerate(remaining):
            redundancy = 0.0
            if candidate["record"].embedding:
                redundancy = max(
                    (
                        cosine_similarity(candidate["record"].embedding or [], prior["record"].embedding or [])
                        for prior in selected
                        if prior["record"].embedding
                    ),
                    default=0.0,
                )
            mmr_score = settings.memory_mmr_lambda * candidate["final_score"] - (1 - settings.memory_mmr_lambda) * redundancy
            if best_score is None or mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score
        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = best_score if best_score is not None else chosen["final_score"]
        selected.append(chosen)
    return selected


def _record_hit(record: MemoryRecord, score: float, relation: str) -> dict[str, Any]:
    payload = {
        "source": f"{record.source_type}_{record.source_id}#memory_{record.chunk_index}",
        "text": record.text[:220],
        "relation": relation,
        "score": round(max(0.0, min(0.99, score)), 4),
    }
    if record.term:
        payload["term"] = record.term
    if record.source_type == "document":
        payload["document_title"] = record.meta.get("document_title")
    return payload


def _normalize_text(text: str) -> str:
    return "".join(text.split())


def _classify_candidates(
    extraction: ExtractionResult,
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    memory_conflicts: list[dict[str, Any]] = []
    memory_supports: list[dict[str, Any]] = []
    definition_hits: list[dict[str, Any]] = []
    counterexample_hits: list[dict[str, Any]] = []
    revision_hits: list[dict[str, Any]] = []
    doc_hits: list[dict[str, Any]] = []

    current_claim = extraction.claim or ""
    current_negative = _is_negative(current_claim)
    current_absolute = extraction.flags.get("absolutist", False) or any(
        point.get("type") == "absolute_claim" for point in extraction.attackable_points
    )

    for candidate in selected:
        record = candidate["record"]
        score = candidate.get("mmr_score", candidate["final_score"])
        if record.kind == "definition":
            if record.term and (record.term in extraction.focus_terms or record.term in current_claim):
                definition_hits.append(_record_hit(record, score, "definition"))
            continue

        if record.source_type == "document":
            relation = "counterpoint" if current_absolute else "support"
            doc_hits.append(_record_hit(record, score, relation))
            if current_absolute:
                counterexample_hits.append(_record_hit(record, score, "counterexample"))
            continue

        if record.kind == "claim":
            past_negative = _is_negative(record.text)
            if current_negative != past_negative:
                conflict = _record_hit(record, score, "conflict")
                memory_conflicts.append(conflict)
                if current_absolute or (record.meta.get("flags") or {}).get("absolutist"):
                    counterexample_hits.append(_record_hit(record, score, "counterexample"))
            else:
                memory_supports.append(_record_hit(record, score, "support"))
            if current_claim and _normalize_text(current_claim) != _normalize_text(record.text):
                revision_hits.append(_record_hit(record, score, "revision"))
            continue

        if record.kind in {"value", "reason"}:
            memory_supports.append(_record_hit(record, score, "support"))

    return (
        memory_conflicts[:3],
        memory_supports[:3],
        definition_hits[:3],
        counterexample_hits[:3],
        revision_hits[:3],
        doc_hits[:3],
    )


def _load_profile_snapshot(db: Session, session_id: str, extraction: ExtractionResult) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = db.get(ChatSession, session_id)
    if not session:
        return {}, []
    profile = db.query(UserProfile).filter(UserProfile.user_id == session.user_id).first()
    if not profile:
        return {}, []

    snapshot = {
        "dialogue_style": profile.dialogue_style,
        "stable_definitions": profile.stable_definitions or {},
        "value_hierarchy": profile.value_hierarchy or [],
        "philosophical_tendency": profile.philosophical_tendency or [],
        "long_term_goals": profile.long_term_goals or [],
        "constraints": profile.constraints or [],
    }
    hits: list[dict[str, Any]] = []
    for term in extraction.focus_terms[:3]:
        definition = snapshot["stable_definitions"].get(term)
        if definition:
            hits.append(
                {
                    "source": f"profile_definition:{term}",
                    "term": term,
                    "text": definition,
                    "relation": "profile_definition",
                    "score": 0.93,
                }
            )
    if snapshot["dialogue_style"] != "balanced_socratic":
        hits.append(
            {
                "source": "profile_dialogue_style",
                "text": snapshot["dialogue_style"],
                "relation": "style",
                "score": 0.8,
            }
        )
    return snapshot, hits


def _finalize_rag(
    memory_conflicts: list[dict[str, Any]],
    memory_supports: list[dict[str, Any]],
    definition_hits: list[dict[str, Any]],
    counterexample_hits: list[dict[str, Any]],
    revision_hits: list[dict[str, Any]],
    doc_hits: list[dict[str, Any]],
    profile_hits: list[dict[str, Any]] | None = None,
    profile_snapshot: dict[str, Any] | None = None,
    summary_override: dict[str, Any] | None = None,
) -> PlanningRAG:
    profile_hits = profile_hits or []
    profile_snapshot = profile_snapshot or {}
    signal_counts = sum(
        1
        for bucket in (
            memory_conflicts,
            definition_hits,
            counterexample_hits,
            revision_hits,
            doc_hits,
            profile_hits,
        )
        if bucket
    )

    top_signal = None
    if memory_conflicts:
        top_signal = "conflict"
    elif definition_hits:
        top_signal = "definition"
    elif counterexample_hits:
        top_signal = "counterexample"
    elif revision_hits:
        top_signal = "revision"
    elif doc_hits:
        top_signal = "document"
    elif profile_hits:
        top_signal = "profile"
    elif memory_supports:
        top_signal = "support"

    summary = {
        "has_conflict": bool(memory_conflicts),
        "has_support": bool(memory_supports),
        "has_prior_definition": bool(definition_hits) or any(hit.get("relation") == "profile_definition" for hit in profile_hits),
        "has_counterexample": bool(counterexample_hits),
        "has_revision": bool(revision_hits),
        "has_doc_hits": bool(doc_hits),
        "has_profile": bool(profile_snapshot),
        "top_signal": top_signal,
        "is_complex": signal_counts >= 2,
        "signal_count": signal_counts,
    }
    if summary_override:
        summary.update(summary_override)

    return PlanningRAG(
        memory_conflicts=memory_conflicts,
        memory_supports=memory_supports,
        definition_hits=definition_hits,
        counterexample_hits=counterexample_hits,
        revision_hits=revision_hits,
        doc_hits=doc_hits,
        relevance_summary=summary,
        profile_hits=profile_hits,
        profile_snapshot=profile_snapshot,
    )


def build_planning_rag(
    db: Session,
    session_id: str,
    extraction: ExtractionResult,
    exclude_turn_id: str | None = None,
    limit: int = 18,
) -> PlanningRAG:
    profile_snapshot, profile_hits = _load_profile_snapshot(db, session_id, extraction)
    candidates, summary = _hybrid_candidates(db, session_id, extraction, exclude_turn_id)
    if not candidates:
        legacy = _legacy_build_planning_rag(db, session_id, extraction, exclude_turn_id=exclude_turn_id, limit=limit)
        legacy.profile_snapshot = profile_snapshot
        legacy.profile_hits = profile_hits
        legacy.relevance_summary["has_profile"] = bool(profile_snapshot)
        legacy.relevance_summary["has_prior_definition"] = legacy.relevance_summary.get("has_prior_definition", False) or any(
            hit.get("relation") == "profile_definition" for hit in profile_hits
        )
        return legacy

    selected = _mmr_rank(candidates)
    for item in selected:
        record = item["record"]
        record.access_count = (record.access_count or 0) + 1
        record.last_accessed_at = datetime.utcnow()

    (
        memory_conflicts,
        memory_supports,
        definition_hits,
        counterexample_hits,
        revision_hits,
        doc_hits,
    ) = _classify_candidates(extraction, selected)

    if not any((memory_conflicts, memory_supports, definition_hits, counterexample_hits, revision_hits, doc_hits)):
        legacy = _legacy_build_planning_rag(db, session_id, extraction, exclude_turn_id=exclude_turn_id, limit=limit)
        legacy.relevance_summary.update(summary)
        legacy.relevance_summary["fallback_used"] = True
        legacy.relevance_summary["has_profile"] = bool(profile_snapshot)
        legacy.relevance_summary["has_prior_definition"] = legacy.relevance_summary.get("has_prior_definition", False) or any(
            hit.get("relation") == "profile_definition" for hit in profile_hits
        )
        legacy.profile_snapshot = profile_snapshot
        legacy.profile_hits = profile_hits
        return legacy

    summary.update(
        {
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_sources": [item["record"].id for item in selected],
        }
    )
    return _finalize_rag(
        memory_conflicts=memory_conflicts,
        memory_supports=memory_supports,
        definition_hits=definition_hits,
        counterexample_hits=counterexample_hits,
        revision_hits=revision_hits,
        doc_hits=doc_hits,
        profile_hits=profile_hits,
        profile_snapshot=profile_snapshot,
        summary_override=summary,
    )
