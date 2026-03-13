from dataclasses import dataclass, asdict
from typing import Any

from sqlalchemy.orm import Session

from app.models import ArgumentUnit, DocumentChunk, Document
from .extractor import ExtractionResult


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
    relevance_summary: dict[str, Any]

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


def build_planning_rag(
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

    memory_conflicts = sorted(memory_conflicts, key=lambda item: item["score"], reverse=True)[:3]
    memory_supports = sorted(memory_supports, key=lambda item: item["score"], reverse=True)[:3]
    definition_hits = sorted(definition_hits, key=lambda item: item["score"], reverse=True)[:3]
    counterexample_hits = sorted(counterexample_hits, key=lambda item: item["score"], reverse=True)[:3]
    revision_hits = sorted(revision_hits, key=lambda item: item["score"], reverse=True)[:3]

    signal_counts = sum(
        1
        for bucket in (
            memory_conflicts,
            definition_hits,
            counterexample_hits,
            revision_hits,
            doc_hits,
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
    elif memory_supports:
        top_signal = "support"

    return PlanningRAG(
        memory_conflicts=memory_conflicts,
        memory_supports=memory_supports,
        definition_hits=definition_hits,
        counterexample_hits=counterexample_hits,
        revision_hits=revision_hits,
        doc_hits=doc_hits,
        relevance_summary={
            "has_conflict": bool(memory_conflicts),
            "has_support": bool(memory_supports),
            "has_prior_definition": bool(definition_hits),
            "has_counterexample": bool(counterexample_hits),
            "has_revision": bool(revision_hits),
            "has_doc_hits": bool(doc_hits),
            "top_signal": top_signal,
            "is_complex": signal_counts >= 2,
            "signal_count": signal_counts,
        },
    )
