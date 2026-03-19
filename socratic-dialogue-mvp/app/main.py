from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from .config import settings
from .database import Base, engine, get_db, ensure_memory_fts, ensure_memory_schema
from .models import User, Session as ChatSession, Turn, ArgumentUnit, StateTransition, QuestionPlan, Summary, Metric, Document, DocumentChunk, MemoryRecord, UserProfile
from .schemas import CreateSessionRequest, SessionResponse, ChatTurnRequest, ChatTurnResponse, SummaryConfirmRequest, CreateDocumentRequest, DocumentResponse, DocumentChunkResponse
from .services.extractor import extract_structure
from .services.state_machine import decide_next
from .services.retrieval import build_planning_rag
from .services.questioning import generate_response
from .services.documents import persist_document
from .services.memory_store import capture_document_memories, capture_turn_memories
from .services.memory_flush import build_flush_preview, flush_session_memory

app = FastAPI(title=settings.app_name)
WEB_INDEX = Path(__file__).parent / "web" / "index.html"
DEV_INDEX = Path(__file__).parent / "web" / "dev.html"


@app.on_event("startup")
def startup_event() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_memory_schema()
    ensure_memory_fts()


@app.get("/")
def web_home():
    if WEB_INDEX.exists():
        return FileResponse(WEB_INDEX)
    raise HTTPException(status_code=404, detail="WEB_UI_NOT_FOUND")


@app.get("/dev")
def web_dev():
    if DEV_INDEX.exists():
        return FileResponse(DEV_INDEX)
    raise HTTPException(status_code=404, detail="DEV_UI_NOT_FOUND")


@app.get("/api/v1/debug/config")
def debug_config():
    return {
        "extractor_mode": settings.extractor_mode,
        "llm_api_key_loaded": bool(settings.llm_api_key),
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "generation_mode": settings.generation_mode,
        "generation_model": settings.generation_model,
        "planner_mode": settings.planner_mode,
        "planner_model": settings.planner_model,
        "memory_embedding_mode": settings.memory_embedding_mode,
        "memory_embedding_model": settings.memory_embedding_model,
        "memory_embedding_dimensions": settings.memory_embedding_dimensions,
    }


def _get_or_create_user(db: Session, external_id: str) -> User:
    user = db.scalar(select(User).where(User.external_id == external_id))
    if user:
        return user
    user = User(external_id=external_id)
    db.add(user)
    db.flush()
    return user


def _serialize_memory_record(record: MemoryRecord, include_text: bool = False) -> dict:
    payload = {
        "id": record.id,
        "scope": record.scope,
        "status": record.status,
        "source_type": record.source_type,
        "source_id": record.source_id,
        "kind": record.kind,
        "term": record.term,
        "profile_key": record.profile_key,
        "origin_memory_id": record.origin_memory_id,
        "score_hints": {
            "importance": record.importance,
            "confidence": record.confidence,
            "stability": record.stability,
        },
        "is_evergreen": record.is_evergreen,
        "embedding_source": record.embedding_source,
        "created_at": record.created_at,
        "last_confirmed_at": record.last_confirmed_at,
        "promoted_at": record.promoted_at,
        "meta": record.meta,
    }
    if include_text:
        payload["text"] = record.text
    return payload


def _serialize_profile(profile: UserProfile | None) -> dict | None:
    if not profile:
        return None
    return {
        "dialogue_style": profile.dialogue_style,
        "stable_definitions": profile.stable_definitions,
        "value_hierarchy": profile.value_hierarchy,
        "philosophical_tendency": profile.philosophical_tendency,
        "long_term_goals": profile.long_term_goals,
        "constraints": profile.constraints,
        "updated_at": profile.updated_at,
        "source_memory_ids": profile.source_memory_ids,
    }


def _normalize_profile_snapshot(snapshot: dict | None) -> dict | None:
    if not snapshot:
        return None
    return {
        "dialogue_style": snapshot.get("dialogue_style", "balanced_socratic"),
        "stable_definitions": snapshot.get("stable_definitions") or {},
        "value_hierarchy": snapshot.get("value_hierarchy") or [],
        "philosophical_tendency": snapshot.get("philosophical_tendency") or [],
        "long_term_goals": snapshot.get("long_term_goals") or [],
        "constraints": snapshot.get("constraints") or [],
        "source_memory_ids": snapshot.get("source_memory_ids") or [],
        "updated_at": snapshot.get("updated_at"),
    }


def _serialize_flush_candidate(entry: dict) -> dict:
    return {
        **_serialize_memory_record(entry["record"], include_text=True),
        "decision": entry["decision"],
        "reason_code": entry["reason_code"],
        "reason": entry["reason"],
        "repeat_count": entry["repeat_count"],
        "matched_durable_id": entry["matched_durable_id"],
    }


def _memory_debug_payload(db: Session, session: ChatSession, session_limit: int = 8, durable_limit: int = 8) -> dict:
    session_memory_records = (
        db.query(MemoryRecord)
        .filter(MemoryRecord.session_id == session.id, MemoryRecord.scope == "session")
        .order_by(MemoryRecord.created_at.desc())
        .limit(session_limit)
        .all()
    )
    durable_memory_records = (
        db.query(MemoryRecord)
        .filter(MemoryRecord.user_id == session.user_id, MemoryRecord.scope == "durable")
        .order_by(MemoryRecord.created_at.desc())
        .limit(durable_limit)
        .all()
    )
    profile = db.query(UserProfile).filter(UserProfile.user_id == session.user_id).first()
    return {
        "session_record_count": db.query(MemoryRecord).filter(MemoryRecord.session_id == session.id, MemoryRecord.scope == "session").count(),
        "durable_record_count": db.query(MemoryRecord).filter(MemoryRecord.user_id == session.user_id, MemoryRecord.scope == "durable").count(),
        "recent_session_records": [_serialize_memory_record(record) for record in session_memory_records],
        "recent_durable_records": [_serialize_memory_record(record, include_text=True) for record in durable_memory_records],
        "profile": _serialize_profile(profile),
    }


@app.post("/api/v1/sessions", response_model=SessionResponse)
def create_session(payload: CreateSessionRequest, db: Session = Depends(get_db)):
    user = _get_or_create_user(db, payload.user_id)
    session = ChatSession(user_id=user.id, title=payload.title, current_stage="S0", status="active")
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse(session_id=session.id, current_stage=session.current_stage, status=session.status)


@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
    return SessionResponse(session_id=session.id, current_stage=session.current_stage, status=session.status)


@app.get("/api/v1/sessions/{session_id}/debug-snapshot")
def get_debug_snapshot(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")

    turns = (
        db.query(Turn)
        .filter(Turn.session_id == session_id)
        .order_by(Turn.turn_index.desc())
        .limit(12)
        .all()
    )
    latest_argument = (
        db.query(ArgumentUnit)
        .filter(ArgumentUnit.session_id == session_id)
        .order_by(ArgumentUnit.created_at.desc())
        .first()
    )
    latest_transition = (
        db.query(StateTransition)
        .filter(StateTransition.session_id == session_id)
        .order_by(StateTransition.created_at.desc())
        .first()
    )
    latest_plan = (
        db.query(QuestionPlan)
        .filter(QuestionPlan.session_id == session_id)
        .order_by(QuestionPlan.created_at.desc())
        .first()
    )
    latest_summary = (
        db.query(Summary)
        .filter(Summary.session_id == session_id)
        .order_by(Summary.created_at.desc())
        .first()
    )
    metric = db.scalar(select(Metric).where(Metric.session_id == session_id))
    documents = (
        db.query(Document)
        .filter(Document.session_id == session_id)
        .order_by(Document.created_at.desc())
        .limit(6)
        .all()
    )
    memory_payload = _memory_debug_payload(db, session)

    return {
        "session": {
            "id": session.id,
            "status": session.status,
            "current_stage": session.current_stage,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        },
        "latest_argument_unit": None if not latest_argument else {
            "id": latest_argument.id,
            "turn_id": latest_argument.turn_id,
            "claim": latest_argument.claim,
            "reasons": latest_argument.reasons,
            "evidence": latest_argument.evidence,
            "value_premises": latest_argument.value_premises,
            "definitions": latest_argument.definitions,
            "flags": latest_argument.flags,
            "confidence": latest_argument.confidence,
            "raw_schema": latest_argument.raw_schema,
            "created_at": latest_argument.created_at,
        },
        "latest_transition": None if not latest_transition else {
            "from_stage": latest_transition.from_stage,
            "to_stage": latest_transition.to_stage,
            "trigger_type": latest_transition.trigger_type,
            "trigger_detail": latest_transition.trigger_detail,
            "at_turn_index": latest_transition.at_turn_index,
            "created_at": latest_transition.created_at,
        },
        "latest_question_plan": None if not latest_plan else {
            "current_stage": latest_plan.current_stage,
            "weak_point": latest_plan.weak_point,
            "question_intent": latest_plan.question_intent,
            "constraints": latest_plan.constraints,
            "created_at": latest_plan.created_at,
        },
        "latest_summary": None if not latest_summary else {
            "round_range": latest_summary.round_range,
            "summary_text": latest_summary.summary_text,
            "user_confirmed": latest_summary.user_confirmed,
            "created_at": latest_summary.created_at,
        },
        "documents": [
            {
                "id": document.id,
                "title": document.title,
                "source_type": document.source_type,
                "chunk_count": db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).count(),
                "created_at": document.created_at,
            }
            for document in documents
        ],
        "metrics": {
            "consistency_score": 0.0 if not metric else metric.consistency_score,
            "self_revision_count": 0 if not metric else metric.self_revision_count,
            "question_repeat_rate": 0.0 if not metric else metric.question_repeat_rate,
            "premise_explicit_count": 0 if not metric else metric.premise_explicit_count,
        },
        "memory": memory_payload,
        "recent_turns": [
            {
                "turn_index": turn.turn_index,
                "role": turn.role,
                "content": turn.content,
                "question_intent": turn.question_intent,
                "created_at": turn.created_at,
            }
            for turn in reversed(turns)
        ],
    }


@app.get("/api/v1/sessions/{session_id}/memory/debug")
def get_memory_debug(
    session_id: str,
    query: str | None = Query(default=None, min_length=1, max_length=2000),
    session_limit: int = Query(default=12, ge=1, le=50),
    durable_limit: int = Query(default=12, ge=1, le=50),
    db: Session = Depends(get_db),
):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")

    payload = {
        "session": {
            "id": session.id,
            "status": session.status,
            "current_stage": session.current_stage,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        },
        "memory": _memory_debug_payload(db, session, session_limit=session_limit, durable_limit=durable_limit),
    }

    if query:
        history_turns = (
            db.query(Turn)
            .filter(Turn.session_id == session_id)
            .order_by(Turn.turn_index.asc())
            .all()
        )
        history_user_texts = [turn.content for turn in history_turns if turn.role == "user"]
        extracted = extract_structure(query, history_user_texts)
        planning_rag = build_planning_rag(
            db=db,
            session_id=session_id,
            extraction=extracted,
            exclude_turn_id=None,
        )
        payload["query_debug"] = {
            "query": query,
            "extraction": {
                "claim": extracted.claim,
                "reasons": extracted.reasons,
                "definitions": extracted.definitions,
                "value_premises": extracted.value_premises,
                "focus_terms": extracted.focus_terms,
                "attackable_points": extracted.attackable_points,
                "missing_links": extracted.missing_links,
                "flags": extracted.flags,
                "confidence": extracted.confidence,
                "extractor_source": extracted.raw_schema.get("_meta", {}).get("source"),
                "extractor_error": extracted.raw_schema.get("_meta", {}).get("error"),
            },
            "retrieval_summary": planning_rag.relevance_summary,
            "profile_snapshot": planning_rag.profile_snapshot,
            "profile_hits": planning_rag.profile_hits,
            "definition_hits": planning_rag.definition_hits,
            "memory_conflicts": planning_rag.memory_conflicts,
            "memory_supports": planning_rag.memory_supports,
            "revision_hits": planning_rag.revision_hits,
            "counterexample_hits": planning_rag.counterexample_hits,
            "doc_hits": planning_rag.doc_hits,
        }

    return payload


@app.get("/api/v1/sessions/{session_id}/memory/flush-preview")
def get_memory_flush_preview(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")

    preview = build_flush_preview(db, session_id)
    return {
        "session": {
            "id": session.id,
            "status": session.status,
            "current_stage": session.current_stage,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        },
        "flush_window": {
            "lookback_turns": settings.memory_flush_lookback_turns,
            "candidate_count": preview["candidate_count"],
            "would_promote_count": preview["would_promote_count"],
            "would_confirm_existing_count": preview["would_confirm_existing_count"],
            "would_skip_count": preview["would_skip_count"],
        },
        "candidates": [_serialize_flush_candidate(entry) for entry in preview["candidates"]],
        "profile_before": _normalize_profile_snapshot(preview["profile_before"]),
        "profile_after": _normalize_profile_snapshot(preview["profile_after"]),
        "profile_diff": preview["profile_diff"],
    }


@app.get("/api/v1/sessions/{session_id}/memory/profile-diff")
def get_memory_profile_diff(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")

    preview = build_flush_preview(db, session_id)
    return {
        "session": {
            "id": session.id,
            "status": session.status,
            "current_stage": session.current_stage,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        },
        "flush_window": {
            "candidate_count": preview["candidate_count"],
            "would_promote_count": preview["would_promote_count"],
            "would_confirm_existing_count": preview["would_confirm_existing_count"],
        },
        "profile_before": _normalize_profile_snapshot(preview["profile_before"]),
        "profile_after": _normalize_profile_snapshot(preview["profile_after"]),
        "profile_diff": preview["profile_diff"],
    }


@app.post("/api/v1/sessions/{session_id}/close", response_model=SessionResponse)
def close_session(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
    flush_session_memory(db, session_id=session_id)
    session.status = "closed"
    db.commit()
    return SessionResponse(session_id=session.id, current_stage=session.current_stage, status=session.status)


@app.post("/api/v1/sessions/{session_id}/documents", response_model=DocumentResponse)
def create_document(session_id: str, payload: CreateDocumentRequest, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")

    document, chunks = persist_document(session_id=session_id, title=payload.title, content=payload.content)
    db.add(document)
    db.flush()
    persisted_chunks: list[DocumentChunk] = []
    for chunk in chunks:
        chunk.document_id = document.id
        db.add(chunk)
        persisted_chunks.append(chunk)
    db.flush()
    capture_document_memories(db, session=session, document=document, document_chunks=persisted_chunks)
    db.commit()

    return DocumentResponse(
        document_id=document.id,
        session_id=session_id,
        title=document.title,
        chunk_count=len(persisted_chunks),
        source_type=document.source_type,
    )


@app.get("/api/v1/sessions/{session_id}/documents")
def get_documents(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
    docs = db.query(Document).filter(Document.session_id == session_id).order_by(Document.created_at.desc()).all()
    return [
        {
            "document_id": document.id,
            "session_id": session_id,
            "title": document.title,
            "source_type": document.source_type,
            "chunk_count": db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).count(),
            "created_at": document.created_at,
        }
        for document in docs
    ]


@app.get("/api/v1/documents/{document_id}/chunks")
def get_document_chunks(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="DOCUMENT_NOT_FOUND")
    chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index.asc()).all()
    return [
        {
            "chunk_id": chunk.id,
            "document_id": document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
        }
        for chunk in chunks
    ]


@app.post("/api/v1/chat/turn", response_model=ChatTurnResponse)
def chat_turn(payload: ChatTurnRequest, db: Session = Depends(get_db)):
    session = db.get(ChatSession, payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="SESSION_CLOSED")

    if payload.client_turn_id:
        existing = db.scalar(
            select(Turn).where(Turn.session_id == payload.session_id, Turn.client_turn_id == payload.client_turn_id)
        )
        if existing:
            raise HTTPException(status_code=409, detail="IDEMPOTENCY_CONFLICT")

    turn_count = db.query(Turn).filter(Turn.session_id == payload.session_id).count()
    user_turn = Turn(
        session_id=payload.session_id,
        turn_index=turn_count + 1,
        role="user",
        content=payload.user_text,
        client_turn_id=payload.client_turn_id,
    )
    db.add(user_turn)
    db.flush()

    history_turns = (
        db.query(Turn)
        .filter(Turn.session_id == payload.session_id)
        .order_by(Turn.turn_index.asc())
        .all()
    )
    history_user_texts = [t.content for t in history_turns if t.role == "user"]

    extracted = extract_structure(payload.user_text, history_user_texts)
    arg = ArgumentUnit(
        session_id=payload.session_id,
        turn_id=user_turn.id,
        claim=extracted.claim,
        reasons=extracted.reasons,
        evidence=extracted.evidence,
        value_premises=extracted.value_premises,
        definitions=extracted.definitions,
        flags=extracted.flags,
        confidence=extracted.confidence,
        raw_schema=extracted.raw_schema,
    )
    db.add(arg)

    assistant_intents = [t.question_intent for t in history_turns if t.role == "assistant" and t.question_intent]
    same_intent_recent = 0
    if assistant_intents:
        last_intent = assistant_intents[-1]
        same_intent_recent = sum(1 for x in assistant_intents[-3:] if x == last_intent)

    latest_summary = (
        db.query(Summary)
        .filter(Summary.session_id == payload.session_id)
        .order_by(Summary.created_at.desc())
        .first()
    )
    turns_since_summary = (turn_count + 1) if not latest_summary else max(0, (turn_count + 1) - 4)

    planning_rag = build_planning_rag(
        db=db,
        session_id=payload.session_id,
        extraction=extracted,
        exclude_turn_id=user_turn.id,
    )

    decision = decide_next(
        current_stage=session.current_stage,
        extraction=extracted,
        planning_rag=planning_rag,
        same_intent_recent=same_intent_recent,
        turns_since_summary=turns_since_summary,
        summary_interval=settings.summary_interval,
    )

    transition = StateTransition(
        session_id=payload.session_id,
        from_stage=session.current_stage,
        to_stage=decision.to_stage,
        trigger_type=decision.trigger_reason,
        trigger_detail={
            "weak_point": decision.weak_point,
            "confidence": extracted.confidence,
            "target_node": decision.target_node,
            "target_text": decision.target_text,
            "goal": decision.goal,
            "planner_source": decision.planner_source,
            "retrieval_summary": planning_rag.relevance_summary,
        },
        at_turn_index=user_turn.turn_index,
    )
    db.add(transition)

    generation = generate_response(decision, extracted)

    assistant_turn = Turn(
        session_id=payload.session_id,
        turn_index=user_turn.turn_index + 1,
        role="assistant",
        content=generation.text,
        question_intent=decision.question_intent,
    )
    db.add(assistant_turn)
    db.flush()

    qplan = QuestionPlan(
        session_id=payload.session_id,
        turn_id=assistant_turn.id,
        current_stage=decision.to_stage,
        weak_point=decision.weak_point,
        question_intent=decision.question_intent,
        constraints={
            "single_question": True,
            "max_sentences": 2,
            "dialogue_act": decision.dialogue_act,
            "target_node": decision.target_node,
            "target_text": decision.target_text,
            "target_reason": decision.target_reason,
            "goal": decision.goal,
            "follow_up_chain": decision.follow_up_chain,
            "safety_mode": decision.safety_mode,
            "planner_source": decision.planner_source,
            "planner_error": decision.planner_error,
            "selected_evidence": decision.selected_evidence,
            "retrieval_summary": planning_rag.relevance_summary,
            "profile_snapshot": planning_rag.profile_snapshot,
            "profile_hits": planning_rag.profile_hits,
            "generation_source": generation.source,
            "generation_error": generation.error,
        },
    )
    db.add(qplan)

    session.current_stage = decision.to_stage
    captured_memories = capture_turn_memories(db, session=session, turn=user_turn, extraction=extracted, raw_text=payload.user_text)
    flush_result = {"promoted_count": 0, "profile_updated": False, "dialogue_style": None}

    if decision.summary_required:
        flush_result = flush_session_memory(db, session_id=payload.session_id)
        summary = Summary(
            session_id=payload.session_id,
            round_range=f"{max(1, user_turn.turn_index-3)}-{user_turn.turn_index}",
            summary_text="本阶段你已经给出核心观点、部分依据与待澄清术语。请确认是否准确。",
        )
        db.add(summary)

    metric = db.scalar(select(Metric).where(Metric.session_id == payload.session_id))
    if not metric:
        metric = Metric(session_id=payload.session_id, metric_date=datetime.utcnow().date().isoformat())
        db.add(metric)
    if metric.premise_explicit_count is None:
        metric.premise_explicit_count = 0
    metric.premise_explicit_count += len(extracted.reasons)

    db.commit()

    return ChatTurnResponse(
        assistant_text=assistant_turn.content,
        session_id=payload.session_id,
        turn_index=assistant_turn.turn_index,
        current_stage=decision.to_stage,
        question_intent=decision.question_intent,
        confidence=extracted.confidence,
        summary_required=decision.summary_required,
        meta={
            "transition": {
                "from_stage": transition.from_stage,
                "to_stage": transition.to_stage,
                "trigger_type": transition.trigger_type,
            },
            "weak_point": decision.weak_point,
            "safety_mode": decision.safety_mode,
            "dialogue_act": decision.dialogue_act,
            "target_node": decision.target_node,
            "target_text": decision.target_text,
            "target_reason": decision.target_reason,
            "goal": decision.goal,
            "follow_up_chain": decision.follow_up_chain,
            "selected_evidence": decision.selected_evidence,
            "planner_source": decision.planner_source,
            "planner_error": decision.planner_error,
            "retrieval_summary": planning_rag.relevance_summary,
            "profile_snapshot": planning_rag.profile_snapshot,
            "profile_hits": planning_rag.profile_hits,
            "extractor_source": extracted.raw_schema.get("_meta", {}).get("source"),
            "extractor_error": extracted.raw_schema.get("_meta", {}).get("error"),
            "generation_source": generation.source,
            "generation_error": generation.error,
            "memory_records_captured": len(captured_memories),
            "memory_flush": flush_result,
            "profile_snapshot": planning_rag.profile_snapshot,
        },
    )


@app.get("/api/v1/sessions/{session_id}/turns")
def get_turns(session_id: str, db: Session = Depends(get_db)):
    rows = db.query(Turn).filter(Turn.session_id == session_id).order_by(Turn.turn_index.asc()).all()
    return [
        {
            "id": t.id,
            "turn_index": t.turn_index,
            "role": t.role,
            "content": t.content,
            "question_intent": t.question_intent,
            "created_at": t.created_at,
        }
        for t in rows
    ]


@app.get("/api/v1/sessions/{session_id}/argument-units")
def get_argument_units(session_id: str, db: Session = Depends(get_db)):
    rows = db.query(ArgumentUnit).filter(ArgumentUnit.session_id == session_id).all()
    return [
        {
            "id": r.id,
            "turn_id": r.turn_id,
            "claim": r.claim,
            "reasons": r.reasons,
            "definitions": r.definitions,
            "flags": r.flags,
            "confidence": r.confidence,
        }
        for r in rows
    ]


@app.get("/api/v1/sessions/{session_id}/argument-units/raw")
def get_argument_units_raw(session_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(ArgumentUnit)
        .filter(ArgumentUnit.session_id == session_id)
        .order_by(ArgumentUnit.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "turn_id": r.turn_id,
            "raw_schema": r.raw_schema,
            "created_at": r.created_at,
        }
        for r in rows
    ]


@app.get("/api/v1/sessions/{session_id}/transitions")
def get_transitions(session_id: str, db: Session = Depends(get_db)):
    rows = db.query(StateTransition).filter(StateTransition.session_id == session_id).order_by(StateTransition.created_at.asc()).all()
    return [
        {
            "from_stage": r.from_stage,
            "to_stage": r.to_stage,
            "trigger_type": r.trigger_type,
            "trigger_detail": r.trigger_detail,
            "at_turn_index": r.at_turn_index,
        }
        for r in rows
    ]


@app.get("/api/v1/sessions/{session_id}/question-plans")
def get_question_plans(session_id: str, db: Session = Depends(get_db)):
    rows = db.query(QuestionPlan).filter(QuestionPlan.session_id == session_id).order_by(QuestionPlan.created_at.asc()).all()
    return [
        {
            "current_stage": r.current_stage,
            "weak_point": r.weak_point,
            "question_intent": r.question_intent,
            "constraints": r.constraints,
        }
        for r in rows
    ]


@app.post("/api/v1/sessions/{session_id}/summaries")
def create_summary(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
    turns = db.query(Turn).filter(Turn.session_id == session_id).count()
    summary = Summary(
        session_id=session_id,
        round_range=f"{max(1, turns-3)}-{turns}",
        summary_text="当前观点摘要：你正在从主张走向可执行修订。",
    )
    db.add(summary)
    db.commit()
    return {"summary_id": summary.id, "summary_text": summary.summary_text}


@app.post("/api/v1/summaries/{summary_id}/confirm")
def confirm_summary(summary_id: str, payload: SummaryConfirmRequest, db: Session = Depends(get_db)):
    summary = db.get(Summary, summary_id)
    if not summary:
        raise HTTPException(status_code=404, detail="SUMMARY_NOT_FOUND")
    summary.user_confirmed = payload.confirmed
    if payload.feedback:
        summary.summary_text += f"\n用户反馈：{payload.feedback}"
    db.commit()
    return {"ok": True, "summary_id": summary_id, "confirmed": payload.confirmed}


@app.get("/api/v1/sessions/{session_id}/metrics")
def get_metrics(session_id: str, db: Session = Depends(get_db)):
    metric = db.scalar(select(Metric).where(Metric.session_id == session_id))
    if not metric:
        return {
            "consistency_score": 0.0,
            "self_revision_count": 0,
            "question_repeat_rate": 0.0,
            "premise_explicit_count": 0,
        }
    return {
        "consistency_score": metric.consistency_score,
        "self_revision_count": metric.self_revision_count,
        "question_repeat_rate": metric.question_repeat_rate,
        "premise_explicit_count": metric.premise_explicit_count,
    }
