from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from .config import settings
from .database import Base, engine, get_db
from .models import User, Session as ChatSession, Turn, ArgumentUnit, StateTransition, QuestionPlan, Summary, Metric
from .schemas import CreateSessionRequest, SessionResponse, ChatTurnRequest, ChatTurnResponse, SummaryConfirmRequest
from .services.extractor import extract_structure
from .services.state_machine import decide_next
from .services.questioning import draft_question, light_rewrite

app = FastAPI(title=settings.app_name)
WEB_INDEX = Path(__file__).parent / "web" / "index.html"


@app.on_event("startup")
def startup_event() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/")
def web_home():
    if WEB_INDEX.exists():
        return FileResponse(WEB_INDEX)
    raise HTTPException(status_code=404, detail="WEB_UI_NOT_FOUND")


@app.get("/api/v1/debug/config")
def debug_config():
    return {
        "extractor_mode": settings.extractor_mode,
        "llm_api_key_loaded": bool(settings.llm_api_key),
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
    }


def _get_or_create_user(db: Session, external_id: str) -> User:
    user = db.scalar(select(User).where(User.external_id == external_id))
    if user:
        return user
    user = User(external_id=external_id)
    db.add(user)
    db.flush()
    return user


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


@app.post("/api/v1/sessions/{session_id}/close", response_model=SessionResponse)
def close_session(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
    session.status = "closed"
    db.commit()
    return SessionResponse(session_id=session.id, current_stage=session.current_stage, status=session.status)


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

    decision = decide_next(
        current_stage=session.current_stage,
        features=extracted.features,
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
        },
        at_turn_index=user_turn.turn_index,
    )
    db.add(transition)

    assistant_turn = Turn(
        session_id=payload.session_id,
        turn_index=user_turn.turn_index + 1,
        role="assistant",
        content=light_rewrite(draft_question(decision, extracted.claim), decision.safety_mode),
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
        constraints={"single_question": True, "max_sentences": 2},
    )
    db.add(qplan)

    session.current_stage = decision.to_stage

    if decision.summary_required:
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
            "extractor_source": extracted.raw_schema.get("_meta", {}).get("source"),
            "extractor_error": extracted.raw_schema.get("_meta", {}).get("error"),
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
