import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    external_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")
    current_stage: Mapped[str] = mapped_column(String, default="S0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    turns = relationship("Turn", back_populates="session")
    documents = relationship("Document", back_populates="session")


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), index=True)
    turn_index: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    question_intent: Mapped[str | None] = mapped_column(String, nullable=True)
    client_turn_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="turns")


class ArgumentUnit(Base):
    __tablename__ = "argument_units"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), index=True)
    turn_id: Mapped[str] = mapped_column(String, ForeignKey("turns.id"), index=True)
    claim: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    value_premises: Mapped[list] = mapped_column(JSON, default=list)
    definitions: Mapped[list] = mapped_column(JSON, default=list)
    flags: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float)
    raw_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StateTransition(Base):
    __tablename__ = "state_transitions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), index=True)
    from_stage: Mapped[str] = mapped_column(String)
    to_stage: Mapped[str] = mapped_column(String)
    trigger_type: Mapped[str] = mapped_column(String)
    trigger_detail: Mapped[dict] = mapped_column(JSON, default=dict)
    at_turn_index: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class QuestionPlan(Base):
    __tablename__ = "question_plans"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), index=True)
    turn_id: Mapped[str] = mapped_column(String, ForeignKey("turns.id"), index=True)
    current_stage: Mapped[str] = mapped_column(String)
    weak_point: Mapped[str] = mapped_column(String)
    question_intent: Mapped[str] = mapped_column(String)
    constraints: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), index=True)
    round_range: Mapped[str] = mapped_column(String)
    summary_text: Mapped[str] = mapped_column(Text)
    user_confirmed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), index=True)
    metric_date: Mapped[str] = mapped_column(String)
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)
    self_revision_count: Mapped[int] = mapped_column(Integer, default=0)
    question_repeat_rate: Mapped[float] = mapped_column(Float, default=0.0)
    premise_explicit_count: Mapped[int] = mapped_column(Integer, default=0)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), index=True)
    title: Mapped[str] = mapped_column(String)
    source_type: Mapped[str] = mapped_column(String, default="pasted")
    raw_content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), index=True)
    document_id: Mapped[str] = mapped_column(String, ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
