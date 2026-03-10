from pydantic import BaseModel, Field
from typing import Optional, Literal, Any


Stage = Literal["S0", "S1", "S2", "S3", "S4", "S5"]
Intent = Literal[
    "clarify_term",
    "ask_premise",
    "test_consistency",
    "probe_causality",
    "necessary_vs_sufficient",
    "counterexample",
    "value_priority",
    "operationalize",
]


class CreateSessionRequest(BaseModel):
    user_id: str
    title: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    current_stage: Stage
    status: str


class ChatTurnRequest(BaseModel):
    session_id: str
    user_text: str = Field(min_length=1, max_length=2000)
    client_turn_id: Optional[str] = None
    debug: bool = False


class TransitionMeta(BaseModel):
    from_stage: Stage
    to_stage: Stage
    trigger_type: str


class ChatTurnResponse(BaseModel):
    assistant_text: str
    session_id: str
    turn_index: int
    current_stage: Stage
    question_intent: Intent
    confidence: float
    summary_required: bool
    meta: dict[str, Any]


class SummaryConfirmRequest(BaseModel):
    confirmed: bool
    feedback: Optional[str] = None
