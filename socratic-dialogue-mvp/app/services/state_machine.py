from dataclasses import dataclass
from .extractor import ExtractedFeatures


HIGH_PRESSURE_INTENTS = {"counterexample", "test_consistency"}


@dataclass
class Decision:
    to_stage: str
    question_intent: str
    weak_point: str
    trigger_reason: str
    safety_mode: str
    summary_required: bool


def decide_next(
    current_stage: str,
    features: ExtractedFeatures,
    same_intent_recent: int,
    turns_since_summary: int,
    summary_interval: int,
) -> Decision:
    low_conf = 0.60
    high_ambig = 0.65
    low_premise = 2
    high_gap = 0.60
    high_abs = 0.70
    high_contra = 0.65

    to_stage = "S5"
    question_intent = "operationalize"
    weak_point = "actionability"
    trigger_reason = "default_consolidation"
    safety_mode = "normal"

    if features.extract_confidence < low_conf:
        to_stage = "S1"
        question_intent = "clarify_term"
        weak_point = "definition"
        trigger_reason = "low_confidence"
        safety_mode = "confirm_only"
    elif not features.has_claim:
        to_stage = "S0"
        question_intent = "clarify_term"
        weak_point = "claim"
        trigger_reason = "missing_claim"
    elif features.term_ambiguity >= high_ambig:
        to_stage = "S1"
        question_intent = "clarify_term"
        weak_point = "definition"
        trigger_reason = "term_ambiguity"
    elif features.premise_count < low_premise:
        to_stage = "S2"
        question_intent = "ask_premise"
        weak_point = "premise"
        trigger_reason = "insufficient_premise"
    elif features.contradiction_level >= high_contra:
        to_stage = "S3"
        question_intent = "test_consistency"
        weak_point = "consistency"
        trigger_reason = "contradiction"
    elif features.logic_gap >= high_gap:
        to_stage = "S3"
        question_intent = "necessary_vs_sufficient"
        weak_point = "logic_link"
        trigger_reason = "logic_gap"
    elif features.absolutist_level >= high_abs:
        to_stage = "S4"
        question_intent = "counterexample"
        weak_point = "counterexample"
        trigger_reason = "absolutist"

    if same_intent_recent >= 2:
        if question_intent == "clarify_term":
            question_intent = "ask_premise"
        elif question_intent == "ask_premise":
            question_intent = "necessary_vs_sufficient"
        elif question_intent == "test_consistency":
            question_intent = "probe_causality"

    summary_required = turns_since_summary >= summary_interval

    return Decision(
        to_stage=to_stage,
        question_intent=question_intent,
        weak_point=weak_point,
        trigger_reason=trigger_reason,
        safety_mode=safety_mode,
        summary_required=summary_required,
    )
