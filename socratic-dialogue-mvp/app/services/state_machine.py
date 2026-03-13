import json
from dataclasses import dataclass, replace
from typing import Any

import httpx

from app.config import settings
from .extractor import ExtractionResult
from .retrieval import PlanningRAG


ALT_INTENTS = {
    "clarify_term": "ask_premise",
    "ask_premise": "necessary_vs_sufficient",
    "test_consistency": "probe_causality",
    "counterexample": "necessary_vs_sufficient",
}
VALID_DIALOGUE_ACTS = {
    "restate",
    "clarify",
    "probe",
    "surface_tension",
    "propose_revision",
    "synthesize",
    "challenge",
}
VALID_STAGES = {"S0", "S1", "S2", "S3", "S4", "S5"}
VALID_INTENTS = {
    "clarify_term",
    "ask_premise",
    "test_consistency",
    "probe_causality",
    "necessary_vs_sufficient",
    "counterexample",
    "value_priority",
    "operationalize",
}
POINT_MAP = {
    "missing_claim": ("S0", "clarify_term", "claim", "先收束出一句可辩论的主张。"),
    "undefined_term": ("S1", "clarify_term", "definition", "先把关键词的边界说清楚。"),
    "missing_reason": ("S2", "ask_premise", "premise", "逼出支撑主张的关键前提。"),
    "unsupported_value_premise": ("S3", "value_priority", "value", "让用户说明价值排序的依据。"),
    "consistency_gap": ("S3", "test_consistency", "consistency", "逼用户统一当前说法和历史说法。"),
    "causality_gap": ("S3", "probe_causality", "causality", "区分相关关系和因果关系。"),
    "missing_link": ("S3", "necessary_vs_sufficient", "logic_link", "补齐从理由到结论的缺失推理链。"),
    "absolute_claim": ("S4", "counterexample", "counterexample", "迫使用户给出边界条件和例外。"),
    "overgeneralization": ("S4", "counterexample", "counterexample", "用反例检验过度概括。"),
    "missing_boundary": ("S4", "counterexample", "counterexample", "让用户补上适用范围和边界。"),
    "weak_evidence": ("S3", "ask_premise", "premise", "要求用户给出更强的依据或证据。"),
}


@dataclass
class Decision:
    dialogue_act: str
    to_stage: str
    question_intent: str
    weak_point: str
    trigger_reason: str
    safety_mode: str
    summary_required: bool
    target_node: str
    target_text: str
    target_reason: str
    goal: str
    follow_up_chain: list[str]
    selected_evidence: list[dict[str, Any]]
    planner_source: str
    planner_error: str | None


def _dialogue_act_for(point_type: str, question_intent: str, same_intent_recent: int) -> str:
    if point_type == "missing_claim":
        return "restate"
    if point_type == "undefined_term":
        return "clarify"
    if point_type in {"missing_reason", "weak_evidence", "causality_gap", "missing_link"}:
        return "probe"
    if point_type in {"consistency_gap", "unsupported_value_premise"}:
        return "surface_tension"
    if point_type in {"absolute_claim", "overgeneralization", "missing_boundary"}:
        return "challenge"
    if question_intent == "operationalize":
        return "propose_revision"
    if same_intent_recent >= 2:
        return "synthesize"
    return "probe"


def _fallback_attack_point(extraction: ExtractionResult) -> dict[str, Any]:
    features = extraction.features
    if not features.has_claim:
        return {
            "type": "missing_claim",
            "target_node": "claim",
            "target_text": "你的核心主张",
            "why": "当前输入还没有形成明确主张。",
        }
    if features.term_ambiguity >= 0.65:
        target_text = extraction.focus_terms[0] if extraction.focus_terms else (extraction.claim or "这个概念")
        return {
            "type": "undefined_term",
            "target_node": "focus_term[0]",
            "target_text": target_text,
            "why": "关键词的边界还不清楚。",
        }
    if features.premise_count < 1:
        return {
            "type": "missing_reason",
            "target_node": "claim",
            "target_text": extraction.claim or "这个结论",
            "why": "当前只有结论，还没有给出直接理由。",
        }
    if features.contradiction_level >= 0.65:
        return {
            "type": "consistency_gap",
            "target_node": "claim",
            "target_text": extraction.claim or "当前说法",
            "why": "当前说法和历史观点之间可能存在冲突。",
        }
    if features.logic_gap >= 0.60:
        return {
            "type": "missing_link",
            "target_node": "claim",
            "target_text": extraction.claim or "这个结论",
            "why": "从理由到结论还缺少关键推理链。",
        }
    if features.absolutist_level >= 0.70:
        return {
            "type": "absolute_claim",
            "target_node": "claim",
            "target_text": extraction.claim or "这个主张",
            "why": "表达过于绝对，尚未给出例外和边界。",
        }
    return {
        "type": "weak_evidence",
        "target_node": "claim",
        "target_text": extraction.claim or "当前观点",
        "why": "可以推动用户把观点说得更可执行。",
    }


def _pick_attack_point(extraction: ExtractionResult) -> dict[str, Any]:
    if extraction.attackable_points:
        priority = [
            "missing_claim",
            "undefined_term",
            "missing_reason",
            "consistency_gap",
            "causality_gap",
            "missing_link",
            "unsupported_value_premise",
            "absolute_claim",
            "overgeneralization",
            "missing_boundary",
            "weak_evidence",
        ]
        ranked = {code: i for i, code in enumerate(priority)}
        return sorted(
            extraction.attackable_points,
            key=lambda point: ranked.get(point.get("type", "other"), 99),
        )[0]
    return _fallback_attack_point(extraction)


def _follow_up_chain(intent: str, target_text: str) -> list[str]:
    quoted = f"“{target_text}”" if target_text else "这个点"
    chains = {
        "clarify_term": [
            f"如果用户定义了{quoted}，下一轮就检验这个定义能否覆盖边界案例。",
            f"如果定义仍模糊，要求用户给出一个具体例子区分{quoted}与相近概念。",
        ],
        "ask_premise": [
            f"如果用户给出理由，下一轮就追问这个理由是否足以推出{quoted}。",
            "如果用户仍给不出理由，要求其缩小主张范围。",
        ],
        "test_consistency": [
            "如果用户承认冲突，要求其修订其中一句话。",
            "如果用户否认冲突，要求其给出同时成立的条件。",
        ],
        "probe_causality": [
            "如果用户坚持因果关系，要求其说明中间机制。",
            "如果用户给不出机制，要求其改成更弱的相关性表述。",
        ],
        "necessary_vs_sufficient": [
            "如果用户分不清必要和充分，要求其各举一个例子。",
            "如果用户区分清楚，继续问为何当前条件属于这一类。",
        ],
        "counterexample": [
            "如果用户承认反例存在，要求其补上限制条件。",
            "如果用户否认反例成立，要求其说明为什么该例外不算例外。",
        ],
        "value_priority": [
            "如果用户说明了排序标准，继续追问该标准是否适用于别的场景。",
            "如果用户说不清排序标准，要求其给出一个价值冲突案例。",
        ],
        "operationalize": [
            "如果用户给出可执行版本，继续追问如何验证是否做到。",
            "如果用户给不出行动，要求其把主张缩小到一周内可实践的范围。",
        ],
    }
    return chains.get(intent, [])


def _decision_from_point(point_type: str, target_node: str, target_text: str, target_reason: str, same_intent_recent: int) -> Decision:
    to_stage, question_intent, weak_point, goal = POINT_MAP.get(
        point_type,
        ("S5", "operationalize", "actionability", "把当前观点收束成更精确、可执行的话。"),
    )
    dialogue_act = _dialogue_act_for(point_type, question_intent, same_intent_recent)
    if same_intent_recent >= 2 and question_intent in ALT_INTENTS:
        question_intent = ALT_INTENTS[question_intent]
        if dialogue_act == "challenge":
            dialogue_act = "surface_tension"
    return Decision(
        dialogue_act=dialogue_act,
        to_stage=to_stage,
        question_intent=question_intent,
        weak_point=weak_point,
        trigger_reason=point_type,
        safety_mode="normal",
        summary_required=False,
        target_node=target_node,
        target_text=target_text,
        target_reason=target_reason,
        goal=goal,
        follow_up_chain=_follow_up_chain(question_intent, target_text),
        selected_evidence=[],
        planner_source="rule_fallback",
        planner_error=None,
    )


def _rule_decide(
    current_stage: str,
    extraction: ExtractionResult,
    planning_rag: PlanningRAG,
    same_intent_recent: int,
) -> Decision:
    claim_text = extraction.claim or "你的这句话"

    if planning_rag.memory_conflicts:
        hit = planning_rag.memory_conflicts[0]
        decision = _decision_from_point(
            "consistency_gap",
            "claim",
            claim_text,
            f"你前面说过“{hit['text']}”，这和当前说法之间存在张力。",
            same_intent_recent,
        )
        decision.selected_evidence = [hit]
        return decision

    if extraction.focus_terms and not extraction.definitions:
        target_term = extraction.focus_terms[0]
        reason = "这个关键词还没有被稳定定义。"
        evidence: list[dict[str, Any]] = []
        if planning_rag.definition_hits:
            hit = planning_rag.definition_hits[0]
            reason = f"你之前给过一个相关定义：{hit['text']}。现在这个词的边界又变得模糊了。"
            evidence = [hit]
        decision = _decision_from_point(
            "undefined_term",
            "focus_term[0]",
            target_term,
            reason,
            same_intent_recent,
        )
        decision.selected_evidence = evidence
        return decision

    if not extraction.reasons:
        decision = _decision_from_point(
            "missing_reason",
            "claim",
            claim_text,
            "当前只有结论，还没有给出直接理由。",
            same_intent_recent,
        )
        if planning_rag.memory_supports:
            decision.selected_evidence = planning_rag.memory_supports[:1]
        return decision

    if extraction.missing_links:
        decision = _decision_from_point(
            "missing_link",
            "claim",
            claim_text,
            extraction.missing_links[0],
            same_intent_recent,
        )
        return decision

    if planning_rag.counterexample_hits and (
        extraction.flags.get("absolutist") or any(point.get("type") == "absolute_claim" for point in extraction.attackable_points)
    ):
        hit = planning_rag.counterexample_hits[0]
        decision = _decision_from_point(
            "absolute_claim",
            "claim",
            claim_text,
            f"会话历史里已经出现一个可能击穿当前绝对说法的例子：{hit['text']}",
            same_intent_recent,
        )
        decision.selected_evidence = [hit]
        return decision

    if planning_rag.revision_hits:
        hit = planning_rag.revision_hits[0]
        decision = _decision_from_point(
            "weak_evidence",
            "claim",
            claim_text,
            f"你在前面的表达里已经开始移动立场：{hit['text']}。现在适合继续把观点收束得更稳。",
            same_intent_recent,
        )
        if same_intent_recent >= 1:
            decision.dialogue_act = "propose_revision"
            decision.question_intent = "operationalize"
            decision.to_stage = "S5"
            decision.weak_point = "actionability"
            decision.goal = "推动用户产出一个更稳、更可执行的修订版本。"
            decision.follow_up_chain = _follow_up_chain(decision.question_intent, decision.target_text)
        decision.selected_evidence = [hit]
        return decision

    fallback = _pick_attack_point(extraction)
    decision = _decision_from_point(
        fallback.get("type", "weak_evidence"),
        fallback.get("target_node", "claim"),
        fallback.get("target_text") or claim_text,
        fallback.get("why", "这是当前论证里最薄弱的一环。"),
        same_intent_recent,
    )
    return decision


def _build_evidence_candidates(planning_rag: PlanningRAG) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for bucket_name in [
        "memory_conflicts",
        "memory_supports",
        "definition_hits",
        "counterexample_hits",
        "revision_hits",
        "doc_hits",
    ]:
        for item in getattr(planning_rag, bucket_name, [])[:2]:
            candidates.append(
                {
                    "source": item.get("source", bucket_name),
                    "relation": item.get("relation", bucket_name),
                    "text": item.get("text", ""),
                    "score": item.get("score", 0.0),
                }
            )
    return candidates[:8]


def _fallback_defaults(extraction: ExtractionResult, same_intent_recent: int) -> dict[str, Any]:
    point = _pick_attack_point(extraction)
    decision = _decision_from_point(
        point.get("type", "weak_evidence"),
        point.get("target_node", "claim"),
        point.get("target_text") or extraction.claim or "你的这句话",
        point.get("why", "这是当前论证里最薄弱的一环。"),
        same_intent_recent,
    )
    return {
        "dialogue_act": decision.dialogue_act,
        "to_stage": decision.to_stage,
        "question_intent": decision.question_intent,
        "weak_point": decision.weak_point,
        "target_node": decision.target_node,
        "target_text": decision.target_text,
        "target_reason": decision.target_reason,
        "goal": decision.goal,
    }


def _plan_via_llm(
    current_stage: str,
    extraction: ExtractionResult,
    planning_rag: PlanningRAG,
    same_intent_recent: int,
) -> Decision:
    fallback_defaults = _fallback_defaults(extraction, same_intent_recent)
    evidence_candidates = _build_evidence_candidates(planning_rag)

    system_prompt = """你是“辩论规划裁判器”。你的任务是根据当前用户观点、结构化抽取结果，以及会话历史检索证据，直接决定下一轮对话动作。

硬规则：
1) 你是主裁判，不是润色器；请直接给出本轮最合适的规划。
2) 只输出 JSON，不要解释。
3) 只选择一个最值得击打的点，不要同时打多个点。
4) 优先使用会话历史证据；如果历史冲突足够强，优先处理冲突。
5) 输出必须使用给定枚举值。
6) selected_evidence_sources 只能从 evidence_candidates 里选，最多 2 条。
7) 如果当前只有结论没有理由，优先 probe；如果已经有清晰冲突或现成反例，再考虑 challenge 或 surface_tension。
8) 如果用户已经在收缩观点，可以用 propose_revision 或 synthesize 推动收敛。"""

    user_payload = {
        "current_stage": current_stage,
        "same_intent_recent": same_intent_recent,
        "extraction": {
            "claim": extraction.claim,
            "reasons": extraction.reasons,
            "definitions": extraction.definitions,
            "value_premises": extraction.value_premises,
            "focus_terms": extraction.focus_terms,
            "attackable_points": extraction.attackable_points[:4],
            "missing_links": extraction.missing_links,
            "flags": extraction.flags,
        },
        "planning_rag": planning_rag.as_dict(),
        "evidence_candidates": evidence_candidates,
        "fallback_defaults": fallback_defaults,
        "allowed_dialogue_acts": sorted(VALID_DIALOGUE_ACTS),
        "allowed_intents": sorted(VALID_INTENTS),
        "allowed_stages": sorted(VALID_STAGES),
        "output_schema": {
            "dialogue_act": "one of allowed_dialogue_acts",
            "to_stage": "one of allowed_stages",
            "question_intent": "one of allowed_intents",
            "weak_point": "string",
            "trigger_reason": "string",
            "target_node": "string",
            "target_text": "string",
            "target_reason": "string",
            "goal": "string",
            "selected_evidence_sources": ["string"],
        },
    }

    body = {
        "model": settings.planner_model,
        "temperature": settings.planner_temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"

    with httpx.Client(timeout=12.0) as client:
        response = client.post(url, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        data = json.loads(content)

    dialogue_act = data.get("dialogue_act", fallback_defaults["dialogue_act"])
    to_stage = data.get("to_stage", fallback_defaults["to_stage"])
    question_intent = data.get("question_intent", fallback_defaults["question_intent"])
    if dialogue_act not in VALID_DIALOGUE_ACTS or to_stage not in VALID_STAGES or question_intent not in VALID_INTENTS:
        raise ValueError("planner returned invalid enum")

    candidate_map = {item["source"]: item for item in evidence_candidates}
    selected_sources = [x for x in data.get("selected_evidence_sources", []) if isinstance(x, str)]
    selected_evidence = [candidate_map[src] for src in selected_sources if src in candidate_map][:2]

    target_text = data.get("target_text") or fallback_defaults["target_text"]
    return Decision(
        dialogue_act=dialogue_act,
        to_stage=to_stage,
        question_intent=question_intent,
        weak_point=data.get("weak_point", fallback_defaults["weak_point"]),
        trigger_reason=data.get("trigger_reason", data.get("weak_point", fallback_defaults["weak_point"])),
        safety_mode="normal",
        summary_required=False,
        target_node=data.get("target_node", fallback_defaults["target_node"]),
        target_text=target_text,
        target_reason=data.get("target_reason", fallback_defaults["target_reason"]),
        goal=data.get("goal", fallback_defaults["goal"]),
        follow_up_chain=_follow_up_chain(question_intent, target_text),
        selected_evidence=selected_evidence,
        planner_source="llm_primary",
        planner_error=None,
    )


def decide_next(
    current_stage: str,
    extraction: ExtractionResult,
    planning_rag: PlanningRAG,
    same_intent_recent: int,
    turns_since_summary: int,
    summary_interval: int,
) -> Decision:
    if settings.planner_mode.lower() in {"llm", "llm_primary"} and settings.llm_api_key:
        try:
            decision = _plan_via_llm(current_stage, extraction, planning_rag, same_intent_recent)
        except Exception as exc:
            decision = _rule_decide(current_stage, extraction, planning_rag, same_intent_recent)
            decision = replace(decision, planner_source="rule_fallback", planner_error=str(exc))
    else:
        decision = _rule_decide(current_stage, extraction, planning_rag, same_intent_recent)

    summary_required = turns_since_summary >= summary_interval
    return replace(decision, summary_required=summary_required)
