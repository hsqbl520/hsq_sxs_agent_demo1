import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings


ABSTRACT_TERMS = {"成功", "幸福", "自由", "努力", "公平", "正义", "意义", "价值"}
ABSOLUTIST_TERMS = {"一定", "所有", "绝不", "必须", "总是", "永远"}
REASON_MARKERS = {"因为", "所以", "因此", "由于", "依据", "证明"}
CLAIM_MARKERS = {"我认为", "应该", "是", "不是", "会", "不会"}


@dataclass
class ExtractedFeatures:
    has_claim: bool
    term_ambiguity: float
    premise_count: int
    logic_gap: float
    absolutist_level: float
    contradiction_level: float
    extract_confidence: float


@dataclass
class ExtractionResult:
    claim: str | None
    reasons: list[str]
    evidence: list[str]
    value_premises: list[str]
    definitions: list[dict]
    focus_terms: list[str]
    attackable_points: list[dict[str, Any]]
    missing_links: list[str]
    flags: dict
    confidence: float
    features: ExtractedFeatures
    raw_schema: dict[str, Any]


def _split_sentences(text: str) -> list[str]:
    for p in ["？", "！", "。", "?", "!", ".", "\n"]:
        text = text.replace(p, "\n")
    return [x.strip() for x in text.split("\n") if x.strip()]


def _uniq_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            ordered.append(cleaned)
            seen.add(cleaned)
    return ordered


def _derive_focus_terms(text: str, claim: str | None, definitions: list[dict], ambiguity_hits: list[str]) -> list[str]:
    candidates = [item.get("term", "") for item in definitions if item.get("term")] + ambiguity_hits
    if not candidates and claim:
        for marker in ABSTRACT_TERMS:
            if marker in claim:
                candidates.append(marker)
    return _uniq_texts(candidates)


def _derive_missing_links(claim: str | None, reasons: list[str], value_premises: list[str], flags: dict) -> list[str]:
    links: list[str] = []
    if claim and not reasons:
        links.append("缺少支撑主张成立的直接理由。")
    elif claim and len(reasons) == 1:
        links.append("只有一条理由，尚未说明它为何足以推出结论。")
    if value_premises and not reasons:
        links.append("价值前提已经出现，但尚未说明它如何支撑当前主张。")
    if flags.get("absolutist"):
        links.append("绝对化表达缺少边界条件和例外说明。")
    return _uniq_texts(links)


def _derive_attackable_points(
    claim: str | None,
    reasons: list[str],
    definitions: list[dict],
    value_premises: list[str],
    focus_terms: list[str],
    missing_links: list[str],
    flags: dict,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []

    def add_point(point_type: str, target_node: str, target_text: str, why: str) -> None:
        if not target_text and target_node != "claim":
            return
        entry = {
            "type": point_type,
            "target_node": target_node,
            "target_text": target_text,
            "why": why,
        }
        if entry not in points:
            points.append(entry)

    if not claim:
        add_point("missing_claim", "claim", "你的核心主张", "当前输入还没有形成一个可辩论的明确主张。")
    if focus_terms and not definitions:
        add_point("undefined_term", "focus_term[0]", focus_terms[0], "关键概念出现了，但边界和含义还没有说明。")
    if claim and not reasons:
        add_point("missing_reason", "claim", claim, "提出了结论，但还没有给出直接理由。")
    if claim and flags.get("absolutist"):
        add_point("absolute_claim", "claim", claim, "结论表达得过于绝对，尚未说明例外和边界。")
    if flags.get("causality_risk") and reasons:
        add_point("causality_gap", "reason[0]", reasons[0], "这里像是在把相关关系直接当成因果关系。")
    if flags.get("potential_contradiction") and claim:
        add_point("consistency_gap", "claim", claim, "当前说法和历史观点之间可能存在冲突。")
    if value_premises:
        add_point(
            "unsupported_value_premise",
            "value_premise[0]",
            value_premises[0],
            "出现了价值排序，但尚未说明为什么这个价值应该优先。",
        )
    if missing_links and claim:
        add_point("missing_link", "claim", claim, missing_links[0])

    return points


def _rule_extract_structure(
    text: str,
    history_texts: list[str],
    source: str = "rule",
    error_message: str | None = None,
) -> ExtractionResult:
    sents = _split_sentences(text)
    claim = sents[0] if sents else None

    has_claim = any(marker in text for marker in CLAIM_MARKERS) and bool(claim)

    reasons = [s for s in sents if any(m in s for m in REASON_MARKERS)]
    premise_count = len(reasons)
    value_premises = [s for s in sents if "应该" in s or "值得" in s]

    ambiguity_hits = [t for t in ABSTRACT_TERMS if t in text]
    definition_hits = [
        {"term": t, "definition": ""}
        for t in ambiguity_hits
        if f"{t}是" in text or f"{t}指" in text
    ]
    term_ambiguity = min(1.0, max(0.0, (len(ambiguity_hits) - len(definition_hits)) * 0.35))

    abs_count = sum(1 for t in ABSOLUTIST_TERMS if t in text)
    absolutist_level = min(1.0, abs_count * 0.35)

    logic_gap = 0.2
    if has_claim and premise_count == 0:
        logic_gap = 0.75
    elif has_claim and premise_count == 1:
        logic_gap = 0.55

    contradiction_level = 0.0
    if history_texts:
        latest = " ".join(history_texts[-3:])
        if ("不" in latest and "不" not in text and "是" in text) or ("不" not in latest and "不" in text):
            contradiction_level = 0.45

    confidence = 0.9
    if not has_claim:
        confidence -= 0.25
    if term_ambiguity >= 0.65:
        confidence -= 0.2
    if premise_count == 0:
        confidence -= 0.2
    confidence = max(0.1, min(0.99, confidence))

    flags = {
        "absolutist": absolutist_level >= 0.7,
        "causality_risk": ("导致" in text or "所以" in text) and ("可能" not in text and "也许" not in text),
        "ambiguity": term_ambiguity >= 0.65,
        "potential_contradiction": contradiction_level >= 0.65,
    }
    focus_terms = _derive_focus_terms(text, claim, definition_hits, ambiguity_hits)
    missing_links = _derive_missing_links(claim, reasons, value_premises, flags)
    attackable_points = _derive_attackable_points(
        claim=claim,
        reasons=reasons,
        definitions=definition_hits,
        value_premises=value_premises,
        focus_terms=focus_terms,
        missing_links=missing_links,
        flags=flags,
    )

    raw_schema = {
        "claim": {"text": claim, "type": "mixed", "scope": None},
        "reasons": [{"text": r, "type": "other"} for r in reasons],
        "evidence": [],
        "value_premises": [{"text": v, "priority": "unknown"} for v in value_premises],
        "definitions": definition_hits,
        "focus_terms": focus_terms,
        "attackable_points": attackable_points,
        "missing_links": missing_links,
        "relation_hints": [],
        "flags": flags,
        "conflict_with": [],
        "uncertainty_notes": [],
        "confidence": confidence,
        "_meta": {
            "source": source,
            "error": error_message,
        },
    }

    return ExtractionResult(
        claim=claim,
        reasons=reasons,
        evidence=[],
        value_premises=value_premises,
        definitions=definition_hits,
        focus_terms=focus_terms,
        attackable_points=attackable_points,
        missing_links=missing_links,
        flags=flags,
        confidence=confidence,
        features=ExtractedFeatures(
            has_claim=has_claim,
            term_ambiguity=term_ambiguity,
            premise_count=premise_count,
            logic_gap=logic_gap,
            absolutist_level=absolutist_level,
            contradiction_level=contradiction_level,
            extract_confidence=confidence,
        ),
        raw_schema=raw_schema,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _from_llm_payload(payload: dict[str, Any], text: str, history_texts: list[str]) -> ExtractionResult:
    claim_obj = payload.get("claim") or {}
    claim = claim_obj.get("text") if isinstance(claim_obj, dict) else None

    reasons = [x.get("text", "") for x in payload.get("reasons", []) if isinstance(x, dict) and x.get("text")]
    evidence = [x.get("text", "") for x in payload.get("evidence", []) if isinstance(x, dict) and x.get("text")]
    value_premises = [
        x.get("text", "") for x in payload.get("value_premises", []) if isinstance(x, dict) and x.get("text")
    ]
    definitions = [
        {"term": x.get("term", ""), "definition": x.get("definition", "")}
        for x in payload.get("definitions", [])
        if isinstance(x, dict) and x.get("term")
    ]
    focus_terms = _uniq_texts([x for x in payload.get("focus_terms", []) if isinstance(x, str)])
    missing_links = _uniq_texts([x for x in payload.get("missing_links", []) if isinstance(x, str)])
    attackable_points = [
        {
            "type": x.get("type", "other"),
            "target_node": x.get("target_node", "claim"),
            "target_text": x.get("target_text", ""),
            "why": x.get("why", ""),
        }
        for x in payload.get("attackable_points", [])
        if isinstance(x, dict)
    ]

    flags_obj = payload.get("flags") if isinstance(payload.get("flags"), dict) else {}
    flags = {
        "absolutist": bool(flags_obj.get("absolutist", False)),
        "causality_risk": bool(flags_obj.get("causality_risk", False)),
        "ambiguity": bool(flags_obj.get("ambiguity", False)),
        "potential_contradiction": bool(flags_obj.get("potential_contradiction", False)),
    }

    confidence = max(0.0, min(1.0, _safe_float(payload.get("confidence"), 0.5)))
    has_claim = bool(claim)

    if not focus_terms:
        ambiguity_hits = [t for t in ABSTRACT_TERMS if t in text]
        focus_terms = _derive_focus_terms(text, claim, definitions, ambiguity_hits)
    if not missing_links:
        missing_links = _derive_missing_links(claim, reasons, value_premises, flags)
    if not attackable_points:
        attackable_points = _derive_attackable_points(
            claim=claim,
            reasons=reasons,
            definitions=definitions,
            value_premises=value_premises,
            focus_terms=focus_terms,
            missing_links=missing_links,
            flags=flags,
        )

    term_ambiguity = 0.8 if flags["ambiguity"] or any(p["type"] == "undefined_term" for p in attackable_points) else 0.2
    premise_count = len(reasons)
    logic_gap = 0.8 if missing_links else 0.2 if premise_count >= 2 else 0.55 if premise_count == 1 else 0.75
    absolutist_level = 0.8 if flags["absolutist"] or any(p["type"] == "absolute_claim" for p in attackable_points) else 0.1
    contradiction_level = 0.8 if flags["potential_contradiction"] or any(p["type"] == "consistency_gap" for p in attackable_points) else 0.0

    if history_texts and contradiction_level == 0.0:
        latest = " ".join(history_texts[-3:])
        if ("不" in latest and "不" not in text and "是" in text) or ("不" not in latest and "不" in text):
            contradiction_level = 0.45

    payload["_meta"] = {
        "source": "llm",
        "error": None,
    }

    return ExtractionResult(
        claim=claim,
        reasons=reasons,
        evidence=evidence,
        value_premises=value_premises,
        definitions=definitions,
        focus_terms=focus_terms,
        attackable_points=attackable_points,
        missing_links=missing_links,
        flags=flags,
        confidence=confidence,
        features=ExtractedFeatures(
            has_claim=has_claim,
            term_ambiguity=term_ambiguity,
            premise_count=premise_count,
            logic_gap=logic_gap,
            absolutist_level=absolutist_level,
            contradiction_level=contradiction_level,
            extract_confidence=confidence,
        ),
        raw_schema=payload,
    )


def _extract_via_llm(text: str, history_texts: list[str]) -> ExtractionResult:
    if not settings.llm_api_key:
        return _rule_extract_structure(
            text,
            history_texts,
            source="rule_fallback",
            error_message="LLM_API_KEY is empty",
        )

    system_prompt = """你是“论证结构化抽取器”。你的任务是把用户一句话（结合最近上下文）抽取成可用于苏格拉底质询的结构化结果。你不负责聊天，不负责给建议，只做抽取。

抽取规则：
1) 尽量忠实原文，不补造不存在的信息。
2) claim 最多 1 条核心主张；若无法确定，claim.text=null。
3) reasons/evidence/value_premises/definitions 可为空数组，不要瞎填。
4) 必须区分“直接抽取”和“谨慎推断”：claim/reasons/evidence/definitions 只能依据原文直接提取；value_premises/attackable_points/missing_links 可以谨慎推断，但要保守。
5) 事实判断与价值判断分开。
6) 出现绝对词（如 一定/所有/绝不/必须）时，flags.absolutist=true。
7) 出现“相关即因果”倾向时，flags.causality_risk=true。
8) 抽象词未定义（如 成功/自由/幸福/努力）且影响理解时，flags.ambiguity=true。
9) 与历史明显冲突时，flags.potential_contradiction=true，并给 conflict_with。
10) focus_terms 只保留本轮真正最值得盯住的 1-3 个词。
11) attackable_points 只保留最值得追问的 1-3 个点，每个点都要写清 target_node、target_text、why。
12) missing_links 只写“从前提到结论还缺的那一步”，不要泛泛重复主张。
13) 置信度要保守：信息不足时降低 confidence，不要虚高。

只输出 JSON，不要解释。"""
    user_payload = {
        "user_text": text,
        "conversation_history": history_texts[-6:],
        "schema": {
            "claim": {"text": "string|null", "type": "descriptive|normative|mixed|null", "scope": "string|null"},
            "reasons": [{"text": "string", "type": "fact|experience|authority|analogy|other"}],
            "evidence": [{"text": "string", "kind": "personal|statistical|anecdotal|none|other"}],
            "value_premises": [{"text": "string", "priority": "high|medium|low|unknown"}],
            "definitions": [{"term": "string", "definition": "string"}],
            "focus_terms": ["string"],
            "attackable_points": [
                {
                    "type": "missing_claim|undefined_term|missing_reason|unsupported_value_premise|consistency_gap|causality_gap|missing_link|absolute_claim|overgeneralization|missing_boundary|weak_evidence|other",
                    "target_node": "claim|focus_term[0]|reason[0]|value_premise[0]|general",
                    "target_text": "string",
                    "why": "string"
                }
            ],
            "missing_links": ["string"],
            "relation_hints": [
                {"from": "claim|reason|evidence|value_premise", "to": "claim|reason|evidence|value_premise", "relation": "support|attack|depend|conflict"}
            ],
            "flags": {
                "absolutist": False,
                "causality_risk": False,
                "ambiguity": False,
                "potential_contradiction": False,
            },
            "conflict_with": [{"past_claim": "string", "why_conflict": "string"}],
            "uncertainty_notes": ["string"],
            "confidence": 0.0,
        },
    }

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.llm_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            payload = json.loads(content)
            return _from_llm_payload(payload, text, history_texts)
    except Exception as exc:
        return _rule_extract_structure(
            text,
            history_texts,
            source="rule_fallback",
            error_message=str(exc),
        )


def extract_structure(text: str, history_texts: list[str]) -> ExtractionResult:
    if settings.extractor_mode.lower() == "llm":
        return _extract_via_llm(text, history_texts)
    return _rule_extract_structure(text, history_texts)
