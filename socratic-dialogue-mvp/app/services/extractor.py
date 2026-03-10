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
    flags: dict
    confidence: float
    features: ExtractedFeatures
    raw_schema: dict[str, Any]


def _split_sentences(text: str) -> list[str]:
    for p in ["？", "！", "。", "?", "!", ".", "\n"]:
        text = text.replace(p, "\n")
    return [x.strip() for x in text.split("\n") if x.strip()]


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

    raw_schema = {
        "claim": {"text": claim, "type": "mixed", "scope": None},
        "reasons": [{"text": r, "type": "other"} for r in reasons],
        "evidence": [],
        "value_premises": [{"text": v, "priority": "unknown"} for v in [s for s in sents if "应该" in s or "值得" in s]],
        "definitions": definition_hits,
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
        value_premises=[s for s in sents if "应该" in s or "值得" in s],
        definitions=definition_hits,
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

    flags_obj = payload.get("flags") if isinstance(payload.get("flags"), dict) else {}
    flags = {
        "absolutist": bool(flags_obj.get("absolutist", False)),
        "causality_risk": bool(flags_obj.get("causality_risk", False)),
        "ambiguity": bool(flags_obj.get("ambiguity", False)),
        "potential_contradiction": bool(flags_obj.get("potential_contradiction", False)),
    }

    confidence = max(0.0, min(1.0, _safe_float(payload.get("confidence"), 0.5)))
    has_claim = bool(claim)

    term_ambiguity = 0.8 if flags["ambiguity"] else 0.2
    premise_count = len(reasons)
    logic_gap = 0.2 if premise_count >= 2 else 0.55 if premise_count == 1 else 0.75
    absolutist_level = 0.8 if flags["absolutist"] else 0.1
    contradiction_level = 0.8 if flags["potential_contradiction"] else 0.0

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
4) 事实判断与价值判断分开。
5) 出现绝对词（如 一定/所有/绝不/必须）时，flags.absolutist=true。
6) 出现“相关即因果”倾向时，flags.causality_risk=true。
7) 抽象词未定义（如 成功/自由/幸福/努力）且影响理解时，flags.ambiguity=true。
8) 与历史明显冲突时，flags.potential_contradiction=true，并给 conflict_with。
9) 置信度要保守：信息不足时降低 confidence，不要虚高。

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
