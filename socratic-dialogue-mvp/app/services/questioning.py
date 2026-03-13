import json
from dataclasses import dataclass

import httpx

from app.config import settings
from .extractor import ExtractionResult
from .state_machine import Decision


@dataclass
class GenerationResult:
    text: str
    source: str
    error: str | None


def draft_question(decision: Decision, extracted_claim: str | None) -> str:
    claim = extracted_claim or "你的核心观点"
    target = decision.target_text or claim
    quoted = f"“{target}”" if target else "你的这句话"
    probe_templates = {
        "clarify_term": f"你这里说的{quoted}具体指什么？它的边界在哪里？",
        "ask_premise": f"围绕{quoted}，你最关键的依据是哪一条？",
        "test_consistency": f"围绕{quoted}，你现在的说法和前面的观点如何同时成立？",
        "probe_causality": f"围绕{quoted}，你怎么排除这只是相关，而不是真正的因果？",
        "necessary_vs_sufficient": f"围绕{quoted}，你依赖的条件是必要条件，还是充分条件？",
        "counterexample": f"如果出现一种情况，使{quoted}不成立，你会怎么修正这句话？",
        "value_priority": f"围绕{quoted}背后的价值排序，你为什么把它放在更高位置？",
        "operationalize": f"如果先保留{quoted}，你会怎么把它改写成一句更可执行的话？",
    }
    base_probe = probe_templates.get(decision.question_intent, probe_templates["clarify_term"])

    if decision.dialogue_act == "restate":
        return f"如果我没理解错，你的核心意思是：{quoted}，对吗？"
    if decision.dialogue_act == "clarify":
        return f"我先收住一个点：{quoted}具体指什么？它和相近说法的边界在哪里？"
    if decision.dialogue_act == "surface_tension":
        return f"这里有个张力：{decision.target_reason}。围绕{quoted}，你准备怎么回应这个张力？"
    if decision.dialogue_act == "propose_revision":
        return f"为了把观点说得更稳一些，你愿意把{quoted}改写成一句边界更清楚的话吗？"
    if decision.dialogue_act == "synthesize":
        return f"先总结一下：我们现在卡在{quoted}这一点上，核心问题是{decision.target_reason}。你同意这个判断吗？"
    if decision.dialogue_act == "challenge":
        return f"我直接挑战这一点：{base_probe}"
    return base_probe


def light_rewrite(question: str, safety_mode: str) -> str:
    if safety_mode == "confirm_only":
        return f"为了确认我没有理解偏：{question}"
    return question


def _build_generation_prompt(decision: Decision, extraction: ExtractionResult, fallback_text: str) -> tuple[str, str]:
    system_prompt = """你是“苏格拉底式对话表达器”。你的任务不是决定策略，而是把已经确定好的对话计划，表达成自然、有针对性、有辩论感但不机械的中文回复。

硬规则：
1) 必须忠实执行给定的 dialogue_act、question_intent、target_text、goal。
2) 必须围绕 target_text，不得泛泛而谈。
3) 只能输出最终回复文本，不要输出解释、标签、JSON。
4) 回复控制在 1 到 3 句。
5) 语气要有思辨性，可以锋利，但不能像审讯，也不能像客服模板。
6) 不得脱离当前目标点自行扩展新话题。
7) 如果 dialogue_act 是 restate 或 synthesize，可以不是问句；其他动作至少包含一个明确的推进句。
8) 可以自然引用用户原话中的关键词，让表达更像辩论，不像模板。
9) 如果给定了 selected_evidence，优先自然引用其中 1 条，让回应更有理有据。"""

    user_payload = {
        "dialogue_plan": {
            "dialogue_act": decision.dialogue_act,
            "question_intent": decision.question_intent,
            "target_node": decision.target_node,
            "target_text": decision.target_text,
            "target_reason": decision.target_reason,
            "goal": decision.goal,
            "follow_up_chain": decision.follow_up_chain,
            "safety_mode": decision.safety_mode,
        },
        "extraction_summary": {
            "claim": extraction.claim,
            "reasons": extraction.reasons,
            "value_premises": extraction.value_premises,
            "definitions": extraction.definitions,
            "focus_terms": extraction.focus_terms,
            "attackable_points": extraction.attackable_points[:3],
            "missing_links": extraction.missing_links,
        },
        "selected_evidence": decision.selected_evidence[:2],
        "fallback_text": fallback_text,
        "writing_goal": "生成一段更像真实辩论者的自然中文，不要显得像模板。"
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=False)


def _generate_via_llm(decision: Decision, extraction: ExtractionResult, fallback_text: str) -> GenerationResult:
    if settings.generation_mode.lower() != "llm":
        return GenerationResult(text=fallback_text, source="template", error=None)
    if not settings.llm_api_key:
        return GenerationResult(text=fallback_text, source="template_fallback", error="LLM_API_KEY is empty")

    system_prompt, user_payload = _build_generation_prompt(decision, extraction, fallback_text)
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.generation_model,
        "temperature": settings.generation_temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            return GenerationResult(text=text or fallback_text, source="llm", error=None)
    except Exception as exc:
        return GenerationResult(text=fallback_text, source="template_fallback", error=str(exc))


def generate_response(decision: Decision, extraction: ExtractionResult) -> GenerationResult:
    fallback = light_rewrite(draft_question(decision, extraction.claim), decision.safety_mode)
    return _generate_via_llm(decision, extraction, fallback)
