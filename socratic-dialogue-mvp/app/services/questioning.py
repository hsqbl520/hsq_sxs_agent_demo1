from .state_machine import Decision


def draft_question(decision: Decision, extracted_claim: str | None) -> str:
    claim = extracted_claim or "你的核心观点"
    templates = {
        "clarify_term": f"你这里说的关键概念具体指什么？请用一句话定义它。",
        "ask_premise": f"你这个结论“{claim}”最关键的依据是什么？",
        "test_consistency": f"你现在的说法与前面观点如何同时成立？",
        "probe_causality": f"你如何证明这不是同时出现，而是因果关系？",
        "necessary_vs_sufficient": f"你给出的条件是必要条件，还是充分条件？为什么？",
        "counterexample": f"如果出现一个反例，你这个结论还成立吗？需要加什么限制条件？",
        "value_priority": f"当两个价值冲突时，你为什么优先当前这个价值？",
        "operationalize": f"请把你修订后的观点说成一句可执行的话，你准备先做哪一步？",
    }
    return templates.get(decision.question_intent, templates["clarify_term"])


def light_rewrite(question: str, safety_mode: str) -> str:
    if safety_mode == "confirm_only":
        return f"为了确保我理解准确：{question}"
    return question
