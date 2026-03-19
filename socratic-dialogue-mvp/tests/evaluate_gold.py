import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.services.extractor import extract_structure
from app.services.retrieval import PlanningRAG
from app.services.state_machine import decide_next


def evaluate(gold_path: Path) -> dict:
    data = json.loads(gold_path.read_text(encoding="utf-8"))

    total = len(data)
    stage_hits = 0
    intent_hits = 0
    weak_hits = 0
    failures = []

    for item in data:
        cid = item["id"]
        text = item["user_text"]
        gold = item["gold"]

        extracted = extract_structure(text, history_texts=[])
        decision = decide_next(
            current_stage="S0",
            extraction=extracted,
            planning_rag=PlanningRAG(
                memory_conflicts=[],
                memory_supports=[],
                definition_hits=[],
                counterexample_hits=[],
                revision_hits=[],
                doc_hits=[],
                relevance_summary={},
            ),
            same_intent_recent=0,
            turns_since_summary=0,
            summary_interval=settings.summary_interval,
        )

        stage_ok = decision.to_stage == gold["expected_stage"]
        intent_ok = decision.question_intent == gold["expected_intent"]
        weak_ok = decision.weak_point == gold["weak_point"]

        stage_hits += int(stage_ok)
        intent_hits += int(intent_ok)
        weak_hits += int(weak_ok)

        if not (stage_ok and intent_ok and weak_ok):
            failures.append(
                {
                    "id": cid,
                    "user_text": text,
                    "expected": {
                        "stage": gold["expected_stage"],
                        "intent": gold["expected_intent"],
                        "weak_point": gold["weak_point"],
                    },
                    "actual": {
                        "stage": decision.to_stage,
                        "intent": decision.question_intent,
                        "weak_point": decision.weak_point,
                    },
                }
            )

    report = {
        "dataset": str(gold_path),
        "total": total,
        "metrics": {
            "stage_accuracy": round(stage_hits / total, 4) if total else 0.0,
            "intent_accuracy": round(intent_hits / total, 4) if total else 0.0,
            "weak_point_accuracy": round(weak_hits / total, 4) if total else 0.0,
        },
        "counts": {
            "stage_hits": stage_hits,
            "intent_hits": intent_hits,
            "weak_point_hits": weak_hits,
            "failed_cases": len(failures),
        },
        "failures": failures,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gold dataset accuracy for stage/intent/weak_point.")
    parser.add_argument(
        "--gold",
        default="tests/fixtures/gold_cases_v1.json",
        help="Path to gold dataset JSON file",
    )
    parser.add_argument(
        "--out",
        default="tests/reports/gold_eval_report_v1.json",
        help="Path to write report JSON",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = evaluate(gold_path)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    m = report["metrics"]
    print(f"Total cases: {report['total']}")
    print(f"Stage accuracy: {m['stage_accuracy']:.2%}")
    print(f"Intent accuracy: {m['intent_accuracy']:.2%}")
    print(f"Weak-point accuracy: {m['weak_point_accuracy']:.2%}")
    print(f"Failed cases: {report['counts']['failed_cases']}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
