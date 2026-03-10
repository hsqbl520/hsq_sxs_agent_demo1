from fastapi.testclient import TestClient
from app.main import app


def test_session_and_chat_turn_flow():
    with TestClient(app) as client:
        res = client.post("/api/v1/sessions", json={"user_id": "u1", "title": "test"})
        assert res.status_code == 200
        session_id = res.json()["session_id"]

        res2 = client.post(
            "/api/v1/chat/turn",
            json={"session_id": session_id, "user_text": "努力一定会成功", "client_turn_id": "x1"},
        )
        assert res2.status_code == 200
        body = res2.json()
        assert body["session_id"] == session_id
        assert body["question_intent"] in {
            "clarify_term",
            "ask_premise",
            "test_consistency",
            "probe_causality",
            "necessary_vs_sufficient",
            "counterexample",
            "value_priority",
            "operationalize",
        }
