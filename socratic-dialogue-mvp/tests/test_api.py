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
        assert body["meta"]["planner_source"] in {"llm_primary", "rule_fallback", "rule"}
        assert isinstance(body["meta"]["selected_evidence"], list)
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



def test_document_ingest_and_list():
    with TestClient(app) as client:
        res = client.post("/api/v1/sessions", json={"user_id": "u_docs", "title": "doc-test"})
        assert res.status_code == 200
        session_id = res.json()["session_id"]

        upload = client.post(
            f"/api/v1/sessions/{session_id}/documents",
            json={
                "title": "justice-note",
                "content": "正义并不完全取决于时效性。多年后平反虽然迟到，但仍然具有纠错价值。"
            },
        )
        assert upload.status_code == 200
        body = upload.json()
        assert body["session_id"] == session_id
        assert body["chunk_count"] >= 1

        listed = client.get(f"/api/v1/sessions/{session_id}/documents")
        assert listed.status_code == 200
        docs = listed.json()
        assert len(docs) == 1
        assert docs[0]["title"] == "justice-note"

        chunks = client.get(f"/api/v1/documents/{body['document_id']}/chunks")
        assert chunks.status_code == 200
        assert len(chunks.json()) >= 1
