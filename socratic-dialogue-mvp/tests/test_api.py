from fastapi.testclient import TestClient
from app.config import settings
from app.main import app

settings.extractor_mode = "mock"
settings.generation_mode = "template"
settings.planner_mode = "rule"
settings.memory_embedding_mode = "hash"


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
        assert body["meta"]["memory_records_captured"] >= 1


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

        turn = client.post(
            "/api/v1/chat/turn",
            json={"session_id": session_id, "user_text": "正义一定只和效率有关"},
        )
        assert turn.status_code == 200
        summary = turn.json()["meta"]["retrieval_summary"]
        assert summary["has_doc_hits"] is True
        assert summary["retrieval_mode"] == "memory_hybrid"


def test_prior_definition_memory_is_recalled():
    with TestClient(app) as client:
        res = client.post("/api/v1/sessions", json={"user_id": "u_defs", "title": "defs"})
        assert res.status_code == 200
        session_id = res.json()["session_id"]

        first = client.post(
            "/api/v1/chat/turn",
            json={"session_id": session_id, "user_text": "成功是稳定地实现自己认可的长期目标"},
        )
        assert first.status_code == 200

        second = client.post(
            "/api/v1/chat/turn",
            json={"session_id": session_id, "user_text": "成功一定就是赚很多钱"},
        )
        assert second.status_code == 200
        body = second.json()
        summary = body["meta"]["retrieval_summary"]
        assert summary["has_prior_definition"] is True
        assert summary["retrieval_mode"] == "memory_hybrid"
        assert body["meta"]["memory_records_captured"] >= 1


def test_flush_promotes_durable_memory_and_profile_across_sessions():
    with TestClient(app) as client:
        first_session = client.post("/api/v1/sessions", json={"user_id": "u_profile", "title": "phase-1"})
        assert first_session.status_code == 200
        session_id = first_session.json()["session_id"]

        first_turn = client.post(
            "/api/v1/chat/turn",
            json={"session_id": session_id, "user_text": "请记住，成功是稳定地实现自己认可的长期目标。直接一点，别安慰我。"},
        )
        assert first_turn.status_code == 200

        closed = client.post(f"/api/v1/sessions/{session_id}/close")
        assert closed.status_code == 200

        snapshot = client.get(f"/api/v1/sessions/{session_id}/debug-snapshot")
        assert snapshot.status_code == 200
        memory = snapshot.json()["memory"]
        assert memory["durable_record_count"] >= 1
        assert memory["profile"]["dialogue_style"] == "direct_challenge"
        assert "成功" in memory["profile"]["stable_definitions"]

        second_session = client.post("/api/v1/sessions", json={"user_id": "u_profile", "title": "phase-2"})
        assert second_session.status_code == 200
        second_session_id = second_session.json()["session_id"]

        second_turn = client.post(
            "/api/v1/chat/turn",
            json={"session_id": second_session_id, "user_text": "成功一定就是赚很多钱"},
        )
        assert second_turn.status_code == 200
        body = second_turn.json()
        assert body["meta"]["retrieval_summary"]["has_profile"] is True
        assert body["meta"]["retrieval_summary"]["has_prior_definition"] is True
        assert body["meta"]["profile_snapshot"]["dialogue_style"] == "direct_challenge"


def test_memory_debug_api_returns_memory_layers_and_query_simulation():
    with TestClient(app) as client:
        session_res = client.post("/api/v1/sessions", json={"user_id": "u_memory_debug", "title": "memory-debug"})
        assert session_res.status_code == 200
        session_id = session_res.json()["session_id"]

        first_turn = client.post(
            "/api/v1/chat/turn",
            json={"session_id": session_id, "user_text": "请记住，成功是稳定地实现自己认可的长期目标。"},
        )
        assert first_turn.status_code == 200

        close_res = client.post(f"/api/v1/sessions/{session_id}/close")
        assert close_res.status_code == 200

        debug_res = client.get(
            f"/api/v1/sessions/{session_id}/memory/debug",
            params={"query": "成功一定就是赚很多钱", "session_limit": 5, "durable_limit": 5},
        )
        assert debug_res.status_code == 200
        body = debug_res.json()
        assert body["memory"]["durable_record_count"] >= 1
        assert body["memory"]["profile"] is not None
        assert body["query_debug"]["extraction"]["claim"] is not None
        assert body["query_debug"]["retrieval_summary"]["has_profile"] is True
        assert body["query_debug"]["retrieval_summary"]["has_prior_definition"] is True
        assert isinstance(body["query_debug"]["definition_hits"], list)


def test_memory_flush_preview_and_profile_diff():
    with TestClient(app) as client:
        session_res = client.post("/api/v1/sessions", json={"user_id": "u_flush_preview", "title": "flush-preview"})
        assert session_res.status_code == 200
        session_id = session_res.json()["session_id"]

        turn_res = client.post(
            "/api/v1/chat/turn",
            json={
                "session_id": session_id,
                "user_text": "请记住，成功是稳定地实现自己认可的长期目标。直接一点，别安慰我。",
            },
        )
        assert turn_res.status_code == 200

        preview_res = client.get(f"/api/v1/sessions/{session_id}/memory/flush-preview")
        assert preview_res.status_code == 200
        preview = preview_res.json()
        assert preview["flush_window"]["candidate_count"] >= 1
        assert preview["flush_window"]["would_promote_count"] >= 1
        assert any(candidate["decision"] == "promote" for candidate in preview["candidates"])
        assert any(candidate["reason_code"] == "explicit_memory" for candidate in preview["candidates"])
        assert preview["profile_before"] is None
        assert preview["profile_after"]["dialogue_style"] == "direct_challenge"
        assert "成功" in preview["profile_after"]["stable_definitions"]
        assert preview["profile_diff"]["has_changes"] is True
        assert "dialogue_style" in preview["profile_diff"]["changed_fields"]
        assert "stable_definitions" in preview["profile_diff"]["changed_fields"]

        diff_res = client.get(f"/api/v1/sessions/{session_id}/memory/profile-diff")
        assert diff_res.status_code == 200
        diff_body = diff_res.json()
        assert diff_body["profile_after"]["dialogue_style"] == "direct_challenge"
        assert diff_body["profile_diff"]["dialogue_style"]["after"] == "direct_challenge"
        assert "成功" in diff_body["profile_diff"]["stable_definitions"]["added"]
