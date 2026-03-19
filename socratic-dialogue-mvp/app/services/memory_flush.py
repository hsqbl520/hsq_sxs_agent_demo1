from datetime import datetime

from sqlalchemy.orm import Session

from app.config import settings
from app.models import MemoryRecord, Session as ChatSession, UserProfile
from .memory_store import _materialize_records
from .profile_compiler import build_profile_snapshot, compile_user_profile


PROMOTABLE_KINDS = {"definition", "value", "preference", "philosophy", "goal", "claim"}


def _normalized(value: str | None) -> str:
    return "".join((value or "").split())


def _group_key(record: MemoryRecord) -> tuple[str, str, str]:
    meta = record.meta or {}
    anchor = record.term or meta.get("style_key") or record.profile_key or _normalized(record.text)
    return record.kind, anchor, record.profile_key or ""


def _repeat_count(db: Session, user_id: str, key: tuple[str, str, str]) -> int:
    candidates = (
        db.query(MemoryRecord)
        .filter(
            MemoryRecord.user_id == user_id,
            MemoryRecord.scope == "session",
            MemoryRecord.kind == key[0],
            MemoryRecord.status == "active",
        )
        .all()
    )
    return sum(1 for record in candidates if _group_key(record) == key)


def _session_candidates(db: Session, session_id: str) -> tuple[ChatSession | None, list[MemoryRecord]]:
    session = db.get(ChatSession, session_id)
    if not session:
        return None, []

    candidates = (
        db.query(MemoryRecord)
        .filter(
            MemoryRecord.session_id == session_id,
            MemoryRecord.scope == "session",
            MemoryRecord.status == "active",
        )
        .order_by(MemoryRecord.created_at.desc())
        .limit(settings.memory_flush_lookback_turns * 4)
        .all()
    )
    return session, candidates


def _promotion_basis(db: Session, record: MemoryRecord) -> tuple[bool, str, int]:
    repeat_count = _repeat_count(db, record.user_id or "", _group_key(record))
    if record.kind not in PROMOTABLE_KINDS:
        return False, "kind_not_promotable", repeat_count
    meta = record.meta or {}
    if meta.get("explicit_memory"):
        return True, "explicit_memory", repeat_count
    if record.kind == "claim":
        if record.stability >= 0.72 and repeat_count >= settings.memory_promotion_repeat_threshold:
            return True, "claim_repeat_and_stability", repeat_count
        return False, "claim_below_repeat_or_stability", repeat_count
    if record.stability >= 0.84:
        return True, "high_stability", repeat_count
    if repeat_count >= settings.memory_promotion_repeat_threshold:
        return True, "repeat_threshold", repeat_count
    return False, "below_promotion_threshold", repeat_count


def _existing_durable(db: Session, user_id: str, record: MemoryRecord) -> MemoryRecord | None:
    durable_rows = (
        db.query(MemoryRecord)
        .filter(
            MemoryRecord.user_id == user_id,
            MemoryRecord.scope == "durable",
            MemoryRecord.kind == record.kind,
            MemoryRecord.status != "archived",
        )
        .all()
    )
    for durable in durable_rows:
        same_term = (durable.term or "") == (record.term or "")
        same_text = _normalized(durable.text) == _normalized(record.text)
        same_profile_key = (durable.profile_key or "") == (record.profile_key or "")
        if same_text or (same_term and same_profile_key):
            return durable
    return None


def _reason_text(reason_code: str, repeat_count: int) -> str:
    if reason_code == "explicit_memory":
        return "命中显式长期记忆指令。"
    if reason_code == "claim_repeat_and_stability":
        return f"主张稳定性达标，且近期待重复次数为 {repeat_count}。"
    if reason_code == "high_stability":
        return "稳定性分数足够高，可直接晋升为 durable memory。"
    if reason_code == "repeat_threshold":
        return f"近期重复出现 {repeat_count} 次，达到晋升阈值。"
    if reason_code == "duplicate_durable":
        return "durable memory 中已有同类稳定记录，本次只做确认。"
    if reason_code == "claim_below_repeat_or_stability":
        return f"claim 的稳定性或重复次数不足，当前重复次数为 {repeat_count}。"
    if reason_code == "kind_not_promotable":
        return "当前 kind 不在 durable promotion 范围内。"
    return "当前记录未达到 durable promotion 阈值。"


def _preview_entry(db: Session, session: ChatSession, record: MemoryRecord) -> dict:
    should_promote, reason_code, repeat_count = _promotion_basis(db, record)
    existing = _existing_durable(db, session.user_id, record) if should_promote else None
    decision = "skip"
    if should_promote and existing:
        decision = "confirm_existing"
        reason_code = "duplicate_durable"
    elif should_promote:
        decision = "promote"

    return {
        "record": record,
        "decision": decision,
        "reason_code": reason_code,
        "reason": _reason_text(reason_code, repeat_count),
        "repeat_count": repeat_count,
        "matched_durable_id": None if not existing else existing.id,
    }


def _preview_promoted_records(session: ChatSession, entries: list[dict]) -> list[MemoryRecord]:
    preview_records: list[MemoryRecord] = []
    for entry in entries:
        if entry["decision"] != "promote":
            continue
        record = entry["record"]
        preview_records.append(
            MemoryRecord(
                id=f"preview:{record.id}",
                user_id=session.user_id,
                session_id=None,
                source_type="memory_flush_preview",
                source_id=session.id,
                scope="durable",
                status="preview",
                chunk_index=record.chunk_index,
                kind=record.kind,
                term=record.term,
                profile_key=record.profile_key,
                origin_memory_id=record.id,
                text=record.text,
                search_text=record.search_text,
                importance=max(record.importance, 0.72),
                confidence=record.confidence,
                stability=max(record.stability, 0.86),
                is_evergreen=True,
                embedding=None,
                embedding_model=None,
                embedding_source="preview",
                created_at=record.created_at,
                last_confirmed_at=record.created_at,
                promoted_at=record.created_at,
                access_count=0,
                meta={
                    **(record.meta or {}),
                    "promoted_from_session_id": session.id,
                    "promoted_from_memory_id": record.id,
                    "preview_only": True,
                },
            )
        )
    return preview_records


def _serialize_profile_snapshot(profile: UserProfile | None) -> dict | None:
    if not profile:
        return None
    return {
        "dialogue_style": profile.dialogue_style,
        "stable_definitions": profile.stable_definitions,
        "value_hierarchy": profile.value_hierarchy,
        "philosophical_tendency": profile.philosophical_tendency,
        "long_term_goals": profile.long_term_goals,
        "constraints": profile.constraints,
        "source_memory_ids": profile.source_memory_ids,
        "updated_at": profile.updated_at,
    }


def _profile_diff(before: dict | None, after: dict | None) -> dict:
    before = before or {}
    after = after or {}
    diff: dict = {"has_changes": False, "changed_fields": []}

    def _mark(field: str, payload: dict) -> None:
        diff["has_changes"] = True
        diff["changed_fields"].append(field)
        diff[field] = payload

    if before.get("dialogue_style") != after.get("dialogue_style"):
        _mark(
            "dialogue_style",
            {"before": before.get("dialogue_style"), "after": after.get("dialogue_style")},
        )

    before_defs = before.get("stable_definitions") or {}
    after_defs = after.get("stable_definitions") or {}
    def_added = {key: value for key, value in after_defs.items() if key not in before_defs}
    def_removed = {key: value for key, value in before_defs.items() if key not in after_defs}
    def_changed = {
        key: {"before": before_defs[key], "after": after_defs[key]}
        for key in before_defs.keys() & after_defs.keys()
        if before_defs[key] != after_defs[key]
    }
    if def_added or def_removed or def_changed:
        _mark(
            "stable_definitions",
            {"added": def_added, "removed": def_removed, "changed": def_changed},
        )

    for field in ("value_hierarchy", "philosophical_tendency", "long_term_goals", "constraints"):
        before_values = before.get(field) or []
        after_values = after.get(field) or []
        added = [value for value in after_values if value not in before_values]
        removed = [value for value in before_values if value not in after_values]
        if added or removed:
            _mark(field, {"added": added, "removed": removed})

    return diff


def build_flush_preview(db: Session, session_id: str) -> dict:
    session, candidates = _session_candidates(db, session_id)
    if not session:
        return {
            "candidate_count": 0,
            "would_promote_count": 0,
            "would_confirm_existing_count": 0,
            "would_skip_count": 0,
            "candidates": [],
            "profile_before": None,
            "profile_after": None,
            "profile_diff": {"has_changes": False, "changed_fields": []},
        }

    entries = [_preview_entry(db, session, record) for record in reversed(candidates)]
    profile_before_obj = db.query(UserProfile).filter(UserProfile.user_id == session.user_id).first()
    profile_before = _serialize_profile_snapshot(profile_before_obj)

    durable_records = (
        db.query(MemoryRecord)
        .filter(
            MemoryRecord.user_id == session.user_id,
            MemoryRecord.scope == "durable",
            MemoryRecord.status != "archived",
        )
        .order_by(MemoryRecord.created_at.desc())
        .all()
    )
    preview_records = _preview_promoted_records(session, entries)
    profile_after = None
    if durable_records or preview_records:
        ordered_records = durable_records + preview_records
        ordered_records.sort(key=lambda record: record.created_at, reverse=True)
        profile_after = build_profile_snapshot(ordered_records)

    return {
        "candidate_count": len(entries),
        "would_promote_count": sum(1 for entry in entries if entry["decision"] == "promote"),
        "would_confirm_existing_count": sum(1 for entry in entries if entry["decision"] == "confirm_existing"),
        "would_skip_count": sum(1 for entry in entries if entry["decision"] == "skip"),
        "candidates": entries,
        "profile_before": profile_before,
        "profile_after": profile_after,
        "profile_diff": _profile_diff(profile_before, profile_after),
    }


def _promote_record(db: Session, session: ChatSession, record: MemoryRecord, promoted_at: datetime) -> MemoryRecord | None:
    existing = _existing_durable(db, session.user_id, record)
    if existing:
        existing.last_confirmed_at = promoted_at
        existing.access_count = (existing.access_count or 0) + 1
        existing.confidence = max(existing.confidence, record.confidence)
        existing.stability = max(existing.stability, record.stability)
        existing.importance = max(existing.importance, record.importance)
        record.status = "promoted"
        record.promoted_at = promoted_at
        return None

    payload = {
        "kind": record.kind,
        "term": record.term,
        "profile_key": record.profile_key,
        "origin_memory_id": record.id,
        "text": record.text,
        "importance": max(record.importance, 0.72),
        "confidence": record.confidence,
        "stability": max(record.stability, 0.86),
        "last_confirmed_at": promoted_at,
        "promoted_at": promoted_at,
        "meta": {
            **(record.meta or {}),
            "promoted_from_session_id": session.id,
            "promoted_from_memory_id": record.id,
        },
    }
    records = _materialize_records(
        db,
        session_id=None,
        user_id=session.user_id,
        source_type="memory_flush",
        source_id=session.id,
        payloads=[payload],
        extra_terms=[record.term] if record.term else None,
        scope="durable",
        status="active",
        is_evergreen=True,
    )
    record.status = "promoted"
    record.promoted_at = promoted_at
    return records[0] if records else None


def flush_session_memory(db: Session, session_id: str) -> dict:
    preview = build_flush_preview(db, session_id)
    session = db.get(ChatSession, session_id)
    if not session:
        return {"promoted_count": 0, "profile_updated": False}

    promoted: list[MemoryRecord] = []
    confirmed_ids: set[str] = set()
    promoted_at = datetime.utcnow()
    for entry in preview["candidates"]:
        if entry["decision"] == "skip":
            continue
        result = _promote_record(db, session, entry["record"], promoted_at)
        if result:
            promoted.append(result)
        elif entry["decision"] == "confirm_existing" and entry["matched_durable_id"]:
            confirmed_ids.add(entry["matched_durable_id"])

    profile: UserProfile | None = None
    durable_exists = (
        db.query(MemoryRecord.id)
        .filter(MemoryRecord.user_id == session.user_id, MemoryRecord.scope == "durable")
        .first()
        is not None
    )
    if promoted or confirmed_ids or durable_exists or db.query(UserProfile).filter(UserProfile.user_id == session.user_id).first():
        profile = compile_user_profile(db, session.user_id)

    return {
        "promoted_count": len(promoted),
        "promoted_ids": [record.id for record in promoted],
        "confirmed_existing_count": len(confirmed_ids),
        "confirmed_existing_ids": sorted(confirmed_ids),
        "profile_updated": bool(profile),
        "dialogue_style": None if not profile else profile.dialogue_style,
    }
