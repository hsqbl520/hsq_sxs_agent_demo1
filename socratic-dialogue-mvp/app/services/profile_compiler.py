from datetime import datetime

from sqlalchemy.orm import Session

from app.models import MemoryRecord, UserProfile


def _unique_texts(values: list[str], limit: int = 6) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = (value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        ordered.append(cleaned)
        seen.add(cleaned)
        if len(ordered) >= limit:
            break
    return ordered


def build_profile_snapshot(records: list[MemoryRecord]) -> dict:
    stable_definitions: dict[str, str] = {}
    value_hierarchy: list[str] = []
    long_term_goals: list[str] = []
    philosophy: list[str] = []
    constraints: list[str] = []
    source_memory_ids: list[str] = []
    direct_votes = 0
    gentle_votes = 0

    for record in records:
        source_memory_ids.append(record.id)
        if record.kind == "definition" and record.term and record.term not in stable_definitions:
            stable_definitions[record.term] = record.text
        elif record.kind == "value":
            value_hierarchy.append(record.text)
        elif record.kind == "goal":
            long_term_goals.append(record.text)
        elif record.kind == "philosophy":
            school = (record.meta or {}).get("school") or record.term or record.text
            philosophy.append(school)
        elif record.kind == "preference":
            style_key = (record.meta or {}).get("style_key")
            if style_key == "direct_challenge":
                direct_votes += 1
            elif style_key == "gentle_probe":
                gentle_votes += 1
            constraint = (record.meta or {}).get("constraint")
            if constraint:
                constraints.append(constraint)

    dialogue_style = "balanced_socratic"
    if direct_votes > gentle_votes and direct_votes > 0:
        dialogue_style = "direct_challenge"
    elif gentle_votes > 0:
        dialogue_style = "gentle_probe"

    return {
        "dialogue_style": dialogue_style,
        "stable_definitions": stable_definitions,
        "value_hierarchy": _unique_texts(value_hierarchy),
        "long_term_goals": _unique_texts(long_term_goals),
        "philosophical_tendency": _unique_texts(philosophy),
        "constraints": _unique_texts(constraints),
        "source_memory_ids": source_memory_ids[:24],
    }


def compile_user_profile(db: Session, user_id: str) -> UserProfile:
    records = (
        db.query(MemoryRecord)
        .filter(
            MemoryRecord.user_id == user_id,
            MemoryRecord.scope == "durable",
            MemoryRecord.is_evergreen.is_(True),
            MemoryRecord.status != "archived",
        )
        .order_by(MemoryRecord.created_at.desc())
        .all()
    )
    snapshot = build_profile_snapshot(records)

    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if not profile:
        profile = UserProfile(user_id=user_id)
        db.add(profile)

    profile.dialogue_style = snapshot["dialogue_style"]
    profile.stable_definitions = snapshot["stable_definitions"]
    profile.value_hierarchy = snapshot["value_hierarchy"]
    profile.long_term_goals = snapshot["long_term_goals"]
    profile.philosophical_tendency = snapshot["philosophical_tendency"]
    profile.constraints = snapshot["constraints"]
    profile.source_memory_ids = snapshot["source_memory_ids"]
    profile.updated_at = datetime.utcnow()
    db.flush()
    return profile
