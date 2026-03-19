from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _sqlite_columns(table_name: str) -> set[str]:
    with engine.begin() as conn:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def ensure_memory_schema() -> None:
    if not settings.database_url.startswith("sqlite"):
        return

    columns = _sqlite_columns("memory_records")
    if not columns:
        return

    additions = {
        "scope": "TEXT DEFAULT 'session'",
        "status": "TEXT DEFAULT 'active'",
        "profile_key": "TEXT",
        "origin_memory_id": "TEXT",
        "last_confirmed_at": "DATETIME",
        "promoted_at": "DATETIME",
    }
    with engine.begin() as conn:
        for column, ddl in additions.items():
            if column not in columns:
                conn.exec_driver_sql(f"ALTER TABLE memory_records ADD COLUMN {column} {ddl}")

        conn.exec_driver_sql(
            """
            UPDATE memory_records
            SET scope = CASE
                WHEN scope IS NULL OR scope = '' THEN CASE WHEN is_evergreen = 1 THEN 'durable' ELSE 'session' END
                ELSE scope
            END
            """
        )
        conn.exec_driver_sql(
            """
            UPDATE memory_records
            SET status = CASE
                WHEN status IS NULL OR status = '' THEN 'active'
                ELSE status
            END
            """
        )
        conn.exec_driver_sql(
            """
            UPDATE memory_records
            SET last_confirmed_at = CASE
                WHEN last_confirmed_at IS NULL AND scope = 'durable' THEN created_at
                ELSE last_confirmed_at
            END
            """
        )


def ensure_memory_fts() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_record_fts
            USING fts5(
                record_id UNINDEXED,
                session_id UNINDEXED,
                user_id UNINDEXED,
                source_type UNINDEXED,
                source_id UNINDEXED,
                kind UNINDEXED,
                is_evergreen UNINDEXED,
                search_text
            )
            """
        )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
