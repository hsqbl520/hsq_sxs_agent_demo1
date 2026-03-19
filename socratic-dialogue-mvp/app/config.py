from pathlib import Path
from pydantic import BaseModel
import os
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseModel):
    app_name: str = "Socratic Dialogue MVP"
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./socratic.db")
    redis_url: str = os.getenv("REDIS_URL", "")
    summary_interval: int = int(os.getenv("SUMMARY_INTERVAL", "4"))
    extractor_mode: str = os.getenv("EXTRACTOR_MODE", "llm")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    generation_mode: str = os.getenv("GENERATION_MODE", "llm")
    generation_model: str = os.getenv("GENERATION_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    generation_temperature: float = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))
    planner_mode: str = os.getenv("PLANNER_MODE", "llm")
    planner_model: str = os.getenv("PLANNER_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    planner_temperature: float = float(os.getenv("PLANNER_TEMPERATURE", "0.2"))
    memory_embedding_mode: str = os.getenv("MEMORY_EMBEDDING_MODE", "auto")
    memory_embedding_model: str = os.getenv("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
    memory_embedding_dimensions: int = int(os.getenv("MEMORY_EMBEDDING_DIMENSIONS", "256"))
    memory_embedding_timeout_seconds: float = float(os.getenv("MEMORY_EMBEDDING_TIMEOUT_SECONDS", "3.0"))
    memory_chunk_chars: int = int(os.getenv("MEMORY_CHUNK_CHARS", "280"))
    memory_chunk_overlap_chars: int = int(os.getenv("MEMORY_CHUNK_OVERLAP_CHARS", "48"))
    memory_bm25_top_k: int = int(os.getenv("MEMORY_BM25_TOP_K", "12"))
    memory_vector_top_k: int = int(os.getenv("MEMORY_VECTOR_TOP_K", "12"))
    memory_hybrid_top_k: int = int(os.getenv("MEMORY_HYBRID_TOP_K", "8"))
    memory_vector_weight: float = float(os.getenv("MEMORY_VECTOR_WEIGHT", "0.55"))
    memory_bm25_weight: float = float(os.getenv("MEMORY_BM25_WEIGHT", "0.30"))
    memory_importance_weight: float = float(os.getenv("MEMORY_IMPORTANCE_WEIGHT", "0.10"))
    memory_stability_weight: float = float(os.getenv("MEMORY_STABILITY_WEIGHT", "0.05"))
    memory_half_life_days: float = float(os.getenv("MEMORY_HALF_LIFE_DAYS", "14"))
    memory_document_half_life_days: float = float(os.getenv("MEMORY_DOCUMENT_HALF_LIFE_DAYS", "30"))
    memory_mmr_lambda: float = float(os.getenv("MEMORY_MMR_LAMBDA", "0.75"))
    memory_flush_lookback_turns: int = int(os.getenv("MEMORY_FLUSH_LOOKBACK_TURNS", "8"))
    memory_promotion_repeat_threshold: int = int(os.getenv("MEMORY_PROMOTION_REPEAT_THRESHOLD", "2"))


settings = Settings()
