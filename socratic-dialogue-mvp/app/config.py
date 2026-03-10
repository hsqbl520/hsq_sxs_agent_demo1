from pydantic import BaseModel
import os


class Settings(BaseModel):
    app_name: str = "Socratic Dialogue MVP"
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./socratic.db")
    redis_url: str = os.getenv("REDIS_URL", "")
    summary_interval: int = int(os.getenv("SUMMARY_INTERVAL", "4"))
    extractor_mode: str = os.getenv("EXTRACTOR_MODE", "llm")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")


settings = Settings()
