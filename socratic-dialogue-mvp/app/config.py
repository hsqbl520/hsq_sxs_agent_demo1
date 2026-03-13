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


settings = Settings()
