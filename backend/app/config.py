from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "DermaScan AI API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api"
    max_upload_mb: int = 10
    min_image_dimension: int = 224
    blur_threshold: float = 60.0
    min_brightness: float = 30.0
    max_brightness: float = 225.0
    confidence_threshold: float = 0.60
    min_skin_ratio: float = 0.08
    max_text_region_ratio: float = 0.02
    inference_mode: str = "placeholder"
    allow_smoke_model: bool = False
    model_path: Path = PROJECT_ROOT / "ml" / "outputs" / "models" / "dermascan-acne-scabies-psoriasis-efficientnet-b0.pt"
    model_name: str = "dermascan-placeholder"
    model_version: str = "dummy-v0.1.0"
    allowed_origins: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
    ]

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="DERMASCAN_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
