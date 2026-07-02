from datetime import datetime

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class Explanation(BaseModel):
    summary: str
    next_steps: str
    warning: str


class PredictionResponse(BaseModel):
    scan_id: str
    status: str = "success"
    top_prediction: PredictionItem | None
    top_k: list[PredictionItem]
    confidence_level: str
    risk_level: str
    explanation: Explanation
    disclaimer: str
    model_version: str
    created_at: datetime


class QualityResult(BaseModel):
    is_acceptable: bool
    reason: str | None = None
    width: int | None = None
    height: int | None = None
    brightness: float | None = None
    blur_score: float | None = None


class ModelInfo(BaseModel):
    model_name: str
    version: str
    classes: list[str]
    input_size: list[int]
    is_placeholder: bool
    is_smoke_test: bool = False
    model_available: bool = True
    inference_mode: str
    metrics: dict[str, float | None]
    last_trained: datetime | None


class HealthResponse(BaseModel):
    status: str
