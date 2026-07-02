from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.app.config import Settings, get_settings
from backend.app.models.model_loader import ModelUnavailableError
from backend.app.schemas import PredictionResponse
from backend.app.services.explanation import DISCLAIMER, build_explanation
from backend.app.services.image_quality import validate_image
from backend.app.services.inference import predict as run_inference
from backend.app.storage.scans import scan_store


router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: Annotated[UploadFile, File(description="Skin concern image")],
    age_range: Annotated[str | None, Form()] = None,
    sex: Annotated[str | None, Form()] = None,
    body_location: Annotated[str | None, Form()] = None,
    symptom_duration: Annotated[str | None, Form()] = None,
) -> PredictionResponse:
    settings: Settings = get_settings()
    data = await image.read(settings.max_upload_mb * 1024 * 1024 + 1)

    try:
        validated = validate_image(data, settings)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not validated.quality.is_acceptable:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=validated.quality.reason)

    try:
        inference = run_inference(validated.image, validated.image_bytes, settings)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    explanation = build_explanation(
        inference.top_prediction.class_name if inference.top_prediction else None,
        inference.confidence_level == "uncertain",
    )
    record = scan_store.create(
        image_bytes=validated.image_bytes,
        top_prediction=inference.top_prediction,
        top_k=inference.top_k,
        confidence_level=inference.confidence_level,
        risk_level=inference.risk_level,
        explanation=explanation,
        disclaimer=DISCLAIMER,
        model_version=inference.model_version,
        metadata={
            "age_range": age_range,
            "sex": sex,
            "body_location": body_location,
            "symptom_duration": symptom_duration,
        },
    )
    return PredictionResponse(
        scan_id=record.scan_id,
        top_prediction=record.top_prediction,
        top_k=record.top_k,
        confidence_level=record.confidence_level,
        risk_level=record.risk_level,
        explanation=record.explanation,
        disclaimer=record.disclaimer,
        model_version=record.model_version,
        created_at=record.created_at,
    )
