# DermaScan AI API

Default development base URL: `http://127.0.0.1:8000`

Interactive OpenAPI documentation is available at `/docs` while the backend
is running.

## Health

`GET /health`

```json
{"status": "ok"}
```

## Predict

`POST /api/predict`

Content type: `multipart/form-data`

| Field | Required | Description |
| --- | --- | --- |
| `image` | Yes | JPG, JPEG, PNG, or WEBP image, at most 10 MB by default |
| `age_range` | No | Broad age range |
| `sex` | No | Optional user-provided context |
| `body_location` | No | Location of the skin concern |
| `symptom_duration` | No | Approximate duration |

Successful response:

```json
{
  "scan_id": "uuid",
  "status": "success",
  "top_prediction": {"class_name": "eczema", "confidence": 0.82},
  "top_k": [
    {"class_name": "eczema", "confidence": 0.82},
    {"class_name": "psoriasis", "confidence": 0.11},
    {"class_name": "tinea", "confidence": 0.05}
  ],
  "confidence_level": "confident",
  "risk_level": "low",
  "explanation": {
    "summary": "Non-diagnostic explanation",
    "next_steps": "Professional follow-up guidance",
    "warning": "Urgent symptom warning"
  },
  "disclaimer": "The scan result is not a diagnosis...",
  "model_version": "model version",
  "created_at": "ISO-8601 timestamp"
}
```

When confidence is below the configured threshold, `top_prediction` is `null`,
`confidence_level` is `uncertain`, and the top three possibilities remain
available.

Common errors:

- `400`: empty, unreadable, unsupported, oversized, or undersized image
- `422`: image is too dark, too bright, or blurry
- `503`: checkpoint mode is configured but the model is missing, invalid, or
  blocked because it is a smoke checkpoint

## Report

`GET /api/report/{scan_id}`

Returns `application/pdf`. Scan records and image bytes are process-local and
temporary; a report returns `404` after a server restart or record eviction.

## Model information

`GET /api/model-info`

Returns the provider mode, model availability, placeholder/smoke flags,
classes, input size, version, and production evaluation metrics when present.

## Privacy behavior

Uploaded images are decoded, orientation-corrected, converted to RGB, and
re-encoded without EXIF metadata. They are retained only in bounded process
memory to support the immediate result and PDF workflow.
