def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_marks_placeholder(client) -> None:
    response = client.get("/api/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_placeholder"] is True
    assert payload["inference_mode"] == "placeholder"
    assert payload["model_available"] is True
    assert payload["classes"] == ["acne", "eczema", "tinea", "scabies", "psoriasis", "other"]


def test_predict_and_download_report(client, valid_image_bytes) -> None:
    response = client.post(
        "/api/predict",
        files={"image": ("skin.jpg", valid_image_bytes, "image/jpeg")},
        data={"body_location": "forearm", "symptom_duration": "2 weeks"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert len(payload["top_k"]) == 3
    assert payload["model_version"].startswith("dummy-")
    assert "not a diagnosis" in payload["disclaimer"]

    report = client.get(f'/api/report/{payload["scan_id"]}')
    assert report.status_code == 200
    assert report.headers["content-type"] == "application/pdf"
    assert report.content.startswith(b"%PDF")


def test_rejects_dark_image(client, dark_image_bytes) -> None:
    response = client.post(
        "/api/predict",
        files={"image": ("dark.jpg", dark_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 422
    assert "too dark" in response.json()["detail"].lower()


def test_rejects_non_image(client) -> None:
    response = client.post(
        "/api/predict",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "readable image" in response.json()["detail"].lower()


def test_missing_report_returns_404(client) -> None:
    response = client.get("/api/report/not-a-real-scan")
    assert response.status_code == 404
