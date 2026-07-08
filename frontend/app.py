from base64 import b64encode
from dataclasses import dataclass
from html import escape
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
try:
    from streamlit_cropper import st_cropper
except ModuleNotFoundError:
    st_cropper = None
try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None

from frontend.api import ApiError, download_report, get_model_info, health_check, predict_image
from frontend.styles import apply_styles


RESULTS_DIR = PROJECT_ROOT / "results"
ASSET_DIR = PROJECT_ROOT / "frontend" / "assets"
SUPPORTED_CLASSES = ["acne", "eczema", "psoriasis"]

DISCLAIMER = (
    "The scan result is not a diagnosis. For accurate diagnosis and treatment "
    "recommendations, consult a qualified healthcare professional or dermatologist."
)

DISEASE_DESCRIPTIONS = {
    "acne": "Acne may show blocked or inflamed pores, blackheads, whiteheads, pimples, oily skin, or scarring. The AI compares visible bump patterns and texture, but a clinician must confirm the cause.",
    "eczema": "Eczema may show dry, itchy, cracked, irritated, or inflamed patches. The AI compares dryness, scaling, redness or darkening, and patch-like irritation across skin tones.",
    "psoriasis": "Psoriasis may show raised, scaly, itchy, or inflamed plaques. The AI compares scale, plaque-like texture, and clearly bounded patch patterns.",
}


def _supported_classes(info: dict[str, Any] | None = None) -> list[str]:
    """Keep the public UI aligned with the trained three-class model."""
    active = (info or {}).get("classes") or SUPPORTED_CLASSES
    normalized = [str(name).strip().lower() for name in active]
    if set(normalized) != set(SUPPORTED_CLASSES):
        return SUPPORTED_CLASSES
    return [name for name in SUPPORTED_CLASSES if name in normalized]


def _asset_data_uri(filename: str, mime_type: str = "image/jpeg") -> str:
    path = ASSET_DIR / filename
    if not path.exists():
        return ""
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

URGENT_WARNING = (
    "Seek urgent medical care if the rash is rapidly spreading, painful, bleeding, "
    "infected, or associated with fever."
)

# Bundled baseline keeps Streamlit Cloud informative even when generated
# evaluation files are not deployed. Replace by committing results/model_metrics.json
# after a full evaluation run.
DEFAULT_PERFORMANCE_METRICS: dict[str, Any] = {
    "model_name": "DermaScan AI acne-eczema-psoriasis custom CNN",
    "class_names": ["acne", "eczema", "psoriasis"],
    "sample_count": 264,
    "accuracy": 0.6212,
    "validation_accuracy": 0.6515,
    "weighted_f1": 0.5922,
    "source_note": "Bundled fallback metrics for the retrained three-class prototype.",
}


@dataclass(slots=True)
class SelectedImage:
    data: bytes
    filename: str
    content_type: str


@dataclass(slots=True)
class CroppedImage:
    data: bytes
    filename: str
    content_type: str
    size: tuple[int, int]


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "page": "home",
        "result": None,
        "scan_image": None,
        "report_bytes": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _page_label(page: str) -> str:
    labels = {
        "home": "Home",
        "scan": "Scan Image",
        "performance": "Model Performance",
        "about": "About",
    }
    return labels.get(page, "Home")


def render_sidebar_navigation() -> None:
    if st.session_state.page == "result":
        return
    pages = ["Home", "Scan Image", "Model Performance", "About"]
    selected = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(_page_label(st.session_state.page)),
    )
    page_map = {
        "Home": "home",
        "Scan Image": "scan",
        "Model Performance": "performance",
        "About": "about",
    }
    st.session_state.page = page_map[selected]


def navigate(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def render_brand() -> None:
    st.markdown(
        """
        <div class="brand-bar">
          <div class="brand-lockup">
            <div class="brand-mark">D</div>
            <div>
              <div class="brand-name">DermaScan AI</div>
              <div class="brand-subtitle">Skin screening support</div>
            </div>
          </div>
          <div class="nav-badge">Education &amp; decision support only</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar() -> None:
    st.markdown('<div class="top-strip"></div>', unsafe_allow_html=True)


def render_disclaimer() -> None:
    st.markdown(
        f'<div class="disclaimer"><strong>Important:</strong> {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if 0 <= number <= 1:
        return f"{number * 100:.1f}%"
    return f"{number:.3g}"


def render_model_metrics(info: dict[str, Any] | None, title: str = "Model Performance Metrics") -> None:
    performance_metrics = _read_performance_metrics() or {}
    info_metrics = (info or {}).get("metrics") or {}
    metrics = {**info_metrics, **performance_metrics}
    metric_items = [
        ("Accuracy", metrics.get("accuracy")),
        ("Validation Accuracy", metrics.get("validation_accuracy")),
        ("Weighted F1-score", metrics.get("weighted_f1")),
    ]
    metric_items = [(label, value) for label, value in metric_items if value is not None]
    st.markdown(f'<h2 class="section-title">{escape(title)}</h2>', unsafe_allow_html=True)
    if not metric_items:
        st.markdown(
            """
            <div class="model-metrics">
              <div class="metric-card"><span>Accuracy</span><strong>N/A</strong><p>No saved evaluation metrics found for the trained checkpoint.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    source_note = metrics.get("source_note")
    if source_note:
        st.markdown(
            f'<div class="status-note"><strong>Metrics source:</strong> {escape(str(source_note))}</div>',
            unsafe_allow_html=True,
        )
    cards = "".join(
        f'<div class="metric-card"><span>{escape(label)}</span><strong>{_format_metric(value)}</strong></div>'
        for label, value in metric_items
    )
    st.markdown(f'<div class="model-metrics">{cards}</div>', unsafe_allow_html=True)


def render_model_notice() -> dict[str, Any] | None:
    info = get_model_info()
    if info is None:
        st.markdown(
            '<div class="placeholder-note"><strong>Model status unavailable:</strong> The backend model metadata could not be loaded.</div>',
            unsafe_allow_html=True,
        )
    elif info.get("is_placeholder"):
        st.markdown(
            '<div class="placeholder-note"><strong>Development mode:</strong> Predictions currently come from a deterministic placeholder, not a trained medical model.</div>',
            unsafe_allow_html=True,
        )
    elif not info.get("model_available"):
        st.markdown(
            '<div class="placeholder-note"><strong>Model unavailable:</strong> The configured checkpoint is missing, invalid, or blocked as a smoke model.</div>',
            unsafe_allow_html=True,
        )
    elif info.get("is_smoke_test"):
        st.markdown(
            '<div class="placeholder-note"><strong>Smoke model active:</strong> This checkpoint is only for development verification and is not a research result.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-note"><strong>Model active:</strong> {info["model_name"]} ({info["version"]})</div>',
            unsafe_allow_html=True,
        )
    return info


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_performance_metrics() -> dict[str, Any] | None:
    candidate_paths = [
        RESULTS_DIR / "model_metrics.json",
        PROJECT_ROOT / "ml" / "outputs" / "reports" / "acne_eczema_psoriasis" / "evaluation_metrics.json",
    ]
    for path in candidate_paths:
        metrics = _load_json(path)
        if metrics:
            return metrics
    return DEFAULT_PERFORMANCE_METRICS


def _classification_report_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    per_class = metrics.get("per_class") or {}
    rows: list[dict[str, Any]] = []
    for class_name, values in per_class.items():
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "class_name": class_name,
                "precision": values.get("precision"),
                "recall": values.get("recall", values.get("recall_sensitivity")),
                "f1_score": values.get("f1_score"),
                "support": values.get("support"),
            }
        )
    return pd.DataFrame(rows)


def _confusion_matrix_dataframe(metrics: dict[str, Any], normalized: bool = False) -> pd.DataFrame | None:
    matrix = metrics.get("confusion_matrix")
    class_names = metrics.get("class_names") or SUPPORTED_CLASSES
    if set(str(name).lower() for name in class_names) != set(SUPPORTED_CLASSES):
        class_names = SUPPORTED_CLASSES
    if not matrix:
        return None
    frame = pd.DataFrame(
        matrix,
        index=[name.title() for name in class_names],
        columns=[name.title() for name in class_names],
    )
    if normalized:
        row_sums = frame.sum(axis=1).replace(0, 1)
        frame = frame.div(row_sums, axis=0).round(3)
    return frame


def _performance_file(name: str) -> Path | None:
    candidate_paths = [
        RESULTS_DIR / name,
        PROJECT_ROOT / "ml" / "outputs" / "reports" / "acne_eczema_psoriasis" / name,
        PROJECT_ROOT / "ml" / "outputs" / "plots" / "acne_eczema_psoriasis" / name,
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return None


def _metric_value(metrics: dict[str, Any] | None, *keys: str) -> float | None:
    if not metrics:
        return None
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def render_probability_chart(top_k: list[dict[str, Any]]) -> None:
    if not top_k:
        return
    chart_data = pd.DataFrame(
        {
            "Class": [item["class_name"].title() for item in top_k],
            "Probability": [float(item["confidence"]) * 100 for item in top_k],
        }
    )
    if px is None:
        st.markdown("### Model probabilities")
        st.bar_chart(chart_data.set_index("Class"))
        return
    figure = px.bar(
        chart_data,
        x="Class",
        y="Probability",
        text=chart_data["Probability"].map(lambda value: f"{value:.1f}%"),
        color="Class",
        color_discrete_sequence=["#295289", "#FF6324", "#060C40", "#EA9118"],
    )
    figure.update_layout(
        title="Model Confidence Breakdown",
        yaxis_title="Probability (%)",
        xaxis_title="Possible condition",
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="rgba(245,249,255,0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#2B3C4C"),
        title_font=dict(size=15, color="#060C40"),
    )
    figure.update_yaxes(range=[0, 100], gridcolor="rgba(210,228,242,0.6)")
    figure.update_xaxes(showgrid=False)
    st.plotly_chart(figure, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────

def render_home() -> None:
    render_brand()
    model_info = get_model_info()
    class_names = _supported_classes(model_info)
    classes_display = ", ".join(name.title() for name in class_names)
    hero_image = _asset_data_uri("landing-hero-clinical.jpg")
    gallery_images = [
        _asset_data_uri("landing-derm-01.jpg"),
        _asset_data_uri("landing-derm-02.jpg"),
        _asset_data_uri("landing-derm-03.jpg"),
        _asset_data_uri("landing-derm-04.jpg"),
    ]

    st.markdown(
        f'<div class="landing-bg"><img src="{hero_image}" alt="Dermatology AI background"></div>',
        unsafe_allow_html=True,
    )

    # ── Split welcome section (Aidoc "See what matters" style) ─
    st.markdown(
        '<div class="welcome-split landing-card-panel"><div class="welcome-split-text"><div class="headline-wrap"><span class="orange-bar"></span><h2>Welcome to DermaScan AI.<br>Clear support for skin screening.</h2></div><p>Start here: upload or capture a skin image, crop the area you want checked, and review an educational AI result designed to support safer dermatology decisions.</p><div class="trusted-note">Educational and decision support tool only, not a medical diagnosis</div></div><div class="welcome-split-image"></div></div><div id="scan-start"></div>',
        unsafe_allow_html=True,
    )
    if st.button("Get started →", type="primary"):
        navigate("scan")

    # ── Hero — full-bleed Aidoc style ─────────────────────────
    st.markdown(
        f'<div class="hero-fullbleed"><img class="hero-bg-img" src="{hero_image}" alt="Doctor using digital healthcare tools"><h1>AI Solutions Deliver<br>Smarter Skin Screening</h1><p>The pressure on medical professionals to provide quality and effective care is enormous. DermaScan AI uses deep learning to help support dermatological triage for {escape(classes_display)} and turns images into actionable educational insights.</p><a class="hero-cta-btn" href="#scan-start">SEE THE SOLUTION</a></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="clinical-gallery"><div class="clinical-gallery-copy"><h3>Clinical photos, calmer decisions.</h3><p>Dermatology workflows depend on clear visual review. These clinical images create the same calm healthcare style as your reference design while keeping the project focused on screening support.</p></div><div class="clinical-gallery-grid" aria-label="Dermatology consultation image gallery">'
        + "".join(
            f'<div class="clinical-shot"><img src="{image_uri}" alt="Clinical dermatology support image"></div>'
            for image_uri in gallery_images
            if image_uri
        )
        + "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Stats strip ───────────────────────────────────────────
    st.markdown(
        f"""
        <div class="stats-strip">
          <div class="stat-cell">
            <span class="stat-value">{len(class_names)}</span>
            <span class="stat-label">Supported conditions</span>
          </div>
          <div class="stat-cell">
            <span class="stat-value">AI</span>
            <span class="stat-label">Deep learning model</span>
          </div>
          <div class="stat-cell">
            <span class="stat-value">100%</span>
            <span class="stat-label">Educational use only</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Supported conditions ──────────────────────────────────
    st.markdown(
        """
        <div class="section-header landing-section-header">
          <h2>Supported Conditions</h2>
          <p>The model is trained to classify the following skin conditions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards = "".join(
        f'<div class="disease-card"><h3>{escape(class_name.title())}</h3><p>{escape(DISEASE_DESCRIPTIONS.get(class_name, "Supported model class for educational screening."))}</p></div>'
        for class_name in class_names
    )
    st.markdown(f'<div class="disease-grid">{cards}</div>', unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header landing-section-header">
          <h2>How It Works</h2>
          <p>Three simple steps to receive an educational AI prediction.</p>
        </div>
        <div class="workflow-grid">
          <div class="info-card">
            <strong>1</strong>
            <h3>Upload a clear skin image</h3>
            <p>Use JPG, JPEG, PNG, WEBP or your camera. Ensure good lighting and a focused view of the affected area.</p>
          </div>
          <div class="info-card">
            <strong>2</strong>
            <h3>AI model analyzes the image</h3>
            <p>The backend checks image quality and runs the trained deep learning model on your cropped selection.</p>
          </div>
          <div class="info-card">
            <strong>3</strong>
            <h3>Review prediction &amp; guidance</h3>
            <p>See the possible condition, confidence score, probability breakdown, and next-step safety guidance.</p>
          </div>
        </div>
        <div class="warning-card landing-warning-card">
          <strong>⚠ Educational use only:</strong> This system is not a medical diagnosis.
          Please consult a qualified health professional for diagnosis and treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS — image handling
# ─────────────────────────────────────────────────────────────

def _selected_image(uploaded_file: Any, camera_file: Any) -> SelectedImage | None:
    source = camera_file or uploaded_file
    if source is None:
        return None
    content_type = getattr(source, "type", None) or "image/jpeg"
    filename = getattr(source, "name", None) or "camera-capture.jpg"
    return SelectedImage(source.getvalue(), filename, content_type)


def _load_image(image_bytes: bytes) -> Image.Image | None:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95, optimize=True)
    return buffer.getvalue()


def _render_cropper(selected: SelectedImage) -> CroppedImage | None:
    image = _load_image(selected.data)
    if image is None:
        st.error("This image could not be opened. Please upload a clear JPG, PNG, or WEBP file.")
        return None
    if st_cropper is None:
        st.error(
            "The interactive crop tool is not available in the Python environment running Streamlit. "
            "Install dependencies with `pip install -r requirements.txt`, then stop and restart the app. "
            "If `.venv` fails, reinstall Python 3.11 and recreate the virtual environment."
        )
        return None

    render_top_bar()
    st.markdown(
        """
        <div class="crop-heading">
          <h2>Crop the area of concern</h2>
          <p>Drag and resize the crop box on the image. The model will analyze only the selected area.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="crop-preview-frame">Crop your photo below</div>', unsafe_allow_html=True)
    cropped = st_cropper(
        image,
        realtime_update=True,
        box_color="#FF6324",
        aspect_ratio=None,
        return_type="image",
        key=f"cropper-{selected.filename}-{len(selected.data)}",
    )
    if cropped is None:
        return None

    cropped = cropped.convert("RGB")
    cropped_bytes = _image_to_jpeg_bytes(cropped)
    cropped_preview = b64encode(cropped_bytes).decode("ascii")

    st.markdown('<div class="crop-preview-frame">Selected crop — sent to model</div>', unsafe_allow_html=True)
    st.image(cropped, caption=f"Cropped image size: {cropped.width} × {cropped.height} px", width="stretch")
    st.markdown(
        f"""
        <div class="quality-card">
          <div class="quality-thumb"><img src="data:image/jpeg;base64,{cropped_preview}" alt="Crop preview"></div>
          <div>
            <strong>Cropped photo ready</strong>
            <p>Only this cropped area will be analyzed by the model.</p>
          </div>
          <div class="quality-check">✓</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return CroppedImage(
        data=cropped_bytes,
        filename=f"cropped-{selected.filename.rsplit('.', 1)[0]}.jpg",
        content_type="image/jpeg",
        size=cropped.size,
    )


# ─────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────

def render_performance() -> None:
    render_brand()
    st.markdown(
        """
        <div class="page-heading-card">
          <h1>Model Performance</h1>
          <p>Evaluation dashboard for the trained DermaScan AI model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metrics = _read_performance_metrics()
    source_note = metrics.get("source_note")
    if source_note:
        st.markdown(
            f'<div class="status-note"><strong>Metrics source:</strong> {escape(str(source_note))}</div>',
            unsafe_allow_html=True,
        )
    metric_columns = st.columns(3)
    metric_specs = [
        ("Accuracy", _metric_value(metrics, "accuracy")),
        ("Validation Accuracy", _metric_value(metrics, "validation_accuracy")),
        ("Weighted F1", _metric_value(metrics, "weighted_f1")),
    ]
    for column, (label, value) in zip(metric_columns, metric_specs):
        column.metric(label, _format_metric(value))

    st.markdown(
        """
        <div class="welcome-card">
          <h3>Performance note</h3>
          <p>Only the three approved model measures are shown here: overall accuracy,
             validation accuracy, and weighted F1-score. This keeps the project report
             focused on the reliability indicators requested for the final system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_disclaimer()


# ─────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────

def render_about() -> None:
    render_brand()
    model_info = get_model_info() or {}
    classes = ", ".join(name.title() for name in _supported_classes(model_info))
    model_name = model_info.get("model_name", "Not available")
    st.markdown(
        f"""
        <div class="page-heading-card">
          <h1>About DermaScan AI</h1>
          <p>Final-year Computer Science research prototype for educational skin screening support.</p>
        </div>
        <div class="about-grid">
          <div class="info-card"><strong style="background:none;color:#295289;font-size:0.85rem;font-weight:600;display:block;margin-bottom:0.4rem">Project name</strong><p>DermaScan AI</p></div>
          <div class="info-card"><strong style="background:none;color:#295289;font-size:0.85rem;font-weight:600;display:block;margin-bottom:0.4rem">Project type</strong><p>Final-year Computer Science research prototype</p></div>
          <div class="info-card"><strong style="background:none;color:#295289;font-size:0.85rem;font-weight:600;display:block;margin-bottom:0.4rem">Dataset</strong><p>SCIN dataset filtered for the trained model classes: {escape(classes)}</p></div>
          <div class="info-card"><strong style="background:none;color:#295289;font-size:0.85rem;font-weight:600;display:block;margin-bottom:0.4rem">Model</strong><p>{escape(str(model_name))}</p></div>
          <div class="info-card"><strong style="background:none;color:#295289;font-size:0.85rem;font-weight:600;display:block;margin-bottom:0.4rem">Frameworks</strong><p>Python, PyTorch, FastAPI, Streamlit</p></div>
          <div class="info-card"><strong style="background:none;color:#295289;font-size:0.85rem;font-weight:600;display:block;margin-bottom:0.4rem">Deployment target</strong><p>Streamlit Community Cloud</p></div>
        </div>
        <div class="warning-card">
          <strong>⚠ Disclaimer:</strong> This application is not a replacement for dermatologists
          or qualified healthcare professionals.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_disclaimer()


# ─────────────────────────────────────────────────────────────
# PAGE: SCAN
# ─────────────────────────────────────────────────────────────

def render_scan() -> None:
    render_brand()
    if st.button("← Back to landing page", key="back-to-landing"):
        navigate("home")

    model_info = render_model_notice()
    class_names = _supported_classes(model_info)

    st.markdown(
        f"""
        <div class="scan-intro-panel">
          <h2>Scan a Skin Image</h2>
          <p>Upload or capture a clear skin image, crop the area of concern, then receive
             an educational AI prediction for {", ".join(name.title() for name in class_names)}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not health_check():
        st.error("The backend is not reachable at http://127.0.0.1:8000. Start it before analyzing an image.")

    left_column, right_column = st.columns([1.1, 0.9], gap="large")
    with left_column:
        st.markdown('<h2 class="section-title">Upload Image</h2>', unsafe_allow_html=True)
        upload_tab, camera_tab = st.tabs(["📁 Upload an image", "📷 Use camera"])
        with upload_tab:
            uploaded = st.file_uploader(
                "Accepted formats: JPG, JPEG, PNG, WEBP",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=False,
            )
        with camera_tab:
            captured = st.camera_input("Take a photo")

        selected = _selected_image(uploaded, captured)
        cropped_image = None
        if selected:
            cropped_image = _render_cropper(selected)
        else:
            st.markdown(
                """
                <div class="upload-card">
                  <h3>📋 Image quality tips</h3>
                  <ul>
                    <li>Use good, even lighting — avoid harsh shadows.</li>
                    <li>Keep the skin area clear and in focus.</li>
                    <li>Avoid blurry or low-resolution images.</li>
                    <li>Do not include faces or personal identifiers.</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Optional context"):
            col_a, col_b = st.columns(2)
            with col_a:
                age_range = st.selectbox("Age range", ["", "Under 18", "18-29", "30-44", "45-59", "60+"])
                body_location = st.text_input("Body location", placeholder="For example: forearm")
            with col_b:
                sex = st.selectbox("Sex", ["", "Female", "Male", "Prefer not to say"])
                symptom_duration = st.text_input("Symptom duration", placeholder="For example: 2 weeks")

        render_model_metrics(model_info)
        consent = st.checkbox("I understand this tool is not a diagnosis.")
        analyze_disabled = cropped_image is None or not consent
        if st.button("Analyze Image", type="primary", use_container_width=True, disabled=analyze_disabled):
            metadata = {
                "age_range": age_range,
                "sex": sex,
                "body_location": body_location,
                "symptom_duration": symptom_duration,
            }
            try:
                with st.status("Checking image quality and preparing the screening result...", expanded=True) as status:
                    result = predict_image(
                        cropped_image.data,
                        cropped_image.filename,
                        cropped_image.content_type,
                        metadata,
                    )
                    status.update(label="Screening result ready", state="complete", expanded=False)
                st.session_state.result = result
                st.session_state.scan_image = cropped_image.data
                st.session_state.report_bytes = None
                st.session_state.page = "result"
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))

    with right_column:
        st.markdown(
            """
            <div class="result-card">
              <h3>🔍 AI Prediction</h3>
              <p>Your educational support result will appear here after you crop the image and click <strong>Analyze Image</strong>.</p>
              <p><strong>Output includes:</strong> possible condition, confidence score, probability chart, explanation, and safety guidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="warning-card">⚠ {URGENT_WARNING}</div>', unsafe_allow_html=True)
    render_disclaimer()


# ─────────────────────────────────────────────────────────────
# PAGE: RESULT
# ─────────────────────────────────────────────────────────────

def render_result() -> None:
    result = st.session_state.result
    if not result:
        navigate("scan")
        return

    render_brand()
    st.markdown(
        """
        <div class="result-heading">
          <h1>Your Screening Result</h1>
          <p>Review the possible class and confidence carefully. Visual similarity is not a medical diagnosis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    model_info = render_model_notice()
    top = result.get("top_prediction")
    if top and str(top.get("class_name", "")).lower() not in SUPPORTED_CLASSES:
        top = None
    top_k = [
        item
        for item in result.get("top_k", [])
        if str(item.get("class_name", "")).lower() in SUPPORTED_CLASSES
    ]
    if top:
        name = escape(top["class_name"].title())
        confidence_value = top["confidence"] * 100
        confidence = f"{confidence_value:.1f}% confidence"
        initial = name[0]
    else:
        name = "Uncertain result"
        confidence_value = 0.0
        confidence = "No single category met the confidence threshold"
        initial = "?"

    st.image(st.session_state.scan_image, caption="Submitted image", width="stretch")
    st.markdown(
        f"""
        <div class="result-feature-card">
          <div class="result-letter">{initial}</div>
          <div>
            <h2>Possible condition: {name}</h2>
            <p>Confidence score: <strong>{confidence}</strong>. This is an educational support result, not a final diagnosis.</p>
          </div>
          <div class="confidence-ring" style="--score:{confidence_value:.1f}%">{confidence_value:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if top:
        st.metric("AI Prediction Confidence", f"{confidence_value:.1f}%")
    render_probability_chart(top_k)
    render_model_metrics(model_info, "Model Performance After Analysis")

    rows = "".join(
        f'<div class="prediction-row"><span>{escape(item["class_name"].title())}</span><span>{item["confidence"] * 100:.1f}%</span></div>'
        for item in top_k
    )
    st.markdown(
        f'<div class="prediction-panel"><h3>Possible categories</h3>{rows}</div>',
        unsafe_allow_html=True,
    )
    if top:
        class_key = top["class_name"].lower()
        st.markdown(
            f"""
            <div class="welcome-card">
              <h3>Brief explanation</h3>
              <p>{escape(DISEASE_DESCRIPTIONS.get(class_key, "This possible condition should be reviewed by a qualified health professional."))}</p>
            </div>
            <div class="warning-card">⚠ {URGENT_WARNING}</div>
            """,
            unsafe_allow_html=True,
        )

    explanation = result["explanation"]
    if top is None:
        explanation = {
            "summary": "The result does not match the active three-class model. Please scan again with the current Acne, Eczema, and Psoriasis model.",
            "next_steps": "Clear the old result and upload a fresh human skin image for analysis.",
            "warning": URGENT_WARNING,
        }
    st.markdown(
        f"""
        <div class="consultant">
          <h3>AI consultant explanation</h3>
          <p>{explanation["summary"]}</p>
          <p><strong>What to do next:</strong> {explanation["next_steps"]}</p>
          <p><strong>Urgent warning:</strong> {explanation["warning"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.report_bytes is None:
        try:
            st.session_state.report_bytes = download_report(result["scan_id"])
        except ApiError as exc:
            st.warning(str(exc))

    action_a, action_b = st.columns(2)
    with action_a:
        if st.session_state.report_bytes:
            st.download_button(
                "⬇ Download PDF Report",
                data=st.session_state.report_bytes,
                file_name=f'dermascan-{result["scan_id"]}.pdf',
                mime="application/pdf",
                use_container_width=True,
            )
    with action_b:
        if st.button("↩ Scan Another Image", use_container_width=True):
            st.session_state.result = None
            st.session_state.scan_image = None
            st.session_state.report_bytes = None
            navigate("scan")
    render_disclaimer()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="DermaScan AI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_styles()
    initialize_state()
    render_sidebar_navigation()
    if st.session_state.page == "scan":
        render_scan()
    elif st.session_state.page == "performance":
        render_performance()
    elif st.session_state.page == "about":
        render_about()
    elif st.session_state.page == "result":
        render_result()
    else:
        render_home()


if __name__ == "__main__":
    main()
