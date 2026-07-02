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

DISCLAIMER = (
    "The scan result is not a diagnosis. For accurate diagnosis and treatment "
    "recommendations, consult a qualified healthcare professional or dermatologist."
)

DISEASE_DESCRIPTIONS = {
    "acne": "Acne is a common skin condition that occurs when hair follicles become clogged with oil and dead skin cells.",
    "eczema": "Eczema, also known as atopic dermatitis, is an inflammatory skin condition that may cause itching, dryness, redness, and irritation.",
    "scabies": "Scabies is a contagious skin condition caused by tiny mites and often causes intense itching and rash.",
    "psoriasis": "Psoriasis is an inflammatory skin condition that can cause raised, scaly, or irritated patches.",
    "tinea": "Tinea is a superficial fungal skin condition that can cause ring-shaped, itchy, or scaly patches.",
    "other": "This category means the image may not clearly match one of the supported classes.",
}

URGENT_WARNING = (
    "Seek urgent medical care if the rash is rapidly spreading, painful, bleeding, "
    "infected, or associated with fever."
)


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
        <div class="brand-row">
          <div class="brand-lockup">
            <div class="brand-mark">D</div>
            <div>
              <div class="brand-name">DermaScan AI</div>
              <div class="brand-subtitle">Skin screening support</div>
            </div>
          </div>
          <div class="nav-note">Education and decision support only</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar() -> None:
    st.markdown('<div class="top-strip"></div>', unsafe_allow_html=True)


def render_disclaimer() -> None:
    st.markdown(f'<div class="disclaimer"><strong>Important:</strong> {DISCLAIMER}</div>', unsafe_allow_html=True)


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


def render_model_metrics(info: dict[str, Any] | None) -> None:
    if not info:
        return
    metrics = info.get("metrics") or {}
    metric_items = [
        ("Accuracy", metrics.get("accuracy")),
        ("Macro F1", metrics.get("macro_f1")),
        ("Macro Recall", metrics.get("macro_recall")),
        ("Val Accuracy", metrics.get("validation_accuracy")),
        ("Top-3 Accuracy", metrics.get("top_3_accuracy")),
    ]
    metric_items = [(label, value) for label, value in metric_items if value is not None]
    if not metric_items:
        st.markdown(
            """
            <div class="model-metrics">
              <div class="metric-card"><span>Model Metrics</span><strong>N/A</strong><p>No saved evaluation metrics found for this checkpoint.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    cards = "".join(
        f'<div class="metric-card"><span>{escape(label)}</span><strong>{_format_metric(value)}</strong></div>'
        for label, value in metric_items[:4]
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
    return None


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
        color_discrete_sequence=["#347fe2", "#44c3cc", "#0f3f83", "#7dd3fc"],
    )
    figure.update_layout(
        title="Model probabilities",
        yaxis_title="Probability (%)",
        xaxis_title="Possible condition",
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_yaxes(range=[0, 100])
    st.plotly_chart(figure, use_container_width=True)


def render_home() -> None:
    render_brand()
    model_info = get_model_info()
    class_names = (model_info or {}).get("classes") or ["acne", "eczema", "scabies"]
    st.markdown(
        f"""
        <section class="hero-card">
          <div>
            <span class="hero-kicker">Academic medical AI prototype</span>
            <h1>DermaScan AI</h1>
            <h2>AI-Based Skin Disease Screening for {", ".join(name.title() for name in class_names)}</h2>
            <p>DermaScan AI is an academic prototype that uses deep learning to analyze skin images and provide educational screening support.</p>
          </div>
          <div class="hero-side-card">
            <strong>Decision-support workflow</strong>
            <p>Upload a clear image, crop the area of concern, and review AI prediction probabilities with caution.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start Skin Scan", type="primary", use_container_width=False):
        navigate("scan")

    cards = "".join(
        f"""
        <div class="disease-card">
          <h3>{escape(class_name.title())}</h3>
          <p>{escape(DISEASE_DESCRIPTIONS.get(class_name, "Supported model class for educational screening."))}</p>
        </div>
        """
        for class_name in class_names
    )
    st.markdown(
        f"""
        <h2 class="section-title">Supported Conditions</h2>
        <div class="disease-grid">{cards}</div>
        <h2 class="section-title">How It Works</h2>
        <div class="workflow-grid">
          <div class="info-card"><strong>Step 1</strong><h3>Upload a clear skin image</h3><p>Use JPG, JPEG, PNG, WEBP, or camera capture.</p></div>
          <div class="info-card"><strong>Step 2</strong><h3>The AI model analyzes the image</h3><p>The backend checks image quality and runs the trained model.</p></div>
          <div class="info-card"><strong>Step 3</strong><h3>View prediction and guidance</h3><p>Review possible condition, confidence score, probabilities, and safety guidance.</p></div>
        </div>
        <div class="warning-card">This system is for educational support only and is not a medical diagnosis. Please consult a qualified health professional for diagnosis and treatment.</div>
        """,
        unsafe_allow_html=True,
    )


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
          <h2>Let's crop the photo!</h2>
          <p>Drag and resize the crop box on the image. The model will analyze only the selected area.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="crop-preview-frame">Crop your photo</div>', unsafe_allow_html=True)
    cropped = st_cropper(
        image,
        realtime_update=True,
        box_color="#347fe2",
        aspect_ratio=None,
        return_type="image",
        key=f"cropper-{selected.filename}-{len(selected.data)}",
    )
    if cropped is None:
        return None

    cropped = cropped.convert("RGB")
    cropped_bytes = _image_to_jpeg_bytes(cropped)
    cropped_preview = b64encode(cropped_bytes).decode("ascii")

    st.markdown('<div class="crop-preview-frame">Selected crop sent to model</div>', unsafe_allow_html=True)
    st.image(cropped, caption=f"Cropped image size: {cropped.width} x {cropped.height}px", width="stretch")
    st.markdown(
        f"""
        <div class="quality-card">
          <div class="quality-thumb"><img src="data:image/jpeg;base64,{cropped_preview}" alt=""></div>
          <div>
            <strong>The cropped photo is ready</strong>
            <p>Only this cropped area will be analyzed by the model.</p>
          </div>
          <div class="quality-check">OK</div>
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
    if not metrics:
        st.warning("Model evaluation results are not available yet. Run python src/evaluate_model.py first.")
        render_disclaimer()
        return

    metric_columns = st.columns(6)
    metric_specs = [
        ("Accuracy", _metric_value(metrics, "accuracy")),
        ("Precision", _metric_value(metrics, "macro_precision", "precision")),
        ("Recall", _metric_value(metrics, "macro_recall", "recall")),
        ("F1-score", _metric_value(metrics, "macro_f1", "f1_score")),
        ("Macro F1-score", _metric_value(metrics, "macro_f1")),
        ("Weighted F1-score", _metric_value(metrics, "weighted_f1")),
    ]
    for column, (label, value) in zip(metric_columns, metric_specs):
        column.metric(label, _format_metric(value))

    confusion_path = _performance_file("confusion_matrix.png")
    normalized_path = _performance_file("confusion_matrix_normalized.png")
    image_a, image_b = st.columns(2)
    with image_a:
        st.markdown("### Confusion Matrix")
        if confusion_path:
            st.image(str(confusion_path), width="stretch")
        else:
            st.info("confusion_matrix.png is missing.")
    with image_b:
        st.markdown("### Normalized Confusion Matrix")
        if normalized_path:
            st.image(str(normalized_path), width="stretch")
        else:
            st.info("confusion_matrix_normalized.png is missing.")

    st.markdown(
        """
        <div class="info-card">
          <h3>What the confusion matrix means</h3>
          <p>The confusion matrix shows how many images were correctly and incorrectly classified for each disease class. It helps identify which diseases the model confuses most often.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    report_path = _performance_file("classification_report.csv")
    st.markdown("### Classification Report")
    if report_path:
        st.dataframe(pd.read_csv(report_path), use_container_width=True)
    else:
        st.info("classification_report.csv is missing.")
    render_disclaimer()


def render_about() -> None:
    render_brand()
    model_info = get_model_info() or {}
    classes = ", ".join(name.title() for name in model_info.get("classes", [])) or "Not available"
    model_name = model_info.get("model_name", "Not available")
    st.markdown(
        f"""
        <div class="page-heading-card">
          <h1>About DermaScan AI</h1>
          <p>Final-year Computer Science research prototype for educational skin screening support.</p>
        </div>
        <div class="about-grid">
          <div class="info-card"><strong>Project name</strong><p>DermaScan AI</p></div>
          <div class="info-card"><strong>Project type</strong><p>Final-year Computer Science research prototype</p></div>
          <div class="info-card"><strong>Dataset</strong><p>SCIN dataset filtered for the trained model classes: {escape(classes)}</p></div>
          <div class="info-card"><strong>Model</strong><p>{escape(str(model_name))}</p></div>
          <div class="info-card"><strong>Frameworks</strong><p>Python, PyTorch, FastAPI, Streamlit</p></div>
          <div class="info-card"><strong>Deployment target</strong><p>Streamlit Community Cloud</p></div>
        </div>
        <div class="warning-card">This application is not a replacement for dermatologists or qualified healthcare professionals.</div>
        """,
        unsafe_allow_html=True,
    )
    render_disclaimer()


def render_scan() -> None:
    render_brand()
    model_info = render_model_notice()
    class_names = (model_info or {}).get("classes") or ["acne", "eczema", "scabies"]
    st.markdown(
        f"""
        <div class="page-heading-card">
          <h1>Scan Image</h1>
          <p>Upload or capture a clear skin image, crop the area of concern, then review an educational AI prediction for {", ".join(name.title() for name in class_names)}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not health_check():
        st.error("The backend is not reachable at http://127.0.0.1:8000. Start it before analyzing an image.")

    left_column, right_column = st.columns([1.1, 0.9], gap="large")
    with left_column:
        st.markdown('<h2 class="section-title">Upload Image</h2>', unsafe_allow_html=True)
        upload_tab, camera_tab = st.tabs(["Upload an image", "Use camera"])
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
            render_model_metrics(model_info)
        else:
            st.markdown(
                """
                <div class="upload-card">
                  <h3>Image quality tips</h3>
                  <ul>
                    <li>Use good lighting.</li>
                    <li>Keep the skin area clear and focused.</li>
                    <li>Avoid blurry images.</li>
                    <li>Avoid images with faces or personal identifiers.</li>
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
              <h3>AI Prediction</h3>
              <p>Your educational support result will appear after you crop the image and click Analyze Image.</p>
              <p><strong>Output includes:</strong> possible condition, confidence score, probability chart, explanation, and safety warning.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="warning-card">{URGENT_WARNING}</div>', unsafe_allow_html=True)
    render_disclaimer()


def render_result() -> None:
    result = st.session_state.result
    if not result:
        navigate("scan")
        return

    render_brand()
    st.markdown(
        """
        <div class="result-heading">
          <h1>Your screening result</h1>
          <p>Review the possible class and confidence carefully. Visual similarity is not a medical diagnosis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    model_info = render_model_notice()
    render_model_metrics(model_info)
    top = result.get("top_prediction")
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
            <p>Confidence score: {confidence}. This is an educational support result, not a final diagnosis.</p>
          </div>
          <div class="confidence-ring" style="--score:{confidence_value:.1f}%">{confidence_value:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if top:
        st.metric("AI Prediction Confidence", f"{confidence_value:.1f}%")
    render_probability_chart(result.get("top_k", []))

    rows = "".join(
        f'<div class="prediction-row"><span>{escape(item["class_name"].title())}</span><span>{item["confidence"] * 100:.1f}%</span></div>'
        for item in result["top_k"]
    )
    st.markdown(f'<div class="prediction-panel"><h3>Possible categories</h3>{rows}</div>', unsafe_allow_html=True)
    if top:
        class_key = top["class_name"].lower()
        st.markdown(
            f"""
            <div class="info-card">
              <h3>Brief explanation</h3>
              <p>{escape(DISEASE_DESCRIPTIONS.get(class_key, "This possible condition should be reviewed by a qualified health professional."))}</p>
            </div>
            <div class="warning-card">{URGENT_WARNING}</div>
            """,
            unsafe_allow_html=True,
        )

    explanation = result["explanation"]
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
                "DOWNLOAD PDF REPORT",
                data=st.session_state.report_bytes,
                file_name=f'dermascan-{result["scan_id"]}.pdf',
                mime="application/pdf",
                use_container_width=True,
            )
    with action_b:
        if st.button("SCAN ANOTHER IMAGE", use_container_width=True):
            st.session_state.result = None
            st.session_state.scan_image = None
            st.session_state.report_bytes = None
            navigate("scan")
    render_disclaimer()


def main() -> None:
    st.set_page_config(
        page_title="DermaScan AI",
        page_icon="D",
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
