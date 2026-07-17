from base64 import b64encode
from dataclasses import dataclass
from html import escape
from io import BytesIO
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
    import plotly.express as px
except ModuleNotFoundError:
    px = None

from frontend.api import ApiError, download_report, get_model_info, health_check, predict_image
from frontend.styles import apply_styles


ASSET_DIR = PROJECT_ROOT / "frontend" / "assets"
SUPPORTED_CLASSES = ["acne", "scabies", "psoriasis"]

DISCLAIMER = (
    "The scan result is not a diagnosis. For accurate diagnosis and treatment "
    "recommendations, consult a qualified healthcare professional or dermatologist."
)

DISEASE_DESCRIPTIONS = {
    "acne": "Acne may show blocked or inflamed pores, blackheads, whiteheads, pimples, oily skin, or scarring. The AI compares visible bump patterns and texture, but a clinician must confirm the cause.",
    "scabies": "Scabies may show small itchy bumps or thin burrow-like lines, often around fingers, wrists, the waist, or skin folds. A clinician must confirm it because several rashes can look similar.",
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

@dataclass(slots=True)
class SelectedImage:
    data: bytes
    filename: str
    content_type: str


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
        "about": "About",
    }
    return labels.get(page, "Home")


def render_sidebar_navigation() -> None:
    if st.session_state.page == "result":
        return
    pages = ["Home", "Scan Image", "About"]
    selected = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(_page_label(st.session_state.page)),
    )
    page_map = {
        "Home": "home",
        "Scan Image": "scan",
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


def render_disclaimer() -> None:
    st.markdown(
        f'<div class="disclaimer"><strong>Important:</strong> {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


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

    # ── Split welcome section (Aidoc "See what matters" style) ─
    st.markdown(
        '<div class="welcome-split landing-card-panel"><div class="welcome-split-text"><div class="headline-wrap"><span class="orange-bar"></span><h2>Welcome to DermaScan AI.<br>Clear support for skin screening.</h2></div><p>Start here: upload or capture a skin image. The backend prepares it automatically before the AI model reviews it for educational support.</p><div class="trusted-note">Educational and decision support tool only, not a medical diagnosis</div></div><div class="welcome-split-image"></div></div><div id="scan-start"></div>',
        unsafe_allow_html=True,
    )
    if st.button("Get started →", type="primary"):
        navigate("scan")

    # ── Hero — full-bleed Aidoc style ─────────────────────────
    st.markdown(
        f'<div class="hero-fullbleed"><h1>AI Solutions Deliver<br>Smarter Skin Screening</h1><p>The pressure on medical professionals to provide quality and effective care is enormous. DermaScan AI uses deep learning to help support dermatological triage for {escape(classes_display)} and turns images into actionable educational insights.</p></div>',
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
            <p>The backend checks image quality, prepares the image automatically, and runs the trained deep learning model.</p>
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


def _render_selected_image(selected: SelectedImage) -> SelectedImage | None:
    image = _load_image(selected.data)
    if image is None:
        st.error("This image could not be opened. Please upload a clear JPG, PNG, or WEBP file.")
        return None
    st.markdown(
        """
        <div class="quality-card">
          <div>
            <strong>Image ready</strong>
            <p>The backend will automatically prepare and center-crop this image before analysis.</p>
          </div>
          <div class="quality-check">Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image(image, caption="Submitted image preview", width="stretch")
    return selected


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

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
          <p>Upload or capture a clear skin image, then receive
             an educational AI prediction for {", ".join(name.title() for name in class_names)}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not health_check():
        st.error("The backend is not reachable at http://127.0.0.1:8000. Start it before analyzing an image.")

    left_column, right_column = st.columns([1.1, 0.9], gap="large")
    with left_column:
        st.markdown(
            '<h2 class="section-title force-white-text" style="color:#ffffff !important;">Upload Image</h2>',
            unsafe_allow_html=True,
        )
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
        analysis_image = None
        if selected:
            analysis_image = _render_selected_image(selected)
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

        consent = st.checkbox("I understand this tool is not a diagnosis.")
        analyze_disabled = analysis_image is None or not consent
        if st.button("Analyze Image", type="primary", use_container_width=True, disabled=analyze_disabled):
            metadata: dict[str, str] = {}
            try:
                with st.status("Checking image quality and preparing the screening result...", expanded=True) as status:
                    result = predict_image(
                        analysis_image.data,
                        analysis_image.filename,
                        analysis_image.content_type,
                        metadata,
                    )
                    status.update(label="Screening result ready", state="complete", expanded=False)
                st.session_state.result = result
                st.session_state.scan_image = analysis_image.data
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
              <p>Your educational support result will appear here after you upload or capture an image and click <strong>Analyze Image</strong>.</p>
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
            "summary": "The result does not match the active three-class model. Please scan again with the current Acne, Scabies, and Psoriasis model.",
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
    elif st.session_state.page == "about":
        render_about()
    elif st.session_state.page == "result":
        render_result()
    else:
        render_home()


if __name__ == "__main__":
    main()
