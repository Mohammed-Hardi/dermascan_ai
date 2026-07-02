from base64 import b64encode
from pathlib import Path

import streamlit as st


ASSET_DIR = Path(__file__).resolve().parent / "assets"
HERO_BACKGROUND = ASSET_DIR / "dermatology-background.jpg"
SCAN_BACKGROUND = ASSET_DIR / "scan-section-background.webp"


def _image_layer(path: Path, mime_type: str) -> str:
    if not path.exists():
        return "none"
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f'url("data:{mime_type};base64,{encoded}")'


APP_CSS = """
<style>
:root {
  --navy: #082158;
  --navy-2: #0f3f83;
  --blue: #347fe2;
  --blue-dark: #2368c2;
  --cyan: #44c3cc;
  --red: #ff3030;
  --green: #04bf3a;
  --text: #071b4c;
  --muted: #4d5577;
  --line: #ccd8ef;
  --soft: #f4f8ff;
  --white: #ffffff;
}

.stApp {
  background: var(--white);
  color: var(--text);
  font-family: "Segoe UI Variable", "Segoe UI", Arial, sans-serif;
}

.block-container {
  max-width: 980px;
  padding-top: 1.2rem;
  padding-bottom: 4rem;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }

h1, h2, h3, p, label, .stMarkdown {
  color: var(--text);
}

h1, h2, h3 {
  letter-spacing: -0.035em;
}

.brand-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 0 1.1rem;
}

.brand-lockup {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.brand-mark {
  width: 2.6rem;
  height: 2.6rem;
  border-radius: 999px;
  display: grid;
  place-items: center;
  color: white;
  font-weight: 850;
  background: linear-gradient(145deg, var(--blue), var(--navy-2));
  box-shadow: 0 10px 24px rgba(52, 127, 226, 0.25);
}

.brand-name {
  font-weight: 850;
  font-size: 1.08rem;
}

.brand-subtitle, .nav-note {
  color: var(--muted);
  font-size: 0.84rem;
}

.hero-card {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(16rem, 0.65fr);
  gap: 1.4rem;
  align-items: stretch;
  margin: 1rem 0 1.2rem;
  padding: 2rem;
  border: 1px solid var(--line);
  border-radius: 22px;
  background:
    radial-gradient(circle at 92% 12%, rgba(68, 195, 204, 0.22), transparent 18rem),
    linear-gradient(135deg, #ffffff, #f4f8ff);
  box-shadow: 0 22px 50px rgba(7, 33, 88, 0.10);
}

.hero-kicker {
  display: inline-flex;
  margin-bottom: 0.7rem;
  color: var(--blue);
  font-size: 0.78rem;
  font-weight: 850;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero-card h1 {
  margin: 0;
  color: var(--navy);
  font-size: clamp(2.4rem, 5vw, 4rem);
}

.hero-card h2 {
  margin: 0.45rem 0 0.7rem;
  color: var(--navy-2);
  font-size: clamp(1.05rem, 2vw, 1.45rem);
  letter-spacing: -0.02em;
}

.hero-card p,
.page-heading-card p {
  color: var(--muted);
  line-height: 1.65;
}

.hero-side-card {
  padding: 1.35rem;
  border-radius: 18px;
  background: var(--navy);
  color: white;
}

.hero-side-card strong,
.hero-side-card p {
  color: white;
}

.page-heading-card {
  margin: 1rem 0 1.2rem;
  padding: 1.5rem;
  border: 1px solid var(--line);
  border-radius: 20px;
  background: linear-gradient(135deg, #ffffff, #f3f9ff);
  box-shadow: 0 18px 36px rgba(7, 33, 88, 0.07);
}

.page-heading-card h1 {
  margin: 0 0 0.35rem;
  color: var(--navy);
  font-size: clamp(2rem, 4vw, 3rem);
}

.section-title {
  margin: 1.6rem 0 0.85rem;
  color: var(--navy);
  font-size: 1.35rem;
  font-weight: 850;
}

.disease-grid,
.workflow-grid,
.about-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

.about-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.disease-card,
.info-card,
.result-card,
.upload-card {
  padding: 1.1rem 1.2rem;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: white;
  box-shadow: 0 14px 30px rgba(7, 33, 88, 0.07);
}

.disease-card {
  border-top: 4px solid var(--cyan);
}

.disease-card h3,
.info-card h3,
.result-card h3,
.upload-card h3 {
  margin: 0 0 0.45rem;
  color: var(--navy);
}

.disease-card p,
.info-card p,
.result-card p,
.upload-card li {
  color: var(--muted);
  line-height: 1.55;
}

.info-card strong {
  display: block;
  color: var(--blue);
  margin-bottom: 0.35rem;
}

.warning-card {
  max-width: 100%;
  margin: 1rem 0;
  padding: 1rem 1.1rem;
  border: 1px solid #ffd0b0;
  border-radius: 16px;
  background: #fff2e7;
  color: #6b3414;
  line-height: 1.55;
}

.skin-hero {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(22rem, 0.95fr) minmax(24rem, 1.05fr);
  gap: 2rem;
  min-height: 26rem;
  padding: 1.7rem 2rem 1.5rem;
  border-radius: 0 0 22px 22px;
  background:
    radial-gradient(circle at 0% 22%, rgba(51, 127, 226, 0.42), transparent 18rem),
    linear-gradient(120deg, #074da2 0%, #073c87 46%, #06407e 100%);
  color: white;
}

.skin-hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 25% 20%, rgba(255,255,255,0.10), transparent 11rem),
    linear-gradient(58deg, transparent 0 34%, rgba(255,255,255,0.06) 34% 51%, transparent 51% 100%);
  pointer-events: none;
}

.skin-hero-content {
  position: relative;
  z-index: 1;
}

.skin-hero h1 {
  margin: 0 0 1.6rem;
  color: white;
  font-size: clamp(2.2rem, 5vw, 3.4rem);
  line-height: 1;
}

.hero-steps {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1.05rem 1.25rem;
}

.hero-step {
  display: grid;
  grid-template-columns: 4.1rem 1fr;
  gap: 0.8rem;
  align-items: center;
}

.step-icon {
  width: 4.1rem;
  height: 4.1rem;
  border-radius: 999px;
  display: grid;
  place-items: center;
  color: white;
  font-weight: 850;
  background: linear-gradient(145deg, #1da8f5, #175ad6);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.35), 0 16px 28px rgba(0,0,0,0.18);
}

.hero-step span {
  display: inline-flex;
  padding: 0.16rem 0.58rem;
  border: 1px solid rgba(255,255,255,0.82);
  border-radius: 999px;
  color: white;
  font-size: 0.66rem;
  font-weight: 800;
}

.hero-step p {
  margin: 0.48rem 0 0;
  color: white;
  font-size: 1rem;
  line-height: 1.28;
}

.hero-photo-card {
  position: relative;
  z-index: 1;
  min-height: 22rem;
  align-self: stretch;
  border-radius: 0 0 12rem 12rem;
  background:
    linear-gradient(90deg, rgba(7, 33, 88, 0.14), rgba(7, 33, 88, 0.04)),
    __HERO_BACKGROUND__;
  background-size: cover;
  background-position: center;
}

.scan-target {
  position: absolute;
  left: 12%;
  top: 30%;
  width: 7rem;
  height: 7rem;
  border: 3px solid white;
  border-radius: 10px;
  box-shadow: 0 0 0 999px rgba(7, 33, 88, 0.04);
}

.hero-photo-note {
  position: absolute;
  left: 35%;
  bottom: 32%;
  max-width: 15rem;
  color: white;
  font-weight: 600;
  line-height: 1.35;
  text-shadow: 0 1px 8px rgba(0,0,0,0.35);
}

.home-trust-card {
  margin: 2rem auto 0;
  max-width: 47rem;
}

.home-trust-card h2 {
  font-size: clamp(2rem, 5vw, 3rem);
  margin: 0 0 0.8rem;
}

.home-trust-card p {
  color: var(--muted);
  line-height: 1.65;
  max-width: 42rem;
}

.trust-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.85rem;
  margin-top: 1.3rem;
}

.trust-grid div {
  padding: 1.1rem;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--soft);
}

.trust-grid strong {
  display: block;
  color: var(--blue);
  font-size: 1.45rem;
}

.trust-grid span {
  color: var(--muted);
  font-size: 0.9rem;
}

.upload-intro {
  max-width: 42rem;
  margin: 0 auto 1.4rem;
}

.upload-intro h1 {
  margin: 0.3rem 0 1.2rem;
  font-size: clamp(2rem, 5vw, 3rem);
}

.upload-photo {
  min-height: 16.5rem;
  border-radius: 16px;
  background:
    linear-gradient(90deg, rgba(255,255,255,0.05), rgba(255,255,255,0.14)),
    __SCAN_BACKGROUND__;
  background-size: cover;
  background-position: center;
  box-shadow: 0 22px 46px rgba(7, 33, 88, 0.12);
}

.trust-list {
  margin: 1.4rem 0;
  color: var(--text);
}

.trust-list strong {
  display: block;
  margin-bottom: 0.55rem;
}

.trust-list ul {
  margin: 0;
  padding-left: 1.25rem;
  color: var(--text);
}

.trust-list li {
  margin: 0.18rem 0;
}

.top-strip {
  height: 22px;
  margin: 1.8rem -10rem 1.6rem;
  background: var(--navy-2);
}

.crop-heading {
  max-width: 42rem;
  margin: 0 auto 1rem;
}

.crop-heading h2 {
  margin: 0;
  font-size: clamp(1.8rem, 4vw, 2.6rem);
}

.crop-heading p {
  color: var(--muted);
}

.crop-preview-frame {
  max-width: 42rem;
  margin: 0 auto 0.45rem;
  color: var(--muted);
  font-weight: 700;
}

.crop-controls-title {
  max-width: 42rem;
  margin: 1rem auto 0.45rem;
  color: var(--text);
  font-weight: 850;
}

[data-testid="stImage"] img {
  border-radius: 4px;
}

[data-testid="stSlider"] {
  max-width: 42rem;
  margin: 0 auto 0.8rem;
}

.quality-card {
  display: grid;
  grid-template-columns: 3.2rem 1fr auto;
  align-items: center;
  gap: 1rem;
  max-width: 42rem;
  margin: 1rem auto 1.3rem;
  padding: 0.9rem 1.2rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: white;
}

.quality-thumb {
  width: 3.2rem;
  height: 3.2rem;
  overflow: hidden;
  border-radius: 3px;
}

.quality-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.quality-card strong {
  color: var(--green);
  font-weight: 700;
}

.quality-card p {
  margin: 0.18rem 0 0;
  color: var(--muted);
  font-size: 0.9rem;
}

.quality-check {
  color: var(--green);
  font-weight: 900;
}

.status-note, .placeholder-note {
  max-width: 42rem;
  margin: 0 auto 1rem;
  border: 1px solid var(--line);
  background: var(--soft);
  border-radius: 14px;
  padding: 0.95rem 1.05rem;
  color: var(--muted);
}

.placeholder-note {
  border-color: #ffd2d2;
  background: #fff6f6;
}

.placeholder-note strong {
  color: #bd2323;
}

.model-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  max-width: 42rem;
  margin: 0.8rem auto 1.2rem;
}

.metric-card {
  padding: 0.95rem;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: white;
  box-shadow: 0 12px 28px rgba(7, 33, 88, 0.06);
}

.metric-card span {
  display: block;
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 800;
}

.metric-card strong {
  display: block;
  margin-top: 0.3rem;
  color: var(--blue);
  font-size: 1.25rem;
}

.metric-card p {
  margin: 0.35rem 0 0;
  color: var(--muted);
  font-size: 0.86rem;
  line-height: 1.35;
}

.result-heading {
  max-width: 42rem;
  margin: 1rem auto 1.3rem;
}

.result-heading h1 {
  margin: 0 0 0.45rem;
  font-size: clamp(2rem, 5vw, 3rem);
}

.result-heading p {
  color: var(--muted);
}

.result-photo-wrap {
  max-width: 42rem;
  margin: 0 auto 1rem;
}

.result-feature-card {
  display: grid;
  grid-template-columns: 3.4rem 1fr auto;
  align-items: center;
  gap: 1rem;
  max-width: 42rem;
  margin: 1.2rem auto;
  padding: 1.25rem;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: white;
  box-shadow: 0 18px 34px rgba(7, 33, 88, 0.08);
}

.result-letter {
  color: var(--text);
  font-size: 3.2rem;
  font-weight: 500;
  line-height: 1;
}

.result-feature-card h2 {
  margin: 0 0 0.22rem;
  color: var(--text);
  font-size: 1.25rem;
}

.result-feature-card p {
  margin: 0;
  color: var(--text);
  line-height: 1.35;
}

.confidence-ring {
  width: 3.1rem;
  height: 3.1rem;
  border-radius: 999px;
  display: grid;
  place-items: center;
  color: var(--blue);
  font-size: 0.78rem;
  font-weight: 850;
  background:
    radial-gradient(circle at center, white 58%, transparent 59%),
    conic-gradient(var(--blue) var(--score), #d9e6fb 0);
}

.prediction-panel {
  max-width: 42rem;
  margin: 1rem auto;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--soft);
  padding: 1.1rem 1.2rem;
}

.prediction-panel h3 {
  margin: 0 0 0.45rem;
}

.prediction-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 1rem;
  padding: 0.8rem 0;
  border-top: 1px solid var(--line);
}

.prediction-row span:last-child {
  color: var(--blue);
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}

.consultant {
  max-width: 42rem;
  margin: 1.2rem auto;
  padding: 1.1rem 1.2rem;
  border-left: 4px solid var(--blue);
  border-radius: 10px;
  background: var(--soft);
}

.consultant h3 {
  margin-top: 0;
}

.consultant p {
  color: var(--muted);
  line-height: 1.6;
}

.disclaimer {
  max-width: 42rem;
  margin: 1.5rem auto 0;
  padding: 0.95rem 1rem;
  color: var(--muted);
  border-radius: 12px;
  background: #fff2e7;
  border: 1px solid #ffdcbf;
  font-size: 0.92rem;
  line-height: 1.55;
}

div.stButton > button, div.stDownloadButton > button {
  min-height: 3.05rem;
  border-radius: 999px;
  border: 1px solid var(--blue);
  background: var(--blue);
  color: white;
  font-weight: 850;
  padding: 0.65rem 1.3rem;
  transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
}

div.stButton > button:hover, div.stDownloadButton > button:hover {
  border-color: var(--blue-dark);
  background: var(--blue-dark);
  color: white;
}

div.stButton > button:active, div.stDownloadButton > button:active {
  transform: translateY(1px);
}

div.stButton > button:disabled {
  opacity: 0.48;
}

div.stButton:first-of-type > button[kind="primary"] {
  border-color: var(--red);
  background: var(--red);
}

[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
  max-width: 42rem;
  margin: 0 auto;
}

[data-testid="stFileUploader"] section {
  border-radius: 18px;
  border-color: var(--line);
  background: var(--soft);
}

[data-baseweb="tab-list"] {
  max-width: 42rem;
  margin: 0 auto 0.5rem;
  gap: 0.5rem;
}

[data-baseweb="tab"] {
  color: var(--muted);
  border-radius: 999px;
}

[data-testid="stExpander"] {
  max-width: 42rem;
  margin: 0.8rem auto;
  border-radius: 14px;
}

[data-testid="stCheckbox"] {
  max-width: 42rem;
  margin: 0.5rem auto;
}

@media (max-width: 800px) {
  .block-container {
    padding: 0.8rem 1rem 3rem;
  }

  .nav-note {
    display: none;
  }

  .skin-hero {
    grid-template-columns: 1fr;
    padding: 1.4rem;
  }

  .hero-steps {
    grid-template-columns: 1fr;
  }

  .hero-photo-card {
    min-height: 18rem;
    border-radius: 22px;
  }

  .hero-photo-note {
    left: 1rem;
    right: 1rem;
    bottom: 1rem;
  }

  .trust-grid {
    grid-template-columns: 1fr;
  }

  .hero-card,
  .disease-grid,
  .workflow-grid,
  .about-grid {
    grid-template-columns: 1fr;
  }

  .model-metrics {
    grid-template-columns: 1fr 1fr;
  }

  .top-strip {
    margin-left: -1rem;
    margin-right: -1rem;
  }

  .result-feature-card {
    grid-template-columns: 3.2rem 1fr;
  }

  .confidence-ring {
    grid-column: 1 / -1;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition: none !important;
  }
}
</style>
"""


def apply_styles() -> None:
    css = APP_CSS.replace("__HERO_BACKGROUND__", _image_layer(HERO_BACKGROUND, "image/jpeg"))
    css = css.replace("__SCAN_BACKGROUND__", _image_layer(SCAN_BACKGROUND, "image/webp"))
    st.markdown(css, unsafe_allow_html=True)
