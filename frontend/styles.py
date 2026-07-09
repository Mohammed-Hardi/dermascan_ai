from base64 import b64encode
from pathlib import Path

import streamlit as st


ASSET_DIR = Path(__file__).resolve().parent / "assets"
SCAN_BACKGROUND     = ASSET_DIR / "scan-page-bg.png"
HERO_DOCTOR_BG      = ASSET_DIR / "hero-dermatologist-bg.png"
DOCTOR_AT_WORK      = ASSET_DIR / "dermatologist-at-work.png"
GLOBAL_BACKGROUND   = ASSET_DIR / "global-dermatology-bg.png"


def _image_layer(path: Path, mime_type: str) -> str:
    if not path.exists():
        return "none"
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f'url("data:{mime_type};base64,{encoded}")'


APP_CSS = """
<style>
/* ============================================================
   GOOGLE FONTS — Inter
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ============================================================
   DESIGN TOKENS — Aidoc-inspired system (info/design.md)
   ============================================================ */
:root {
  /* Primary palette */
  --navy-deep:        #060C40;
  --navy-medium:      #2B3C4C;

  /* Accent */
  --orange-primary:   #FF6324;
  --orange-secondary: #EA9118;

  /* Interactive */
  --btn-blue:         #295289;
  --btn-blue-hover:   #184080;
  --btn-blue-active:  #060C40;
  --btn-disabled:     #C2CDE2;

  /* Neutral scale */
  --slate-light:      #899BB9;
  --slate-medium:     #657A8E;
  --grey:             #707070;

  /* Surfaces */
  --bg-white:         #FFFFFF;
  --bg-faint-blue:    #F5F9FF;
  --bg-light-blue:    #F2F9FE;
  --bg-pale-blue:     #EEF4FF;
  --lavender:         #C2CDE2;

  /* Borders */
  --border-card:      #D2E4F2;
  --border-input:     #C2CDE2;

  /* Status */
  --error-red:        #CF2E2E;
  --warning-bg:       #FFF7ED;
  --warning-border:   #FED7AA;
  --warning-text:     #7C2D12;

  /* Shadows (navy-derived) */
  --shadow-1: rgba(9, 19, 33, 0.08) 0px 1px 2px 0px;
  --shadow-2: rgba(9, 19, 33, 0.12) 0px 2px 4px 0px;
  --shadow-3: rgba(9, 19, 33, 0.16) 0px 4px 8px 0px;
  --shadow-4: rgba(9, 19, 33, 0.20) 0px 8px 16px 0px;
}

/* ============================================================
   GLOBAL RESET & BASE
   ============================================================ */
*, *::before, *::after {
  box-sizing: border-box;
}

.stApp {
  background:
    linear-gradient(115deg, rgba(3, 12, 67, 0.96) 0%, rgba(11, 77, 162, 0.90) 48%, rgba(2, 8, 47, 0.96) 100%),
    __GLOBAL_BACKGROUND__;
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  color: var(--navy-medium);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 16px;
  line-height: 1.5;
}

.block-container {
  position: relative;
  z-index: 1;
  max-width: 1100px;
  padding-top: 0;
  padding-bottom: 5rem;
  padding-left: 1.5rem;
  padding-right: 1.5rem;
}

[data-testid="stHeader"]  { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

h1, h2, h3, h4, h5, h6 {
  color: var(--navy-medium);
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  line-height: 1.15;
  margin: 0;
}

p, label, .stMarkdown {
  color: var(--navy-medium);
  font-family: 'Inter', sans-serif;
}

/* ============================================================
   BRAND / NAVIGATION HEADER
   ============================================================ */
.brand-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 0 1.5rem;
  border-bottom: 1px solid var(--border-card);
  margin-bottom: 0;
}

.brand-lockup {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.brand-mark {
  width: 2.75rem;
  height: 2.75rem;
  border-radius: 999px;
  display: grid;
  place-items: center;
  color: #fff;
  font-size: 1.1rem;
  font-weight: 600;
  background: linear-gradient(135deg, var(--btn-blue), var(--navy-deep));
  box-shadow: 0 6px 20px rgba(41, 82, 137, 0.35);
  flex-shrink: 0;
}

.brand-name {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--navy-deep);
  letter-spacing: -0.02em;
}

.brand-subtitle {
  font-size: 0.78rem;
  color: var(--slate-medium);
  margin-top: 1px;
}

.nav-badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 12px;
  border: 1px solid var(--lavender);
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  color: var(--navy-medium);
  background: var(--bg-faint-blue);
}

/* ============================================================
   HERO SECTION — Full-bleed Aidoc style (dark blue + doctor bg)
   ============================================================ */
.landing-bg {
  display: none;
}

.brand-bar,
.hero-fullbleed,
.welcome-split,
.clinical-gallery,
.stats-strip,
.section-header,
.disease-grid,
.workflow-grid,
.warning-card,
[data-testid="stButton"] {
  position: relative;
  z-index: 1;
}

.brand-bar {
  padding-left: 1rem;
  padding-right: 1rem;
  border-radius: 0 0 16px 16px;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(10px);
}

.hero-fullbleed {
  position: relative;
  isolation: isolate;
  overflow: hidden;
  /* no border-radius — extends edge to edge within Streamlit's container */
  margin: 1.5rem -1.5rem 0;
  padding: 5rem 8% 4.5rem;
  min-height: 28rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  background: var(--navy-deep);
  color: #fff;
  box-shadow: var(--shadow-4);
}

.hero-fullbleed::before {
  content: "";
  position: absolute;
  inset: 0;
  z-index: 1;
  background:
    radial-gradient(circle at 16% 18%, rgba(32, 135, 255, 0.62), transparent 34%),
    linear-gradient(100deg, rgba(3, 12, 67, 0.95) 0%, rgba(5, 27, 105, 0.88) 48%, rgba(6, 12, 64, 0.78) 100%);
}

.hero-bg-img {
  display: none;
}

.hero-fullbleed h1 {
  position: relative;
  z-index: 2;
  font-size: clamp(2rem, 4.5vw, 3.2rem);
  font-weight: 600;
  color: #fff;
  line-height: 1.15;
  max-width: 44rem;
  margin: 0 auto 1.2rem;
}

.hero-fullbleed p {
  position: relative;
  z-index: 2;
  font-size: 1rem;
  color: rgba(255,255,255,0.82);
  line-height: 1.7;
  max-width: 38rem;
  margin: 0 auto 2rem;
}

/* ============================================================
   WELCOME SPLIT SECTION (Aidoc "See what matters" style)
   ============================================================ */
.welcome-split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
  align-items: center;
  margin: 1.5rem -1.5rem 1.25rem;
  padding: 3.5rem 8%;
  background:
    linear-gradient(135deg, rgba(4, 24, 95, 0.92), rgba(11, 77, 162, 0.78)),
    radial-gradient(circle at 90% 12%, rgba(255, 255, 255, 0.16), transparent 28%);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 18px;
  box-shadow: 0 24px 60px rgba(2, 8, 47, 0.28);
  backdrop-filter: blur(8px);
}

.welcome-split-text {
  padding-right: 1rem;
}

.welcome-split-text .orange-bar {
  display: inline-block;
  width: 4px;
  height: 3.2rem;
  background: var(--orange-primary);
  border-radius: 2px;
  margin-right: 1rem;
  vertical-align: middle;
  flex-shrink: 0;
}

.welcome-split-text .headline-wrap {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 1.2rem;
}

.welcome-split-text h2 {
  font-size: clamp(1.6rem, 3vw, 2.4rem);
  font-weight: 600;
  color: #fff;
  line-height: 1.2;
  margin: 0;
}

.welcome-split-text p {
  color: rgba(255, 255, 255, 0.82);
  font-size: 1rem;
  line-height: 1.7;
  margin: 0 0 1.8rem;
  max-width: 30rem;
}

.welcome-split-text .trusted-note {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: rgba(255, 255, 255, 0.82);
  font-size: 0.85rem;
  padding-top: 1.2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.22);
}

.welcome-split-text .trusted-note::before {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border-card);
  display: none;
}

.welcome-split-image {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: var(--shadow-3);
  aspect-ratio: 4/3;
  background:
    __DOCTOR_AT_WORK__;
  background-size: cover;
  background-position: center top;
  min-height: 18rem;
}

/* ============================================================
   LANDING CLINICAL IMAGE ANIMATION
   ============================================================ */
.clinical-gallery {
  display: grid;
  grid-template-columns: 0.82fr 1.18fr;
  gap: 2rem;
  align-items: center;
  margin: 2rem 0 2.6rem;
  padding: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 16px;
  background:
    linear-gradient(135deg, rgba(4, 24, 95, 0.90), rgba(11, 77, 162, 0.72)),
    radial-gradient(circle at 12% 20%, rgba(255, 255, 255, 0.14), transparent 32%);
  box-shadow: 0 22px 50px rgba(2, 8, 47, 0.24);
  backdrop-filter: blur(8px);
}

.clinical-gallery-copy h3 {
  font-size: clamp(1.55rem, 2.4vw, 2.25rem);
  color: #fff;
  letter-spacing: -0.04em;
  margin-bottom: 0.75rem;
}

.clinical-gallery-copy p {
  color: rgba(255, 255, 255, 0.82);
  margin: 0;
  max-width: 28rem;
  line-height: 1.7;
}

.clinical-gallery-grid {
  display: block;
  min-height: 16rem;
  position: relative;
  overflow: hidden;
  border-radius: 16px;
  border: 1px solid rgba(210, 228, 242, 0.9);
  box-shadow: 0 18px 44px rgba(2, 8, 47, 0.24);
}

.clinical-shot {
  position: absolute;
  inset: 0;
  overflow: hidden;
  opacity: 0;
  transform: translateX(1.8rem) scale(1.02);
  animation: clinical-slide 12s ease-in-out infinite;
}

.clinical-shot::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, transparent 50%, rgba(6, 12, 64, 0.34));
}

.clinical-shot:nth-child(2),
.clinical-shot:nth-child(4) {
  animation-delay: 3s;
}

.clinical-shot:nth-child(3) {
  animation-delay: 6s;
}

.clinical-shot:nth-child(4) {
  animation-delay: 9s;
}

.clinical-shot img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

@keyframes clinical-slide {
  0% { opacity: 0; transform: translateX(1.8rem) scale(1.02); }
  8%, 24% { opacity: 1; transform: translateX(0) scale(1); }
  32%, 100% { opacity: 0; transform: translateX(-1.8rem) scale(1.02); }
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(1.3); }
}

/* ============================================================
   TRUST / STATS STRIP
   ============================================================ */
.stats-strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: rgba(255, 255, 255, 0.18);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 12px;
  overflow: hidden;
  margin: 2rem 0;
  box-shadow: 0 18px 44px rgba(2, 8, 47, 0.22);
  backdrop-filter: blur(8px);
}

.stat-cell {
  background: rgba(4, 24, 95, 0.72);
  padding: 1.4rem 1.5rem;
  text-align: center;
}

.stat-value {
  display: block;
  font-size: 2rem;
  font-weight: 600;
  color: #fff;
  line-height: 1;
  margin-bottom: 0.3rem;
}

.stat-label {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.78);
}

/* ============================================================
   SECTION TITLES
   ============================================================ */
.section-header {
  margin: 2.5rem 0 1.2rem;
}

.section-header h2 {
  font-size: 1.7rem;
  font-weight: 600;
  color: var(--navy-deep);
  margin-bottom: 0.35rem;
}

.section-header p {
  color: var(--slate-medium);
  font-size: 0.98rem;
}

.landing-section-header {
  padding: 1.2rem 1.4rem;
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: rgba(4, 24, 95, 0.72);
  box-shadow: 0 16px 38px rgba(2, 8, 47, 0.20);
  backdrop-filter: blur(8px);
}

.landing-section-header h2 {
  color: #fff;
}

.landing-section-header p {
  color: rgba(255, 255, 255, 0.80);
  margin: 0;
}

.section-title {
  margin: 2rem 0 1rem;
  color: #ffffff !important;
  text-shadow: 0 2px 12px rgba(2, 8, 47, 0.55);
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: -0.02em;
}

.section-title a,
.section-title svg,
.section-title .anchor-link {
  color: #ffffff !important;
  fill: #ffffff !important;
}

/* ============================================================
   DISEASE / FEATURE CARDS
   ============================================================ */
.disease-grid,
.workflow-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.about-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}

/* Elevated card — design-spec default */
.disease-card,
.info-card,
.result-card,
.upload-card {
  padding: 1.25rem 1.35rem 1.5rem;
  border: 1px solid var(--border-card);
  border-radius: 12px;
  background: var(--bg-white);
  box-shadow: var(--shadow-1);
  transition: box-shadow 180ms ease, transform 180ms ease;
}

.disease-card:hover,
.info-card:hover {
  box-shadow: var(--shadow-2);
  transform: translateY(-2px);
}

/* Orange top-bar accent on disease cards */
.disease-card {
  border-top: 3px solid var(--orange-primary);
}

.disease-card,
.workflow-grid .info-card {
  border-color: rgba(255, 255, 255, 0.18);
  background:
    linear-gradient(160deg, rgba(5, 27, 105, 0.88), rgba(11, 77, 162, 0.66)),
    radial-gradient(circle at 90% 10%, rgba(255, 255, 255, 0.14), transparent 26%);
  box-shadow: 0 18px 44px rgba(2, 8, 47, 0.22);
  backdrop-filter: blur(8px);
}

.disease-card h3,
.info-card h3,
.result-card h3,
.upload-card h3 {
  font-size: 1.1rem;
  font-weight: 500;
  color: var(--navy-deep);
  margin-bottom: 0.5rem;
}

.disease-card h3,
.workflow-grid .info-card h3 {
  color: #fff;
}

.disease-card p,
.info-card p,
.result-card p,
.upload-card li {
  color: var(--slate-medium);
  line-height: 1.6;
  font-size: 0.95rem;
}

.disease-card p,
.workflow-grid .info-card p {
  color: rgba(255, 255, 255, 0.80);
}

/* Step number badge inside info cards */
.info-card strong {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  border-radius: 999px;
  background: var(--bg-pale-blue);
  color: var(--btn-blue);
  font-size: 0.82rem;
  font-weight: 600;
  margin-bottom: 0.6rem;
}

.workflow-grid .info-card strong {
  background: rgba(255, 255, 255, 0.16);
  color: #fff;
}

/* Accent card variant (dark navy) */
.accent-card {
  padding: 1.25rem 1.35rem 1.5rem;
  border-radius: 12px;
  background: var(--btn-blue);
  color: #fff;
  box-shadow: var(--shadow-2);
}

.accent-card h3, .accent-card p, .accent-card strong {
  color: #fff;
}

/* ============================================================
   WELCOME / INTRO CARD
   ============================================================ */
.welcome-card {
  margin: 1.5rem 0;
  padding: 1.4rem 1.5rem;
  border: 1px solid rgba(210, 228, 242, 0.72);
  border-left: 4px solid var(--orange-primary);
  border-radius: 12px;
  background: rgba(245, 249, 255, 0.94);
  box-shadow: var(--shadow-1);
}

.welcome-card h3 {
  font-size: 1.15rem;
  font-weight: 500;
  color: var(--navy-deep);
  margin-bottom: 0.4rem;
}

.welcome-card p {
  color: var(--slate-medium);
  line-height: 1.6;
  margin: 0;
}

/* ============================================================
   PAGE HEADING CARD
   ============================================================ */
.page-heading-card {
  margin: 1.5rem 0 1.5rem;
  padding: 2rem 2rem;
  border-radius: 12px;
  border: 1px solid rgba(210, 228, 242, 0.72);
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.96) 60%, rgba(238, 244, 255, 0.94));
  box-shadow: var(--shadow-1);
  border-left: 4px solid var(--btn-blue);
}

.page-heading-card h1 {
  font-size: clamp(1.9rem, 4vw, 2.7rem);
  font-weight: 600;
  color: var(--navy-deep);
  margin-bottom: 0.4rem;
}

.page-heading-card p {
  color: var(--slate-medium);
  font-size: 1rem;
  margin: 0;
}

/* ============================================================
   WARNING / DISCLAIMER
   ============================================================ */
.warning-card {
  margin: 1.2rem 0;
  padding: 1rem 1.2rem;
  border: 1px solid var(--warning-border);
  border-left: 4px solid var(--orange-secondary);
  border-radius: 12px;
  background: var(--warning-bg);
  color: var(--warning-text);
  font-size: 0.95rem;
  line-height: 1.6;
}

.landing-warning-card {
  border-color: rgba(255, 255, 255, 0.20);
  border-left-color: var(--orange-primary);
  background: rgba(4, 24, 95, 0.78);
  color: rgba(255, 255, 255, 0.86);
  box-shadow: 0 16px 38px rgba(2, 8, 47, 0.20);
  backdrop-filter: blur(8px);
}

.disclaimer {
  margin: 1.8rem 0 0;
  padding: 1rem 1.2rem;
  color: var(--slate-medium);
  border-radius: 12px;
  background: rgba(245, 249, 255, 0.94);
  border: 1px solid rgba(210, 228, 242, 0.72);
  font-size: 0.9rem;
  line-height: 1.6;
}

/* ============================================================
   STATUS / PLACEHOLDER NOTES
   ============================================================ */
.status-note, .placeholder-note {
  margin: 0 0 1rem;
  border: 1px solid rgba(210, 228, 242, 0.72);
  background: rgba(245, 249, 255, 0.94);
  border-radius: 10px;
  padding: 0.9rem 1rem;
  color: var(--slate-medium);
  font-size: 0.92rem;
}

.placeholder-note {
  border-color: #FECACA;
  background: #FEF2F2;
}

.placeholder-note strong { color: var(--error-red); }

/* ============================================================
   METRIC CARDS (performance page)
   ============================================================ */
.model-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  margin: 1rem 0 1.5rem;
}

.metric-card {
  padding: 1rem 1rem 1.2rem;
  border: 1px solid rgba(210, 228, 242, 0.72);
  border-radius: 12px;
  background: linear-gradient(160deg, rgba(255, 255, 255, 0.96), rgba(238, 244, 255, 0.94));
  box-shadow: var(--shadow-1);
  transition: box-shadow 180ms ease;
}

.metric-card:hover { box-shadow: var(--shadow-2); }

.metric-card span {
  display: block;
  color: var(--slate-medium);
  font-size: 0.76rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.3rem;
}

.metric-card strong {
  display: block;
  color: var(--btn-blue);
  font-size: 1.35rem;
  font-weight: 600;
}

.metric-card p {
  margin: 0.3rem 0 0;
  color: var(--slate-medium);
  font-size: 0.84rem;
  line-height: 1.4;
}

/* ============================================================
   SCAN PAGE UPLOAD CARD
   ============================================================ */
.scan-intro-panel {
  position: relative;
  overflow: hidden;
  border-radius: 12px;
  padding: 2rem 2rem 1.75rem;
  background:
    linear-gradient(160deg, rgba(6,12,64,0.88) 0%, rgba(41,82,137,0.78) 100%),
    radial-gradient(circle at 88% 12%, rgba(255, 255, 255, 0.12), transparent 34%);
  color: #fff;
  margin-bottom: 1.5rem;
  min-height: 13rem;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  box-shadow: var(--shadow-3);
}

.scan-intro-panel h2 {
  color: #fff;
  font-size: 1.6rem;
  font-weight: 600;
  margin-bottom: 0.45rem;
}

.scan-intro-panel p {
  color: rgba(255,255,255,0.80);
  font-size: 0.95rem;
  line-height: 1.6;
  margin: 0;
}

/* ============================================================
   RESULT PAGE
   ============================================================ */
.result-heading {
  margin: 1rem 0 1.5rem;
}

.result-heading h1 {
  font-size: clamp(1.8rem, 4vw, 2.6rem);
  font-weight: 600;
  color: #fff;
  margin-bottom: 0.4rem;
}

.result-heading p { color: rgba(255, 255, 255, 0.80); }

.result-feature-card {
  display: grid;
  grid-template-columns: 3.6rem 1fr auto;
  align-items: center;
  gap: 1.1rem;
  margin: 1.2rem 0;
  padding: 1.35rem 1.5rem;
  border: 1px solid var(--border-card);
  border-radius: 12px;
  background: var(--bg-white);
  box-shadow: var(--shadow-2);
}

.result-letter {
  width: 3.6rem;
  height: 3.6rem;
  border-radius: 12px;
  background: linear-gradient(135deg, var(--btn-blue), var(--navy-deep));
  color: #fff;
  display: grid;
  place-items: center;
  font-size: 1.6rem;
  font-weight: 600;
  flex-shrink: 0;
}

.result-feature-card h2 {
  font-size: 1.2rem;
  font-weight: 500;
  color: var(--navy-deep);
  margin-bottom: 0.2rem;
}

.result-feature-card p {
  margin: 0;
  color: var(--slate-medium);
  font-size: 0.93rem;
  line-height: 1.4;
}

.confidence-ring {
  width: 3.4rem;
  height: 3.4rem;
  border-radius: 999px;
  display: grid;
  place-items: center;
  color: var(--btn-blue);
  font-size: 0.75rem;
  font-weight: 600;
  background:
    radial-gradient(circle at center, white 55%, transparent 56%),
    conic-gradient(var(--orange-primary) var(--score), var(--bg-pale-blue) 0);
}

.prediction-panel {
  margin: 1rem 0;
  border: 1px solid var(--border-card);
  border-radius: 12px;
  background: var(--bg-faint-blue);
  padding: 1.2rem 1.4rem;
  box-shadow: var(--shadow-1);
}

.prediction-panel h3 {
  margin: 0 0 0.5rem;
  font-size: 1.05rem;
  font-weight: 500;
  color: var(--navy-deep);
}

.prediction-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 1rem;
  padding: 0.75rem 0;
  border-top: 1px solid var(--border-card);
}

.prediction-row span { color: var(--navy-medium); }

.prediction-row span:last-child {
  color: var(--btn-blue);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.consultant {
  margin: 1.2rem 0;
  padding: 1.2rem 1.4rem;
  border-left: 4px solid var(--orange-primary);
  border-radius: 12px;
  background: var(--bg-faint-blue);
  border-top: 1px solid var(--border-card);
  border-right: 1px solid var(--border-card);
  border-bottom: 1px solid var(--border-card);
}

.consultant h3 {
  margin: 0 0 0.6rem;
  font-size: 1.05rem;
  font-weight: 500;
  color: var(--navy-deep);
}

.consultant p {
  color: var(--slate-medium);
  line-height: 1.65;
  font-size: 0.95rem;
}

/* ============================================================
   CROP / UPLOAD TOOL
   ============================================================ */
.crop-heading {
  margin: 0 0 1rem;
}

.crop-heading h2 {
  font-size: clamp(1.5rem, 3vw, 2rem);
  font-weight: 600;
  color: #fff;
}

.crop-heading p { color: rgba(255, 255, 255, 0.80); }

.crop-preview-frame {
  margin-bottom: 0.45rem;
  color: rgba(255, 255, 255, 0.80);
  font-size: 0.88rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.quality-card {
  display: grid;
  grid-template-columns: 3.2rem 1fr auto;
  align-items: center;
  gap: 1rem;
  margin: 1rem 0 1.3rem;
  padding: 0.95rem 1.2rem;
  border: 1px solid var(--border-card);
  border-radius: 10px;
  background: var(--bg-white);
  box-shadow: var(--shadow-1);
}

.quality-thumb {
  width: 3.2rem;
  height: 3.2rem;
  overflow: hidden;
  border-radius: 6px;
  border: 1px solid var(--border-card);
}

.quality-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.quality-card strong { color: #059669; font-weight: 600; }
.quality-card p { margin: 0.18rem 0 0; color: var(--slate-medium); font-size: 0.88rem; }
.quality-check { color: #059669; font-weight: 700; font-size: 1.1rem; }

/* ============================================================
   BUTTONS
   ============================================================ */
div.stButton > button,
div.stDownloadButton > button {
  min-height: 46px;
  border-radius: 999px;
  border: 1px solid var(--orange-primary);
  background: var(--orange-primary);
  color: #fff;
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  font-size: 1rem;
  padding: 12px 24px;
  transition: background 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
  cursor: pointer;
}

div.stButton > button[kind="primary"] {
  min-height: 50px;
  font-size: 1.05rem;
  box-shadow: 0 8px 24px rgba(255, 99, 36, 0.32);
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
  background: var(--orange-secondary);
  border-color: var(--orange-secondary);
  box-shadow: 0 10px 26px rgba(255, 99, 36, 0.34);
}

div.stButton > button:active,
div.stDownloadButton > button:active {
  background: #D94F17;
  border-color: #D94F17;
  box-shadow: var(--shadow-2);
}

div.stButton > button:disabled {
  background: var(--btn-disabled);
  border-color: var(--btn-disabled);
  color: #fff;
  cursor: not-allowed;
  box-shadow: none;
  opacity: 1;
}

/* ============================================================
   FORM ELEMENTS
   ============================================================ */
[data-testid="stFileUploader"],
[data-testid="stCameraInput"] {
  margin: 0 0 1rem;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stCameraInput"] label,
[data-testid="stCameraInput"] p {
  color: #ffffff !important;
  text-shadow: 0 2px 10px rgba(2, 8, 47, 0.55);
}

[data-testid="stFileUploader"] section {
  border-radius: 12px;
  border: 1.5px dashed var(--border-input);
  background: var(--bg-faint-blue);
}

[data-testid="stFileUploader"] section:hover {
  border-color: var(--border-input);
  background: var(--bg-faint-blue);
}

[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:focus,
[data-testid="stFileUploader"] button:active {
  background: #ffffff !important;
  border-color: var(--border-input) !important;
  color: var(--navy-deep) !important;
  box-shadow: none !important;
}

[data-baseweb="tab-list"] {
  gap: 0.4rem;
  margin-bottom: 0.75rem;
}

[data-baseweb="tab"] {
  border-radius: 999px;
  color: rgba(255, 255, 255, 0.86) !important;
  font-family: 'Inter', sans-serif;
  font-size: 0.9rem;
}

[data-baseweb="tab"][aria-selected="true"] {
  color: #ffffff !important;
  background: rgba(255, 255, 255, 0.14);
}

[data-baseweb="tab"] p,
[data-baseweb="tab"] span {
  color: inherit !important;
}

[data-testid="stExpander"] {
  border-radius: 12px;
  border: 1px solid var(--border-card) !important;
  margin: 0.8rem 0;
}

[data-testid="stCheckbox"] {
  margin: 0.5rem 0;
}

[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] p,
[data-testid="stCheckbox"] span {
  color: #ffffff !important;
  text-shadow: 0 2px 10px rgba(2, 8, 47, 0.55);
}

/* Streamlit native metric widget */
[data-testid="stMetricValue"] {
  color: var(--btn-blue) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
}

[data-testid="stMetricLabel"] {
  color: var(--slate-medium) !important;
  font-size: 0.82rem !important;
}

/* Images */
[data-testid="stImage"] img {
  border-radius: 10px;
  border: 1px solid var(--border-card);
}

/* Sidebar nav */
[data-testid="stSidebar"] {
  background: var(--bg-faint-blue);
  border-right: 1px solid var(--border-card);
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] {
  color: var(--navy-medium);
}

/* ============================================================
   TOP STRIP (cropper separator)
   ============================================================ */
.top-strip {
  height: 3px;
  margin: 1.5rem -1.5rem 1.5rem;
  background: linear-gradient(90deg, var(--btn-blue), var(--orange-primary));
  border-radius: 0;
}

/* ============================================================
   RESPONSIVE BREAKPOINTS
   ============================================================ */
@media (max-width: 900px) {
  .hero-fullbleed {
    padding: 3.5rem 6% 3rem;
    min-height: 22rem;
  }

  .welcome-split {
    grid-template-columns: 1fr;
    padding: 2.5rem 6%;
    gap: 2rem;
  }

  .welcome-split-image {
    min-height: 14rem;
    aspect-ratio: 16/9;
  }

  .disease-grid,
  .workflow-grid {
    grid-template-columns: 1fr 1fr;
  }

  .clinical-gallery {
    grid-template-columns: 1fr;
  }

  .clinical-gallery-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .about-grid {
    grid-template-columns: 1fr;
  }

  .model-metrics {
    grid-template-columns: repeat(2, 1fr);
  }

  .stats-strip {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 600px) {
  .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
  }

  .hero-fullbleed {
    padding: 2.5rem 5% 2.5rem;
    margin: 0 -1rem;
  }

  .welcome-split {
    margin: 0 -1rem;
    padding: 2rem 5%;
  }

  .disease-grid,
  .workflow-grid,
  .about-grid,
  .hero-card {
    grid-template-columns: 1fr;
  }

  .clinical-gallery {
    padding: 1.25rem;
  }

  .clinical-gallery-grid {
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem;
    min-height: 12rem;
  }

  .clinical-shot {
    min-height: 12rem;
  }

  .model-metrics {
    grid-template-columns: 1fr 1fr;
  }

  .result-feature-card {
    grid-template-columns: 3rem 1fr;
  }

  .confidence-ring {
    grid-column: 1 / -1;
  }

  .nav-badge { display: none; }
}

/* ============================================================
   REDUCED MOTION
   ============================================================ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    transition: none !important;
    animation: none !important;
  }
}
</style>
"""


def apply_styles() -> None:
    css = APP_CSS.replace("__GLOBAL_BACKGROUND__", _image_layer(GLOBAL_BACKGROUND, "image/png"))
    css = css.replace("__SCAN_BACKGROUND__",       _image_layer(SCAN_BACKGROUND,   "image/png"))
    css = css.replace("__HERO_DOCTOR_BG__",       _image_layer(HERO_DOCTOR_BG,    "image/png"))
    css = css.replace("__DOCTOR_AT_WORK__",       _image_layer(DOCTOR_AT_WORK,    "image/png"))
    st.markdown(css, unsafe_allow_html=True)
