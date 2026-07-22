import streamlit as st
import pandas as pd
import joblib
import os
import json
import textwrap

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Prestige — Customer Churn Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL (cached — loaded once per server process, not
# once per widget interaction)
# ==========================================================

MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "models/metrics.json"
IMPORTANCE_PATH = "models/feature_importance.csv"
HISTORY_FILE = "data/prediction_history.csv"


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_scaler():
    m = joblib.load(MODEL_PATH)
    s = joblib.load(SCALER_PATH)
    return m, s


@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_feature_importance():
    if not os.path.exists(IMPORTANCE_PATH):
        return None
    try:
        df = pd.read_csv(IMPORTANCE_PATH)
        return df.sort_values("importance", ascending=False)
    except Exception:
        return None


if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error(
        "Model files not found. Run 'python train.py' locally first to "
        "generate 'models/churn_model.pkl' and 'models/scaler.pkl'."
    )
    st.stop()

model, scaler = load_model_and_scaler()
metrics = load_metrics()
feature_importance = load_feature_importance()

# ==========================================================
# LUXURY THEME — FONTS, PALETTE, LIVE BACKGROUND
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;1,500&family=Jost:wght@300;400;500;600;700&display=swap');

:root{
    --bg-0:#07060b;
    --bg-1:#120e1c;
    --bg-2:#1a1428;
    --gold:#c9a961;
    --gold-light:#eddcb0;
    --gold-dim:rgba(201,169,97,0.35);
    --emerald:#123c2e;
    --emerald-light:#1f6e52;
    --burgundy:#5c1620;
    --burgundy-light:#8f2433;
    --cream:#f3ecdd;
    --muted:#9b927e;
    --glass:rgba(255,255,255,0.035);
    --glass-border:rgba(201,169,97,0.22);
}

html, body, [class*="css"]{
    font-family:'Jost', sans-serif;
}

h1,h2,h3,h4,h5,h6{
    font-family:'Playfair Display', serif !important;
    letter-spacing:0.3px;
}

/* ---------- LIVE ANIMATED BACKGROUND ---------- */

.stApp{
    background:
        radial-gradient(ellipse 900px 600px at 12% 8%, rgba(201,169,97,0.10), transparent 60%),
        radial-gradient(ellipse 800px 700px at 88% 18%, rgba(31,110,82,0.14), transparent 60%),
        radial-gradient(ellipse 1000px 800px at 50% 100%, rgba(92,22,32,0.10), transparent 55%),
        linear-gradient(160deg, var(--bg-0) 0%, var(--bg-1) 55%, var(--bg-2) 100%);
    background-size: 140% 140%, 140% 140%, 140% 140%, 100% 100%;
    animation: auroraShift 26s ease-in-out infinite alternate;
}

@keyframes auroraShift{
    0%   { background-position: 0% 0%, 100% 0%, 50% 100%, 0 0; }
    50%  { background-position: 20% 15%, 75% 25%, 60% 85%, 0 0; }
    100% { background-position: 5% 25%, 90% 5%, 40% 100%, 0 0; }
}

.aurora-orbs{
    position:fixed;
    inset:0;
    z-index:0;
    overflow:hidden;
    pointer-events:none;
}
.aurora-orbs span{
    position:absolute;
    border-radius:50%;
    filter:blur(80px);
    opacity:0.28;
}
.orb-a{
    width:420px;height:420px;
    top:-120px;left:-100px;
    background:radial-gradient(circle,var(--gold) 0%, transparent 70%);
    animation:driftA 22s ease-in-out infinite alternate;
}
.orb-b{
    width:460px;height:460px;
    top:20%;right:-140px;
    background:radial-gradient(circle,var(--emerald-light) 0%, transparent 70%);
    animation:driftB 28s ease-in-out infinite alternate;
}
.orb-c{
    width:380px;height:380px;
    bottom:-140px;left:35%;
    background:radial-gradient(circle,var(--burgundy-light) 0%, transparent 70%);
    animation:driftC 24s ease-in-out infinite alternate;
}
@keyframes driftA{ from{transform:translate(0,0);} to{transform:translate(60px,50px);} }
@keyframes driftB{ from{transform:translate(0,0);} to{transform:translate(-50px,60px);} }
@keyframes driftC{ from{transform:translate(0,0);} to{transform:translate(40px,-40px);} }

@media (prefers-reduced-motion: reduce){
    .stApp{ animation:none; }
    .orb-a,.orb-b,.orb-c{ animation:none; }
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg, #0a0810 0%, #100c1a 100%);
    border-right:1px solid var(--glass-border);
}
section[data-testid="stSidebar"] *{
    color:var(--cream) !important;
}

.brand-mark{
    font-family:'Playfair Display', serif;
    font-size:26px;
    font-weight:700;
    letter-spacing:2px;
    color:var(--gold-light) !important;
    text-align:center;
    margin:6px 0 0 0;
}
.brand-sub{
    text-align:center;
    font-size:11px;
    letter-spacing:3px;
    text-transform:uppercase;
    color:var(--muted) !important;
    margin-bottom:18px;
}

.gold-rule{
    display:flex;
    align-items:center;
    justify-content:center;
    gap:10px;
    margin:18px 0;
}
.gold-rule::before, .gold-rule::after{
    content:"";
    height:1px;
    flex:1;
    background:linear-gradient(90deg, transparent, var(--gold-dim), transparent);
}
.gold-rule span{
    display:inline-block;
    width:7px;height:7px;
    background:var(--gold);
    transform:rotate(45deg);
    box-shadow:0 0 8px var(--gold-dim);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label{
    background:var(--glass);
    border:1px solid transparent;
    border-radius:8px;
    padding:10px 12px;
    margin-bottom:6px;
    transition:all .25s ease;
    text-transform:uppercase;
    font-size:12.5px;
    letter-spacing:1.5px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
    border-color:var(--glass-border);
    background:rgba(201,169,97,0.06);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label div:first-child{
    border-color:var(--gold) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label div:first-child > div{
    background-color:var(--gold) !important;
}

.status-chip{
    background:var(--glass);
    border:1px solid var(--glass-border);
    border-radius:8px;
    padding:10px 14px;
    font-size:12px;
    letter-spacing:1px;
    text-transform:uppercase;
    text-align:center;
    margin-bottom:8px;
    color:var(--gold-light) !important;
}

.sidebar-footer{
    text-align:center;
    font-size:11px;
    letter-spacing:1px;
    color:var(--muted) !important;
    margin-top:10px;
}
.sidebar-footer .name{
    font-family:'Playfair Display', serif;
    font-size:15px;
    color:var(--cream) !important;
    letter-spacing:0.5px;
    margin-bottom:2px;
}

/* ---------- HERO ---------- */

.eyebrow{
    text-transform:uppercase;
    letter-spacing:5px;
    font-size:12px;
    color:var(--gold) !important;
    margin-bottom:6px;
}
.hero-title{
    font-family:'Playfair Display', serif;
    font-size:48px;
    font-weight:700;
    color:var(--cream) !important;
    line-height:1.15;
    margin:0;
}
.hero-title em{
    font-style:italic;
    color:var(--gold-light) !important;
}
.hero-sub{
    font-size:16px;
    color:var(--muted) !important;
    margin-top:10px;
    max-width:640px;
    font-weight:300;
}

.section-divider{
    display:flex;
    align-items:center;
    gap:14px;
    margin:34px 0 26px 0;
}
.section-divider::before, .section-divider::after{
    content:"";
    height:1px;
    flex:1;
    background:linear-gradient(90deg, transparent, var(--gold-dim), transparent);
}
.section-divider .diamond{
    width:8px;height:8px;
    background:var(--gold);
    transform:rotate(45deg);
    box-shadow:0 0 10px var(--gold-dim);
}

/* ---------- METRIC / GLASS CARDS ---------- */

.metric-card{
    background:var(--glass);
    border:1px solid var(--glass-border);
    padding:26px 18px;
    border-radius:14px;
    text-align:center;
    backdrop-filter:blur(6px);
    box-shadow:0 8px 30px rgba(0,0,0,0.35);
    transition:transform .25s ease, border-color .25s ease;
}
.metric-card:hover{
    transform:translateY(-3px);
    border-color:var(--gold-dim);
}
.metric-number{
    font-family:'Playfair Display', serif;
    font-size:36px;
    font-weight:700;
    color:var(--gold-light);
    line-height:1;
}
.metric-text{
    margin-top:8px;
    font-size:11.5px;
    letter-spacing:2px;
    text-transform:uppercase;
    color:var(--muted);
}

.section-label{
    text-transform:uppercase;
    letter-spacing:3px;
    font-size:12.5px;
    color:var(--gold) !important;
    margin-bottom:2px;
}
.section-title{
    font-family:'Playfair Display', serif;
    font-size:26px;
    color:var(--cream) !important;
    margin-top:0;
}

/* ---------- FORM CONTROLS ---------- */

.main .block-container,
.main .block-container p,
.main .block-container span,
.main .block-container li,
.main .block-container ul,
.main .block-container ol{
    color:var(--cream) !important;
}
.main .block-container strong,
.main .block-container b{
    color:var(--gold-light) !important;
}
.main .block-container em{
    color:var(--gold) !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] span,
.stMarkdown p, .stMarkdown li, .stMarkdown ul, .stMarkdown span{
    color:var(--cream) !important;
}
[data-testid="stMarkdownContainer"] strong,
.stMarkdown strong{
    color:var(--gold-light) !important;
}
label{
    color:var(--cream) !important;
}
.main .block-container div[data-testid="stWidgetLabel"] p{
    text-transform:uppercase;
    letter-spacing:1.5px;
    font-size:11.5px !important;
    color:var(--muted) !important;
}

div[data-baseweb="select"]{
    background:rgba(255,255,255,0.04) !important;
    border:1px solid var(--glass-border) !important;
    border-radius:10px !important;
}
div[data-baseweb="select"] *{
    color:var(--cream) !important;
}
div[data-baseweb="select"]:hover{
    border-color:var(--gold-dim) !important;
}

.stNumberInput input{
    background:rgba(255,255,255,0.04) !important;
    color:var(--cream) !important;
    border:1px solid var(--glass-border) !important;
    border-radius:10px !important;
}

.stSlider [data-baseweb="slider"] div[role="slider"]{
    background:var(--gold) !important;
    box-shadow:0 0 10px var(--gold-dim) !important;
}
.stSlider [data-baseweb="slider"] > div > div{
    background:var(--gold-dim) !important;
}

/* ---------- BUTTONS ---------- */

.stButton>button{
    width:100%;
    height:54px;
    border-radius:10px;
    border:1px solid var(--gold);
    background:linear-gradient(90deg, #b6923f, var(--gold) 45%, #b6923f);
    color:#151022;
    font-family:'Jost', sans-serif;
    font-size:14px;
    font-weight:600;
    letter-spacing:3px;
    text-transform:uppercase;
    transition:all .25s ease;
    box-shadow:0 6px 22px rgba(201,169,97,0.18);
}
.stButton>button:hover{
    background:linear-gradient(90deg, var(--gold-light), var(--gold) 55%, var(--gold-light));
    box-shadow:0 8px 28px rgba(201,169,97,0.35);
    transform:translateY(-1px);
}

.stDownloadButton>button{
    border-radius:10px;
    border:1px solid var(--gold-dim);
    background:transparent;
    color:var(--gold-light) !important;
    letter-spacing:1.5px;
    text-transform:uppercase;
    font-size:12.5px;
}
.stDownloadButton>button:hover{
    border-color:var(--gold);
    background:rgba(201,169,97,0.08);
}

/* ---------- RESULT BANNERS ---------- */

.result-banner{
    padding:30px;
    border-radius:14px;
    text-align:center;
    border:1px solid var(--gold-dim);
    box-shadow:0 10px 40px rgba(0,0,0,0.4);
}
.result-banner.stay{
    background:linear-gradient(135deg, var(--emerald), #0c2620);
}
.result-banner.churn{
    background:linear-gradient(135deg, var(--burgundy), #3a0e15);
}
.result-eyebrow{
    text-transform:uppercase;
    letter-spacing:4px;
    font-size:11.5px;
    color:var(--gold-light);
    margin-bottom:10px;
}
.result-headline{
    font-family:'Playfair Display', serif;
    font-size:32px;
    font-weight:700;
    color:var(--cream);
    margin-bottom:14px;
}

/* ---------- RISK BADGE ---------- */

.risk-badge{
    display:inline-block;
    padding:6px 18px;
    border-radius:999px;
    font-size:11.5px;
    letter-spacing:2px;
    text-transform:uppercase;
    font-weight:600;
    border:1px solid;
}
.risk-badge.low{
    color:#8fd6b4;
    border-color:var(--emerald-light);
    background:rgba(31,110,82,0.15);
}
.risk-badge.medium{
    color:var(--gold-light);
    border-color:var(--gold);
    background:rgba(201,169,97,0.15);
}
.risk-badge.high{
    color:#ff9d92;
    border-color:var(--burgundy-light);
    background:rgba(143,36,51,0.18);
}

/* ---------- PROGRESS ---------- */

div[data-testid="stProgress"] > div > div{
    background:linear-gradient(90deg, var(--gold), var(--gold-light)) !important;
}

/* ---------- METRIC WIDGETS ---------- */

div[data-testid="stMetric"]{
    background:var(--glass);
    border:1px solid var(--glass-border);
    border-radius:12px;
    padding:14px 10px;
}
div[data-testid="stMetricLabel"]{
    text-transform:uppercase;
    letter-spacing:1.5px;
    color:var(--muted) !important;
}
div[data-testid="stMetricValue"]{
    font-family:'Playfair Display', serif;
    color:var(--gold-light) !important;
}

/* ---------- DATAFRAME ---------- */

div[data-testid="stDataFrame"]{
    border:1px solid var(--glass-border);
    border-radius:10px;
    overflow:hidden;
}

/* ---------- FEATURE IMPORTANCE BARS ---------- */

.imp-row{
    margin-bottom:14px;
}
.imp-label{
    display:flex;
    justify-content:space-between;
    font-size:12.5px;
    letter-spacing:0.5px;
    color:var(--cream);
    margin-bottom:5px;
}
.imp-label span.val{
    color:var(--gold-light);
}
.imp-track{
    background:rgba(255,255,255,0.05);
    border-radius:6px;
    height:8px;
    overflow:hidden;
}
.imp-fill{
    height:100%;
    background:linear-gradient(90deg, var(--gold), var(--gold-light));
    border-radius:6px;
}

/* ---------- RATIO BAR (History) ---------- */

.ratio-bar{
    display:flex;
    height:14px;
    border-radius:8px;
    overflow:hidden;
    border:1px solid var(--glass-border);
    margin-bottom:8px;
}
.ratio-fill.churn{
    background:linear-gradient(90deg, var(--burgundy-light), var(--burgundy));
}
.ratio-fill.stay{
    background:linear-gradient(90deg, var(--emerald-light), var(--emerald));
}
.ratio-legend{
    display:flex;
    justify-content:space-between;
    font-size:11.5px;
    letter-spacing:1px;
    color:var(--muted);
    text-transform:uppercase;
}

/* ---------- TABS ---------- */

button[data-baseweb="tab"]{
    color:var(--muted) !important;
    font-size:12.5px;
    letter-spacing:1.5px;
    text-transform:uppercase;
}
button[data-baseweb="tab"][aria-selected="true"]{
    color:var(--gold-light) !important;
}
div[data-baseweb="tab-highlight"]{
    background-color:var(--gold) !important;
}
div[data-baseweb="tab-border"]{
    background-color:var(--glass-border) !important;
}

/* ---------- FILE UPLOADER ---------- */

[data-testid="stFileUploader"] section{
    background:var(--glass) !important;
    border:1px dashed var(--glass-border) !important;
    border-radius:12px !important;
}

/* ---------- MISC ---------- */

hr{
    border-color:var(--glass-border) !important;
}
::selection{
    background:var(--gold-dim);
}

</style>

<div class="aurora-orbs">
    <span class="orb-a"></span>
    <span class="orb-b"></span>
    <span class="orb-c"></span>
</div>
""", unsafe_allow_html=True)


def gold_divider():
    st.markdown('<div class="section-divider"><span class="diamond"></span></div>', unsafe_allow_html=True)


def risk_tier(probability):
    """Return (label, css_class) for a churn probability."""
    if probability is None:
        return None, None
    if probability < 0.40:
        return "Low Risk", "low"
    elif probability < 0.70:
        return "Medium Risk", "medium"
    else:
        return "High Risk", "high"


# ==========================================================
# ENCODING (single source of truth — mirrors train.py exactly)
# ==========================================================

COLUMN_ENCODINGS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
}

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

WIDGET_KEYS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


def encode_dataframe(df):
    """Encode a raw (human-readable) dataframe into model-ready numeric form.
    Returns (encoded_df, n_dropped) where invalid rows are dropped."""
    out = df.copy()
    for col, mapping in COLUMN_ENCODINGS.items():
        if col in out.columns:
            out[col] = out[col].map(mapping)

    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=FEATURE_COLUMNS)
    dropped = before - len(out)
    return out, dropped


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown('<div class="brand-mark">PRESTIGE</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Churn Intelligence Suite</div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"><span></span></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Prediction", "History", "About"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="gold-rule"><span></span></div>', unsafe_allow_html=True)

    st.markdown('<div class="status-chip">● Model Loaded</div>', unsafe_allow_html=True)

    if metrics:
        st.markdown(f'<div class="status-chip">{metrics.get("model_name", "Model")}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-chip">Version 1.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"><span></span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">
        <div class="name">Palavali Charan Kumar Reddy</div>
        Developer
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# HEADER (shown on every page)
# ==========================================================

st.markdown("""
<div class="eyebrow">Premium Customer Intelligence</div>
<div class="hero-title">Customer Churn <em>Prediction</em></div>
<div class="hero-sub">
An AI-driven retention system that reads a customer's profile and tells you,
with confidence, whether they are about to walk away.
</div>
""", unsafe_allow_html=True)

gold_divider()

# ==========================================================
# PAGE ROUTER
# ==========================================================

if page == "Prediction":

    # ------------------------------------------------------
    # DASHBOARD METRICS (real metrics if train.py has been run)
    # ------------------------------------------------------

    dataset_rows_display = f"{metrics['dataset_rows']:,}" if metrics else "7,043"
    accuracy_display = f"{metrics['test_accuracy'] * 100:.0f}%" if metrics else "82%"
    model_display = metrics["model_name"] if metrics else "AI"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{dataset_rows_display}</div>
            <div class="metric-text">Customers</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(FEATURE_COLUMNS)}</div>
            <div class="metric-text">Features</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{accuracy_display}</div>
            <div class="metric-text">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{model_display}</div>
            <div class="metric-text">Model Type</div>
        </div>
        """, unsafe_allow_html=True)

    if not metrics:
        st.caption("Showing placeholder stats — run `python train.py` to populate real model metrics.")

    gold_divider()

    tab_single, tab_batch = st.tabs(["Single Customer", "Batch Upload (CSV)"])

    # ========================================================
    # TAB 1 — SINGLE CUSTOMER PREDICTION
    # ========================================================

    with tab_single:

        top_label, top_reset = st.columns([4, 1])
        with top_label:
            st.markdown('<div class="section-label">Step One</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
        with top_reset:
            st.write("")
            if st.button("Reset Form"):
                for k in WIDGET_KEYS:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        st.write("")

        left, right = st.columns(2)

        with left:

            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="SeniorCitizen")
            partner = st.selectbox("Partner", ["Yes", "No"], key="Partner")
            dependents = st.selectbox("Dependents", ["Yes", "No"], key="Dependents")
            tenure = st.slider("Tenure (Months)", 0, 72, 12, key="tenure")
            phone = st.selectbox("Phone Service", ["Yes", "No"], key="PhoneService")

            multiple = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
                key="MultipleLines"
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
                key="InternetService"
            )

            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
                key="OnlineSecurity"
            )

        with right:

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
                key="OnlineBackup"
            )

            device = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
                key="DeviceProtection"
            )

            tech = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
                key="TechSupport"
            )

            tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
                key="StreamingTV"
            )

            movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
                key="StreamingMovies"
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
                key="Contract"
            )

            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="PaperlessBilling")

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                key="PaymentMethod"
            )

            monthly = st.number_input(
                "Monthly Charges (₹)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                key="MonthlyCharges"
            )

            total = st.number_input(
                "Total Charges (₹)",
                min_value=0.0,
                max_value=10000.0,
                value=1000.0,
                key="TotalCharges"
            )

        gold_divider()

        st.markdown('<div class="section-label">Step Two</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Run the Prediction</div>', unsafe_allow_html=True)
        st.write("")

        predict = st.button("Predict Customer Churn")

        if predict:

            row = pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device,
                "TechSupport": tech,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total
            }])

            encoded, dropped_rows = encode_dataframe(row)

            if encoded.empty:
                st.error(
                    "Couldn't process this input — one of the selected values "
                    "didn't match an expected category. Try adjusting your "
                    "selections and predicting again."
                )
                st.stop()

            expected_n = getattr(scaler, "n_features_in_", len(FEATURE_COLUMNS))
            if expected_n != len(FEATURE_COLUMNS):
                st.error(
                    f"Model/scaler mismatch: the deployed scaler expects "
                    f"{expected_n} feature(s), but the app is sending "
                    f"{len(FEATURE_COLUMNS)}. The 'models/' files in this "
                    f"deployment are likely out of date — retrain by running "
                    f"`python main.py` locally and redeploy the updated "
                    f"'models/churn_model.pkl' and 'models/scaler.pkl'."
                )
                st.stop()

            try:
                sample_scaled = scaler.transform(encoded[FEATURE_COLUMNS])
            except ValueError as e:
                st.error(
                    "The model couldn't process this input. This usually means "
                    "the deployed 'models/' files don't match this version of "
                    "the app. Retrain with `python main.py` and redeploy the "
                    "updated model files."
                )
                st.caption(f"Technical detail: {e}")
                st.stop()

            prediction = model.predict(sample_scaled)[0]

            probability = None
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(sample_scaled)[0][1])

            tier_label, tier_class = risk_tier(probability)

            gold_divider()

            st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prediction Outcome</div>', unsafe_allow_html=True)
            st.write("")

            badge_html = f'<span class="risk-badge {tier_class}">{tier_label}</span>' if tier_label else ""

            if prediction == 1:
                st.markdown(f"""
                <div class="result-banner churn">
                    <div class="result-eyebrow">Retention Risk Detected</div>
                    <div class="result-headline">This customer is likely to churn</div>
                    {badge_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-banner stay">
                    <div class="result-eyebrow">Retention Risk Low</div>
                    <div class="result-headline">This customer is likely to stay</div>
                    {badge_html}
                </div>
                """, unsafe_allow_html=True)

            if probability is not None:
                st.write("")
                st.markdown('<div class="section-label">Confidence</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Churn Probability</div>', unsafe_allow_html=True)
                st.progress(probability)
                st.metric("Probability", f"{probability * 100:.2f}%")

            gold_divider()

            st.markdown('<div class="section-label">Record</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Customer Summary</div>', unsafe_allow_html=True)
            st.write("")

            col1, col2, col3 = st.columns(3)

            col1.metric("Gender", gender)
            col2.metric("Tenure", f"{tenure} mo")
            col3.metric("Monthly Charges", f"${monthly}")

            col1.metric("Internet", internet)
            col2.metric("Contract", contract)
            col3.metric("Payment", payment)

            result_row = pd.DataFrame({
                "Gender": [gender],
                "Tenure": [tenure],
                "Internet": [internet],
                "Contract": [contract],
                "MonthlyCharges": [monthly],
                "TotalCharges": [total],
                "Prediction": ["Churn" if prediction == 1 else "Stay"],
                "Probability": [round(probability, 4) if probability is not None else None]
            })

            os.makedirs("data", exist_ok=True)

            if os.path.exists(HISTORY_FILE):
                old = pd.read_csv(HISTORY_FILE)
                combined = pd.concat([old, result_row], ignore_index=True)
            else:
                combined = result_row

            combined.to_csv(HISTORY_FILE, index=False)

            st.success("Prediction saved to history.")

            st.download_button(
                "Download This Result (CSV)",
                data=result_row.to_csv(index=False).encode("utf-8"),
                file_name="churn_prediction_result.csv",
                mime="text/csv"
            )

    # ========================================================
    # TAB 2 — BATCH PREDICTION VIA CSV UPLOAD
    # ========================================================

    with tab_batch:

        st.markdown('<div class="section-label">Bulk Processing</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Batch Prediction</div>', unsafe_allow_html=True)
        st.write("")

        st.markdown(textwrap.dedent(f"""\
        Upload a CSV with one row per customer. Required columns:

        `{"`, `".join(FEATURE_COLUMNS)}`

        An optional `customerID` column is kept for reference in the results.
        """))

        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

        if uploaded is not None:

            try:
                raw_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                raw_df = None

            if raw_df is not None:

                missing_cols = [c for c in FEATURE_COLUMNS if c not in raw_df.columns]

                if missing_cols:
                    st.error(
                        "The uploaded file is missing required column(s): "
                        + ", ".join(missing_cols)
                    )
                else:
                    encoded, dropped = encode_dataframe(raw_df)

                    if encoded.empty:
                        st.warning("No valid rows found after validation — check category spellings match the expected values.")
                    else:
                        try:
                            scaled = scaler.transform(encoded[FEATURE_COLUMNS])
                        except ValueError as e:
                            st.error(
                                "The model couldn't process this file. This usually "
                                "means the deployed 'models/' files don't match this "
                                "version of the app. Retrain with `python main.py` "
                                "and redeploy the updated model files."
                            )
                            st.caption(f"Technical detail: {e}")
                            st.stop()

                        preds = model.predict(scaled)

                        probs = None
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(scaled)[:, 1]

                        results = raw_df.loc[encoded.index].copy()
                        results["Prediction"] = ["Churn" if p == 1 else "Stay" for p in preds]
                        if probs is not None:
                            results["ChurnProbability"] = [round(float(p), 4) for p in probs]

                        gold_divider()

                        churn_n = int((results["Prediction"] == "Churn").sum())
                        stay_n = int((results["Prediction"] == "Stay").sum())

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Rows Processed", len(results))
                        m2.metric("Predicted Churn", churn_n)
                        m3.metric("Predicted Stay", stay_n)
                        m4.metric("Rows Skipped", dropped)

                        if dropped:
                            st.caption(f"{dropped} row(s) were skipped due to missing or unrecognized values.")

                        st.write("")
                        st.dataframe(results, use_container_width=True)

                        st.download_button(
                            "Download Batch Results (CSV)",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="batch_churn_predictions.csv",
                            mime="text/csv"
                        )


elif page == "History":

    st.markdown('<div class="section-label">Archive</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
    st.write("")

    if os.path.exists(HISTORY_FILE):

        history_df = pd.read_csv(HISTORY_FILE)

        if history_df.empty:
            st.info("No predictions saved yet. Make a prediction on the Prediction page first.")
        else:
            churn_count = int((history_df["Prediction"] == "Churn").sum())
            stay_count = int((history_df["Prediction"] == "Stay").sum())
            total_count = len(history_df)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Records", total_count)
            m2.metric("Predicted Churn", churn_count)
            m3.metric("Predicted Stay", stay_count)

            churn_pct = (churn_count / total_count * 100) if total_count else 0
            stay_pct = 100 - churn_pct

            st.write("")
            st.markdown(f"""
            <div class="ratio-bar">
                <div class="ratio-fill churn" style="width:{churn_pct}%;"></div>
                <div class="ratio-fill stay" style="width:{stay_pct}%;"></div>
            </div>
            <div class="ratio-legend">
                <span>Churn — {churn_pct:.1f}%</span>
                <span>Stay — {stay_pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

            gold_divider()

            f1, f2 = st.columns([1, 1])
            with f1:
                filter_choice = st.selectbox("Filter", ["All", "Churn", "Stay"])
            with f2:
                sort_choice = st.selectbox(
                    "Sort by",
                    ["Most Recent", "Tenure", "Monthly Charges", "Total Charges"]
                )

            filtered = history_df.copy()
            if filter_choice != "All":
                filtered = filtered[filtered["Prediction"] == filter_choice]

            if sort_choice == "Tenure":
                filtered = filtered.sort_values("Tenure", ascending=False)
            elif sort_choice == "Monthly Charges":
                filtered = filtered.sort_values("MonthlyCharges", ascending=False)
            elif sort_choice == "Total Charges":
                filtered = filtered.sort_values("TotalCharges", ascending=False)
            # "Most Recent" keeps the natural (append) order, reversed
            elif sort_choice == "Most Recent":
                filtered = filtered.iloc[::-1]

            st.dataframe(filtered, use_container_width=True)

            csv_bytes = history_df.to_csv(index=False).encode("utf-8")

            dl_col, clear_col = st.columns(2)

            with dl_col:
                st.download_button(
                    "Download Full History (CSV)",
                    data=csv_bytes,
                    file_name="prediction_history.csv",
                    mime="text/csv"
                )

            with clear_col:
                if st.button("Clear History"):
                    os.remove(HISTORY_FILE)
                    st.success("History cleared. Refresh the page to see the change.")
    else:
        st.info("No predictions saved yet. Make a prediction on the Prediction page first.")


elif page == "About":

    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About This App</div>', unsafe_allow_html=True)
    st.write("")

    st.markdown(textwrap.dedent("""\
    This dashboard predicts whether a telecom customer is likely to **churn**
    (cancel their service) based on account and usage details, using a
    machine learning model trained on historical customer data.
    """))

    gold_divider()

    a1, a2 = st.columns(2)

    with a1:
        st.markdown('<div class="section-label">Process</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
        st.markdown(textwrap.dedent("""\
        - Enter a customer's details on the **Prediction** page (or upload a CSV for batch scoring)
        - Inputs are encoded and scaled the same way the training data was scaled
        - A trained classifier predicts churn risk and shows the probability, with a Low / Medium / High risk tier
        - Every single prediction is saved and viewable, filterable, and sortable on the **History** page
        """))

    with a2:
        st.markdown('<div class="section-label">Specification</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">The Model</div>', unsafe_allow_html=True)

        if metrics:
            st.markdown(textwrap.dedent(f"""\
            - Model: **{metrics.get('model_name', 'n/a')}** (auto-selected by cross-validation)
            - Trained on **{metrics.get('dataset_rows', 'n/a'):,}** customers · {len(FEATURE_COLUMNS)} features
            - Test Accuracy: **{metrics.get('test_accuracy', 0) * 100:.2f}%**
            - Precision: **{metrics.get('precision', 0) * 100:.2f}%** · Recall: **{metrics.get('recall', 0) * 100:.2f}%** · F1: **{metrics.get('f1_score', 0) * 100:.2f}%**
            - Last trained: {metrics.get('trained_at', 'n/a')}
            """))
        else:
            st.markdown(textwrap.dedent("""\
            - Trained on the Telco Customer Churn dataset
            - 7,043 customers · 19 features
            - Run `python train.py` to generate real accuracy, precision, recall, and F1 metrics here
            """))

    if feature_importance is not None and not feature_importance.empty:

        gold_divider()

        st.markdown('<div class="section-label">Signal</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Predictive Features</div>', unsafe_allow_html=True)
        st.write("")

        top_features = feature_importance.head(8)
        max_importance = top_features["importance"].max()

        bars_html = ""
        for _, row in top_features.iterrows():
            pct = (row["importance"] / max_importance * 100) if max_importance else 0
            bars_html += f"""
            <div class="imp-row">
                <div class="imp-label">
                    <span>{row['feature']}</span>
                    <span class="val">{row['importance']:.3f}</span>
                </div>
                <div class="imp-track">
                    <div class="imp-fill" style="width:{pct}%;"></div>
                </div>
            </div>
            """

        st.markdown(bars_html, unsafe_allow_html=True)

    gold_divider()

    st.markdown('<div class="section-label">Credit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Developer</div>', unsafe_allow_html=True)
    st.markdown("Palavali Charan Kumar Reddy")
