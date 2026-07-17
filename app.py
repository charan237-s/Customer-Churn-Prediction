import streamlit as st
import pandas as pd
import joblib
import os
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
# LOAD MODEL
# ==========================================================

MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error(
        "Model files not found. Make sure 'models/churn_model.pkl' and "
        "'models/scaler.pkl' exist relative to this script."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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

/* Container-level fallback: any text in the main app area is cream by default */
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
/* Belt-and-braces: cover both old and new Streamlit markdown wrappers */
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
# ENCODING MAPS (needed by Prediction page)
# ==========================================================

maps = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "Phone": {"No": 0, "Yes": 1},
    "Multiple": {"No": 0, "No phone service": 1, "Yes": 2},
    "Internet": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "Security": {"No": 0, "No internet service": 1, "Yes": 2},
    "Backup": {"No": 0, "No internet service": 1, "Yes": 2},
    "Device": {"No": 0, "No internet service": 1, "Yes": 2},
    "Tech": {"No": 0, "No internet service": 1, "Yes": 2},
    "TV": {"No": 0, "No internet service": 1, "Yes": 2},
    "Movies": {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "Paperless": {"No": 0, "Yes": 1},
    "Payment": {
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

HISTORY_FILE = "data/prediction_history.csv"

# ==========================================================
# PAGE ROUTER
# ==========================================================

if page == "Prediction":

    # ------------------------------------------------------
    # DASHBOARD METRICS
    # ------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">7,043</div>
            <div class="metric-text">Customers</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">19</div>
            <div class="metric-text">Features</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">82%</div>
            <div class="metric-text">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">AI</div>
            <div class="metric-text">Model Type</div>
        </div>
        """, unsafe_allow_html=True)

    gold_divider()

    # ------------------------------------------------------
    # INPUT FORM
    # ------------------------------------------------------

    st.markdown('<div class="section-label">Step One</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2)

    with left:

        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])

        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

    with right:

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        tech = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0
        )

    gold_divider()

    # ------------------------------------------------------
    # PREDICT BUTTON
    # ------------------------------------------------------

    st.markdown('<div class="section-label">Step Two</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Run the Prediction</div>', unsafe_allow_html=True)
    st.write("")

    predict = st.button("Predict Customer Churn")

    if predict:

        values = [[
            maps["gender"][gender],
            senior,
            maps["Partner"][partner],
            maps["Dependents"][dependents],
            tenure,
            maps["Phone"][phone],
            maps["Multiple"][multiple],
            maps["Internet"][internet],
            maps["Security"][online_security],
            maps["Backup"][online_backup],
            maps["Device"][device],
            maps["Tech"][tech],
            maps["TV"][tv],
            maps["Movies"][movies],
            maps["Contract"][contract],
            maps["Paperless"][paperless],
            maps["Payment"][payment],
            monthly,
            total
        ]]

        sample = pd.DataFrame(values, columns=FEATURE_COLUMNS)
        sample_scaled = scaler.transform(sample)

        prediction = model.predict(sample_scaled)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(sample_scaled)[0][1]

        gold_divider()

        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Outcome</div>', unsafe_allow_html=True)
        st.write("")

        if prediction == 1:
            st.markdown("""
            <div class="result-banner churn">
                <div class="result-eyebrow">Retention Risk Detected</div>
                <div class="result-headline">This customer is likely to churn</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-banner stay">
                <div class="result-eyebrow">Retention Risk Low</div>
                <div class="result-headline">This customer is likely to stay</div>
            </div>
            """, unsafe_allow_html=True)

        if probability is not None:
            st.write("")
            st.markdown('<div class="section-label">Confidence</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Churn Probability</div>', unsafe_allow_html=True)
            st.progress(float(probability))
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

        history = pd.DataFrame({
            "Gender": [gender],
            "Tenure": [tenure],
            "Internet": [internet],
            "Contract": [contract],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total],
            "Prediction": ["Churn" if prediction == 1 else "Stay"]
        })

        os.makedirs("data", exist_ok=True)

        if os.path.exists(HISTORY_FILE):
            old = pd.read_csv(HISTORY_FILE)
            history = pd.concat([old, history], ignore_index=True)

        history.to_csv(HISTORY_FILE, index=False)

        st.success("Prediction saved to history.")


elif page == "History":

    st.markdown('<div class="section-label">Archive</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
    st.write("")

    if os.path.exists(HISTORY_FILE):

        history_df = pd.read_csv(HISTORY_FILE)

        if history_df.empty:
            st.info("No predictions saved yet. Make a prediction on the Prediction page first.")
        else:
            churn_count = (history_df["Prediction"] == "Churn").sum()
            stay_count = (history_df["Prediction"] == "Stay").sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Records", len(history_df))
            m2.metric("Predicted Churn", churn_count)
            m3.metric("Predicted Stay", stay_count)

            gold_divider()

            st.dataframe(history_df, use_container_width=True)

            csv_bytes = history_df.to_csv(index=False).encode("utf-8")

            dl_col, clear_col = st.columns(2)

            with dl_col:
                st.download_button(
                    "Download History (CSV)",
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
        - Enter a customer's details on the **Prediction** page
        - Inputs are encoded and scaled the same way the training data was scaled
        - A trained classifier predicts churn risk and shows the probability
        - Every prediction is saved and viewable on the **History** page
        """))

    with a2:
        st.markdown('<div class="section-label">Specification</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">The Model</div>', unsafe_allow_html=True)
        st.markdown(textwrap.dedent("""\
        - Trained on the Telco Customer Churn dataset
        - 7,043 customers · 19 features
        - Reported accuracy: ~82%
        """))

    gold_divider()

    st.markdown('<div class="section-label">Credit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Developer</div>', unsafe_allow_html=True)
    st.markdown("PALAVALI CHARAN KUMAR REDDY")