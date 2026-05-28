import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnScope · Customer Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0A0B0F;
    --card: #111318;
    --card2: #161921;
    --border: rgba(255,255,255,0.07);
    --accent: #C8F04B;
    --accent2: #FF6B6B;
    --accent3: #6BFFC8;
    --text: #E8EAF0;
    --muted: #6B7280;
    --radius: 16px;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg); }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0D0E14 !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }

/* HEADER */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #C8F04B 0%, #6BFFC8 60%, #C8F04B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* METRIC CARDS */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
    border-radius: 2px 0 0 2px;
}
.metric-card.danger::before { background: var(--accent2); }
.metric-card.safe::before { background: var(--accent3); }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-val {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--accent);
    letter-spacing: -0.03em;
    line-height: 1;
}
.metric-val.danger { color: var(--accent2); }
.metric-val.safe { color: var(--accent3); }
.metric-change {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* PREDICTION BOX */
.pred-box {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-box.churn {
    background: linear-gradient(135deg, rgba(255,107,107,0.15) 0%, rgba(255,107,107,0.05) 100%);
    border: 1px solid rgba(255,107,107,0.3);
}
.pred-box.stay {
    background: linear-gradient(135deg, rgba(107,255,200,0.15) 0%, rgba(107,255,200,0.05) 100%);
    border: 1px solid rgba(107,255,200,0.3);
}
.pred-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.pred-label { font-size: 0.7rem; font-family: 'DM Mono', monospace; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; }
.pred-result { font-size: 2rem; font-weight: 800; }
.pred-result.churn { color: var(--accent2); }
.pred-result.stay { color: var(--accent3); }
.pred-prob { font-size: 0.8rem; font-family: 'DM Mono', monospace; color: var(--muted); margin-top: 0.3rem; }

/* SECTION HEADERS */
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1.2rem;
    letter-spacing: -0.02em;
}

/* RISK BADGE */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.1em;
}
.risk-low { background: rgba(107,255,200,0.15); color: #6BFFC8; border: 1px solid rgba(107,255,200,0.3); }
.risk-medium { background: rgba(200,240,75,0.15); color: #C8F04B; border: 1px solid rgba(200,240,75,0.3); }
.risk-high { background: rgba(255,165,0,0.15); color: #FFA500; border: 1px solid rgba(255,165,0,0.3); }
.risk-critical { background: rgba(255,107,107,0.15); color: #FF6B6B; border: 1px solid rgba(255,107,107,0.3); }

/* DIVIDER */
.div-line { height: 1px; background: var(--border); margin: 1.5rem 0; }

/* Streamlit overrides */
div[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label { 
    font-family: 'DM Mono', monospace !important; 
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
div[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #d4ff5e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(200,240,75,0.25) !important;
}

/* Sidebar sliders and inputs */
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stRadio > label {
    color: #9CA3AF !important;
}

/* Info box */
.info-box {
    background: rgba(200,240,75,0.06);
    border: 1px solid rgba(200,240,75,0.18);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #9CA3AF;
    line-height: 1.7;
}
.info-box strong { color: var(--accent); }

/* Feature importance row */
.feat-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
}
.feat-name { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--muted); width: 200px; flex-shrink: 0; }
.feat-bar-bg { flex: 1; height: 6px; background: var(--card2); border-radius: 3px; overflow: hidden; }
.feat-bar { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--accent), var(--accent3)); }
.feat-val { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--text); width: 50px; text-align: right; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD DATA & MODEL ──────────────────────────────────────────────────────
FEATURE_COLS = [
    'SeniorCitizen','tenure','MonthlyCharges','TotalCharges',
    'gender_Female','gender_Male',
    'Partner_No','Partner_Yes',
    'Dependents_No','Dependents_Yes',
    'PhoneService_No','PhoneService_Yes',
    'MultipleLines_No','MultipleLines_No phone service','MultipleLines_Yes',
    'InternetService_DSL','InternetService_Fiber optic','InternetService_No',
    'OnlineSecurity_No','OnlineSecurity_No internet service','OnlineSecurity_Yes',
    'OnlineBackup_No','OnlineBackup_No internet service','OnlineBackup_Yes',
    'DeviceProtection_No','DeviceProtection_No internet service','DeviceProtection_Yes',
    'TechSupport_No','TechSupport_No internet service','TechSupport_Yes',
    'StreamingTV_No','StreamingTV_No internet service','StreamingTV_Yes',
    'StreamingMovies_No','StreamingMovies_No internet service','StreamingMovies_Yes',
    'Contract_Month-to-month','Contract_One year','Contract_Two year',
    'PaperlessBilling_No','PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer','PaymentMethod_Credit card',
    'PaymentMethod_Electronic check','PaymentMethod_Mailed check',
]

@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn_prediction_dataset.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

@st.cache_resource
def load_model():
    from sklearn.linear_model import LogisticRegression
    try:
        with open("Logistic.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        data = pd.read_csv("customer_churn_prediction_dataset.csv")
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
        y = (data['Churn'] == 'Yes').astype(int)
        X = pd.get_dummies(data.drop(columns=['customerID', 'Churn']), drop_first=False)
        for col in FEATURE_COLS:
            if col not in X.columns:
                X[col] = 0
        X = X[FEATURE_COLS].fillna(0)
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        clf.fit(X, y)
        clf.feature_names_in_ = np.array(FEATURE_COLS)
        clf.classes_ = np.array([0, 1])
        return clf

df = load_data()
model = load_model()

def build_input(vals: dict) -> pd.DataFrame:
    row = {}
    row['SeniorCitizen'] = vals['SeniorCitizen']
    row['tenure'] = vals['tenure']
    row['MonthlyCharges'] = vals['MonthlyCharges']
    row['TotalCharges'] = vals['TotalCharges']
    cats = {
        'gender': ['Female','Male'],
        'Partner': ['No','Yes'],
        'Dependents': ['No','Yes'],
        'PhoneService': ['No','Yes'],
        'MultipleLines': ['No','No phone service','Yes'],
        'InternetService': ['DSL','Fiber optic','No'],
        'OnlineSecurity': ['No','No internet service','Yes'],
        'OnlineBackup': ['No','No internet service','Yes'],
        'DeviceProtection': ['No','No internet service','Yes'],
        'TechSupport': ['No','No internet service','Yes'],
        'StreamingTV': ['No','No internet service','Yes'],
        'StreamingMovies': ['No','No internet service','Yes'],
        'Contract': ['Month-to-month','One year','Two year'],
        'PaperlessBilling': ['No','Yes'],
        'PaymentMethod': ['Bank transfer','Credit card','Electronic check','Mailed check'],
    }
    for col, options in cats.items():
        chosen = vals[col]
        for opt in options:
            key = f"{col}_{opt}"
            row[key] = 1 if chosen == opt else 0
    return pd.DataFrame([row])[FEATURE_COLS]


# ─── SIDEBAR INPUTS ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='padding: 1rem 0 0.5rem;'>
        <div style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.2em;color:#6B7280;text-transform:uppercase;margin-bottom:0.3rem;'>ChurnScope</div>
        <div style='font-size:1.1rem;font-weight:700;color:#E8EAF0;'>Customer Profile</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-tag'>● Demographics</div>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-tag'>● Account</div>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", 1, 72, 24)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-tag'>● Services</div>", unsafe_allow_html=True)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-tag'>● Billing</div>", unsafe_allow_html=True)
    monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total = st.slider("Total Charges ($)", 0.0, 9000.0, monthly * tenure, step=50.0)

vals = dict(
    gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents,
    tenure=tenure, PhoneService=phone, MultipleLines=multi, InternetService=internet,
    OnlineSecurity=security, OnlineBackup=backup, DeviceProtection=device,
    TechSupport=tech, StreamingTV=tv, StreamingMovies=movies, Contract=contract,
    PaperlessBilling=paperless, PaymentMethod=payment,
    MonthlyCharges=monthly, TotalCharges=total
)
X_input = build_input(vals)
proba = model.predict_proba(X_input)[0]
churn_prob = proba[1]
stay_prob = proba[0]
predicted_churn = churn_prob >= 0.5

if churn_prob < 0.25: risk_level, risk_class = "LOW RISK", "risk-low"
elif churn_prob < 0.50: risk_level, risk_class = "MEDIUM RISK", "risk-medium"
elif churn_prob < 0.75: risk_level, risk_class = "HIGH RISK", "risk-high"
else: risk_level, risk_class = "CRITICAL RISK", "risk-critical"


# ─── MAIN LAYOUT ────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>ChurnScope</div>
<div class='hero-sub'>// Customer Churn Intelligence Platform · Logistic Regression Engine</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮  Prediction", "📊  Dataset Analytics", "⚙️  Model Insights"])


# ══════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════
with tab1:
    col_pred, col_gauge, col_meta = st.columns([1, 1.4, 1])

    with col_pred:
        box_cls = "churn" if predicted_churn else "stay"
        emoji = "⚠️" if predicted_churn else "✅"
        result_text = "WILL CHURN" if predicted_churn else "WILL STAY"
        prob_text = f"Churn probability: {churn_prob:.1%}"
        st.markdown(f"""
        <div class='pred-box {box_cls}'>
            <div class='pred-emoji'>{emoji}</div>
            <div class='pred-label'>Prediction</div>
            <div class='pred-result {box_cls}'>{result_text}</div>
            <div class='pred-prob'>{prob_text}</div>
            <br>
            <span class='{risk_class} risk-badge'>{risk_level}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            number={'suffix': '%', 'font': {'size': 36, 'color': '#FF6B6B' if churn_prob >= 0.5 else '#6BFFC8', 'family': 'Syne'}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'color': '#6B7280', 'size': 10, 'family': 'DM Mono'}, 'tickcolor': '#6B7280'},
                'bar': {'color': '#FF6B6B' if churn_prob >= 0.5 else '#6BFFC8', 'thickness': 0.22},
                'bgcolor': '#161921',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(107,255,200,0.12)'},
                    {'range': [25, 50], 'color': 'rgba(200,240,75,0.12)'},
                    {'range': [50, 75], 'color': 'rgba(255,165,0,0.12)'},
                    {'range': [75, 100], 'color': 'rgba(255,107,107,0.12)'},
                ],
                'threshold': {'line': {'color': '#C8F04B', 'width': 2}, 'thickness': 0.75, 'value': 50}
            },
            title={'text': "CHURN RISK SCORE", 'font': {'color': '#6B7280', 'size': 11, 'family': 'DM Mono'}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=260, margin=dict(t=40, b=0, l=20, r=20),
            font_color='#E8EAF0'
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    with col_meta:
        st.markdown(f"""
        <div class='metric-card {"danger" if predicted_churn else "safe"}' style='margin-bottom:0.8rem;'>
            <div class='metric-label'>Churn Prob</div>
            <div class='metric-val {"danger" if predicted_churn else "safe"}'>{churn_prob:.1%}</div>
            <div class='metric-change'>Threshold @ 50%</div>
        </div>
        <div class='metric-card safe'>
            <div class='metric-label'>Retention Prob</div>
            <div class='metric-val safe'>{stay_prob:.1%}</div>
            <div class='metric-change'>Likelihood to stay</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)

    # ── PROBABILITY BAR ─────────────────────────────────────────
    st.markdown("<div class='section-tag'>● Probability Distribution</div>", unsafe_allow_html=True)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[stay_prob], y=[""], orientation='h', name='Will Stay',
        marker_color='#6BFFC8', text=[f"STAY  {stay_prob:.1%}"],
        textposition='inside', insidetextanchor='start',
        textfont=dict(color='#000', size=12, family='Syne'),
        hoverinfo='skip'
    ))
    fig_bar.add_trace(go.Bar(
        x=[churn_prob], y=[""], orientation='h', name='Will Churn',
        marker_color='#FF6B6B', text=[f"{churn_prob:.1%}  CHURN"],
        textposition='inside', insidetextanchor='end',
        textfont=dict(color='#fff', size=12, family='Syne'),
        hoverinfo='skip'
    ))
    fig_bar.update_layout(
        barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=70, margin=dict(t=0, b=0, l=0, r=0), showlegend=False,
        xaxis=dict(visible=False, range=[0,1]), yaxis=dict(visible=False),
        bargap=0
    )
    fig_bar.update_traces(marker_line_width=0)
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    # ── RISK FACTORS ─────────────────────────────────────────────
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-tag'>● Profile Risk Signals</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Key Risk Factors</div>", unsafe_allow_html=True)

    signals = []
    if contract == "Month-to-month": signals.append(("📋 Month-to-month contract", "High churn indicator", "danger"))
    elif contract == "Two year": signals.append(("📋 Two-year contract", "Strong retention signal", "safe"))
    if internet == "Fiber optic": signals.append(("🌐 Fiber optic internet", "Associated with higher churn", "danger"))
    if payment == "Electronic check": signals.append(("💳 Electronic check payment", "Linked to higher churn rates", "danger"))
    if tenure < 12: signals.append((f"⏱ Short tenure ({tenure}mo)", "New customers churn more often", "danger"))
    elif tenure > 48: signals.append((f"⏱ Long tenure ({tenure}mo)", "Loyal long-term customer", "safe"))
    if security == "No" and internet != "No": signals.append(("🔒 No online security", "Moderate churn risk factor", "warn"))
    if tech == "Yes": signals.append(("🛠 Tech support enabled", "Positive retention factor", "safe"))
    if not signals: signals.append(("📊 Balanced profile", "No extreme risk signals detected", "neutral"))

    cols_sig = st.columns(min(len(signals), 3))
    for i, (icon_label, desc, stype) in enumerate(signals[:3]):
        color = "#FF6B6B" if stype=="danger" else "#6BFFC8" if stype=="safe" else "#C8F04B"
        border = f"rgba({','.join(str(int(color[i:i+2],16)) for i in (1,3,5))},0.25)"
        with cols_sig[i % 3]:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid {border};border-radius:12px;padding:1rem;margin-bottom:0.5rem;'>
                <div style='font-size:0.85rem;font-weight:600;color:{color};margin-bottom:0.3rem;'>{icon_label}</div>
                <div style='font-family:DM Mono,monospace;font-size:0.7rem;color:#6B7280;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── RETENTION ACTIONS ────────────────────────────────────────
    if predicted_churn:
        st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-tag'>● Recommended Actions</div>", unsafe_allow_html=True)
        actions = []
        if contract == "Month-to-month": actions.append("🎯 Offer annual contract discount (15–20%)")
        if internet == "Fiber optic": actions.append("🌐 Investigate service quality issues")
        if security == "No" and internet != "No": actions.append("🔒 Bundle online security at reduced rate")
        if tenure < 12: actions.append("🎁 Early loyalty bonus / onboarding support")
        actions.append("📞 Proactive outreach from retention team")
        st.markdown(f"""
        <div class='info-box'>
            {'<br>'.join(f'<strong>→</strong> {a}' for a in actions)}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 2 — DATASET ANALYTICS
# ══════════════════════════════════════════
with tab2:
    # ── KPI ROW ─────────────────────────────────────────────────
    total_c = len(df)
    churned = (df['Churn'] == 'Yes').sum()
    churn_rate = churned / total_c
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['MonthlyCharges'].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Total Customers</div>
            <div class='metric-val'>{total_c:,}</div>
            <div class='metric-change'>Dataset size</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='metric-card danger'>
            <div class='metric-label'>Churned</div>
            <div class='metric-val danger'>{churned}</div>
            <div class='metric-change'>{churn_rate:.1%} of total</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='metric-card safe'>
            <div class='metric-label'>Avg Tenure</div>
            <div class='metric-val safe'>{avg_tenure:.0f}<span style='font-size:1rem'>mo</span></div>
            <div class='metric-change'>Months with service</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Avg Monthly</div>
            <div class='metric-val'>${avg_monthly:.0f}</div>
            <div class='metric-change'>Monthly charges</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)

    # ── CHARTS ROW 1 ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='section-tag'>● Churn Distribution</div>", unsafe_allow_html=True)
        churn_counts = df['Churn'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=['Stay', 'Churn'], values=[churn_counts.get('No',0), churn_counts.get('Yes',0)],
            hole=0.65, marker_colors=['#6BFFC8', '#FF6B6B'],
            textinfo='none', hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
        ))
        fig_pie.add_annotation(text=f"{churn_rate:.0%}", x=0.5, y=0.55, font=dict(size=24, color='#FF6B6B', family='Syne'), showarrow=False)
        fig_pie.add_annotation(text="churn rate", x=0.5, y=0.42, font=dict(size=10, color='#6B7280', family='DM Mono'), showarrow=False)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              height=230, margin=dict(t=10,b=10,l=0,r=0), showlegend=True,
                              legend=dict(font=dict(color='#9CA3AF', size=11, family='DM Mono'),
                                          orientation='h', x=0.2, y=-0.05))
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    with c2:
        st.markdown("<div class='section-tag'>● Contract vs Churn</div>", unsafe_allow_html=True)
        ct = df.groupby(['Contract','Churn']).size().reset_index(name='count')
        fig_ct = go.Figure()
        for churn_val, color in [('No','#6BFFC8'), ('Yes','#FF6B6B')]:
            d = ct[ct['Churn']==churn_val]
            fig_ct.add_trace(go.Bar(x=d['Contract'], y=d['count'], name=churn_val,
                                    marker_color=color, marker_line_width=0,
                                    hovertemplate='%{x}: %{y}<extra></extra>'))
        fig_ct.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=230, margin=dict(t=10,b=30,l=30,r=10),
            xaxis=dict(tickfont=dict(color='#6B7280', size=9, family='DM Mono'), gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(tickfont=dict(color='#6B7280', size=9, family='DM Mono'), gridcolor='rgba(255,255,255,0.04)'),
            legend=dict(font=dict(color='#9CA3AF', size=10, family='DM Mono'), orientation='h', x=0, y=-0.15),
            bargap=0.25
        )
        st.plotly_chart(fig_ct, use_container_width=True, config={'displayModeBar': False})

    with c3:
        st.markdown("<div class='section-tag'>● Internet Service vs Churn</div>", unsafe_allow_html=True)
        it = df.groupby(['InternetService','Churn']).size().reset_index(name='count')
        fig_it = go.Figure()
        for churn_val, color in [('No','#6BFFC8'), ('Yes','#FF6B6B')]:
            d = it[it['Churn']==churn_val]
            fig_it.add_trace(go.Bar(x=d['InternetService'], y=d['count'], name=churn_val,
                                    marker_color=color, marker_line_width=0,
                                    hovertemplate='%{x}: %{y}<extra></extra>'))
        fig_it.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=230, margin=dict(t=10,b=30,l=30,r=10),
            xaxis=dict(tickfont=dict(color='#6B7280', size=9, family='DM Mono'), gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(tickfont=dict(color='#6B7280', size=9, family='DM Mono'), gridcolor='rgba(255,255,255,0.04)'),
            showlegend=False, bargap=0.25
        )
        st.plotly_chart(fig_it, use_container_width=True, config={'displayModeBar': False})

    # ── CHARTS ROW 2 ────────────────────────────────────────────
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    c4, c5 = st.columns([1.5, 1])

    with c4:
        st.markdown("<div class='section-tag'>● Tenure Distribution by Churn</div>", unsafe_allow_html=True)
        fig_ten = go.Figure()
        for churn_val, color, name in [('No','#6BFFC8','Stay'), ('Yes','#FF6B6B','Churn')]:
            d = df[df['Churn']==churn_val]['tenure']
            fig_ten.add_trace(go.Histogram(x=d, name=name, nbinsx=24,
                                           marker_color=color, opacity=0.7,
                                           hovertemplate='Tenure %{x}mo: %{y}<extra></extra>'))
        fig_ten.update_layout(
            barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=260, margin=dict(t=10,b=30,l=40,r=10),
            xaxis=dict(title=dict(text='Months', font=dict(color='#6B7280', size=10, family='DM Mono')),
                       tickfont=dict(color='#6B7280', size=9), gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(tickfont=dict(color='#6B7280', size=9), gridcolor='rgba(255,255,255,0.04)'),
            legend=dict(font=dict(color='#9CA3AF', size=10, family='DM Mono'), orientation='h', x=0, y=1.1)
        )
        st.plotly_chart(fig_ten, use_container_width=True, config={'displayModeBar': False})

    with c5:
        st.markdown("<div class='section-tag'>● Monthly Charges by Churn</div>", unsafe_allow_html=True)
        fig_box = go.Figure()
        for churn_val, color, name in [('No','#6BFFC8','Stay'), ('Yes','#FF6B6B','Churn')]:
            d = df[df['Churn']==churn_val]['MonthlyCharges']
            fig_box.add_trace(go.Box(y=d, name=name, marker_color=color,
                                     line_color=color, fillcolor=f'rgba({",".join(str(int(color[i:i+2],16)) for i in (1,3,5))},0.15)',
                                     boxpoints='outliers', jitter=0.3,
                                     hovertemplate='$%{y:.2f}<extra></extra>'))
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=260, margin=dict(t=10,b=10,l=40,r=10),
            yaxis=dict(title=dict(text='$/month', font=dict(color='#6B7280', size=10, family='DM Mono')),
                       tickfont=dict(color='#6B7280', size=9), gridcolor='rgba(255,255,255,0.04)'),
            xaxis=dict(tickfont=dict(color='#9CA3AF', size=11, family='Syne')),
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})

    # ── PAYMENT METHOD ───────────────────────────────────────────
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-tag'>● Payment Method Churn Rate</div>", unsafe_allow_html=True)
    pay_churn = df.groupby('PaymentMethod').apply(lambda x: (x['Churn']=='Yes').mean()).reset_index()
    pay_churn.columns = ['Method', 'ChurnRate']
    pay_churn = pay_churn.sort_values('ChurnRate', ascending=True)
    fig_pay = go.Figure(go.Bar(
        x=pay_churn['ChurnRate'], y=pay_churn['Method'], orientation='h',
        marker_color=['#6BFFC8' if r < 0.3 else '#C8F04B' if r < 0.45 else '#FF6B6B' for r in pay_churn['ChurnRate']],
        marker_line_width=0, text=[f"{r:.0%}" for r in pay_churn['ChurnRate']],
        textposition='outside', textfont=dict(color='#9CA3AF', size=10, family='DM Mono'),
        hovertemplate='%{y}: %{x:.1%}<extra></extra>'
    ))
    fig_pay.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=220, margin=dict(t=10,b=10,l=10,r=60),
        xaxis=dict(tickformat='.0%', tickfont=dict(color='#6B7280', size=9), gridcolor='rgba(255,255,255,0.04)', range=[0,0.65]),
        yaxis=dict(tickfont=dict(color='#9CA3AF', size=10, family='DM Mono'))
    )
    st.plotly_chart(fig_pay, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-tag'>● Model Architecture</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Logistic Regression · Coefficient Analysis</div>", unsafe_allow_html=True)

    # Feature importances from model coefficients
    coef = model.coef_[0]
    feat_imp = pd.DataFrame({'feature': FEATURE_COLS, 'coef': coef})
    feat_imp['abs_coef'] = feat_imp['coef'].abs()
    feat_imp = feat_imp.sort_values('abs_coef', ascending=False).head(20)

    # Top positive (churn drivers) & negative (retention drivers)
    churn_drivers = feat_imp[feat_imp['coef'] > 0].head(8)
    retain_drivers = feat_imp[feat_imp['coef'] < 0].head(8)

    cd1, cd2 = st.columns(2)
    with cd1:
        st.markdown("<div class='section-tag'>🔴 Churn Drivers (Top 8)</div>", unsafe_allow_html=True)
        fig_cd = go.Figure(go.Bar(
            x=churn_drivers['coef'].values[::-1],
            y=churn_drivers['feature'].values[::-1],
            orientation='h', marker_color='#FF6B6B', marker_line_width=0,
            hovertemplate='%{y}: %{x:.3f}<extra></extra>'
        ))
        fig_cd.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(t=10,b=10,l=10,r=10),
            xaxis=dict(tickfont=dict(color='#6B7280', size=8, family='DM Mono'), gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(tickfont=dict(color='#9CA3AF', size=9, family='DM Mono'))
        )
        st.plotly_chart(fig_cd, use_container_width=True, config={'displayModeBar': False})

    with cd2:
        st.markdown("<div class='section-tag'>🟢 Retention Drivers (Top 8)</div>", unsafe_allow_html=True)
        fig_rd = go.Figure(go.Bar(
            x=retain_drivers['coef'].abs().values[::-1],
            y=retain_drivers['feature'].values[::-1],
            orientation='h', marker_color='#6BFFC8', marker_line_width=0,
            hovertemplate='%{y}: -%{x:.3f}<extra></extra>'
        ))
        fig_rd.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(t=10,b=10,l=10,r=10),
            xaxis=dict(tickfont=dict(color='#6B7280', size=8, family='DM Mono'), gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(tickfont=dict(color='#9CA3AF', size=9, family='DM Mono'))
        )
        st.plotly_chart(fig_rd, use_container_width=True, config={'displayModeBar': False})

    # Model stats
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-tag'>● Model Performance on Dataset</div>", unsafe_allow_html=True)

    # Build full predictions on dataset
    df_proc = pd.get_dummies(df.drop(columns=['customerID','Churn','TotalCharges']), drop_first=False)
    X_full = pd.DataFrame(columns=FEATURE_COLS)
    for col in FEATURE_COLS:
        if col in df_proc.columns:
            X_full[col] = df_proc[col]
        elif col == 'TotalCharges':
            X_full[col] = df['TotalCharges']
        else:
            X_full[col] = 0
    X_full = X_full.fillna(0)

    try:
        y_pred = model.predict(X_full)
        y_true = (df['Churn'] == 'Yes').astype(int)
        y_pred_bin = (y_pred == model.classes_[1]).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_true, y_pred_bin)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        rec = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)

        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, note in [(m1,"Accuracy",acc,"Overall"), (m2,"Precision",prec,"Churn class"), (m3,"Recall",rec,"Churn class"), (m4,"F1 Score",f1,"Harmonic mean")]:
            with col:
                color = "#6BFFC8" if val >= 0.7 else "#C8F04B" if val >= 0.55 else "#FF6B6B"
                st.markdown(f"""<div class='metric-card' style='border-left-color:{color}'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-val' style='color:{color}'>{val:.1%}</div>
                    <div class='metric-change'>{note}</div></div>""", unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Could not compute full metrics on dataset: {e}")

    # Model info card
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    params = model.get_params()
    st.markdown(f"""
    <div class='info-box'>
        <strong>Model:</strong> Logistic Regression &nbsp;·&nbsp; <strong>Solver:</strong> {params.get('solver','lbfgs')} &nbsp;·&nbsp;
        <strong>Penalty:</strong> {params.get('penalty','l2')} &nbsp;·&nbsp; <strong>C:</strong> {params.get('C',1.0)} &nbsp;·&nbsp;
        <strong>Max Iterations:</strong> {params.get('max_iter',100)} &nbsp;·&nbsp;
        <strong>Features:</strong> {len(FEATURE_COLS)} &nbsp;·&nbsp; <strong>Version:</strong> sklearn 1.6.1
    </div>
    """, unsafe_allow_html=True)

    # Coefficient scatter
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-tag'>● All Feature Coefficients</div>", unsafe_allow_html=True)
    all_coef = pd.DataFrame({'feature': FEATURE_COLS, 'coef': coef}).sort_values('coef')
    colors = ['#FF6B6B' if c > 0 else '#6BFFC8' for c in all_coef['coef']]
    fig_all = go.Figure(go.Bar(
        x=all_coef['feature'], y=all_coef['coef'],
        marker_color=colors, marker_line_width=0,
        hovertemplate='%{x}: %{y:.4f}<extra></extra>'
    ))
    fig_all.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=320, margin=dict(t=10,b=80,l=50,r=10),
        xaxis=dict(tickfont=dict(color='#6B7280', size=7, family='DM Mono'), gridcolor='rgba(255,255,255,0.04)', tickangle=45),
        yaxis=dict(title=dict(text='Coefficient', font=dict(color='#6B7280', size=10)),
                   tickfont=dict(color='#6B7280', size=9), gridcolor='rgba(255,255,255,0.04)',
                   zerolinecolor='rgba(255,255,255,0.15)'),
    )
    st.plotly_chart(fig_all, use_container_width=True, config={'displayModeBar': False})