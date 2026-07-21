# pyrefly: ignore [missing-import]
import streamlit as st
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
import joblib
import os
import json
import hashlib
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import RobustScaler
from src.models import CardShieldLSTM
from src.evaluate import assign_card_risk_tier

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="CardShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
.stApp {background: linear-gradient(135deg,#0a0a0f 0%,#0d1117 50%,#0a0f1a 100%);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0d1117,#161b22);border-right:1px solid rgba(0,240,255,0.1);}
[data-testid="stMetricValue"] {font-family:'Orbitron',monospace;color:#00f0ff;font-size:1.4rem;}
[data-testid="stMetricLabel"] {color:#8b949e;}
h1,h2,h3 {font-family:'Orbitron',monospace !important;letter-spacing:1px;}
.hero-title {
    background:linear-gradient(135deg,#00f0ff,#ff007f,#00ff88);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-family:'Orbitron',monospace;font-size:2.8rem;font-weight:900;text-align:center;
    animation:glow 3s ease-in-out infinite alternate;
}
@keyframes glow {from{filter:brightness(1)} to{filter:brightness(1.3)}}
.hero-sub {text-align:center;color:#8b949e;font-size:1.1rem;font-family:'Inter',sans-serif;margin-bottom:2rem;}
.glass-card {
    background:rgba(22,27,34,0.7);backdrop-filter:blur(12px);
    border:1px solid rgba(0,240,255,0.12);border-radius:16px;padding:1.5rem;margin:0.5rem 0;
}
.risk-safe {border-left:4px solid #00ff88;} .risk-suspicious {border-left:4px solid #ffcc00;} .risk-critical {border-left:4px solid #ff007f;}
.stat-number {font-family:'Orbitron',monospace;font-size:2rem;font-weight:700;}
.cyan {color:#00f0ff;} .magenta {color:#ff007f;} .green {color:#00ff88;} .yellow {color:#ffcc00;}
.tag {display:inline-block;padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:600;font-family:'Inter',sans-serif;}
.tag-safe {background:rgba(0,255,136,0.15);color:#00ff88;border:1px solid rgba(0,255,136,0.3);}
.tag-suspicious {background:rgba(255,204,0,0.15);color:#ffcc00;border:1px solid rgba(255,204,0,0.3);}
.tag-critical {background:rgba(255,0,127,0.15);color:#ff007f;border:1px solid rgba(255,0,127,0.3);}
div.stButton > button {
    background:linear-gradient(135deg,#00f0ff,#0080ff);color:#0a0a0f;font-weight:700;
    border:none;border-radius:12px;padding:0.6rem 2rem;font-family:'Inter',sans-serif;
    transition:all 0.3s;
}
div.stButton > button:hover {transform:translateY(-2px);box-shadow:0 8px 25px rgba(0,240,255,0.3);}
.login-wrapper {display:flex;min-height:100vh;margin:-6rem -4rem;}
.login-left {
    flex:1;background:linear-gradient(135deg,#070d1a 0%,#0a1628 50%,#0d1f3c 100%);
    display:flex;flex-direction:column;justify-content:center;align-items:center;padding:3rem;
    position:relative;overflow:hidden;
}
.login-left::before {
    content:'';position:absolute;top:50%;left:50%;width:500px;height:500px;
    background:radial-gradient(circle,rgba(0,240,255,0.08) 0%,transparent 70%);
    transform:translate(-50%,-50%);border-radius:50%;
}
.login-logo {font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:700;color:#00f0ff;margin-bottom:2rem;letter-spacing:2px;}
.login-logo span {color:#e0e0e0;}
.login-headline {
    font-family:'Inter',sans-serif;font-size:2.2rem;font-weight:800;color:#e0e0e0;
    line-height:1.2;margin-bottom:1rem;text-align:center;
}
.login-headline em {font-style:normal;color:#00f0ff;}
.login-tagline {color:#6c7086;font-size:0.95rem;text-align:center;max-width:380px;line-height:1.6;margin-bottom:2rem;font-family:'Inter',sans-serif;}
.login-features {display:flex;gap:2rem;margin-top:2rem;}
.login-feat-item {text-align:center;}
.login-feat-icon {
    width:48px;height:48px;border-radius:14px;margin:0 auto 0.6rem;
    display:flex;align-items:center;justify-content:center;font-size:1.3rem;
    background:rgba(0,240,255,0.08);border:1px solid rgba(0,240,255,0.15);
}
.login-feat-title {font-family:'Inter',sans-serif;font-weight:600;font-size:0.8rem;color:#e0e0e0;}
.login-feat-desc {font-family:'Inter',sans-serif;font-size:0.65rem;color:#6c7086;max-width:120px;margin:0 auto;}
.login-right-card {
    background:rgba(13,17,23,0.95);backdrop-filter:blur(20px);
    border:1px solid rgba(0,240,255,0.08);border-radius:20px;padding:2rem;
    box-shadow:0 8px 40px rgba(0,0,0,0.4);
}
.login-form-title {font-family:'Inter',sans-serif;font-size:1.6rem;font-weight:700;color:#e0e0e0;margin-bottom:0.2rem;}
.login-form-sub {color:#6c7086;font-size:0.85rem;margin-bottom:1.5rem;font-family:'Inter',sans-serif;}
.login-form-sub b {color:#00f0ff;}

.login-switch {text-align:center;color:#6c7086;font-size:0.85rem;margin-top:1.2rem;font-family:'Inter',sans-serif;}
.login-switch a {color:#00f0ff;text-decoration:none;font-weight:600;cursor:pointer;}
.login-badge-row {display:flex;justify-content:center;gap:2rem;margin-top:1.5rem;}
.login-badge {display:flex;align-items:center;gap:6px;color:#4a5068;font-size:0.75rem;font-family:'Inter',sans-serif;}
.login-error {background:rgba(255,0,127,0.1);border:1px solid rgba(255,0,127,0.3);color:#ff007f;padding:0.7rem 1rem;border-radius:10px;text-align:center;font-size:0.85rem;margin-top:0.5rem;}
.login-success {background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3);color:#00ff88;padding:0.7rem 1rem;border-radius:10px;text-align:center;font-size:0.85rem;margin-top:0.5rem;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# AUTH SYSTEM
# ──────────────────────────────────────────────
USERS_FILE = "users.json"
HASH_ITERATIONS = 260000  # OWASP-recommended for PBKDF2-SHA256

def hash_pw(pw, salt=None):
    """Hash password with PBKDF2-HMAC-SHA256 and a random 32-byte salt."""
    if salt is None:
        salt = os.urandom(32)
    dk = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, HASH_ITERATIONS)
    return salt.hex() + ":" + dk.hex()

def verify_pw(pw, stored_hash):
    """Verify password. Supports legacy SHA-256 hashes for transparent migration."""
    if ":" in stored_hash:
        salt_hex, hash_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, HASH_ITERATIONS)
        return dk.hex() == hash_hex
    else:
        return hashlib.sha256(pw.encode()).hexdigest() == stored_hash

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    # Default users
    defaults = {
        "admin@cardshield.ai": {"password": hash_pw("admin123"), "name": "Admin"},
        "danish@cardshield.ai": {"password": hash_pw("cardshield"), "name": "Danish"},
        "demo@demo.com": {"password": hash_pw("demo"), "name": "Demo User"},
    }
    save_users(defaults)
    return defaults

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def get_hero_base64():
    img_path = "assets/hero_shield.png"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# ──────────────────────────────────────────────
# LOGIN / SIGNUP PAGE
# ──────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.user_display = ""
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"

if not st.session_state.authenticated:
    users = load_users()
    hero_b64 = get_hero_base64()
    hero_html = f'<img src="data:image/png;base64,{hero_b64}" style="max-width:340px;border-radius:16px;margin-bottom:1rem;filter:drop-shadow(0 0 30px rgba(0,240,255,0.15));">' if hero_b64 else ""

    left_col, right_col = st.columns([1, 1], gap="large")

    # ── LEFT: Hero Section ──
    with left_col:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:80vh;padding:1rem;">
            <div class="login-logo">🛡️ AI CREDIT <span>SHIELD AI</span></div>
            <div class="login-headline">Smarter <em>Credit</em> Decisions.<br>Stronger <em>Financial</em> Future.</div>
            <div class="login-tagline">AI-powered credit risk analysis and fraud detection to protect lenders and empower smarter lending.</div>
            {hero_html}
            <div class="login-features">
                <div class="login-feat-item">
                    <div class="login-feat-icon">🧠</div>
                    <div class="login-feat-title">AI Powered</div>
                    <div class="login-feat-desc">Advanced ML for accurate predictions</div>
                </div>
                <div class="login-feat-item">
                    <div class="login-feat-icon">🛡️</div>
                    <div class="login-feat-title">Risk Protection</div>
                    <div class="login-feat-desc">Detect fraud and minimize credit risk</div>
                </div>
                <div class="login-feat-item">
                    <div class="login-feat-icon">📊</div>
                    <div class="login-feat-title">Better Decisions</div>
                    <div class="login-feat-desc">Data-driven insights for smarter lending</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── RIGHT: Auth Form ──
    with right_col:
        st.markdown('<div style="min-height:10vh;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="login-right-card">', unsafe_allow_html=True)

        if st.session_state.auth_page == "login":
            # ── SIGN IN ──
            st.markdown('<div class="login-form-title">Welcome Back</div>', unsafe_allow_html=True)
            st.markdown('<div class="login-form-sub">Sign in to continue to <b>CardShield AI</b></div>', unsafe_allow_html=True)

            with st.form("login_form"):
                email = st.text_input("📧 Email Address", placeholder="Enter your email")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
                rem_col, forgot_col = st.columns([1, 1])
                with rem_col:
                    remember = st.checkbox("Remember me", value=True)
                login_btn = st.form_submit_button("Sign In  →", use_container_width=True)

            if login_btn:
                if email in users and verify_pw(password, users[email]["password"]):
                    # Silently migrate legacy SHA-256 hash to salted PBKDF2
                    if ":" not in users[email]["password"]:
                        users[email]["password"] = hash_pw(password)
                        save_users(users)
                    st.session_state.authenticated = True
                    st.session_state.username = email
                    st.session_state.user_display = users[email]["name"]
                    st.rerun()
                else:
                    st.markdown('<div class="login-error">⚠️ Invalid email or password. Access denied.</div>', unsafe_allow_html=True)



            # Switch to Sign Up
            switch_col1, switch_col2 = st.columns([3, 1])
            with switch_col1:
                st.markdown('<div class="login-switch" style="text-align:left;margin-top:1rem;">Don\'t have an account?</div>', unsafe_allow_html=True)
            with switch_col2:
                if st.button("Sign up", key="go_signup"):
                    st.session_state.auth_page = "signup"
                    st.rerun()

        else:
            # ── SIGN UP ──
            st.markdown('<div class="login-form-title">Create Account</div>', unsafe_allow_html=True)
            st.markdown('<div class="login-form-sub">Join <b>CardShield AI</b> fraud detection platform</div>', unsafe_allow_html=True)

            with st.form("signup_form"):
                new_name = st.text_input("👤 Full Name", placeholder="Enter your full name")
                new_email = st.text_input("📧 Email Address", placeholder="Enter your email")
                new_pass = st.text_input("🔒 Password", type="password", placeholder="Create a password")
                confirm_pass = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
                agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")
                signup_btn = st.form_submit_button("Create Account  →", use_container_width=True)

            if signup_btn:
                if not new_name or not new_email or not new_pass:
                    st.markdown('<div class="login-error">⚠️ All fields are required.</div>', unsafe_allow_html=True)
                elif new_pass != confirm_pass:
                    st.markdown('<div class="login-error">⚠️ Passwords do not match.</div>', unsafe_allow_html=True)
                elif len(new_pass) < 4:
                    st.markdown('<div class="login-error">⚠️ Password must be at least 4 characters.</div>', unsafe_allow_html=True)
                elif new_email in users:
                    st.markdown('<div class="login-error">⚠️ Email already registered. Please sign in.</div>', unsafe_allow_html=True)
                elif not agree:
                    st.markdown('<div class="login-error">⚠️ Please accept the terms to continue.</div>', unsafe_allow_html=True)
                else:
                    users[new_email] = {"password": hash_pw(new_pass), "name": new_name}
                    save_users(users)
                    st.markdown('<div class="login-success">✅ Account created! You can now sign in.</div>', unsafe_allow_html=True)
                    st.session_state.auth_page = "login"
                    st.rerun()



            switch_col1, switch_col2 = st.columns([3, 1])
            with switch_col1:
                st.markdown('<div class="login-switch" style="text-align:left;margin-top:1rem;">Already have an account?</div>', unsafe_allow_html=True)
            with switch_col2:
                if st.button("Sign in", key="go_login"):
                    st.session_state.auth_page = "login"
                    st.rerun()

        # Footer badges
        st.markdown("""
        <div class="login-badge-row">
            <div class="login-badge">🔒 Secure</div>
            <div class="login-badge">🛡️ Reliable</div>
            <div class="login-badge">🧠 Intelligent</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(18,18,26,0.6)",
    font=dict(family="Inter", color="#e0e0e0"),
    margin=dict(l=40, r=40, t=50, b=40),
)

# ──────────────────────────────────────────────
# CACHED LOADERS
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("saved_models/card_rf.pkl"):
        models["Random Forest"] = joblib.load("saved_models/card_rf.pkl")
    if os.path.exists("saved_models/card_xgboost.pkl"):
        models["XGBoost"] = joblib.load("saved_models/card_xgboost.pkl")
    if os.path.exists("saved_models/card_lstm_model.pth"):
        lstm = CardShieldLSTM(input_dim=30, hidden_dim=32)
        lstm.load_state_dict(torch.load("saved_models/card_lstm_model.pth", map_location="cpu"))
        lstm.eval()
        models["LSTM"] = lstm
    return models

@st.cache_resource
def load_scalers():
    a_path, t_path = "saved_models/amount_scaler.pkl", "saved_models/time_scaler.pkl"
    if os.path.exists(a_path) and os.path.exists(t_path):
        return joblib.load(a_path), joblib.load(t_path)
    if os.path.exists("data/creditcard.csv"):
        df = pd.read_csv("data/creditcard.csv")
        a_sc, t_sc = RobustScaler(), RobustScaler()
        a_sc.fit(df[["Amount"]]); t_sc.fit(df[["Time"]])
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(a_sc, a_path); joblib.dump(t_sc, t_path)
        return a_sc, t_sc
    return None, None

# ──────────────────────────────────────────────
# PREDICTION HELPERS
# ──────────────────────────────────────────────
def predict_single(models, features):
    """features: np array shape (30,)"""
    results = {}
    for name, mdl in models.items():
        if name == "LSTM":
            with torch.no_grad():
                prob = mdl(torch.FloatTensor(features).reshape(1, 1, -1)).item()
        else:
            prob = mdl.predict_proba(features.reshape(1, -1))[0][1]
        results[name] = {"prob": prob, "pred": int(prob >= 0.5), "tier": assign_card_risk_tier(prob)}
    return results

def predict_batch(models, X, model_name):
    mdl = models[model_name]
    if model_name == "LSTM":
        with torch.no_grad():
            probs = np.atleast_1d(mdl(torch.FloatTensor(X.reshape(X.shape[0], 1, X.shape[1]))).numpy())
    else:
        probs = mdl.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    tiers = [assign_card_risk_tier(p) for p in probs]
    return probs, preds, tiers

def risk_color(tier):
    return {"Safe": "#00ff88", "Suspicious": "#ffcc00", "Critical Risk": "#ff007f"}.get(tier, "#e0e0e0")

def risk_tag(tier):
    cls = {"Safe": "tag-safe", "Suspicious": "tag-suspicious", "Critical Risk": "tag-critical"}.get(tier, "tag-safe")
    return f'<span class="tag {cls}">{tier}</span>'

def make_gauge(prob, title="Fraud Probability"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 36, "family": "Orbitron"}},
        title={"text": title, "font": {"size": 14, "color": "#8b949e"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
            "bar": {"color": "#00f0ff", "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 25], "color": "rgba(0,255,136,0.15)"},
                {"range": [25, 70], "color": "rgba(255,204,0,0.15)"},
                {"range": [70, 100], "color": "rgba(255,0,127,0.15)"},
            ],
            "threshold": {"line": {"color": "#ff007f", "width": 3}, "thickness": 0.8, "value": prob * 100},
        },
    ))
    fig.update_layout(height=260, **PLOTLY_LAYOUT)
    return fig

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Orbitron;font-size:1.4rem;color:#00f0ff;text-align:center;">🛡️ CardShield AI</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#8b949e;font-size:0.8rem;">Fraud Detection Engine</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center;color:#00ff88;font-size:0.75rem;">Logged in as: <b>{st.session_state.user_display or st.session_state.username}</b></p>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", ["🏠 Dashboard", "🔍 Real-Time Scan", "📁 Batch Analysis", "📊 Model Analytics", "🧪 Train Your Model"], label_visibility="collapsed")
    st.divider()
    models = load_models()
    amt_scaler, time_scaler = load_scalers()
    st.markdown("**Models Loaded**")
    for m in models:
        st.markdown(f"✅ {m}")
    if not models:
        st.warning("No models found in saved_models/")
    st.markdown(f"**Scalers:** {'✅ Ready' if amt_scaler else '❌ Missing'}")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.rerun()

# ──────────────────────────────────────────────
# PAGE: DASHBOARD
# ──────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown('<p class="hero-title">CardShield AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">AI-Powered Credit Card Fraud Detection • LSTM + Random Forest + XGBoost</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models Active", len(models))
    c2.metric("Input Features", "30")
    c3.metric("Architecture", "Hybrid")
    c4.metric("Risk Tiers", "3")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Model Architecture")
        st.markdown("""
        | Component | Details |
        |-----------|---------|
        | **LSTM** | 32 hidden units, BCE loss, Adam optimizer |
        | **Random Forest** | 100 trees, max depth 12 |
        | **XGBoost** | Gradient-boosted ensemble |
        | **Preprocessing** | RobustScaler + SMOTE oversampling |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Risk Classification")
        st.markdown(f"""
        | Probability | Tier | Action |
        |-------------|------|--------|
        | < 25% | {risk_tag("Safe")} | Auto-approve |
        | 25% – 70% | {risk_tag("Suspicious")} | Manual review |
        | > 70% | {risk_tag("Critical Risk")} | Block & alert |
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if os.path.exists("outputs/roc_auc_curve.png"):
        st.markdown("### 📈 ROC-AUC Performance")
        st.image("outputs/roc_auc_curve.png", use_container_width=True)

# ──────────────────────────────────────────────
# PAGE: REAL-TIME SCAN
# ──────────────────────────────────────────────
elif page == "🔍 Real-Time Scan":
    st.markdown("## 🔍 Real-Time Transaction Scan")
    st.markdown('<p style="color:#8b949e;">Choose a model and enter transaction details for instant fraud detection.</p>', unsafe_allow_html=True)

    if not models:
        st.error("No models loaded. Train models first with `python main.py`.")
        st.stop()
    if not amt_scaler:
        st.error("Scalers not found. Place `creditcard.csv` in `data/` and restart.")
        st.stop()

    # ── Model Selector ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    sel_col1, sel_col2 = st.columns([1, 1])
    with sel_col1:
        model_options = list(models.keys()) + ["🔀 All Models (Ensemble)"]
        selected_model = st.selectbox("🧠 Select Model", model_options, help="Choose which model to use for prediction")
    with sel_col2:
        scan_mode = st.radio("Scan Mode", ["⚡ Quick Scan", "🔬 Advanced Scan"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("scan_form"):
        if scan_mode == "⚡ Quick Scan":
            st.info("Quick Scan uses Amount & Time only. PCA features (V1-V28) default to 0.")
            qc1, qc2 = st.columns(2)
            amount = qc1.number_input("💰 Transaction Amount ($)", min_value=0.0, max_value=30000.0, value=150.0, step=0.01)
            time_val = qc2.number_input("⏱️ Time (seconds from first txn)", min_value=0.0, max_value=200000.0, value=50000.0, step=1.0)
            v_features = [0.0] * 28
        else:
            st.info("Enter all 28 PCA features for maximum accuracy.")
            fc1, fc2 = st.columns(2)
            amount = fc1.number_input("💰 Amount ($)", min_value=0.0, max_value=30000.0, value=150.0, step=0.01)
            time_val = fc2.number_input("⏱️ Time (sec)", min_value=0.0, max_value=200000.0, value=50000.0, step=1.0)
            v_features = []
            cols = st.columns(4)
            for i in range(28):
                v_features.append(cols[i % 4].number_input(f"V{i+1}", value=0.0, format="%.4f", key=f"v{i+1}"))

        submitted = st.form_submit_button("🛡️ Run Fraud Scan", use_container_width=True)

    if submitted:
        scaled_amt = amt_scaler.transform([[amount]])[0][0]
        scaled_time = time_scaler.transform([[time_val]])[0][0]
        features = np.array(v_features + [scaled_amt, scaled_time], dtype=np.float32)

        # Decide which models to run
        if selected_model == "🔀 All Models (Ensemble)":
            run_models = models
        else:
            run_models = {selected_model: models[selected_model]}

        with st.spinner(f"Analyzing with {selected_model}..."):
            results = predict_single(run_models, features)

        st.markdown("---")

        if selected_model == "🔀 All Models (Ensemble)":
            # ── ENSEMBLE MODE ──
            st.markdown("### 🎯 Ensemble Scan Results")
            avg_prob = np.mean([r["prob"] for r in results.values()])
            ensemble_tier = assign_card_risk_tier(avg_prob)

            gc1, gc2 = st.columns([1, 1])
            with gc1:
                st.plotly_chart(make_gauge(avg_prob, "Ensemble Fraud Score"), use_container_width=True)
            with gc2:
                st.markdown(f'<div class="glass-card risk-{ensemble_tier.split()[0].lower()}">', unsafe_allow_html=True)
                st.markdown(f"#### Verdict: {risk_tag(ensemble_tier)}", unsafe_allow_html=True)
                st.markdown(f'<p class="stat-number cyan">{avg_prob*100:.1f}%</p>', unsafe_allow_html=True)
                st.markdown(f"**Amount:** ${amount:,.2f}")
                st.markdown(f"**Models Consulted:** {len(results)}")
                action = {"Safe": "✅ Auto-Approve", "Suspicious": "⚠️ Flag for Review", "Critical Risk": "🚨 Block Transaction"}
                st.markdown(f"**Action:** {action.get(ensemble_tier, 'Review')}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 📊 Per-Model Breakdown")
            mcols = st.columns(len(results))
            for i, (name, r) in enumerate(results.items()):
                with mcols[i]:
                    st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f"**{name}**")
                    st.metric("Probability", f"{r['prob']*100:.1f}%")
                    st.markdown(f"Risk: {risk_tag(r['tier'])}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # Comparison chart
            fig = go.Figure()
            names = list(results.keys())
            probs_list = [results[n]["prob"] * 100 for n in names]
            colors = [risk_color(results[n]["tier"]) for n in names]
            fig.add_trace(go.Bar(x=names, y=probs_list, marker_color=colors, text=[f"{p:.1f}%" for p in probs_list], textposition="outside", textfont=dict(color="#e0e0e0")))
            fig.add_hline(y=50, line_dash="dash", line_color="#ff007f", annotation_text="Fraud Threshold")
            fig.update_layout(title="Model Comparison", yaxis_title="Fraud Probability %", yaxis_range=[0, 105], **PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig, use_container_width=True)

        else:
            # ── SINGLE MODEL MODE ──
            r = results[selected_model]
            st.markdown(f"### 🎯 {selected_model} — Scan Result")

            gc1, gc2 = st.columns([1, 1])
            with gc1:
                st.plotly_chart(make_gauge(r["prob"], f"{selected_model} Fraud Score"), use_container_width=True)
            with gc2:
                tier_key = r["tier"].split()[0].lower()
                st.markdown(f'<div class="glass-card risk-{tier_key}">', unsafe_allow_html=True)
                st.markdown(f"#### Verdict: {risk_tag(r['tier'])}", unsafe_allow_html=True)
                st.markdown(f'<p class="stat-number cyan">{r["prob"]*100:.1f}%</p>', unsafe_allow_html=True)
                st.markdown(f"**Model:** {selected_model}")
                st.markdown(f"**Amount:** ${amount:,.2f}")
                st.markdown(f"**Prediction:** {'🚨 FRAUD' if r['pred'] == 1 else '✅ LEGITIMATE'}")
                action = {"Safe": "✅ Auto-Approve", "Suspicious": "⚠️ Flag for Review", "Critical Risk": "🚨 Block Transaction"}
                st.markdown(f"**Action:** {action.get(r['tier'], 'Review')}")
                st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PAGE: BATCH ANALYSIS
# ──────────────────────────────────────────────
elif page == "📁 Batch Analysis":
    st.markdown("## 📁 Batch Data Analysis & Fraud Detection")
    st.markdown('<p style="color:#8b949e;">Upload any credit-type CSV — get full analysis, interactive graphs, and fraud predictions (if format matches).</p>', unsafe_allow_html=True)

    # Model selector for this page
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    ba_c1, ba_c2 = st.columns([1, 1])
    with ba_c1:
        if models:
            model_opts = list(models.keys()) + ["🔀 All Models (Ensemble)"]
            model_choice = st.selectbox("🧠 Model for Prediction", model_opts, key="batch_model")
        else:
            st.warning("No models loaded — EDA only mode.")
            model_choice = None
    with ba_c2:
        st.markdown("""
        **Accepted Formats:**
        - Any CSV with numeric data → Full EDA
        - `Time, V1-V28, Amount` → EDA + Fraud Prediction
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # ── DATA OVERVIEW ──
        st.markdown("---")
        st.markdown("### 📋 Data Overview")
        ov1, ov2, ov3, ov4 = st.columns(4)
        ov1.metric("Rows", f"{df.shape[0]:,}")
        ov2.metric("Columns", df.shape[1])
        ov3.metric("Numeric Cols", df.select_dtypes(include=np.number).shape[1])
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        ov4.metric("Missing %", f"{missing_pct:.1f}%")

        st.markdown("#### 🔎 Data Preview")
        st.dataframe(df.head(20), use_container_width=True, height=300)

        # ── STATISTICAL SUMMARY ──
        st.markdown("---")
        st.markdown("### 📊 Statistical Summary")
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            st.dataframe(num_df.describe().round(3), use_container_width=True)

        # ── MISSING VALUES ──
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.markdown("### ⚠️ Missing Values")
            fig = go.Figure(go.Bar(
                x=missing.index, y=missing.values,
                marker_color="#ff007f", text=missing.values, textposition="outside",
                textfont=dict(color="#e0e0e0"),
            ))
            fig.update_layout(title="Missing Values Per Column", yaxis_title="Count", **PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig, use_container_width=True)

        # ── DISTRIBUTION CHARTS ──
        if not num_df.empty:
            st.markdown("---")
            st.markdown("### 📈 Column Distributions")
            dist_cols = list(num_df.columns)
            # Show up to 8 most interesting columns
            if len(dist_cols) > 8:
                # Prioritize by variance
                variances = num_df.var().sort_values(ascending=False)
                dist_cols = list(variances.index[:8])

            n_charts = len(dist_cols)
            rows_needed = (n_charts + 1) // 2
            for row_i in range(rows_needed):
                dc1, dc2 = st.columns(2)
                for ci, col_widget in enumerate([dc1, dc2]):
                    idx = row_i * 2 + ci
                    if idx >= n_charts:
                        break
                    col_name = dist_cols[idx]
                    with col_widget:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=num_df[col_name].dropna(), nbinsx=40,
                            marker_color="#00f0ff", opacity=0.8,
                            name=col_name,
                        ))
                        fig.update_layout(
                            title=f"{col_name}", xaxis_title=col_name,
                            yaxis_title="Frequency", **PLOTLY_LAYOUT, height=280,
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # ── BOX PLOTS (Outlier Detection) ──
        if not num_df.empty and len(num_df.columns) <= 30:
            st.markdown("---")
            st.markdown("### 📦 Box Plots — Outlier Detection")
            box_cols = list(num_df.columns[:12])  # limit to 12 for readability
            fig = go.Figure()
            colors = ["#00f0ff", "#ff007f", "#00ff88", "#ffcc00", "#a855f7", "#f97316",
                       "#06b6d4", "#ec4899", "#22c55e", "#eab308", "#8b5cf6", "#fb923c"]
            for i, col in enumerate(box_cols):
                fig.add_trace(go.Box(
                    y=num_df[col].dropna(), name=col,
                    marker_color=colors[i % len(colors)], boxmean=True,
                ))
            fig.update_layout(title="Feature Distributions", **PLOTLY_LAYOUT, height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── CORRELATION HEATMAP ──
        if not num_df.empty and 2 <= len(num_df.columns) <= 40:
            st.markdown("---")
            st.markdown("### 🔥 Correlation Heatmap")
            corr = num_df.corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale=[[0, "#0a0a0f"], [0.5, "#00f0ff"], [1, "#ff007f"]],
                zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}",
                textfont=dict(size=9),
            ))
            fig.update_layout(title="Feature Correlations", **PLOTLY_LAYOUT,
                              height=max(400, len(corr.columns) * 28))
            st.plotly_chart(fig, use_container_width=True)

        # ── TARGET / CLASS DISTRIBUTION ──
        # Auto-detect target column
        target_col = None
        for candidate in ["Class", "class", "TARGET", "target", "Label", "label", "Fraud", "fraud", "is_fraud", "isFraud", "default", "Default"]:
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col:
            st.markdown("---")
            st.markdown(f"### 🎯 Target Column: `{target_col}`")
            tc1, tc2 = st.columns(2)
            with tc1:
                val_counts = df[target_col].value_counts()
                fig = go.Figure(go.Pie(
                    labels=[str(v) for v in val_counts.index], values=val_counts.values,
                    marker_colors=["#00ff88", "#ff007f"] + ["#ffcc00", "#a855f7"][:len(val_counts)-2],
                    hole=0.5, textfont=dict(color="#e0e0e0"),
                ))
                fig.update_layout(title=f"{target_col} Distribution", **PLOTLY_LAYOUT, height=350)
                st.plotly_chart(fig, use_container_width=True)
            with tc2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                for val, count in val_counts.items():
                    pct = count / len(df) * 100
                    st.markdown(f"**{target_col} = {val}:** {count:,} ({pct:.2f}%)")
                imbalance = val_counts.max() / val_counts.min() if val_counts.min() > 0 else float('inf')
                if imbalance > 10:
                    st.warning(f"⚠️ High imbalance ratio: {imbalance:.1f}:1")
                elif imbalance > 3:
                    st.info(f"Moderate imbalance: {imbalance:.1f}:1")
                else:
                    st.success(f"Balanced: {imbalance:.1f}:1")
                st.markdown('</div>', unsafe_allow_html=True)

        # ──────────────────────────────────────
        # FRAUD PREDICTION (if format matches)
        # ──────────────────────────────────────
        has_raw = "Amount" in df.columns and "Time" in df.columns
        has_v_cols = all(f"V{i}" in df.columns for i in range(1, 29))
        has_preprocessed = all(c in df.columns for c in FEATURE_COLS)
        can_predict = models and model_choice and (has_preprocessed or (has_raw and has_v_cols and amt_scaler))

        if can_predict:
            st.markdown("---")
            st.markdown("### 🛡️ Fraud Prediction Results")
            chosen_model = model_choice

            has_class = "Class" in df.columns

            if has_raw and has_v_cols and amt_scaler:
                with st.spinner(f"Running predictions with {chosen_model}..."):
                    df_proc = df.copy()
                    df_proc["scaled_amount"] = amt_scaler.transform(df_proc[["Amount"]])
                    df_proc["scaled_time"] = time_scaler.transform(df_proc[["Time"]])
                    df_proc.drop(["Time", "Amount"], axis=1, inplace=True, errors="ignore")
                    if has_class:
                        y_true = df_proc["Class"].values
                        df_proc.drop("Class", axis=1, inplace=True, errors="ignore")
                    X = df_proc.values
            else:
                with st.spinner(f"Running predictions with {chosen_model}..."):
                    X = df[FEATURE_COLS].values
                    has_class = False

            # Run predictions — ensemble or single
            if chosen_model == "🔀 All Models (Ensemble)":
                all_probs = {}
                for mname, mdl in models.items():
                    if mname == "LSTM":
                        with torch.no_grad():
                            all_probs[mname] = np.atleast_1d(mdl(torch.FloatTensor(X.reshape(X.shape[0], 1, X.shape[1]))).numpy())
                    else:
                        all_probs[mname] = mdl.predict_proba(X)[:, 1]
                probs = np.mean(list(all_probs.values()), axis=0)
            else:
                probs, _, _ = predict_batch(models, X, chosen_model)

            preds = (probs >= 0.5).astype(int)
            tiers = [assign_card_risk_tier(p) for p in probs]

            result_df = pd.DataFrame({
                "Transaction": range(1, len(probs)+1),
                "Fraud_Probability": np.round(probs, 4),
                "Prediction": preds,
                "Risk_Tier": tiers,
            })
            if has_class:
                result_df["Actual"] = y_true.astype(int)

            # Summary metrics
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Total Scanned", len(result_df))
            rc2.metric("Flagged Fraud", int(result_df["Prediction"].sum()))
            rc3.metric("Critical Risk", int((result_df["Risk_Tier"] == "Critical Risk").sum()))
            if has_class:
                acc = (result_df["Prediction"] == result_df["Actual"]).mean()
                rc4.metric("Accuracy", f"{acc*100:.2f}%")
            else:
                rc4.metric("Avg Prob", f"{probs.mean()*100:.2f}%")

            # Prediction charts
            pr1, pr2 = st.columns(2)
            with pr1:
                tier_counts = result_df["Risk_Tier"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=tier_counts.index, values=tier_counts.values,
                    marker_colors=[risk_color(t) for t in tier_counts.index],
                    hole=0.5, textfont=dict(color="#e0e0e0"),
                ))
                fig.update_layout(title=f"Risk Distribution ({chosen_model})", **PLOTLY_LAYOUT, height=350)
                st.plotly_chart(fig, use_container_width=True)

            with pr2:
                fig = go.Figure(go.Histogram(x=probs, nbinsx=50, marker_color="#00f0ff", opacity=0.8))
                fig.add_vline(x=0.5, line_dash="dash", line_color="#ff007f", annotation_text="Threshold")
                fig.update_layout(title="Fraud Probability Distribution", xaxis_title="Probability", yaxis_title="Count", **PLOTLY_LAYOUT, height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Results table
            st.markdown("#### 📋 Prediction Details")
            st.dataframe(result_df, use_container_width=True, height=400)

            csv_out = result_df.to_csv(index=False)
            st.download_button("📥 Download Predictions CSV", csv_out, "cardshield_results.csv", "text/csv", use_container_width=True)

        elif models:
            st.markdown("---")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.info("ℹ️ **Prediction not available** — This CSV doesn't match the fraud model format (`Time, V1-V28, Amount`). Full EDA is shown above.")
            st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PAGE: MODEL ANALYTICS
# ──────────────────────────────────────────────
elif page == "📊 Model Analytics":
    st.markdown("## 📊 Model Performance Analytics")
    st.markdown('<p style="color:#8b949e;">Training metrics, confusion matrices, and ROC curves from the latest training run.</p>', unsafe_allow_html=True)

    output_dir = "outputs"
    plots = {
        "Class Distribution": "class_distribution_graph.png",
        "Training vs Validation Loss": "training_vs_validation_loss.png",
        "Confusion Matrices": "confusion_matrices.png",
        "ROC-AUC Curve": "roc_auc_curve.png",
        "Precision-Recall Curve": "precision_recall_curve.png",
    }

    found = {k: v for k, v in plots.items() if os.path.exists(os.path.join(output_dir, v))}

    if not found:
        st.warning("No output plots found. Run `python main.py` to train and generate visualizations.")
        st.stop()

    tabs = st.tabs(list(found.keys()))
    for tab, (title, fname) in zip(tabs, found.items()):
        with tab:
            st.image(os.path.join(output_dir, fname), caption=title, use_container_width=True)

    # Feature importance from RF
    if "Random Forest" in models:
        st.markdown("---")
        st.markdown("### 🌲 Random Forest Feature Importance")
        rf = models["Random Forest"]
        importances = rf.feature_importances_
        feat_names = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
        top_idx = np.argsort(importances)[-15:]
        fig = go.Figure(go.Bar(
            x=importances[top_idx], y=[feat_names[i] for i in top_idx],
            orientation="h", marker_color="#00f0ff",
        ))
        fig.update_layout(title="Top 15 Features", xaxis_title="Importance", **PLOTLY_LAYOUT, height=450)
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# PAGE: TRAIN YOUR MODEL
# ──────────────────────────────────────────────
elif page == "🧪 Train Your Model":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    )
    from xgboost import XGBClassifier

    st.markdown("## 🧪 Train Your Own Model")
    st.markdown('<p style="color:#8b949e;">Upload any credit-card-type CSV. We\'ll auto-detect the target, train RF + XGBoost, and give you full results — no coding required.</p>', unsafe_allow_html=True)

    # ── Upload ──
    train_file = st.file_uploader("📂 Drop your CSV here", type=["csv"], key="train_upload")

    if train_file:
        raw_df = pd.read_csv(train_file)
        st.markdown("---")

        # ── Quick Data Preview ──
        st.markdown("### 📋 Data Preview")
        pv1, pv2, pv3, pv4 = st.columns(4)
        pv1.metric("Rows", f"{raw_df.shape[0]:,}")
        pv2.metric("Columns", raw_df.shape[1])
        pv3.metric("Numeric", raw_df.select_dtypes(include=np.number).shape[1])
        missing_pct = (raw_df.isnull().sum().sum() / (raw_df.shape[0] * raw_df.shape[1])) * 100
        pv4.metric("Missing %", f"{missing_pct:.1f}%")
        st.dataframe(raw_df.head(10), use_container_width=True, height=280)

        # ── Target Column Selection ──
        st.markdown("---")
        st.markdown("### 🎯 Select Target Column")
        st.markdown('<p style="color:#8b949e;">Choose the column your model should predict (e.g. Class, Fraud, Default).</p>', unsafe_allow_html=True)

        # Auto-detect candidates
        target_candidates = []
        for col in raw_df.columns:
            nunique = raw_df[col].nunique()
            if nunique == 2:
                target_candidates.insert(0, col)  # binary cols first
            elif 2 < nunique <= 10:
                target_candidates.append(col)

        # Also check by common names
        common_names = ["Class", "class", "TARGET", "target", "Label", "label",
                        "Fraud", "fraud", "is_fraud", "isFraud", "default",
                        "Default", "default.payment.next.month", "SeriousDlqin2yrs"]
        auto_pick = None
        for cn in common_names:
            if cn in raw_df.columns:
                auto_pick = cn
                break

        all_cols = list(raw_df.columns)
        default_idx = all_cols.index(auto_pick) if auto_pick and auto_pick in all_cols else 0
        target_col = st.selectbox("Target column", all_cols, index=default_idx,
                                  help="The column containing the label (0/1, Yes/No, etc.)")

        # Show target distribution
        tc1, tc2 = st.columns([1, 2])
        with tc1:
            val_counts = raw_df[target_col].value_counts()
            for val, cnt in val_counts.items():
                pct = cnt / len(raw_df) * 100
                st.markdown(f"**{target_col} = {val}:** {cnt:,} ({pct:.1f}%)")
            if val_counts.min() > 0:
                ratio = val_counts.max() / val_counts.min()
                if ratio > 10:
                    st.warning(f"⚠️ High imbalance ({ratio:.0f}:1) — SMOTE will be applied.")
                elif ratio > 3:
                    st.info(f"Moderate imbalance ({ratio:.1f}:1)")
        with tc2:
            fig = go.Figure(go.Pie(
                labels=[str(v) for v in val_counts.index], values=val_counts.values,
                marker_colors=["#00ff88", "#ff007f"] + ["#ffcc00", "#a855f7"][:max(0, len(val_counts)-2)],
                hole=0.5, textfont=dict(color="#e0e0e0"),
            ))
            fig.update_layout(title=f"{target_col} Distribution", **PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig, use_container_width=True)

        # ── Training Config ──
        st.markdown("---")
        st.markdown("### ⚙️ Training Configuration")
        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1:
            test_size = st.slider("Test Split %", 10, 40, 20, 5) / 100
        with cfg2:
            n_trees = st.selectbox("RF Trees", [50, 100, 200, 300], index=1)
        with cfg3:
            xgb_rounds = st.selectbox("XGB Rounds", [50, 100, 200, 300], index=1)

        use_smote = st.checkbox("Apply SMOTE (recommended for imbalanced data)", value=True)

        # ── TRAIN BUTTON ──
        if st.button("🚀 Train Models", use_container_width=True, type="primary"):
            with st.spinner("Preprocessing data..."):
                df = raw_df.copy()
                # Encode target if non-numeric
                le = None
                if not np.issubdtype(np.dtype(df[target_col].dtype), np.number):
                    st.error("Target column Should be numeric (0/1). Non-numeric targets will be label-encoded.")
                    le = LabelEncoder()
                    df[target_col] = le.fit_transform(df[target_col])

                y = df[target_col].values
                X_df = df.drop(columns=[target_col])

                # Drop non-numeric columns
                non_num = X_df.select_dtypes(exclude=np.number).columns.tolist()
                if non_num:
                    st.info(f"Dropped non-numeric columns: {', '.join(non_num)}")
                    X_df = X_df.select_dtypes(include=np.number)

                # Handle missing values
                if X_df.isnull().sum().sum() > 0:
                    X_df = X_df.fillna(X_df.median())
                    st.info("Missing values filled with column medians.")

                feature_names = list(X_df.columns)

                # Scale
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X_df)

                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, stratify=y, random_state=42
                )

                # SMOTE
                if use_smote:
                    try:
                        from imblearn.over_sampling import SMOTE
                        sm = SMOTE(random_state=42)
                        X_train, y_train = sm.fit_resample(X_train, y_train)
                        st.info(f"SMOTE applied — training set resampled to {X_train.shape[0]:,} rows.")
                    except Exception:
                        st.warning("SMOTE failed — training on original distribution.")

            # ── Train Random Forest ──
            with st.spinner("Training Random Forest..."):
                rf = RandomForestClassifier(
                    n_estimators=n_trees, max_depth=12, random_state=42, n_jobs=-1
                )
                rf.fit(X_train, y_train)
                rf_probs = rf.predict_proba(X_test)[:, 1]
                rf_preds = rf.predict(X_test)

            # ── Train XGBoost ──
            with st.spinner("Training XGBoost..."):
                neg_c = int(np.sum(y_train == 0))
                pos_c = int(np.sum(y_train == 1))
                spw = neg_c / pos_c if pos_c > 0 else 1.0
                xgb = XGBClassifier(
                    n_estimators=xgb_rounds, max_depth=8, learning_rate=0.1,
                    scale_pos_weight=spw, random_state=42, n_jobs=-1,
                    eval_metric='logloss',
                )
                xgb.fit(X_train, y_train)
                xgb_probs = xgb.predict_proba(X_test)[:, 1]
                xgb_preds = xgb.predict(X_test)

            # ══════════════════════════════════════
            #  RESULTS
            # ══════════════════════════════════════
            st.markdown("---")
            st.markdown("## 🏆 Training Results")

            rf_rep = classification_report(y_test, rf_preds, output_dict=True)
            xgb_rep = classification_report(y_test, xgb_preds, output_dict=True)

            # ── Accuracy Cards ──
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("RF Accuracy", f"{rf_rep['accuracy']*100:.2f}%")
            mc2.metric("RF F1 (fraud)", f"{rf_rep.get('1', rf_rep.get('1.0', {})).get('f1-score', 0)*100:.2f}%")
            mc3.metric("XGB Accuracy", f"{xgb_rep['accuracy']*100:.2f}%")
            mc4.metric("XGB F1 (fraud)", f"{xgb_rep.get('1', xgb_rep.get('1.0', {})).get('f1-score', 0)*100:.2f}%")

            # ── Confusion Matrices ──
            st.markdown("### 🔢 Confusion Matrices")
            cm1, cm2 = st.columns(2)
            for col_widget, preds, name, color in [
                (cm1, rf_preds, "Random Forest", "#00f0ff"),
                (cm2, xgb_preds, "XGBoost", "#ff007f"),
            ]:
                with col_widget:
                    cm = confusion_matrix(y_test, preds)
                    fig = go.Figure(go.Heatmap(
                        z=cm, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],
                        colorscale=[[0, "#0a0a0f"], [1, color]],
                        text=cm, texttemplate="%{text}", textfont=dict(size=18, color="#e0e0e0"),
                        showscale=False,
                    ))
                    fig.update_layout(title=f"{name} Confusion Matrix", **PLOTLY_LAYOUT, height=320)
                    st.plotly_chart(fig, use_container_width=True)

            # ── ROC-AUC Curves ──
            st.markdown("### 📈 ROC-AUC Curves")
            try:
                rf_auc = roc_auc_score(y_test, rf_probs)
                xgb_auc = roc_auc_score(y_test, xgb_probs)
                fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
                fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name=f"RF (AUC={rf_auc:.4f})",
                                         line=dict(color="#00f0ff", width=2)))
                fig.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, name=f"XGB (AUC={xgb_auc:.4f})",
                                         line=dict(color="#ff007f", width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                         line=dict(color="#3d4250", dash="dot")))
                fig.update_layout(title="ROC-AUC Comparison", xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate", **PLOTLY_LAYOUT, height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("ROC-AUC could not be computed (target may not be binary).")

            # ── Precision-Recall Curve ──
            st.markdown("### 🎯 Precision-Recall Curves")
            try:
                rf_ap = average_precision_score(y_test, rf_probs)
                xgb_ap = average_precision_score(y_test, xgb_probs)
                prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_probs)
                prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, xgb_probs)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rec_rf, y=prec_rf, name=f"RF (AP={rf_ap:.4f})",
                                         line=dict(color="#00f0ff", width=2)))
                fig.add_trace(go.Scatter(x=rec_xgb, y=prec_xgb, name=f"XGB (AP={xgb_ap:.4f})",
                                         line=dict(color="#ff007f", width=2)))
                fig.update_layout(title="Precision-Recall Comparison", xaxis_title="Recall",
                                  yaxis_title="Precision", **PLOTLY_LAYOUT, height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Precision-Recall curve could not be computed.")

            # ── Feature Importance ──
            st.markdown("### 🌲 Feature Importance (Top 15)")
            fi1, fi2 = st.columns(2)
            for col_widget, model_obj, name, color in [
                (fi1, rf, "Random Forest", "#00f0ff"),
                (fi2, xgb, "XGBoost", "#ff007f"),
            ]:
                with col_widget:
                    importances = model_obj.feature_importances_
                    top_n = min(15, len(importances))
                    top_idx = np.argsort(importances)[-top_n:]
                    fig = go.Figure(go.Bar(
                        x=importances[top_idx],
                        y=[feature_names[i] for i in top_idx],
                        orientation="h", marker_color=color,
                    ))
                    fig.update_layout(title=f"{name} — Top {top_n}", xaxis_title="Importance",
                                      **PLOTLY_LAYOUT, height=420)
                    st.plotly_chart(fig, use_container_width=True)

            # ── Probability Distribution ──
            st.markdown("### 📊 Prediction Probability Distribution")
            pd1, pd2 = st.columns(2)
            for col_widget, probs_arr, name, color in [
                (pd1, rf_probs, "Random Forest", "#00f0ff"),
                (pd2, xgb_probs, "XGBoost", "#ff007f"),
            ]:
                with col_widget:
                    fig = go.Figure(go.Histogram(x=probs_arr, nbinsx=50, marker_color=color, opacity=0.8))
                    fig.add_vline(x=0.5, line_dash="dash", line_color="#ffcc00", annotation_text="Threshold")
                    fig.update_layout(title=f"{name} — Prob. Distribution",
                                      xaxis_title="Fraud Probability", yaxis_title="Count",
                                      **PLOTLY_LAYOUT, height=320)
                    st.plotly_chart(fig, use_container_width=True)

            # ── Detailed Classification Report ──
            st.markdown("### 📝 Classification Report")
            rep1, rep2 = st.columns(2)
            with rep1:
                st.markdown("**Random Forest**")
                rf_report_df = pd.DataFrame(classification_report(y_test, rf_preds, output_dict=True)).T
                st.dataframe(rf_report_df.round(4), use_container_width=True)
            with rep2:
                st.markdown("**XGBoost**")
                xgb_report_df = pd.DataFrame(classification_report(y_test, xgb_preds, output_dict=True)).T
                st.dataframe(xgb_report_df.round(4), use_container_width=True)

            # ── Per-Row Predictions Download ──
            st.markdown("---")
            st.markdown("### 📥 Download Full Predictions")

            # Build ensemble prediction
            ens_probs = (rf_probs + xgb_probs) / 2
            ens_preds = (ens_probs >= 0.5).astype(int)
            ens_tiers = [assign_card_risk_tier(p) for p in ens_probs]

            result_df = pd.DataFrame({
                "Row": range(1, len(y_test) + 1),
                "Actual": y_test,
                "RF_Prob": np.round(rf_probs, 4),
                "RF_Pred": rf_preds,
                "XGB_Prob": np.round(xgb_probs, 4),
                "XGB_Pred": xgb_preds,
                "Ensemble_Prob": np.round(ens_probs, 4),
                "Ensemble_Pred": ens_preds,
                "Risk_Tier": ens_tiers,
            })

            # Summary row
            sr1, sr2, sr3, sr4 = st.columns(4)
            sr1.metric("Total Test Rows", f"{len(result_df):,}")
            sr2.metric("Ensemble Fraud", int(ens_preds.sum()))
            sr3.metric("Critical Risk", int((result_df["Risk_Tier"] == "Critical Risk").sum()))
            ens_acc = (ens_preds == y_test).mean()
            sr4.metric("Ensemble Accuracy", f"{ens_acc*100:.2f}%")

            st.dataframe(result_df, use_container_width=True, height=400)

            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions CSV", csv_out,
                "trained_model_predictions.csv", "text/csv",
                use_container_width=True,
            )
