"""
======================================================================
AI-NIDS — Streamlit Frontend
PHASE 12 (Enhanced Edition)

AI-powered Network Intrusion Detection System Dashboard

Backend:
    FastAPI
    http://127.0.0.1:8000

Frontend:
    Streamlit
======================================================================
"""

import json
import time
from typing import Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ======================================================================
# CONFIGURATION
# ======================================================================

API_URL = "http://127.0.0.1:8000"

HEALTH_URL = f"{API_URL}/health"
MODEL_INFO_URL = f"{API_URL}/model-info"
FEATURES_URL = f"{API_URL}/features"
ATTACK_CLASSES_URL = f"{API_URL}/attack-classes"
PREDICT_URL = f"{API_URL}/predict"


# ======================================================================
# PAGE CONFIGURATION
# ======================================================================

st.set_page_config(
    page_title="AI-NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================
# CUSTOM CSS — THEME, LAYOUT & ANIMATIONS
# ======================================================================

st.markdown(
    """
    <style>

    /* ---------------------------------------------------------- */
    /* Global font / background polish                            */
    /* ---------------------------------------------------------- */

    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #0f1c2e 0%, #0a0f1a 45%, #060a12 100%);
    }

    /* ---------------------------------------------------------- */
    /* Keyframe animations                                        */
    /* ---------------------------------------------------------- */

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }

    @keyframes pulseGlowGreen {
        0%   { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55); }
        70%  { box-shadow: 0 0 0 14px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    @keyframes pulseGlowRed {
        0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.55); }
        70%  { box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes shimmer {
        0%   { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes float {
        0%   { transform: translateY(0px); }
        50%  { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }

    /* ---------------------------------------------------------- */
    /* Hero header                                                 */
    /* ---------------------------------------------------------- */

    .hero-banner {
        background: linear-gradient(120deg, #0ea5e9, #6366f1, #0ea5e9);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite, fadeInUp 0.7s ease;
        border-radius: 18px;
        padding: 32px 40px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.25);
    }

    .hero-icon {
        font-size: 46px;
        display: inline-block;
        animation: float 3.5s ease-in-out infinite;
        margin-right: 14px;
    }

    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #ffffff;
        display: inline-block;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: rgba(255,255,255,0.88);
        margin-top: 6px;
        font-weight: 400;
    }

    .badge-row {
        margin-top: 18px;
    }

    .pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        margin-right: 10px;
        background: rgba(255,255,255,0.16);
        color: #ffffff;
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.25);
    }

    /* ---------------------------------------------------------- */
    /* Status badges (sidebar + system)                            */
    /* ---------------------------------------------------------- */

    .status-online {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #22c55e;
        padding: 12px 16px;
        border-radius: 12px;
        font-weight: 700;
        animation: fadeIn 0.6s ease;
    }

    .status-offline {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #ef4444;
        padding: 12px 16px;
        border-radius: 12px;
        font-weight: 700;
        animation: fadeIn 0.6s ease;
    }

    .dot-green {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: #22c55e;
        animation: pulseGlowGreen 2s infinite;
    }

    .dot-red {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: #ef4444;
        animation: pulseGlowRed 2s infinite;
    }

    /* ---------------------------------------------------------- */
    /* Section cards                                                */
    /* ---------------------------------------------------------- */

    .section-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 22px;
        animation: fadeInUp 0.6s ease;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .section-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
    }

    .attack-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 10px;
        padding: 10px 12px;
        margin: 5px 0;
        font-size: 13px;
        transition: transform 0.2s ease, background 0.2s ease;
        animation: fadeInUp 0.5s ease;
    }

    .attack-chip:hover {
        transform: scale(1.03);
        background: rgba(99, 102, 241, 0.22);
    }

    /* ---------------------------------------------------------- */
    /* Result banners                                               */
    /* ---------------------------------------------------------- */

    .result-box-safe {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 18px;
        margin-bottom: 22px;
        background: linear-gradient(135deg, rgba(34,197,94,0.16), rgba(34,197,94,0.04));
        border: 1px solid rgba(34, 197, 94, 0.45);
        animation: fadeInUp 0.6s ease, pulseGlowGreen 2.4s ease-in-out 1;
        color: #eafff1;
    }

    .result-box-danger {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 18px;
        margin-bottom: 22px;
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.05));
        border: 1px solid rgba(239, 68, 68, 0.5);
        animation: fadeInUp 0.6s ease, pulseGlowRed 2.4s ease-in-out 1;
        color: #fff0f0;
    }

    .result-box-safe h1, .result-box-danger h1 {
        font-size: 30px;
        margin-bottom: 6px;
    }

    /* ---------------------------------------------------------- */
    /* Footer                                                       */
    /* ---------------------------------------------------------- */

    .footer-text {
        text-align: center;
        opacity: 0.55;
        font-size: 12.5px;
        margin-top: 10px;
        animation: fadeIn 1s ease;
    }

    /* ---------------------------------------------------------- */
    /* Streamlit widget polish                                      */
    /* ---------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px 10px;
        transition: transform 0.2s ease;
        animation: fadeInUp 0.5s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
    }

    .stButton>button {
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.4px;
        transition: transform 0.15s ease, box-shadow 0.25s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.35);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# SESSION STATE
# ======================================================================

if "features" not in st.session_state:
    st.session_state.features = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "feature_values" not in st.session_state:
    st.session_state.feature_values = {}


# ======================================================================
# API HELPER
# ======================================================================

def api_get(url: str):
    """
    Perform GET request to FastAPI.
    """

    try:

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:

        return None

    except requests.exceptions.RequestException as error:

        st.error(
            f"API request failed:\n\n{error}"
        )

        return None


def api_post(url: str, payload: dict):
    """
    Perform POST request to FastAPI.
    """

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=30
        )

        if response.status_code == 422:

            try:
                error_data = response.json()

                st.error(
                    "❌ Validation Error (422)"
                )

                st.json(error_data)

            except Exception:

                st.error(
                    response.text
                )

            return None

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to AI-NIDS FastAPI server.\n\n"
            "Make sure this is running:\n"
            "`python src/api.py`"
        )

        return None

    except requests.exceptions.RequestException as error:

        st.error(
            f"Prediction request failed:\n\n{error}"
        )

        return None


# ======================================================================
# GAUGE HELPER (for confidence visualization)
# ======================================================================

def render_confidence_gauge(value: float, label: str, color: str):
    """
    Render an animated-feel Plotly gauge for a confidence score (0-1).
    Falls back to st.progress if Plotly isn't available.
    """

    pct = max(0.0, min(1.0, float(value))) * 100

    if not PLOTLY_AVAILABLE:
        st.caption(label)
        st.progress(pct / 100)
        return

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 26, "color": "white"}},
            title={"text": label, "font": {"size": 14, "color": "rgba(255,255,255,0.75)"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)"},
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(255,255,255,0.04)"},
                    {"range": [50, 80], "color": "rgba(255,255,255,0.07)"},
                    {"range": [80, 100], "color": "rgba(255,255,255,0.1)"},
                ],
            },
        )
    )

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ======================================================================
# LOAD BACKEND INFORMATION
# ======================================================================

health_data = api_get(HEALTH_URL)

model_data = api_get(MODEL_INFO_URL)

features_data = api_get(FEATURES_URL)

attack_classes_data = api_get(ATTACK_CLASSES_URL)


# ======================================================================
# CHECK BACKEND
# ======================================================================

backend_running = (
    health_data is not None
    and health_data.get("status") == "healthy"
)


# ======================================================================
# HERO HEADER
# ======================================================================

st.markdown(
    f"""
    <div class="hero-banner">
        <span class="hero-icon">🛡️</span>
        <span class="main-title">AI-NIDS</span>
        <div class="subtitle">
            AI-Powered Network Intrusion Detection System &nbsp;•&nbsp;
            Enterprise Security Analytics Dashboard
        </div>
        <div class="badge-row">
            <span class="pill">🧠 Random Forest Ensemble</span>
            <span class="pill">⚡ Real-time Inference</span>
            <span class="pill">🔒 50-Feature Flow Analysis</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# SIDEBAR
# ======================================================================

with st.sidebar:

    st.markdown("### ⚙️ System Status")

    if backend_running:

        st.markdown(
            '<div class="status-online">'
            '<span class="dot-green"></span> API Connected'
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="status-offline">'
            '<span class="dot-red"></span> API Offline'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Architecture")

    st.code(
        "Network Flow\n"
        "     ↓\n"
        "50 Features\n"
        "     ↓\n"
        "Binary Random Forest\n"
        "     ↓\n"
        "BENIGN / ATTACK\n"
        "     ↓\n"
        "Multi-class Random Forest\n"
        "     ↓\n"
        "Attack Type",
        language="text"
    )

    st.divider()

    st.caption(
        "AI-NIDS v1.0.0 — Enhanced UI"
    )


# ======================================================================
# BACKEND OFFLINE
# ======================================================================

if not backend_running:

    st.warning(
        "⚠️ AI-NIDS backend is not running."
    )

    st.markdown(
        """
        ### Start the backend first

        Open another PowerShell window and run:

        ```powershell
        cd C:\\Users\\PRATI\\OneDrive\\Desktop\\AI-NIDS\\ml-service
        python src/api.py
        ```

        Then refresh this page.
        """
    )

    st.stop()


# ======================================================================
# EXTRACT FEATURE LIST
# ======================================================================

if features_data:

    st.session_state.features = (
        features_data.get("features", [])
    )

features = st.session_state.features


# ======================================================================
# DASHBOARD METRICS
# ======================================================================

st.subheader("📊 System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "API Status",
        "ONLINE"
    )

with col2:

    st.metric(
        "Features",
        len(features)
    )

with col3:

    if model_data:

        classes = model_data.get(
            "attack_classes",
            []
        )

        st.metric(
            "Attack Classes",
            len(classes)
        )

    else:

        st.metric(
            "Attack Classes",
            "-"
        )

with col4:

    st.metric(
        "Version",
        "1.0.0"
    )


# ======================================================================
# MODEL INFORMATION
# ======================================================================

st.subheader("🤖 Model Information")

if model_data:

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"""
**Binary Model**

`{model_data.get("binary_model", "Unknown")}`

Purpose:

BENIGN vs ATTACK
"""
        )

    with col2:

        st.info(
            f"""
**Multi-class Model**

`{model_data.get("multiclass_model", "Unknown")}`

Purpose:

Attack type classification
"""
        )

    st.success(
        f"Architecture: "
        f"{model_data.get('architecture', 'Unknown')}"
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================================
# ATTACK CLASSES
# ======================================================================

if attack_classes_data:

    st.subheader("🚨 Supported Attack Classes")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    class_mapping = attack_classes_data.get(
        "classes",
        {}
    )

    class_columns = st.columns(5)

    for index, (key, value) in enumerate(
        class_mapping.items()
    ):

        with class_columns[index % 5]:

            st.markdown(
                f"""
                <div class="attack-chip">
                <b>{key}</b><br>{value}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================================
# FEATURE INPUT
# ======================================================================

st.divider()

st.header("🔍 Network Flow Prediction")

st.write(
    "Enter the 50 network-flow features below. "
    "The frontend automatically uses the exact feature "
    "order required by the trained models."
)


# ======================================================================
# INPUT MODE
# ======================================================================

input_mode = st.radio(
    "Input Method",
    [
        "Manual Feature Input",
        "JSON Input",
        "CSV Input"
    ],
    horizontal=True
)


# ======================================================================
# MANUAL INPUT
# ======================================================================

feature_values: Dict[str, float] = {}


if input_mode == "Manual Feature Input":

    if not features:

        st.error(
            "Feature configuration could not be loaded."
        )

        st.stop()

    st.subheader(
        f"📋 {len(features)} Network Features"
    )

    # --------------------------------------------------------------
    # Generate columns
    # --------------------------------------------------------------

    columns = st.columns(3)

    for index, feature in enumerate(features):

        with columns[index % 3]:

            value = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"feature_{index}"
            )

            feature_values[feature] = value


# ======================================================================
# JSON INPUT
# ======================================================================

elif input_mode == "JSON Input":

    st.subheader("📄 JSON Feature Input")

    st.caption(
        "Paste a JSON object containing all 50 features."
    )

    example = {
        feature: 0.0
        for feature in features
    }

    json_text = st.text_area(
        "Feature JSON",
        value=json.dumps(
            example,
            indent=2
        ),
        height=400
    )

    try:

        parsed_json = json.loads(
            json_text
        )

        if not isinstance(
            parsed_json,
            dict
        ):

            st.error(
                "JSON must be an object."
            )

        else:

            feature_values = parsed_json

    except json.JSONDecodeError as error:

        st.error(
            f"Invalid JSON:\n{error}"
        )


# ======================================================================
# CSV INPUT
# ======================================================================

elif input_mode == "CSV Input":

    st.subheader("📁 CSV Network Flow")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file:

        try:

            dataframe = pd.read_csv(
                uploaded_file
            )

            st.write(
                f"Rows: {len(dataframe)} | "
                f"Columns: {len(dataframe.columns)}"
            )

            st.dataframe(
                dataframe.head(),
                use_container_width=True
            )

            if len(dataframe) > 0:

                selected_row = st.number_input(
                    "Select row",
                    min_value=0,
                    max_value=len(dataframe) - 1,
                    value=0,
                    step=1
                )

                row = dataframe.iloc[
                    selected_row
                ]

                feature_values = {
                    feature: row[feature]
                    for feature in features
                    if feature in dataframe.columns
                }

        except Exception as error:

            st.error(
                f"Could not read CSV:\n{error}"
            )


# ======================================================================
# VALIDATE FEATURES BEFORE PREDICTION
# ======================================================================

if feature_values:

    provided = set(
        feature_values.keys()
    )

    expected = set(
        features
    )

    missing = sorted(
        expected - provided
    )

    extra = sorted(
        provided - expected
    )

    if missing:

        st.warning(
            f"⚠️ Missing {len(missing)} feature(s)."
        )

        with st.expander(
            "Show missing features"
        ):

            st.write(
                missing
            )

    if extra:

        st.warning(
            f"⚠️ {len(extra)} unknown feature(s) will "
            f"cause validation failure."
        )

        with st.expander(
            "Show extra features"
        ):

            st.write(
                extra
            )


# ======================================================================
# PREDICTION BUTTON
# ======================================================================

st.divider()

predict_button = st.button(
    "🚀 ANALYZE NETWORK FLOW",
    type="primary",
    use_container_width=True
)


# ======================================================================
# RUN PREDICTION
# ======================================================================

if predict_button:

    # --------------------------------------------------------------
    # Check all features
    # --------------------------------------------------------------

    missing = sorted(
        set(features)
        - set(feature_values.keys())
    )

    if missing:

        st.error(
            f"❌ {len(missing)} required feature(s) "
            "are missing."
        )

        with st.expander(
            "Missing Features"
        ):

            st.write(
                missing
            )

        st.stop()

    # --------------------------------------------------------------
    # Build exact feature dictionary
    # --------------------------------------------------------------

    final_features = {}

    invalid_features = []

    for feature in features:

        try:

            value = float(
                feature_values[feature]
            )

            if not np.isfinite(value):

                invalid_features.append(
                    feature
                )

            else:

                final_features[feature] = value

        except (
            TypeError,
            ValueError
        ):

            invalid_features.append(
                feature
            )

    if invalid_features:

        st.error(
            "❌ Invalid numeric values found."
        )

        st.write(
            invalid_features
        )

        st.stop()

    # --------------------------------------------------------------
    # Build API payload
    # --------------------------------------------------------------

    payload = {
        "features": final_features
    }

    # --------------------------------------------------------------
    # Show request
    # --------------------------------------------------------------

    with st.expander(
        "🔧 API Request"
    ):

        st.json(
            payload
        )

    # --------------------------------------------------------------
    # Call FastAPI (with staged progress animation)
    # --------------------------------------------------------------

    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text="Preparing network flow vector...")

    stages = [
        (25, "Preparing network flow vector..."),
        (55, "Running Binary Random Forest..."),
        (80, "Running Multi-class Random Forest..."),
        (100, "Finalizing detection result..."),
    ]

    for pct, msg in stages:
        time.sleep(0.15)
        progress_bar.progress(pct, text=msg)

    result = api_post(
        PREDICT_URL,
        payload
    )

    progress_placeholder.empty()

    if result:

        st.session_state.prediction_result = (
            result
        )

        if result.get("is_attack", False):
            st.toast("🚨 Attack detected in network flow!", icon="🚨")
        else:
            st.toast("✅ Flow classified as benign.", icon="✅")


# ======================================================================
# DISPLAY RESULT
# ======================================================================

result = st.session_state.prediction_result


if result:

    st.divider()

    st.header("🎯 Detection Result")

    is_attack = result.get(
        "is_attack",
        False
    )

    prediction = result.get(
        "prediction",
        "UNKNOWN"
    )

    attack_type = result.get(
        "attack_type",
        "UNKNOWN"
    )

    confidence = float(
        result.get(
            "confidence",
            0
        )
    )

    binary_confidence = float(
        result.get(
            "binary_confidence",
            0
        )
    )

    attack_confidence = result.get(
        "attack_confidence"
    )

    # --------------------------------------------------------------
    # BENIGN
    # --------------------------------------------------------------

    if not is_attack:

        st.balloons()

        st.markdown(
            f"""
            <div class="result-box-safe">
            <h1>🟢 NO ATTACK DETECTED</h1>
            <p>
            The Binary Random Forest classified this
            network flow as <b>BENIGN</b>.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------------
    # ATTACK
    # --------------------------------------------------------------

    else:

        st.markdown(
            f"""
            <div class="result-box-danger">
            <h1>🚨 {attack_type}</h1>
            <p>
            Network intrusion detected.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------------
    # RESULT METRICS
    # --------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Prediction",
            prediction
        )

    with col2:

        st.metric(
            "Attack Type",
            attack_type
        )

    with col3:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Binary Confidence",
            f"{binary_confidence * 100:.2f}%"
        )

    # --------------------------------------------------------------
    # CONFIDENCE GAUGES
    # --------------------------------------------------------------

    gauge_color = "#ef4444" if is_attack else "#22c55e"

    gcol1, gcol2 = st.columns(2)

    with gcol1:
        render_confidence_gauge(binary_confidence, "Binary Confidence", gauge_color)

    with gcol2:
        if attack_confidence is not None:
            render_confidence_gauge(float(attack_confidence), "Multi-class Attack Confidence", "#6366f1")
        else:
            render_confidence_gauge(confidence, "Overall Confidence", gauge_color)

    # --------------------------------------------------------------
    # RAW RESPONSE
    # --------------------------------------------------------------

    with st.expander(
        "📦 API Response"
    ):

        st.json(
            result
        )


# ======================================================================
# FEATURE REFERENCE
# ======================================================================

with st.expander(
    "📚 Model Feature Reference"
):

    if features:

        feature_df = pd.DataFrame(
            {
                "Index": range(
                    1,
                    len(features) + 1
                ),
                "Feature": features
            }
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )


# ======================================================================
# FOOTER
# ======================================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
    AI-NIDS v1.0.0 — Enhanced UI &nbsp;|&nbsp;
    Binary Random Forest → Multi-class Random Forest &nbsp;|&nbsp;
    FastAPI: http://127.0.0.1:8000
    </div>
    """,
    unsafe_allow_html=True,
)