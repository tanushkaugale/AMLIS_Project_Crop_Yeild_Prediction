"""
app.py
======
Streamlit Web Application: Crop Yield Prediction using Weather and Soil Data
Designed for Indian Farmers with an Evaluator Section for 5th-Semester IT College Viva.

Pages:
1. 🌾 Yield Prediction & Suggestions (Farmer-Facing)
2. 🌱 Crop Recommendation (Farmer-Facing)
3. 🤖 Farmer Assistant (Chatbot with MCP & LLM)
4. 📖 About Model (for evaluators)
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Suppress sklearn unpickle warnings
warnings.filterwarnings("ignore")

# Ensure local imports work reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import farm_logic
import chatbot

# ---------------------------------------------------------------------------
# Streamlit Page Setup & Agricultural Theme Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroYield | Crop Prediction & Farmer Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, accessible, mobile-friendly green agriculture aesthetic
st.markdown("""
<style>
    /* Main body background & typography */
    .stApp {
        background-color: #FAF8F5;
        color: #1F2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Ensure clear text visibility */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    .main p, .main span, .main label, .main div {
        color: inherit;
    }

    /* Input labels */
    .main label[data-baseweb="label"],
    .main [data-testid="stWidgetLabel"] label,
    .main [data-testid="stWidgetLabel"] p {
        color: #1F2937 !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1E4620 0%, #2D5A27 100%);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 20px;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(30, 70, 32, 0.15);
        border-left: 6px solid #D97706;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        margin-top: 6px;
        margin-bottom: 0;
        color: #E2E8F0 !important;
    }

    /* Primary yield card */
    .big-yield-card {
        background-color: #FFFFFF;
        border: 2px solid #2E7D32;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.12);
        text-align: center;
    }
    .big-yield-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .big-yield-number {
        font-size: 3.2rem;
        font-weight: 800;
        color: #1E4620;
        line-height: 1.1;
        margin: 8px 0;
    }
    .big-yield-unit {
        font-size: 1.15rem;
        font-weight: 500;
        color: #4B5563;
    }
    .big-yield-range {
        display: inline-block;
        background-color: #E8F5E9;
        color: #1B5E20;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 5px 14px;
        border-radius: 20px;
        margin-top: 8px;
    }

    /* Suggestion checklist cards */
    .suggestion-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 5px solid #2E7D32;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .suggestion-gain {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 3px 8px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .package-card {
        background-color: #F0FDF4;
        border: 2px dashed #22C55E;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    /* Weather tips box */
    .weather-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 14px;
    }

    /* Disclaimer box */
    .disclaimer-card {
        background-color: #FEF3C7;
        border: 1px solid #FDE68A;
        border-left: 5px solid #D97706;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 20px;
        font-size: 0.88rem;
        color: #78350F;
        line-height: 1.45;
    }

    /* Metric cards for evaluators */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #6B7280;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #1E4620;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.80rem;
        font-weight: 600;
        color: #2E7D32;
    }

    /* Primary buttons */
    div.stButton > button:first-child {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        font-size: 1.0rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1E4620 !important;
        box-shadow: 0 4px 10px rgba(30, 70, 32, 0.3) !important;
    }

    /* Dark readable sidebar */
    section[data-testid="stSidebar"] {
        background-color: #172F19;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #F8FAFC !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Artifacts and Setup Session State
# ---------------------------------------------------------------------------
try:
    metadata, lr_model, rf_model, test_preds = farm_logic.load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}. Please ensure 'models/' exists and contains trained pipelines.")
    st.stop()

crops_list = metadata.get("crops", [])
metrics_rf = metadata["evaluation_metrics"]["Random_Forest"]
metrics_lr = metadata["evaluation_metrics"]["Linear_Regression"]

# Initialize default session state values if not present
if "crop" not in st.session_state:
    st.session_state.crop = "Rice" if "Rice" in crops_list else crops_list[0]
if "rainfall" not in st.session_state:
    st.session_state.rainfall = 1150.0
if "temperature" not in st.session_state:
    st.session_state.temperature = 26.5
if "humidity" not in st.session_state:
    st.session_state.humidity = 72.0
if "pH" not in st.session_state:
    st.session_state.pH = 6.5
if "N" not in st.session_state:
    st.session_state.N = 78.0
if "P" not in st.session_state:
    st.session_state.P = 35.0
if "K" not in st.session_state:
    st.session_state.K = 28.0
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "suggestions_result" not in st.session_state:
    st.session_state.suggestions_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 Agrowise Assistant")
    st.caption("Crop Yield Prediction & Smart Farming Advisory")
    st.markdown("---")

    selected_page = st.radio(
        "Choose Page:",
        [
            "🌾  Yield Prediction & Suggestions",
            "🌱  Crop Recommendation",
            "🤖  Farmer Assistant (Chatbot)",
            "📖  About Model (for evaluators)"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 💡 Quick Stats")
    st.markdown(f"**Champion Model:** Random Forest")
    st.markdown(f"**Test Accuracy (R²):** `{metrics_rf['R2']:.4f}`")
    st.markdown(f"**Average Error (MAE):** `±{metrics_rf['MAE']:.2f} t/ha`")
    st.markdown(f"**Supported Crops:** {len(crops_list)} Indian varieties")

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.78rem; color: #94A3B8; text-align: center;'>"
        "College IT Capstone Project &bull; Weather & Soil ML"
        "</div>",
        unsafe_allow_html=True
    )


# ===========================================================================
# PAGE 1: YIELD PREDICTION & SUGGESTIONS (FARMER-FACING)
# ===========================================================================
if selected_page == "🌾 1. Yield Prediction & Suggestions":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🌾 Crop Yield Prediction & Soil Guidance</h1>
        <p class="hero-subtitle">Enter your weather and soil conditions to forecast crop yield in Tonnes per Hectare (t/ha) and get smart soil improvement tips.</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_out = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown("### 📝 Enter Farm Inputs")
        
        c_crop, c_rain = st.columns(2)
        with c_crop:
            st.session_state.crop = st.selectbox(
                "Select Crop",
                crops_list,
                index=crops_list.index(st.session_state.crop) if st.session_state.crop in crops_list else 0,
                help="Select the crop you plan to cultivate."
            )
        with c_rain:
            st.session_state.rainfall = st.slider(
                "Annual Rainfall (mm)",
                min_value=100.0,
                max_value=3500.0,
                value=float(st.session_state.rainfall),
                step=25.0,
                help="Estimated annual rainfall in millimeters (rain cannot be changed by farmer)."
            )

        c_temp, c_hum = st.columns(2)
        with c_temp:
            st.session_state.temperature = st.slider(
                "Temperature (°C)",
                min_value=10.0,
                max_value=45.0,
                value=float(st.session_state.temperature),
                step=0.5,
                help="Average temperature during crop season."
            )
        with c_hum:
            st.session_state.humidity = st.slider(
                "Relative Humidity (%)",
                min_value=20.0,
                max_value=100.0,
                value=float(st.session_state.humidity),
                step=1.0,
                help="Average atmospheric humidity percentage."
            )

        st.markdown("#### 🧪 Soil Test Parameters *(Farmer Controllable)*")
        c_ph, c_n = st.columns(2)
        with c_ph:
            st.session_state.pH = st.slider(
                "Soil pH",
                min_value=3.5,
                max_value=9.5,
                value=float(st.session_state.pH),
                step=0.1,
                help="Soil acidity/alkalinity (<5.5 acidic, 6.0-7.2 optimal, >7.5 alkaline)."
            )
        with c_n:
            st.session_state.N = st.slider(
                "Nitrogen (N) [kg/ha]",
                min_value=10.0,
                max_value=200.0,
                value=float(st.session_state.N),
                step=1.0,
                help="Available Nitrogen nutrient in soil."
            )

        c_p, c_k = st.columns(2)
        with c_p:
            st.session_state.P = st.slider(
                "Phosphorus (P) [kg/ha]",
                min_value=5.0,
                max_value=120.0,
                value=float(st.session_state.P),
                step=1.0,
                help="Available Phosphorus nutrient in soil."
            )
        with c_k:
            st.session_state.K = st.slider(
                "Potassium (K) [kg/ha]",
                min_value=5.0,
                max_value=120.0,
                value=float(st.session_state.K),
                step=1.0,
                help="Available Potassium nutrient in soil."
            )

        st.write("")
        if st.button("🚀 Predict Crop Yield", type="primary", use_container_width=True):
            with st.spinner("Calculating prediction and testing soil improvements..."):
                pred_res = farm_logic.predict_yield(
                    crop=st.session_state.crop,
                    rainfall=st.session_state.rainfall,
                    temperature=st.session_state.temperature,
                    humidity=st.session_state.humidity,
                    pH=st.session_state.pH,
                    N=st.session_state.N,
                    P=st.session_state.P,
                    K=st.session_state.K
                )
                sugg_res = farm_logic.suggest_improvements(
                    crop=st.session_state.crop,
                    rainfall=st.session_state.rainfall,
                    temperature=st.session_state.temperature,
                    humidity=st.session_state.humidity,
                    pH=st.session_state.pH,
                    N=st.session_state.N,
                    P=st.session_state.P,
                    K=st.session_state.K
                )
                st.session_state.prediction_result = pred_res
                st.session_state.suggestions_result = sugg_res
                st.session_state.predicted = True

    with col_out:
        st.markdown("### 📊 Estimated Harvest Result")
        
        if not st.session_state.predicted or st.session_state.prediction_result is None:
            st.info("👈 Enter your soil and weather details on the left, then click **'Predict Crop Yield'** to see your forecast.")
        else:
            p_res = st.session_state.prediction_result
            s_res = st.session_state.suggestions_result
            
            # Big Primary Result Card
            st.markdown(f"""
            <div class="big-yield-card">
                <div class="big-yield-title">Expected Yield for {p_res['crop']}</div>
                <div class="big-yield-number">{p_res['predicted_yield']:.2f} <span class="big-yield-unit">t/ha</span></div>
                <div class="big-yield-range">
                    Typical Margin: {p_res['yield_range_min']:.2f} – {p_res['yield_range_max']:.2f} t/ha (±{p_res['mae']:.2f} MAE)
                </div>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 10px;">
                    Tonnes per Hectare (1 t/ha ≈ 1000 kg per 2.47 acres)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Report Download Button
            report_lines = [
                "CROP YIELD PREDICTION & SOIL REPORT",
                "====================================",
                f"Crop Variety: {p_res['crop']}",
                f"Predicted Yield: {p_res['predicted_yield']} Tonnes per Hectare (t/ha)",
                f"Confidence Range: {p_res['yield_range_min']} to {p_res['yield_range_max']} t/ha (±{p_res['mae']} MAE)",
                "",
                "INPUT PARAMETERS:",
                f"Annual Rainfall: {st.session_state.rainfall} mm",
                f"Temperature: {st.session_state.temperature} °C",
                f"Relative Humidity: {st.session_state.humidity} %",
                f"Soil pH: {st.session_state.pH}",
                f"Nitrogen (N): {st.session_state.N} kg/ha",
                f"Phosphorus (P): {st.session_state.P} kg/ha",
                f"Potassium (K): {st.session_state.K} kg/ha",
                "",
                "RECOMMENDED SOIL IMPROVEMENTS:",
            ]
            if s_res and s_res.get("suggestions"):
                for idx, s in enumerate(s_res["suggestions"], 1):
                    report_lines.append(f"{idx}. {s['action']} -> Gain: +{s['yield_gain']} t/ha ({s['amendment_advice']})")
            else:
                report_lines.append("Current soil nutrients are already close to optimal for this crop.")
            report_lines.append("")
            report_lines.append("DISCLAIMER:")
            report_lines.append(farm_logic.MANDATORY_DISCLAIMER)

            report_text = "\n".join(report_lines)
            st.download_button(
                label="📥 Download Prediction Report (TXT)",
                data=report_text,
                file_name=f"crop_yield_report_{p_res['crop'].lower()}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # -----------------------------------------------------------------------
    # Section: Smart Yield Improvement Suggestions (Innovation)
    # -----------------------------------------------------------------------
    if st.session_state.predicted and st.session_state.suggestions_result is not None:
        st.markdown("---")
        st.markdown("### 💡 Smart Yield Improvement Suggestions")
        st.caption("Weather cannot be altered, but soil nutrients and pH can be conditioned. Below are model-tested adjustments sorted by estimated impact:")

        s_res = st.session_state.suggestions_result

        # Combined Package Card (if significant gain found)
        pkg = s_res.get("package_suggestion")
        if pkg:
            st.markdown(f"""
            <div class="package-card">
                <div style="font-weight: 700; color: #166534; font-size: 1.05rem;">⭐ {pkg['title']}</div>
                <div style="font-size: 0.92rem; color: #1F2937; margin: 4px 0 10px 0;">{pkg['description']}</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #1E4620;">
                    Potential Overall Yield: <strong>{pkg['improved_yield']:.2f} t/ha</strong> (Gain: <span style="color: #15803D;">+{pkg['yield_gain']:.2f} t/ha</span>)
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_sugg, col_whatif = st.columns([1.1, 1], gap="large")

        with col_sugg:
            st.markdown("#### ✅ What You Can Do (Sorted by Impact)")
            if s_res.get("suggestions"):
                for s in s_res["suggestions"]:
                    st.markdown(f"""
                    <div class="suggestion-card">
                        <span class="suggestion-gain">+{s['yield_gain']:.2f} t/ha Gain</span>
                        <div style="font-weight: 700; color: #1F2937; font-size: 0.98rem;">{s['action']}</div>
                        <div style="font-size: 0.88rem; color: #4B5563; margin-top: 4px;">{s['explanation']}</div>
                        <div style="font-size: 0.84rem; color: #2E7D32; margin-top: 6px; font-weight: 500;">
                            🌱 <strong>Agronomic Advice:</strong> {s['amendment_advice']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🎉 Your current soil parameters (N, P, K, pH) are well-balanced for this crop. No single nutrient adjustment yielded significant improvement.")

            # Weather Management Tips
            st.markdown("#### 🌦️ Weather Risk & Management Tips")
            tips = s_res.get("weather_tips", [])
            for tip in tips:
                st.markdown(f"""
                <div class="weather-box" style="margin-bottom: 8px; font-size: 0.90rem;">
                    {tip}
                </div>
                """, unsafe_allow_html=True)

        with col_whatif:
            st.markdown("#### ⚖️ What-If Comparison (Current vs. Improved)")
            st.caption("Test your own custom soil adjustments and see the predicted difference in real time:")

            # Pre-populate improved values from package or current
            init_ph = float(pkg["improved_inputs"]["pH"]) if pkg else float(st.session_state.pH)
            init_n = float(pkg["improved_inputs"]["N"]) if pkg else float(st.session_state.N)
            init_p = float(pkg["improved_inputs"]["P"]) if pkg else float(st.session_state.P)
            init_k = float(pkg["improved_inputs"]["K"]) if pkg else float(st.session_state.K)

            wi_c1, wi_c2 = st.columns(2)
            with wi_c1:
                wi_ph = st.number_input("Target Soil pH", 3.5, 9.5, value=init_ph, step=0.1)
                wi_n = st.number_input("Target Nitrogen (kg/ha)", 10.0, 200.0, value=init_n, step=5.0)
            with wi_c2:
                wi_p = st.number_input("Target Phosphorus (kg/ha)", 5.0, 120.0, value=init_p, step=5.0)
                wi_k = st.number_input("Target Potassium (kg/ha)", 5.0, 120.0, value=init_k, step=5.0)

            comparison = farm_logic.compare_what_if(
                crop=st.session_state.crop,
                rainfall=st.session_state.rainfall,
                temperature=st.session_state.temperature,
                humidity=st.session_state.humidity,
                cur_pH=st.session_state.pH,
                cur_N=st.session_state.N,
                cur_P=st.session_state.P,
                cur_K=st.session_state.K,
                imp_pH=wi_ph,
                imp_N=wi_n,
                imp_P=wi_p,
                imp_K=wi_k
            )

            diff_val = comparison["difference"]
            diff_color = "#15803D" if diff_val >= 0 else "#DC2626"
            diff_symbol = "+" if diff_val >= 0 else ""

            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 18px; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; text-align: center;">
                    <div>
                        <div style="font-size: 0.80rem; color: #6B7280; font-weight: 600;">CURRENT YIELD</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #1F2937;">{comparison['current_yield']:.2f} t/ha</div>
                    </div>
                    <div style="font-size: 1.5rem; color: #9CA3AF; align-self: center;">➔</div>
                    <div>
                        <div style="font-size: 0.80rem; color: #6B7280; font-weight: 600;">TARGET YIELD</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #1E4620;">{comparison['improved_yield']:.2f} t/ha</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 14px; padding-top: 12px; border-top: 1px solid #F3F4F6;">
                    <span style="font-size: 1.15rem; font-weight: 800; color: {diff_color};">
                        Predicted Difference: {diff_symbol}{diff_val:.2f} t/ha ({diff_symbol}{comparison['percent_change']}%)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Mandatory Academic & Agronomic Disclaimer
    st.markdown(f"""
    <div class="disclaimer-card">
        <strong>⚠️ Mandatory Academic & Agronomic Notice:</strong><br>
        {farm_logic.MANDATORY_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 2: CROP RECOMMENDATION (FARMER-FACING)
# ===========================================================================
elif selected_page == "🌱 2. Crop Recommendation":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🌱 Crop Recommendation Engine</h1>
        <p class="hero-subtitle">Discover the most productive crops for your exact climate and soil indicators, evaluated across 54 Indian varieties.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"**Evaluating based on active session conditions:** "
        f"Rainfall: `{st.session_state.rainfall:.0f} mm` | "
        f"Temp: `{st.session_state.temperature:.1f} °C` | "
        f"Humidity: `{st.session_state.humidity:.0f} %` | "
        f"Soil pH: `{st.session_state.pH:.1f}` | "
        f"N-P-K: `{st.session_state.N:.0f}-{st.session_state.P:.0f}-{st.session_state.K:.0f} kg/ha`"
    )

    top_count = st.slider("Number of Top Crops to Display:", min_value=3, max_value=10, value=5, step=1)

    with st.spinner("Simulating yields across all supported crops..."):
        recommended = farm_logic.recommend_crops(
            rainfall=st.session_state.rainfall,
            temperature=st.session_state.temperature,
            humidity=st.session_state.humidity,
            pH=st.session_state.pH,
            N=st.session_state.N,
            P=st.session_state.P,
            K=st.session_state.K,
            top_n=top_count
        )

    df_rec = pd.DataFrame(recommended)

    col_chart, col_tbl = st.columns([1.3, 1], gap="large")

    with col_chart:
        st.markdown("#### 📊 Top Crops Ranked by Estimated Yield")
        fig_rec = px.bar(
            df_rec,
            x="predicted_yield",
            y="crop",
            orientation="h",
            color="predicted_yield",
            color_continuous_scale="Greens",
            text="predicted_yield",
            labels={"predicted_yield": "Predicted Yield (Tonnes per Hectare)", "crop": "Crop Variety"}
        )
        fig_rec.update_layout(
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#1F2937"),
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False
        )
        fig_rec.update_traces(texttemplate="%{text:.2f} t/ha", textposition="outside")
        st.plotly_chart(fig_rec, use_container_width=True)

    with col_tbl:
        st.markdown("#### 📋 Detailed Yield Summary Table")
        df_display = df_rec.copy()
        df_display.columns = ["Crop Variety", "Predicted Yield (t/ha)", "Units"]
        df_display.index = range(1, len(df_display) + 1)
        st.dataframe(df_display, use_container_width=True)

        best_crop = recommended[0]["crop"]
        best_yield = recommended[0]["predicted_yield"]
        st.success(f"🌾 **Top Recommendation:** **{best_crop}** is predicted to produce the highest yield (**{best_yield:.2f} t/ha**) under your current climate and soil profile.")

    st.markdown(f"""
    <div class="disclaimer-card">
        <strong>⚠️ Advisory Notice:</strong><br>
        {farm_logic.MANDATORY_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 3: FARMER ASSISTANT (CHATBOT WITH MCP)
# ===========================================================================
elif selected_page == "🤖 3. Farmer Assistant (Chatbot)":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🤖 Kisan Mitra — Farmer Assistant</h1>
        <p class="hero-subtitle">Ask questions about crop yields, soil nutrients, or crop recommendations. Answers in simple English, Hindi (हिन्दी), or Marathi (मराठी).</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick prompt starter buttons
    st.markdown("##### 💬 Quick Questions to Ask:")
    q_cols = st.columns(3)
    with q_cols[0]:
        if st.button("🌾 What is my rice yield with 1150mm rain?"):
            st.session_state.chat_history.append({"role": "user", "content": "What is my expected rice yield with 1150 mm rainfall, 26.5 C temp, 72% humidity, pH 6.5, N 78, P 35, K 28?"})
            with st.spinner("Kisan Mitra is checking the model..."):
                reply = chatbot.ask_farmer_assistant(st.session_state.chat_history[-1]["content"], st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()
    with q_cols[1]:
        if st.button("🌱 Which crops give highest yield in my soil?"):
            st.session_state.chat_history.append({"role": "user", "content": "Which top crops will give highest yield with rainfall 1150 mm, temperature 26.5 C, humidity 72%, pH 6.5, N 78, P 35, K 28?"})
            with st.spinner("Kisan Mitra is evaluating crops..."):
                reply = chatbot.ask_farmer_assistant(st.session_state.chat_history[-1]["content"], st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()
    with q_cols[2]:
        if st.button("हिन्दी: गेहूँ की पैदावार कैसे बढ़ाएं?"):
            st.session_state.chat_history.append({"role": "user", "content": "नमस्ते किसान मित्र, गेहूँ (Wheat) की पैदावार बढ़ाने के लिए मिट्टी में क्या सुधार करें? (वर्षा 800mm, तापमान 22C, नमी 60%, pH 6.5, N 70, P 30, K 25)"})
            with st.spinner("किसान मित्र जवाब तैयार कर रहे हैं..."):
                reply = chatbot.ask_farmer_assistant(st.session_state.chat_history[-1]["content"], st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    st.write("")

    # Display Chat History
    for msg in st.session_state.chat_history:
        avatar = "👨‍🌾" if msg["role"] == "user" else "🌾"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat Input
    user_prompt = st.chat_input("Type your farming question here (English / हिन्दी / मराठी)...")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="👨‍🌾"):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar="🌾"):
            with st.spinner("Kisan Mitra is consulting agricultural tools..."):
                response_text = chatbot.ask_farmer_assistant(
                    user_prompt,
                    conversation_history=st.session_state.chat_history[:-1]
                )
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    # Clear Chat Button
    if st.session_state.chat_history:
        st.write("")
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ===========================================================================
# PAGE 4: ABOUT MODEL (FOR EVALUATORS & CAPSTONE VIVA)
# ===========================================================================
elif selected_page == "📖 4. About Model (for evaluators)":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">📖 Machine Learning Architecture & Evaluation</h1>
        <p class="hero-subtitle">Rigorous benchmark of Linear Regression vs Random Forest Regressor on identical held-out test partition (20%, n = 3,904 samples).</p>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic metrics computation (No hardcoded percentages)
    lr_mae = metrics_lr["MAE"]
    rf_mae = metrics_rf["MAE"]
    lr_rmse = metrics_lr["RMSE"]
    rf_rmse = metrics_rf["RMSE"]
    lr_r2 = metrics_lr["R2"]
    rf_r2 = metrics_rf["R2"]

    mae_diff_pct = ((lr_mae - rf_mae) / lr_mae) * 100
    rmse_diff_pct = ((lr_rmse - rf_rmse) / lr_rmse) * 100
    r2_diff = rf_r2 - lr_r2

    # Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">R² Score (Variance Explained)</div>
            <div class="metric-value">{rf_r2:.4f}</div>
            <div class="metric-sub">RF vs LR ({lr_r2:.4f}) &bull; +{r2_diff:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Absolute Error (MAE)</div>
            <div class="metric-value">{rf_mae:.2f} t/ha</div>
            <div class="metric-sub">{mae_diff_pct:.1f}% lower error than LR ({lr_mae:.2f})</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Root Mean Squared Error (RMSE)</div>
            <div class="metric-value">{rf_rmse:.2f} t/ha</div>
            <div class="metric-sub">{rmse_diff_pct:.1f}% lower error than LR ({lr_rmse:.2f})</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test Sample Size</div>
            <div class="metric-value">{metadata.get('test_sample_count', 3904):,}</div>
            <div class="metric-sub">Strict 20% Held-Out Split</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col_chart, col_rationale = st.columns([1.1, 1], gap="large")

    with col_chart:
        st.markdown("#### 📈 Benchmark Metric Comparison")
        metrics_df = pd.DataFrame({
            "Metric": ["MAE (t/ha)", "RMSE (t/ha)", "MSE", "R² Score"],
            "Linear Regression (Baseline)": [metrics_lr["MAE"], metrics_lr["RMSE"], metrics_lr["MSE"], metrics_lr["R2"]],
            "Random Forest (Champion)": [metrics_rf["MAE"], metrics_rf["RMSE"], metrics_rf["MSE"], metrics_rf["R2"]]
        })

        fig_m = go.Figure(data=[
            go.Bar(
                name='Linear Regression (Baseline)',
                x=metrics_df["Metric"],
                y=metrics_df["Linear Regression (Baseline)"],
                marker_color='#93C5FD',
                text=[f"{v:.2f}" for v in metrics_df["Linear Regression (Baseline)"]],
                textposition='outside'
            ),
            go.Bar(
                name='Random Forest (Champion)',
                x=metrics_df["Metric"],
                y=metrics_df["Random Forest (Champion)"],
                marker_color='#2E7D32',
                text=[f"{v:.2f}" for v in metrics_df["Random Forest (Champion)"]],
                textposition='outside'
            )
        ])
        fig_m.update_layout(
            barmode='group',
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1F2937'),
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_m, use_container_width=True)

    with col_rationale:
        st.markdown("#### 🧠 Agronomic & ML Rationale")
        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 18px; font-size: 0.92rem; line-height: 1.6;">
            <strong>1. Non-Linear Diminishing Returns:</strong> Biological crops follow non-linear growth curves. Extreme rainfall triggers root waterlogging, while excessive temperatures cause floral abortion. Decision trees partition these optimal zones naturally, whereas linear equations impose a rigid plane.<br><br>
            <strong>2. High-Order Nutrient Interactions:</strong> Soil pH dictates the bioavailability of Nitrogen, Phosphorus, and Potassium. Random Forest splits recursively on pH thresholds before testing nutrient values, capturing non-linear interactions without requiring manual feature engineering.<br><br>
            <strong>3. Zero Data Leakage:</strong> Standard scalers and One-Hot Encoders were fit strictly on the 80% training split. The 20% test partition remained completely unobserved during training.
        </div>
        """, unsafe_allow_html=True)

    # Actual vs Predicted Scatter Visualization
    st.markdown("---")
    st.markdown("#### 🎯 Actual vs. Predicted Yield on Held-Out Test Data")

    if test_preds is not None:
        model_view = st.radio(
            "Display predictions for:",
            ["Random Forest Regressor", "Linear Regression", "Both Models"],
            horizontal=True
        )

        fig_scatter = go.Figure()
        max_val = max(test_preds["Actual_Yield"].max(), test_preds["Predicted_RF"].max(), test_preds["Predicted_LR"].max())

        # 45-degree reference line
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="#DC2626", dash="dash", width=2),
            name="Ideal 1:1 Line (Perfect Fit)"
        ))

        if model_view in ["Random Forest Regressor", "Both Models"]:
            fig_scatter.add_trace(go.Scatter(
                x=test_preds["Actual_Yield"],
                y=test_preds["Predicted_RF"],
                mode="markers",
                marker=dict(color="#2E7D32", opacity=0.75, size=6),
                name="Random Forest",
                text=test_preds["crop_type"],
                hovertemplate="<b>Crop:</b> %{text}<br><b>Actual:</b> %{x:.2f} t/ha<br><b>Predicted:</b> %{y:.2f} t/ha<extra></extra>"
            ))

        if model_view in ["Linear Regression", "Both Models"]:
            fig_scatter.add_trace(go.Scatter(
                x=test_preds["Actual_Yield"],
                y=test_preds["Predicted_LR"],
                mode="markers",
                marker=dict(color="#2563EB", opacity=0.55, size=6),
                name="Linear Regression",
                text=test_preds["crop_type"],
                hovertemplate="<b>Crop:</b> %{text}<br><b>Actual:</b> %{x:.2f} t/ha<br><b>Predicted:</b> %{y:.2f} t/ha<extra></extra>"
            ))

        fig_scatter.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1F2937'),
            xaxis=dict(title="Actual Historical Yield (Tonnes per Hectare)", gridcolor="#F3F4F6"),
            yaxis=dict(title="Predicted Yield (Tonnes per Hectare)", gridcolor="#F3F4F6"),
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------------
    # Dynamic Sensitivity Simulation (Using Active User Inputs)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🧪 Dynamic Sensitivity Simulation (Trained RF Response)")
    st.markdown(
        f"Sweeps one variable across realistic bounds while holding all other inputs at your **current active session values** "
        f"(Crop: `{st.session_state.crop}`, Rain: `{st.session_state.rainfall:.0f} mm`, Temp: `{st.session_state.temperature:.1f} °C`, "
        f"Humidity: `{st.session_state.humidity:.0f} %`, pH: `{st.session_state.pH:.1f}`, N: `{st.session_state.N:.0f}`, P: `{st.session_state.P:.0f}`, K: `{st.session_state.K:.0f}`)."
    )

    sc1, sc2 = st.columns([1, 2.5], gap="large")
    with sc1:
        sens_param = st.selectbox(
            "Select Variable to Sweep:",
            ["Annual Rainfall (mm)", "Temperature (°C)", "Relative Humidity (%)", "Soil pH", "Nitrogen (N)"]
        )

    with sc2:
        param_configs = {
            "Annual Rainfall (mm)": ("rainfall", np.linspace(300, 3000, 30)),
            "Temperature (°C)": ("temperature", np.linspace(12, 42, 30)),
            "Relative Humidity (%)": ("humidity", np.linspace(30, 95, 30)),
            "Soil pH": ("pH", np.linspace(4.5, 8.5, 30)),
            "Nitrogen (N)": ("N", np.linspace(20, 180, 30))
        }

        col_name, sweep_vals = param_configs[sens_param]

        # Base vector uses the user's active inputs from session state!
        base_vector = {
            "rainfall": float(st.session_state.rainfall),
            "temperature": float(st.session_state.temperature),
            "humidity": float(st.session_state.humidity),
            "pH": float(st.session_state.pH),
            "N": float(st.session_state.N),
            "P": float(st.session_state.P),
            "K": float(st.session_state.K),
            "crop_type": st.session_state.crop
        }

        sim_rows = []
        for val in sweep_vals:
            row = base_vector.copy()
            row[col_name] = val
            sim_rows.append(row)

        sim_df = pd.DataFrame(sim_rows)
        sim_preds = rf_model.predict(sim_df)

        fig_sens = px.line(
            x=sweep_vals,
            y=sim_preds,
            labels={"x": sens_param, "y": "Predicted Yield (t/ha)"},
            title=f"Predicted Yield Sensitivity to {sens_param} for {st.session_state.crop}"
        )
        fig_sens.update_traces(line_color="#2E7D32", line_width=3)
        fig_sens.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1F2937'),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    # Dataset & Academic Limitations Tabs
    st.markdown("---")
    st.markdown("#### 📚 Academic Provenance & Execution Details")

    tab_prov, tab_setup, tab_limits = st.tabs(["1. Dataset Provenance", "2. Windows Run Guide", "3. Project Scope & Limitations"])

    with tab_prov:
        st.markdown("""
        - **Source:** Kaggle Agricultural Benchmark Dataset (`anshumish/crop-yield-data-with-soil-and-weather-dataset`).
        - **Primary Provenance:** Directorate of Economics and Statistics, Ministry of Agriculture & Farmers Welfare (Govt. of India), merged with regional India Meteorological Department (IMD) historical records.
        - **Observations:** 19,517 cleaned records covering 54 crop varieties across Indian states (1997–2020).
        - **Target Metric:** Historical Crop Yield in **Tonnes per Hectare (t/ha)** ($\text{Production in Tonnes} / \text{Area in Hectares}$).
        - **Data Cleaning Decision:** Coconut was isolated from the continuous tonne-based benchmark because Coconut yield is officially recorded in *thousands of nuts per hectare* rather than metric tonnes.
        """)

    with tab_setup:
        st.code("""# 1. Activate Windows Virtual Environment:
.\\venv\\Scripts\\activate

# 2. Install Dependencies:
pip install -r requirements.txt

# 3. Configure Chatbot API Key (Optional):
copy .env.example .env
# Open .env and add: LLM_API_KEY=your_actual_key

# 4. Run Automated ML & FastMCP Verification Suite:
python test_pipeline.py

# 5. Launch FastMCP Agricultural Server (Optional standalone):
python mcp_server.py

# 6. Launch the Streamlit Web Application:
streamlit run app.py
""", language="powershell")

    with tab_limits:
        st.markdown("""
        1. **Regional Macro-Data:** Climate and soil metrics represent state and district statistical aggregations. Field-level precision soil tests will exhibit localized micro-variances.
        2. **Unmodeled Exogenous Agronomic Factors:** Seed genetics (hybrid vs local cultivars), pest outbreaks, micro-irrigation schedules, and extreme weather hail shocks are outside tabular dataset scope.
        3. **Academic Capstone Scope:** Developed as an IT college viva demonstration and educational prototype, not a commercially certified agronomic prescription platform.
        """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.85rem; padding-bottom: 20px;'>"
    "Crop Yield Prediction using Weather and Soil Data &bull; Department of Information Technology &bull; Capstone Viva Project"
    "</div>",
    unsafe_allow_html=True
)
