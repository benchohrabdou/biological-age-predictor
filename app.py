"""
app.py
Production-grade Streamlit Web Application for Biological Age Prediction & SHAP Interpretability.

Features:
- Interactive Biomarker Input Dashboard with clinical presets.
- Live Anthropometric calculation (BMI & Waist-to-Height Ratio).
- Machine Learning inference using trained XGBoost Regressor (models/final_model.pkl).
- Real-time SHAP Waterfall explanation deconstructing the user's Biological Age Gap.
- Actionable, personalized longevity and lifestyle recommendations.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# ── Page Configuration & Theming ──────────────────────────────────────────────
st.set_page_config(
    page_title="Biological Age Predictor | NHANES AI Clock",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Sleek Dark Glassmorphism Design ─────────────────────────────
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Container */
    .hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    /* Scorecard Container */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .gap-youthful {
        color: #34D399 !important;
        font-weight: 700;
    }
    
    .gap-accelerated {
        color: #F87171 !important;
        font-weight: 700;
    }
    
    .gap-neutral {
        color: #FBBF24 !important;
        font-weight: 700;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
    }
    
    .badge-green {
        background: rgba(52, 211, 153, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .badge-red {
        background: rgba(248, 113, 113, 0.15);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    
    .badge-amber {
        background: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ── Model & Scaler Loading ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts() -> Tuple[Any, Any]:
    """Loads the trained XGBoost model and fitted StandardScaler."""
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "models" / "final_model.pkl"
    scaler_path = project_root / "models" / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        st.error("Model artifacts missing. Please run src/models.py first.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_artifacts()


# ── Named Feature Columns ────────────────────────────────────────────────────
CONTINUOUS_SCALING_COLS = [
    "log_LBXSCR", "log_LBXSGL", "log_LBXSTR", "log_LBXGH",
    "log_LBXSGTSI", "log_LBXSATSI", "log_PAD680", "WHtR",
    "total_pa_min_wk", "LBXHGB", "LBXTC", "LBXSUA", "LBXSCA",
    "LBXWBCSI", "LBXRBCSI", "LBXPLTSI", "LBXMCVSI",
    "BMXHT", "BMXWT", "BMXWAIST", "BMXBMI",
]

MODEL_FEATURE_COLS = [
    "log_LBXSCR", "log_LBXSGL", "log_LBXSTR", "log_LBXGH",
    "log_LBXSGTSI", "log_LBXSATSI", "log_PAD680", "WHtR",
    "total_pa_min_wk", "LBXHGB", "LBXTC", "LBXSUA", "LBXSCA",
    "LBXWBCSI", "LBXRBCSI", "LBXPLTSI", "LBXMCVSI",
    "BMXHT", "BMXWT", "BMXWAIST", "BMXBMI", "INDFMPIR",
    "was_fasting_sample", "sex_encoded", "pa_level_encoded",
    "smoking_Former Smoker", "smoking_Current Smoker"
]


# ── Sidebar Configuration & Presets ──────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=64)
    st.title("NHANES AI Clock")
    st.caption("Trained on 5,995 Multi-Ethnic NHANES Participants (2021–2023)")
    
    st.markdown("---")
    st.subheader("⚡ Quick Clinical Presets")
    preset_choice = st.selectbox(
        "Load Sample Profile:",
        ["Custom Input", "Healthy / Active Longevity Profile", "Metabolic / High-Risk Profile", "Average Baseline (Age 50)"]
    )
    
    st.markdown("---")
    st.markdown("""
    **Model Performance:**
    - **Architecture:** XGBoost Regressor
    - **Test Set $R^2$:** `0.532` (53.2% Variance)
    - **Test Set MAE:** `9.65 Years`
    """)


# ── Preset Values Helper ──────────────────────────────────────────────────────
def get_defaults(preset: str) -> Dict[str, Any]:
    if preset == "Healthy / Active Longevity Profile":
        return {
            "age": 52, "sex": "Female", "height": 168.0, "weight": 62.0, "waist": 74.0,
            "glucose": 88.0, "hba1c": 5.1, "cholesterol": 185.0, "triglycerides": 85.0,
            "creatinine": 0.72, "uric_acid": 4.2, "calcium": 9.4, "alt": 18.0, "ggt": 16.0,
            "wbc": 5.4, "rbc": 4.6, "platelets": 240.0, "mcv": 88.0, "hemoglobin": 13.5,
            "sedentary": 300.0, "pa_level": "High (>300m/wk)", "smoking": "Never Smoker", "pir": 4.5
        }
    elif preset == "Metabolic / High-Risk Profile":
        return {
            "age": 48, "sex": "Male", "height": 175.0, "weight": 105.0, "waist": 112.0,
            "glucose": 145.0, "hba1c": 7.4, "cholesterol": 240.0, "triglycerides": 280.0,
            "creatinine": 1.25, "uric_acid": 7.8, "calcium": 9.8, "alt": 48.0, "ggt": 65.0,
            "wbc": 8.8, "rbc": 5.2, "platelets": 310.0, "mcv": 96.0, "hemoglobin": 15.8,
            "sedentary": 660.0, "pa_level": "Low (<150m/wk)", "smoking": "Current Smoker", "pir": 1.8
        }
    else:
        return {
            "age": 50, "sex": "Male", "height": 175.0, "weight": 82.0, "waist": 92.0,
            "glucose": 100.0, "hba1c": 5.5, "cholesterol": 195.0, "triglycerides": 130.0,
            "creatinine": 0.95, "uric_acid": 5.5, "calcium": 9.5, "alt": 25.0, "ggt": 28.0,
            "wbc": 6.5, "rbc": 4.8, "platelets": 245.0, "mcv": 90.0, "hemoglobin": 14.8,
            "sedentary": 480.0, "pa_level": "Medium (150-300m/wk)", "smoking": "Never Smoker", "pir": 3.0
        }

d = get_defaults(preset_choice)


# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Biological Age Predictor & SHAP Biomarker Interpreter</div>
    <div class="hero-subtitle">
        Quantify your physiological biological age and deconstruct the exact clinical biomarker drivers of longevity using explainable machine learning.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("biomarker_form"):
    st.subheader("📋 Enter Clinical Biomarkers & Lifestyle Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🧬 Demographics & Anthropometrics")
        age = st.slider("Chronological Age (Years)", 18, 79, int(d["age"]))
        sex = st.selectbox("Biological Sex", ["Male", "Female"], index=0 if d["sex"] == "Male" else 1)
        height = st.number_input("Height (cm)", 130.0, 215.0, float(d["height"]), step=0.5)
        weight = st.number_input("Weight (kg)", 35.0, 200.0, float(d["weight"]), step=0.5)
        waist = st.number_input("Waist Circumference (cm)", 50.0, 160.0, float(d["waist"]), step=0.5)
        
        # Computed anthropometrics
        bmi = weight / ((height / 100.0) ** 2)
        whtr = waist / height
        st.caption(f"📊 **Live BMI:** `{bmi:.1f}` kg/m² | **Waist-to-Height Ratio (WHtR):** `{whtr:.3f}`")
        
    with col2:
        st.markdown("##### 🩸 Metabolic & Organ Function")
        glucose = st.number_input("Fasting Glucose (mg/dL)", 50.0, 350.0, float(d["glucose"]), step=1.0)
        hba1c = st.number_input("Glycohemoglobin HbA1c (%)", 4.0, 15.0, float(d["hba1c"]), step=0.1)
        cholesterol = st.number_input("Total Cholesterol (mg/dL)", 80.0, 400.0, float(d["cholesterol"]), step=1.0)
        triglycerides = st.number_input("Triglycerides (mg/dL)", 30.0, 700.0, float(d["triglycerides"]), step=1.0)
        creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.3, 5.0, float(d["creatinine"]), step=0.01)
        uric_acid = st.number_input("Serum Uric Acid (mg/dL)", 1.5, 14.0, float(d["uric_acid"]), step=0.1)
        calcium = st.number_input("Total Calcium (mg/dL)", 7.0, 13.0, float(d["calcium"]), step=0.1)
        alt = st.number_input("ALT Liver Enzyme (U/L)", 4.0, 200.0, float(d["alt"]), step=1.0)
        ggt = st.number_input("GGT Liver Enzyme (U/L)", 4.0, 300.0, float(d["ggt"]), step=1.0)
        
    with col3:
        st.markdown("##### 🧪 Complete Blood Count (CBC) & Lifestyle")
        wbc = st.number_input("White Blood Cell Count (10³/µL)", 2.0, 25.0, float(d["wbc"]), step=0.1)
        rbc = st.number_input("Red Blood Cell Count (10⁶/µL)", 2.0, 7.5, float(d["rbc"]), step=0.01)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 8.0, 20.0, float(d["hemoglobin"]), step=0.1)
        platelets = st.number_input("Platelet Count (10³/µL)", 80.0, 700.0, float(d["platelets"]), step=1.0)
        mcv = st.number_input("Mean Corpuscular Volume (fL)", 60.0, 120.0, float(d["mcv"]), step=0.1)
        
        st.markdown("---")
        smoking = st.selectbox(
            "Smoking Status",
            ["Never Smoker", "Former Smoker", "Current Smoker"],
            index=["Never Smoker", "Former Smoker", "Current Smoker"].index(d["smoking"])
        )
        pa_level = st.selectbox(
            "Physical Activity Level",
            ["Low (<150m/wk)", "Medium (150-300m/wk)", "High (>300m/wk)"],
            index=["Low (<150m/wk)", "Medium (150-300m/wk)", "High (>300m/wk)"].index(d["pa_level"])
        )
        sedentary = st.slider("Sedentary Time (min/day)", 60, 900, int(d["sedentary"]))
        pir = st.slider("Poverty Income Ratio (PIR)", 0.0, 5.0, float(d["pir"]), step=0.1)

    submitted = st.form_submit_button("🚀 Calculate Biological Age & Generate Report", use_container_width=True)


# ── Feature Vector Assembly & Inference ───────────────────────────────────────
if submitted or preset_choice != "Custom Input":
    # 1. Feature transformations
    log_LBXSCR = np.log1p(creatinine)
    log_LBXSGL = np.log1p(glucose)
    log_LBXSTR = np.log1p(triglycerides)
    log_LBXGH = np.log1p(hba1c)
    log_LBXSGTSI = np.log1p(ggt)
    log_LBXSATSI = np.log1p(alt)
    log_PAD680 = np.log1p(min(sedentary, 720.0))  # 95th percentile cap

    # Physical activity weekly minutes approximation
    pa_min_map = {"Low (<150m/wk)": 60.0, "Medium (150-300m/wk)": 210.0, "High (>300m/wk)": 450.0}
    total_pa_min_wk = pa_min_map[pa_level]

    # Encodings
    sex_encoded = 1 if sex == "Male" else 0
    pa_map = {"Low (<150m/wk)": 0, "Medium (150-300m/wk)": 1, "High (>300m/wk)": 2}
    pa_level_encoded = pa_map[pa_level]
    smoking_Former = 1 if smoking == "Former Smoker" else 0
    smoking_Current = 1 if smoking == "Current Smoker" else 0

    # Build unscaled continuous feature vector
    raw_continuous_vals = [
        log_LBXSCR, log_LBXSGL, log_LBXSTR, log_LBXGH,
        log_LBXSGTSI, log_LBXSATSI, log_PAD680, whtr,
        total_pa_min_wk, hemoglobin, cholesterol, uric_acid, calcium,
        wbc, rbc, platelets, mcv,
        height, weight, waist, bmi
    ]
    
    # Scale continuous features using fitted StandardScaler
    scaled_continuous_vals = scaler.transform([raw_continuous_vals])[0]
    scaled_dict = dict(zip(CONTINUOUS_SCALING_COLS, scaled_continuous_vals))

    # Assemble complete model input DataFrame
    input_row = {
        "log_LBXSCR": scaled_dict["log_LBXSCR"],
        "log_LBXSGL": scaled_dict["log_LBXSGL"],
        "log_LBXSTR": scaled_dict["log_LBXSTR"],
        "log_LBXGH": scaled_dict["log_LBXGH"],
        "log_LBXSGTSI": scaled_dict["log_LBXSGTSI"],
        "log_LBXSATSI": scaled_dict["log_LBXSATSI"],
        "log_PAD680": scaled_dict["log_PAD680"],
        "WHtR": scaled_dict["WHtR"],
        "total_pa_min_wk": scaled_dict["total_pa_min_wk"],
        "LBXHGB": scaled_dict["LBXHGB"],
        "LBXTC": scaled_dict["LBXTC"],
        "LBXSUA": scaled_dict["LBXSUA"],
        "LBXSCA": scaled_dict["LBXSCA"],
        "LBXWBCSI": scaled_dict["LBXWBCSI"],
        "LBXRBCSI": scaled_dict["LBXRBCSI"],
        "LBXPLTSI": scaled_dict["LBXPLTSI"],
        "LBXMCVSI": scaled_dict["LBXMCVSI"],
        "BMXHT": scaled_dict["BMXHT"],
        "BMXWT": scaled_dict["BMXWT"],
        "BMXWAIST": scaled_dict["BMXWAIST"],
        "BMXBMI": scaled_dict["BMXBMI"],
        "INDFMPIR": pir,
        "was_fasting_sample": 1,
        "sex_encoded": sex_encoded,
        "pa_level_encoded": pa_level_encoded,
        "smoking_Former Smoker": smoking_Former,
        "smoking_Current Smoker": smoking_Current,
    }

    input_df = pd.DataFrame([input_row])[MODEL_FEATURE_COLS]

    # Model Inference
    predicted_bio_age = float(model.predict(input_df)[0])
    age_gap = predicted_bio_age - float(age)

    # ── Scorecard Section ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Biological Age Assessment Scorecard")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.95rem; font-weight: 500;">Chronological Age</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: #F1F5F9; margin-top: 4px;">{age:.1f} <span style="font-size: 1.1rem; color: #64748B;">Years</span></div>
            <div class="badge badge-amber">Passport Age</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.95rem; font-weight: 500;">Predicted Biological Age</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">{predicted_bio_age:.1f} <span style="font-size: 1.1rem; color: #64748B;">Years</span></div>
            <div class="badge badge-green">Physiological Clock</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        if age_gap < -2.5:
            gap_class = "gap-youthful"
            badge_class = "badge-green"
            badge_text = "Decelerated Aging (Youthful)"
            sign = ""
        elif age_gap > 2.5:
            gap_class = "gap-accelerated"
            badge_class = "badge-red"
            badge_text = "Accelerated Aging (Elevated Risk)"
            sign = "+"
        else:
            gap_class = "gap-neutral"
            badge_class = "badge-amber"
            badge_text = "Synchronous (Normal Baseline)"
            sign = "+" if age_gap >= 0 else ""

        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.95rem; font-weight: 500;">Biological Age Gap (Δ)</div>
            <div class="{gap_class}" style="font-size: 2.4rem; margin-top: 4px;">{sign}{age_gap:.1f} <span style="font-size: 1.1rem;">Years</span></div>
            <div class="badge {badge_class}">{badge_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP Real-Time Waterfall Explanation ──────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Personal SHAP Waterfall Explanation")
    st.markdown("""
    This waterfall chart explains **why** the model produced your specific biological age. It shows how each biomarker added (red) or subtracted (blue) years from the population baseline age.
    """)

    explainer = shap.TreeExplainer(model)
    shap_val = explainer(input_df)[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_val, max_display=10, show=False)
    plt.title(f"Personal Biomarker Impact on Biological Age (Gap: {sign}{age_gap:.1f} Yrs)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Personalized Recommendations ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Targeted Longevity & Biomarker Recommendations")
    
    rec_cols = st.columns(2)
    
    with rec_cols[0]:
        st.markdown("##### ⚠️ Top Potential Risk Drivers to Address")
        risks = []
        if whtr > 0.55:
            risks.append(f"**Waist-to-Height Ratio (`{whtr:.2f}` > 0.55):** Visceral adiposity accelerates metabolic aging. Aim to reduce waist circumference to below {height * 0.5:.0f} cm.")
        if hba1c >= 5.7:
            risks.append(f"**HbA1c (`{hba1c:.1f}%` ≥ 5.7%):** Pre-diabetic glycation range. Optimizing dietary glycemic load and post-meal walks can lower biological age acceleration.")
        if smoking in ["Current Smoker", "Former Smoker"]:
            risks.append(f"**Smoking History ({smoking}):** Tobacco exposure increases systemic inflammatory markers (WBC count). Smoking cessation lowers active inflammatory age drift.")
        if sedentary > 540:
            risks.append(f"**Sedentary Time (`{sedentary / 60:.1f}` hrs/day):** Prolonged inactivity slows lipid and glucose clearance. Incorporate standing desks and active micro-breaks.")
        
        if risks:
            for r in risks:
                st.warning(r)
        else:
            st.success("No major clinical biomarker risk elevations detected! Keep maintaining your current regimen.")

    with rec_cols[1]:
        st.markdown("##### 🛡️ Protective Longevity Assets")
        strengths = []
        if pa_level == "High (>300m/wk)":
            strengths.append("**High Physical Activity Volume:** Meeting >300 moderate-equivalent min/wk confers strong cardiovascular and metabolic youthfulness.")
        if whtr <= 0.50:
            strengths.append(f"**Optimal Waist-to-Height Ratio (`{whtr:.2f}`):** Low visceral fat is one of the strongest protective buffers against premature cellular aging.")
        if hba1c < 5.4:
            strengths.append(f"**Excellent Glycemic Regulation (HbA1c `{hba1c:.1f}%`):** Minimal advanced glycation end-products preserving endothelial vascular health.")
        if smoking == "Never Smoker":
            strengths.append("**Never Smoker Status:** Avoided cumulative pulmonary and vascular oxidative burden.")
            
        if strengths:
            for s in strengths:
                st.info(s)
        else:
            st.info("Continue building consistency across physical activity and metabolic regulation.")
