"""
app.py
Research Demonstration Platform for Biomarker-Based Age Prediction.

Features:
- Professional informatics dashboard layout (zero emojis, typography-driven UI).
- Real-time anthropometric profiling (Body Mass Index & Waist-to-Height Ratio).
- Gradient Boosted Machine Learning inference (XGBoost Regressor).
- Bias-corrected age gap computation via linear residualization from out-of-fold cross-validation.
- Native SHAP TreeExplainer feature decomposition and local waterfall attribution.
- Descriptive biomarker profile context (non-clinical, non-causal).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NHANES Biological Age Predictor (Research Demo)",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dashboard CSS Design ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header */
    .dashboard-header {
        background: #0B0F19;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 22px 28px;
        margin-bottom: 20px;
    }
    
    .dashboard-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    
    .dashboard-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        line-height: 1.5;
        font-weight: 400;
    }
    
    .meta-tag-container {
        display: flex;
        gap: 8px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    
    .meta-tag {
        background: #1E293B;
        color: #CBD5E1;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #334155;
    }
    
    /* Metric Scorecard Containers */
    .metric-panel {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 18px 20px;
        text-align: left;
        height: 100%;
    }
    
    .metric-label {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    
    .metric-unit {
        font-size: 0.95rem;
        font-weight: 500;
        color: #64748B;
        margin-left: 2px;
    }
    
    .status-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        margin-top: 10px;
        letter-spacing: 0.02em;
    }
    
    .badge-lower {
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }
    
    .badge-higher {
        background: rgba(239, 68, 68, 0.12);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.28);
    }
    
    .badge-typical {
        background: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.28);
    }
    
    /* Section Cards */
    .content-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .content-card-header {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94A3B8;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
        margin-bottom: 14px;
    }

    /* Structured List Items */
    .obs-item {
        border-left: 2px solid #334155;
        padding-left: 12px;
        margin-bottom: 12px;
    }
    
    .obs-item-high {
        border-left-color: #EF4444;
    }
    
    .obs-item-low {
        border-left-color: #10B981;
    }
    
    .obs-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 2px;
    }
    
    .obs-desc {
        font-size: 0.8rem;
        color: #94A3B8;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ── Model & Scaler Artifact Loading ──────────────────────────────────────────
@st.cache_resource
def load_artifacts() -> Tuple[Any, Any]:
    """Loads serialized model and scaler."""
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "models" / "final_model.pkl"
    scaler_path = project_root / "models" / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        st.error("Model artifacts missing. Run src/models.py first.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


@st.cache_resource
def load_correction_coefficients() -> Tuple[float, float]:
    """Loads linear age-gap bias correction coefficients (intercept, slope).

    Formula:
        expected_raw_gap = intercept + slope * age
        corrected_gap = raw_gap - expected_raw_gap
                      = raw_gap - (intercept + slope * age)
    """
    project_root = Path(__file__).resolve().parent
    for candidate in [
        project_root / "models" / "age_gap_correction.json",
        project_root / "results" / "age_gap_correction.json",
    ]:
        if candidate.exists():
            try:
                with open(candidate, "r") as f:
                    data = json.load(f)
                    return float(data.get("intercept", 26.6535)), float(data.get("slope", -0.5247))
            except Exception:
                pass
    # Fallback default from 5-fold cross-validation on XGBoost
    return 26.6535, -0.5247


model, scaler = load_artifacts()
bias_intercept, bias_slope = load_correction_coefficients()

# ── Feature Definitions ───────────────────────────────────────────────────────
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
    "has_lab_values", "sex_encoded", "pa_level_encoded",
    "smoking_Former Smoker", "smoking_Current Smoker"
]


# ── Sidebar Configuration & Benchmark Profiles ────────────────────────────────
with st.sidebar:
    st.markdown("### System Specifications")
    st.markdown("""
    **Core Engine:** XGBoost Regressor  
    **Evaluation:** 5-Fold Stratified CV  
    **Cross-Validation $R^2$:** `0.525 ± 0.016`  
    **Cross-Validation MAE:** `9.71 ± 0.15 Yrs`  
    **Cohort Source:** NHANES 2021–2023 (N = 5,995)  
    """)
    
    st.markdown("---")
    st.markdown("### Benchmark Profiles")
    preset_choice = st.selectbox(
        "Load Benchmark Profile:",
        [
            "Custom Parameters",
            "Profile A (Lower predicted age)",
            "Profile B (Higher predicted age)",
            "Median Profile (Age 50)"
        ]
    )
    
    st.markdown("---")
    st.caption("Research prototype for evaluating biomarker-based age prediction on NHANES survey data. Not a health assessment or clinical diagnostic tool.")


def get_profile_data(preset: str) -> Dict[str, Any]:
    if preset == "Profile A (Lower predicted age)":
        return {
            "age": 54, "sex": "Female", "height": 168.0, "weight": 60.0, "waist": 72.0,
            "glucose": 86.0, "hba1c": 5.0, "cholesterol": 180.0, "triglycerides": 80.0,
            "creatinine": 0.72, "uric_acid": 4.0, "calcium": 9.4, "alt": 16.0, "ggt": 15.0,
            "wbc": 5.2, "rbc": 4.5, "platelets": 235.0, "mcv": 87.0, "hemoglobin": 13.4,
            "sedentary": 280.0, "pa_level": "High (>300m/wk)", "smoking": "Never Smoker", "pir": 4.5
        }
    elif preset == "Profile B (Higher predicted age)":
        return {
            "age": 46, "sex": "Male", "height": 174.0, "weight": 104.0, "waist": 110.0,
            "glucose": 142.0, "hba1c": 7.2, "cholesterol": 235.0, "triglycerides": 270.0,
            "creatinine": 1.22, "uric_acid": 7.6, "calcium": 9.8, "alt": 46.0, "ggt": 62.0,
            "wbc": 8.6, "rbc": 5.1, "platelets": 305.0, "mcv": 95.0, "hemoglobin": 15.6,
            "sedentary": 640.0, "pa_level": "Low (<150m/wk)", "smoking": "Current Smoker", "pir": 1.8
        }
    else:
        return {
            "age": 50, "sex": "Male", "height": 175.0, "weight": 82.0, "waist": 92.0,
            "glucose": 100.0, "hba1c": 5.5, "cholesterol": 195.0, "triglycerides": 130.0,
            "creatinine": 0.95, "uric_acid": 5.5, "calcium": 9.5, "alt": 25.0, "ggt": 28.0,
            "wbc": 6.5, "rbc": 4.8, "platelets": 245.0, "mcv": 90.0, "hemoglobin": 14.8,
            "sedentary": 480.0, "pa_level": "Medium (150-300m/wk)", "smoking": "Never Smoker", "pir": 3.0
        }

p = get_profile_data(preset_choice)


# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">NHANES Biological Age Predictor</div>
    <div class="dashboard-subtitle">
        Predicting chronological age and exploring local SHAP feature attributions from routine NHANES clinical chemistry, hematology, and anthropometry.
    </div>
    <div class="meta-tag-container">
        <div class="meta-tag">COHORT N = 5,995</div>
        <div class="meta-tag">AGE SPAN: 18–79 YRS</div>
        <div class="meta-tag">MODEL: GRADIENT BOOSTED TREES (XGBOOST)</div>
        <div class="meta-tag">EXPLAINER: TREE-SHAP</div>
        <div class="meta-tag">CORRECTION: LINEAR RESIDUALIZATION (r = 0.000)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Visible research demo notice
st.info(
    "**Research Demonstration Only:** This application is a machine learning research prototype predicting chronological age "
    "from routine NHANES biomarkers. It is intended solely for methodological demonstration and exploratory data analysis. "
    "It is not a clinical assessment, medical diagnostic tool, or measure of personal health."
)


# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("biomarker_input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics & Body Metrics")
        age = st.slider("Chronological Age (Years)", 18, 79, int(p["age"]))
        sex = st.selectbox("Biological Sex", ["Male", "Female"], index=0 if p["sex"] == "Male" else 1)
        height = st.number_input("Standing Height (cm)", 130.0, 215.0, float(p["height"]), step=0.5)
        weight = st.number_input("Body Weight (kg)", 35.0, 200.0, float(p["weight"]), step=0.5)
        waist = st.number_input("Waist Circumference (cm)", 50.0, 160.0, float(p["waist"]), step=0.5)
        
        bmi_calc = weight / ((height / 100.0) ** 2)
        whtr_calc = waist / height
        st.caption(f"Calculated BMI: **{bmi_calc:.1f} kg/m²** | WHtR: **{whtr_calc:.3f}**")
        
    with col2:
        st.markdown("#### Metabolic & Serum Biochemistry")
        glucose = st.number_input("Serum Glucose (mg/dL)", 50.0, 350.0, float(p["glucose"]), step=1.0)
        hba1c = st.number_input("Glycohemoglobin HbA1c (%)", 4.0, 15.0, float(p["hba1c"]), step=0.1)
        cholesterol = st.number_input("Total Serum Cholesterol (mg/dL)", 80.0, 400.0, float(p["cholesterol"]), step=1.0)
        triglycerides = st.number_input("Serum Triglycerides (mg/dL)", 30.0, 700.0, float(p["triglycerides"]), step=1.0)
        creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.3, 5.0, float(p["creatinine"]), step=0.01)
        uric_acid = st.number_input("Serum Uric Acid (mg/dL)", 1.5, 14.0, float(p["uric_acid"]), step=0.1)
        calcium = st.number_input("Serum Total Calcium (mg/dL)", 7.0, 13.0, float(p["calcium"]), step=0.1)
        alt = st.number_input("Alanine Aminotransferase ALT (U/L)", 4.0, 200.0, float(p["alt"]), step=1.0)
        ggt = st.number_input("Gamma-Glutamyl Transferase GGT (U/L)", 4.0, 300.0, float(p["ggt"]), step=1.0)
        
    with col3:
        st.markdown("#### Hematology & Behavioral Metrics")
        wbc = st.number_input("White Blood Cell Count (10³/µL)", 2.0, 25.0, float(p["wbc"]), step=0.1)
        rbc = st.number_input("Red Blood Cell Count (10⁶/µL)", 2.0, 7.5, float(p["rbc"]), step=0.01)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 8.0, 20.0, float(p["hemoglobin"]), step=0.1)
        platelets = st.number_input("Platelet Count (10³/µL)", 80.0, 700.0, float(p["platelets"]), step=1.0)
        mcv = st.number_input("Mean Corpuscular Volume (fL)", 60.0, 120.0, float(p["mcv"]), step=0.1)
        
        smoking = st.selectbox(
            "Smoking History",
            ["Never Smoker", "Former Smoker", "Current Smoker"],
            index=["Never Smoker", "Former Smoker", "Current Smoker"].index(p["smoking"])
        )
        pa_level = st.selectbox(
            "Physical Activity Category",
            ["Low (<150m/wk)", "Medium (150-300m/wk)", "High (>300m/wk)"],
            index=["Low (<150m/wk)", "Medium (150-300m/wk)", "High (>300m/wk)"].index(p["pa_level"])
        )
        sedentary = st.slider("Sedentary Duration (min/day)", 60, 900, int(p["sedentary"]))
        pir = st.slider("Poverty-Income Ratio (PIR)", 0.0, 5.0, float(p["pir"]), step=0.1)

    submitted = st.form_submit_button("Predict Age", use_container_width=True)


# ── Inference Execution & Display ─────────────────────────────────────────────
if submitted or preset_choice != "Custom Parameters":
    # 1. Feature Engineering
    log_LBXSCR = np.log1p(creatinine)
    log_LBXSGL = np.log1p(glucose)
    log_LBXSTR = np.log1p(triglycerides)
    log_LBXGH = np.log1p(hba1c)
    log_LBXSGTSI = np.log1p(ggt)
    log_LBXSATSI = np.log1p(alt)
    log_PAD680 = np.log1p(min(sedentary, 720.0))

    pa_min_map = {"Low (<150m/wk)": 60.0, "Medium (150-300m/wk)": 210.0, "High (>300m/wk)": 450.0}
    total_pa_min_wk = pa_min_map[pa_level]

    sex_encoded = 1 if sex == "Male" else 0
    pa_map = {"Low (<150m/wk)": 0, "Medium (150-300m/wk)": 1, "High (>300m/wk)": 2}
    pa_level_encoded = pa_map[pa_level]
    smoking_Former = 1 if smoking == "Former Smoker" else 0
    smoking_Current = 1 if smoking == "Current Smoker" else 0

    raw_continuous_vals = [
        log_LBXSCR, log_LBXSGL, log_LBXSTR, log_LBXGH,
        log_LBXSGTSI, log_LBXSATSI, log_PAD680, whtr_calc,
        total_pa_min_wk, hemoglobin, cholesterol, uric_acid, calcium,
        wbc, rbc, platelets, mcv,
        height, weight, waist, bmi_calc
    ]
    
    scaled_continuous_vals = scaler.transform([raw_continuous_vals])[0]
    scaled_dict = dict(zip(CONTINUOUS_SCALING_COLS, scaled_continuous_vals))

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
        "has_lab_values": 1,
        "sex_encoded": sex_encoded,
        "pa_level_encoded": pa_level_encoded,
        "smoking_Former Smoker": smoking_Former,
        "smoking_Current Smoker": smoking_Current,
    }

    input_df = pd.DataFrame([input_row])[MODEL_FEATURE_COLS]

    # Predict Chronological Age with Model
    predicted_bio_age = float(model.predict(input_df)[0])
    raw_gap = predicted_bio_age - float(age)

    # Compute Bias-Corrected Age Gap using linear residualization
    # Delta_expected = intercept + slope * age = 26.65 - 0.525 * age
    expected_gap = bias_intercept + bias_slope * float(age)
    corrected_gap = raw_gap - expected_gap

    # ── Scorecard Section ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Prediction Summary")

    mcol1, mcol2, mcol3 = st.columns(3)

    with mcol1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-label">Chronological Age</div>
            <div class="metric-value">{age:.1f}<span class="metric-unit">YRS</span></div>
            <div class="status-badge" style="background:#1E293B; color:#94A3B8; border:1px solid #334155;">Reference Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol2:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-label">Predicted Age</div>
            <div class="metric-value" style="color: #38BDF8;">{predicted_bio_age:.1f}<span class="metric-unit">YRS</span></div>
            <div class="status-badge" style="background: rgba(56, 189, 248, 0.12); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3);">Model Regression Output</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol3:
        if corrected_gap > 2.5:
            gap_color = "#EF4444"
            badge_cls = "badge-higher"
            badge_lbl = f"Predicted age is higher than typical for this age (+{corrected_gap:.1f} Yrs)"
            sign_str = "+"
        elif corrected_gap < -2.5:
            gap_color = "#10B981"
            badge_cls = "badge-lower"
            badge_lbl = f"Predicted age is lower than typical for this age ({corrected_gap:.1f} Yrs)"
            sign_str = ""
        else:
            gap_color = "#F59E0B"
            badge_cls = "badge-typical"
            badge_lbl = f"Predicted age is close to typical for this age ({corrected_gap:+.1f} Yrs)"
            sign_str = "+" if corrected_gap >= 0 else ""

        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-label">Bias-Corrected Age Gap</div>
            <div class="metric-value" style="color: {gap_color};">{sign_str}{corrected_gap:.1f}<span class="metric-unit">YRS</span></div>
            <div class="status-badge {badge_cls}">{badge_lbl}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Raw gap: {raw_gap:+.1f} yrs | Linear age-bias correction: {-expected_gap:+.1f} yrs")

    # ── SHAP Waterfall Plot ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Local Feature Attribution (Tree-SHAP Decomposition)")
    st.caption("Decomposition of marginal contributions (in years) by individual biomarkers relative to the cohort expected value.")

    explainer = shap.TreeExplainer(model)
    shap_val = explainer(input_df)[0]

    # Style Matplotlib Dark Palette
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0B0F19')
    ax.set_facecolor('#0B0F19')
    
    shap.plots.waterfall(shap_val, max_display=10, show=False)
    plt.title(f"Local Biomarker Attribution (SHAP) — Predicted Age: {predicted_bio_age:.1f} Yrs", fontsize=11, color="#E2E8F0", fontweight="600", pad=14)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Biomarker Context Section ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Biomarker Profile Context")
    
    obs_col1, obs_col2 = st.columns(2)
    
    with obs_col1:
        st.markdown("""
        <div class="content-card">
            <div class="content-card-header">Biomarkers Associated with Higher Predicted Age</div>
        """, unsafe_allow_html=True)
        
        higher_found = False
        if whtr_calc > 0.55:
            higher_found = True
            st.markdown(f"""
            <div class="obs-item obs-item-high">
                <div class="obs-title">Waist-to-Height Ratio ({whtr_calc:.2f} > 0.55)</div>
                <div class="obs-desc">Waist-to-height ratio is elevated. In this cohort, higher central adiposity is associated with older chronological age.</div>
            </div>
            """, unsafe_allow_html=True)
            
        if hba1c >= 5.7:
            higher_found = True
            st.markdown(f"""
            <div class="obs-item obs-item-high">
                <div class="obs-title">Glycated Hemoglobin HbA1c ({hba1c:.1f}% ≥ 5.7%)</div>
                <div class="obs-desc">HbA1c level is in the elevated range. Glycated hemoglobin is the strongest predictor of older age in the model.</div>
            </div>
            """, unsafe_allow_html=True)
            
        if smoking in ["Current Smoker", "Former Smoker"]:
            higher_found = True
            st.markdown(f"""
            <div class="obs-item obs-item-high">
                <div class="obs-title">Smoking History ({smoking})</div>
                <div class="obs-desc">Participant reports a history of smoking. In this cohort, former smokers average ~10 years older than never-smokers.</div>
            </div>
            """, unsafe_allow_html=True)
            
        if sedentary > 540:
            higher_found = True
            st.markdown(f"""
            <div class="obs-item obs-item-high">
                <div class="obs-title">Sedentary Duration ({sedentary / 60:.1f} hrs/day)</div>
                <div class="obs-desc">Reported sedentary duration is above cohort average, which correlates positively with predicted age.</div>
            </div>
            """, unsafe_allow_html=True)
            
        if not higher_found:
            st.markdown("""
            <div class="obs-item" style="border-left-color: #10B981;">
                <div class="obs-title">No Elevating Factors Identified</div>
                <div class="obs-desc">Primary metabolic and hematologic features fall within typical baseline levels for this age bracket.</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    with obs_col2:
        st.markdown("""
        <div class="content-card">
            <div class="content-card-header">Biomarkers Associated with Lower Predicted Age</div>
        """, unsafe_allow_html=True)
        
        if pa_level == "High (>300m/wk)":
            st.markdown("""
            <div class="obs-item obs-item-low">
                <div class="obs-title">High Physical Activity Volume (>300 min/wk)</div>
                <div class="obs-desc">Active weekly physical activity is associated with lower predicted age in the model.</div>
            </div>
            """, unsafe_allow_html=True)
            
        if whtr_calc <= 0.50:
            st.markdown(f"""
            <div class="obs-item obs-item-low">
                <div class="obs-title">Lower Waist-to-Height Ratio (WHtR: {whtr_calc:.2f})</div>
                <div class="obs-desc">Waist-to-height ratio falls in the lower reference range (&le; 0.50), associated with lower predicted age.</div>
            </div>
            """, unsafe_allow_html=True)
            
        if hba1c < 5.4:
            st.markdown(f"""
            <div class="obs-item obs-item-low">
                <div class="obs-title">Low Glycated Hemoglobin (HbA1c: {hba1c:.1f}%)</div>
                <div class="obs-desc">HbA1c level is in the lower range, associated with younger model age predictions.</div>
            </div>
            """, unsafe_allow_html=True)
            
        if smoking == "Never Smoker":
            st.markdown("""
            <div class="obs-item obs-item-low">
                <div class="obs-title">Never Smoker Status</div>
                <div class="obs-desc">Absence of smoking history is associated with lower predicted age relative to former and current smokers.</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
