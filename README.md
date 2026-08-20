# Biological Age Predictor & SHAP Biomarker Interpreter

An end-to-end clinical machine learning platform to predict chronological age from multi-system physiological biomarkers, quantify individual **Biological Age Acceleration Gaps** ($\Delta = \hat{Y}_{\text{predicted}} - Y_{\text{actual}}$), and interpret clinical longevity drivers using Tree-SHAP.

Trained and validated on **5,995 multi-ethnic participants (Ages 18–79)** from the post-pandemic **NHANES August 2021–August 2023** survey cycle.

---

## Key Results & Model Benchmarks

Four regression architectures were trained on 80% of the cohort ($N = 4,796$) using age-bracket stratification and evaluated on the held-out test set ($N = 1,199$):

| Model | Test MAE (Years) | Test RMSE (Years) | Test $R^2$ Score | Key Characteristics |
|:---|:---:|:---:|:---:|:---|
| **OLS Linear Regression** | 11.01 yrs | 13.29 yrs | 0.419 | Linear baseline |
| **Ridge Regression ($\alpha=10.0$)** | 11.01 yrs | 13.27 yrs | 0.421 | Regularized linear baseline |
| **Random Forest Regressor** | 9.86 yrs | 12.37 yrs | 0.497 | Non-linear tree ensemble |
| **XGBoost Regressor** | **9.65 yrs** | **11.93 yrs** | **0.532** | **Best Model** ($53.2\%$ aging variance explained) |

---

## Global Biomarker Drivers of Biological Aging (SHAP Analysis)

Using `shap.TreeExplainer`, global Shapley values quantify the exact marginal contribution (in years) of individual biomarkers to biological age acceleration:

| Rank | Biomarker | Feature Column | Mean \|SHAP\| (Years) | Clinical / Physiological Interpretation |
|:---:|:---|:---|:---:|:---|
| **1** | **Glycated Hemoglobin (HbA1c)** | `log_LBXGH` | **5.374 yrs** | **#1 Driver.** Advanced glycation end-products drive systemic vascular & metabolic aging. |
| **2** | **Waist-to-Height Ratio (WHtR)** | `WHtR` | **3.053 yrs** | **#2 Driver.** Central adiposity index ($\text{Waist} / \text{Height} > 0.55$) marks rapid age acceleration. |
| **3** | **Mean Corpuscular Volume (MCV)** | `LBXMCVSI` | **2.793 yrs** | Red blood cell volume reflects hematologic, vascular, and nutrient dynamics. |
| **4** | **Former Tobacco History** | `smoking_Former Smoker` | **1.604 yrs** | Cumulative tobacco exposure shifts baseline biological age upward. |
| **5** | **Serum Creatinine** | `log_LBXSCR` | **1.491 yrs** | Declining renal glomerular filtration rate accelerates biological age. |
| **6** | **Platelet Count** | `LBXPLTSI` | **1.369 yrs** | Hematologic aging dynamics and clotting regulation. |
| **7** | **Body Weight & BMI** | `BMXWT`, `BMXBMI` | **~1.23 yrs** | Metabolic load and excess adiposity burden. |
| **8** | **Poverty-Income Ratio (PIR)** | `INDFMPIR` | **1.114 yrs** | Socio-economic health disparities correlate with accelerated biological aging. |
| **9** | **Fasting Glucose** | `log_LBXSGL` | **1.061 yrs** | Short-term glycemic elevation contributes to metabolic age drift. |

---

## Repository Structure

```
biological-age-predictor/
├── app.py                      # Production Streamlit clinical web application
├── requirements.txt            # Python dependencies (xgboost, shap, streamlit, etc.)
├── notes.md                    # Technical decision log & cohort evolution
├── README.md                   # Project documentation & benchmark overview
├── data/
│   ├── raw/                    # Raw NHANES August 2021–August 2023 XPT survey files
│   └── processed/              # Processed train_data.csv and test_data.csv
├── models/
│   ├── final_model.pkl         # Serialized XGBoost Regressor model
│   └── scaler.pkl              # Fitted StandardScaler (leakage-safe)
├── notebooks/
│   ├── 02_eda.ipynb            # Exploratory Data Analysis & Skewness distributions
│   ├── 04_modeling.ipynb       # Model Training, Evaluation & Benchmarking
│   └── 05_shap_analysis.ipynb  # Global & Local SHAP Interpretability Analysis
└── src/
    ├── data_loader.py          # Modular 8-table XPT ingestion & inner join pipeline
    ├── eda_utils.py            # Reusable plotting & multicollinearity utilities
    ├── feature_engineering.py  # Log-transforms, WHtR, WHO activity, stratification
    ├── models.py               # Linear & tree-based regression training module
    └── explainability.py       # Tree-SHAP computation and attribution module
```

---

## Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/benchohrabdou/biological-age-predictor.git
cd biological-age-predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test custom biomarker profiles, inspect biological age scorecards, and view real-time personal SHAP waterfall breakdowns.

### 3. Re-run Pipelines from Terminal
```bash
# Ingest raw NHANES XPT surveys
python src/data_loader.py

# Run Feature Engineering pipeline (generates train_data.csv & test_data.csv)
python src/feature_engineering.py

# Train and benchmark models (generates models/final_model.pkl)
python src/models.py

# Compute SHAP feature rankings
python src/explainability.py
```

---

## Methodology Highlights

1. **Top-Coding Resolution**: Filtered top-coded participants aged 80+ to eliminate continuous regression target distortion and ceiling effects.
2. **Data Leakage Prevention**: `StandardScaler` is fitted strictly on the 80% training split and applied to transform the 20% test split.
3. **Subsample Missingness Awareness**: Fasting blood subsample structure (~10% missing) is explicitly tracked via `was_fasting_sample` binary indicators.
4. **Multicollinearity Cleanup**: Dropped redundant Hematocrit (`LBXHCT`, $r=0.971$ with Hemoglobin) and engineered clinically validated **Waist-to-Height Ratio (WHtR)**.