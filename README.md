# Biomarker-Based Age Prediction & SHAP Interpretation

Predicting chronological age from routine clinical biomarkers, estimating each person's age gap (predicted age − chronological age), and examining which biomarkers the model relies on, using Tree-SHAP.

**Data:** 5,995 adults aged 18–79 from the NHANES August 2021–August 2023 cycle (U.S. National Health and Nutrition Examination Survey), built from 8 linked survey tables.

**Scope:** The model is trained to predict chronological age. The age gap is a common starting point for biological-age research, but it is only meaningful as a marker of biological aging once it is corrected for its known dependence on age and shown to relate to health outcomes. See [Limitations](#limitations).

---

## Results

Four regression models were trained on 80% of the cohort (N = 4,796, split stratified by age bracket) and evaluated on the held-out 20% (N = 1,199).

| Model | Test MAE (years) | Test RMSE (years) | Test R² |
|:---|:---:|:---:|:---:|
| Linear regression (OLS) | 11.01 | 13.29 | 0.419 |
| Ridge regression (α = 10) | 11.01 | 13.27 | 0.421 |
| Random forest | 9.86 | 12.37 | 0.497 |
| XGBoost | **9.65** | **11.93** | **0.532** |

The tree-based models outperform the linear baselines, consistent with non-linear relationships and interactions between biomarkers. No single biomarker correlates strongly with age on its own (all |r| < 0.35), so age prediction relies on combining many weak signals.

---

## Which biomarkers does the model rely on? (SHAP)

Global feature importance from `shap.TreeExplainer` on the XGBoost model, computed on the test set (N = 1,199). Mean |SHAP| is the average absolute contribution of a feature to the predicted age, in years.

SHAP describes how the model uses each feature to predict chronological age. It shows which biomarkers are most informative about age in this cohort; it does not show that a biomarker causes aging.

| Rank | Feature | Column | Mean \|SHAP\| (years) | Context |
|:---:|:---|:---|:---:|:---|
| 1 | Glycated hemoglobin (HbA1c) | `log_LBXGH` | 5.37 | HbA1c rises with age in this cohort (r = +0.31 after log transform). |
| 2 | Waist-to-height ratio | `WHtR` | 3.05 | Central adiposity tends to increase with age. |
| 3 | Mean corpuscular volume | `LBXMCVSI` | 2.79 | Red blood cell volume increases with age (r = +0.26). |
| 4 | Former smoker | `smoking_Former Smoker` | 1.60 | Former smokers are ~10 years older on average than never-smokers here, so this feature partly acts as an age proxy. |
| 5 | Creatinine | `log_LBXSCR` | 1.49 | Rises with age, consistent with declining kidney function; strongly sex-dependent. |
| 6 | Platelet count | `LBXPLTSI` | 1.37 | Declines with age (r = −0.17). |
| 7 | Body weight | `BMXWT` | 1.24 | Related to adiposity measures above. |
| 8 | BMI | `BMXBMI` | 1.22 | Highly correlated with weight and waist (r ≈ 0.9), so importance is shared among them. |
| 9 | Income-to-poverty ratio | `INDFMPIR` | 1.11 | Socio-economic factor; association may reflect confounding. |
| 10 | Fasting glucose | `log_LBXSGL` | 1.06 | Correlated with HbA1c (r = +0.78). |

---

## Methodology

- **Cohort construction.** 8 NHANES tables (demographics, body measures, biochemistry, glycohemoglobin, cholesterol, smoking, complete blood count, physical activity) merged with inner joins: 11,933 → 6,337 participants. Laboratory tests are run on a subsample and not everyone completed the physical activity questionnaire, which explains most of the reduction. Details in [notes.md](notes.md).
- **Top-coded age.** NHANES records everyone aged 80+ as exactly 80, which distorts a regression target. These participants were removed: 6,337 → 5,995.
- **Skewed biomarkers.** log(1 + x) applied to heavily skewed variables (e.g. creatinine, fasting glucose, triglycerides, HbA1c); sedentary time additionally capped at the 95th percentile.
- **Collinearity.** Hematocrit dropped (r = 0.97 with hemoglobin); waist-to-height ratio added as an adiposity measure.
- **Planned missingness.** Fasting laboratory values exist only for a subsample; this is tracked with a `was_fasting_sample` indicator rather than treated as random missingness.
- **Leakage prevention.** Train/test split before any fitting; StandardScaler fitted on the training set only.

---

## Limitations

- **Single train/test split.** Metrics come from one split, without confidence intervals.
- **Age-gap bias.** A model trained to predict age regresses toward the mean: it tends to over-predict young people and under-predict older people, so the raw age gap is negatively correlated with age. The gap should be corrected for age before being interpreted.
- **No outcome validation.** The age gap has not yet been tested against health outcomes, so it should not be read as a validated measure of biological aging.
- **SHAP is descriptive.** Feature importances describe the model, not biological mechanisms, and some features (e.g. smoking status) are confounded with age.
- **Survey design.** NHANES sampling weights are not used, so results describe this sample rather than the U.S. population.
- **Cross-sectional data.** One measurement per person; no individual aging trajectories.

---

## Interactive demo

A Streamlit app lets you enter a biomarker profile and see the predicted age together with a per-person SHAP waterfall explanation. It is a research prototype and is not intended for clinical use.

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

---

## Repository structure

```
biological-age-predictor/
├── app.py                      # Streamlit demo
├── requirements.txt
├── notes.md                    # Decision log: cohort construction, EDA, feature choices
├── README.md
├── data/
│   ├── raw/                    # NHANES 2021–2023 XPT files
│   └── processed/              # train_data.csv, test_data.csv
├── models/
│   ├── final_model.pkl         # Trained XGBoost model
│   └── scaler.pkl              # StandardScaler fitted on training data
├── notebooks/
│   ├── 02_eda.ipynb            # Exploratory data analysis
│   ├── 04_modeling.ipynb       # Training and evaluation
│   └── 05_shap_analysis.ipynb  # SHAP analysis
└── src/
    ├── data_loader.py          # Loading and merging the 8 NHANES tables
    ├── eda_utils.py            # Plotting and collinearity utilities
    ├── feature_engineering.py  # Transforms, WHtR, activity levels, train/test split
    ├── models.py               # Model training and evaluation
    └── explainability.py       # SHAP computation
```

---

## Reproducing the results

```bash
git clone https://github.com/benchohrabdou/biological-age-predictor.git
cd biological-age-predictor

python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

pip install -r requirements.txt

python src/data_loader.py          # load and merge NHANES tables
python src/feature_engineering.py  # build train_data.csv and test_data.csv
python src/models.py               # train and evaluate models
python src/explainability.py        # compute SHAP values
```