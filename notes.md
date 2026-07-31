# Project Notes

## Cohort Selection & Sample Size

The final dataset size of **6,337 participants** across **128 features** is derived from the initial NHANES August 2021–August 2023 demographic respondents using sequential inner joins across 8 distinct surveys.

Here is the step-by-step cohort size and feature count evolution as the tables were merged:

| Step | Dataset Added | Individual Row Count | Merged Row Count (Intersection) | Total Columns (Cumulative) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Demographics** (`DEMO_L.xpt`) | 11,933 | 11,933 | 27 |
| 2 | **Body Measures** (`BMX_L.xpt`) | 8,860 | 8,860 | 48 |
| 3 | **Biochemistry** (`BIOPRO_L.xpt`) | 7,199 | 7,199 | 89 |
| 4 | **Glycohemoglobin** (`GHB_L.xpt`) | 7,199 | 7,199 | 90 |
| 5 | **Total Cholesterol** (`TCHOL_L.xpt`) | 8,068 | 7,199 | 92 |
| 6 | **Smoking Status** (`SMQ_L.xpt`) | 9,015 | 7,199 | 100 |
| 7 | **Complete Blood Count** (`CBC_L.xpt`) | 8,727 | 7,199 | 121 |
| 8 | **Physical Activity** (`PAQ_L.xpt`) | 8,153 | **6,337** | **128** |

### Key Ingestion Details:
* **Fasting/Lab Selection Limit**: Lab examinations (such as biochemistry and glycohemoglobin) are performed only on a subsample of participants, which drops the available rows from 11,933 to 7,199.
* **Questionnaire Drop**: The physical activity questionnaire (`PAQ_L.xpt`) drops the sample size from 7,199 to 6,337 because some participants did not complete this questionnaire interview.
* **Inner Join Tradeoff**: Using an inner join ensures a clean dataset with full records across all variables, eliminating the need for missing-value imputation in model features. However, it results in a smaller cohort compared to the initial demographic count.

---

## Exploratory Data Analysis Findings

### 2.1 — Univariate Exploration

**Skewness summary (flagging threshold: |skew| > 3 → log-transform required)**

| Biomarker | Column | Skew | Missing % | Feature Engineering Action |
| :--- | :--- | :--- | :--- | :--- |
| Creatinine | `LBXSCR` | ~15.6 | 10.7% | Log-transform (most extreme skew in dataset) |
| Sedentary Time | `PAD680` | ~11.1 | 0.1% | Log-transform + cap outliers at 95th percentile |
| Fasting Glucose | `LBXSGL` | ~6.3 | 10.2% | Log-transform |
| Triglycerides | `LBXSTR` | ~4.4 | 10.2% | Log-transform |
| HbA1c | `LBXGH` | ~3.7 | 5.3% | Log-transform |
| BMI | `BMXBMI` | ~1.2 | 1.6% | Borderline skew, monitor in Phase 3 |

**Data quality issues found:**
- **Planned Lab Subsample Missingness**: Lab-based biomarkers (glucose, creatinine, cholesterol, HbA1c) all show ~10% missingness — caused by NHANES only drawing fasting blood from a subsample, not random data loss. This informs our imputation strategy in Phase 3 (missingness itself may reflect fasting compliance).
- **Age Top-Coding at 80 (`RIDAGEYR`)**: Anyone aged 80+ is recorded as exactly 80, creating an artificial point mass spike. Feature engineering decision: either filter participants to `< 80` or explicitly handle 80+ as a censored category.
- **Sedentary Time Artifacts (`PAD680`)**: Implausible self-reported values (> 1,000 min/day = 16+ hrs) reflect survey rounding artifacts. Requires log-transformation and percentile capping.

---

### 2.2 — Correlation Analysis

**Method note:** Applied log-transforms identified in Section 2.1 *before* computing correlations, since Pearson $r$ is sensitive to outliers and skew. Log-transforming HbA1c boosted its linear correlation with age from $r = +0.263$ (raw) to $r = +0.312$ (log-transformed).

**Top correlations with chronological age (`RIDAGEYR`):**
| Biomarker | Column | Pearson $r$ | Direction |
| :--- | :--- | :---: | :--- |
| log(HbA1c) | `log_LBXGH` | +0.312 | ↑ with age |
| MCV (red cell size) | `LBXMCVSI` | +0.263 | ↑ with age |
| log(Fasting Glucose) | `log_LBXSGL` | +0.241 | ↑ with age |
| log(Creatinine) | `log_LBXSCR` | +0.231 | ↑ with age |
| Waist Circumference | `BMXWAIST` | +0.200 | ↑ with age |
| RBC Count | `LBXRBCSI` | −0.190 | ↓ with age |
| Platelet Count | `LBXPLTSI` | −0.172 | ↓ with age |

**Interpretation:** No single biomarker is strongly predictive alone (all $|r| < 0.35$). This supports the core premise of biological age modeling: age prediction requires a multivariate model, and SHAP analysis in Phase 5 will reveal how complex biomarker combinations interact.

**Multicollinearity Decision Rules ($|r| \ge 0.70$):**
- **$0.70 \le |r| < 0.90$:** Retain for initial modeling or consolidate into ratio features.
- **$|r| \ge 0.90$:** Candidate for dropping one of the redundant features to prevent unstable linear coefficients and SHAP importance splitting.

**Specific Multicollinear Feature Pairs Found:**
| Feature 1 | Feature 2 | $r$ | Feature Engineering Action |
| :--- | :--- | :---: | :--- |
| Hemoglobin (`LBXHGB`) | Hematocrit (`LBXHCT`) | **+0.971** | **Drop Candidate ($\ge 0.90$)**: Keep Hemoglobin, drop Hematocrit due to direct physiological redundancy. |
| BMI (`BMXBMI`) | Waist Circumference (`BMXWAIST`) | **+0.900** | **Drop/Feature Candidate ($\ge 0.90$)**: Engineer Waist-to-Height Ratio (WHtR) or retain Waist Circumference (stronger age correlation). |
| Weight (`BMXWT`) | Waist Circumference (`BMXWAIST`) | **+0.898** | High body mass collinearity; consolidate in feature engineering. |
| BMI (`BMXBMI`) | Weight (`BMXWT`) | **+0.890** | High body mass collinearity; consolidate in feature engineering. |
| RBC Count (`LBXRBCSI`) | Hematocrit (`LBXHCT`) | **+0.794** | Retain or combine in ratio ($0.70 - 0.85$). |
| Fasting Glucose (`log_LBXSGL`) | HbA1c (`log_LBXGH`) | **+0.777** | Retain both ($0.70 - 0.85$); short-term vs long-term glycemic control. Tree models handle both well. |

---

### 2.3 — Subgroup Exploration

**1. Smoking Status (`SMQ020` / `SMQ040`):**
- **Demographic & Age Confounding:** Former smokers are significantly older on average (**59.6 yrs**) than Never smokers (**49.3 yrs**) and Current smokers (**52.5 yrs**).
- **Direct Biological Signal vs. Age Artifact:**
  - *Direct Signal:* Active smoking directly drives acute systemic inflammation — Current smokers show markedly elevated White Blood Cell counts (WBC median **7.5** vs **6.4** in Never smokers), despite being 7.1 years *younger* than Former smokers.
  - *Age Artifact:* Elevated Creatinine (0.95 vs 0.87 mg/dL) and Waist Circumference (103.8 vs 99.0 cm) in Former smokers are primarily artifacts of being **10.3 years older** on average.
- **Feature Engineering Action:** Encode 3-category `smoking_status` and construct `Smoking \times Age` interaction terms to decouple true biological acceleration from chronological age.

**2. Sex (`RIAGENDR`):**
- **Demographic Balance:** Age is perfectly balanced across sexes (Female mean age 52.2 vs. Male 52.5 yrs, difference < 0.4 yrs). Differences are **genuine biological baseline dimorphisms**, not age artifacts.
- **Biomarker Dimorphisms:**
  - Hemoglobin (`LBXHGB`): Male median **14.8 g/dL** vs. Female **13.3 g/dL** (androgen-stimulated erythropoiesis).
  - Creatinine (`LBXSCR`): Male median **0.97 mg/dL** vs. Female **0.75 mg/dL** (muscle mass breakdown).
  - Platelets (`LBXPLTSI`): Female median **265** vs. Male **234** $\times 10^3/\mu\text{L}$.
- **Feature Engineering Action:** Include `Sex` as a predictor feature or apply sex-standardized z-scores so normal female baseline creatinine isn't misclassified as "younger biological age".

**3. Physical Activity Level (`total_pa_min_wk`):**
- **WHO / CDC Bucketing:** $\text{Total Moderate-Equivalent} = \text{Moderate} + 2 \times \text{Vigorous}$. Low (<150m/wk: 47.5%), Medium (150-300m/wk: 19.4%), High (>300m/wk: 33.1%).
- **Age Confounding:** Low activity group is **5.4 years older** on average than High activity group (54.6 vs. 49.2 yrs). Reduced activity in older adults is partially driven by age-related mobility decline.
- **Biomarker Signals:** High Activity correlates with lower Waist Circumference (94.7 vs. 102.6 cm), lower BMI (26.8 vs. 29.7), lower HbA1c (5.4% vs. 5.6%), and lower WBC count (6.4 vs. 6.9 $\times 10^3/\mu\text{L}$).

---

### Feature Engineering & Modeling Plan

1. **Transformations Mandate:** Apply $\log(1+x)$ to `LBXSCR`, `LBXSGL`, `LBXSTR`, `LBXGH`, `PAD680`, `LBXSGTSI`, and `LBXSATSI`. Cap `PAD680` at the 95th percentile.
2. **Multicollinearity Clean-up:** Drop `LBXHCT` (redundant with `LBXHGB` $r=0.971$). Create Waist-to-Height Ratio (`WHtR = BMXWAIST / BMXHT`) and drop raw weight/BMI.
3. **Sex Standardization:** Create sex-adjusted z-scores for dimorphic markers (`LBXHGB`, `LBXSCR`).
4. **Interaction Features:** Construct `smoking_status \times Age` and `pa_level \times Age` interaction terms to isolate true biological acceleration from chronological age confounding.
5. **Target Censoring:** Handle top-coded age observations (`RIDAGEYR == 80`).
