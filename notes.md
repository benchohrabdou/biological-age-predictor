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

## Phase 2 — EDA Findings

### 2.1 — Univariate Exploration

**Skewness summary (flagging threshold: |skew| > 3 → log-transform required)**

| Biomarker | Column | Skew | Missing % | Action for Phase 3 |
| :--- | :--- | :--- | :--- | :--- |
| Creatinine | `LBXSCR` | ~15.6 | 10.7% | Log-transform (most extreme skew in dataset) |
| Sedentary Time | `PAD680` | ~11.1 | 0.1% | Log-transform + cap outliers at 95th percentile |
| Fasting Glucose | `LBXSGL` | ~6.3 | 10.2% | Log-transform |
| Triglycerides | `LBXSTR` | ~4.4 | 10.2% | Log-transform |
| HbA1c | `LBXGH` | ~3.7 | 5.3% | Log-transform |
| BMI | `BMXBMI` | ~1.2 | 1.6% | Borderline skew, monitor in Phase 3 |

**Data quality issues found:**
- **Planned Lab Subsample Missingness**: Lab-based biomarkers (glucose, creatinine, cholesterol, HbA1c) all show ~10% missingness — caused by NHANES only drawing fasting blood from a subsample, not random data loss. This informs our imputation strategy in Phase 3 (missingness itself may reflect fasting compliance).
- **Age Top-Coding at 80 (`RIDAGEYR`)**: Anyone aged 80+ is recorded as exactly 80, creating an artificial point mass spike. Decision needed for Phase 3: either filter participants to `< 80` or explicitly handle 80+ as a censored category.
- **Sedentary Time Artifacts (`PAD680`)**: Implausible self-reported values (> 1,000 min/day = 16+ hrs) reflect survey rounding artifacts. Will require log-transformation and capping at the 95th percentile in Phase 3.

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
- **$0.70 \le |r| < 0.90$:** Retain for initial modeling or consolidate into ratio features during Phase 3.
- **$|r| \ge 0.90$:** Candidate for dropping one of the redundant features to prevent unstable linear coefficients and SHAP importance splitting.

**Specific Multicollinear Feature Pairs Found:**
| Feature 1 | Feature 2 | $r$ | Phase 3 Decision |
| :--- | :--- | :---: | :--- |
| Hemoglobin (`LBXHGB`) | Hematocrit (`LBXHCT`) | **+0.971** | **Drop Candidate ($\ge 0.90$)**: Keep Hemoglobin, drop Hematocrit due to direct physiological redundancy. |
| BMI (`BMXBMI`) | Waist Circumference (`BMXWAIST`) | **+0.900** | **Drop/Feature Candidate ($\ge 0.90$)**: Engineer Waist-to-Height Ratio (WHtR) or retain Waist Circumference (stronger age correlation). |
| Weight (`BMXWT`) | Waist Circumference (`BMXWAIST`) | **+0.898** | High body mass collinearity; consolidate in Phase 3. |
| BMI (`BMXBMI`) | Weight (`BMXWT`) | **+0.890** | High body mass collinearity; consolidate in Phase 3. |
| RBC Count (`LBXRBCSI`) | Hematocrit (`LBXHCT`) | **+0.794** | Retain or combine in ratio ($0.70 - 0.85$). |
| Fasting Glucose (`log_LBXSGL`) | HbA1c (`log_LBXGH`) | **+0.777** | Retain both ($0.70 - 0.85$); short-term vs long-term glycemic control. Tree models handle both well. |

