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
