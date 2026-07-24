"""
src/feature_engineering.py
Production-standard feature engineering pipeline for the Biological Age Predictor.

Modules included:
- handle_age_topcoding: Filters top-coded 80+ age observations.
- handle_missing_data: Constructs subsample indicator and imputes lab biomarkers.
- engineer_composite_features: Log-transforms, ratio engineering, and physical activity bucketing.
- encode_and_scale: One-hot/ordinal encoding and continuous scaling.
- train_test_split: Stratified train/test splitting and saving to data/processed/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Named Constants ───────────────────────────────────────────────────────────
AGE_TOPCODE_LIMIT: int = 80
FASTING_INDICATOR_COL: str = "was_fasting_sample"
DEFAULT_RANDOM_STATE: int = 42

# List of lab biomarkers with subsample missingness (~10%)
LAB_BIOMARKERS: List[str] = [
    "LBXSGL",    # Fasting Glucose
    "LBXGH",     # HbA1c
    "LBXTC",     # Total Cholesterol
    "LBXSTR",    # Triglycerides
    "LBXSCR",    # Creatinine
    "LBXSUA",    # Uric Acid
    "LBXSCA",    # Calcium
    "LBXSATSI",  # ALT
    "LBXSGTSI",  # GGT
]

# All numeric biomarkers needing missing value check & median imputation
NUMERIC_BIOMARKER_COLS: List[str] = [
    "BMXBMI", "BMXWT", "BMXHT", "BMXWAIST",
    "LBXSGL", "LBXGH", "LBXTC", "LBXSTR",
    "LBXSCR", "LBXSUA", "LBXSCA", "LBXSATSI", "LBXSGTSI",
    "LBXWBCSI", "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXPLTSI", "LBXMCVSI",
    "PAD680",
]


def handle_age_topcoding(
    df: pd.DataFrame, 
    age_col: str = "RIDAGEYR", 
    max_age: int = AGE_TOPCODE_LIMIT
) -> pd.DataFrame:
    """Filters out top-coded age observations (RIDAGEYR >= 80) to eliminate continuous target distortion.

    Note:
        Restricts the biological age prediction model scope to participants aged 18–79 years.

    Args:
        df (pd.DataFrame): Raw merged NHANES DataFrame.
        age_col (str): Name of the chronological age column (default 'RIDAGEYR').
        max_age (int): Upper bound cutoff age (default 80).

    Returns:
        pd.DataFrame: Filtered DataFrame containing participants aged < max_age.

    Raises:
        KeyError: If age_col is not present in df.
    """
    if age_col not in df.columns:
        raise KeyError(f"Age column '{age_col}' not found in DataFrame.")

    initial_rows = len(df)
    df_filtered = df[df[age_col] < max_age].copy()
    dropped_rows = initial_rows - len(df_filtered)

    logger.info(
        f"Filtered top-coded age observations ({age_col} >= {max_age}): "
        f"dropped {dropped_rows} rows ({dropped_rows / initial_rows:.2%}). "
        f"New shape: {df_filtered.shape}"
    )

    return df_filtered


def handle_missing_data(
    df: pd.DataFrame,
    lab_cols: Optional[List[str]] = None,
    impute_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Creates a fasting subsample indicator and median-imputes missing numeric biomarker values.

    Args:
        df (pd.DataFrame): DataFrame with top-coded ages filtered out.
        lab_cols (Optional[List[str]]): List of lab biomarker columns to define fasting subsample.
        impute_cols (Optional[List[str]]): List of numeric columns to median-impute.

    Returns:
        pd.DataFrame: Imputed DataFrame with the binary 'was_fasting_sample' indicator feature added.
    """
    lab_cols = lab_cols or LAB_BIOMARKERS
    impute_cols = impute_cols or NUMERIC_BIOMARKER_COLS

    df_out = df.copy()

    # 1. Create binary indicator feature: was_fasting_sample
    existing_lab_cols = [c for c in lab_cols if c in df_out.columns]
    if existing_lab_cols:
        df_out[FASTING_INDICATOR_COL] = df_out[existing_lab_cols].notna().any(axis=1).astype(int)
        fasting_count = df_out[FASTING_INDICATOR_COL].sum()
        logger.info(
            f"Constructed indicator '{FASTING_INDICATOR_COL}': "
            f"{fasting_count}/{len(df_out)} participants ({fasting_count / len(df_out):.1%}) in fasting subsample."
        )

    # 2. Median Imputation for continuous biomarker features
    existing_impute_cols = [c for c in impute_cols if c in df_out.columns]
    missing_before = df_out[existing_impute_cols].isnull().sum().sum()

    if missing_before > 0:
        imputer = SimpleImputer(strategy="median")
        df_out[existing_impute_cols] = imputer.fit_transform(df_out[existing_impute_cols])
        missing_after = df_out[existing_impute_cols].isnull().sum().sum()
        logger.info(
            f"Median imputation completed across {len(existing_impute_cols)} biomarker features. "
            f"Missing values imputed: {missing_before} -> {missing_after}."
        )
    else:
        logger.info("No missing values detected in specified biomarker columns. Skipping imputation.")

    return df_out
