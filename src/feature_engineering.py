"""
src/feature_engineering.py
Production-standard feature engineering pipeline for the Biological Age Predictor.

Modules included:
- handle_age_topcoding: Filters top-coded 80+ age observations.
- handle_missing_data: Constructs subsample indicator and imputes lab biomarkers.
- engineer_composite_features: Log-transforms, ratio engineering, and physical activity bucketing.
- encode_and_scale: One-hot/ordinal encoding and continuous scaling.
- split_and_save_data: Stratified train/test splitting and saving to data/processed/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Heavily right-skewed features requiring log-transform (|skew| > 3)
SKEWED_FEATURES: List[str] = [
    "LBXSCR",    # Creatinine
    "LBXSGL",    # Fasting Glucose
    "LBXSTR",    # Triglycerides
    "LBXGH",     # HbA1c
    "LBXSGTSI",  # GGT
    "LBXSATSI",  # ALT
    "PAD680",    # Sedentary Time
]

# List of lab biomarkers with subsample missingness (~10%)
LAB_BIOMARKERS: List[str] = [
    "LBXSGL", "LBXGH", "LBXTC", "LBXSTR",
    "LBXSCR", "LBXSUA", "LBXSCA", "LBXSATSI", "LBXSGTSI",
]

# All numeric biomarkers needing missing value check & median imputation
NUMERIC_BIOMARKER_COLS: List[str] = [
    "BMXBMI", "BMXWT", "BMXHT", "BMXWAIST",
    "LBXSGL", "LBXGH", "LBXTC", "LBXSTR",
    "LBXSCR", "LBXSUA", "LBXSCA", "LBXSATSI", "LBXSGTSI",
    "LBXWBCSI", "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXPLTSI", "LBXMCVSI",
    "PAD680", "INDFMPIR",
]

# Continuous features to standardize (mean=0, std=1)
CONTINUOUS_SCALING_COLS: List[str] = [
    "log_LBXSCR", "log_LBXSGL", "log_LBXSTR", "log_LBXGH",
    "log_LBXSGTSI", "log_LBXSATSI", "log_PAD680", "WHtR",
    "total_pa_min_wk", "LBXHGB", "LBXTC", "LBXSUA", "LBXSCA",
    "LBXWBCSI", "LBXRBCSI", "LBXPLTSI", "LBXMCVSI",
    "BMXHT", "BMXWT", "BMXWAIST", "BMXBMI",
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


def _clean_pa_sentinel(val: float) -> float:
    """Helper to replace NHANES sentinel response codes (7777, 9999) with 0.0."""
    if pd.isna(val) or val in [7777, 9999, 77777, 99999]:
        return 0.0
    return float(val)


def _calc_weekly_minutes(row: pd.Series, qty_col: str, unit_col: str, dur_col: str) -> float:
    """Helper to convert NHANES frequency & duration into total weekly minutes."""
    q = _clean_pa_sentinel(row.get(qty_col, 0))
    u = row.get(unit_col, None)
    d = _clean_pa_sentinel(row.get(dur_col, 0))

    if q <= 0 or d <= 0:
        return 0.0

    mult_map = {"D": 7.0, "W": 1.0, "M": 1.0 / 4.33, "Y": 1.0 / 52.0}
    multiplier = mult_map.get(u, 0.0)
    return q * d * multiplier


def engineer_composite_features(
    df: pd.DataFrame,
    skewed_cols: Optional[List[str]] = None,
    drop_redundant_hematocrit: bool = True,
    cap_sedentary_95th: bool = True,
) -> pd.DataFrame:
    """Engineers log-transformed, ratio, categorical, and physical activity features.

    Operations performed:
    1. Log-transforms right-skewed biomarkers using log1p.
    2. Caps sedentary time (PAD680) at the 95th percentile.
    3. Drops Hematocrit (LBXHCT) due to r = 0.971 redundancy with Hemoglobin (LBXHGB).
    4. Engineers Waist-to-Height Ratio (WHtR = BMXWAIST / BMXHT).
    5. Derives WHO physical activity metrics (total_pa_min_wk and pa_level).
    6. Derives categorical smoking_status and sex_label.

    Args:
        df (pd.DataFrame): Input DataFrame (imputed, age-filtered).
        skewed_cols (Optional[List[str]]): List of columns to log-transform.
        drop_redundant_hematocrit (bool): Whether to drop LBXHCT (default True).
        cap_sedentary_95th (bool): Whether to cap PAD680 at 95th percentile (default True).

    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    skewed_cols = skewed_cols or SKEWED_FEATURES
    df_out = df.copy()

    # 1. Cap Sedentary Time (PAD680) at 95th percentile to handle self-report rounding artifacts
    if cap_sedentary_95th and "PAD680" in df_out.columns:
        p95 = df_out["PAD680"].quantile(0.95)
        n_capped = (df_out["PAD680"] > p95).sum()
        df_out["PAD680"] = df_out["PAD680"].clip(upper=p95)
        logger.info(f"Capped Sedentary Time (PAD680) at 95th percentile ({p95:.1f} min/day): {n_capped} values capped.")

    # 2. Log-transform skewed biomarkers
    for col in skewed_cols:
        if col in df_out.columns:
            log_col_name = f"log_{col}"
            df_out[log_col_name] = np.log1p(df_out[col].clip(lower=0))
            logger.info(f"Created log-transformed feature '{log_col_name}' from '{col}'.")

    # 3. Drop redundant Hematocrit (LBXHCT)
    if drop_redundant_hematocrit and "LBXHCT" in df_out.columns:
        df_out.drop(columns=["LBXHCT"], inplace=True)
        logger.info("Dropped redundant feature 'LBXHCT' (Hematocrit) to resolve r = 0.971 multicollinearity with Hemoglobin.")

    # 4. Ratio Feature Engineering: Waist-to-Height Ratio (WHtR)
    if "BMXWAIST" in df_out.columns and "BMXHT" in df_out.columns:
        df_out["WHtR"] = df_out["BMXWAIST"] / df_out["BMXHT"]
        logger.info(f"Engineered 'WHtR' (Waist-to-Height Ratio): mean = {df_out['WHtR'].mean():.3f}, std = {df_out['WHtR'].std():.3f}.")

    # 5. Deriving Physical Activity Metrics (WHO / CDC guideline formula)
    if all(c in df_out.columns for c in ["PAD790Q", "PAD790U", "PAD800", "PAD810Q", "PAD810U", "PAD820"]):
        mod_min = df_out.apply(lambda r: _calc_weekly_minutes(r, "PAD790Q", "PAD790U", "PAD800"), axis=1)
        vig_min = df_out.apply(lambda r: _calc_weekly_minutes(r, "PAD810Q", "PAD810U", "PAD820"), axis=1)
        df_out["total_pa_min_wk"] = mod_min + 2.0 * vig_min

        def _bucket_pa(m: float) -> str:
            if m < 150.0:
                return "Low (<150m/wk)"
            elif m <= 300.0:
                return "Medium (150-300m/wk)"
            else:
                return "High (>300m/wk)"

        df_out["pa_level"] = df_out["total_pa_min_wk"].apply(_bucket_pa)
        logger.info(f"Engineered Physical Activity features 'total_pa_min_wk' and 'pa_level'. Breakdown:\n{df_out['pa_level'].value_counts().to_dict()}")

    # 6. Deriving Categorical Subgroups (Smoking & Sex)
    if "SMQ020" in df_out.columns and "SMQ040" in df_out.columns:
        def _categorize_smoking(r: pd.Series) -> str:
            if r["SMQ020"] == 2.0:
                return "Never Smoker"
            elif r["SMQ040"] in [1.0, 2.0]:
                return "Current Smoker"
            elif r["SMQ040"] == 3.0:
                return "Former Smoker"
            else:
                return "Never Smoker"  # Default fallback for 9 missing/refused cases

        df_out["smoking_status"] = df_out.apply(_categorize_smoking, axis=1)
        logger.info(f"Derived 'smoking_status': {df_out['smoking_status'].value_counts().to_dict()}")

    if "RIAGENDR" in df_out.columns:
        sex_map = {1.0: "Male", 2.0: "Female"}
        df_out["sex_label"] = df_out["RIAGENDR"].map(sex_map)
        logger.info(f"Derived 'sex_label': {df_out['sex_label'].value_counts().to_dict()}")

    return df_out


def encode_and_scale(
    df: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Encodes categorical variables and standardizes continuous biomarker features.

    Categorical Encoding Scheme:
    - sex_label: Binary encoding (Male=1, Female=0).
    - pa_level: Ordinal encoding (Low=0, Medium=1, High=2).
    - smoking_status: One-hot encoding with 'Never Smoker' as reference baseline.

    Args:
        df (pd.DataFrame): Input DataFrame (imputed & feature-engineered).
        continuous_cols (Optional[List[str]]): List of continuous columns to standardize.
        scaler (Optional[StandardScaler]): Pre-fitted StandardScaler instance (for test data).

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: (Encoded & scaled DataFrame, fitted StandardScaler instance).
    """
    continuous_cols = continuous_cols or CONTINUOUS_SCALING_COLS
    df_out = df.copy()

    # 1. Binary Encoding: sex_label (Male=1, Female=0)
    if "sex_label" in df_out.columns:
        df_out["sex_encoded"] = (df_out["sex_label"] == "Male").astype(int)
        logger.info(f"Binary encoded 'sex_label' -> 'sex_encoded' (Male=1, Female=0): {df_out['sex_encoded'].value_counts().to_dict()}")

    # 2. Ordinal Encoding: pa_level (Low=0, Medium=1, High=2)
    if "pa_level" in df_out.columns:
        pa_map = {
            "Low (<150m/wk)": 0,
            "Medium (150-300m/wk)": 1,
            "High (>300m/wk)": 2,
        }
        df_out["pa_level_encoded"] = df_out["pa_level"].map(pa_map).fillna(0).astype(int)
        logger.info(f"Ordinal encoded 'pa_level' -> 'pa_level_encoded': {df_out['pa_level_encoded'].value_counts().to_dict()}")

    # 3. One-Hot Encoding: smoking_status (Never Smoker as reference baseline)
    if "smoking_status" in df_out.columns:
        smoking_dummies = pd.get_dummies(df_out["smoking_status"], prefix="smoking", drop_first=False, dtype=int)
        if "smoking_Never Smoker" in smoking_dummies.columns:
            smoking_dummies.drop(columns=["smoking_Never Smoker"], inplace=True)
        df_out = pd.concat([df_out, smoking_dummies], axis=1)
        logger.info(f"One-hot encoded 'smoking_status': generated columns {list(smoking_dummies.columns)}.")

    # 4. Continuous Feature Standardization (StandardScaler)
    existing_continuous = [c for c in continuous_cols if c in df_out.columns]

    if scaler is None:
        scaler = StandardScaler()
        df_out[existing_continuous] = scaler.fit_transform(df_out[existing_continuous])
        logger.info(f"Fitted and applied StandardScaler on {len(existing_continuous)} continuous features.")
    else:
        df_out[existing_continuous] = scaler.transform(df_out[existing_continuous])
        logger.info(f"Applied pre-fitted StandardScaler on {len(existing_continuous)} continuous features.")

    return df_out, scaler


def split_and_save_data(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.20,
    random_state: int = DEFAULT_RANDOM_STATE,
    age_col: str = "RIDAGEYR",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs age-stratified train/test splitting and saves processed datasets to data/processed/.

    Prevents Data Leakage:
    - Splits raw engineered features first into Train and Test.
    - Fits StandardScaler ONLY on Train data, then transforms Test data.

    Args:
        df (pd.DataFrame): Imputed and composite-engineered DataFrame.
        output_dir (Path): Path to output directory (e.g. data/processed/).
        test_size (float): Proportion of dataset for test split (default 0.20).
        random_state (int): Random seed for reproducibility.
        age_col (str): Age column used for age-bracket stratification.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create Age Brackets for Stratified Splitting
    age_bins = pd.cut(df[age_col], bins=[17, 34, 49, 64, 80], labels=["18-34", "35-49", "50-64", "65-79"])

    # 2. Stratified Train/Test Split
    raw_train_df, raw_test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=age_bins,
    )

    logger.info(
        f"Age-stratified split complete (test_size={test_size:.0%}): "
        f"Train shape = {raw_train_df.shape}, Test shape = {raw_test_df.shape}."
    )

    # 3. Fit scaler on Train ONLY, transform both Train and Test
    train_df, fitted_scaler = encode_and_scale(raw_train_df, scaler=None)
    test_df, _ = encode_and_scale(raw_test_df, scaler=fitted_scaler)

    # 4. Save processed train and test datasets and fitted scaler
    train_path = output_dir / "train_data.csv"
    test_path = output_dir / "test_data.csv"
    project_root = output_dir.parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / "scaler.pkl"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    import pickle
    with open(scaler_path, "wb") as f:
        pickle.dump(fitted_scaler, f)

    logger.info(f"Saved processed train dataset to: {train_path}")
    logger.info(f"Saved processed test dataset to: {test_path}")
    logger.info(f"Saved fitted StandardScaler to: {scaler_path}")

    return train_df, test_df



def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main orchestration function for the feature engineering pipeline.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df).
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    processed_dir = project_root / "data" / "processed"
    raw_merged_file = processed_dir / "merged_data.csv"

    if not raw_merged_file.exists():
        raise FileNotFoundError(f"Input file {raw_merged_file} not found. Run src/data_loader.py first.")

    logger.info(f"Loading merged dataset from {raw_merged_file}...")
    raw_df = pd.read_csv(raw_merged_file)

    # 1. Filter age top-coding (RIDAGEYR >= 80)
    df_age_filtered = handle_age_topcoding(raw_df)

    # 2. Handle missing data & construct subsample indicator
    df_imputed = handle_missing_data(df_age_filtered)

    # 3. Engineer composite features & ratio features
    df_composite = engineer_composite_features(df_imputed)

    # 4. Stratified split, fit scaler on train, transform test, and save
    train_df, test_df = split_and_save_data(df_composite, output_dir=processed_dir)

    logger.info("Feature engineering pipeline completed successfully.")
    return train_df, test_df


if __name__ == "__main__":
    main()
