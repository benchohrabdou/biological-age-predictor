"""
src/models.py
Production-standard modeling pipeline for Biological Age Prediction.

Modules included:
- prepare_feature_matrices: Extracts feature matrix X and target y.
- evaluate_model: Computes MAE, RMSE, and R^2 evaluation metrics.
- train_linear_baselines: Fits OLS Linear Regression and Ridge Regression models.
- train_tree_models: Fits Random Forest and XGBoost Regressor models.
- save_model: Serializes the best-performing model to models/final_model.pkl.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Named Constants ───────────────────────────────────────────────────────────
DEFAULT_RANDOM_STATE: int = 42
TARGET_COL: str = "RIDAGEYR"

# Default selected feature set for model training
MODEL_FEATURE_COLS: List[str] = [
    "log_LBXSCR",       # Log Creatinine
    "log_LBXSGL",       # Log Fasting Glucose
    "log_LBXSTR",       # Log Triglycerides
    "log_LBXGH",        # Log HbA1c
    "log_LBXSGTSI",     # Log GGT
    "log_LBXSATSI",     # Log ALT
    "log_PAD680",       # Log Sedentary Time
    "WHtR",             # Waist-to-Height Ratio
    "total_pa_min_wk",  # Weekly Moderate-Equivalent PA minutes
    "LBXHGB",           # Hemoglobin
    "LBXTC",            # Total Cholesterol
    "LBXSUA",           # Uric Acid
    "LBXSCA",           # Calcium
    "LBXWBCSI",         # White Blood Cell Count
    "LBXRBCSI",         # Red Blood Cell Count
    "LBXPLTSI",         # Platelet Count
    "LBXMCVSI",         # Mean Corpuscular Volume
    "BMXHT",            # Standing Height (cm)
    "BMXWT",            # Weight (kg)
    "BMXWAIST",         # Waist Circumference (cm)
    "BMXBMI",           # Body Mass Index
    "INDFMPIR",         # Family Income to Poverty Ratio
    "was_fasting_sample",      # Subsample Fasting Indicator (1/0)
    "sex_encoded",             # Sex Binary Encoded (Male=1, Female=0)
    "pa_level_encoded",        # Activity Level Ordinal (Low=0, Med=1, High=2)
    "smoking_Former Smoker",   # One-hot Former Smoker
    "smoking_Current Smoker",  # One-hot Current Smoker
]


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extracts feature matrices X_train, X_test and target vectors y_train, y_test.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        feature_cols (Optional[List[str]]): List of feature column names.
        target_col (str): Target column name (default 'RIDAGEYR').

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: (X_train, y_train, X_test, y_test).

    Raises:
        KeyError: If target_col is missing from train or test DataFrame.
    """
    feature_cols = feature_cols or MODEL_FEATURE_COLS

    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in training DataFrame.")
    if target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in testing DataFrame.")

    missing_train_feats = [c for c in feature_cols if c not in train_df.columns]
    if missing_train_feats:
        raise KeyError(f"Specified feature columns missing from training DataFrame: {missing_train_feats}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    logger.info(
        f"Extracted feature matrices: X_train = {X_train.shape}, y_train = {y_train.shape}, "
        f"X_test = {X_test.shape}, y_test = {y_test.shape} across {len(feature_cols)} features."
    )

    return X_train, y_train, X_test, y_test


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "Test",
) -> Dict[str, float]:
    """Calculates MAE, RMSE, and R^2 performance metrics for a fitted model.

    Args:
        model (Any): Fitted scikit-learn or XGBoost model instance.
        X (pd.DataFrame): Input feature matrix.
        y (pd.Series): True target values.
        dataset_name (str): Label for logging (e.g. 'Train' or 'Test').

    Returns:
        Dict[str, float]: Dictionary containing 'mae', 'rmse', and 'r2' scores.
    """
    y_pred = model.predict(X)
    mae = float(mean_absolute_error(y, y_pred))
    rmse = float(root_mean_squared_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    logger.info(
        f"[{dataset_name} Performance] MAE: {mae:.2f} yrs | RMSE: {rmse:.2f} yrs | R^2: {r2:.3f}"
    )

    return metrics


def train_linear_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ridge_alpha: float = 10.0,
) -> Dict[str, Any]:
    """Fits and evaluates OLS Linear Regression and Ridge Regression models.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target.
        ridge_alpha (float): L2 regularization parameter for Ridge (default 10.0).

    Returns:
        Dict[str, Any]: Results dictionary containing fitted models and evaluation metrics.
    """
    logger.info("Training OLS Linear Regression baseline...")
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    ols_metrics = evaluate_model(ols_model, X_test, y_test, dataset_name="OLS Test")

    logger.info(f"Training Ridge Regression (alpha={ridge_alpha})...")
    ridge_model = Ridge(alpha=ridge_alpha, random_state=DEFAULT_RANDOM_STATE)
    ridge_model.fit(X_train, y_train)
    ridge_metrics = evaluate_model(ridge_model, X_test, y_test, dataset_name="Ridge Test")

    return {
        "OLS": {"model": ols_model, "metrics": ols_metrics},
        "Ridge": {"model": ridge_model, "metrics": ridge_metrics},
    }


def train_tree_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """Fits and evaluates Random Forest Regressor and XGBoost Regressor models.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test target.
        random_state (int): Random seed for reproducibility.

    Returns:
        Dict[str, Any]: Results dictionary containing fitted models and evaluation metrics.
    """
    logger.info("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, dataset_name="RandomForest Test")

    logger.info("Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, dataset_name="XGBoost Test")

    return {
        "RandomForest": {"model": rf_model, "metrics": rf_metrics},
        "XGBoost": {"model": xgb_model, "metrics": xgb_metrics},
    }


def save_model(model: Any, output_path: Path) -> None:
    """Serializes fitted model instance to disk using pickle.

    Args:
        model (Any): Fitted model object.
        output_path (Path): Destination file path (e.g. models/final_model.pkl).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Successfully saved trained model object to: {output_path}")


def main() -> Dict[str, Any]:
    """Main orchestration function to train and compare baseline & tree regression models."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"

    train_path = processed_dir / "train_data.csv"
    test_path = processed_dir / "test_data.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Processed datasets missing. Run src/feature_engineering.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train, X_test, y_test = prepare_feature_matrices(train_df, test_df)

    # 1. Train Linear Models
    linear_results = train_linear_baselines(X_train, y_train, X_test, y_test)

    # 2. Train Tree Models
    tree_results = train_tree_models(X_train, y_train, X_test, y_test)

    all_results = {**linear_results, **tree_results}

    # Find best model based on lowest Test MAE
    best_model_name = min(all_results, key=lambda k: all_results[k]["metrics"]["mae"])
    best_model = all_results[best_model_name]["model"]
    best_mae = all_results[best_model_name]["metrics"]["mae"]

    logger.info(f"\nModel Comparison Complete! Best model: {best_model_name} with MAE = {best_mae:.2f} years.")

    # Save best model to models/final_model.pkl
    save_model(best_model, models_dir / "final_model.pkl")

    return all_results


if __name__ == "__main__":
    main()
