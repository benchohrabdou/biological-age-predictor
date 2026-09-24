"""
src/evaluate_cv.py
5-Fold Stratified Cross-Validation Pipeline for Biological Age Prediction.

Models evaluated:
- Ordinary Least Squares (OLS) Linear Regression
- Ridge Regression (alpha = 10.0)
- Random Forest Regressor (n_estimators=200, max_depth=12, min_samples_split=5)
- XGBoost Regressor (n_estimators=200, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8)

Methodological constraints:
- Zero data leakage: SimpleImputer and StandardScaler are fitted strictly inside each training fold
  via an sklearn Pipeline and ColumnTransformer.
- Stratified sampling by age brackets matching current split stratification.
- Random seed fixed at random_state=42 for reproducibility.
- Saves results to results/cv_results.csv and results/cv_results_per_fold.csv.
"""



from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from feature_engineering import (
    load_and_preprocess_full_cohort,
    CONTINUOUS_SCALING_COLS,
    DEFAULT_RANDOM_STATE,
)
from models import MODEL_FEATURE_COLS

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """Builds a leakage-proof preprocessor inside the cross-validation pipeline.

    - Fits median imputer and StandardScaler on continuous features inside each training fold only.
    - Fits median imputer on unscaled numeric feature INDFMPIR inside each training fold only.
    - Passes through pre-encoded binary/ordinal indicator columns.

    Returns:
        ColumnTransformer: Preprocessing pipeline object.
    """
    passthrough_cols = [
        "was_fasting_sample",
        "sex_encoded",
        "pa_level_encoded",
        "smoking_Former Smoker",
        "smoking_Current Smoker",
    ]

    return ColumnTransformer(
        transformers=[
            (
                "continuous",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                CONTINUOUS_SCALING_COLS,
            ),
            (
                "unscaled_imputed",
                SimpleImputer(strategy="median"),
                ["INDFMPIR"],
            ),
            (
                "passthrough",
                "passthrough",
                passthrough_cols,
            ),
        ]
    )


def get_models(random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, Any]:
    """Instantiates the four candidate regression model architectures.

    Hyperparameters strictly match src/models.py.

    Args:
        random_state (int): Random seed.

    Returns:
        Dict[str, Any]: Dictionary of un-fitted model instances.
    """
    return {
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (alpha=10)": Ridge(alpha=10.0, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def run_cross_validation(
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs 5-fold stratified cross-validation on full cohort and returns summary and per-fold metrics.

    Args:
        n_splits (int): Number of folds (default 5).
        random_state (int): Random seed (default 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (cv_summary_df, per_fold_df).
    """
    logger.info("Loading full cohort (N = 5,995, ages 18-79)...")
    X, y, seqn, age_bins = load_and_preprocess_full_cohort()

    models = get_models(random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    per_fold_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []

    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} across {n_splits} stratified folds...")
        maes: List[float] = []
        rmses: List[float] = []
        r2s: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, age_bins), start=1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            pipeline = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("regressor", model),
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)

            mae = float(mean_absolute_error(y_val, y_pred))
            rmse = float(root_mean_squared_error(y_val, y_pred))
            r2 = float(r2_score(y_val, y_pred))

            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)

            per_fold_records.append({
                "Model": model_name,
                "Fold": fold_idx,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            })

        mae_mean, mae_std = float(np.mean(maes)), float(np.std(maes))
        rmse_mean, rmse_std = float(np.mean(rmses)), float(np.std(rmses))
        r2_mean, r2_std = float(np.mean(r2s)), float(np.std(r2s))

        logger.info(
            f"[{model_name}] MAE: {mae_mean:.2f} +/- {mae_std:.2f} yrs | "
            f"RMSE: {rmse_mean:.2f} +/- {rmse_std:.2f} yrs | "
            f"R2: {r2_mean:.3f} +/- {r2_std:.3f}"
        )

        summary_records.append({
            "Model": model_name,
            "MAE_mean": mae_mean,
            "MAE_std": mae_std,
            "RMSE_mean": rmse_mean,
            "RMSE_std": rmse_std,
            "R2_mean": r2_mean,
            "R2_std": r2_std,
        })

    cv_summary_df = pd.DataFrame(summary_records)
    per_fold_df = pd.DataFrame(per_fold_records)

    return cv_summary_df, per_fold_df


def main():
    """Main execution entrypoint for Task 1 cross-validation."""
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "cv_results.csv"
    per_fold_path = results_dir / "cv_results_per_fold.csv"

    cv_summary_df, per_fold_df = run_cross_validation()

    cv_summary_df.to_csv(summary_path, index=False)
    per_fold_df.to_csv(per_fold_path, index=False)

    logger.info(f"Saved cross-validation summary to: {summary_path}")
    logger.info(f"Saved per-fold evaluation records to: {per_fold_path}")

    print("\n" + "=" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION RESULTS (N = 5,995, AGES 18-79)")
    print("=" * 70)
    for _, row in cv_summary_df.iterrows():
        print(
            f"{row['Model']:<28} | "
            f"MAE: {row['MAE_mean']:.2f} +/- {row['MAE_std']:.2f} yrs | "
            f"RMSE: {row['RMSE_mean']:.2f} +/- {row['RMSE_std']:.2f} yrs | "
            f"R2: {row['R2_mean']:.3f} +/- {row['R2_std']:.3f}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
