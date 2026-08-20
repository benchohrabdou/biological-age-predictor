"""
src/explainability.py
Production-standard SHAP explainability module for the Biological Age Predictor.

Modules included:
- load_model_and_data: Loads saved model object and test dataset.
- compute_shap_explanation: Computes SHAP values using shap.TreeExplainer.
- get_global_feature_importance: Returns ranked DataFrame of mean absolute SHAP values.
- generate_individual_waterfall: Generates local SHAP explanation for a specific participant.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path 
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import shap

import warnings

# Suppress non-critical library load and font warnings
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_and_data(
    model_path: Path,
    test_data_path: Path,
    train_data_path: Path,
) -> Tuple[Any, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Loads serialized model object and prepared feature matrices.

    Args:
        model_path (Path): Path to serialized model (e.g. models/final_model.pkl).
        test_data_path (Path): Path to test_data.csv.
        train_data_path (Path): Path to train_data.csv.

    Returns:
        Tuple[Any, pd.DataFrame, pd.Series, pd.DataFrame]: (model, X_test, y_test, test_df).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found. Run src/models.py first.")
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test dataset {test_data_path} not found.")

    logger.info(f"Loading trained model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    from models import prepare_feature_matrices, MODEL_FEATURE_COLS
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    _, _, X_test, y_test = prepare_feature_matrices(train_df, test_df, feature_cols=MODEL_FEATURE_COLS)

    return model, X_test, y_test, test_df


def compute_shap_explanation(
    model: Any,
    X: pd.DataFrame,
) -> Tuple[shap.Explanation, shap.TreeExplainer]:
    """Computes SHAP Explanation object and TreeExplainer instance for feature matrix X.

    Args:
        model (Any): Fitted tree-based model (e.g. XGBoost Regressor).
        X (pd.DataFrame): Input feature matrix.

    Returns:
        Tuple[shap.Explanation, shap.TreeExplainer]: (shap_explanation_object, explainer_instance).
    """
    logger.info(f"Computing SHAP values across {len(X)} samples and {X.shape[1]} features using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X)

    logger.info("SHAP values computed successfully.")
    return shap_explanation, explainer


def get_global_feature_importance(
    shap_explanation: shap.Explanation,
    feature_names: List[str],
) -> pd.DataFrame:
    """Calculates mean absolute SHAP value per feature to rank global importance in years.

    Args:
        shap_explanation (shap.Explanation): Computed SHAP Explanation object.
        feature_names (List[str]): List of feature column names.

    Returns:
        pd.DataFrame: Ranked DataFrame containing 'feature' and 'mean_abs_shap_years'.
    """
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap_years": mean_abs_shap
    }).sort_values(by="mean_abs_shap_years", ascending=False).reset_index(drop=True)

    logger.info(f"Top 5 global biological age drivers:\n{importance_df.head(5).to_string(index=False)}")

    return importance_df


def main() -> Tuple[shap.Explanation, pd.DataFrame]:
    """Main orchestration function to load model, compute SHAP values, and log feature rankings."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"

    model_path = models_dir / "final_model.pkl"
    test_data_path = processed_dir / "test_data.csv"
    train_data_path = processed_dir / "train_data.csv"

    model, X_test, y_test, test_df = load_model_and_data(model_path, test_data_path, train_data_path)
    shap_exp, explainer = compute_shap_explanation(model, X_test)
    importance_df = get_global_feature_importance(shap_exp, list(X_test.columns))

    return shap_exp, importance_df


if __name__ == "__main__":
    main()
