"""
src/age_gap.py
Age-Gap Bias Correction Pipeline via Linear Residualization on Out-of-Fold Predictions.

Methodological background:
A regression model trained to predict chronological age regresses toward the population mean:
it tends to over-predict younger participants and under-predict older participants. Consequently,
the raw gap Delta = predicted - actual is negatively correlated with chronological age.
The standard biostatistical solution is to regress Delta on chronological age:
    Delta = alpha + beta * Age + epsilon
and define the bias-corrected gap as the residual:
    Delta_corrected = Delta - (alpha + beta * Age) = epsilon

Key implementation details:
- Uses Out-of-Fold (OOF) predictions from 5-fold stratified cross-validation (XGBoost).
- Computes Pearson correlation before and after correction (target: r_after approx 0).
- Saves per-participant records (SEQN, age, predicted_age, raw_gap, corrected_gap) to results/age_gaps.csv.
- Saves a dual-panel scatter plot (gap vs age, before and after) to results/figures/age_gap_correction.png.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from evaluate_cv import build_preprocessor
from feature_engineering import (
    load_and_preprocess_full_cohort,
    DEFAULT_RANDOM_STATE,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_age_gap_correction(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """Generates out-of-fold predictions with XGBoost and performs linear age-bias correction.

    Args:
        random_state (int): Random seed for stratified splitting and model.

    Returns:
        Tuple[pd.DataFrame, float, float, float, float]:
            (gaps_df, corr_before, p_before, corr_after, p_after)
    """
    logger.info("Loading full cohort (N = 5,995, ages 18-79)...")
    X, y, seqn, age_bins = load_and_preprocess_full_cohort()

    # Build XGBoost Pipeline matching benchmark specification
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", xgb_model),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    logger.info("Generating Out-of-Fold (OOF) predictions across 5 stratified folds...")
    oof_predictions = cross_val_predict(
        pipeline,
        X,
        y,
        cv=skf.split(X, age_bins),
        n_jobs=-1,
    )

    # 1. Raw Age Gap: predicted - chronological
    raw_gap = oof_predictions - y
    corr_before, p_before = pearsonr(y, raw_gap)

    logger.info(f"Raw Gap vs Age correlation: r = {corr_before:.4f} (p = {p_before:.2e})")

    # 2. Linear regression: Delta ~ Age
    reg = LinearRegression()
    reg.fit(y.values.reshape(-1, 1), raw_gap)
    expected_gap = reg.predict(y.values.reshape(-1, 1))

    # 3. Corrected Age Gap: residual epsilon
    corrected_gap = raw_gap - expected_gap
    corr_after, p_after = pearsonr(y, corrected_gap)

    logger.info(
        f"Fitted linear trend: Delta_expected = {reg.intercept_:.4f} + ({reg.coef_[0]:.4f} * Age)"
    )
    logger.info(f"Corrected Gap vs Age correlation: r = {corr_after:.4f} (p = {p_after:.4f})")

    gaps_df = pd.DataFrame({
        "SEQN": seqn.values,
        "age": y.values,
        "predicted_age": np.round(oof_predictions, 3),
        "raw_gap": np.round(raw_gap.values, 3),
        "corrected_gap": np.round(corrected_gap.values, 3),
    })

    return gaps_df, corr_before, p_before, corr_after, p_after


def plot_gap_correction(
    gaps_df: pd.DataFrame,
    corr_before: float,
    corr_after: float,
    output_path: Path,
) -> None:
    """Plots and saves a dual-panel scatter plot comparing Raw vs Corrected Age Gaps.

    Args:
        gaps_df (pd.DataFrame): DataFrame with 'age', 'raw_gap', and 'corrected_gap'.
        corr_before (float): Pearson r before correction.
        corr_after (float): Pearson r after correction.
        output_path (Path): Destination PNG file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    age = gaps_df["age"]
    raw_gap = gaps_df["raw_gap"]
    corrected_gap = gaps_df["corrected_gap"]

    # ── Left Panel: Raw Gap vs Age ──
    axes[0].scatter(age, raw_gap, alpha=0.25, color="#2563EB", edgecolors="none", s=20)
    # Regression trendline
    m_raw, b_raw = np.polyfit(age, raw_gap, 1)
    x_vals = np.linspace(age.min(), age.max(), 100)
    axes[0].plot(x_vals, m_raw * x_vals + b_raw, color="#DC2626", linewidth=2.2, label=f"Fit (slope = {m_raw:.3f})")
    axes[0].axhline(0, color="#64748B", linestyle="--", linewidth=1.2, alpha=0.8)

    axes[0].set_title(
        f"Raw Age Gap vs Chronological Age\nPearson r = {corr_before:.3f} (p < 0.001)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_xlabel("Chronological Age (years)", fontsize=11)
    axes[0].set_ylabel("Age Gap (Predicted − Chronological, years)", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].set_xlim(15, 82)

    # ── Right Panel: Corrected Gap vs Age ──
    axes[1].scatter(age, corrected_gap, alpha=0.25, color="#059669", edgecolors="none", s=20)
    m_corr, b_corr = np.polyfit(age, corrected_gap, 1)
    axes[1].plot(x_vals, m_corr * x_vals + b_corr, color="#059669", linewidth=2.2, label=f"Fit (slope = {m_corr:.3f})")
    axes[1].axhline(0, color="#64748B", linestyle="--", linewidth=1.2, alpha=0.8)

    axes[1].set_title(
        f"Bias-Corrected Age Gap vs Chronological Age\nPearson r = {corr_after:.3f} (orthogonalized)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_xlabel("Chronological Age (years)", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].set_xlim(15, 82)

    fig.suptitle(
        "Age-Gap Bias Correction: Resolving Regression Toward the Mean (XGBoost OOF, N = 5,995)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved age gap correction figure to: {output_path}")


def main():
    """Main execution entrypoint for Task 2."""
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    gaps_path = results_dir / "age_gaps.csv"
    fig_path = figures_dir / "age_gap_correction.png"

    gaps_df, corr_before, p_before, corr_after, p_after = compute_age_gap_correction()

    gaps_df.to_csv(gaps_path, index=False)
    logger.info(f"Saved participant age gaps to: {gaps_path}")

    plot_gap_correction(gaps_df, corr_before, corr_after, fig_path)

    print("\n" + "=" * 70)
    print("AGE-GAP BIAS CORRECTION SUMMARY (OUT-OF-FOLD XGBOOST, N = 5,995)")
    print("=" * 70)
    print(f"Raw Gap vs Age Correlation:       r = {corr_before:+.4f} (p = {p_before:.2e})")
    print(f"Corrected Gap vs Age Correlation: r = {corr_after:+.4f} (p = {p_after:.4f})")
    print(f"Participant Table Saved:          {gaps_path}")
    print(f"Scatter Plot Figure Saved:        {fig_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
