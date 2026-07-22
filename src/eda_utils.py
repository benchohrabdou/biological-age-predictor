"""
src/eda_utils.py
Reusable plotting and summary utilities for EDA notebooks.
All functions accept a DataFrame and a list of column names,
so they can be called cleanly from notebooks without copy-pasting.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 110


# ── Univariate helpers ─────────────────────────────────────────────────────────

def plot_biomarker_distributions(
    df: pd.DataFrame,
    columns: List[str],
    labels: Optional[dict] = None,
    n_cols: int = 3,
    bins: int = 40,
    figsize_per_plot: tuple = (4.5, 3.5),
) -> None:
    """Plot a grid of histogram + KDE plots for the given columns.

    Each subplot also annotates median, skewness, and % missing so that
    a reader can assess distribution shape and data quality at a glance.

    Args:
        df: The merged NHANES DataFrame.
        columns: List of column names to plot.
        labels: Optional dict mapping column name → human-readable label.
        n_cols: Number of columns in the subplot grid.
        bins: Number of histogram bins.
        figsize_per_plot: (width, height) per subplot in inches.
    """
    labels = labels or {}
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig_w = figsize_per_plot[0] * n_cols
    fig_h = figsize_per_plot[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten()

    for idx, col in enumerate(columns):
        ax = axes[idx]
        series = df[col].dropna()
        label = labels.get(col, col)

        # Histogram + KDE
        sns.histplot(series, bins=bins, kde=True, ax=ax, color="#4C72B0", alpha=0.75)

        # Annotation box
        skew = series.skew()
        median = series.median()
        missing_pct = df[col].isnull().mean() * 100
        annotation = (
            f"median={median:.2g}\n"
            f"skew={skew:.2f}\n"
            f"missing={missing_pct:.1f}%"
        )
        ax.text(
            0.97, 0.97, annotation,
            transform=ax.transAxes,
            fontsize=8,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

        # Red flag: flag extreme skew
        if abs(skew) > 3:
            ax.set_title(f"⚠ {label}", fontsize=9, color="crimson", fontweight="bold")
        else:
            ax.set_title(label, fontsize=9)

        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Biomarker Distributions — NHANES 2021–2023", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def biomarker_summary_table(
    df: pd.DataFrame,
    columns: List[str],
    labels: Optional[dict] = None,
) -> pd.DataFrame:
    """Return a summary DataFrame with key statistics for each column.

    Columns returned: N (non-null), Missing %, Mean, Median, Std, Min, Max, Skew.
    Rows with |skew| > 3 are flagged as needing log-transformation.

    Args:
        df: The merged NHANES DataFrame.
        columns: Columns to summarise.
        labels: Optional dict mapping column name → human-readable label.

    Returns:
        A styled pandas DataFrame ready for notebook display.
    """
    labels = labels or {}
    rows = []
    for col in columns:
        s = df[col]
        rows.append({
            "Column": col,
            "Label": labels.get(col, col),
            "N (non-null)": s.notna().sum(),
            "Missing %": f"{s.isnull().mean() * 100:.1f}%",
            "Mean": f"{s.mean():.3g}",
            "Median": f"{s.median():.3g}",
            "Std": f"{s.std():.3g}",
            "Min": f"{s.min():.3g}",
            "Max": f"{s.max():.3g}",
            "Skew": f"{s.skew():.2f}",
            "⚠ Flag": "Log-transform" if abs(s.skew()) > 3 else "",
        })
    summary = pd.DataFrame(rows).set_index("Column")
    return summary
