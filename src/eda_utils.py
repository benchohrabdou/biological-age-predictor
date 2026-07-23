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


# ── Correlation helpers ────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str],
    labels: Optional[dict] = None,
    title: str = "Biomarker Correlation Heatmap",
    figsize: tuple = (12, 10),
    annot: bool = True,
    cmap: str = "coolwarm",
) -> pd.DataFrame:
    """Plot a correlation heatmap for the specified columns.

    Args:
        df: Input DataFrame.
        columns: List of column names to include.
        labels: Optional dict mapping column name → human-readable label.
        title: Plot title.
        figsize: Figure size tuple.
        annot: Whether to annotate values in heat cells.
        cmap: Seaborn colormap string.

    Returns:
        The correlation matrix DataFrame.
    """
    labels = labels or {}
    display_names = [labels.get(c, c) for c in columns]

    corr_df = df[columns].copy()
    corr_df.columns = display_names

    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        annot=annot,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()

    return corr_matrix


def find_multicollinear_pairs(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 0.70,
    labels: Optional[dict] = None,
) -> pd.DataFrame:
    """Scan feature pairs and return a DataFrame of pairs with |r| >= threshold.

    Args:
        df: Input DataFrame.
        columns: Feature column names to check.
        threshold: Absolute correlation threshold cutoff (default 0.70).
        labels: Optional dict mapping column name → human-readable label.

    Returns:
        pd.DataFrame with columns ['Feature 1', 'Feature 2', 'Correlation (r)'].
    """
    labels = labels or {}
    corr_matrix = df[columns].corr()

    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            f1, f2 = columns[i], columns[j]
            r = corr_matrix.loc[f1, f2]
            if abs(r) >= threshold:
                pairs.append({
                    "Feature 1": labels.get(f1, f1),
                    "Feature 2": labels.get(f2, f2),
                    "Raw_Col_1": f1,
                    "Raw_Col_2": f2,
                    "Correlation (r)": round(r, 3),
                    "|r|": round(abs(r), 3),
                })

    res = pd.DataFrame(pairs)
    if not res.empty:
        res = res.sort_values(by="|r|", ascending=False).drop(columns=["|r|"]).reset_index(drop=True)
    return res

