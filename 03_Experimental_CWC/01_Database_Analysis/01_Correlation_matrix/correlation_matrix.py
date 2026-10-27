"""Pearson correlation matrix for Study 3: experimental CWC database.

The analysis uses the complete cases for the eight engineering inputs and the
experimental resistance. The observation with the maximum F_exp is excluded,
consistent with the original correlation-matrix analysis.

Run from the repository root:
    python 03_Experimental_CWC/01_Database_Analysis/01_Correlation_Matrix/correlation_matrix.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_03_data.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

INPUT_COLUMNS = ["h_c", "t_wc", "b_fc", "t_fc", "r_c", "t_fb", "N/Npl", "f_y"]
TARGET_COLUMN = "F_exp"
DISPLAY_NAMES = [
    r"$h_c$", r"$t_{wc}$", r"$b_{fc}$", r"$t_{fc}$", r"$r_c$",
    r"$t_{fb}$", r"$N/N_{pl}$", r"$f_y$", r"$F_{exp}$",
]


def prepare_data():
    """Preserve the original complete-case and maximum-target exclusion rules."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Required dataset not found: {DATA_FILE}")
    frame = pd.read_excel(DATA_FILE)
    columns = INPUT_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    # As in the source analysis, remove rows with missing input/target values.
    frame = frame[columns].dropna().copy()
    if frame.empty:
        raise ValueError("No complete observations remain after dropping missing data.")
    removed_row = frame[TARGET_COLUMN].idxmax()
    print("Excluded highest-resistance observation:")
    print(frame.loc[removed_row, columns].to_string())
    frame = frame.drop(index=removed_row).reset_index(drop=True)
    print(f"Observations in correlation analysis: {len(frame)}")
    return frame


def make_plot(frame):
    """Compute the Pearson correlation matrix and save its annotated heatmap."""
    columns = INPUT_COLUMNS + [TARGET_COLUMN]
    numeric = frame[columns].copy()
    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna()
    corr = numeric.corr(method="pearson")
    corr.index = DISPLAY_NAMES
    corr.columns = DISPLAY_NAMES
    n = len(columns)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(DISPLAY_NAMES, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(DISPLAY_NAMES, fontsize=10)
    for i in range(n):
        for j in range(n):
            value = corr.iloc[i, j]
            label = "" if np.isnan(value) else f"{value:.2f}"
            color = "white" if not np.isnan(value) and abs(value) > 0.55 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    bar.ax.tick_params(labelsize=9)
    ax.set_title("Experimental CWC database", fontsize=14, pad=15)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = OUTPUT_DIR / "Correlation_Experimental_CWC.png"
    fig.savefig(image, dpi=600, bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(OUTPUT_DIR / "experimental_cwc_pearson_correlations.csv")
    print(f"Saved correlation matrix and figure in {OUTPUT_DIR}")


def main():
    make_plot(prepare_data())


if __name__ == "__main__":
    main()
