# =============================================================================
# STUDY 4 — WELDED JOINTS: PEARSON AND SPEARMAN CORRELATION
# =============================================================================
# Purpose:
# - Read the joint dataset.
# - Compute Pearson and Spearman correlation matrices.
# - Save correlation matrices as CSV files.
# - Save heatmap plots.
#
# Notes:
# - Pearson correlation captures linear relationships.
# - Spearman correlation captures monotonic relationships and is more robust
#   for nonlinear but ordered trends.
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. FILES AND FOLDERS
# =============================================================================

# Resolve paths from the final public repository layout.
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_04_data.xlsx"
RESULTS_DIR = SCRIPT_DIR / "outputs"

PEARSON_CSV_PATH = RESULTS_DIR / "pearson_correlation_matrix.csv"
SPEARMAN_CSV_PATH = RESULTS_DIR / "spearman_correlation_matrix.csv"

PEARSON_HEATMAP_PATH = RESULTS_DIR / "pearson_correlation_heatmap.png"
SPEARMAN_HEATMAP_PATH = RESULTS_DIR / "spearman_correlation_heatmap.png"

TARGET_CORRELATION_PATH = RESULTS_DIR / "correlation_with_moment_resistance.csv"
TARGET_CORRELATION_PLOT_PATH = RESULTS_DIR / "correlation_with_moment_resistance_barplot.png"

SUMMARY_TXT_PATH = RESULTS_DIR / "correlation_summary.txt"


# =============================================================================
# 2. COLUMNS
# =============================================================================

INPUT_COLUMNS = [
    "hc",
    "bc",
    "twc",
    "tfc",
    "rc",
    "hb1",
    "bb1",
    "twb1",
    "tfb1",
    "rb1",
    "axial force percentage",
    "stiffening",
    "joint location",
    "moment ratio",
    "fy",
]

OUTPUT_COLUMN = "moment resistance"


COLUMNS_TO_USE = INPUT_COLUMNS + [OUTPUT_COLUMN]


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def save_correlation_heatmap(corr_df, title, output_path):
    labels = corr_df.columns.tolist()
    matrix = corr_df.to_numpy(dtype=float)

    fig_size = max(8, 0.55 * len(labels))

    plt.figure(figsize=(fig_size, fig_size))

    im = plt.imshow(matrix, vmin=-1, vmax=1)

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation coefficient")

    plt.xticks(
        ticks=np.arange(len(labels)),
        labels=labels,
        rotation=90,
        fontsize=8,
    )

    plt.yticks(
        ticks=np.arange(len(labels)),
        labels=labels,
        fontsize=8,
    )

    # Write numbers inside cells.
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
            )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_target_correlation_plot(target_corr_df, output_path):
    plot_df = target_corr_df.copy()
    plot_df = plot_df[plot_df["variable"] != OUTPUT_COLUMN].copy()
    plot_df = plot_df.sort_values("pearson_abs", ascending=True)

    plt.figure(figsize=(8, max(5, 0.35 * len(plot_df))))

    plt.barh(
        plot_df["variable"],
        plot_df["pearson"],
    )

    plt.axvline(0.0, linewidth=1)

    plt.xlabel("Pearson correlation with moment resistance")
    plt.ylabel("Variable")
    plt.title("Correlation with moment resistance")
    plt.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    # =============================================================================
    # 4. READ DATA
    # =============================================================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Study 4 dataset not found: {DATA_FILE}\n"
            "Place the authorized Excel file at data/study_04_data.xlsx "
            "in the repository root."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(DATA_FILE)

    missing_required_columns = [col for col in COLUMNS_TO_USE if col not in df.columns]

    if missing_required_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_required_columns}")

    df_corr = df[COLUMNS_TO_USE].replace([np.inf, -np.inf], np.nan).dropna().copy()


    # =============================================================================
    # 5. CORRELATION MATRICES
    # =============================================================================

    pearson_corr = df_corr.corr(method="pearson")
    spearman_corr = df_corr.corr(method="spearman")

    pearson_corr.to_csv(PEARSON_CSV_PATH)
    spearman_corr.to_csv(SPEARMAN_CSV_PATH)

    save_correlation_heatmap(
        corr_df=pearson_corr,
        title="Pearson correlation matrix - joint dataset",
        output_path=PEARSON_HEATMAP_PATH,
    )

    save_correlation_heatmap(
        corr_df=spearman_corr,
        title="Spearman correlation matrix - joint dataset",
        output_path=SPEARMAN_HEATMAP_PATH,
    )


    # =============================================================================
    # 6. CORRELATION WITH TARGET
    # =============================================================================

    target_records = []

    for col in COLUMNS_TO_USE:
        target_records.append(
            {
                "variable": col,
                "pearson": pearson_corr.loc[col, OUTPUT_COLUMN],
                "pearson_abs": abs(pearson_corr.loc[col, OUTPUT_COLUMN]),
                "spearman": spearman_corr.loc[col, OUTPUT_COLUMN],
                "spearman_abs": abs(spearman_corr.loc[col, OUTPUT_COLUMN]),
            }
        )

    target_corr_df = pd.DataFrame(target_records)
    target_corr_df = target_corr_df.sort_values("pearson_abs", ascending=False)

    target_corr_df.to_csv(TARGET_CORRELATION_PATH, index=False)

    save_target_correlation_plot(
        target_corr_df=target_corr_df,
        output_path=TARGET_CORRELATION_PLOT_PATH,
    )


    # =============================================================================
    # 7. SAVE SUMMARY
    # =============================================================================

    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("CORRELATION MATRIX SUMMARY - JOINT DATASET\n")
        f.write("==========================================\n\n")

        f.write(f"Dataset file: {DATA_FILE}\n")
        f.write(f"Rows used after dropna: {len(df_corr)}\n")
        f.write(f"Columns used: {COLUMNS_TO_USE}\n\n")

        f.write("Top correlations with moment resistance by absolute Pearson correlation:\n")
        f.write("=======================================================================\n")
        f.write(target_corr_df.to_string(index=False))
        f.write("\n\n")

        f.write("Notes:\n")
        f.write("======\n")
        f.write("- Pearson correlation measures linear association.\n")
        f.write("- Spearman correlation measures monotonic association.\n")
        f.write("- Strong correlation does not necessarily mean the variable should be removed.\n")


    print("====================================================")
    print("CORRELATION MATRIX COMPLETE")
    print("====================================================")
    print(f"Dataset file: {DATA_FILE}")
    print(f"Rows used after dropna: {len(df_corr)}")
    print(f"Columns used: {COLUMNS_TO_USE}")
    print()
    print(f"Pearson matrix saved to: {PEARSON_CSV_PATH}")
    print(f"Spearman matrix saved to: {SPEARMAN_CSV_PATH}")
    print(f"Pearson heatmap saved to: {PEARSON_HEATMAP_PATH}")
    print(f"Spearman heatmap saved to: {SPEARMAN_HEATMAP_PATH}")
    print(f"Target correlation table saved to: {TARGET_CORRELATION_PATH}")
    print(f"Target correlation plot saved to: {TARGET_CORRELATION_PLOT_PATH}")
    print(f"Summary saved to: {SUMMARY_TXT_PATH}")
    print(f"Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
