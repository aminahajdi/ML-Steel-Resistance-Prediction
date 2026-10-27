# =============================================================================
# CORRELATION MATRIX FOR EUROCODE DATASET
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. FILES AND FOLDERS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "Parametrized set.xlsx"

RESULTS_DIR = BASE_DIR / "EC_correlation_results"
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# 2. INPUT / OUTPUT COLUMNS
# =============================================================================
INPUT_COLUMNS = [
    "h_c",
    "t_wc",
    "b_fc",
    "t_fc",
    "r_c",
    "t_fb",
    "N/Npl",
    "f_y",
]

OUTPUT_COLUMN = "F_c_wc_Rd_kN"


# =============================================================================
# 3. READ DATA
# =============================================================================
df = pd.read_excel(DATA_FILE)

# Keep only selected columns
corr_df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()


# =============================================================================
# 4. COMPUTE CORRELATION MATRIX
# =============================================================================
correlation_matrix = corr_df.corr(method="pearson")

# Save numerical values
correlation_matrix.to_csv(
    RESULTS_DIR / "correlation_matrix.csv"
)


# =============================================================================
# 5. PLOT CORRELATION MATRIX
# =======================================================
plt.figure(figsize=(10, 8))

im = plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    im,
    label="Pearson correlation coefficient"
)

plt.xticks(
    ticks=np.arange(len(correlation_matrix.columns)),
    labels=correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(correlation_matrix.columns)),
    labels=correlation_matrix.columns
)

# Write correlation values inside cells
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Correlation Matrix - Eurocode Dataset")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "correlation_matrix.png",
    dpi=300
)

plt.savefig(
    RESULTS_DIR / "correlation_matrix.pdf",
    dpi=300
)

plt.close()


# =============================================================================
# 6. PRINT RESULTS
# =============================================================================
print("====================================================")
print("CORRELATION MATRIX SAVED")
print("====================================================")
print(f"Results folder: {RESULTS_DIR}")
print("Saved files:")
print(" - correlation_matrix.csv")
print(" - correlation_matrix.png")
print(" - correlation_matrix.pdf")