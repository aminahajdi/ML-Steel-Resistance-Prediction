"""Study 1 — IDEA StatiCa CWC database diagnostics.

Uses the same sequential upper-tail filtering as the Study 1 model scripts.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_01_data.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

NUM_BINS = 30
COUNT_THRESHOLD = 5
INPUT_COLUMNS = ["D [value]", "B [value]", "tf [value]", "tw [value]", "rw [value]", "x", "a", "fy", "epsilon_lim"]
OUTPUT_COLUMN = "F_plRd_CBFEM"
COLUMNS = INPUT_COLUMNS + [OUTPUT_COLUMN]
DISPLAY_NAMES = [r"$D$", r"$B$", r"$t_f$", r"$t_w$", r"$r_w$", r"$x$", r"$a$", r"$f_y$", r"$\varepsilon_{lim}$", r"$F_{pl,Rd}$"]


def remove_sparse_upper_tail(df, column):
    """Truncate at the upper edge of the first bin containing fewer than five rows."""
    counts, edges = np.histogram(df[column], bins=NUM_BINS)
    low_count_bins = np.where(counts < COUNT_THRESHOLD)[0]
    if len(low_count_bins) == 0:
        return df.copy(), None
    cutoff = edges[low_count_bins[0] + 1]
    return df.loc[df[column] <= cutoff].copy(), cutoff


def load_filtered_data():
    """Read the original workbook and apply the archived filtering in order."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Missing Study 1 data file: {DATA_FILE}")
    data = pd.read_excel(DATA_FILE)
    missing = [col for col in COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Validate relevant numeric inputs; do not silently drop or reorder observations.
    for col in COLUMNS:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column must be numeric: {col}")
        if not np.isfinite(data[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values found in: {col}")
    original_size = len(data)
    for col in ["tf [value]", "tw [value]", OUTPUT_COLUMN]:
        data, cutoff = remove_sparse_upper_tail(data, col)
        print(f"{col}: upper-tail cutoff = {cutoff}")
    print(f"Observations: {original_size} before filtering; {len(data)} after filtering")
    return data[COLUMNS].copy()

def main():
    """Generate the Study 1 Pearson correlation heatmap (one PNG)."""
    data = load_filtered_data()
    corr = data.corr(method="pearson")
    matrix = corr.to_numpy(dtype=float)
    n = len(COLUMNS)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(DISPLAY_NAMES, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(DISPLAY_NAMES, fontsize=10)
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(value) > 0.55 else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=9)
    ax.set_title("IDEA CWC database", fontsize=14, pad=15)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "Correlation_IDEA_CWC.png"
    fig.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
