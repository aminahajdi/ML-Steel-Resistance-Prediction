"""Study 1: histograms of nine IDEA CWC inputs and one resistance output.

Reads data/study_01_data.xlsx from the repository root. Applies the sequential 30-bin sparse-tail filter
used for the Study 1 model dataset. Saves a single 3-column histogram figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
NUM_BINS = 30
COUNT_THRESHOLD = 5
INPUT_COLUMNS = [
    "D [value]", "B [value]", "tf [value]", "tw [value]", "rw [value]",
    "x", "a", "fy", "epsilon_lim",
]
TARGET_COLUMN = "F_plRd_CBFEM"
COLUMNS = INPUT_COLUMNS + [TARGET_COLUMN]
TITLES = [
    r"Column depth $D$ [mm]",
    r"Column flange width $B$ [mm]",
    r"Flange thickness $t_f$ [mm]",
    r"Web thickness $t_w$ [mm]",
    r"Root radius $r_w$ [mm]",
    r"Normalized load position $x$ [-]",
    r"Thickness ratio $a$ [-]",
    r"Yield strength $f_y$ [MPa]",
    r"Strain limit $\varepsilon_{lim}$ [-]",
    r"Resistance $F_{pl,Rd}$ [kN]",
]


# Expected location: 01_IDEA_CWC/04_Database_Analysis/02_Histogram/histograms.py
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_01_data.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "outputs"


def locate_dataset():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Study 1 dataset not found: {DATA_FILE}\n"
            "Place the authorized workbook in the repository's data/ directory."
        )
    return DATA_FILE


def filter_upper_tail(df, column):
    counts, edges = np.histogram(df[column], bins=NUM_BINS)
    sparse = np.flatnonzero(counts < COUNT_THRESHOLD)
    if not len(sparse):
        return df.copy(), None
    cutoff = float(edges[sparse[0] + 1])
    return df.loc[df[column] <= cutoff].copy(), cutoff


def main():
    path = locate_dataset()
    data = pd.read_excel(path)
    missing = [c for c in COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = data[COLUMNS].copy()
    for column in COLUMNS:
        if not pd.api.types.is_numeric_dtype(data[column]) or not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid numeric values in column: {column}")

    original_count = len(data)
    for column in ["tf [value]", "tw [value]", TARGET_COLUMN]:
        data, cutoff = filter_upper_tail(data, column)
        print(f"{column}: cutoff = {cutoff}")
    print(f"Observations: {original_count} original, {len(data)} filtered")

    # Same visual language as the reference: three columns, blue bars,
    # black outlines, unit-bearing subplot titles, no overall title.
    fig, axes = plt.subplots(4, 3, figsize=(13.4, 12.0))
    for ax, column, title in zip(axes.flat, COLUMNS, TITLES):
        ax.hist(data[column].to_numpy(dtype=float), bins=NUM_BINS,
                color="#1f77b4", edgecolor="black", linewidth=0.9)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(False)
    for ax in list(axes.flat)[len(COLUMNS):]:
        fig.delaxes(ax)
    fig.tight_layout(h_pad=2.1, w_pad=1.1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path_out = OUTPUT_DIR / "Histograms_IDEA_CWC.png"
    fig.savefig(path_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_out}")


if __name__ == "__main__":
    main()
