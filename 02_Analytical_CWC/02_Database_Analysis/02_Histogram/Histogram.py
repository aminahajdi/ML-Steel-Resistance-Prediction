from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. FILES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "Parametrized set.xlsx"

OUTPUT_DIR = BASE_DIR / "analytical database histogram"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. SETTINGS
# SAME AS ANALYTICAL MODEL SCRIPT
# ============================================================

USE_DATA_FILTERING = True

NUM_BINS = 30
COUNT_THRESHOLD = 5

FILTER_COLUMN_1 = "t_fc"
FILTER_COLUMN_2 = "t_wc"

OUTPUT_COLUMN = "F_c_wc_Rd_kN"

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

INPUT_DISPLAY_NAMES = [
    r"Column depth $h_c$ [mm]",
    r"Web thickness $t_{wc}$ [mm]",
    r"Flange width $b_{fc}$ [mm]",
    r"Flange thickness $t_{fc}$ [mm]",
    r"Fillet radius $r_c$ [mm]",
    r"Loaded flange thickness $t_{fb}$ [mm]",
    r"$N/N_{pl}$ [-]",
    r"$f_y$ [MPa]",
]


# ============================================================
# 3. READ DATABASE
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

df = pd.read_excel(DATA_FILE)

print("\nOriginal database size:", len(df))


# ============================================================
# 4. APPLY SAME FILTERING AS MODEL SCRIPT
# ============================================================

if USE_DATA_FILTERING:

    # --------------------------------------------------------
    # Filter t_fc
    # --------------------------------------------------------

    counts_1, bin_edges_1 = np.histogram(
        df[FILTER_COLUMN_1],
        bins=NUM_BINS
    )

    low_count_bins_1 = np.where(
        counts_1 < COUNT_THRESHOLD
    )[0]

    if len(low_count_bins_1) > 0:

        cutoff_1 = bin_edges_1[
            low_count_bins_1[0] + 1
        ]

        print(
            f"{FILTER_COLUMN_1} cutoff: "
            f"{cutoff_1:.6f}"
        )

        df = df[
            df[FILTER_COLUMN_1] <= cutoff_1
        ].copy()

    # --------------------------------------------------------
    # Filter t_wc
    # --------------------------------------------------------

    counts_2, bin_edges_2 = np.histogram(
        df[FILTER_COLUMN_2],
        bins=NUM_BINS
    )

    low_count_bins_2 = np.where(
        counts_2 < COUNT_THRESHOLD
    )[0]

    if len(low_count_bins_2) > 0:

        cutoff_2 = bin_edges_2[
            low_count_bins_2[0] + 1
        ]

        print(
            f"{FILTER_COLUMN_2} cutoff: "
            f"{cutoff_2:.6f}"
        )

        df = df[
            df[FILTER_COLUMN_2] <= cutoff_2
        ].copy()

    # --------------------------------------------------------
    # Filter output
    # --------------------------------------------------------

    counts_out, bin_edges_out = np.histogram(
        df[OUTPUT_COLUMN],
        bins=NUM_BINS
    )

    low_count_bins_out = np.where(
        counts_out < COUNT_THRESHOLD
    )[0]

    if len(low_count_bins_out) > 0:

        output_cutoff = bin_edges_out[
            low_count_bins_out[0] + 1
        ]

        print(
            f"{OUTPUT_COLUMN} cutoff: "
            f"{output_cutoff:.6f}"
        )

        df = df[
            df[OUTPUT_COLUMN] <= output_cutoff
        ].copy()


print(
    "\nFiltered database size:",
    len(df)
)


# ============================================================
# 5. MINIMUM AND MAXIMUM INPUT VALUES
# AFTER FILTERING
# ============================================================

range_table = pd.DataFrame({
    "Feature": INPUT_DISPLAY_NAMES,
    "Minimum": [
        df[col].min()
        for col in INPUT_COLUMNS
    ],
    "Maximum": [
        df[col].max()
        for col in INPUT_COLUMNS
    ],
})

print("\nFILTERED ANALYTICAL CWC INPUT RANGES")
print("====================================")

print(
    range_table.to_string(
        index=False
    )
)


# ============================================================
# 6. SAVE MIN/MAX TABLE
# ============================================================

EXCEL_PATH = (
    OUTPUT_DIR
    / "Analytical_CWC_input_ranges.xlsx"
)

CSV_PATH = (
    OUTPUT_DIR
    / "Analytical_CWC_input_ranges.csv"
)

range_table.to_excel(
    EXCEL_PATH,
    index=False
)

range_table.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# 7. HISTOGRAMS
# INPUTS + RESISTANCE
# AFTER FILTERING
# ============================================================

PLOT_COLUMNS = [
    "h_c",
    "t_wc",
    "b_fc",
    "t_fc",
    "r_c",
    "t_fb",
    "N/Npl",
    "f_y",
    OUTPUT_COLUMN,
]

PLOT_TITLES = [
    r"Column depth $h_c$ [mm]",
    r"Web thickness $t_{wc}$ [mm]",
    r"Flange width $b_{fc}$ [mm]",
    r"Flange thickness $t_{fc}$ [mm]",
    r"Fillet radius $r_c$ [mm]",
    r"Loaded flange thickness $t_{fb}$ [mm]",
    r"$N/N_{pl}$ [-]",
    r"$f_y$ [MPa]",
    r"$F_{c,wc,Rd}$ [kN]",
]


ncols = 3
nrows = 3

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(12, 9)
)

axes = axes.flatten()


for i, (col, title) in enumerate(
    zip(PLOT_COLUMNS, PLOT_TITLES)
):

    axes[i].hist(
        df[col].dropna(),
        bins=30,
        edgecolor="black",
        linewidth=1.0
    )

    axes[i].set_title(
        title,
        fontsize=13
    )

    axes[i].tick_params(
        axis="both",
        labelsize=10
    )

    axes[i].grid(False)


plt.tight_layout()


# ============================================================
# 8. SAVE HISTOGRAM FIGURE
# ============================================================

HISTOGRAM_PATH = (
    OUTPUT_DIR
    / "Analytical_CWC_histograms.png"
)

plt.savefig(
    HISTOGRAM_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. FINAL OUTPUT
# ============================================================

print("\nDONE")

print("\nFiles saved in:")
print(OUTPUT_DIR)

print("\nExcel table:")
print(EXCEL_PATH)

print("\nCSV table:")
print(CSV_PATH)

print("\nHistogram figure:")
print(HISTOGRAM_PATH)
