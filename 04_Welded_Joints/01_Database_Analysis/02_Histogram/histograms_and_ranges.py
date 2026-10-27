"""Study 4: welded-joint database input ranges and histograms.

Reads the Study 4 numerical dataset and saves descriptive database diagnostics.
No sparse-domain filtering is applied.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. FILES
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_04_data.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "outputs"


# ============================================================
# 2. DATABASE SETTINGS
# SAME AS WELDED FE MODEL SCRIPT
# ============================================================

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


# ============================================================
# 3. DISPLAY NAMES
# ============================================================

INPUT_DISPLAY_NAMES = [
    r"Column depth $h_c$ [mm]",
    r"Column width $b_c$ [mm]",
    r"Column web thickness $t_{wc}$ [mm]",
    r"Column flange thickness $t_{fc}$ [mm]",
    r"Column fillet radius $r_c$ [mm]",
    r"Beam depth $h_{b1}$ [mm]",
    r"Beam width $b_{b1}$ [mm]",
    r"Beam web thickness $t_{wb1}$ [mm]",
    r"Beam flange thickness $t_{fb1}$ [mm]",
    r"Beam fillet radius $r_{b1}$ [mm]",
    r"Axial force percentage [%]",
    r"Stiffening [-]",
    r"Joint location [-]",
    r"Moment ratio [-]",
    r"$f_y$ [MPa]",
]



def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Study 4 dataset not found: {DATA_FILE}\n"
            "Place the authorized workbook at data/study_04_data.xlsx in the repository root."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 4. READ DATABASE
    # ============================================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_excel(DATA_FILE)

    print("\nOriginal database size:", len(df))

    print("\nAvailable columns:")
    print(df.columns.tolist())


    # ============================================================
    # 5. CHECK REQUIRED COLUMNS
    # ============================================================

    required_columns = INPUT_COLUMNS + [OUTPUT_COLUMN]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nThe following columns were not found:\n"
            + "\n".join(missing_columns)
            + "\n\nAvailable columns are:\n"
            + "\n".join(df.columns.astype(str))
        )


    # ============================================================
    # 6. SAME DATA PREPARATION AS MODEL SCRIPT
    # NO SPARSE-DOMAIN FILTERING
    # ============================================================

    df = df[
        INPUT_COLUMNS + [OUTPUT_COLUMN]
    ].dropna().copy()

    print(
        "\nFinal welded FE database size:",
        len(df)
    )


    # ============================================================
    # 7. MINIMUM AND MAXIMUM INPUT VALUES
    # ============================================================

    range_table = pd.DataFrame({
        "Feature": INPUT_COLUMNS,
        "Minimum": [
            df[col].min()
            for col in INPUT_COLUMNS
        ],
        "Maximum": [
            df[col].max()
            for col in INPUT_COLUMNS
        ],
    })

    print("\nWELDED FE INPUT RANGES")
    print("======================")

    print(
        range_table.to_string(
            index=False
        )
    )


    # ============================================================
    # 8. SAVE MIN/MAX TABLE
    # ============================================================

    EXCEL_PATH = (
        OUTPUT_DIR
        / "Welded_FE_input_ranges.xlsx"
    )

    CSV_PATH = (
        OUTPUT_DIR
        / "Welded_FE_input_ranges.csv"
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
    # 9. HISTOGRAM SETTINGS
    # INPUTS + MOMENT RESISTANCE
    # ============================================================

    PLOT_COLUMNS = INPUT_COLUMNS + [OUTPUT_COLUMN]

    PLOT_TITLES = [
        r"Column depth $h_c$ [mm]",
        r"Column width $b_c$ [mm]",
        r"Column web thickness $t_{wc}$ [mm]",
        r"Column flange thickness $t_{fc}$ [mm]",
        r"Column fillet radius $r_c$ [mm]",
        r"Beam depth $h_{b1}$ [mm]",
        r"Beam width $b_{b1}$ [mm]",
        r"Beam web thickness $t_{wb1}$ [mm]",
        r"Beam flange thickness $t_{fb1}$ [mm]",
        r"Beam fillet radius $r_{b1}$ [mm]",
        r"Axial force percentage [%]",
        r"Stiffening [-]",
        r"Joint location [-]",
        r"Moment ratio [-]",
        r"$f_y$ [MPa]",
        r"Moment resistance [kNm]",
    ]


    # ============================================================
    # 10. HISTOGRAMS
    # SAME STYLE AS THE OTHER DATABASES
    # ============================================================

    ncols = 4
    nrows = 4

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 12)
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
            fontsize=12
        )

        axes[i].tick_params(
            axis="both",
            labelsize=9
        )

        axes[i].grid(False)


    # Remove unused panels
    for j in range(
        len(PLOT_COLUMNS),
        len(axes)
    ):
        fig.delaxes(
            axes[j]
        )


    plt.tight_layout()


    # ============================================================
    # 11. SAVE HISTOGRAM FIGURE
    # ============================================================

    HISTOGRAM_PATH = (
        OUTPUT_DIR
        / "Welded_FE_histograms.png"
    )

    plt.savefig(
        HISTOGRAM_PATH,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ============================================================
    # 12. FINAL OUTPUT
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


if __name__ == "__main__":
    main()
