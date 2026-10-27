"""Study 3 — Experimental CWC database distributions and input ranges.

Generate nine histograms and input-variable min/max tables from the curated
experimental CWC dataset. The analysis removes rows with missing required
values, followed by the single observation of maximum experimental resistance.

Input:  repository_root/data/study_03_data.xlsx
Output: 02_Histograms/outputs/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. FILE PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_03_data.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "outputs"


# ============================================================
# 2. ANALYSIS CONFIGURATION
# ============================================================

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

TARGET_COLUMN = "F_exp"

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




def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ============================================================
    # 3. READ DATABASE
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
    # 4. REMOVE MISSING VALUES
    # ============================================================

    required_columns = INPUT_COLUMNS + [TARGET_COLUMN]

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

    df = df.dropna(
        subset=required_columns
    ).copy()

    print(
        "\nDatabase size after removing missing values:",
        len(df)
    )


    # ============================================================
    # 5. REMOVE INFLUENTIAL OBSERVATION / OUTLIER
    # ============================================================

    # Use complete observations, then exclude the observation with the
    # largest experimental resistance, as in the study database definition.

    outlier_index = df[TARGET_COLUMN].idxmax()

    print(
        "\nRemoved influential observation:"
    )

    print(
        df.loc[
            outlier_index,
            required_columns
        ]
    )

    df = df.drop(
        index=outlier_index
    ).reset_index(drop=True)

    print(
        "\nFinal experimental database size:",
        len(df)
    )


    # ============================================================
    # 6. MINIMUM AND MAXIMUM INPUT VALUES
    # AFTER FINAL PREPROCESSING
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

    print("\nFINAL EXPERIMENTAL CWC INPUT RANGES")
    print("===================================")

    print(
        range_table.to_string(
            index=False
        )
    )


    # ============================================================
    # 7. SAVE MIN/MAX TABLE
    # ============================================================

    EXCEL_PATH = (
        OUTPUT_DIR
        / "Experimental_CWC_input_ranges.xlsx"
    )

    CSV_PATH = (
        OUTPUT_DIR
        / "Experimental_CWC_input_ranges.csv"
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
    # 8. HISTOGRAM SETTINGS
    # INPUTS + EXPERIMENTAL RESISTANCE
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
        TARGET_COLUMN,
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
        r"Experimental resistance $F_{exp}$ [kN]",
    ]


    # ============================================================
    # 9. HISTOGRAMS
    # Nine histograms: eight input variables and experimental resistance
    # ============================================================

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
    # 10. SAVE HISTOGRAM FIGURE
    # ============================================================

    HISTOGRAM_PATH = (
        OUTPUT_DIR
        / "Experimental_CWC_histograms.png"
    )

    plt.savefig(
        HISTOGRAM_PATH,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ============================================================
    # 11. FINAL OUTPUT
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
