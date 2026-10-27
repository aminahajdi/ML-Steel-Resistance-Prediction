# Study 3 — Experimental CWC: Input Distributions and Ranges

This directory contains the code used to characterize the experimental column-web-in-transverse-compression (CWC) database in *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Analysis

`histograms_and_ranges.py` reads the experimental dataset, retains complete observations across the eight input variables and experimental resistance (`F_exp`), and removes the single observation with the maximum `F_exp`. It then calculates the minimum and maximum of each input variable and plots 30-bin histograms for the eight inputs and `F_exp`.

The observation exclusion is part of the database definition used for this analysis; it is not a statistical outlier test. Keep the original row order and column names so that the same observation is excluded if maximum values are tied.

## Dataset

Place the separately distributed dataset at `data/study_03_data.xlsx` relative to the repository root. Required columns: `h_c`, `t_wc`, `b_fc`, `t_fc`, `r_c`, `t_fb`, `N/Npl`, `f_y`, `F_exp`. The workbook is not included here.

## Run

From the repository root, after installing the project's requirements:

```bash
python 03_Experimental_CWC/01_Database_Analysis/02_Histograms/histograms_and_ranges.py
```

The script requires `numpy`, `pandas`, `matplotlib`, and `openpyxl`.

## Outputs

Generated files are placed in `outputs/`:

- `Experimental_CWC_histograms.png` — 3 × 3 panel of input/target distributions.
- `Experimental_CWC_input_ranges.csv` — input minimum and maximum values.
- `Experimental_CWC_input_ranges.xlsx` — the same ranges in Excel format.

The `outputs/` folder contains archived figures and tables. They are not regenerated simply by packaging the scripts. Verify newly generated outputs against these results using the authorized dataset before citing a reproduced figure.
