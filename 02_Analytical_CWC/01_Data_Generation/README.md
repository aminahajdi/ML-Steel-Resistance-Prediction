# Study 2 — Analytical CWC: Data Generation

This directory contains the Python script and the cross-section workbook used to generate a parametric database of column-web-in-transverse-compression (CWC) resistance values for Study 2 of *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Files

| File | Purpose |
|---|---|
| `generate_analytical_cwc.py` | Samples the input combinations and calculates yielding, buckling, and governing resistances |
| `CWC_inputs_2.xlsx` | Source cross-section table (worksheet `Parametric`) |

The source workbook contains section identifiers and the geometric columns `h_c`, `t_wc`, `b_fc`, `t_fc`, and `r_c`. The script reads the first worksheet, strips surrounding spaces from column headings, converts the five geometric columns to numbers, and excludes rows with missing values in those columns. The `number` and `Section` columns are also retained in the generated output.

## Generation procedure

The script generates **10,000 sampled combinations** (`N_CALCULATIONS = 10000`), with replacement from the valid cross-section records. The Python sampling seed is `42`.

For each sampled section, it assigns:

- A steel grade sampled from S235, S275, S355, S420, S460, S500, S550, S620, S690, and S700; the yield strengths are the nominal values defined in the script (MPa).
- An axial-force ratio `N/Npl` sampled from `0.0, 0.1, ..., 0.7`.
- A beam-flange thickness `t_fb` sampled from the available column-flange thickness values in the source workbook that satisfy `0.70 t_fc ≤ t_fb ≤ 1.30 t_fc`. If no value falls within the range, the nearest available thickness is used.

The analytical calculations use `E = 210000 MPa`, `a_b = 0 mm`, `γ_M0 = γ_M1 = 1.0`, `s = r_c`, `β = 0`, and `ω = 1`. They calculate effective width, column-web depth, axial-force reduction factor, plate slenderness, reduction factor, and yielding/buckling resistances. The governing resistance `F_c_wc_Rd_kN` is the lower of the calculated yielding and buckling resistances.

The expressions and coefficients used for these calculations are implemented explicitly in the Python script. Consult the accompanying paper for their theoretical and design-code context.

## Run

Install Python and the required packages: `numpy`, `pandas`, and `openpyxl` (the last is needed to read and write Excel files). From the repository root, run:

```bash
python 02_Analytical_CWC/01_Data_Generation/generate_analytical_cwc.py
```

The script creates `outputs/analytical_cwc_generated_10000.xlsx` alongside this README and the source files. Its rows include input values, intermediate analytical quantities, resistance values in kN, and the governing mode (`yielding` or `buckling`).

## Relationship to the modelling database

The generated file is the **initial parametric analytical database**, not necessarily the final training matrix. The model-development analyses in Study 2 apply their own documented database filtering and select eight input variables to predict `F_c_wc_Rd_kN`. Those downstream steps are separate from the generator.

The separately distributed, approved modelling dataset should be stored in the repository's `data/` directory as `study_02_data.xlsx`. Generating a fresh dataset does not replace that archived dataset unless equivalence has been verified.

## Reproducibility

Given an unchanged cross-section workbook, the same random seed, and compatible Python packages, the script is intended to reproduce the sampling and calculated output. Preserve workbook row order and values. The exported database is written to a script-relative location so execution does not depend on the current terminal directory.
