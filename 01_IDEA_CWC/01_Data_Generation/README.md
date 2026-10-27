# Study 1 — IDEA StatiCa CWC: Numerical Data Generation

This directory contains the model template and Python scripts used to generate numerical observations for the **column web in transverse compression (CWC)** benchmark in *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

The workflow parametrically modifies an IDEA StatiCa Connection model, runs connection analyses through the IDEA StatiCa Connection API, and extracts resistance and buckling-related quantities. The resulting numerical database is used separately for the machine-learning analyses described in the paper.

## Files

| File | Purpose |
|---|---|
| `LoadNearUnstiffenedEnd.ideaCon` | IDEA StatiCa Connection model template for loading near an unstiffened column end |
| `LoadNearUnstiffenedEnd.py` | Main sampling, model modification, calculation, and result-extraction script |
| `Functions.py` | Supporting functions, including cross-section substitution and reading section names from Excel |
| `Cross-sections.xlsx` | **Required additional input, not included here:** list of eligible cross-section names |

## Software and input requirements

- A compatible, licensed installation of **IDEA StatiCa Connection** with its Connection REST API service and Python client (`ideastatica_connection_api`). The scripts were developed for an IDEA StatiCa 25.x-era API; other versions may require changes.
- Python with `numpy` and `openpyxl`. `Functions.py` additionally imports `pandas`, `matplotlib`, `scikit-learn`, and `torch`, even though these are not used by the data-generation routine itself.
- The supplied `.ideaCon` template, containing the connection, its named member **`M3`**, parameter definitions **`x`** and **`a`**, and the load-effect arrangement assumed by the script.
- A file named `Cross-sections.xlsx` in the working directory. The script reads cross-section names from **column A, starting at row 5** of the active worksheet; the preceding four rows are skipped. Names must be recognized by the installed IDEA StatiCa section catalogue.

The connection template and the cross-section list determine which models can be generated. The cross-section workbook must be provided separately; its contents cannot be reconstructed from these scripts alone.

## Generation procedure

For each requested calculation, the main script:

1. Randomly selects a cross-section from `Cross-sections.xlsx` and a steel grade from the predefined list.
2. Samples the relative eccentricity parameter **`x`** from 100 logarithmically spaced candidate values between **0.05 and 1.0** (rounded to three decimal places), and **`a`** uniformly between **1 and 5** (rounded to two decimals).
3. Replaces the cross-section and material assigned to member `M3`, then updates the model parameters `x` and `a`.
4. Reads the material strengths `fy` and `fu` from the connection parameters, and computes the limiting strain as

   `epsilon_u = max(0.06, 0.6 * (1 - fy / fu))`

   `epsilon_lim = 0.25 * epsilon_u`

5. Updates the connection settings to stop at the limiting plastic strain and requests a buckling analysis.
6. Adjusts the applied load where required by the analysis check value: multiplies it by `analysis * 1.05` when the check value is below 0.3, or doubles it repeatedly while the check value equals 1.
7. Extracts the resistance and buckling-related results and prints a semicolon-separated record.

The recorded fields are `i`, `steel_grade`, `fy`, `fu`, `epsilon_lim`, `css`, `x`, `a`, `plate_strain`, `analysis`, `F_plRd_CBFEM`, `buckling`, `Fcr`, and `alpha_cr`. Material strengths are printed in MPa, and calculated forces in kN, following the unit conversions in the script.

The nominal resistance is calculated as `F_plRd_CBFEM = AppliedLoad * analysis / -1000`; the critical load is calculated as `Fcr = AppliedLoad * buckling / -1000`, and `alpha_cr = Fcr / F_plRd_CBFEM`.

## Running the script

1. Place `LoadNearUnstiffenedEnd.py`, `Functions.py`, `LoadNearUnstiffenedEnd.ideaCon`, and the separately supplied `Cross-sections.xlsx` in the same working directory.
2. Start the compatible IDEA StatiCa Connection API service and confirm it can be reached at `http://localhost:5000` (the address configured in the script). The script **attaches to an already running service**; it does not launch the service.
3. Set `Number_of_calculations` in `LoadNearUnstiffenedEnd.py` to the required number of attempted analyses. Its provided value is `100`.
4. From that directory, run:

   ```bash
   python LoadNearUnstiffenedEnd.py
   ```

The script prints the results to the terminal rather than saving a database file. To retain the console output, it can be redirected to a text file, for example:

```bash
python LoadNearUnstiffenedEnd.py > generation_log.txt
```

**Important:** The console output includes progress and diagnostic lines, not only data records. `generation_log.txt` is therefore **not directly a clean CSV dataset**. Extract and validate the semicolon-separated result rows before importing the observations into the study database.

## Database preparation and reproducibility

The provided script performs **100 attempted calculations by default**. It does not, by itself, reproduce the full reported Study 1 database of 9,241 initially generated observations or the **9,006 observations retained for machine learning**. Generating the full database requires an adequate section list, additional sampling runs, collection and quality checks, and the database-filtering procedure described in the paper and modelling scripts.

Sampling uses Python's `random` module without an active fixed seed; therefore, rerunning the script generally produces a different sample. The resulting numerical values can also depend on the IDEA StatiCa version, section/material catalogues, model template, and analysis settings. For an auditable new dataset, record the software/client versions, input cross-section list, sampling seed (if one is introduced), model-template version, analysis logs, and filtering decisions.

The script assumes specific API response positions and project entities (including member `M3`, two load-effect member loadings, and named result keys). Users should verify these against their IDEA StatiCa installation and the supplied template. It has no general retry or exception-handling mechanism for failed analyses.

The independently distributed **Study 1 numerical dataset** is distinct from this model-generation code. Refer to the paper and associated data repository for the dataset used in the reported machine-learning experiments.
