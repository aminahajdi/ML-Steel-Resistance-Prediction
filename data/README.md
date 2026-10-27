# Benchmark datasets

The four benchmark studies expect their modelling data in this `data/` directory. Source data are not automatically included in the code repository. Access and redistribution are subject to the original data owners' permissions.

| Study | Required local filename | Notes |
|---|---|---|
| 1 — Numerical CWC (IDEA StatiCa) | `study_01_data.xlsx` | The model-generation template, scripts and section input workbook are in `01_IDEA_CWC/01_Data_Generation/`. The complete prepared modelling dataset is not included here. |
| 2 — Analytical CWC | `study_02_data.xlsx` | A database can be generated using `02_Analytical_CWC/01_Data_Generation/`, which includes `CWC_inputs_2.xlsx`. The fresh generated dataset may differ from the archived modelling database. |
| 3 — Experimental CWC | `study_03_data.xlsx` | Experimental observations are not publicly distributed. Only users with authorized access may obtain and use the prepared dataset. |
| 4 — Welded joints | `study_04_data.xlsx` | Obtain the dataset from an authorized source and verify its redistribution terms. |

Keep the column labels, measurement units, sample ordering and any preparation or filtering decisions consistent with the relevant analysis scripts. Input files named above are ignored by the repository-level `.gitignore` and should not be committed without checking the corresponding data-sharing permissions.

The two spreadsheets under the data-generation directories (`Cross-sections.xlsx` and `CWC_inputs_2.xlsx`) are generator inputs, **not** the prepared benchmark modelling databases.

All relative locations are resolved from the repository root in the standardized analysis scripts. Generated files are generally written to the respective analysis `outputs/` directories; where supplied, `reference_results/` contains selected reference material.
