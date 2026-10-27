# Study 3 — Experimental CWC: H2O AutoML

This directory contains the H2O AutoML implementation used to develop and assess regression models for experimental column-web transverse-compression resistance in *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Dataset

Download the experimental CWC dataset from the accompanying research-data repository and place it at `data/study_03_data.xlsx` in the repository root.

The script requires eight input columns: `h_c`, `t_wc`, `b_fc`, `t_fc`, `r_c`, `t_fb`, `N/Npl`, `f_y`, and the target `F_exp`.

The modelling dataset was prepared by excluding an influential observation prior to model development. The script subsequently removes rows with missing or nonfinite required values and observations with nonpositive `F_exp`. No additional outlier removal is performed.

## Model development

An 80/20 training/test partition is generated using seed **7**. H2O AutoML uses a **3,600-second** time budget, **five internal folds**, and **RMSE** for model ranking and early stopping. The H2O server is initialized with up to **4 GB RAM** and all available CPU threads.

The 20% test partition is provided as the `leaderboard_frame` during H2O AutoML training. This allows test-set performance to influence model ranking, so the reported test metrics should not be interpreted as fully independent of model selection.

The script calculates train/test bias and CoV from predicted-to-experimental resistance ratios, plus R², RMSE, and MAE. Model selection by H2O is based on RMSE, not on resistance-ratio CoV.

## Execution

Install H2O, `openpyxl`, and the scientific Python dependencies specified in the root requirements file. H2O requires a compatible Java runtime.

Run from the repository root:

```bash
python 03_Experimental_CWC/02_AutoML/H2O/h2o_benchmark.py
```

## Outputs

Files are written to `03_Experimental_CWC/02_AutoML/H2O/outputs/seed_7/`:

| File | Contents |
|---|---|
| `h2o_leaderboard_seed7.csv` | H2O leaderboard |
| `h2o_best_model_metrics_seed7.csv` | Train/test bias, CoV, R², RMSE, MAE |
| `h2o_true_vs_predicted_seed7.png` | Resistance prediction comparison |
| `H2OModels/` | Saved H2O leader model (may be large) |
| `h2o_variable_importance_seed7.csv` | Variable importance, when supported by the leader |
| `h2o_variable_importance_seed7.png` | Corresponding importance plot, when available |

An optional `h2o.explain()` report is generated in the active session but is not explicitly exported to a file by this script.

H2O's time-bounded search and its leaderboard may vary across environments. For reproducibility, document installed H2O, Java, and Python versions and CPU/RAM specifications, and compare new runs against the reported reference outputs.