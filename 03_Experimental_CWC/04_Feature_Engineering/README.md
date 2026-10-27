# Study 3 — Experimental CWC: MLJAR Golden Features

This directory contains the MLJAR AutoML implementation used to assess the effect of automatically generated **Golden Features** on experimental column-web transverse-compression resistance prediction in *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Dataset

Download the experimental CWC modelling dataset from the accompanying research-data repository and place it at `data/study_03_data.xlsx` in the repository root.

The script uses eight input variables: `h_c`, `t_wc`, `b_fc`, `t_fc`, `r_c`, `t_fb`, `N/Npl`, and `f_y`. The target is experimental resistance `F_exp`.

The modelling dataset was prepared with the influential observation excluded before model development. The script removes missing or nonfinite required values and retains only positive experimental resistance values. It does not perform any additional outlier exclusion.

## Model development

An 80/20 training/test partition is generated with seed **7**. MLJAR AutoML uses **Compete** mode, a **3,600-second** time budget, **five-fold cross-validation**, and **RMSE** for model evaluation.

The candidate algorithm families are linear regression, random forest, extra trees, LightGBM, XGBoost, CatBoost, and neural networks. Automatic feature selection, ensembling, and stacking are enabled. **Golden Features generation is enabled** (`golden_features=True`).

The configuration otherwise matches the corresponding baseline MLJAR experiment (`golden_features=False`), allowing the results to be compared under the same random split and search settings. Since these are time-bounded AutoML runs, enabling feature generation can also alter the allocation of computational time among model candidates.

MLJAR selects models according to RMSE, not resistance-ratio CoV. The script subsequently computes training/test bias and CoV from predicted-to-experimental resistance ratios, as well as R², RMSE, and MAE. The held-out test partition is not passed to `AutoML.fit()`.

## Execution

Install `mljar-supervised`, `openpyxl`, and the scientific Python dependencies specified in the root requirements file.

Run from the repository root:

```bash
python 03_Experimental_CWC/04_Feature_Engineering/mljar_golden_features.py
```

## Outputs

Files are written to `03_Experimental_CWC/04_Feature_Engineering/outputs/seed_7/`:

| File | Contents |
|---|---|
| `mljar_golden_features_metrics_seed7.csv` | Train/test bias, CoV, R², RMSE, and MAE |
| `mljar_golden_features_predictions_seed7.csv` | Reference and predicted resistance values and their ratios |
| `mljar_golden_features_true_vs_predicted_seed7.png` | Resistance prediction comparison |
| `MLJARModels/` | MLJAR models and automatically generated reports |
| `mljar_golden_features_leaderboard_seed7.csv` | Leaderboard, if the installed MLJAR version exposes `get_leaderboard()` |
| `golden_features_extracted_seed7.json` | Copy of `golden_features.json`, if such a file is generated and found in `MLJARModels/` |

MLJAR may store Golden Features information in version-dependent forms. Absence of the optional JSON copy does not establish that Golden Features were not used; inspect the generated MLJAR reports when needed.

Time-limited searches may yield different selected models across software or hardware environments. Record package versions and computational resources and compare reproduced results with the reported reference outputs.
