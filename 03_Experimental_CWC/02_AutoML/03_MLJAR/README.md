# Study 3 — Experimental CWC: MLJAR AutoML

This directory contains the MLJAR AutoML implementation used to develop and assess regression models for experimental column-web transverse-compression resistance in *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Dataset

Download the experimental CWC dataset from the accompanying research-data repository and place it at `data/study_03_data.xlsx` in the repository root. The script requires eight input columns: `h_c`, `t_wc`, `b_fc`, `t_fc`, `r_c`, `t_fb`, `N/Npl`, `f_y`, and the target `F_exp`.

The modelling dataset was prepared by excluding an influential observation prior to model development. The script subsequently removes rows containing missing or nonfinite required values and observations with nonpositive `F_exp`. No additional outlier removal is performed.

## Model development

An 80/20 training/test partition is generated using seed **7**. MLJAR AutoML uses **Compete** mode, a **3,600-second** total search budget, **five-fold cross-validation**, and **RMSE** as its evaluation metric. All available CPU threads are permitted.

The search includes linear models, random forests, extra trees, LightGBM, XGBoost, CatBoost, and neural networks. Feature selection, ensembling, and model stacking are enabled; golden-feature generation is disabled. MLJAR's explainability level is set to 2.

The script calculates training/test bias and CoV from predicted-to-experimental resistance ratios, together with R², RMSE, and MAE. MLJAR model development is guided by RMSE rather than resistance-ratio CoV. The 20% test subset is not passed to `AutoML.fit()` and is reserved for evaluation after fitting.

## Execution

Install `mljar-supervised`, `openpyxl`, and the scientific Python dependencies specified in the root requirements file. Additional environment requirements depend on the installed MLJAR version and its supported estimators.

Run from the repository root:

```bash
python 03_Experimental_CWC/02_AutoML/MLJAR/mljar_benchmark.py
```

## Outputs

Files are written to `03_Experimental_CWC/02_AutoML/MLJAR/outputs/seed_7/`:

| File | Contents |
|---|---|
| `mljar_best_model_metrics_seed7.csv` | Train/test bias, CoV, R², RMSE, MAE |
| `mljar_predictions_seed7.csv` | Observed and predicted resistance, dataset split, and prediction ratio |
| `mljar_true_vs_predicted_seed7.png` | Resistance prediction comparison |
| `mljar_leaderboard_seed7.csv` | Model leaderboard, if direct export is supported by the installed MLJAR version |
| `MLJARModels/` | Automatically generated models, reports, and supporting files (may be large) |

The direct leaderboard export is optional: the script attempts `automl.get_leaderboard()` and reports an exception if that method is unavailable. MLJAR also generates reports within `MLJARModels/`.

Time-bounded AutoML searches may vary across environments. For reproducibility, record the Python and MLJAR versions, dependencies, and computational resources, and compare new runs against the reported reference outputs.
