# Study 2 — Analytical CWC: Optuna-based model selection

This directory contains scripts for **Optuna-based model-family selection and hyperparameter optimization** for analytical column-web transverse-compression resistance prediction. Optuna is an optimization library; this experiment should not be confused with standalone AutoML systems such as AutoGluon, H2O AutoML, or MLJAR.

## Data and experimental protocol

Place the analytical database at `data/study_02_data.xlsx` in the repository root. The target is `F_c_wc_Rd_kN`; the eight input features and sequential sparse-tail filtering are defined in the script. The workflow uses an 80/20 train/test split (seed 42), five-fold cross-validation on the training pool, and a search objective based on mean cross-validation CoV. The selected model is refitted on the training pool before holdout evaluation.

The Optuna search compares multiple model families (DNN, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging, and optional XGBoost, LightGBM, and CatBoost) and their hyperparameters. The **set of available model families depends on installed packages**, so all intended optional dependencies should be installed to reproduce the original search space.

## Run the optimization

From the repository root:

```bash
python "02_Analytical_CWC/04_Optuna/Optuna optimisation.py"
```

The default budget is `TIMEOUT_SECONDS = 900`. To reproduce other time-budget experiments, set this to `360`, `1800`, or `3600` seconds before running.

New outputs are written to `04_Optuna/outputs/<seconds>s/`, keeping separate experiments distinct. Since optimization is time-limited, the number of completed trials can vary with hardware and software versions.

## Files and outputs

- `Optuna optimisation.py` — Optuna model-family and hyperparameter search, final model retraining, and test evaluation.
- `shap_selected_dnn.py` — computes SHAP values only when the saved 3600-second Optuna result selects a DNN; it reads `outputs/3600s/best_params.txt`. It is not applicable to the supplied 900-second LightGBM selection.
- `outputs/<seconds>s/optuna_results.csv` — completed search trials summarized by cross-validation metrics.
- `outputs/<seconds>s/all_optuna_trials.csv` — all Optuna trial records.
- `outputs/<seconds>s/best_params.txt` — selected model family, parameters, and holdout test metrics.
- `outputs/<seconds>s/optuna_convergence_cov_r2.png` — best observed CV CoV and highest observed CV R² across completed trials. The two traces need not correspond to the same model at each trial.
- `outputs/<seconds>s/best_optuna_model_all_data_test_plot.png` — final predictions for all observations and the independent test subset.
- `outputs/<seconds>s/optuna_optimization_history.png`, `optuna_param_importance.png`, `optuna_parallel_coordinate.png`, `optuna_slice_plot.png`, and `optuna_edf_plot.png` — Optuna diagnostic visualizations.

### Archived reference results

The repository also contains results from the four optimization budgets, stored in separate directories:

- `01_outputs - 360 sec run/`
- `02_outputs - 900 sec run/`
- `03_outputs - 1800 sec run/`
- `04_outputs - 3600 sec run/`
- `05_outputs -SHAP - selected run/`

These directories contain previously generated results and are separate from the `outputs/` directory used by new executions. Some archived files retain duplicated extensions such as `.png.png` and `.csv.csv`.

## Reproducibility

The archived 900-second results selected **LightGBM**, with mean CV CoV 0.019786 and test CoV approximately 0.0167. The included source implements the documented search spaces and modelling settings.

Exact numerical reproduction depends on the underlying database, software versions, and computational environment. Results from time-limited optimization may vary between executions.