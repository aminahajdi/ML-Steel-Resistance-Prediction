# Study 4 — Welded Joints: MLJAR AutoML with Golden Features

This directory contains the MLJAR AutoML implementation with Golden Features enabled for regression-based prediction of numerical welded-joint moment resistance, as part of *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Dataset

Place the numerical welded-joint dataset at `data/study_04_data.xlsx` in the repository root. The script requires 15 input columns: `hc`, `bc`, `twc`, `tfc`, `rc`, `hb1`, `bb1`, `twb1`, `tfb1`, `rb1`, `axial force percentage`, `stiffening`, `joint location`, `moment ratio`, and `fy`. The prediction target is `moment resistance`.

Rows with missing or non-finite required values and samples with nonpositive target resistance are excluded. No histogram-based tail filtering is applied.

## Model development

An 80/20 training/test split is generated using seed **42**. MLJAR is run in **Compete** mode with a **3,600-second** time budget, **five-fold cross-validation**, and **RMSE** as its internal evaluation metric.

The candidate algorithm families are Linear, Random Forest, Extra Trees, LightGBM, XGBoost, CatBoost, and Neural Network. Golden Features, feature selection, ensembling, and stacking are enabled (`golden_features=True`, `features_selection=True`, `train_ensemble=True`, `stack_models=True`), with `explain_level=2`.

The training and test predictions are evaluated using bias and coefficient of variation (CoV) of predicted-to-reference resistance ratios, R², RMSE, and MAE. This is an AutoML experiment; it is not an Optuna optimization. Comparison with a baseline MLJAR run without Golden Features is meaningful when the same training/test split, time limit, and other settings are used. Under a fixed time budget, the optimization trajectories may still differ.

## Execution

Install `mljar-supervised`, `openpyxl`, and the scientific Python dependencies specified in the repository's root requirements file. From the repository root, run:

```bash
python 04_Welded_Joints/03_AutoML/MLJAR/mljar_golden_features.py
```

## Outputs

Files are written to `04_Welded_Joints/03_AutoML/MLJAR/outputs/seed_42/`:

| File | Contents |
|---|---|
| `mljar_golden_features_metrics_seed42.csv` | Train/test bias, CoV, R², RMSE and MAE |
| `mljar_golden_features_predictions_seed42.csv` | Predictions and individual numerical records |
| `mljar_golden_features_true_vs_predicted_seed42.png` | Predicted versus reference moment resistance |
| `mljar_golden_features_summary_seed42.txt` | Experiment configuration, metrics and extracted features |
| `MLJARModels/` | MLJAR model artifacts and generated reports |
| `mljar_golden_features_leaderboard_seed42.csv` | Leaderboard, if supported by the installed MLJAR version |
| `golden_features_extracted_seed42.json` | Copy of an existing `golden_features.json`, if found in MLJAR outputs |

MLJAR may store feature engineering information in other report files depending on its version. Absence of `golden_features_extracted_seed42.json` does not establish that no feature engineering occurred.

## Reproducibility

The data split, model families, time budget, and evaluation metrics are specified in the script. Time-bounded AutoML results may vary with library versions and compute resources; record the Python, MLJAR, and dependency versions when repeating the analysis.

The prediction CSV contains individual numerical database records; check dataset redistribution permissions before including it in a public repository.
