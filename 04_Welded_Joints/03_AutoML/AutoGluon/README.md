# Study 4 — Welded Joints: AutoGluon AutoML

This directory contains the AutoGluon AutoML implementation used to develop and evaluate regression models for numerical welded-joint moment resistance in *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Dataset

Place the welded-joint dataset at `data/study_04_data.xlsx` in the repository root. The script requires 15 input columns: `hc`, `bc`, `twc`, `tfc`, `rc`, `hb1`, `bb1`, `twb1`, `tfb1`, `rb1`, `axial force percentage`, `stiffening`, `joint location`, `moment ratio`, and `fy`. The prediction target is `moment resistance`.

Rows with missing or nonfinite values in the required columns and observations with nonpositive moment resistance are excluded. No histogram-based tail filtering is applied.

## Model development

An **80/20 training/test partition** is generated using seed **42**. AutoGluon is configured with a **3,600-second** training budget, the `best_quality` preset, **five bagging folds**, **one stacking level**, and CPU training (`num_gpus=0`). Its internal evaluation metric is **RMSE** (`root_mean_squared_error`).

The model selected by AutoGluon is evaluated on the training and held-out test partitions using bias and coefficient of variation (CoV) of predicted-to-reference resistance ratios, R², RMSE, and MAE. Model selection is based on AutoGluon's internal RMSE evaluation, not on CoV.

Permutation feature importance is calculated using the held-out test partition. These importance values therefore describe sensitivity on the test data and should be interpreted separately from independent test-set performance estimation.

## Execution

Install AutoGluon, `openpyxl`, and the scientific Python dependencies specified in the repository's root requirements file. Run from the repository root:

```bash
python 04_Welded_Joints/03_AutoML/AutoGluon/autogluon_benchmark.py
```

## Outputs

Files are written to `04_Welded_Joints/03_AutoML/AutoGluon/outputs/seed_42/`:

| File | Contents |
|---|---|
| `autogluon_leaderboard_seed42.csv` | AutoGluon model leaderboard evaluated on the test partition |
| `best_model_metrics_seed42.csv` | Train/test bias, CoV, R², RMSE, and MAE |
| `predictions_seed42.csv` | Individual records, predictions, and prediction-to-reference ratios |
| `feature_importance_seed42.csv` | Permutation feature importance on the test partition |
| `autogluon_summary_seed42.txt` | Training configuration, selected model, and metrics |
| `AutogluonModels/` | Saved AutoGluon models (may be large) |

The predictions CSV contains row-level dataset values. Review data-sharing permissions before committing it or the trained models to a public repository.

AutoGluon's time-limited search may vary between software versions and computational environments. Record package versions and computing resources when reproducing the experiment.
