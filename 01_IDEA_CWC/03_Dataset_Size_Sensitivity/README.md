# Study 1 — IDEA StatiCa CWC: Dataset-Size Sensitivity

This directory contains the dataset-size sensitivity analysis for Random Forest (RF), XGBoost, and a fixed deep neural network (DNN), accompanying *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Script

`dataset_size_sensitivity.py` evaluates the three selected model configurations at increasing training-set sizes while keeping the test set fixed. No hyperparameter search is performed in this analysis.

## Dataset

The script expects the IDEA StatiCa CWC numerical database at `data/study_01_data.xlsx`, relative to the repository root. The nine inputs are `D [value]`, `B [value]`, `tf [value]`, `tw [value]`, `rw [value]`, `x`, `a`, `fy`, and `epsilon_lim`; the prediction target is `F_plRd_CBFEM`. Preserve the original variable names and observation order. The numerical database is not included in this directory; access depends on the associated data-sharing arrangements.

Before splitting, the script sequentially applies the original 30-bin sparse-tail filter (threshold: five observations) to `tf [value]`, `tw [value]`, and `F_plRd_CBFEM`.

## Dataset-size evaluation

The filtered database is divided into 80% training and 20% testing using random seed 42. One fixed shuffled ordering of the training pool is used to create **nested subsets** of 1,000, 2,000, 3,000, 4,000, 5,000, 6,000, 7,000, and all available training observations. Requested sizes exceeding the training pool are skipped.

All three model families are evaluated on the **same 20% test set** for every training size. RF and XGBoost use the previously selected hyperparameters; the DNN uses four hidden layers (128–64–32–16), GELU, AdamW, a learning rate of 0.01, weight decay of 0.01, and 4,500 training epochs. Input standardization is fitted on the current training subset only. The DNN target is scaled using the maximum training target value and predictions are converted back to resistance units for evaluation.

The DNN random initialization uses `SEED + size_idx` as in the original experiment, so initialization varies across training sizes. The resulting size trends therefore reflect both training-subset changes and this initialization variation.

Evaluation metrics are prediction bias, coefficient of variation (CoV) of the prediction/reference ratio, and R². The fixed test set is repeatedly evaluated across training sizes; the results describe dataset-size sensitivity rather than an untouched single-use test estimate.

## Execution

From the repository root, install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the analysis:

```bash
python 01_IDEA_CWC/03_Dataset_Size_Sensitivity/dataset_size_sensitivity.py
```

## Outputs

All generated results and figures are saved directly under `03_Dataset_Size_Sensitivity/outputs/`:

| File | Description |
|---|---|
| `dataset_size_sensitivity_results.csv` | Training-size-specific RF, XGBoost, and DNN test metrics and training times |
| `dataset_size_sensitivity_summary.txt` | Protocol, selected model settings, and tabulated results |
| `dataset_size_test_cov_plot.png` | Test CoV versus training dataset size |
| `dataset_size_test_r2_plot.png` | Test R² versus training dataset size |

The script retains the reported sampling, modelling, and evaluation procedure. Numerical equivalence with the archived results requires rerunning the experiment using the original dataset and a compatible software environment.
