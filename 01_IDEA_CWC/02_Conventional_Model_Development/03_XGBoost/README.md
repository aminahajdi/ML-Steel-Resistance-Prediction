# Study 1 — IDEA StatiCa CWC: XGBoost Development

This directory contains the XGBoost implementation for hyperparameter selection, five-fold cross-validation, independent-test evaluation, and native feature-importance analysis for predicting the resistance of column webs in transverse compression (CWC). The code accompanies *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## 1. Dataset

The IDEA StatiCa CWC dataset is distributed separately from this code repository. Save the authorized Excel dataset at `data/study_01_data.xlsx` in the repository root. Preserve the original column names, units, and observation order.

The input variables are `D [value]`, `B [value]`, `tf [value]`, `tw [value]`, `rw [value]`, `x`, `a`, `fy`, and `epsilon_lim`; the target is `F_plRd_CBFEM`.

The script applies sequential sparse-tail filtering to `tf [value]`, `tw [value]`, and the target resistance. For the reported dataset, this produces **9,006 modelling observations**. The filtering rule uses 30 histogram bins and a minimum count of five; it truncates the upper tail at the first sparsely populated bin rather than removing sparse bins individually.

## 2. Model development and validation

The curated modelling dataset is split into **80% training and 20% independent testing**, using random seed 42. **Five-fold cross-validation** within the 80% training subset evaluates candidate hyperparameters. The candidate with the lowest mean CV coefficient of variation (CoV) is selected, with mean CV R² as a secondary ranking criterion. The final model is retrained on the full 80% training subset and evaluated on the held-out 20% test set (1,802 observations).

Input standardization uses `StandardScaler`, fitted **only on each training fold** for cross-validation, and only on the complete training subset for final model fitting. The target is kept in its original resistance units.

### 2.1 Focused hyperparameter search — reported results

The supplied script evaluates a **focused search of nine candidate configurations**, varying the number of boosting trees and the learning rate while fixing the remaining parameters:

```python
N_ESTIMATORS_LIST = [4000, 6000, 8000]
MAX_DEPTH_LIST = [7]
LEARNING_RATE_LIST = [0.01, 0.02, 0.03]
SUBSAMPLE_LIST = [0.8]
COLSAMPLE_BYTREE_LIST = [0.8]
MIN_CHILD_WEIGHT_LIST = [3]
REG_ALPHA_LIST = [0.0]
REG_LAMBDA_LIST = [1.0]
```

The total number of candidates is **3 × 3 = 9**, each evaluated in five CV folds. This is the search supported by the supplied script and results; no separate broader XGBoost search is documented here.

The reference results identify the selected setting as **8,000 boosting trees**, `max_depth=7`, `learning_rate=0.01`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`, `reg_alpha=0.0` and `reg_lambda=1.0` (configuration ID 7). The reported mean CV CoV is **0.029484**; the held-out test results are **bias 1.003370, CoV 0.025801 and R² 0.998145**.

### 2.2 Adapting the search

To investigate additional settings, make a copy of `xgboost_grid_search.py`, change the eight hyperparameter lists under **EXPERIMENT CONFIGURATION**, and give `RESULTS_DIR` in that copy a separate path (for example `Path(__file__).resolve().parent / "outputs_extended"`). Keep the data partition, CV procedure and selection criterion unchanged for comparable experiments. Outputs from modified grids are **new experiments**, not the supplied nine-configuration reference results.

## 3. Execution

Install the dependencies from the repository's root `requirements.txt`. The analysis requires `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, and `openpyxl` for Excel input. From the repository root run:

```bash
python 01_IDEA_CWC/02_Conventional_Model_Development/03_XGBoost/xgboost_grid_search.py
```

New outputs are written to `01_IDEA_CWC/02_Conventional_Model_Development/03_XGBoost/outputs/`. Existing files with the same names in that folder will be overwritten; preserve any published reference results separately before rerunning.

## 4. Outputs

| File | Description |
|---|---|
| `xgboost_5fold_cv_results.csv` | Nine candidate configurations, fold-aggregated validation statistics and final metrics for the selected model |
| `best_xgboost_5fold_summary.txt` | Selected hyperparameters, cross-validation statistics and independent-test results |
| `xgboost_feature_importance.csv` | Native XGBoost feature importance (`feature_importances_`; gain-based for the default tree booster) |
| `A_ranked_mean_cv_cov_xgboost.png` | Configurations ranked by mean five-fold CV CoV |
| `B_best_xgboost_final_prediction_plot.png` | Predicted versus reference resistance for all observations and independent test observations |
| `C_xgboost_feature_importance.png` | Native XGBoost feature-importance scores |

Bias and CoV are computed from the prediction/reference resistance ratio; R² uses the original resistance values. The all-observations metrics include training data and are descriptive; the independent-test results provide the held-out performance assessment. Native feature importance is model-specific and should not be interpreted as a direct physical sensitivity measure.

The reported numerical results have not been regenerated during repository preparation. Reproduction should be confirmed with the distributed dataset and tested software versions before making claims of exact numerical agreement. For methodological context and interpretation, refer to the accompanying paper.
