# Study 1 — IDEA StatiCa CWC: Random Forest Development

This directory contains a conventional Random Forest (RF) grid-search and evaluation script for predicting column-web transverse-compression (CWC) resistance. It accompanies *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Script

`random_forest_grid_search.py` evaluates eight manually specified RF configurations, selects the best configuration by five-fold cross-validation, and retrains it on the complete training portion before evaluating the held-out test set.

## Dataset

The script expects the IDEA StatiCa numerical database at `data/study_01_data.xlsx`, relative to the repository root. It uses nine inputs (`D [value]`, `B [value]`, `tf [value]`, `tw [value]`, `rw [value]`, `x`, `a`, `fy`, `epsilon_lim`) and the target `F_plRd_CBFEM`. Retain the original variable names and row order.

The script sequentially applies the original 30-bin sparse-tail filter (threshold of five observations) to flange thickness, web thickness, and the target. The source database and filtering are described in the paper. Dataset availability depends on the associated data-sharing arrangements.

## Model development and evaluation

Random Forest development involved two stages: an initial broad hyperparameter exploration covering **432 configurations**, followed by a focused search of **eight configurations**. The initial exploration informed the selection of the hyperparameter ranges for the focused search.

The publicly provided `random_forest_grid_search.py` implements only the focused eight-configuration search.

The filtered observations are split 80/20 into training and independent testing (seed 42). Five-fold cross-validation is performed within the training portion. Input `StandardScaler` is fitted on each training fold only and, for final training, on the full training portion only. Although scaling is unnecessary for tree-based models, it is retained from the original experiment.

The focused search evaluates four tree counts (800, 1000, 2000, 4000) and two bootstrap settings (`True` and `False`), with `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, and `max_features=1.0`, resulting in **eight configurations**.

The selection criterion is minimum mean cross-validation CoV, with maximum mean CV R² as the secondary criterion. The selected configuration is subsequently retrained on the complete training portion and evaluated on the independent test set.

Model performance is assessed using prediction bias, coefficient of variation (CoV), and coefficient of determination (R²). Bias and CoV are calculated from prediction-to-reference resistance ratios.

The initial 432-configuration search is documented as part of the model-development process but is not included in the public scripts or outputs.

## Execution

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run from the repository root:

```bash
python 01_IDEA_CWC/02_Conventional_Model_Development/02_Random_Forest/random_forest_grid_search.py
```

## Outputs

Generated files are placed in `outputs/`. They include:

- `random_forest_5fold_cv_results.csv` — cross-validation results for the eight configurations.
- `best_random_forest_5fold_summary.txt` — selected configuration and performance metrics.
- `random_forest_feature_importance.csv` — feature-importance scores.
- Figures illustrating ranked cross-validation results, final predictions, and feature importance.

Random seeds, hyperparameter configurations, data partitioning, filtering, and model-selection criteria are specified in the script. Numerical results may vary depending on the software versions and computational environment.

Further methodological details and interpretation are provided in the accompanying paper.