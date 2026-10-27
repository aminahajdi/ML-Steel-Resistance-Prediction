# Study 4 — Welded Joints: Feature Ablation

This analysis evaluates the effect of removing one input variable at a time from the fixed deep neural network (DNN) for numerical joint moment-resistance prediction. The full model uses 15 input variables. Each leave-one-feature-out configuration uses 14 variables; the DNN is trained again for each configuration.

## Dataset

The script reads `data/study_04_data.xlsx` from the repository root. The workbook must contain the 15 named inputs and the target `moment resistance`. Data are not bundled with this folder; use a dataset you are authorized to access and redistribute. Each case drops missing and non-finite values for its selected variables and excludes nonpositive target values.

## Evaluation protocol

- **Split:** one shuffled 80/20 training/test division, random seed **42**, for every feature set.
- **Model:** fixed fully connected DNN, hidden layers **64–32–16**, **GELU**, **AdamW**, learning rate **0.005**, weight decay **0**, relative squared-error loss.
- **Scaling:** input `StandardScaler` fitted only on the applicable training partition; training target scaled as `10 × y / max(y_train)`. Scaling is fitted separately within each CV fold and on each final training partition.
- **Epoch selection:** five-fold CV within the training partition; validation CoV is used for epoch selection (evaluation begins at epoch 100, every 25 epochs, with maximum 5,000 epochs and early stopping patience 500). The final model is retrained on all training observations for an epoch count derived from CV.
- **Comparisons:** a full-input baseline and 15 separate single-feature removals. Each case reports bias, CoV, R², RMSE and MAE; the holdout test partition is evaluated after training.

The feature-ablation analysis uses **one seed**, not an average over multiple splits. The script preserves per-case complete-case filtering: if a removed feature has missing values, the available observations and hence the train/test partition can differ across cases. Interpretation as a controlled feature-only comparison therefore assumes a consistent complete dataset.

## Outputs

Running the script creates `outputs/`, containing:

- `feature_ablation_summary.csv` — full results and changes relative to the full model.
- `feature_ablation_compact_table.csv` — concise comparison table.
- `feature_ablation_case_results_seed42.csv` and `feature_ablation_all_cv_results.csv` — per-case and cross-validation results.
- `feature_ablation_delta_test_cov.png`, `feature_ablation_delta_test_r2.png`, and `feature_ablation_performance_loss_score.png` — comparative plots.
- `feature_ablation_summary.txt` — settings and interpretation notes.

The composite **performance-loss score** is calculated as `Δtest_CoV − Δtest_R²`. It is a descriptive, study-specific ranking measure, not a calibrated or causal feature-importance statistic. Correlated variables can substitute for omitted inputs, and results from a single split have limited stability.

## Execution

From the repository root:

```bash
python 04_Welded_Joints/07_Ablation_Study/ablation_study.py
```

Required packages include NumPy, pandas, Matplotlib, PyTorch, scikit-learn and an Excel reader such as openpyxl (see the repository-level `requirements.txt`).

## Reproducibility

The script provides the fixed model settings, data processing, CV procedure and evaluation calculations. Numerical replication requires the same input database and compatible package versions. Training results have not been independently verified against the archived analysis. The `outputs/` directory is intentionally excluded from source-control commits by default; reviewed aggregate results may be copied into a separately curated `reference_results/` directory.
