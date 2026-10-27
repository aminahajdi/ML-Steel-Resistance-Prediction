# Study 4 — Welded Joints: H2O AutoML

This directory contains the H2O AutoML benchmark for numerical welded-joint moment resistance and a companion script for interpreting the saved selected model, as used in *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Dataset

Place the Study 4 dataset at `data/study_04_data.xlsx` in the repository root. The 15 required inputs are `hc`, `bc`, `twc`, `tfc`, `rc`, `hb1`, `bb1`, `twb1`, `tfb1`, `rb1`, `axial force percentage`, `stiffening`, `joint location`, `moment ratio`, and `fy`; the target is `moment resistance`.

The benchmark excludes rows with missing or non-finite required values and nonpositive target resistance. It does not apply additional tail filtering.

## Model development

An 80/20 training/test split uses **seed 42**. H2O AutoML uses a **3,600-second** time budget, **five-fold cross-validation**, **RMSE** for model ranking and stopping, **4 GB** maximum H2O memory, and all available CPU threads. The analysis computes bias and CoV from predicted-to-reference resistance ratios, alongside R², RMSE and MAE.

**Evaluation protocol:** The test partition is passed to H2O as `leaderboard_frame`. As a result, test scores may influence model ranking, and the reported test performance is not a fully untouched independent model-selection holdout. This preserves the original benchmark configuration.

## Execution

Install the root repository requirements, including H2O and `openpyxl`, and a compatible Java runtime. From the repository root, run:

```bash
python 04_Welded_Joints/03_AutoML/H2O/h2o_benchmark.py
```

After the benchmark has saved the selected model, run:

```bash
python 04_Welded_Joints/03_AutoML/H2O/h2o_shap.py
```

The SHAP script uses the original split and the selected model recorded in `selected_model_path.txt`; it does not retrain AutoML. The H2O selected leader may be a stacked ensemble: support for `shap_summary_plot()` with `background_frame` depends on the H2O model type and installed version. An unsupported combination is reported rather than represented as a completed SHAP explanation.

## Outputs

All results are written to `04_Welded_Joints/03_AutoML/H2O/outputs/seed_42/`.

| File | Contents |
|---|---|
| `h2o_leaderboard_seed42.csv` | H2O leaderboard |
| `h2o_best_model_metrics_seed42.csv` | Training and test performance |
| `h2o_predictions_seed42.csv` | Specimen-level features, predictions and ratios |
| `h2o_true_vs_predicted_seed42.png` | Predicted vs. reference moment resistance |
| `h2o_summary_seed42.txt` | Settings and metrics |
| `H2OModels/` | Saved leader model |
| `selected_model_path.txt` | Identity of saved leader model for follow-on analyses |
| `h2o_variable_importance_seed42.csv` and `.png` | Variable importance, where supported |
| `h2o_shap_summary_saved_model_seed42.png` | SHAP summary, if supported by the saved leader |

The optional `h2o.explain()` visualization is generated during an active session but is not explicitly exported. Specimen-level prediction files and model artifacts should only be published where dataset-sharing permissions permit.

## Reproducibility

Runtime-limited AutoML results can vary across software versions and hardware. Record the versions of Python, H2O and Java, and the available CPU and RAM resources when comparing reruns with the archived reference results. Syntax validation does not establish numerical reproducibility.
