# Study 3 — Experimental CWC: Seed Sensitivity

This directory contains the fixed-DNN seed-sensitivity analysis and SHAP interpretation for experimental column-web transverse-compression resistance in *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Dataset

The experimental CWC modelling dataset is not publicly distributed with this repository. Users with authorized access should place the prepared dataset at `data/study_03_data.xlsx` in the repository root. The influential observation was excluded during database preparation, before these analyses. Both scripts remove missing or nonfinite required values and require positive `F_exp`.

**Seven input features** are used in this experiment: `h_c`, `t_wc`, `b_fc`, `r_c`, `t_fb`, `N/Npl`, and `f_y`; the target is `F_exp`. In contrast to the eight-input AutoML benchmarks, `t_fc` is intentionally **not included** in this DNN configuration. The SHAP script uses the same seven inputs as the trained model.

## Model development

A fixed DNN is evaluated on five shuffled 80/20 training/test partitions with seeds **1, 7, 21, 42, and 100**. Within the training portion of each split, **five-fold cross-validation** estimates a suitable training duration. A final model is fitted to the training portion and evaluated on that seed's held-out 20% test set. Input standardization and target scaling are fitted on training data.

The DNN uses architecture **16–8**, **GELU**, **AdamW**, learning rate **0.001**, weight decay **0**, dropout **0.05**, batch size **4**, and relative mean-squared error loss. Training is controlled by validation CoV, with a minimum of **300**, maximum of **3,000** epochs, and an early-stopping patience of **500** epochs. The script reports bias, CoV, R², RMSE, and MAE for training and test sets. Across-seed standard deviations use the pandas default sample standard deviation (`ddof=1`).

The representative model and its training/test prediction plot are saved for **seed 7**. The representative split does not replace the five-seed performance summary.

## SHAP interpretation

`representative_dnn_shap.py` loads the saved seed-7 DNN and input scaler; it does not repeat model training. It uses a random background sample of up to **50** records and an explanation sample of up to **200** records, selected from the complete prepared dataset. SHAP values are calculated for predicted resistance in physical units. The script saves beeswarm and mean absolute SHAP plots. These explanations are descriptive; the explanation sample is not restricted to the seed-7 test partition.

## Execution

Install the dependencies specified in the repository's root requirements file, including PyTorch, scikit-learn, SHAP, pandas, matplotlib, and `openpyxl`. Run from the repository root:

```bash
python 03_Experimental_CWC/03_Seed_Sensitivity/seed_sensitivity.py
python 03_Experimental_CWC/03_Seed_Sensitivity/representative_dnn_shap.py
```

## Outputs

New runs save results in `03_Experimental_CWC/03_Seed_Sensitivity/outputs/`:

| File | Contents |
|---|---|
| `multi_seed_dnn_results.csv` | Metrics and selected epochs for each random seed |
| `multi_seed_dnn_summary.csv` | Mean and standard deviation of train/test metrics across seeds |
| `representative_seed_dnn_train_test_plot.png` | Prediction comparison for seed 7 |
| `representative_seed_dnn_results.txt` | Settings and summary metrics |
| `representative_seed_dnn_model.pth` | Saved seed-7 DNN checkpoint |
| `representative_seed_input_scaler.pkl` | Input scaler fitted for seed 7 |
| `representative_seed_output_scaler_info.pkl` | Target scaling information |
| `SHAP_results/representative_seed_dnn_shap_beeswarm.png` | SHAP distribution plot |
| `SHAP_results/representative_seed_dnn_shap_bar.png` | Mean absolute SHAP importance |

`reference_results/` contains the supplied historical results, trained checkpoint, scaler files, and figures. The SHAP script uses a new checkpoint from `outputs/` when available and otherwise falls back to the provided reference checkpoint. Load checkpoint and pickle files only from trusted sources.

## Reproducibility

The archived results span five seeds, although the supplied working copy of the training script listed only seed 7. The public script explicitly lists the five seeds documented in the archived summary. The original SHAP working copy included `t_fc`, which does not match the saved DNN's seven-feature input configuration; this has been aligned with the checkpoint. Numerical results were not independently reproduced because the modelling workbook was not included in this archive. Hardware, package versions and GPU availability may affect DNN training results.
