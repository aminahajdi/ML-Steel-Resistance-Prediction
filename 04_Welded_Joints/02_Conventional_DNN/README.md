# Study 4 — Welded Joints: Fixed DNN

This directory contains a fixed deep neural network (DNN) implementation for predicting welded-joint moment resistance, as part of *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*. The network configuration was selected in a preceding Optuna-based model-development experiment; this script evaluates that fixed configuration and does not rerun hyperparameter optimization.

## Dataset

Place the Study 4 database at `data/study_04_data.xlsx` in the repository root. The 15 required input variables are `hc`, `bc`, `twc`, `tfc`, `rc`, `hb1`, `bb1`, `twb1`, `tfb1`, `rb1`, `axial force percentage`, `stiffening`, `joint location`, `moment ratio`, and `fy`. The prediction target is `moment resistance`.

Rows with missing values in these required columns are excluded. The optional sequential histogram-based tail filtering is **disabled** in this experiment (`USE_DATA_FILTERING = False`). No additional dataset filtering is applied by default. The code expects numerical input columns.

## Model development

An 80/20 training/test partition is generated with seed **42**. Five-fold cross-validation is performed within the training partition. Input standardization and target scaling are fitted using each fold's training data only. The DNN is trained with mean squared relative error and the lowest validation resistance-ratio CoV selects the epoch for each fold. The final network is retrained on the complete training partition for the rounded mean of the selected epochs, subject to the minimum epoch count.

| Parameter | Value |
|---|---|
| Architecture | 64–32–16 |
| Activation | GELU |
| Optimizer | AdamW |
| Learning rate | 0.005 |
| Weight decay | 0 |
| Training/test partition | 80/20 |
| Cross-validation | 5 folds |
| Minimum / maximum training epochs per fold | 300 / 5,000 |
| Validation evaluation interval | 25 epochs, starting at epoch 100 |
| Early-stopping patience | 500 epochs |

Evaluation metrics comprise bias and CoV of predicted-to-reference resistance ratios, R², RMSE, and MAE. Results are calculated for the training, independent test, and combined datasets. Combined-dataset scores include training observations and are not independent test metrics.

## Execution

Install the dependencies listed in the repository's root requirements file. From the repository root, run:

```bash
python 04_Welded_Joints/02_Fixed_DNN/fixed_dnn.py
```

## Outputs

Outputs are created in `04_Welded_Joints/02_Fixed_DNN/outputs/`.

| File | Contents |
|---|---|
| `fixed_dnn_5fold_cv_results.csv` | Per-fold validation metrics and selected epochs |
| `fixed_dnn_metrics.txt` | Model configuration, cross-validation summary, and evaluation metrics |
| `fixed_dnn_predictions.csv` | Individual input records, predicted moment resistance, and prediction-to-reference ratios |
| `fixed_dnn_all_data_test_plot.png` | Predictions compared against reference values; test observations highlighted |
| `fixed_dnn_training_loss.png` | Final-network training loss |
| `fixed_dnn_model_state_dict.pt` | Trained PyTorch model weights and metadata |
| `fixed_dnn_input_scaler.pkl` | Fitted input standardizer |

`fixed_dnn_predictions.csv`, the trained weights, and the fitted scaler should not be uploaded to a public repository without considering any applicable dataset-sharing restrictions.

## Reproducibility

The script provides the fixed model configuration, random seed, data partition, five-fold evaluation, and final training procedure. Numerical reproduction requires the original Study 4 database and compatible software versions. The script has been checked for Python syntax but the model has not been retrained as part of preparing this repository version.
