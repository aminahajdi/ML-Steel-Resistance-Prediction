# Study 4 — Welded Joints: Multi-Seed Validation Sensitivity

This analysis evaluates the stability of a fixed deep neural network (DNN) for predicting joint **moment resistance** under changes in the random train/test partition. The model configuration was selected previously and is not retuned during this analysis.

## Dataset

The script expects `data/study_04_data.xlsx` relative to the repository root. The workbook must contain the 15 input columns listed in `dnn_multiseed.py` and the target column `moment resistance`. It removes rows containing missing or nonfinite values and requires positive target values. Optional histogram-based filtering is disabled by default.

## Method

Five random seeds (`1`, `7`, `21`, `42`, and `100`) define independent shuffled **80/20 train/test splits**. Within each 80% training pool, five-fold cross-validation is used to select an epoch count according to the lowest validation coefficient of variation (CoV) of the prediction/reference ratio. A final model is then retrained on that complete training pool and evaluated once on its held-out 20% test partition.

The fixed DNN has hidden layers `(64, 32, 16)`, GELU activation, AdamW optimization, learning rate `0.005`, weight decay `0`, and relative squared-error loss. Inputs are standardized with a `StandardScaler` fitted only on the relevant training fold or final training pool. The output is scaled using `y_scaled = 10 × y / max(y_training)` and converted back for assessment. The maximum training length is 5,000 epochs, with validation inspected every 25 epochs beginning at epoch 100 and an early-stopping patience of 500 epochs. The five folds use different training initialization seeds derived from the outer seed.

Performance is assessed by bias, CoV, R², RMSE, and MAE. Reported mean ± standard deviation across seeds reflects variation in both data partitions and random model initialization. **Seed 7** is prespecified for representative plots and saved model artifacts; it is not selected as the best-performing seed.

## Execution

From the repository root:

```bash
python 04_Welded_Joints/06_Validation_Sensitivity/Multi_Seed_Evaluation/dnn_multiseed.py
```

Dependencies include NumPy, pandas, matplotlib, scikit-learn, PyTorch, and an Excel reader such as openpyxl.

## Generated outputs

The script writes per-seed metrics, cross-validation results, across-seed summary tables, and representative seed-7 plots to `outputs/`. It also writes representative individual predictions, a PyTorch checkpoint, and fitted scaling information for local use. The `outputs/` directory is excluded from version control; curated aggregate results can be copied into a separate `reference_results/` directory if redistribution is permitted.

The script reports the lowest test CoV across seeds as a descriptive comparison only. Choosing a model or seed by that test result would compromise independent test-based model selection.

## Reproducibility

The script documents the modelling settings and partitioning protocol. Reproducing numerical results requires the underlying numerical joint dataset and a compatible software environment; hardware and software versions can affect neural-network results.
