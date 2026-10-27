# Study 4 — Welded Joints: No Output Scaling

This analysis evaluates the effect of omitting target scaling when training a fixed deep neural network (DNN) for welded-joint moment-resistance prediction. Its intended comparison is the fixed DNN with target scaling documented in `02_Fixed_DNN/`.

## Dataset

The script expects `data/study_04_data.xlsx` at the repository root. It requires the 15 model inputs listed in the script and the response column `moment resistance`. The database is not bundled with this package; its distribution is subject to the owner's permissions.

## Method

The fixed DNN has hidden layers (64, 32, 16), GELU activations, AdamW optimizer (learning rate 0.005; weight decay 0), and relative squared error loss. The random seed is 42. A shuffled 80/20 train/test split is used, with five-fold cross-validation on the training portion. The epoch minimizing validation CoV is selected in each fold, and the final DNN is retrained on the full training portion for the rounded mean number of selected epochs (at least 300). Original epoch evaluation and stopping criteria are retained.

Inputs are standardized using `StandardScaler`, fitted separately to each training fold and subsequently to the full training pool. **The target is not scaled**: training and predictions use the original moment-resistance units. Optional sparse-tail filtering remains disabled (`USE_DATA_FILTERING = False`), as in the source experiment.

Evaluation reports bias (mean predicted/reference ratio), CoV of this ratio, R², RMSE, and MAE. The script also reports metrics on all records, which should not be interpreted as independent test performance.

## Run

From the repository root:

```bash
python 04_Welded_Joints/04_Preprocessing/No_Output_Scaling/dnn_no_output_scaling.py
```

Required Python packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `joblib`, and `openpyxl`. See the repository-level dependency file.

## Outputs

The script writes its results locally under `outputs/`: per-fold CV results (`.csv`), a performance-metrics summary (`.txt`), prediction and training-loss plots (`.png`), individual predictions (`.csv`), a fitted input scaler (`.pkl`), and a model checkpoint (`.pt`). Aggregate metrics and plots may be placed in `reference_results/` if suitable for public release.

Outputs containing row-level data, trained model parameters, and fitted scalers should not be committed when data-sharing permissions do not allow their release. No UMAP analysis is part of this script or of this section.

## Reproducibility

All training settings and calculations are specified in the script. Reproducing numerical results requires the original dataset and a compatible software environment. The code has been syntax-checked, but model training and numerical equivalence have not been independently verified.
