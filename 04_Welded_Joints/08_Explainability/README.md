# Study 4 — Welded Joints: DNN Explainability (SHAP)

This directory documents SHAP feature attribution for the fixed deep neural network (DNN) predicting welded-joint moment resistance. The model configuration is the Optuna-selected 64–32–16 architecture with GELU activation, trained in `02_Fixed_DNN/`.

## Data and model dependencies

The analysis reads `data/study_04_data.xlsx` at the repository root. The workbook must contain the 15 inputs defined in `dnn_shap.py` and the target `moment resistance`.

Before running SHAP, execute `04_Welded_Joints/02_Fixed_DNN/fixed_dnn.py` with the same underlying dataset. Its generated model checkpoint and fitted input scaler must be available as:

- `04_Welded_Joints/02_Fixed_DNN/outputs/fixed_dnn_model_state_dict.pt`
- `04_Welded_Joints/02_Fixed_DNN/outputs/fixed_dnn_input_scaler.pkl`

The pretrained checkpoint must contain the original training maximum (`max_y`) to invert the model's output scaling. The script checks the stored feature order and target name.

## Procedure

The script reconstructs the stored model and predicts moment resistance in the original target units. It recreates the 80/20 train/test division using seed 42. With NumPy's seed-42 random generator, it samples up to 100 background observations from the training partition and up to 500 observations for explanation from the test partition. SHAP `KernelExplainer` is then run with `nsamples=300`. It exports mean absolute feature-attribution values and the original beeswarm, bar, and first-sample waterfall visualizations.

These explanations characterize model behavior relative to the sampled background distribution; they do not establish a causal effect of any input. Computational time may be substantial.

## Run

From the repository root:

```bash
python 04_Welded_Joints/02_Fixed_DNN/fixed_dnn.py
python 04_Welded_Joints/08_Explainability/SHAP/dnn_shap.py
```

Install the dependencies from the repository requirements, including `shap`, `torch`, `scikit-learn`, `joblib`, `pandas`, `numpy`, `matplotlib`, and `openpyxl`.

## Outputs and sharing

Runtime results are saved under `outputs/`: `shap_mean_absolute_importance.csv`, `shap_beeswarm.png`, `shap_bar_mean_abs.png`, `shap_waterfall_first_sample.png`, and `shap_summary.txt`. The script also generates `shap_values.csv` and `shap_explained_samples.csv`, which contain individual attributions and input values.

The dataset, individual explained samples, model checkpoint, fitted scaler, and specimen-level SHAP values should only be distributed when permitted by the relevant data-sharing arrangements. The public repository may contain the scripts and aggregate feature-importance results without distributing these artifacts. Full numerical reproduction requires the original dataset and compatible software versions.
