# Study 1 — IDEA StatiCa CWC: DNN Hyperparameter Sensitivity

This directory contains five manually configured sensitivity analyses supporting conventional deep neural network (DNN) development for column-web transverse-compression resistance prediction in *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Scripts

| Script | Parameter investigated |
|---|---|
| `architecture_sensitivity.py` | Hidden-layer architectures |
| `activation_sensitivity.py` | Activation functions |
| `optimizer_sensitivity.py` | Optimization algorithms |
| `learning_rate_sensitivity.py` | Learning rates |
| `weight_decay_sensitivity.py` | Weight-decay coefficients |

## Dataset

All scripts read `data/study_01_data.xlsx` relative to the repository root. They expect the original column names and ordering. In these sensitivity scripts, the **last column is treated as the prediction target** (`OUTPUT_COLUMN = None`); all preceding columns become predictors. The input file should therefore match the dataset schema used in the original experiments. The numerical dataset must be obtained separately if it is not included in the code repository.

## Protocol

Each experiment uses random seed 42 and a 70/15/15 training/validation/test split. A sequential sparse-bin filter is applied to flange thickness (`tf [value]`), web thickness (`tw [value]`), then the selected target: 30 histogram bins and a minimum bin count of five. This filtering occurs **before** data partitioning, and includes the target distribution.

The scripts fit input `StandardScaler` parameters and the target scaling factor (training-target maximum, multiplied to a scale of 10) using **training observations only**. Candidate configurations are compared using validation coefficient of variation (CoV); bias and R² are also reported. Training uses relative squared-error loss, learning-rate scheduling, and epoch selection based on validation CoV. Final test performance is reported for each candidate after restoring its validation-selected weights.

These experiments are **manual hyperparameter sensitivity studies**, not AutoML. The fixed settings and parameter grids differ between scripts and are defined directly in each script. The separate final DNN evaluation is provided in `../final_dnn.py` and uses an 80/20 partition with five-fold cross-validation.

## Execution

From the repository root, install dependencies using `pip install -r requirements.txt`, then run, for example:

```bash
python 01_IDEA_CWC/02_Conventional_Model_Development/01_DNN/sensitivity/architecture_sensitivity.py
```

Replace the script name to run the other four analyses. Each script writes its CSV results and figures to a dedicated subdirectory under `sensitivity/outputs/`.

## Reproducibility and interpretation

The random seeds, model settings, data filtering, and evaluation procedures remain specified in the scripts. Test results for the individual hyperparameter candidates are descriptive; comparisons and candidate selection are based on validation performance. The global pre-split filtering step should be considered when interpreting generalization claims. Numerical reproduction requires the original dataset and compatible library versions; computational results have not been independently rerun for this release.
