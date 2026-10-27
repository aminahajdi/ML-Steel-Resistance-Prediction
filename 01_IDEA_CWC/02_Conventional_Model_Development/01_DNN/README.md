# Study 1 — IDEA StatiCa CWC: Conventional DNN Development

This directory contains the fixed DNN training and evaluation script for the IDEA StatiCa column-web transverse-compression resistance database. The associated hyperparameter sensitivity scripts belong in the `sensitivity/` subdirectory.

## Model and evaluation

The selected model uses four hidden layers (128–64–32–16), GELU activation, AdamW optimization, learning rate 0.01, and weight decay 0.01. The script uses an 80/20 train/test partition (seed 42), with five-fold cross-validation within the training portion. The epoch count for final training is derived from the fold-specific validation-CoV selections.

The analysis retains the original sequential sparse-bin filtering (`tf`, `tw`, then resistance), train-only input standardization, and train-only target scaling. Performance is reported through bias, CoV, and R².

## Data and execution

Place the dataset at `data/study_01_data.xlsx` relative to the repository root. Preserve the required column names. Install dependencies using the root `requirements.txt`, then run from the repository root:

```bash
python 01_IDEA_CWC/02_Conventional_Model_Development/01_DNN/final_dnn.py
```

Generated results are written to `outputs/final_dnn/` alongside the script. The dataset is not contained in this code package. Numerical reproduction requires the associated dataset and a compatible environment.
