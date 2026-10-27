# Study 3 — Experimental CWC: Yielding and Buckling Subsets

This section documents a resistance-mechanism-based evaluation of a fixed deep neural network (DNN) using an experimental column-web transverse-compression (CWC) database, as part of *A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*. The governing resistance mechanism is classified using the Eurocode-based calculations implemented in `classify_yielding_buckling.py`.

## Dataset access

Individual experimental measurements are **not included** in this public directory. Users with authorized access to the experimental database can place the preprocessed Excel file at `data/study_03_data.xlsx` in the repository root.

The dataset must contain the required input variables, experimental resistance `F_exp`, and the additional properties `a_b` and `E` used for analytical classification. The influential observation was excluded during the original database preparation, before the analyses were performed.

## Procedure

1. `classify_yielding_buckling.py` calculates the analytical yielding and buckling resistances and identifies the governing mechanism based on the lower calculated resistance. The resulting subsets are saved locally as `CWC_EC_yielding.xlsx` and `CWC_EC_buckling.xlsx` in `outputs/`.
2. `Yielding/yielding_dnn.py` trains and evaluates the fixed DNN using the yielding-classified subset.
3. `Buckling/buckling_dnn.py` applies the same modelling procedure to the buckling-classified subset.

The DNN evaluation uses five random seeds (1, 7, 21, 42, 100), an 80/20 training/test partition for each seed, and five-fold cross-validation within the training portion. Seed 7 is used as the representative case.

The separate analyses investigate how the governing resistance mechanism influences prediction accuracy.

## Public outputs

Aggregate performance results are provided as `multi_seed_dnn_summary.csv` files in the corresponding `reference_results/` directories for the yielding and buckling subsets.

Individual experimental measurements, specimen-level predictions, prediction scatterplots, saved model parameters, and fitted scalers are not distributed.

The scripts document the analytical classification, model development, and evaluation procedures. Numerical reproduction requires access to the underlying experimental database.

## Execution

Install the required dependencies specified in the repository's root `requirements.txt` file.

Run the following scripts from the repository root after placing the authorized experimental dataset in the expected location:

```bash
python 03_Experimental_CWC/05_Yielding_vs_Buckling/classify_yielding_buckling.py
python 03_Experimental_CWC/05_Yielding_vs_Buckling/Yielding/yielding_dnn.py
python 03_Experimental_CWC/05_Yielding_vs_Buckling/Buckling/buckling_dnn.py
```

The generated Excel files are intermediate datasets used for model development and should not be included in public commits when redistribution of the experimental data is restricted.

## Reproducibility

The analytical classification procedure, DNN configuration, random seeds, data partitions, and evaluation metrics are defined in the supplied scripts.

Results may vary across computational environments. Reproduction of the reported numerical results requires the original experimental observations and compatible software dependencies.