# Study 2 — Analytical CWC: Dataset-Size Sensitivity

This directory provides the fixed deep neural network (DNN) analysis used to
investigate the influence of database size on resistance prediction for the
analytical column-web-in-transverse-compression (CWC) benchmark accompanying
*A Systematic Framework for Machine Learning-Based Structural Resistance Prediction*.

## Data

Download the authorized Study 2 analytical database separately and place it at
`data/study_02_data.xlsx` in the repository root. Preserve the original row order,
column names and units. The script requires the inputs `h_c`, `t_wc`, `b_fc`,
`t_fc`, `r_c`, `t_fb`, `N/Npl`, and `f_y`, with target `F_c_wc_Rd_kN`.

The script applies the sequential sparse-tail filtering procedure to `t_fc`,
`t_wc`, and `F_c_wc_Rd_kN`, using 30 histogram bins and a threshold of fewer than
five observations. Filtering is performed before sampling dataset sizes.

## Methodology

The evaluated dataset sizes are 1,000 to 10,000 observations in steps of 1,000,
provided enough filtered observations are available. For each size, a separate
random subset is drawn without replacement using a seeded generator (seed 42).
**These subsets are not nested.** Each subset is split 80/20, and five-fold
cross-validation is performed within its 80% training pool. The remaining 20%
is reserved and **not used to generate the reported dataset-size curves**.

The fixed DNN has hidden layers 128–64–32, ReLU activation, Adam optimizer,
learning rate 0.0005, and no weight decay. Training minimizes squared relative
error; model checkpoints are selected using validation CoV. Input standardization
and target scaling are fitted only on each fold's training observations.

The plotted quantities are the **mean five-fold validation CoV and R²**, with
between-fold standard deviations as error bars. They are not independent-test
metrics. The code does not retrain a final model for each dataset size.

## Run

From the repository root, install the required packages (including `openpyxl`
for Excel input), then run:

```bash
python 02_Analytical_CWC/05_Dataset_Size/dataset_size_sensitivity.py
```

## Outputs

- `results/fixed_dnn_dataset_size_results.csv` — mean and standard deviation
  across folds for bias, CoV and R²; mean selected epoch and execution time.
- `figures/fixed_dnn_dataset_size_cov_r2.png` — the two dataset-size curves.

The output files are generated upon execution and should be compared with the
published results using the approved database and tested software environment.
Results can vary with software, hardware and nondeterministic computation.
