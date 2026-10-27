"""Study 2 — Analytical CWC: dataset-size sensitivity of a fixed DNN.

For each sampled database size, reserve 20% of observations and calculate
five-fold cross-validation metrics on the remaining 80%. The saved CoV and R²
are cross-validation results; the held-out 20% is not evaluated here.

Random subsets are sampled separately at each size using a single seeded
random-number generator (they are not nested subsets). Input and target scaling
are fitted on each training fold. See the accompanying README for details.
"""

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# 1. FILES AND FOLDERS
# =============================================================================

# Script: repository/02_Analytical_CWC/05_Dataset_Size/dataset_size_sensitivity.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_02_data.xlsx"

RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
RESULTS_CSV_PATH = RESULTS_DIR / "fixed_dnn_dataset_size_results.csv"
PLOT_PATH = FIGURES_DIR / "fixed_dnn_dataset_size_cov_r2.png"


# =============================================================================
# 2. EXPERIMENT CONFIGURATION
# =============================================================================

TEST_SIZE = 0.20  # Reserved subset, not evaluated in this CV study
N_SPLITS = 5

DATASET_SIZES = [
    1000, 2000, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000
]

USE_DATA_FILTERING = True

NUM_BINS = 30
COUNT_THRESHOLD = 5

FILTER_COLUMN_1 = "t_fc"
FILTER_COLUMN_2 = "t_wc"

OUTPUT_COLUMN = "F_c_wc_Rd_kN"

INPUT_COLUMNS = [
    "h_c",
    "t_wc",
    "b_fc",
    "t_fc",
    "r_c",
    "t_fb",
    "N/Npl",
    "f_y",
]


# =============================================================================
# 3. FIXED DNN CONFIGURATION
# =============================================================================

ARCHITECTURE = (128, 64, 32)

LEARNING_RATE = 0.0005
ACTIVATION_NAME = "ReLU"
OPTIMIZER_NAME = "Adam"
WEIGHT_DECAY = 0.0

MIN_EPOCHS = 1500
MAX_EPOCHS = 5000

EVAL_STEP = 100
START_EVAL_EPOCH = 500
EARLY_STOP_PATIENCE = 1000


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def compute_metrics(y_true, y_pred):
    """Prediction/reference bias, population CoV and resistance-based R²."""
    ratios = y_pred / y_true
    bias = np.mean(ratios)
    cov = np.std(ratios) / abs(bias)
    r2 = r2_score(y_true, y_pred)
    return bias, cov, r2


def get_activation(name):
    if name == "ReLU":
        return nn.ReLU()
    if name == "LeakyReLU":
        return nn.LeakyReLU()
    if name == "ELU":
        return nn.ELU()
    if name == "GELU":
        return nn.GELU()
    if name == "SiLU":
        return nn.SiLU()
    if name == "Tanh":
        return nn.Tanh()

    raise ValueError(f"Unknown activation: {name}")


def get_optimizer(name, model, learning_rate, weight_decay):
    if name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name == "SGD_Momentum":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

    if name == "Adagrad":
        return optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name == "RMSprop":
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if name == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unknown optimizer: {name}")


class RelativeSquaredError(nn.Module):
    """Mean squared relative prediction error in scaled target units."""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        relative_error = (y_true - y_pred) / (y_true + self.epsilon)
        return torch.mean(relative_error ** 2)


class RegressionModel(nn.Module):
    """Fully connected DNN with the fixed architecture specified above."""
    def __init__(self, num_features, architecture, activation_name):
        super().__init__()

        layers = []
        in_features = num_features

        for hidden_size in architecture:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(get_activation(activation_name))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_dnn_fold(
    X_train_np,
    X_val_np,
    y_train_real,
    y_val_real,
    num_features,
):
    # Fit the input and target scaling parameters only on the CV training fold.
    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)
    X_val_scaled = input_scaler.transform(X_val_np)

    max_y = np.max(y_train_real)

    y_train_scaled = (y_train_real / max_y) * 10.0

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)

    y_train = torch.tensor(
        y_train_scaled.reshape(-1, 1),
        dtype=torch.float32
    )

    model = RegressionModel(
        num_features=num_features,
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION_NAME
    )

    criterion = RelativeSquaredError()

    optimizer = get_optimizer(
        name=OPTIMIZER_NAME,
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_cov = float("inf")
    best_bias = None
    best_r2 = None
    best_epoch = None

    # Retain the weights corresponding to the minimum validation CoV.
    best_model_state = copy.deepcopy(model.state_dict())

    early_stop_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()

        optimizer.zero_grad()

        y_pred_train_scaled = model(X_train)

        loss = criterion(
            y_pred_train_scaled,
            y_train
        )

        loss.backward()
        optimizer.step()

        if epoch >= START_EVAL_EPOCH and epoch % EVAL_STEP == 0:
            model.eval()

            with torch.no_grad():
                y_pred_val_scaled = model(X_val).cpu().numpy().flatten()

            y_pred_val_real = (y_pred_val_scaled / 10.0) * max_y

            bias_val, cov_val, r2_val = compute_metrics(
                y_val_real,
                y_pred_val_real
            )

            if np.isnan(cov_val) or np.isinf(cov_val):
                continue

            if cov_val < best_cov:
                best_cov = cov_val
                best_bias = bias_val
                best_r2 = r2_val
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += EVAL_STEP

        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            break

    model.load_state_dict(best_model_state)

    return best_bias, best_cov, best_r2, best_epoch


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # 5. READ AND FILTER DATA
    # =============================================================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)

    missing_input_columns = [
        col for col in INPUT_COLUMNS
        if col not in df.columns
    ]

    if missing_input_columns:
        raise ValueError(
            f"Missing input columns in Excel file: {missing_input_columns}"
        )

    if OUTPUT_COLUMN not in df.columns:
        raise ValueError(
            f"Output column not found in Excel file: {OUTPUT_COLUMN}"
        )

    output_column = OUTPUT_COLUMN

    if USE_DATA_FILTERING:
        df_full_length = len(df)

        counts_1, bin_edges_1 = np.histogram(
            df[FILTER_COLUMN_1],
            bins=NUM_BINS
        )

        low_count_bins_1 = np.where(counts_1 < COUNT_THRESHOLD)[0]

        if len(low_count_bins_1) > 0:
            cutoff_1 = bin_edges_1[low_count_bins_1[0] + 1]
            df = df[df[FILTER_COLUMN_1] <= cutoff_1].copy()
        else:
            cutoff_1 = None

        counts_2, bin_edges_2 = np.histogram(
            df[FILTER_COLUMN_2],
            bins=NUM_BINS
        )

        low_count_bins_2 = np.where(counts_2 < COUNT_THRESHOLD)[0]

        if len(low_count_bins_2) > 0:
            cutoff_2 = bin_edges_2[low_count_bins_2[0] + 1]
            df = df[df[FILTER_COLUMN_2] <= cutoff_2].copy()
        else:
            cutoff_2 = None

        counts_out, bin_edges_out = np.histogram(
            df[output_column],
            bins=NUM_BINS
        )

        low_count_bins_out = np.where(counts_out < COUNT_THRESHOLD)[0]

        if len(low_count_bins_out) > 0:
            output_cutoff = bin_edges_out[low_count_bins_out[0] + 1]
            df = df[df[output_column] <= output_cutoff].copy()
        else:
            output_cutoff = None

        print("====================================================")
        print("DATA FILTERING")
        print("====================================================")
        print(f"Filtered data length: {len(df)} from {df_full_length}")
        print(f"{FILTER_COLUMN_1} cutoff: {cutoff_1}")
        print(f"{FILTER_COLUMN_2} cutoff: {cutoff_2}")
        print(f"{output_column} cutoff: {output_cutoff}")


    # =============================================================================
    # 6. INPUTS AND OUTPUT
    # =============================================================================

    X_df = df[INPUT_COLUMNS].copy()
    y_array = df[output_column].to_numpy(dtype=float)

    X_np = X_df.to_numpy(dtype=float)
    y_np = y_array.copy()

    print("====================================================")
    print("DATA VARIABLES")
    print("====================================================")
    print(f"Input columns           = {INPUT_COLUMNS}")
    print(f"Output column           = {OUTPUT_COLUMN}")
    print(f"Input features          = {X_np.shape[1]}")
    print(f"Available samples       = {len(X_np)}")


    # =============================================================================
    # 7. DATASET-SIZE CROSS-VALIDATION
    # =============================================================================

    # Advance one seeded generator across dataset sizes. Each draw is independent
    # (not a progressively nested subset of the previous draw).
    rng = np.random.default_rng(SEED)

    results = []

    for dataset_size in DATASET_SIZES:

        if dataset_size > len(X_np):
            print(
                f"Skipping {dataset_size}: only {len(X_np)} samples available."
            )
            continue

        print("\n====================================================")
        print(f"DATASET SIZE = {dataset_size}")
        print("====================================================")

        selected_indices = rng.choice(
            np.arange(len(X_np)),
            size=dataset_size,
            replace=False
        )

        X_subset = X_np[selected_indices]
        y_subset = y_np[selected_indices]

        # The independent 20% partition is reserved, not used to compute the curves.
        X_trainval_np, X_test_np, y_trainval_np, y_test_np = train_test_split(
            X_subset,
            y_subset,
            test_size=TEST_SIZE,
            random_state=SEED,
            shuffle=True
        )

        print(f"Train/validation pool   = {len(X_trainval_np)}")
        print(f"Independent test set    = {len(X_test_np)}")

        kf = KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED
        )

        fold_covs = []
        fold_biases = []
        fold_r2s = []
        fold_epochs = []

        start_time = time.time()

        for fold_id, (train_idx, val_idx) in enumerate(
            kf.split(X_trainval_np),
            start=1
        ):
            X_fold_train = X_trainval_np[train_idx]
            X_fold_val = X_trainval_np[val_idx]

            y_fold_train = y_trainval_np[train_idx]
            y_fold_val = y_trainval_np[val_idx]

            bias_val, cov_val, r2_val, best_epoch = train_dnn_fold(
                X_train_np=X_fold_train,
                X_val_np=X_fold_val,
                y_train_real=y_fold_train,
                y_val_real=y_fold_val,
                num_features=X_np.shape[1]
            )

            fold_biases.append(bias_val)
            fold_covs.append(cov_val)
            fold_r2s.append(r2_val)
            fold_epochs.append(best_epoch)

            print(
                f"Fold {fold_id}: "
                f"Bias = {bias_val:.6f}, "
                f"CoV = {cov_val:.6f}, "
                f"R² = {r2_val:.6f}, "
                f"Best epoch = {best_epoch}"
            )

        training_time = time.time() - start_time

        mean_cov = float(np.mean(fold_covs))
        std_cov = float(np.std(fold_covs))

        mean_bias = float(np.mean(fold_biases))
        std_bias = float(np.std(fold_biases))

        mean_r2 = float(np.mean(fold_r2s))
        std_r2 = float(np.std(fold_r2s))

        mean_epoch = float(np.nanmean(fold_epochs))

        print("\nSummary:")
        print(f"Mean CV Bias = {mean_bias:.6f} ± {std_bias:.6f}")
        print(f"Mean CV CoV  = {mean_cov:.6f} ± {std_cov:.6f}")
        print(f"Mean CV R²   = {mean_r2:.6f} ± {std_r2:.6f}")
        print(f"Mean epoch   = {mean_epoch:.1f}")
        print(f"Time         = {training_time:.2f} s")

        results.append({
            "dataset_size": dataset_size,
            "trainval_size": len(X_trainval_np),
            "test_size": len(X_test_np),
            "mean_cv_bias": mean_bias,
            "std_cv_bias": std_bias,
            "mean_cv_cov": mean_cov,
            "std_cv_cov": std_cov,
            "mean_cv_r2": mean_r2,
            "std_cv_r2": std_r2,
            "mean_best_epoch": mean_epoch,
            "training_time_s": training_time,
        })


    # =============================================================================
    # 8. SAVE RESULTS
    # =============================================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_CSV_PATH,
        index=False
    )


    # =============================================================================
    # 9. PLOT FIVE-FOLD CV CoV AND R² VS DATASET SIZE
    # =============================================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 10),
        sharex=True
    )

    axes[0].errorbar(
        results_df["dataset_size"],
        results_df["mean_cv_cov"],
        yerr=results_df["std_cv_cov"],
        marker="o",
        capsize=4,
        linewidth=2
    )

    axes[0].set_ylabel("Mean CV CoV")
    axes[0].set_title("Fixed DNN Dataset Size Study - CoV")
    axes[0].grid(
        True,
        linestyle="--",
        alpha=0.6
    )

    axes[1].errorbar(
        results_df["dataset_size"],
        results_df["mean_cv_r2"],
        yerr=results_df["std_cv_r2"],
        marker="o",
        capsize=4,
        linewidth=2
    )

    axes[1].set_xlabel("Dataset size")
    axes[1].set_ylabel("Mean CV R²")
    axes[1].set_title("Fixed DNN Dataset Size Study - R²")
    axes[1].grid(
        True,
        linestyle="--",
        alpha=0.6
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=300
    )

    plt.close()


    # =============================================================================
    # 10. DONE
    # =============================================================================

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Results saved to: {RESULTS_CSV_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")

if __name__ == "__main__":
    main()
