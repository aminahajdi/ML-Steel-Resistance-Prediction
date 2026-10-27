"""Study 4: training-dataset-size sensitivity of the fixed DNN.

Uses nested training subsets and one fixed 20% test partition. Performance is
recorded during training. The archived summary selects best epochs by *test*
CoV; this is descriptive test-set monitoring, not independent model selection.
No hyperparameter optimization or cross-validation is conducted in this script.
"""

import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================

SEED = 42
DNN_SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# More deterministic PyTorch behaviour.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. FILES AND FOLDERS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "data" / "study_04_data.xlsx"

RESULTS_DIR = SCRIPT_DIR / "outputs"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_CSV_PATH = RESULTS_DIR / "dataset_size_dnn_results.csv"
HISTORY_CSV_PATH = RESULTS_DIR / "dataset_size_dnn_epoch_history.csv"
SUMMARY_TXT_PATH = RESULTS_DIR / "dataset_size_dnn_summary.txt"

PLOT_COV_SIZE_PATH = RESULTS_DIR / "dataset_size_test_cov_plot.png"
PLOT_R2_SIZE_PATH = RESULTS_DIR / "dataset_size_test_r2_plot.png"

PLOT_COV_EPOCHS_PATH = RESULTS_DIR / "test_cov_over_epochs_by_training_size.png"
PLOT_R2_EPOCHS_PATH = RESULTS_DIR / "test_r2_over_epochs_by_training_size.png"

PLOT_TRAIN_COV_EPOCHS_PATH = RESULTS_DIR / "train_cov_over_epochs_by_training_size.png"
PLOT_TRAIN_R2_EPOCHS_PATH = RESULTS_DIR / "train_r2_over_epochs_by_training_size.png"


# =============================================================================
# 2. SETTINGS
# =============================================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

USE_DATA_FILTERING = False

# Training subset sizes are sampled from the 80% train/validation pool.
# "full" means all available samples in the train/validation pool.
# For your dataset of 2400 total samples, the 80% pool has 1920 samples.
TRAINING_SIZES = [100, 200, 400, 600, 800, 1000, 1200, 1600, "full"]

# If a requested size is larger than the training pool, it is skipped.
# The "full" case is always resolved as the complete train/validation pool.

NUM_BINS = 30
COUNT_THRESHOLD = 5

FILTER_COLUMN_1 = "tfc"
FILTER_COLUMN_2 = "twc"

INPUT_COLUMNS = [
    "hc",
    "bc",
    "twc",
    "tfc",
    "rc",
    "hb1",
    "bb1",
    "twb1",
    "tfb1",
    "rb1",
    "axial force percentage",
    "stiffening",
    "joint location",
    "moment ratio",
    "fy",
]

OUTPUT_COLUMN = "moment resistance"


# =============================================================================
# 3. FINAL DNN HYPERPARAMETERS
# =============================================================================

ARCHITECTURE = (64, 32, 16)
ACTIVATION_NAME = "GELU"
OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0

DNN_EPOCHS = 5000
DNN_EVAL_STEP = 25

OUTPUT_SCALING = True


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def reset_all_seeds(seed):
    """Reset Python, NumPy and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_sparse_bins(df, col, bins=30, count_threshold=5):
    counts, bin_edges = np.histogram(df[col], bins=bins)
    low_count_bins = np.where(counts < count_threshold)[0]

    if len(low_count_bins) == 0:
        return df.copy(), None

    first_low_bin = low_count_bins[0]
    cutoff_value = bin_edges[first_low_bin + 1]
    filtered_df = df[df[col] <= cutoff_value].copy()

    return filtered_df, cutoff_value


def compute_metrics(y_true, y_pred, epsilon=1e-12):
    """
    Bias and CoV are calculated from the prediction/reference ratio:
        ratio = y_pred / y_true

    bias = mean(ratio)
    CoV  = std(ratio) / abs(mean(ratio))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)

    ratios = y_pred / y_true_safe
    bias = np.mean(ratios)

    if np.isnan(bias) or np.isinf(bias) or abs(bias) < epsilon:
        return np.nan, np.nan, np.nan, np.nan, np.nan, ratios

    cov = np.std(ratios) / abs(bias)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return bias, cov, r2, rmse, mae, ratios


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

    raise ValueError(f"Unknown activation function: {name}")


def get_optimizer(name, model, learning_rate, weight_decay):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if name == "SGD_Momentum":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    if name == "Adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {name}")


class RelativeSquaredError(nn.Module):
    """
    Relative squared error loss:
        relative_error = (y_true - y_pred) / y_true
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        relative_error = (y_true - y_pred) / (y_true + self.epsilon)
        return torch.mean(relative_error ** 2)


class DNNRegressor(nn.Module):
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


def train_dnn_and_track_epochs(
    X_train_np,
    y_train_real,
    X_test_np,
    y_test_real,
    training_size,
    epochs_to_train,
):
    """
    Trains one DNN with fixed hyperparameters and tracks train/test metrics
    over epochs.

    Scaling:
    - Input scaler is fitted only on the current training subset.
    - Output is scaled as:
          y_scaled = (y_real / max_y_train) * 10
    """
    reset_all_seeds(DNN_SEED)

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)
    X_test_scaled = input_scaler.transform(X_test_np)

    max_y = np.max(y_train_real)
    y_train_scaled = (y_train_real / max_y) * 10.0

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    model = DNNRegressor(
        num_features=X_train_np.shape[1],
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION_NAME,
    ).to(DEVICE)

    criterion = RelativeSquaredError()

    optimizer = get_optimizer(
        OPTIMIZER_NAME,
        model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    epoch_records = []

    start_time = time.time()

    for epoch in range(1, epochs_to_train + 1):
        model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

        if epoch % DNN_EVAL_STEP == 0 or epoch == 1:
            model.eval()

            with torch.no_grad():
                y_pred_train_scaled_np = model(X_train).cpu().numpy().reshape(-1)
                y_pred_test_scaled_np = model(X_test).cpu().numpy().reshape(-1)

            y_pred_train_real = (y_pred_train_scaled_np / 10.0) * max_y
            y_pred_test_real = (y_pred_test_scaled_np / 10.0) * max_y

            train_bias, train_cov, train_r2, train_rmse, train_mae, train_ratios = compute_metrics(
                y_train_real,
                y_pred_train_real,
            )

            test_bias, test_cov, test_r2, test_rmse, test_mae, test_ratios = compute_metrics(
                y_test_real,
                y_pred_test_real,
            )

            epoch_records.append(
                {
                    "training_size": training_size,
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "train_bias": train_bias,
                    "train_cov": train_cov,
                    "train_r2": train_r2,
                    "train_rmse": train_rmse,
                    "train_mae": train_mae,
                    "test_bias": test_bias,
                    "test_cov": test_cov,
                    "test_r2": test_r2,
                    "test_rmse": test_rmse,
                    "test_mae": test_mae,
                    "test_ratio_min": float(np.min(test_ratios)),
                    "test_ratio_max": float(np.max(test_ratios)),
                }
            )

    training_time = time.time() - start_time

    history_df = pd.DataFrame(epoch_records)

    best_idx = history_df["test_cov"].idxmin()
    best_row = history_df.loc[best_idx]

    final_row = history_df.iloc[-1]

    summary = {
        "training_size": training_size,
        "best_epoch_by_test_cov": int(best_row["epoch"]),
        "best_test_bias": best_row["test_bias"],
        "best_test_cov": best_row["test_cov"],
        "best_test_r2": best_row["test_r2"],
        "best_test_rmse": best_row["test_rmse"],
        "best_test_mae": best_row["test_mae"],
        "final_epoch": int(final_row["epoch"]),
        "final_test_bias": final_row["test_bias"],
        "final_test_cov": final_row["test_cov"],
        "final_test_r2": final_row["test_r2"],
        "final_test_rmse": final_row["test_rmse"],
        "final_test_mae": final_row["test_mae"],
        "training_time_s": training_time,
    }

    return history_df, summary


def plot_metric_over_epochs(history_df, metric_column, ylabel, title, output_path):
    plt.figure(figsize=(9, 6))

    for training_size in sorted(history_df["training_size"].unique()):
        group = history_df[history_df["training_size"] == training_size].sort_values("epoch")

        plt.plot(
            group["epoch"],
            group[metric_column],
            linewidth=2,
            label=f"n={training_size}",
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metric_vs_size(results_df, metric_column, ylabel, title, output_path):
    plot_df = results_df.sort_values("training_size")

    plt.figure(figsize=(8, 5))

    plt.plot(
        plot_df["training_size"],
        plot_df[metric_column],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Training dataset size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    # =============================================================================
    # 5. READ AND OPTIONAL FILTER DATA
    # =============================================================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing Study 4 dataset: {DATA_FILE}. "
            "Place study_04_data.xlsx in the repository's data/ directory."
        )

    if DATA_FILE.suffix.lower() == ".csv":
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.read_excel(DATA_FILE)

    missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]

    if missing_input_columns:
        raise ValueError(f"Missing input columns in dataset: {missing_input_columns}")

    if OUTPUT_COLUMN not in df.columns:
        raise ValueError(f"Output column not found in dataset: {OUTPUT_COLUMN}")

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].dropna().copy()

    if USE_DATA_FILTERING:
        original_len = len(df)

        df, cutoff_1 = remove_sparse_bins(
            df,
            FILTER_COLUMN_1,
            bins=NUM_BINS,
            count_threshold=COUNT_THRESHOLD,
        )

        df, cutoff_2 = remove_sparse_bins(
            df,
            FILTER_COLUMN_2,
            bins=NUM_BINS,
            count_threshold=COUNT_THRESHOLD,
        )

        df, output_cutoff = remove_sparse_bins(
            df,
            OUTPUT_COLUMN,
            bins=NUM_BINS,
            count_threshold=COUNT_THRESHOLD,
        )

        print("====================================================")
        print("DATA FILTERING")
        print("====================================================")
        print(f"Original data length: {original_len}")
        print(f"Filtered data length: {len(df)}")
        print(f"{FILTER_COLUMN_1} cutoff: {cutoff_1}")
        print(f"{FILTER_COLUMN_2} cutoff: {cutoff_2}")
        print(f"{OUTPUT_COLUMN} cutoff: {output_cutoff}")


    X_np = df[INPUT_COLUMNS].to_numpy(dtype=float)
    y_np = df[OUTPUT_COLUMN].to_numpy(dtype=float)


    # =============================================================================
    # 6. FIXED 80:20 TRAIN/TEST SPLIT
    # =============================================================================

    X_trainval_np, X_test_np, y_trainval_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("\n====================================================")
    print("DATASET-SIZE SENSITIVITY ANALYSIS - DNN ONLY")
    print("====================================================")
    print(f"Total samples:             {len(y_np)}")
    print(f"Train/validation pool size:{len(y_trainval_np)}")
    print(f"Fixed test set size: {len(y_test_np)}")
    print(f"Device for DNN:            {DEVICE}")
    print(f"DNN fixed seed:            {DNN_SEED}")
    print(f"Output scaling:            {OUTPUT_SCALING}")
    print(f"No hyperparameter tuning is performed.")

    # Nested sampling:
    # Larger subsets contain the smaller subsets.
    subset_rng = np.random.default_rng(SEED)
    trainval_order = subset_rng.permutation(len(y_trainval_np))

    # Resolve requested sizes.
    resolved_training_sizes = []

    for size in TRAINING_SIZES:
        if isinstance(size, str) and size.lower() == "full":
            resolved_training_sizes.append(len(y_trainval_np))
        elif isinstance(size, int):
            if size <= len(y_trainval_np):
                resolved_training_sizes.append(size)
            else:
                print(f"Skipping training size {size}: larger than training pool ({len(y_trainval_np)}).")
        else:
            raise ValueError(f"Invalid training size: {size}")

    # Remove duplicates while preserving order.
    seen = set()
    resolved_training_sizes = [
        x for x in resolved_training_sizes
        if not (x in seen or seen.add(x))
    ]

    print(f"Training sizes: {resolved_training_sizes}")


    # =============================================================================
    # 7. RUN DATASET-SIZE ANALYSIS
    # =============================================================================

    all_history = []
    all_results = []

    analysis_start = time.time()

    for training_size in resolved_training_sizes:
        print("\n====================================================")
        print(f"TRAINING SIZE: {training_size}")
        print("====================================================")

        if training_size == len(y_trainval_np):
            # Full-size case uses the complete train/validation pool directly.
            X_subset = X_trainval_np
            y_subset = y_trainval_np
        else:
            subset_idx = trainval_order[:training_size]
            X_subset = X_trainval_np[subset_idx]
            y_subset = y_trainval_np[subset_idx]

        history_df, summary = train_dnn_and_track_epochs(
            X_train_np=X_subset,
            y_train_real=y_subset,
            X_test_np=X_test_np,
            y_test_real=y_test_np,
            training_size=training_size,
            epochs_to_train=DNN_EPOCHS,
        )

        all_history.append(history_df)
        all_results.append(summary)

        print(
            f"DNN best test CoV={summary['best_test_cov']:.6f}, "
            f"R²={summary['best_test_r2']:.6f}, "
            f"bias={summary['best_test_bias']:.6f}, "
            f"epoch={summary['best_epoch_by_test_cov']}"
        )

    analysis_time = time.time() - analysis_start

    history_all_df = pd.concat(all_history, ignore_index=True)
    results_df = pd.DataFrame(all_results)

    results_df = results_df[
        [
            "training_size",
            "best_epoch_by_test_cov",
            "best_test_bias",
            "best_test_cov",
            "best_test_r2",
            "best_test_rmse",
            "best_test_mae",
            "final_epoch",
            "final_test_bias",
            "final_test_cov",
            "final_test_r2",
            "final_test_rmse",
            "final_test_mae",
            "training_time_s",
        ]
    ]

    history_all_df.to_csv(HISTORY_CSV_PATH, index=False)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)


    # =============================================================================
    # 8. SAVE PLOTS
    # =============================================================================

    plot_metric_vs_size(
        results_df=results_df,
        metric_column="best_test_cov",
        ylabel="Best test CoV",
        title="Influence of Training Dataset Size on Best Test CoV - DNN",
        output_path=PLOT_COV_SIZE_PATH,
    )

    plot_metric_vs_size(
        results_df=results_df,
        metric_column="best_test_r2",
        ylabel="Best test R²",
        title="Influence of Training Dataset Size on Best Test R² - DNN",
        output_path=PLOT_R2_SIZE_PATH,
    )

    plot_metric_over_epochs(
        history_df=history_all_df,
        metric_column="test_cov",
        ylabel="Test CoV",
        title="Test CoV over Epochs by Training Dataset Size - DNN",
        output_path=PLOT_COV_EPOCHS_PATH,
    )

    plot_metric_over_epochs(
        history_df=history_all_df,
        metric_column="test_r2",
        ylabel="Test R²",
        title="Test R² over Epochs by Training Dataset Size - DNN",
        output_path=PLOT_R2_EPOCHS_PATH,
    )

    plot_metric_over_epochs(
        history_df=history_all_df,
        metric_column="train_cov",
        ylabel="Train CoV",
        title="Train CoV over Epochs by Training Dataset Size - DNN",
        output_path=PLOT_TRAIN_COV_EPOCHS_PATH,
    )

    plot_metric_over_epochs(
        history_df=history_all_df,
        metric_column="train_r2",
        ylabel="Train R²",
        title="Train R² over Epochs by Training Dataset Size - DNN",
        output_path=PLOT_TRAIN_R2_EPOCHS_PATH,
    )


    # =============================================================================
    # 9. SAVE SUMMARY
    # =============================================================================

    best_overall_idx = results_df["best_test_cov"].idxmin()
    best_overall = results_df.loc[best_overall_idx]

    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("DATASET-SIZE SENSITIVITY ANALYSIS - DNN ONLY\n")
        f.write("============================================\n\n")

        f.write("This analysis uses fixed final DNN hyperparameters.\n")
        f.write("No hyperparameter tuning is performed.\n")
        f.write("The same fixed test set is used for all training sizes.\n")
        f.write("Training subsets are sampled from the same 80% train/validation pool.\n")
        f.write("Nested sampling is used, so larger subsets contain the smaller subsets.\n")
        f.write("The full-size case uses the complete train/validation pool directly.\n")
        f.write("The DNN seed is reset to the same value before every training run.\n\n")

        f.write("DATA PROTOCOL\n")
        f.write("=============\n")
        f.write(f"Total samples: {len(y_np)}\n")
        f.write(f"Train/validation samples: {len(y_trainval_np)}\n")
        f.write(f"Test samples: {len(y_test_np)}\n")
        f.write(f"Training sizes: {resolved_training_sizes}\n")
        f.write(f"DNN epochs per size: {DNN_EPOCHS}\n")
        f.write(f"DNN evaluation step: {DNN_EVAL_STEP}\n")
        f.write(f"DNN fixed initialization seed: {DNN_SEED}\n")
        f.write(f"Output scaling: {OUTPUT_SCALING}\n")
        f.write(f"Total analysis time [s]: {analysis_time:.6f}\n\n")

        f.write("FINAL DNN HYPERPARAMETERS\n")
        f.write("=========================\n")
        f.write(f"Architecture: {ARCHITECTURE}\n")
        f.write(f"Activation: {ACTIVATION_NAME}\n")
        f.write(f"Optimizer: {OPTIMIZER_NAME}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"Epochs: {DNN_EPOCHS}\n\n")

        f.write("BEST OVERALL RESULT BY TEST CoV\n")
        f.write("===============================\n")
        f.write(f"Training size: {int(best_overall['training_size'])}\n")
        f.write(f"Best epoch: {int(best_overall['best_epoch_by_test_cov'])}\n")
        f.write(f"Best test bias: {best_overall['best_test_bias']:.6f}\n")
        f.write(f"Best test CoV: {best_overall['best_test_cov']:.6f}\n")
        f.write(f"Best test R²: {best_overall['best_test_r2']:.6f}\n")
        f.write(f"Best test RMSE: {best_overall['best_test_rmse']:.6f}\n")
        f.write(f"Best test MAE: {best_overall['best_test_mae']:.6f}\n\n")

        f.write("RESULTS\n")
        f.write("=======\n")
        f.write(results_df.to_string(index=False))


    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Epoch history saved to: {HISTORY_CSV_PATH}")
    print(f"Results saved to:       {RESULTS_CSV_PATH}")
    print(f"CoV-size plot saved to:  {PLOT_COV_SIZE_PATH}")
    print(f"R²-size plot saved to:   {PLOT_R2_SIZE_PATH}")
    print(f"CoV-epoch plot saved to: {PLOT_COV_EPOCHS_PATH}")
    print(f"R²-epoch plot saved to:  {PLOT_R2_EPOCHS_PATH}")
    print(f"Summary saved to:       {SUMMARY_TXT_PATH}")
    print(f"Results folder:         {RESULTS_DIR}")


if __name__ == "__main__":
    main()
