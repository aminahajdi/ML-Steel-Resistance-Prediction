# =============================================================================
# FIXED DNN FOR JOINT DATASET - MOMENT RESISTANCE
# MULTI-SEED 80:20 VERSION - STANDARD SCALER
#
# Purpose:
# - Use the selected fixed DNN hyperparameters.
# - Use StandardScaler for input variables.
# - Use output scaling:
#       y_scaled = (y_real / max_y_train) * 10
# - Evaluate the same DNN over several random 80:20 train/test splits.
# - For each seed: 5-fold CV is performed inside the 80% training pool only.
# - The independent 20% test set is evaluated ONCE at the end for each seed.
# - Report mean ± std train/test performance across seeds.
# - Save one representative train/test plot using REPRESENTATIVE_SEED.
# =============================================================================

import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# 0. SETTINGS
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SIZE = 0.20
N_SPLITS = 5

# Seeds used to evaluate split sensitivity.
SEEDS = [1, 7, 21, 42, 100]

# Prespecified seed used for representative plots and saved artifacts.
REPRESENTATIVE_SEED = 7

BASE_DIR = Path(__file__).resolve().parent

# Location when installed at:
# 04_Welded_Joints/06_Validation_Sensitivity/Multi_Seed_Evaluation/
REPOSITORY_ROOT = BASE_DIR.parents[2]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_04_data.xlsx"
RESULTS_DIR = BASE_DIR / "outputs"

MULTI_SEED_RESULTS_PATH = RESULTS_DIR / "multi_seed_dnn_results.csv"
MULTI_SEED_SUMMARY_PATH = RESULTS_DIR / "multi_seed_dnn_summary.csv"
ALL_CV_RESULTS_PATH = RESULTS_DIR / "multi_seed_dnn_all_cv_results.csv"

# Compact tables for reporting sensitivity to random data partitioning.
DIFFERENT_SEEDS_RESULTS_TABLE_PATH = RESULTS_DIR / "different_seeds_results_table.csv"
DIFFERENT_SEEDS_MEAN_STD_TABLE_PATH = RESULTS_DIR / "different_seeds_mean_std_table.csv"

REPRESENTATIVE_PLOT_PATH = RESULTS_DIR / "representative_seed_dnn_train_test_plot.png"
REPRESENTATIVE_ALL_DATA_TEST_PLOT_PATH = RESULTS_DIR / "representative_seed_dnn_all_data_test_plot.png"
REPRESENTATIVE_RESULTS_TXT_PATH = RESULTS_DIR / "representative_seed_dnn_results.txt"
REPRESENTATIVE_MODEL_PATH = RESULTS_DIR / "representative_seed_dnn_model_state_dict.pt"
REPRESENTATIVE_INPUT_SCALER_PATH = RESULTS_DIR / "representative_seed_input_scaler.pkl"
REPRESENTATIVE_OUTPUT_SCALER_PATH = RESULTS_DIR / "representative_seed_output_scaler_info.pkl"
REPRESENTATIVE_TRAINING_LOSS_PATH = RESULTS_DIR / "representative_seed_training_loss.png"
REPRESENTATIVE_PREDICTIONS_PATH = RESULTS_DIR / "representative_seed_predictions.csv"


# =============================================================================
# 1. DATA SETTINGS
# =============================================================================

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

USE_DATA_FILTERING = False

NUM_BINS = 30
COUNT_THRESHOLD = 5

FILTER_COLUMN_1 = "tfc"
FILTER_COLUMN_2 = "twc"


# =============================================================================
# 2. FIXED BEST DNN HYPERPARAMETERS
# =============================================================================

ARCHITECTURE = (64, 32, 16)
LEARNING_RATE = 0.005
ACTIVATION_NAME = "GELU"
OPTIMIZER_NAME = "AdamW"
WEIGHT_DECAY = 0.0
LOSS_NAME = "relative_mse"

# Training control.
MIN_EPOCHS = 300
MAX_EPOCHS = 5000
EVAL_STEP = 25
START_EVAL_EPOCH = 100
EARLY_STOP_PATIENCE = 500


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)


def compute_metrics(y_true, y_pred):
    """Return bias, CoV, R2, RMSE, and MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ratios = y_pred / y_true

    bias = float(np.mean(ratios))
    cov = float(np.std(ratios) / abs(bias)) if abs(bias) > 1e-12 else float("inf")
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return bias, cov, r2, rmse, mae


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


class RelativeSquaredError(nn.Module):
    """Relative squared error loss."""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        relative_error = (y_true - y_pred) / (y_true + self.epsilon)
        return torch.mean(relative_error ** 2)


class RegressionModel(nn.Module):
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


def get_loss_function(loss_name):
    if loss_name == "relative_mse":
        return RelativeSquaredError()

    raise ValueError(f"Unknown loss function: {loss_name}")


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


def scale_y(y_real, max_y):
    return (y_real / max_y) * 10.0


def unscale_y(y_scaled, max_y):
    return (y_scaled / 10.0) * max_y


def build_model(num_features):
    return RegressionModel(
        num_features=num_features,
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION_NAME,
    ).to(DEVICE)


def predict_real(model, input_scaler, max_y, X_np):
    X_scaled = input_scaler.transform(X_np)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    return unscale_y(y_pred_scaled, max_y)


def train_model_with_validation_selection(
    X_train_np,
    y_train_real,
    X_val_np,
    y_val_real,
    seed,
):
    """
    Train one CV fold and select best epoch by lowest validation CoV.
    This follows the fixed DNN script logic.
    """
    set_all_seeds(seed)

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)
    X_val_scaled = input_scaler.transform(X_val_np)

    max_y = float(np.max(y_train_real))
    y_train_scaled = scale_y(y_train_real, max_y)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    model = build_model(num_features=X_train_np.shape[1])
    criterion = get_loss_function(LOSS_NAME)

    optimizer = get_optimizer(
        OPTIMIZER_NAME,
        model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state_dict = None
    best_epoch = None
    best_val_bias = None
    best_val_cov = float("inf")
    best_val_r2 = None
    best_val_rmse = None
    best_val_mae = None

    early_stop_counter = 0

    history = []

    for epoch in range(MAX_EPOCHS):
        model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

        if epoch >= START_EVAL_EPOCH and epoch % EVAL_STEP == 0:
            model.eval()

            with torch.no_grad():
                y_pred_val_scaled = model(X_val).cpu().numpy().flatten()

            y_pred_val_real = unscale_y(y_pred_val_scaled, max_y)

            val_bias, val_cov, val_r2, val_rmse, val_mae = compute_metrics(
                y_val_real,
                y_pred_val_real,
            )

            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "val_bias": val_bias,
                    "val_cov": val_cov,
                    "val_r2": val_r2,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                }
            )

            if np.isnan(val_cov) or np.isinf(val_cov):
                continue

            if val_cov < best_val_cov:
                best_val_cov = val_cov
                best_val_bias = val_bias
                best_val_r2 = val_r2
                best_val_rmse = val_rmse
                best_val_mae = val_mae
                best_epoch = epoch
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                early_stop_counter = 0
            else:
                early_stop_counter += EVAL_STEP

        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if best_epoch is None:
        best_epoch = MAX_EPOCHS - 1

    return {
        "model": model,
        "input_scaler": input_scaler,
        "max_y": max_y,
        "best_epoch": best_epoch,
        "best_val_bias": best_val_bias,
        "best_val_cov": best_val_cov,
        "best_val_r2": best_val_r2,
        "best_val_rmse": best_val_rmse,
        "best_val_mae": best_val_mae,
        "history": history,
    }


def train_final_model_fixed_epochs(
    X_train_np,
    y_train_real,
    epochs_to_train,
    seed,
):
    """Retrain final DNN on full 80% training pool."""
    set_all_seeds(seed)

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)

    max_y = float(np.max(y_train_real))
    y_train_scaled = scale_y(y_train_real, max_y)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    model = build_model(num_features=X_train_np.shape[1])
    criterion = get_loss_function(LOSS_NAME)

    optimizer = get_optimizer(
        OPTIMIZER_NAME,
        model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_history = []

    for epoch in range(epochs_to_train):
        model.train()

        optimizer.zero_grad()
        y_pred_train_scaled = model(X_train)
        loss = criterion(y_pred_train_scaled, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
            }
        )

    return model, input_scaler, max_y, loss_history


def make_prediction_plot(
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    title,
    save_path,
):
    bias_train, cov_train, r2_train, rmse_train, mae_train = compute_metrics(y_train, y_pred_train)
    bias_test, cov_test, r2_test, rmse_test, mae_test = compute_metrics(y_test, y_pred_test)

    plt.figure(figsize=(7, 7))

    combined_values = np.concatenate((y_train, y_pred_train, y_test, y_pred_test))
    min_val = float(np.min(combined_values))
    max_val = float(np.max(combined_values))
    line_x = np.linspace(min_val, max_val, 200)

    plt.scatter(
        y_pred_train,
        y_train,
        alpha=0.65,
        edgecolors="black",
        s=35,
        label=(
            f"Train data\n"
            f"n={len(y_train)}\n"
            f"b={bias_train:.4f}\n"
            f"CoV={cov_train:.4f}\n"
            f"R²={r2_train:.4f}"
        ),
    )

    plt.scatter(
        y_pred_test,
        y_test,
        alpha=0.85,
        edgecolors="black",
        s=50,
        label=(
            f"Test data\n"
            f"n={len(y_test)}\n"
            f"b={bias_test:.4f}\n"
            f"CoV={cov_test:.4f}\n"
            f"R²={r2_test:.4f}"
        ),
    )

    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="Ideal fit")
    plt.plot(line_x, 1.1 * line_x, "r--", linewidth=1.5, label="±10% error")
    plt.plot(line_x, 0.9 * line_x, "r--", linewidth=1.5)

    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.xlabel("Predicted moment resistance")
    plt.ylabel("Reference moment resistance")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        "train_bias": bias_train,
        "train_cov": cov_train,
        "train_r2": r2_train,
        "train_rmse": rmse_train,
        "train_mae": mae_train,
        "test_bias": bias_test,
        "test_cov": cov_test,
        "test_r2": r2_test,
        "test_rmse": rmse_test,
        "test_mae": mae_test,
    }


def make_all_data_test_plot(
    y_all,
    y_pred_all,
    y_test,
    y_pred_test,
    title,
    save_path,
):
    bias_all, cov_all, r2_all, rmse_all, mae_all = compute_metrics(y_all, y_pred_all)
    bias_test, cov_test, r2_test, rmse_test, mae_test = compute_metrics(y_test, y_pred_test)

    plt.figure(figsize=(7, 7))

    combined_values = np.concatenate((y_all, y_pred_all, y_test, y_pred_test))
    min_val = float(np.min(combined_values))
    max_val = float(np.max(combined_values))
    line_x = np.linspace(min_val, max_val, 200)

    plt.scatter(
        y_pred_all,
        y_all,
        alpha=0.5,
        edgecolors="black",
        s=30,
        label=(
            f"All data\n"
            f"n={len(y_all)}\n"
            f"b={bias_all:.4f}\n"
            f"CoV={cov_all:.4f}\n"
            f"R²={r2_all:.4f}"
        ),
    )

    plt.scatter(
        y_pred_test,
        y_test,
        alpha=0.85,
        edgecolors="black",
        s=45,
        label=(
            f"Test data\n"
            f"n={len(y_test)}\n"
            f"b={bias_test:.4f}\n"
            f"CoV={cov_test:.4f}\n"
            f"R²={r2_test:.4f}"
        ),
    )

    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="Ideal fit")
    plt.plot(line_x, 1.1 * line_x, "r--", linewidth=1.5, label="±10% error")
    plt.plot(line_x, 0.9 * line_x, "r--", linewidth=1.5)

    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.xlabel("Predicted moment resistance")
    plt.ylabel("Reference moment resistance")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        "all_bias": bias_all,
        "all_cov": cov_all,
        "all_r2": r2_all,
        "all_rmse": rmse_all,
        "all_mae": mae_all,
        "test_bias": bias_test,
        "test_cov": cov_test,
        "test_r2": r2_test,
        "test_rmse": rmse_test,
        "test_mae": mae_test,
    }


def train_evaluate_for_seed(X_np, y_np, df_original, seed):
    set_all_seeds(seed)

    X_train_np, X_test_np, y_train_np, y_test_np, train_idx, test_idx = train_test_split(
        X_np,
        y_np,
        np.arange(len(y_np)),
        test_size=TEST_SIZE,
        random_state=seed,
        shuffle=True,
    )

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    cv_records = []

    for fold_id, (fold_train_idx, fold_val_idx) in enumerate(kf.split(X_train_np), start=1):
        X_fold_train = X_train_np[fold_train_idx]
        X_fold_val = X_train_np[fold_val_idx]

        y_fold_train = y_train_np[fold_train_idx]
        y_fold_val = y_train_np[fold_val_idx]

        fold_result = train_model_with_validation_selection(
            X_train_np=X_fold_train,
            y_train_real=y_fold_train,
            X_val_np=X_fold_val,
            y_val_real=y_fold_val,
            seed=seed + fold_id,
        )

        cv_records.append(
            {
                "seed": seed,
                "fold": fold_id,
                "best_epoch": fold_result["best_epoch"],
                "val_bias": fold_result["best_val_bias"],
                "val_cov": fold_result["best_val_cov"],
                "val_r2": fold_result["best_val_r2"],
                "val_rmse": fold_result["best_val_rmse"],
                "val_mae": fold_result["best_val_mae"],
            }
        )

    cv_df = pd.DataFrame(cv_records)

    epochs_to_train = int(round(cv_df["best_epoch"].mean()))
    epochs_to_train = max(epochs_to_train, MIN_EPOCHS)
    epochs_to_train = min(epochs_to_train, MAX_EPOCHS)

    final_model, final_input_scaler, final_max_y, final_loss_history = train_final_model_fixed_epochs(
        X_train_np=X_train_np,
        y_train_real=y_train_np,
        epochs_to_train=epochs_to_train,
        seed=seed,
    )

    y_pred_train = predict_real(final_model, final_input_scaler, final_max_y, X_train_np)
    y_pred_test = predict_real(final_model, final_input_scaler, final_max_y, X_test_np)
    y_pred_all = predict_real(final_model, final_input_scaler, final_max_y, X_np)

    train_bias, train_cov, train_r2, train_rmse, train_mae = compute_metrics(y_train_np, y_pred_train)
    test_bias, test_cov, test_r2, test_rmse, test_mae = compute_metrics(y_test_np, y_pred_test)
    all_bias, all_cov, all_r2, all_rmse, all_mae = compute_metrics(y_np, y_pred_all)

    record = {
        "seed": seed,
        "epochs_to_train": epochs_to_train,
        "cv_val_bias": float(cv_df["val_bias"].mean()),
        "cv_val_cov": float(cv_df["val_cov"].mean()),
        "cv_val_r2": float(cv_df["val_r2"].mean()),
        "cv_val_rmse": float(cv_df["val_rmse"].mean()),
        "cv_val_mae": float(cv_df["val_mae"].mean()),
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
        "all_bias": all_bias,
        "all_cov": all_cov,
        "all_r2": all_r2,
        "all_rmse": all_rmse,
        "all_mae": all_mae,
    }

    predictions_df = df_original.copy()
    predictions_df["prediction"] = y_pred_all
    predictions_df["prediction_to_reference_ratio"] = y_pred_all / y_np
    predictions_df["split"] = "train"
    predictions_df.loc[test_idx, "split"] = "test"

    data = {
        "model": final_model,
        "input_scaler": final_input_scaler,
        "max_y": final_max_y,
        "loss_history": final_loss_history,
        "cv_df": cv_df,
        "X_train": X_train_np,
        "X_test": X_test_np,
        "y_train": y_train_np,
        "y_test": y_test_np,
        "y_all": y_np,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_pred_all": y_pred_all,
        "predictions_df": predictions_df,
        "record": record,
    }

    return record, data, cv_df


def main():
    """Run the five-seed DNN split-sensitivity assessment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # 4. READ AND OPTIONAL FILTER DATA
    # =============================================================================

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Study 4 dataset not found: {DATA_FILE}. "
            "Place the Excel workbook in the repository data/ directory."
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

    df = df[INPUT_COLUMNS + [OUTPUT_COLUMN]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    df = df[df[OUTPUT_COLUMN] > 0].copy()

    if USE_DATA_FILTERING:
        df_full_length = len(df)

        counts_1, bin_edges_1 = np.histogram(df[FILTER_COLUMN_1], bins=NUM_BINS)
        low_count_bins_1 = np.where(counts_1 < COUNT_THRESHOLD)[0]

        if len(low_count_bins_1) > 0:
            cutoff_1 = bin_edges_1[low_count_bins_1[0] + 1]
            df = df[df[FILTER_COLUMN_1] <= cutoff_1].copy()
        else:
            cutoff_1 = None

        counts_2, bin_edges_2 = np.histogram(df[FILTER_COLUMN_2], bins=NUM_BINS)
        low_count_bins_2 = np.where(counts_2 < COUNT_THRESHOLD)[0]

        if len(low_count_bins_2) > 0:
            cutoff_2 = bin_edges_2[low_count_bins_2[0] + 1]
            df = df[df[FILTER_COLUMN_2] <= cutoff_2].copy()
        else:
            cutoff_2 = None

        counts_out, bin_edges_out = np.histogram(df[OUTPUT_COLUMN], bins=NUM_BINS)
        low_count_bins_out = np.where(counts_out < COUNT_THRESHOLD)[0]

        if len(low_count_bins_out) > 0:
            output_cutoff = bin_edges_out[low_count_bins_out[0] + 1]
            df = df[df[OUTPUT_COLUMN] <= output_cutoff].copy()
        else:
            output_cutoff = None

        print("====================================================")
        print("DATA FILTERING")
        print("====================================================")
        print(f"Filtered data length: {len(df)} from {df_full_length}")
        print(f"{FILTER_COLUMN_1} cutoff: {cutoff_1}")
        print(f"{FILTER_COLUMN_2} cutoff: {cutoff_2}")
        print(f"{OUTPUT_COLUMN} cutoff: {output_cutoff}")


    # Position-based split indices refer to row positions in this cleaned table.
    # Reset labels before using .loc on these indices for representative predictions.
    df = df.reset_index(drop=True)

    X_np = df[INPUT_COLUMNS].to_numpy(dtype=float)
    y_np = df[OUTPUT_COLUMN].to_numpy(dtype=float)


    print("====================================================")
    print("FIXED DNN - MULTI-SEED 80:20 JOINT DATASET - STANDARD SCALER")
    print("====================================================")
    print(f"Device                 = {DEVICE}")
    print(f"Dataset file            = {DATA_FILE}")
    print(f"Input columns           = {INPUT_COLUMNS}")
    print(f"Output column           = {OUTPUT_COLUMN}")
    print(f"Total clean samples     = {len(X_np)}")
    print(f"Train/test split        = {int((1 - TEST_SIZE) * 100)}:{int(TEST_SIZE * 100)}")
    print(f"Seeds                   = {SEEDS}")
    print(f"Representative seed     = {REPRESENTATIVE_SEED}")
    print(f"CV folds per seed       = {N_SPLITS}")
    print(f"Input scaler            = StandardScaler")
    print(f"Output scaling          = y_scaled = y / max_y_train * 10")
    print("\nFixed hyperparameters:")
    print(f"Architecture            = {ARCHITECTURE}")
    print(f"Learning rate           = {LEARNING_RATE}")
    print(f"Activation              = {ACTIVATION_NAME}")
    print(f"Optimizer               = {OPTIMIZER_NAME}")
    print(f"Weight decay            = {WEIGHT_DECAY}")
    print(f"Loss                    = {LOSS_NAME}")
    print(f"Max epochs              = {MAX_EPOCHS}")


    # =============================================================================
    # 5. MULTI-SEED EVALUATION
    # =============================================================================

    seed_records = []
    all_cv_records = []
    representative_data = None

    for seed in SEEDS:
        print("\n====================================================")
        print(f"SEED = {seed}")
        print("====================================================")

        record, data, cv_df = train_evaluate_for_seed(
            X_np=X_np,
            y_np=y_np,
            df_original=df,
            seed=seed,
        )

        seed_records.append(record)
        all_cv_records.append(cv_df)

        print(
            f"Seed {seed:>3} | "
            f"epochs={record['epochs_to_train']} | "
            f"CV CoV={record['cv_val_cov']:.4f}, "
            f"CV R²={record['cv_val_r2']:.4f}, "
            f"Test CoV={record['test_cov']:.4f}, "
            f"Test R²={record['test_r2']:.4f}, "
            f"Test bias={record['test_bias']:.4f}"
        )

        if seed == REPRESENTATIVE_SEED:
            representative_data = data


    seed_df = pd.DataFrame(seed_records)
    seed_df.to_csv(MULTI_SEED_RESULTS_PATH, index=False)

    # -------------------------------------------------------------------------
    # Tables formatted like the previous "Different seeds - results" slide
    # -------------------------------------------------------------------------
    different_seeds_results_table = seed_df[
        [
            "seed",
            "train_bias",
            "train_cov",
            "train_r2",
            "test_bias",
            "test_cov",
            "test_r2",
        ]
    ].copy()

    different_seeds_results_table.to_csv(
        DIFFERENT_SEEDS_RESULTS_TABLE_PATH,
        index=False,
    )

    different_seeds_mean_std_table = pd.DataFrame(
        [
            {
                "metric": "bias",
                "train_mean": seed_df["train_bias"].mean(),
                "train_std": seed_df["train_bias"].std(),
                "test_mean": seed_df["test_bias"].mean(),
                "test_std": seed_df["test_bias"].std(),
            },
            {
                "metric": "cov",
                "train_mean": seed_df["train_cov"].mean(),
                "train_std": seed_df["train_cov"].std(),
                "test_mean": seed_df["test_cov"].mean(),
                "test_std": seed_df["test_cov"].std(),
            },
            {
                "metric": "r2",
                "train_mean": seed_df["train_r2"].mean(),
                "train_std": seed_df["train_r2"].std(),
                "test_mean": seed_df["test_r2"].mean(),
                "test_std": seed_df["test_r2"].std(),
            },
        ]
    )

    different_seeds_mean_std_table.to_csv(
        DIFFERENT_SEEDS_MEAN_STD_TABLE_PATH,
        index=False,
    )

    all_cv_df = pd.concat(all_cv_records, ignore_index=True)
    all_cv_df.to_csv(ALL_CV_RESULTS_PATH, index=False)


    summary_rows = []

    for metric in ["bias", "cov", "r2", "rmse", "mae"]:
        summary_rows.append(
            {
                "metric": metric,
                "cv_val_mean": seed_df[f"cv_val_{metric}"].mean(),
                "cv_val_std": seed_df[f"cv_val_{metric}"].std(),
                "train_mean": seed_df[f"train_{metric}"].mean(),
                "train_std": seed_df[f"train_{metric}"].std(),
                "test_mean": seed_df[f"test_{metric}"].mean(),
                "test_std": seed_df[f"test_{metric}"].std(),
                "all_mean": seed_df[f"all_{metric}"].mean(),
                "all_std": seed_df[f"all_{metric}"].std(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(MULTI_SEED_SUMMARY_PATH, index=False)

    # Best seed by test CoV, then test R².
    best_seed_row = seed_df.sort_values(
        by=["test_cov", "test_r2"],
        ascending=[True, False],
    ).iloc[0]


    print("\n====================================================")
    print("MULTI-SEED SUMMARY")
    print("====================================================")
    print(f"Number of seeds = {len(SEEDS)}")
    print(f"Seeds = {SEEDS}")
    print("\nTest-set performance across seeds:")
    print(f"Mean Test CoV  = {seed_df['test_cov'].mean():.4f} ± {seed_df['test_cov'].std():.4f}")
    print(f"Mean Test R²   = {seed_df['test_r2'].mean():.4f} ± {seed_df['test_r2'].std():.4f}")
    print(f"Mean Test Bias = {seed_df['test_bias'].mean():.4f} ± {seed_df['test_bias'].std():.4f}")
    print(f"Mean Test RMSE = {seed_df['test_rmse'].mean():.4f} ± {seed_df['test_rmse'].std():.4f}")
    print(f"Mean Test MAE  = {seed_df['test_mae'].mean():.4f} ± {seed_df['test_mae'].std():.4f}")

    print("\nBest seed by test CoV, for reference only:")
    print(
        f"Seed {int(best_seed_row['seed'])} | "
        f"Test CoV={best_seed_row['test_cov']:.6f}, "
        f"Test R²={best_seed_row['test_r2']:.6f}, "
        f"Test bias={best_seed_row['test_bias']:.6f}"
    )


    # =============================================================================
    # 6. REPRESENTATIVE FINAL PLOT AND MODEL SAVE
    # =============================================================================

    if representative_data is None:
        raise RuntimeError(f"Representative seed {REPRESENTATIVE_SEED} was not included in SEEDS.")

    representative_train_test_metrics = make_prediction_plot(
        y_train=representative_data["y_train"],
        y_pred_train=representative_data["y_pred_train"],
        y_test=representative_data["y_test"],
        y_pred_test=representative_data["y_pred_test"],
        title=f"Fixed DNN - StandardScaler - Representative Split Seed {REPRESENTATIVE_SEED}",
        save_path=REPRESENTATIVE_PLOT_PATH,
    )

    representative_all_test_metrics = make_all_data_test_plot(
        y_all=representative_data["y_all"],
        y_pred_all=representative_data["y_pred_all"],
        y_test=representative_data["y_test"],
        y_pred_test=representative_data["y_pred_test"],
        title=f"Fixed DNN - StandardScaler - All Data and Test - Seed {REPRESENTATIVE_SEED}",
        save_path=REPRESENTATIVE_ALL_DATA_TEST_PLOT_PATH,
    )

    representative_data["predictions_df"].to_csv(REPRESENTATIVE_PREDICTIONS_PATH, index=False)

    # Save representative model and scalers.
    torch.save(
        {
            "model_state_dict": representative_data["model"].state_dict(),
            "input_columns": INPUT_COLUMNS,
            "output_column": OUTPUT_COLUMN,
            "architecture": ARCHITECTURE,
            "activation": ACTIVATION_NAME,
            "input_scaler": "StandardScaler",
            "max_y": representative_data["max_y"],
            "loss": LOSS_NAME,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "test_size": TEST_SIZE,
            "n_splits": N_SPLITS,
            "representative_seed": REPRESENTATIVE_SEED,
            "representative_train_test_metrics": representative_train_test_metrics,
            "representative_all_test_metrics": representative_all_test_metrics,
            "multi_seed_summary": summary_df.to_dict(orient="records"),
        },
        REPRESENTATIVE_MODEL_PATH,
    )

    with open(REPRESENTATIVE_INPUT_SCALER_PATH, "wb") as f:
        pickle.dump(representative_data["input_scaler"], f)

    with open(REPRESENTATIVE_OUTPUT_SCALER_PATH, "wb") as f:
        pickle.dump(
            {
                "max_y": representative_data["max_y"],
                "scaling": "y_scaled = y / max_y * 10",
            },
            f,
        )

    # Training loss plot for representative seed.
    loss_df = pd.DataFrame(representative_data["loss_history"])

    plt.figure(figsize=(8, 5))
    plt.plot(loss_df["epoch"], loss_df["loss"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Relative squared error loss")
    plt.title(f"Representative seed {REPRESENTATIVE_SEED} training loss - StandardScaler")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(REPRESENTATIVE_TRAINING_LOSS_PATH, dpi=300)
    plt.close()


    with open(REPRESENTATIVE_RESULTS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("FIXED DNN MULTI-SEED RESULTS FOR JOINT DATASET - STANDARD SCALER\n")
        f.write("==============================================================\n\n")

        f.write("Methodology:\n")
        f.write("- 80:20 train/test split repeated across multiple random seeds\n")
        f.write("- 5-fold CV inside the training pool for each seed\n")
        f.write("- final independent test set evaluated once per seed\n")
        f.write("- reported performance should be mean ± std across seeds\n")
        f.write("- StandardScaler is used for input variables\n")
        f.write("- output scaling is used: y_scaled = y / max_y_train * 10\n\n")

        f.write(f"Seeds = {SEEDS}\n")
        f.write(f"Representative seed = {REPRESENTATIVE_SEED}\n\n")

        f.write("Fixed hyperparameters:\n")
        f.write(f"Architecture = {ARCHITECTURE}\n")
        f.write(f"Learning rate = {LEARNING_RATE}\n")
        f.write(f"Activation = {ACTIVATION_NAME}\n")
        f.write(f"Optimizer = {OPTIMIZER_NAME}\n")
        f.write(f"Weight decay = {WEIGHT_DECAY}\n")
        f.write(f"Loss = {LOSS_NAME}\n")
        f.write(f"Max epochs = {MAX_EPOCHS}\n")
        f.write(f"Early stop patience = {EARLY_STOP_PATIENCE}\n\n")

        f.write("Different seeds - results table:\n")
        f.write(different_seeds_results_table.to_string(index=False))
        f.write("\n\n")

        f.write("Mean and std of all seeds:\n")
        f.write(different_seeds_mean_std_table.to_string(index=False))
        f.write("\n\n")

        f.write("Full multi-seed summary:\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"{row['metric']}: "
                f"CV = {row['cv_val_mean']:.6f} ± {row['cv_val_std']:.6f}, "
                f"train = {row['train_mean']:.6f} ± {row['train_std']:.6f}, "
                f"test = {row['test_mean']:.6f} ± {row['test_std']:.6f}, "
                f"all = {row['all_mean']:.6f} ± {row['all_std']:.6f}\n"
            )

        f.write("\nBest seed by test CoV, for reference only:\n")
        f.write(
            f"Seed {int(best_seed_row['seed'])}: "
            f"Test CoV={best_seed_row['test_cov']:.6f}, "
            f"Test R²={best_seed_row['test_r2']:.6f}, "
            f"Test bias={best_seed_row['test_bias']:.6f}\n"
        )

        f.write("\nRepresentative train/test metrics:\n")
        for key, value in representative_train_test_metrics.items():
            f.write(f"{key}: {value:.6f}\n")

        f.write("\nRepresentative all/test metrics:\n")
        for key, value in representative_all_test_metrics.items():
            f.write(f"{key}: {value:.6f}\n")


    print("\n====================================================")
    print(f"REPRESENTATIVE 80:20 TRAIN/TEST RESULTS, SEED = {REPRESENTATIVE_SEED}")
    print("====================================================")
    print(
        f"Train data: bias = {representative_train_test_metrics['train_bias']:.4f}, "
        f"CoV = {representative_train_test_metrics['train_cov']:.4f}, "
        f"R² = {representative_train_test_metrics['train_r2']:.6f}, "
        f"RMSE = {representative_train_test_metrics['train_rmse']:.4f}, "
        f"MAE = {representative_train_test_metrics['train_mae']:.4f}"
    )
    print(
        f"Test data:  bias = {representative_train_test_metrics['test_bias']:.4f}, "
        f"CoV = {representative_train_test_metrics['test_cov']:.4f}, "
        f"R² = {representative_train_test_metrics['test_r2']:.6f}, "
        f"RMSE = {representative_train_test_metrics['test_rmse']:.4f}, "
        f"MAE = {representative_train_test_metrics['test_mae']:.4f}"
    )


    # =============================================================================
    # 7. DONE
    # =============================================================================

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Multi-seed results saved to: {MULTI_SEED_RESULTS_PATH}")
    print(f"Multi-seed summary saved to: {MULTI_SEED_SUMMARY_PATH}")
    print(f"All CV results saved to: {ALL_CV_RESULTS_PATH}")
    print(f"Different-seeds results table saved to: {DIFFERENT_SEEDS_RESULTS_TABLE_PATH}")
    print(f"Different-seeds mean/std table saved to: {DIFFERENT_SEEDS_MEAN_STD_TABLE_PATH}")
    print(f"Representative plot saved to: {REPRESENTATIVE_PLOT_PATH}")
    print(f"Representative all/test plot saved to: {REPRESENTATIVE_ALL_DATA_TEST_PLOT_PATH}")
    print(f"Representative predictions saved to: {REPRESENTATIVE_PREDICTIONS_PATH}")
    print(f"Representative model saved to: {REPRESENTATIVE_MODEL_PATH}")
    print(f"Representative input scaler saved to: {REPRESENTATIVE_INPUT_SCALER_PATH}")
    print(f"Representative output scaler saved to: {REPRESENTATIVE_OUTPUT_SCALER_PATH}")
    print(f"Representative training loss plot saved to: {REPRESENTATIVE_TRAINING_LOSS_PATH}")
    print(f"Representative text results saved to: {REPRESENTATIVE_RESULTS_TXT_PATH}")
    print(f"Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
