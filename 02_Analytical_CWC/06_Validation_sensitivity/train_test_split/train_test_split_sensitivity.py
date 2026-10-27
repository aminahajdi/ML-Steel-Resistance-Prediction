"""Study 2: sensitivity of DNN performance to train/test partition size.

The analytical CWC database is filtered using the original sequential
sparse-tail procedure. Three holdout fractions (10%, 20%, 30%) are considered,
with five-fold cross-validation within each training partition. A fixed DNN
configuration is used throughout. See README.md for experiment details.
"""

# =============================================================================
# FIXED BEST DNN - TRAIN/TEST SPLIT SENSITIVITY STUDY
# Fixed:
# 5-fold CV
# 10000 epochs
# batch size = full
# architecture = (256, 128, 64)
# activation = GELU
# optimizer = AdamW
# learning rate = 0.001
# weight decay = 0.0001
# =============================================================================

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


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
DATA_FILE = REPOSITORY_ROOT / "data" / "study_02_data.xlsx"

RESULTS_DIR = SCRIPT_DIR / "outputs"
RESULTS_DIR.mkdir(exist_ok=True)

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CONVERGENCE_DIR = PLOTS_DIR / "convergence"
CONVERGENCE_DIR.mkdir(exist_ok=True)

SUMMARY_CSV_PATH = RESULTS_DIR / "train_test_split_sensitivity_results.csv"
SUMMARY_TXT_PATH = RESULTS_DIR / "best_train_test_split_summary.txt"


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


TEST_SIZE_OPTIONS = [
    0.10,   # 90:10
    0.20,   # 80:20
    0.30,   # 70:30
]

N_SPLITS = 5
MAX_EPOCHS = 10000

MIN_EPOCHS = 1500
START_EVAL_EPOCH = 500
EVAL_STEP = 100
EARLY_STOP_PATIENCE = 1000

BATCH_SIZE = 1024

USE_OUTPUT_SCALING = True
OUTPUT_SCALE = 10.0


ARCHITECTURE = (256, 128, 64)
ACTIVATION = "GELU"
OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def compute_metrics(y_true, y_pred):
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
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "Adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


class RelativeSquaredError(nn.Module):
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


def make_batches(X_tensor, y_tensor, batch_size):
    if batch_size == "full":
        return [(X_tensor, y_tensor)]

    n_samples = X_tensor.shape[0]
    indices = torch.randperm(n_samples)

    batches = []

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        batches.append((X_tensor[batch_idx], y_tensor[batch_idx]))

    return batches


def plot_convergence(history, save_path, title):
    plt.figure(figsize=(8, 6))

    plt.plot(
        history["epoch"],
        history["val_cov"],
        marker="o",
        linewidth=2,
        label="Validation CoV"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation CoV")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train_dnn_fold(
    X_train_np,
    X_val_np,
    y_train_real,
    y_val_real,
    num_features,
    return_model=False,
):

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train_np)
    X_val_scaled = input_scaler.transform(X_val_np)

    if USE_OUTPUT_SCALING:
        max_y = np.max(y_train_real)
        y_train_scaled = (y_train_real / max_y) * OUTPUT_SCALE
    else:
        max_y = None
        y_train_scaled = y_train_real.copy()

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)

    model = RegressionModel(
        num_features=num_features,
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION
    )

    criterion = RelativeSquaredError()

    optimizer = get_optimizer(
        OPTIMIZER_NAME,
        model,
        LEARNING_RATE,
        WEIGHT_DECAY
    )

    best_cov = float("inf")
    best_bias = None
    best_r2 = None
    best_epoch = None
    best_state_dict = copy.deepcopy(model.state_dict())

    early_stop_counter = 0

    history = {
        "epoch": [],
        "val_cov": [],
        "val_r2": [],
        "val_bias": [],
    }

    for epoch in range(MAX_EPOCHS):

        model.train()

        batches = make_batches(
            X_tensor=X_train,
            y_tensor=y_train,
            batch_size=BATCH_SIZE
        )

        for X_batch, y_batch in batches:

            optimizer.zero_grad()

            y_pred_train = model(X_batch)

            loss = criterion(
                y_pred_train,
                y_batch
            )

            loss.backward()
            optimizer.step()

        if epoch >= START_EVAL_EPOCH and epoch % EVAL_STEP == 0:

            model.eval()

            with torch.no_grad():

                y_pred_val_scaled = model(X_val).cpu().numpy().flatten()

                if USE_OUTPUT_SCALING:
                    y_pred_val_real = (y_pred_val_scaled / OUTPUT_SCALE) * max_y
                else:
                    y_pred_val_real = y_pred_val_scaled.copy()

            bias_val, cov_val, r2_val = compute_metrics(
                y_val_real,
                y_pred_val_real
            )

            history["epoch"].append(epoch)
            history["val_cov"].append(cov_val)
            history["val_r2"].append(r2_val)
            history["val_bias"].append(bias_val)

            if np.isnan(cov_val) or np.isinf(cov_val):
                continue

            if cov_val < best_cov:

                best_cov = cov_val
                best_bias = bias_val
                best_r2 = r2_val
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

                early_stop_counter = 0

            else:
                early_stop_counter += EVAL_STEP

        if epoch >= MIN_EPOCHS and early_stop_counter >= EARLY_STOP_PATIENCE:
            break

    model.load_state_dict(best_state_dict)

    if return_model:
        return {
            "model": model,
            "input_scaler": input_scaler,
            "max_y": max_y,
            "best_cov": best_cov,
            "best_bias": best_bias,
            "best_r2": best_r2,
            "best_epoch": best_epoch,
            "history": history,
        }

    return best_cov, best_bias, best_r2, best_epoch, history


def final_train_and_test(
    X_trainval_np,
    X_test_np,
    y_trainval_np,
    y_test_np,
    X_all_np,
    y_all_np,
):

    X_internal_train, X_internal_val, y_internal_train, y_internal_val = train_test_split(
        X_trainval_np,
        y_trainval_np,
        test_size=0.15,
        random_state=SEED,
        shuffle=True
    )

    internal_result = train_dnn_fold(
        X_train_np=X_internal_train,
        X_val_np=X_internal_val,
        y_train_real=y_internal_train,
        y_val_real=y_internal_val,
        num_features=X_all_np.shape[1],
        return_model=True,
    )

    final_epoch = internal_result["best_epoch"]

    epochs_to_train = final_epoch if final_epoch is not None else MIN_EPOCHS
    epochs_to_train = max(int(epochs_to_train), MIN_EPOCHS)

    final_scaler = StandardScaler()

    X_trainval_scaled = final_scaler.fit_transform(X_trainval_np)
    X_test_scaled = final_scaler.transform(X_test_np)
    X_all_scaled = final_scaler.transform(X_all_np)

    if USE_OUTPUT_SCALING:
        final_max_y = np.max(y_trainval_np)
        y_trainval_scaled = (y_trainval_np / final_max_y) * OUTPUT_SCALE
    else:
        final_max_y = None
        y_trainval_scaled = y_trainval_np.copy()

    X_trainval_tensor = torch.tensor(X_trainval_scaled, dtype=torch.float32)
    y_trainval_tensor = torch.tensor(y_trainval_scaled.reshape(-1, 1), dtype=torch.float32)

    final_model = RegressionModel(
        num_features=X_all_np.shape[1],
        architecture=ARCHITECTURE,
        activation_name=ACTIVATION
    )

    criterion = RelativeSquaredError()

    optimizer = get_optimizer(
        OPTIMIZER_NAME,
        final_model,
        LEARNING_RATE,
        WEIGHT_DECAY
    )

    for epoch in range(epochs_to_train):

        final_model.train()

        batches = make_batches(
            X_tensor=X_trainval_tensor,
            y_tensor=y_trainval_tensor,
            batch_size=BATCH_SIZE
        )

        for X_batch, y_batch in batches:

            optimizer.zero_grad()

            y_pred_train = final_model(X_batch)

            loss = criterion(
                y_pred_train,
                y_batch
            )

            loss.backward()
            optimizer.step()

    final_model.eval()

    with torch.no_grad():

        y_pred_test_scaled = final_model(
            torch.tensor(X_test_scaled, dtype=torch.float32)
        ).cpu().numpy().flatten()

        y_pred_all_scaled = final_model(
            torch.tensor(X_all_scaled, dtype=torch.float32)
        ).cpu().numpy().flatten()

    if USE_OUTPUT_SCALING:
        y_pred_test = (y_pred_test_scaled / OUTPUT_SCALE) * final_max_y
        y_pred_all = (y_pred_all_scaled / OUTPUT_SCALE) * final_max_y
    else:
        y_pred_test = y_pred_test_scaled.copy()
        y_pred_all = y_pred_all_scaled.copy()

    bias_test, cov_test, r2_test = compute_metrics(y_test_np, y_pred_test)
    bias_all, cov_all, r2_all = compute_metrics(y_all_np, y_pred_all)

    return {
        "final_epoch": epochs_to_train,
        "test_bias": bias_test,
        "test_cov": cov_test,
        "test_r2": r2_test,
        "all_bias": bias_all,
        "all_cov": cov_all,
        "all_r2": r2_all,
    }


def plot_metric(results_df, metric_col, ylabel, title, save_path):
    plt.figure(figsize=(8, 6))

    x_labels = results_df["train_test_split"].astype(str).tolist()
    y_values = results_df[metric_col].to_numpy()

    plt.plot(
        x_labels,
        y_values,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Train:test split")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================================================================
# READ AND FILTER DATA
# =============================================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_excel(DATA_FILE)

missing_input_columns = [col for col in INPUT_COLUMNS if col not in df.columns]

if missing_input_columns:
    raise ValueError(f"Missing input columns in Excel file: {missing_input_columns}")

if OUTPUT_COLUMN not in df.columns:
    raise ValueError(f"Output column not found in Excel file: {OUTPUT_COLUMN}")

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


X_np = df[INPUT_COLUMNS].to_numpy(dtype=float)
y_np = df[OUTPUT_COLUMN].to_numpy(dtype=float)


# =============================================================================
# RUN TRAIN/TEST SPLIT STUDY
# =============================================================================

records = []

print("\n====================================================")
print("FIXED DNN - TRAIN/TEST SPLIT SENSITIVITY STUDY")
print("====================================================")
print(f"Architecture  = {ARCHITECTURE}")
print(f"Activation    = {ACTIVATION}")
print(f"Optimizer     = {OPTIMIZER_NAME}")
print(f"Learning rate = {LEARNING_RATE}")
print(f"Weight decay  = {WEIGHT_DECAY}")
print(f"Folds         = {N_SPLITS}")
print(f"Max epochs    = {MAX_EPOCHS}")
print(f"Batch size    = {BATCH_SIZE}")
print(f"Splits tested = {TEST_SIZE_OPTIONS}")


for test_size in TEST_SIZE_OPTIONS:

    train_percent = int((1 - test_size) * 100)
    test_percent = int(test_size * 100)
    split_label = f"{train_percent}:{test_percent}"

    print("\n====================================================")
    print(f"RUNNING: train/test split = {split_label}")
    print("====================================================")

    start_time = time.time()

    X_trainval_np, X_test_np, y_trainval_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=test_size,
        random_state=SEED,
        shuffle=True
    )

    fold_covs = []
    fold_biases = []
    fold_r2s = []
    fold_epochs = []

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_trainval_np), start=1):

        X_fold_train = X_trainval_np[train_idx]
        X_fold_val = X_trainval_np[val_idx]

        y_fold_train = y_trainval_np[train_idx]
        y_fold_val = y_trainval_np[val_idx]

        cov_val, bias_val, r2_val, best_epoch, history = train_dnn_fold(
            X_train_np=X_fold_train,
            X_val_np=X_fold_val,
            y_train_real=y_fold_train,
            y_val_real=y_fold_val,
            num_features=X_np.shape[1],
            return_model=False,
        )

        fold_covs.append(cov_val)
        fold_biases.append(bias_val)
        fold_r2s.append(r2_val)
        fold_epochs.append(best_epoch)

        conv_path = CONVERGENCE_DIR / (
            f"convergence_cov_split_{split_label}_fold{fold_id}.png"
        )

        plot_convergence(
            history,
            conv_path,
            f"Validation CoV Convergence\nTrain/test split = {split_label}, fold {fold_id}"
        )

        print(
            f"Fold {fold_id}: "
            f"best_epoch={best_epoch}, "
            f"bias={bias_val:.6f}, "
            f"CoV={cov_val:.6f}, "
            f"R²={r2_val:.6f}"
        )

    cv_mean_bias = float(np.mean(fold_biases))
    cv_std_bias = float(np.std(fold_biases))
    cv_mean_cov = float(np.mean(fold_covs))
    cv_std_cov = float(np.std(fold_covs))
    cv_mean_r2 = float(np.mean(fold_r2s))
    cv_std_r2 = float(np.std(fold_r2s))
    cv_mean_best_epoch = float(np.mean(fold_epochs))

    final_results = final_train_and_test(
        X_trainval_np=X_trainval_np,
        X_test_np=X_test_np,
        y_trainval_np=y_trainval_np,
        y_test_np=y_test_np,
        X_all_np=X_np,
        y_all_np=y_np,
    )

    total_time = time.time() - start_time

    record = {
        "test_size": test_size,
        "train_test_split": split_label,

        "cv_mean_bias": cv_mean_bias,
        "cv_std_bias": cv_std_bias,
        "cv_mean_cov": cv_mean_cov,
        "cv_std_cov": cv_std_cov,
        "cv_mean_r2": cv_mean_r2,
        "cv_std_r2": cv_std_r2,
        "cv_mean_best_epoch": cv_mean_best_epoch,

        "final_epoch": final_results["final_epoch"],

        "test_bias": final_results["test_bias"],
        "test_cov": final_results["test_cov"],
        "test_r2": final_results["test_r2"],

        "all_bias": final_results["all_bias"],
        "all_cov": final_results["all_cov"],
        "all_r2": final_results["all_r2"],

        "training_time_s": total_time,

        "architecture": ARCHITECTURE,
        "activation": ACTIVATION,
        "optimizer": OPTIMIZER_NAME,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "n_splits": N_SPLITS,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
    }

    records.append(record)

    pd.DataFrame(records).to_csv(SUMMARY_CSV_PATH, index=False)

    print("\nSUMMARY")
    print(f"CV mean CoV  = {cv_mean_cov:.6f} ± {cv_std_cov:.6f}")
    print(f"CV mean R²   = {cv_mean_r2:.6f} ± {cv_std_r2:.6f}")
    print(f"Test bias    = {final_results['test_bias']:.6f}")
    print(f"Test CoV     = {final_results['test_cov']:.6f}")
    print(f"Test R²      = {final_results['test_r2']:.6f}")
    print(f"Final epoch  = {final_results['final_epoch']}")
    print(f"Time [s]     = {total_time:.2f}")


# =============================================================================
# SAVE FINAL RESULTS
# =============================================================================

results_df = pd.DataFrame(records)

results_df = results_df.sort_values(
    by=["test_cov", "cv_std_cov", "test_r2"],
    ascending=[True, True, False]
).reset_index(drop=True)

results_df.to_csv(SUMMARY_CSV_PATH, index=False)

best_row = results_df.iloc[0]

with open(SUMMARY_TXT_PATH, "w") as f:
    f.write("FIXED DNN - TRAIN/TEST SPLIT SENSITIVITY STUDY\n")
    f.write("==============================================\n\n")
    f.write("Fixed model and protocol:\n")
    f.write(f"Architecture: {ARCHITECTURE}\n")
    f.write(f"Activation: {ACTIVATION}\n")
    f.write(f"Optimizer: {OPTIMIZER_NAME}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Weight decay: {WEIGHT_DECAY}\n")
    f.write(f"K-fold: {N_SPLITS}\n")
    f.write(f"Max epochs: {MAX_EPOCHS}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n\n")
    f.write("Best selected row:\n")
    f.write(str(best_row))
    f.write("\n\n")
    f.write("Selection rule:\n")
    f.write("lowest test CoV, then lowest CV std CoV, then highest test R².\n")


# =============================================================================
# PLOTS
# =============================================================================

plot_metric(
    results_df,
    metric_col="test_cov",
    ylabel="Test CoV",
    title="Test CoV vs Train/Test Split",
    save_path=PLOTS_DIR / "test_cov_vs_train_test_split.png"
)

plot_metric(
    results_df,
    metric_col="test_r2",
    ylabel="Test R²",
    title="Test R² vs Train/Test Split",
    save_path=PLOTS_DIR / "test_r2_vs_train_test_split.png"
)

plot_metric(
    results_df,
    metric_col="test_bias",
    ylabel="Test Bias",
    title="Test Bias vs Train/Test Split",
    save_path=PLOTS_DIR / "test_bias_vs_train_test_split.png"
)

plot_metric(
    results_df,
    metric_col="cv_mean_cov",
    ylabel="Mean CV CoV",
    title="Mean CV CoV vs Train/Test Split",
    save_path=PLOTS_DIR / "mean_cv_cov_vs_train_test_split.png"
)


print("\n====================================================")
print("BEST RESULT")
print("====================================================")
print(best_row)

print("\n====================================================")
print("DONE")
print("====================================================")
print(f"Summary CSV saved to: {SUMMARY_CSV_PATH}")
print(f"Summary TXT saved to: {SUMMARY_TXT_PATH}")
print(f"Plots saved to: {PLOTS_DIR}")
print(f"Convergence plots saved to: {CONVERGENCE_DIR}")